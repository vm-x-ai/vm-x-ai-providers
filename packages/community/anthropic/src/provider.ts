import Anthropic from '@anthropic-ai/sdk';
import type {
  MessageCreateParamsNonStreaming,
  MessageCreateParamsStreaming,
  ToolChoice,
  MessageParam,
  RawMessageStartEvent,
  RawContentBlockDeltaEvent,
  RawContentBlockStartEvent,
  RawMessageDeltaEvent,
  ContentBlock,
  TextBlock,
  ToolUseBlock,
} from '@anthropic-ai/sdk/resources/messages';
import { Stream } from '@anthropic-ai/sdk/streaming';
import type { Logger } from '@nestjs/common';
import type {
  CompletionMetadata,
  ICompletionProvider,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  TokenMessage,
  AIProviderConfig,
} from '@vm-x-ai/completion-provider';
import { BaseCompletionProvider, TokenCounter } from '@vm-x-ai/completion-provider';
import { Span } from 'nestjs-otel';
import type { Subject } from 'rxjs';

const API_TIMEOUT_SECONDS = 300;

export type CallAPIRequest = MessageCreateParamsNonStreaming | MessageCreateParamsStreaming;

export type AnthropicConnectionConfig = {
  apiKey: string;
};

export class AnthropicLLMProvider extends BaseCompletionProvider<Anthropic> implements ICompletionProvider {
  private modelsMaxTokensMap: Record<string, number | undefined> = {};

  constructor(logger: Logger, provider: AIProviderConfig) {
    super(logger, provider);
    this.modelsMaxTokensMap = provider.config.models.reduce((acc, item) => {
      return { ...acc, [item.value]: item?.options?.maxTokens };
    }, {});
  }

  @Span('Anthropic.getMaxReplyTokens')
  getMaxReplyTokens(request: CompletionRequest, modelConfig: ResourceModelConfig): number {
    const maxTok = request.config?.max_tokens || this.modelsMaxTokensMap[modelConfig.model] || undefined;
    return maxTok;
  }

  @Span('Anthropic.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection<AnthropicConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    return await this.callCompletion(request, connection, model, metadata, observable);
  }

  private async getRequestTokensDetails(
    request: CompletionRequest,
  ): Promise<{ usedTokens: number; completionUsedTokens: number; promptUsedTokens: number }> {
    const usageClient = new TokenCounter({
      model: 'cl100k_base',
      messages: request.messages.map(({ toolCalls, ...msg }) => ({
        ...msg,
        role: msg.role,
        content: msg.content ?? '',
        tool_calls: toolCalls as TokenMessage['tool_calls'],
      })),
    });
    const { usedTokens, completionUsedTokens, promptUsedTokens } = usageClient;
    return { usedTokens, completionUsedTokens, promptUsedTokens };
  }

  @Span('Anthropic.getRequestTokens')
  public async getRequestTokens(request: CompletionRequest): Promise<number> {
    const { usedTokens } = await this.getRequestTokensDetails(request);
    return usedTokens;
  }

  private async callCompletion(
    request: CompletionRequest,
    connection: AIConnection<AnthropicConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const client = await this.createClient(connection);

    let toolChoice: ToolChoice = { type: 'auto' }; // default to auto
    if (request.toolChoice?.auto) {
      toolChoice = {
        type: 'auto',
      };
    } else if (request.toolChoice?.none) {
      toolChoice = {
        type: 'any',
      };
    } else if (request.toolChoice?.tool && request.toolChoice.tool.function?.name) {
      toolChoice = {
        type: 'tool',
        name: request.toolChoice.tool.function.name,
      };
    }

    // Handle tool choice and streaming as per Anthropic API
    const anthropicRequest: CallAPIRequest = {
      ...(request.config || {}),
      max_tokens: request?.config?.max_tokens || this.modelsMaxTokensMap[model.model] || 4096,
      model: model.model,
      stream: request.stream,
      ...(request.config?.temperature && { temperature: request.config?.temperature }),
      ...((request.tools || []).length > 0 && {
        tool_choice: toolChoice,
        tools: request.tools as unknown as CallAPIRequest['tools'],
      }),
      messages: this.parseRequestMessagesToAnthropicFormat(request),
    };

    this.logger.log({ request: anthropicRequest }, 'Calling Anthropic API');

    let message;
    let timeToFirstToken: number | null = null;
    const startTime = Date.now();

    const { data } = await client.messages
      .create(anthropicRequest, {
        timeout: API_TIMEOUT_SECONDS * 1000,
      })
      .withResponse();

    if (data instanceof Stream) {
      ({ message, timeToFirstToken } = await this.parseStreamingResponse(
        request,
        data,
        model,
        metadata,
        startTime,
        observable,
      ));
    } else {
      message = data;
    }

    this.logger.log({ responseMessage: message }, 'Anthropic response');

    const responseTimestamp = new Date();
    const total_tokens = message?.usage?.output_tokens + message?.usage?.input_tokens;
    return {
      id: message.id,
      role: message.role,
      toolCalls: [],
      message: request.stream ? '' : message?.content.find((block) => block.type === 'text')?.text ?? '',
      responseTimestamp: responseTimestamp.getTime(),
      usage: message.usage
        ? {
            total: total_tokens,
            completion: message.usage.output_tokens,
            prompt: message.usage.input_tokens,
          }
        : undefined,
      metadata: {
        ...this.getMetadata(model, metadata),
        done: true,
      },
      metrics: message.usage
        ? {
            timeToFirstToken: timeToFirstToken ?? undefined,
            tokensPerSecond: total_tokens / ((Date.now() - startTime) / 1000),
          }
        : undefined,
      finishReason: this.parseFinishReason(message?.stop_reason) || 'stop',
    };
  }

  protected override async createClient(connection: AIConnection<AnthropicConnectionConfig>): Promise<Anthropic> {
    if (!connection.config?.apiKey) {
      throw new Error('API Key cannot be found in the AI connection config');
    }

    return new Anthropic({ apiKey: connection.config.apiKey });
  }

  private parseFinishReason(
    stopReason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | null,
  ): 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call' | null {
    switch (stopReason) {
      case 'end_turn':
        return 'stop';
      case 'max_tokens':
        return 'length';
      case 'stop_sequence':
        return 'stop';
      case 'tool_use':
        return 'tool_calls';
      default:
        return null;
    }
  }

  private parseRequestMessagesToAnthropicFormat(request: CompletionRequest) {
    return request.messages.map<MessageParam>((msg) => ({
      name: msg.name,
      role: msg.role as never,
      content: msg.content || '',
      tool_call_id: msg.toolCallId,
      tool_calls:
        (msg.toolCalls || []).length > 0
          ? (msg.toolCalls || [])
              .map((toolCall) => {
                if (!toolCall.function?.name || !toolCall.function?.arguments) {
                  return null;
                }

                return {
                  id: toolCall.id,
                  type: 'function',
                  function: {
                    name: toolCall.function.name,
                    arguments: toolCall.function.arguments,
                  },
                };
              })
              .filter((toolCall) => toolCall !== null)
          : undefined,
    }));
  }

  private async parseStreamingResponse(
    request: CompletionRequest,
    data: Stream<Anthropic.RawMessageStreamEvent>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    startTime: number,
    observable: Subject<CompletionResponse>,
  ): Promise<{ message: Anthropic.Message; timeToFirstToken: number }> {
    let timeToFirstToken = 0;
    const contentBlocks: ContentBlock[] = [];

    let message: Anthropic.Message = {
      id: '',
      model: '',
      role: 'assistant',
      content: [],
      type: 'message',
      usage: {
        input_tokens: 0,
        output_tokens: 0,
      },
      stop_reason: null,
      stop_sequence: null,
    };

    for await (const event of data) {
      if (timeToFirstToken === 0) {
        timeToFirstToken = Date.now() - startTime;
      }

      switch (event.type) {
        case 'message_start': {
          const startEvent = event as RawMessageStartEvent;

          // ACCUMULATE: if message_start, then initialize the message
          message = startEvent.message;

          // SEND: translate and pass on each chunk to observable
          observable.next({
            id: message.id,
            message: message?.content[0]?.type === 'text' ? message.content[0].text : '',
            role: message.role,
            toolCalls: [],
            metadata: {
              ...this.getMetadata(model, metadata),
              done: false,
            },
            finishReason: this.parseFinishReason(message?.stop_reason) || 'stop',
          });
          break;
        }
        case 'content_block_start': {
          const startEvent = event as RawContentBlockStartEvent;

          // ACCUMULATE: if content_block_start, then initialize the block
          contentBlocks[startEvent.index] = startEvent.content_block;

          // SEND: translate and pass on each chunk to observable
          if (startEvent.content_block.type === 'text') {
            observable.next({
              id: message.id,
              message: startEvent.content_block.text,
              role: message.role,
              toolCalls: [], // there's no tool call in the text block
              metadata: {
                ...this.getMetadata(model, metadata),
                done: false,
              },
            });
          } else if (startEvent.content_block.type === 'tool_use') {
            observable.next({
              id: message.id,
              message: '', // there's no message in the tool_use block
              role: message.role,
              toolCalls: [
                {
                  id: startEvent.content_block.id,
                  type: 'function',
                  function: {
                    name: startEvent.content_block.name,
                    arguments: '',
                  },
                },
              ],
              metadata: {
                ...this.getMetadata(model, metadata),
                done: false,
              },
            });
          }
          break;
        }
        case 'message_delta': {
          // TODO: determine for each case, which things need to be pushed to observable. The observable format is ours, and is different than what is defined here
          const deltaMessage = event as RawMessageDeltaEvent;
          if (deltaMessage.delta.stop_reason) {
            message.stop_reason = deltaMessage.delta.stop_reason;
          }
          if (deltaMessage.delta.stop_sequence) {
            message.stop_sequence = deltaMessage.delta.stop_sequence;
          }
          if (deltaMessage.usage) {
            message.usage = {
              input_tokens: message.usage?.input_tokens, // Retain input tokens from message_start
              output_tokens: deltaMessage.usage.output_tokens,
            };
          }

          observable.next({
            id: message.id,
            role: message.role,
            toolCalls: [],
            metadata: {
              ...this.getMetadata(model, metadata),
              done: false,
            },
            usage: {
              prompt: 0,
              completion: deltaMessage.usage.output_tokens,
              total: deltaMessage.usage.output_tokens,
            },
            ...(deltaMessage.delta.stop_reason
              ? {
                  finishReason: this.parseFinishReason(deltaMessage.delta.stop_reason) || 'stop',
                }
              : {}),
          });

          break;
        }
        case 'content_block_delta': {
          const deltaEvent = event as RawContentBlockDeltaEvent;
          const index = deltaEvent.index;

          if (deltaEvent.delta.type === 'text_delta') {
            // ACCUMULATE: add text to the existing text block
            if (contentBlocks[index] && contentBlocks[index].type === 'text') {
              (contentBlocks[index] as TextBlock).text += deltaEvent.delta.text;
            }

            // SEND: translate and pass on each chunk to observable
            observable.next({
              id: message.id,
              role: message.role,
              message: deltaEvent.delta.text,
              toolCalls: [],
              metadata: {
                ...this.getMetadata(model, metadata),
                done: false,
              },
            });
          }

          if (deltaEvent.delta.type === 'input_json_delta') {
            if (contentBlocks[index] && contentBlocks[index].type === 'tool_use') {
              const toolUseBlock = contentBlocks[index] as ToolUseBlock;

              // ACCUMULATE: add partial JSON to the existing input block
              toolUseBlock.input += deltaEvent.delta.partial_json;

              // Note, we do not SEND / stream the partial toolcall to the observable, because a partial toolcall is useless
            }
          }

          break;
        }
        case 'content_block_stop': {
          // no action needed for content_block_stop
          break;
        }
        case 'message_stop': {
          // no action needed for message_stop
          break;
        }
      }
    }

    return { message, timeToFirstToken };
  }
}
