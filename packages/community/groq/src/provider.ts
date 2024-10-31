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
import Groq from 'groq-sdk';
import { Stream } from 'groq-sdk/lib/streaming';
import type {
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionSystemMessageParam,
} from 'groq-sdk/resources/chat/completions';
import { Span } from 'nestjs-otel';
import type { Subject } from 'rxjs';

const API_TIMEOUT_SECONDS = 300;

export type CallAPIRequest = ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming;

export type GroqConnectionConfig = {
  apiKey: string;
};

export class GroqLLMProvider extends BaseCompletionProvider<Groq> implements ICompletionProvider {
  constructor(logger: Logger, provider: AIProviderConfig) {
    super(logger, provider);
  }

  getMaxReplyTokens(request: CompletionRequest): number {
    const maxTok = request.config?.max_tokens || 100; // REPLACE THIS!!
    this.logger.log('maxTok:', maxTok);
    return maxTok;
  }

  @Span('Groq.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection<GroqConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    //const callCompletion = this.errorHandling(this.callCompletion.bind(this));
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

  public async getRequestTokens(request: CompletionRequest): Promise<number> {
    const { usedTokens } = await this.getRequestTokensDetails(request);
    return usedTokens;
  }

  private async callCompletion(
    request: CompletionRequest,
    connection: AIConnection<GroqConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const client = await this.createClient(connection);

    // Handle tool choice and streaming as per Groq API
    const groqRequest: ChatCompletionCreateParams = {
      ...(request.config || {}),
      model: model.model,
      max_tokens: this.getMaxReplyTokens(request),
      temperature: request.config?.temperature ?? 0.5,
      stream: request.stream,
      tool_choice: request.toolChoice?.auto ? 'auto' : undefined,
      tools: (request.tools || []).length > 0 ? (request.tools as any[]) : undefined,
      messages: this.parseRequestMessagesToGroqFormat(request),
    };

    this.logger.log('Calling Groq API', {
      request: groqRequest, // WHY DOES THIS NOT GET LOGGED?
    });

    let message;
    let timeToFirstToken: number | null = null;
    const startTime = Date.now();

    const { data } = await client.chat.completions
      .create(groqRequest, {
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
    this.logger.log('Groq response:', { message });

    const responseTimestamp = new Date();
    return {
      id: message.id,
      role: message.choices[0].message.role,
      toolCalls: [],
      message: request.stream ? '' : message?.choices[0]?.message?.content ?? '',
      responseTimestamp: responseTimestamp.getTime(),
      usage: message.usage
        ? {
            total: message.usage.total_tokens,
            completion: message.usage.completion_tokens,
            prompt: message.usage.prompt_tokens,
          }
        : undefined,
      metadata: {
        ...this.getMetadata(model, metadata),
        done: true,
      },
      finishReason: message?.choices[0]?.finish_reason || 'stop',
    };
  }

  protected override async createClient(connection: AIConnection<GroqConnectionConfig>): Promise<Groq> {
    if (!connection.config?.apiKey) {
      throw new Error('API Key cannot be found in the AI connection config');
    }

    return new Groq({ apiKey: connection.config.apiKey });
  }

  private parseRequestMessagesToGroqFormat(
    request: CompletionRequest,
  ): Array<ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam | ChatCompletionSystemMessageParam> {
    return request.messages.map((msg) => {
      const baseMessage = { name: msg.name || '', content: msg.content ?? '' };

      switch (msg.role) {
        case 'user':
          return { ...baseMessage, role: 'user' } as ChatCompletionUserMessageParam;
        case 'assistant':
          return {
            ...baseMessage,
            role: 'assistant',
            tool_calls: msg.toolCalls as TokenMessage['tool_calls'],
          } as ChatCompletionAssistantMessageParam;
        case 'system':
          return { ...baseMessage, role: 'system' } as ChatCompletionSystemMessageParam;
        default:
          throw new Error(`Unsupported role: ${msg.role}`);
      }
    });
  }

  private async parseStreamingResponse(
    request: CompletionRequest,
    data: Stream<Groq.Chat.Completions.ChatCompletionChunk>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    startTime: number,
    observable: Subject<CompletionResponse>,
  ): Promise<{ message: Groq.Chat.Completions.ChatCompletion; timeToFirstToken: number }> {
    let timeToFirstToken = 0;
    const messageChoice: Groq.Chat.Completions.ChatCompletion.Choice = {
      finish_reason: null as never,
      index: 0,
      message: {
        content: null,
        role: 'assistant',
      },
      logprobs: null,
    };

    const message: Groq.Chat.Completions.ChatCompletion = {
      id: '',
      created: 0,
      model: '',
      object: 'chat.completion',
      usage: undefined,
      choices: [messageChoice],
    };

    for await (const part of data) {
      if (timeToFirstToken === 0) {
        timeToFirstToken = Date.now() - startTime;
      }

      message.id = part.id;
      message.created = part.created;
      message.model = part.model;

      if (!part.choices || part.choices.length === 0) {
        if (request.includeRawResponse) {
          observable.next({
            id: message.id,
            message: '',
            role: messageChoice.message.role,
            toolCalls: [],
            metadata: {
              ...this.getMetadata(model, metadata),
              done: false,
            },
            rawResponse: part,
            finishReason: undefined,
          });
        }
        continue;
      }

      const choice = part.choices[0];
      messageChoice.index = choice.index;
      messageChoice.logprobs = choice.logprobs ?? null;

      if (choice.delta?.content) {
        if (messageChoice.message.content === null) {
          messageChoice.message.content = '';
        }
        messageChoice.message.content += choice.delta.content;
      }

      if (choice.finish_reason) {
        messageChoice.finish_reason = choice.finish_reason;
      }

      if (choice.delta?.tool_calls) {
        messageChoice.message.tool_calls = messageChoice.message.tool_calls || [];
        for (const toolCall of choice.delta.tool_calls) {
          if (!messageChoice.message.tool_calls[toolCall.index]) {
            messageChoice.message.tool_calls[toolCall.index] = {
              id: toolCall.id,
              type: toolCall.type,
              function: toolCall.function,
            } as never;
          } else if (toolCall.function?.arguments) {
            messageChoice.message.tool_calls[toolCall.index].function.arguments += toolCall.function.arguments;
          }
        }
      }

      if ((choice.finish_reason === null || choice.finish_reason !== 'tool_calls') && !choice.delta?.tool_calls) {
        if (choice.delta?.content && !request.includeRawResponse) {
          observable.next({
            id: message.id,
            message: choice.delta.content,
            role: messageChoice.message.role,
            toolCalls: messageChoice.message.tool_calls ?? [],
            metadata: {
              ...this.getMetadata(model, metadata),
              done: false,
            },
            finishReason: choice.finish_reason ?? undefined,
          });
        }
      }

      if (request.includeRawResponse) {
        observable.next({
          id: message.id,
          message: choice.delta?.content ?? '',
          role: messageChoice.message.role,
          toolCalls: messageChoice.message.tool_calls ?? [],
          metadata: {
            ...this.getMetadata(model, metadata),
            done: false,
          },
          rawResponse: part,
          finishReason: choice.finish_reason ?? undefined,
        });
      }

      if (part.x_groq?.usage) {
        const usage = part.x_groq.usage;
        message.usage = {
          total_tokens: usage.total_tokens,
          completion_tokens: usage.completion_tokens,
          prompt_tokens: usage.prompt_tokens,
        };
      }
    } // end of for loop (data parts)

    return { message, timeToFirstToken };
  }
}
