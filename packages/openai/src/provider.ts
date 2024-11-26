import { status } from '@grpc/grpc-js';
import {} from '@nestjs/common';
import { Logger, HttpStatus } from '@nestjs/common';
import { BaseCompletionProvider, HTTP_STATUS_TO_GRPC } from '@vm-x-ai/completion-provider';
import { TokenCounter, CompletionRpcException } from '@vm-x-ai/completion-provider';
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
import { Span } from 'nestjs-otel';
import { APIError, OpenAI, RateLimitError } from 'openai';
import type { Headers } from 'openai/core';
import type { ChatCompletionCreateParamsNonStreaming, ChatCompletionCreateParamsStreaming } from 'openai/resources';
import type {
  ChatCompletionCreateParamsBase,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
} from 'openai/resources/chat/completions';
import { Stream } from 'openai/streaming';
import type { Subject } from 'rxjs';

const API_TIMEOUT_SECONDS = 300;

export type CallAPIRequest = ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming;

export type OpenAIConnectionConfig = {
  apiKey: string;
};

export class OpenAILLMProvider extends BaseCompletionProvider<OpenAI> implements ICompletionProvider {
  constructor(logger: Logger, provider: AIProviderConfig) {
    super(logger, provider);
  }

  @Span('OpenAI.getMaxReplyTokens')
  getMaxReplyTokens(request: CompletionRequest): number {
    return request.config?.max_tokens ?? 0;
  }

  @Span('OpenAI.completion')
  async completion(
    request: CompletionRequest,
    connection: AIConnection,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const callCompletion = this.errorHandling(this.callCompletion.bind(this));

    return await callCompletion(request, connection, model, metadata, observable);
  }

  @Span('OpenAI.getRequestTokens')
  public async getRequestTokens(request: CompletionRequest, modelConfig: ResourceModelConfig): Promise<number> {
    const usageClient = new TokenCounter({
      model: modelConfig.model,
      messages: this.parseRequestMessagesToOpenAIFormat(request) as TokenMessage[],
    });

    return usageClient.usedTokens;
  }

  private async callCompletion(
    request: CompletionRequest,
    connection: AIConnection<OpenAIConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const client = await this.createClient(connection);

    let toolChoice: ChatCompletionCreateParamsBase['tool_choice'] = undefined;
    if (request.toolChoice?.auto) {
      toolChoice = 'auto';
    } else if (request.toolChoice?.none) {
      toolChoice = 'none';
    } else if (request.toolChoice?.tool && request.toolChoice.tool.function?.name) {
      toolChoice = {
        type: 'function',
        function: {
          name: request.toolChoice.tool.function.name,
        },
      };
    }

    const openaiRequest: CallAPIRequest = {
      ...(request.config || {}),
      ...(request.stream
        ? {
            stream_options: {
              include_usage: true,
            },
          }
        : {}),
      model: model.model,
      stream: request.stream,
      tool_choice: toolChoice,
      tools: (request.tools || []).length > 0 ? (request.tools as ChatCompletionTool[]) : undefined,
      messages: this.parseRequestMessagesToOpenAIFormat(request),
    };

    this.logger.log(
      {
        request: openaiRequest,
      },
      'Calling OpenAI API',
    );

    let message: OpenAI.Chat.Completions.ChatCompletion;
    let timeToFirstToken: number | null = null;
    const startTime = Date.now();
    const { data } = await client.chat.completions
      .create(openaiRequest, {
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
      message = data as OpenAI.Chat.Completions.ChatCompletion;
    }

    const responseTimestamp = new Date();
    return {
      id: message.id,
      role: message.choices[0].message.role,
      toolCalls: message.choices[0].message.tool_calls || [],
      message: request.stream ? '' : message.choices[0].message.content ?? '',
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
      metrics: message.usage
        ? {
            timeToFirstToken: timeToFirstToken ?? undefined,
            tokensPerSecond: message.usage.total_tokens / ((Date.now() - startTime) / 1000),
          }
        : undefined,
      rawResponse: message,
      finishReason: message.choices[0].finish_reason,
    };
  }

  protected override async createClient(connection: AIConnection<OpenAIConnectionConfig>): Promise<OpenAI> {
    if (!connection.config?.apiKey) {
      throw new Error('API Key cannot be found in the AI connection config');
    }

    return new OpenAI({ apiKey: connection.config?.apiKey });
  }

  private errorHandling(delegate: ICompletionProvider['completion']) {
    return async (
      request: CompletionRequest,
      connection: AIConnection,
      model: ResourceModelConfig,
      metadata: CompletionMetadata,
      observable: Subject<CompletionResponse>,
    ) => {
      try {
        return await delegate(request, connection, model, metadata, observable);
      } catch (error) {
        this.logger.error('error calling model', {
          error,
        });

        if (error instanceof RateLimitError) {
          const resetTime = this.extractRateLimitResetTime(error.headers);
          throw new CompletionRpcException({
            rate: true,
            code: status.RESOURCE_EXHAUSTED,
            message: error.message,
            statusCode: error.status,
            retryable: true,
            retryDelay: Math.max(resetTime.resetRequests, resetTime.resetTokens),
            failureReason: 'Rate limit exceeded',
            metadata: {
              code: error.code,
              type: error.type,
              param: error.param,
            },
          });
        }

        if (error instanceof APIError) {
          const retryableStatus = [500, 502, 503, 504];
          const statusCode = error.status ?? HttpStatus.INTERNAL_SERVER_ERROR;

          throw new CompletionRpcException({
            rate: false,
            code: HTTP_STATUS_TO_GRPC[statusCode] ?? status.UNKNOWN,
            message: error.message,
            statusCode: statusCode,
            retryable: retryableStatus.includes(statusCode),
            failureReason: 'External API error',
            metadata: {
              code: error.code,
              type: error.type,
              param: error.param,
            },
          });
        }

        throw new CompletionRpcException({
          rate: false,
          code: status.UNKNOWN,
          message: (error as Error).message,
          statusCode: HttpStatus.INTERNAL_SERVER_ERROR,
          retryable: false,
          failureReason: 'External API error',
        });
      }
    };
  }

  private parseRequestMessagesToOpenAIFormat(request: CompletionRequest) {
    return request.messages.map<ChatCompletionMessageParam>((msg) => ({
      name: msg.name,
      role: msg.role as never,
      content: msg.content || null,
      tool_call_id: msg.toolCallId,
      tool_calls:
        (msg.toolCalls || []).length > 0
          ? (msg.toolCalls || [])
              .map<ChatCompletionMessageToolCall | null>((toolCall) => {
                if (!toolCall.function?.name || toolCall.function?.arguments) {
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
    data: Stream<OpenAI.Chat.Completions.ChatCompletionChunk>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    startTime: number,
    observable: Subject<CompletionResponse>,
  ): Promise<{ message: OpenAI.Chat.Completions.ChatCompletion; timeToFirstToken: number }> {
    let timeToFirstToken = 0;
    const messageChoice: OpenAI.Chat.Completions.ChatCompletion.Choice = {
      finish_reason: null as never,
      index: 0,
      message: {
        content: null,
        role: 'assistant',
      },
      logprobs: null,
    };

    const message: OpenAI.Chat.Completions.ChatCompletion = {
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

      if (part.usage) {
        message.usage = part.usage;
      }

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
      if (choice.delta.content) {
        if (messageChoice.message.content === null) {
          messageChoice.message.content = '';
        }

        messageChoice.message.content += choice.delta.content;
      }

      if (choice.finish_reason) {
        messageChoice.finish_reason = choice.finish_reason;
      }

      if (choice.delta.tool_calls) {
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

      if ((choice.finish_reason === null || choice.finish_reason !== 'tool_calls') && !choice.delta.tool_calls) {
        if (choice.delta.content && !request.includeRawResponse) {
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
          message: choice.delta.content ?? '',
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
    }

    return { message, timeToFirstToken };
  }

  private extractRateLimitResetTime(headers?: Headers) {
    if (!headers) {
      return {
        resetRequests: 0,
        resetTokens: 0,
      };
    }

    const resetRequests = headers['x-ratelimit-reset-requests'];
    const resetTokens = headers['x-ratelimit-reset-tokens'];

    return {
      resetRequests: resetRequests ? this.parseDuration(resetRequests) : 0,
      resetTokens: resetTokens ? this.parseDuration(resetTokens) : 0,
    };
  }

  private parseDuration(durationStr: string | undefined): number {
    if (durationStr === undefined) {
      return 0;
    }

    let totalMillis = 0;
    const timeUnits = {
      h: 3600000,
      m: 60000,
      ms: 1,
      s: 1000,
    };

    const orderedUnits = ['h', 'm', 'ms', 's'];

    for (const unit of orderedUnits) {
      const match = new RegExp(`(\\d+(?:\\.\\d+)?)${unit}(?!\\w)`).exec(durationStr);
      if (match) {
        const value = parseFloat(match[1]);
        if (unit === 'h') {
          totalMillis += value * timeUnits['h'];
        } else if (unit === 'm') {
          totalMillis += value * timeUnits['m'];
        } else if (unit === 's') {
          totalMillis += value * timeUnits['s'];
        } else if (unit === 'ms') {
          totalMillis += value;
        }
      }
    }

    return totalMillis;
  }
}
