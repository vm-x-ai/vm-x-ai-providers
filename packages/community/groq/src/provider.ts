import type { Logger } from '@nestjs/common';
import type {
  CompletionMetadata,
  ICompletionProvider,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  TokenMessage,
} from '@vm-x-ai/completion-provider';
import { BaseCompletionProvider } from '@vm-x-ai/completion-provider';
import { TokenCounter } from '@vm-x-ai/completion-provider';
import Groq from 'groq-sdk';
import type {
  ChatCompletion,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  CompletionCreateParams,
} from 'groq-sdk/resources/chat/completions';
import { Span } from 'nestjs-otel';
import type { Subject } from 'rxjs';
import { v4 as uuidv4 } from 'uuid';

const API_TIMEOUT_SECONDS = 300;

export type CallAPIRequest = ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming;

export type GroqConnectionConfig = {
  apiKey: string;
};

export class GroqLLMProvider extends BaseCompletionProvider<Groq> implements ICompletionProvider {
  constructor(private readonly logger: Logger) {
    super();
  }

  getMaxReplyTokens(request: CompletionRequest): number {
    return request.config?.max_tokens || 0;
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
      stream: false, //request.stream,
      tool_choice: request.toolChoice?.auto ? 'auto' : undefined,
      tools: (request.tools || []).length > 0 ? (request.tools as any[]) : undefined,
      messages: [],
    };

    this.logger.log('Calling Groq API', {
      request: groqRequest,
    });

    //let message: any;
    const timeToFirstToken: number | null = null;
    const startTime = Date.now();

    const { data } = await client.chat.completions
      .create(groqRequest, {
        timeout: API_TIMEOUT_SECONDS * 1000,
      })
      .withResponse();

    // if (request.stream && data instanceof Stream) {
    //   ({ message, timeToFirstToken } = await this.parseStreamingResponse(
    //     request,
    //     data,
    //     model,
    //     metadata,
    //     startTime,
    //     observable,
    //   ));
    // } else {
    const message = data;
    // }

    const responseTimestamp = new Date();
    return {
      id: data.id,
      role: data.choices[0].message.role,
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

  // private async parseStreamingResponse(
  //   request: CompletionRequest,
  //   data: any,
  //   model: ResourceModelConfig,
  //   metadata: CompletionMetadata,
  //   startTime: number,
  //   observable: Subject<CompletionResponse>,
  // ): Promise<{ message: any; timeToFirstToken: number }> {
  //   let timeToFirstToken = 0;
  //   let message;

  //   for await (const chunk of data) {
  //     if (timeToFirstToken === 0) {
  //       timeToFirstToken = Date.now() - startTime;
  //     }

  //     message = chunk;
  //     observable.next({
  //       id: chunk.id,
  //       role: chunk.choices[0].delta.role || 'assistant',
  //       message: chunk.choices[0].delta.content || '',
  //       toolCalls: [],
  //       metadata: {
  //         ...this.getMetadata(model, metadata),
  //         done: false,
  //       },
  //       finishReason: chunk.choices[0].finish_reason,
  //     });
  //   }

  //   return { message, timeToFirstToken };
  // }
}
