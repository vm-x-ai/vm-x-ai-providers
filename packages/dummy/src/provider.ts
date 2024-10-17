import { Logger } from '@nestjs/common';
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
import { Span } from 'nestjs-otel';
import type { Subject } from 'rxjs';
import { v4 as uuidv4 } from 'uuid';

export class DummyLLMProvider extends BaseCompletionProvider<object> implements ICompletionProvider {
  constructor(private readonly logger: Logger) {
    super();
  }

  getModel(request: CompletionRequest): string {
    return request.config?.model ?? 'unknown';
  }

  getMaxReplyTokens(request: CompletionRequest): number {
    return request.config?.maxTokens || 0;
  }

  @Span('Dummy.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const messageId = uuidv4();
    const messageContent =
      "Hi, I'm a dummy model, which always replies the same content. Please go to https://console.vm-x.ai and add a model of your choice.";

    const usageClient = new TokenCounter({
      messages: [
        ...this.parseRequestMessages(request),
        {
          content: messageContent,
          role: 'assistant',
        },
      ],
    });

    const usage = {
      total: usageClient.usedTokens,
      completion: usageClient.completionUsedTokens,
      prompt: usageClient.promptUsedTokens,
    };

    if (request.tools) {
      const functionTokens = new TokenCounter({
        messages: request.tools.map((fn) => ({
          role: 'assistant',
          content: JSON.stringify(fn),
        })),
      }).usedTokens;

      usage.total += functionTokens;
      usage.prompt += functionTokens;
    }

    if (request.stream) {
      observable.next({
        id: messageId,
        role: 'assistant',
        message: messageContent,
        toolCalls: [],
        metadata: {
          ...this.getMetadata(model, metadata),
          done: true,
        },
        finishReason: undefined,
      });
    }

    const responseTimestamp = new Date();
    return {
      id: messageId,
      role: 'assistant',
      toolCalls: [],
      message: request.stream ? '' : messageContent,
      responseTimestamp: responseTimestamp.getTime(),
      usage,
      metadata: {
        ...this.getMetadata(model, metadata),
        done: true,
      },
      finishReason: 'stop',
    };
  }

  public async getRequestTokens(request: CompletionRequest): Promise<number> {
    const usageClient = new TokenCounter({
      model: 'gpt-4', // Dummy provider does have a model, so, we can use any model here
      messages: this.parseRequestMessages(request),
    });

    return usageClient.usedTokens;
  }

  protected override async createClient(): Promise<object> {
    return {};
  }

  private parseRequestMessages(request: CompletionRequest): TokenMessage[] {
    return request.messages.map<TokenMessage>(({ toolCalls, ...msg }) => ({
      ...msg,
      role: msg.role,
      content: msg.content ?? '',
      tool_calls: toolCalls as TokenMessage['tool_calls'],
    }));
  }
}
