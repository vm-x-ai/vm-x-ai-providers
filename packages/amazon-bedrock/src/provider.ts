import { Logger } from '@nestjs/common';
import {
  CompletionMetadata,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  ICompletionProvider,
} from '@vm-x-ai/completion-provider';
import { Span } from 'nestjs-otel';
import { Subject } from 'rxjs';
import { AmazonBedrockAIConnectionConfig } from './base.provider';
import { AmazonBedrockLlama3Provider } from './llama3.provider';
import { AmazonBedrockMistralProvider } from './mistral.provider';

export class AmazonBedrockProvider implements ICompletionProvider {
  private readonly llama3: AmazonBedrockLlama3Provider;
  private readonly mistral: AmazonBedrockMistralProvider;

  constructor(private logger: Logger) {
    this.llama3 = new AmazonBedrockLlama3Provider(logger);
    this.mistral = new AmazonBedrockMistralProvider(logger);
  }

  @Span('AmazonBedrockProvider.getRequestTokens')
  getRequestTokens(request: CompletionRequest, modelConfig: ResourceModelConfig): Promise<number> {
    if (modelConfig.model.startsWith('meta')) {
      return this.llama3.getRequestTokens(request);
    }

    return this.mistral.getRequestTokens(request);
  }

  @Span('AmazonBedrockProvider.getMaxReplyTokens')
  getMaxReplyTokens(request: CompletionRequest): number {
    return request.config?.maxTokens || 0;
  }

  @Span('AmazonBedrockProvider.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection<AmazonBedrockAIConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    if (model.model.startsWith('meta')) {
      return this.llama3.completion(request, connection, model, metadata, observable);
    }

    return this.mistral.completion(request, connection, model, metadata, observable);
  }
}
