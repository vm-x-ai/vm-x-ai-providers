import { InvokeModelCommandInput } from '@aws-sdk/client-bedrock-runtime';
import { Logger } from '@nestjs/common';
import {
  CompletionMetadata,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  CompletionUsage,
  ICompletionProvider,
  RequestMessage,
  AIProviderConfig,
} from '@vm-x-ai/completion-provider';
import { Span } from 'nestjs-otel';
import { Subject } from 'rxjs';
import dedent from 'string-dedent';
import { AmazonBedrockAIConnectionConfig, AmazonBedrockProvider } from './base.provider';

type Llama3InvokeResponse = {
  generation: string;
  prompt_token_count: number;
  generation_token_count: number;
};

export class AmazonBedrockLlama3Provider
  extends AmazonBedrockProvider<Llama3InvokeResponse>
  implements ICompletionProvider
{
  constructor(logger: Logger, provider: AIProviderConfig) {
    super(logger, provider);
  }

  getMaxReplyTokens(request: CompletionRequest): number {
    return request.config?.maxTokens || 0;
  }

  private getToolsContext(request: CompletionRequest): string {
    return dedent`
      If you need to call a tool please return in the following format:

      [TOOL_CALL]
      [
        {
          "function": "{functionName}",
          "args": {
            "{key}": "{value}"
          }
        }
      ]
      [/TOOL_CALL]

      - You can call multiple tools in the same request.

      Here are the available tools:

      \`\`\`json
      ${JSON.stringify(request.tools?.map((tool) => tool.function) || [], null, 2)}
      \`\`\`

      please only return this pattern if the tool name matches one of the ones listed here

      - If you already have enough information, you don't need to call a tool.
      `;
  }

  @Span('AmazonBedrockLlama3Provider.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection<AmazonBedrockAIConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const context = request.tools && request.tools.length > 0 ? this.getToolsContext(request) : '';

    const prompt = `<|begin_of_text|>\n\n${[
      {
        role: 'system',
        content: context,
      } as RequestMessage,
      ...request.messages,
    ]
      .map((message: RequestMessage) => {
        if (
          (['system', 'user'].includes(message.role) && message.content) ||
          (message.role === 'assistant' && !message.toolCalls)
        ) {
          return `<|start_header_id|>${message.role}<|end_header_id|>\n${message.name ? `${message.name}: ` : ''}${
            message.content
          }<|eot_id|>`;
        } else if (message.role === 'tool') {
          return `<|start_header_id|>system<|end_header_id|>\nTool call id '${message.toolCallId}' returned: ${message.content}<|eot_id|>`;
        } else if (message.role === 'assistant' && message.toolCalls) {
          return `<|start_header_id|>system<|end_header_id|>\nYou called the following tools: '${JSON.stringify(message.toolCalls, null, 2)}'<|eot_id|>`;
        }

        return '';
      })
      .join('\n\n')}\n\n<|start_header_id|>assistant<|end_header_id|>`;

    this.logger.log(
      {
        prompt,
        model: model.model,
      },
      'Amazon Bedrock Llama3 prompt',
    );

    const requestInput = {
      contentType: 'application/json',
      body: JSON.stringify({
        prompt,
        max_gen_len: request.config?.maxTokens,
      }),
      modelId: model.model,
    };

    return await this.invoke(requestInput, connection, request, model, metadata, observable);
  }

  protected override getResponseUsage(
    _request: CompletionRequest,
    requestInput: InvokeModelCommandInput,
    response: Llama3InvokeResponse,
  ): CompletionUsage {
    return {
      completion: response.generation_token_count,
      prompt: response.prompt_token_count,
      total: response.generation_token_count + response.prompt_token_count,
    };
  }

  protected override getResponseMessage(_request: CompletionRequest, response: Llama3InvokeResponse): string {
    return response.generation;
  }
}
