import { InvokeModelCommandInput, InvokeModelWithResponseStreamCommandOutput } from '@aws-sdk/client-bedrock-runtime';
import { Injectable, Logger } from '@nestjs/common';
import { ResponseMetadata } from '@smithy/types';
import {
  CompletionMetadata,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  CompletionUsage,
  ICompletionProvider,
  RequestMessage,
  TokenCounter,
  AIProviderConfig,
} from '@vm-x-ai/completion-provider';
import { Span } from 'nestjs-otel';
import { Subject } from 'rxjs';
import dedent from 'string-dedent';
import { v4 as uuidv4 } from 'uuid';
import { AmazonBedrockAIConnectionConfig, AmazonBedrockProvider, StreamLastChunkBaseResponse } from './base.provider';

type MistralInvokeResponse = {
  outputs: {
    text: string;
  }[];
};

type MistralLargeInvokeResponse = {
  choices: {
    index: number;
    message: {
      role: string;
      content: string;
      tool_calls?: {
        id: string;
        function: {
          name: string;
          arguments: string;
        };
      }[];
    };
    stop_reason: string;
  }[];
};

const MISTRAL_LARGE_MODEL_ID = 'mistral.mistral-large-2402-v1:0';

@Injectable()
export class AmazonBedrockMistralProvider
  extends AmazonBedrockProvider<MistralInvokeResponse | MistralLargeInvokeResponse>
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

  @Span('AmazonBedrockMistralProvider.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection<AmazonBedrockAIConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    if (model.model === MISTRAL_LARGE_MODEL_ID) {
      const requestInput = {
        contentType: 'application/json',
        body: JSON.stringify({
          messages: request.messages.map((message) => ({
            role: message.role,
            content: message.content,
            tool_calls: message.toolCalls?.map((toolCall) => ({
              id: toolCall.id,
              function: {
                name: toolCall.function?.name,
                arguments: toolCall.function?.arguments,
              },
            })),
            tool_call_id: message.toolCallId,
          })),
          tools: request.tools?.map((tool) => ({
            type: 'function',
            function: {
              name: tool.function?.name,
              description: tool.function?.description,
              parameters: tool.function?.parameters,
            },
          })),
          tool_choice: request.toolChoice?.auto
            ? 'auto'
            : request.toolChoice?.none
              ? 'none'
              : request.toolChoice?.tool
                ? 'any'
                : 'auto',
          max_tokens: request.config?.maxTokens,
        }),
        modelId: model.model,
      };

      this.logger.log(
        {
          requestInput,
        },
        'Amazon Bedrock Mistral Large Prompt',
      );

      return await this.invoke(requestInput, connection, request, model, metadata, observable);
    } else {
      const context = request.tools && request.tools.length > 0 ? this.getToolsContext(request) : '';
      const prompt = `<s>\n\n${[
        ...request.messages,
        {
          role: 'system',
          content: context,
        } as RequestMessage,
      ]
        .map((message: RequestMessage) => {
          if (['user', 'system'].includes(message.role) && message.content) {
            return `[INST]${message.content}[/INST]`;
          } else if (message.role === 'assistant' && !message.toolCalls) {
            return message.name ? `${message.name}: ${message.content}` : message.content;
          } else if (message.role === 'tool') {
            return `[INST]\nTool call id '${message.toolCallId}' returned: ${message.content}[/INST]`;
          } else if (message.role === 'assistant' && message.toolCalls) {
            return `[INST]\nYou called the following tools: '${JSON.stringify(message.toolCalls, null, 2)}[/INST]`;
          }

          return '';
        })
        .join('\n\n')}`;

      this.logger.log(
        {
          prompt,
        },
        'Amazon Bedrock Mistral prompt',
      );

      const requestInput = {
        contentType: 'application/json',
        body: JSON.stringify({
          prompt,
          max_tokens: request.config?.maxTokens,
        }),
        modelId: model.model,
      };

      return await this.invoke(requestInput, connection, request, model, metadata, observable);
    }
  }

  private isLargeModel(
    model: ResourceModelConfig,
    response: MistralInvokeResponse | MistralLargeInvokeResponse,
  ): response is MistralLargeInvokeResponse {
    return model.model === MISTRAL_LARGE_MODEL_ID;
  }

  protected override getResponseUsage(
    _request: CompletionRequest,
    requestInput: InvokeModelCommandInput,
    response: MistralInvokeResponse | MistralLargeInvokeResponse,
    model: ResourceModelConfig,
  ): CompletionUsage {
    const usageClient = new TokenCounter({
      messages: [
        {
          role: 'user',
          content: JSON.parse(requestInput.body as string).prompt,
        },
        {
          role: 'assistant',
          content: this.isLargeModel(model, response) ? response.choices[0].message.content : response.outputs[0].text,
        },
      ],
    });

    const usage = {
      total: usageClient.usedTokens,
      completion: usageClient.completionUsedTokens,
      prompt: usageClient.promptUsedTokens,
    };

    return usage;
  }

  protected override getResponseMessage(
    request: CompletionRequest,
    response: MistralInvokeResponse | MistralLargeInvokeResponse,
    model: ResourceModelConfig,
  ): string {
    return this.isLargeModel(model, response) ? response.choices[0].message.content : response.outputs[0].text;
  }

  protected override getNormalizedMessageContent(content: string): string {
    return content.replace(/\\_/gm, '_');
  }

  protected override getToolCallResult(
    bedrockMetadata: ResponseMetadata,
    response:
      | MistralInvokeResponse
      | MistralLargeInvokeResponse
      | ((MistralInvokeResponse | MistralLargeInvokeResponse) & StreamLastChunkBaseResponse),
    content: string,
    usage: CompletionUsage,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
  ): CompletionResponse | undefined {
    if (this.isLargeModel(model, response) && response.choices[0].message.tool_calls) {
      return {
        responseTimestamp: Date.now(),
        id: bedrockMetadata.requestId ?? uuidv4(),
        role: 'assistant',
        message: '',
        usage,
        toolCalls: response.choices[0].message.tool_calls.map((toolCall) => ({
          id: toolCall.id,
          type: 'function',
          function: {
            name: toolCall.function.name,
            arguments: toolCall.function.arguments,
          },
        })),
        metadata: {
          ...this.getMetadata(model, metadata),
          done: false,
        },
        finishReason: 'tool_calls',
      };
    }

    return super.getToolCallResult(bedrockMetadata, response, content, usage, model, metadata);
  }

  protected override processChunk(
    request: CompletionRequest,
    chunk: (MistralInvokeResponse | MistralLargeInvokeResponse) & StreamLastChunkBaseResponse,
    model: ResourceModelConfig,
    observable: Subject<CompletionResponse>,
    fullMessage: string,
    trimmedMessage: string,
    toolCall: boolean | undefined,
    bufferToStream: boolean,
    responseTimestamp: number,
    result: InvokeModelWithResponseStreamCommandOutput,
    metadata: CompletionMetadata,
    usage: CompletionUsage,
  ): {
    fullMessage: string;
    trimmedMessage: string;
    toolCall: boolean | undefined;
    bufferToStream: boolean;
    completion?: CompletionResponse;
  } {
    if (this.isLargeModel(model, chunk) && chunk.choices[0].message.tool_calls?.length) {
      return {
        fullMessage,
        trimmedMessage,
        toolCall: true,
        bufferToStream,
        completion: {
          responseTimestamp,
          id: result.$metadata.requestId ?? uuidv4(),
          role: 'assistant',
          message: '',
          usage,
          toolCalls: chunk.choices[0].message.tool_calls.map((toolCall) => ({
            id: toolCall.id,
            type: 'function',
            function: {
              name: toolCall.function.name,
              arguments: toolCall.function.arguments,
            },
          })),
          metadata: {
            ...this.getMetadata(model, metadata),
            done: false,
          },
          finishReason: 'tool_calls',
        },
      };
    }

    return super.processChunk(
      request,
      chunk,
      model,
      observable,
      fullMessage,
      trimmedMessage,
      toolCall,
      bufferToStream,
      responseTimestamp,
      result,
      metadata,
      usage,
    );
  }
}
