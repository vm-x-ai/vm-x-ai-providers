import {
  BedrockRuntimeClient,
  ContentBlock,
  ConverseCommand,
  ConverseCommandInput,
  ConverseStreamCommand,
  Message,
  StopReason,
  TokenUsage,
  Tool,
  ToolChoice,
  ToolInputSchema,
} from '@aws-sdk/client-bedrock-runtime';
import { fromTemporaryCredentials } from '@aws-sdk/credential-providers';
import { status } from '@grpc/grpc-js';
import { HttpStatus, Logger } from '@nestjs/common';
import { AwsCredentialIdentityProvider } from '@smithy/types';
import {
  CompletionMetadata,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  ICompletionProvider,
  AIProviderConfig,
  BaseCompletionProvider,
  TokenCounter,
  TokenMessage,
  RequestMessageToolCall,
  CompletionRpcException,
  CompletionUsage,
} from '@vm-x-ai/completion-provider';
import { Span } from 'nestjs-otel';
import { Subject } from 'rxjs';
import { v4 as uuid } from 'uuid';

const cachedProviders = new Map<string, AwsCredentialIdentityProvider>();

export type AmazonBedrockAIConnectionConfig = {
  iamRoleArn: string;
  region: string;
  performanceConfig?: {
    latency?: 'standard' | 'optimized';
  };
};

const stopReasonMap: Record<StopReason, string> = {
  [StopReason.TOOL_USE]: 'tool_calls',
  [StopReason.END_TURN]: 'stop',
  [StopReason.MAX_TOKENS]: 'length',
  [StopReason.STOP_SEQUENCE]: 'stop',
  [StopReason.CONTENT_FILTERED]: 'content_filter',
  [StopReason.GUARDRAIL_INTERVENED]: 'guardrail',
};

export class AmazonBedrockProvider extends BaseCompletionProvider<BedrockRuntimeClient> implements ICompletionProvider {
  constructor(logger: Logger, provider: AIProviderConfig) {
    super(logger, provider);
  }

  protected override initializeTokenCounterModels(): void {
    // We don't need to initialize the token counter for the bedrock models, only for the GPT models
    TokenCounter.getEncodingForModelCached('cl100k_base');
  }

  @Span('AmazonBedrockProvider.getRequestTokens')
  public async getRequestTokens(request: CompletionRequest): Promise<number> {
    const usageClient = new TokenCounter({
      model: 'cl100k_base',
      messages: request.messages.map(({ toolCalls, ...msg }) => ({
        ...msg,
        role: msg.role,
        content: msg.content ?? '',
        tool_calls: toolCalls as TokenMessage['tool_calls'],
      })),
    });

    return usageClient.usedTokens;
  }

  @Span('AmazonBedrockProvider.getMaxReplyTokens')
  public getMaxReplyTokens(request: CompletionRequest): number {
    return request.config?.max_tokens ?? 0;
  }

  @Span('AmazonBedrockProvider.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection<AmazonBedrockAIConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const client = await this.createClient(connection);

    const toolChoice: ToolChoice | undefined = request.toolChoice?.auto
      ? {
          auto: {},
        }
      : request.toolChoice?.tool
        ? {
            tool: {
              name: request.toolChoice.tool.function?.name,
            },
          }
        : undefined;

    const tools =
      request.tools?.map<Tool>((tool) => ({
        toolSpec: {
          name: tool.function?.name,
          description: tool.function?.description,
          inputSchema: {
            json: tool.function?.parameters,
          } as ToolInputSchema.JsonMember,
        },
      })) ?? [];

    const input: ConverseCommandInput = {
      modelId: model.model,
      inferenceConfig: {
        maxTokens: request.config?.max_tokens,
        temperature: request.config?.temperature,
      },
      toolConfig:
        tools.length > 0
          ? {
              tools,
              toolChoice,
            }
          : undefined,
      messages: this.parseMessageToBedrockFormat(request),
      system: request.messages
        .filter((msg) => msg.role === 'system' && msg.content)
        .map((msg) => ({
          text: msg.content ?? '',
        })),
    };

    if (connection.config?.performanceConfig?.latency) {
      input.performanceConfig = {
        latency: connection.config.performanceConfig.latency,
      };
    }

    this.logger.log({ input }, 'Sending request to Amazon Bedrock');

    const startTime = Date.now();
    let timeToFirstToken: number | undefined = undefined;
    if (request.stream) {
      const stream = await client.send(new ConverseStreamCommand(input));
      if (!stream.stream) {
        throw new CompletionRpcException({
          rate: true,
          code: status.INTERNAL,
          message: 'Stream not available',
          statusCode: HttpStatus.INTERNAL_SERVER_ERROR,
          retryable: false,
          failureReason: 'Stream not available',
        });
      }
      const msgId = uuid();
      const toolCalls: RequestMessageToolCall[] = [];
      let stopReason = 'stop';
      let usage: CompletionUsage | undefined = undefined;
      let messageContent = '';

      for await (const chunk of stream.stream) {
        if (!timeToFirstToken) {
          timeToFirstToken = Date.now() - startTime;
        }

        if (chunk.contentBlockStart?.start?.toolUse && chunk.contentBlockStart.contentBlockIndex) {
          toolCalls[chunk.contentBlockStart.contentBlockIndex] = {
            type: 'function',
            function: {
              name: chunk.contentBlockStart.start.toolUse.name ?? '',
              arguments: '',
            },
            id: chunk.contentBlockStart.start.toolUse.toolUseId ?? uuid(),
          };
        } else if (chunk.contentBlockDelta) {
          if (chunk.contentBlockDelta.delta?.text) {
            messageContent += chunk.contentBlockDelta.delta.text;
            observable.next({
              id: msgId,
              message: chunk.contentBlockDelta.delta?.text,
              role: 'assistant',
              toolCalls: [],
              metadata: {
                ...this.getMetadata(model, metadata),
                done: false,
              },
              finishReason: undefined,
            });
          } else if (chunk.contentBlockDelta.delta?.toolUse && chunk.contentBlockDelta.contentBlockIndex) {
            const fn = toolCalls[chunk.contentBlockDelta.contentBlockIndex].function;
            if (fn) {
              fn.arguments += chunk.contentBlockDelta.delta.toolUse.input ?? '';
            }
          }
        } else if (chunk.messageStop && chunk.messageStop.stopReason) {
          stopReason = stopReasonMap[chunk.messageStop.stopReason];
        } else if (
          chunk.metadata &&
          chunk.metadata.usage &&
          chunk.metadata.usage.outputTokens &&
          chunk.metadata.usage.inputTokens &&
          chunk.metadata.usage.totalTokens
        ) {
          usage = {
            completion: chunk.metadata.usage.outputTokens,
            prompt: chunk.metadata.usage.inputTokens,
            total: chunk.metadata.usage.totalTokens,
          };
        }
      }

      return {
        id: msgId,
        role: 'assistant',
        responseTimestamp: Date.now(),
        finishReason: stopReason,
        message: '',
        toolCalls: toolCalls.filter((call) => !!call),
        usage: usage || this.generateCompletionUsage(undefined, model, request, messageContent, toolCalls),
        metrics: usage
          ? {
              timeToFirstToken,
              tokensPerSecond: usage.total / ((Date.now() - startTime) / 1000),
            }
          : undefined,
        metadata: {
          ...this.getMetadata(model, metadata),
          done: true,
        },
      };
    } else {
      const result = await client.send(new ConverseCommand(input));

      const responseTimestamp = new Date();
      const messageContent = result.output?.message?.content?.[0].text ?? '';
      const toolCalls =
        result.stopReason === StopReason.TOOL_USE
          ? result.output?.message?.content
              ?.filter((msg): msg is ContentBlock.ToolUseMember => !!msg.toolUse)
              .map<RequestMessageToolCall>((msg) => ({
                id: msg.toolUse.toolUseId ?? uuid(),
                type: 'function',
                function: {
                  name: msg.toolUse.name ?? '',
                  arguments:
                    typeof msg.toolUse.input === 'object' ? JSON.stringify(msg.toolUse.input) : `${msg.toolUse?.input}`,
                },
              })) ?? []
          : [];

      const usage = this.generateCompletionUsage(result.usage, model, request, messageContent, toolCalls);

      return {
        id: result.$metadata.requestId || uuid(),
        role: 'assistant',
        responseTimestamp: responseTimestamp.getTime(),
        finishReason: result.stopReason ? stopReasonMap[result.stopReason] : 'stop',
        message: messageContent,
        toolCalls,
        usage,
        metrics: result.usage?.totalTokens
          ? {
              timeToFirstToken,
              tokensPerSecond: result.usage.totalTokens / ((Date.now() - startTime) / 1000),
            }
          : undefined,
        metadata: {
          ...this.getMetadata(model, metadata),
          done: true,
        },
      };
    }
  }

  private generateCompletionUsage(
    resultUsage: TokenUsage | undefined,
    model: ResourceModelConfig,
    request: CompletionRequest,
    messageContent: string,
    toolCalls: RequestMessageToolCall[],
  ): CompletionUsage {
    let usage: CompletionUsage | undefined = undefined;
    if (resultUsage && resultUsage.outputTokens && resultUsage.inputTokens && resultUsage.totalTokens) {
      usage = {
        completion: resultUsage.outputTokens,
        prompt: resultUsage.inputTokens,
        total: resultUsage.totalTokens,
      };
    } else {
      const counter = new TokenCounter({
        model: 'cl100k_base',
        messages: [
          ...request.messages.map(({ toolCalls, ...msg }) => ({
            ...msg,
            role: msg.role,
            content: msg.content ?? '',
            tool_calls: toolCalls as TokenMessage['tool_calls'],
          })),
          {
            role: 'assistant',
            content: messageContent,
            tool_calls: toolCalls as TokenMessage['tool_calls'],
          },
        ],
      });

      usage = {
        completion: counter.completionUsedTokens,
        prompt: counter.promptUsedTokens,
        total: counter.usedTokens,
      };
    }
    return usage;
  }

  private parseMessageToBedrockFormat(request: CompletionRequest): Message[] {
    const messages: Message[] = [];
    for (const msg of request.messages) {
      if (msg.role === 'system') {
        continue;
      }

      const role = msg.role === 'assistant' ? 'assistant' : 'user';

      if (msg.toolCalls) {
        messages.push({
          role,
          content: msg.toolCalls.map<ContentBlock.ToolUseMember>((tool) => ({
            toolUse: {
              input: this.parseToolArguments(tool),
              name: tool.function?.name,
              toolUseId: tool.id,
            },
          })),
        });
        continue;
      }

      if (msg.toolCallId) {
        const lastMessage = messages[messages.length - 1];
        if (lastMessage?.content?.[0]?.toolResult) {
          lastMessage.content.push({
            toolResult: {
              toolUseId: msg.toolCallId,
              content: [
                {
                  text: msg.content ?? '',
                },
              ],
              status: 'success',
            },
          });
        } else {
          messages.push({
            role,
            content: [
              {
                toolResult: {
                  toolUseId: msg.toolCallId,
                  content: [
                    {
                      text: msg.content ?? '',
                    },
                  ],
                  status: 'success',
                },
              },
            ],
          });
        }

        continue;
      }

      messages.push({
        role,
        content: [
          {
            text: msg.content ?? '',
          },
        ],
      });
    }

    return messages;
  }

  private parseToolArguments(tool: RequestMessageToolCall): ContentBlock.ToolUseMember['toolUse']['input'] {
    try {
      return JSON.parse(tool.function?.arguments ?? '{}');
    } catch (e) {
      return tool.function?.arguments;
    }
  }

  protected override async createClient(
    connection: AIConnection<AmazonBedrockAIConnectionConfig>,
  ): Promise<BedrockRuntimeClient> {
    if (connection.config) {
      this.logger.log(
        {
          roleArn: connection.config.iamRoleArn,
        },
        'Using custom IAM role for Bedrock client',
      );

      const cacheKey = connection.config.iamRoleArn;
      if (cachedProviders.has(cacheKey)) {
        this.logger.log(
          {
            roleArn: connection.config.iamRoleArn,
          },
          'Using cached IAM role credentials provider for Bedrock client',
        );
        return new BedrockRuntimeClient({
          region: connection.config.region,
          credentials: cachedProviders.get(cacheKey),
        });
      }

      const credentials = fromTemporaryCredentials({
        params: {
          RoleArn: connection.config.iamRoleArn,
          RoleSessionName: 'vm-x-server-cross-account-session',
          ExternalId: connection.workspaceEnvironmentId,
        },
      });

      cachedProviders.set(cacheKey, credentials);

      return new BedrockRuntimeClient({
        region: connection.config.region,
        credentials,
      });
    }

    return new BedrockRuntimeClient({});
  }
}
