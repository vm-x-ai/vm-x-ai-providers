import {
  BedrockRuntimeClient,
  InvokeModelCommand,
  InvokeModelCommandInput,
  InvokeModelWithResponseStreamCommand,
  InvokeModelWithResponseStreamCommandOutput,
} from '@aws-sdk/client-bedrock-runtime';
import { fromTemporaryCredentials } from '@aws-sdk/credential-providers';
import { status } from '@grpc/grpc-js';
import { HttpStatus, Logger } from '@nestjs/common';
import { AwsCredentialIdentityProvider, ResponseMetadata } from '@smithy/types';
import {
  CompletionMetadata,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  CompletionUsage,
  CompletionRpcException,
} from '@vm-x-ai/completion-provider';
import { BaseCompletionProvider, TokenCounter } from '@vm-x-ai/completion-provider';
import { Subject } from 'rxjs';
import { v4 as uuidv4 } from 'uuid';

const TOOL_CALL_TOKEN = '[TOOL_CALL]';

export type StreamLastChunkBaseResponse = {
  'amazon-bedrock-invocationMetrics'?: {
    inputTokenCount: number;
    outputTokenCount: number;
    invocationLatency: number;
    firstByteLatency: number;
  };
  stop_reason?: string;
};

const cachedProviders = new Map<string, AwsCredentialIdentityProvider>();

export type AmazonBedrockAIConnectionConfig = {
  iamRoleArn: string;
  region: string;
};

export abstract class AmazonBedrockProvider<
  TResponse,
  TChunkResponse extends TResponse & StreamLastChunkBaseResponse = TResponse & StreamLastChunkBaseResponse,
> extends BaseCompletionProvider<BedrockRuntimeClient> {
  constructor(protected readonly logger: Logger) {
    super();
  }

  public async invoke(
    requestInput: InvokeModelCommandInput,
    aiConnection: AIConnection<AmazonBedrockAIConnectionConfig>,
    request: CompletionRequest,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    const client = await this.createClient(aiConnection);

    const startTime = Date.now();
    if (request.stream) {
      let timeToFirstToken = 0;
      const result = await client.send(new InvokeModelWithResponseStreamCommand(requestInput));

      let toolCall: boolean | undefined = undefined;
      let bufferToStream = false;
      let fullMessage = '';
      let trimmedMessage = '';
      const responseTimestamp = Date.now();

      const usage: CompletionUsage = {
        completion: 0,
        prompt: 0,
        total: 0,
      };

      for await (const event of result.body ?? []) {
        if (timeToFirstToken === 0) {
          timeToFirstToken = Date.now() - startTime;
        }
        const chunk = JSON.parse(new TextDecoder().decode(event.chunk?.bytes)) as TChunkResponse;
        const invocationMetrics = chunk['amazon-bedrock-invocationMetrics'];
        if (invocationMetrics) {
          usage.completion = invocationMetrics.outputTokenCount;
          usage.prompt = invocationMetrics.inputTokenCount;
          usage.total = invocationMetrics.inputTokenCount + invocationMetrics.outputTokenCount;
        }

        let completion: CompletionResponse | undefined;
        ({ fullMessage, trimmedMessage, toolCall, bufferToStream, completion } = this.processChunk(
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
        ));

        if (completion) {
          return completion;
        }
      }

      if (toolCall) {
        const toolResult = this.getToolCallResult(result.$metadata, null, fullMessage, usage, model, metadata);
        if (!toolResult) {
          throw new CompletionRpcException({
            rate: true,
            code: status.INTERNAL,
            message:
              'Message started as function call but failed to match the "\\[FUNCTION_CALL\\](.*)\\[\\/FUNCTION_CALL\\]" regex, original response was: ' +
              fullMessage,
            statusCode: HttpStatus.INTERNAL_SERVER_ERROR,
            retryable: false,
            failureReason: 'Function call failed to match regex',
          });
        }

        return toolResult;
      }

      return {
        id: result.$metadata.requestId ?? uuidv4(),
        role: 'assistant',
        message: '',
        usage,
        responseTimestamp,
        toolCalls: [],
        metadata: {
          ...this.getMetadata(model, metadata),
          done: true,
        },
        metrics: {
          timeToFirstToken,
          tokensPerSecond: usage.total / ((Date.now() - startTime) / 1000),
        },
        finishReason: 'stop',
      };
    } else {
      const result = await client.send(new InvokeModelCommand(requestInput));

      const nativeResponse = JSON.parse(new TextDecoder().decode(result.body)) as TResponse;

      const usage = this.getResponseUsage(request, requestInput, nativeResponse, model);
      const message = this.getResponseMessage(request, nativeResponse, model);
      const toolResult = this.getToolCallResult(result.$metadata, nativeResponse, message, usage, model, metadata);
      if (toolResult) {
        return toolResult;
      }

      return {
        id: result.$metadata.requestId ?? uuidv4(),
        role: 'assistant',
        message,
        responseTimestamp: Date.now(),
        usage,
        toolCalls: [],
        metadata: {
          ...this.getMetadata(model, metadata),
          done: true,
        },
        metrics: {
          tokensPerSecond: usage.total / ((Date.now() - startTime) / 1000),
        },
        finishReason: 'stop',
      };
    }
  }

  protected processChunk(
    request: CompletionRequest,
    chunk: TChunkResponse,
    model: ResourceModelConfig,
    observable: Subject<CompletionResponse>,
    fullMessage: string,
    trimmedMessage: string,
    toolCall: boolean | undefined,
    bufferToStream: boolean,
    responseTimestamp: number,
    result: InvokeModelWithResponseStreamCommandOutput,
    metadata: CompletionMetadata,
    _usage: CompletionUsage,
  ): {
    fullMessage: string;
    trimmedMessage: string;
    toolCall: boolean | undefined;
    bufferToStream: boolean;
    completion?: CompletionResponse;
  } {
    const chunkMessage = this.getResponseMessage(request, chunk, model);
    fullMessage += chunkMessage;
    trimmedMessage += this.getNormalizedMessageContent
      ? this.getNormalizedMessageContent(chunkMessage.trim())
      : chunkMessage.trim();

    if (TOOL_CALL_TOKEN.length > trimmedMessage.length && TOOL_CALL_TOKEN.startsWith(trimmedMessage)) {
      return {
        fullMessage,
        trimmedMessage,
        toolCall,
        bufferToStream,
      };
    } else if (
      TOOL_CALL_TOKEN.length <= trimmedMessage.length &&
      TOOL_CALL_TOKEN === trimmedMessage.substring(0, TOOL_CALL_TOKEN.length)
    ) {
      toolCall = true;
      return {
        fullMessage,
        trimmedMessage,
        toolCall,
        bufferToStream,
      };
    } else if (toolCall === undefined) {
      toolCall = false;
      bufferToStream = true;
    }

    if (toolCall === false) {
      observable.next({
        id: result.$metadata.requestId ?? uuidv4(),
        role: 'assistant',
        message: bufferToStream ? fullMessage : chunkMessage,
        responseTimestamp,
        toolCalls: [],
        metadata: {
          ...this.getMetadata(model, metadata),
          done: false,
        },
        finishReason: chunk.stop_reason || undefined,
      });

      bufferToStream = false;
    }

    return {
      fullMessage,
      trimmedMessage,
      toolCall,
      bufferToStream,
    };
  }

  protected abstract getResponseUsage(
    request: CompletionRequest,
    requestInput: InvokeModelCommandInput,
    response: TResponse,
    model: ResourceModelConfig,
  ): CompletionUsage;

  protected abstract getResponseMessage(
    request: CompletionRequest,
    response: TResponse,
    model: ResourceModelConfig,
  ): string;

  protected getNormalizedMessageContent?(content: string): string;

  protected getToolCallResult(
    bedrockMetadata: ResponseMetadata,
    response: TResponse | TChunkResponse | null,
    content: string,
    usage: CompletionUsage,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
  ): CompletionResponse | undefined {
    const normalizedContent = this.getNormalizedMessageContent ? this.getNormalizedMessageContent(content) : content;
    const match = /\[TOOL_CALL\](.*)\[\/TOOL_CALL\]/gms.exec(normalizedContent);
    if (!match) {
      return undefined;
    }

    try {
      const fnResult = JSON.parse(match[1]);

      return {
        responseTimestamp: Date.now(),
        id: bedrockMetadata.requestId ?? uuidv4(),
        role: 'assistant',
        message: '',
        usage,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        toolCalls: fnResult.map((fn: any) => ({
          id: uuidv4(),
          type: 'function',
          function: {
            name: fn.function,
            arguments: JSON.stringify(fn.args),
          },
        })),
        metadata: {
          ...this.getMetadata(model, metadata),
          done: false,
        },
        finishReason: 'tool_calls',
      };
    } catch (error) {
      throw new CompletionRpcException({
        rate: true,
        code: status.INTERNAL,
        message: 'Cannot parse the model result to JSON format, original response was: ' + content,
        statusCode: HttpStatus.INTERNAL_SERVER_ERROR,
        retryable: false,
        failureReason: 'Failed to parse model result',
      });
    }
  }

  public async getRequestTokens(request: CompletionRequest): Promise<number> {
    const usageClient = new TokenCounter({
      messages: request.messages.map((message) => ({
        role: message.role,
        content: message.content ?? '',
      })),
    });

    return usageClient.usedTokens;
  }

  protected override async createClient(
    connection: AIConnection<AmazonBedrockAIConnectionConfig>,
  ): Promise<BedrockRuntimeClient> {
    if (connection.config) {
      this.logger.log('Using custom IAM role for Bedrock client', {
        roleArn: connection.config.iamRoleArn,
      });

      const cacheKey = connection.config.iamRoleArn;
      if (cachedProviders.has(cacheKey)) {
        this.logger.log('Using cached IAM role credentials provider for Bedrock client', {
          roleArn: connection.config.iamRoleArn,
        });
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
