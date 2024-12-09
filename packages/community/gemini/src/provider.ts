import { GoogleGenerativeAI, FunctionCallingMode, FinishReason } from '@google/generative-ai';
import type {
  Part,
  SingleRequestOptions,
  GenerationConfig,
  GenerativeModel,
  EnhancedGenerateContentResponse,
  Content,
  TextPart,
  FunctionCallPart,
  FunctionResponsePart,
  ToolConfig,
  FunctionCallingConfig,
  FunctionDeclarationSchema,
  Tool,
  SchemaType,
} from '@google/generative-ai';
import type { Logger } from '@nestjs/common';
import type {
  CompletionMetadata,
  ICompletionProvider,
  CompletionRequest,
  CompletionResponse,
  ResourceModelConfig,
  AIConnection,
  AIProviderConfig,
  TokenMessage,
} from '@vm-x-ai/completion-provider';
import { BaseCompletionProvider, TokenCounter } from '@vm-x-ai/completion-provider';
import { Span } from 'nestjs-otel';
import type { Subject } from 'rxjs';
import { v4 as uuid } from 'uuid';

const API_TIMEOUT_SECONDS = 300;

export type GeminiConnectionConfig = {
  apiKey: string;
};

export class GeminiLLMProvider extends BaseCompletionProvider<GoogleGenerativeAI> implements ICompletionProvider {
  private modelsMaxTokensMap: Record<string, number | undefined> = {};

  constructor(logger: Logger, provider: AIProviderConfig) {
    super(logger, provider);
    this.modelsMaxTokensMap = provider.config.models.reduce((acc, item) => {
      return { ...acc, [item.value]: item?.options?.maxTokens };
    }, {});
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

  @Span('GoogleGenerativeAI.getRequestTokens')
  public async getRequestTokens(request: CompletionRequest): Promise<number> {
    const { usedTokens } = await this.getRequestTokensDetails(request);
    return usedTokens;
  }

  @Span('GoogleGenerativeAI.getMaxReplyTokens')
  getMaxReplyTokens(request: CompletionRequest, modelConfig: ResourceModelConfig): number {
    const maxTok = request.config?.max_tokens || this.modelsMaxTokensMap[modelConfig.model] || undefined;
    return maxTok;
  }

  @Span('GoogleGenerativeAI.completion')
  public async completion(
    request: CompletionRequest,
    connection: AIConnection<GeminiConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    return await this.callCompletion(request, connection, model, metadata, observable);
  }

  protected override async createClient(connection: AIConnection<GeminiConnectionConfig>): Promise<GoogleGenerativeAI> {
    if (!connection.config?.apiKey) {
      throw new Error('API Key cannot be found in the AI connection config');
    }
    return new GoogleGenerativeAI(connection.config.apiKey);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private createGeminiGenerationConfig(config?: { [key: string]: any }): GenerationConfig {
    if (!config) {
      return {};
    }
    return {
      ...(config.temperature !== undefined && { temperature: config.temperature }),
      ...(config.top_p !== undefined && { topP: config.top_p }),
      ...(config.top_k !== undefined && { topK: config.top_k }),
      ...(config.max_tokens !== undefined && { maxOutputTokens: config.max_tokens }),
      ...(config.response_format && { responseMimeType: config.response_format }),
      ...(config.response_schema && { responseSchema: config.response_schema }),
      ...(config.presence_penalty !== undefined && { presencePenalty: config.presence_penalty }),
      ...(config.frequency_penalty !== undefined && { frequencyPenalty: config.frequency_penalty }),
      ...(config.responseLogprobs !== undefined && { responseLogprobs: config.responseLogprobs }),
      ...(config.logprobs !== undefined && { logprobs: config.logprobs }),
    };
  }

  private parseMessagesForGemini(request: CompletionRequest): {
    systemInstruction: Content | null;
    history: Content[];
    prompt: string | Array<string | Part>;
  } {
    // Separate system messages
    const systemMessages = request.messages.filter((message) => message.role === 'system');
    const nonSystemMessages = request.messages.filter((message) => message.role !== 'system');

    // Convert all non-system messages to Gemini format
    const allContents: Content[] = nonSystemMessages.map<Content>((message) => {
      const parts: Part[] = [];

      // Handle text content
      if (message.content) {
        parts.push({ text: message.content });
      }

      // Handle tool calls
      if (message.toolCalls?.length > 0) {
        message.toolCalls.forEach((toolCall) => {
          if (toolCall.function?.name && toolCall.function?.arguments) {
            let parsedArguments: object | null = null;
            try {
              parsedArguments = JSON.parse(toolCall.function.arguments);
            } catch (error) {
              this.logger.log({}, `Failed to parse tool call arguments: ${toolCall.function.arguments}`);
              throw new Error(`Failed to parse tool call arguments: ${toolCall.function.arguments}`);
            }
            if (parsedArguments) {
              const functionCallPart: FunctionCallPart = {
                functionCall: {
                  name: toolCall.function.name,
                  args: parsedArguments,
                },
              };
              parts.push(functionCallPart);
            }
          }
        });
      }

      // Handle tool responses
      if (message.role === 'tool' && message.content && message.toolCallId) {
        const toolCallIdParts = message.toolCallId.split('|');
        parts.push({
          functionResponse: {
            name: toolCallIdParts.length > 1 ? toolCallIdParts[1] : '', // Use the part after '|' if it exists
            response: { content: message.content },
          },
        });
      }

      return { role: message.role, parts };
    });

    // Extract history (all except the last message)
    const history = allContents.slice(0, -1);

    // Extract prompt (last message in allContents)
    const promptContent = allContents[allContents.length - 1] || { parts: [] };
    const prompt: string | (string | Part)[] =
      promptContent.parts.length === 1
        ? promptContent.parts[0].text || '' // Single part: use its text
        : promptContent.parts; // Multiple parts (or empty array)

    // Combine system messages into a single system instruction
    const systemInstruction: Content | null =
      systemMessages.length > 0
        ? {
            role: 'system',
            parts: systemMessages.map((message) => ({ text: message.content || '' })),
          }
        : null;

    return { systemInstruction, history, prompt };
  }

  private createGeminiToolConfig(request: CompletionRequest): ToolConfig {
    let functionCallingConfig: FunctionCallingConfig;

    if (request.toolChoice?.auto) {
      functionCallingConfig = { mode: FunctionCallingMode.AUTO };
    } else if (request.toolChoice?.none) {
      functionCallingConfig = { mode: FunctionCallingMode.NONE };
    } else if (request.toolChoice?.tool && request.toolChoice.tool.function?.name) {
      functionCallingConfig = {
        mode: FunctionCallingMode.ANY,
        allowedFunctionNames: [request.toolChoice.tool.function.name],
      };
    } else {
      functionCallingConfig = { mode: FunctionCallingMode.MODE_UNSPECIFIED };
    }

    const toolConfig = { functionCallingConfig };
    return toolConfig;
  }

  private createGeminiTools(request: CompletionRequest): Tool[] {
    if (!request.tools || request.tools.length === 0) {
      return [];
    }

    return [
      {
        functionDeclarations: request.tools.map((tool) => ({
          name: tool.function?.name || '',
          description: tool.function?.description,
          parameters: tool.function?.parameters
            ? {
                type: tool.function.parameters.type as SchemaType,
                properties: tool.function.parameters.properties || {},
                description: tool.function.parameters.description,
                required: tool.function.parameters.required || [],
              }
            : undefined,
        })),
      },
    ];
  }

  private parseFinishReason(
    geminiReason: FinishReason | undefined,
  ): 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call' | undefined {
    if (!geminiReason) {
      return undefined;
    }

    switch (geminiReason) {
      case FinishReason.STOP:
        return 'stop';
      case FinishReason.MAX_TOKENS:
        return 'length';
      case FinishReason.SAFETY: // safety content filters
      case FinishReason.RECITATION: // means that the model was reciting directly from its training data, and thus gets blocked
      case FinishReason.OTHER: // mostly when the model is asked about openai, msft, etc.
        return 'content_filter';
      case FinishReason.FINISH_REASON_UNSPECIFIED:
      default:
        return undefined;
    }
  }

  private async callCompletion(
    request: CompletionRequest,
    connection: AIConnection<GeminiConnectionConfig>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    observable: Subject<CompletionResponse>,
  ): Promise<CompletionResponse> {
    try {
      // set up the clients needed for Gemini
      const client = await this.createClient(connection);
      const modelClient: GenerativeModel = client.getGenerativeModel({ model: model.model });

      // interpret the VM-X request to Gemini format
      const generationConfig = this.createGeminiGenerationConfig(request.config);
      const { history, prompt, systemInstruction } = this.parseMessagesForGemini(request);
      const toolConfig = this.createGeminiToolConfig(request);
      const tools = this.createGeminiTools(request);

      // setup the chat
      const startChatParams = {
        ...(history && { history }),
        ...(systemInstruction && { systemInstruction }),
        ...(generationConfig && { generationConfig }),
        ...((request.tools || []).length > 0 && {
          toolConfig,
          tools,
        }),
      };
      const chatSession =
        Object.keys(startChatParams).length > 0 ? modelClient.startChat(startChatParams) : modelClient.startChat(); // startChat doesn't like to get an empty object {}

      this.logger.log({ startChatParams, prompt }, 'Calling Gemini API');

      // make the AI model request
      let chatResult: EnhancedGenerateContentResponse | undefined;
      let timeToFirstToken: number | null = null;
      const startTime = Date.now();

      if (request.stream) {
        const result = await chatSession.sendMessageStream(prompt);
        ({ chatResult, timeToFirstToken } = await this.parseStreamingResponse(
          request,
          result.stream,
          model,
          metadata,
          startTime,
          observable,
        ));
      } else {
        chatResult = (await chatSession.sendMessage(prompt)).response;
      }

      this.logger.log({ chatResult }, 'Gemini response');
      this.logger.log({ timeToFirstToken }, 'timeToFirstToken');

      const responseTimestamp = new Date();
      const candidateZero =
        Array.isArray(chatResult.candidates) && chatResult.candidates.length > 0 ? chatResult.candidates[0] : null;

      const completion = {
        id: `${uuid()}`,
        role: 'assistant',
        toolCalls: [], // chatResult.response.functionCalls() || [],
        message: chatResult.text() || '',
        responseTimestamp: responseTimestamp.getTime(),
        usage: chatResult.usageMetadata
          ? {
              total: chatResult.usageMetadata.totalTokenCount,
              completion: chatResult.usageMetadata.candidatesTokenCount,
              prompt: chatResult.usageMetadata.promptTokenCount,
            }
          : undefined,
        metrics: chatResult.usageMetadata
          ? {
              timeToFirstToken: timeToFirstToken ?? undefined,
              tokensPerSecond: chatResult.usageMetadata.totalTokenCount / ((Date.now() - startTime) / 1000),
            }
          : undefined,
        metadata: {
          ...this.getMetadata(model, metadata),
          done: true,
        },
        finishReason: this.parseFinishReason(candidateZero?.finishReason),
      };

      this.logger.log({ completion }, 'RETURN this to VM-X server');

      return completion;
    } catch (error) {
      this.logger.log({ error }, 'Error in callCompletion method');
      throw error;
    }
  }

  private async parseStreamingResponse(
    request: CompletionRequest,
    data: AsyncGenerator<EnhancedGenerateContentResponse>,
    model: ResourceModelConfig,
    metadata: CompletionMetadata,
    startTime: number,
    observable: Subject<CompletionResponse>,
  ): Promise<{ chatResult: EnhancedGenerateContentResponse; timeToFirstToken: number }> {
    let timeToFirstToken = 0;
    let chatResult: EnhancedGenerateContentResponse | undefined;
    const accumulatedContent: Content = { role: 'assistant', parts: [] };
    let currentPartIndex = -1;

    for await (const chunk of data) {
      if (timeToFirstToken === 0) {
        timeToFirstToken = Date.now() - startTime;
      }

      const candidate = chunk.candidates?.[0];
      if (candidate?.content) {
        for (const part of candidate.content.parts) {
          const currentPart = accumulatedContent.parts[currentPartIndex];

          // Increment the index if needed
          const needsNewPart =
            !currentPart || // No part exists at this index
            ('text' in currentPart && !('text' in part)) || // Type has changed, so we need to start a new part
            'functionCall' in part; // Always increment for new functionCall (each functioncall is a separate part, and functioncalls are always delivered in 1 chunk)
          if (needsNewPart) {
            currentPartIndex++;
          }

          // If the current part doesn't exist at the index, initialize it
          if (!accumulatedContent.parts[currentPartIndex]) {
            if ('text' in part) {
              accumulatedContent.parts[currentPartIndex] = { text: '' };
            } else if ('functionCall' in part) {
              accumulatedContent.parts[currentPartIndex] = {
                functionCall: { name: '', args: {} },
              };
            } else {
              continue; // Skip all other part types
            }
          }

          // TEXT PART
          if (
            'text' in part &&
            accumulatedContent.parts[currentPartIndex] &&
            'text' in accumulatedContent.parts[currentPartIndex]
          ) {
            // accumulate content
            (accumulatedContent.parts[currentPartIndex] as TextPart).text += part.text;

            // send to the observable
            observable.next({
              id: `${uuid()}`,
              role: candidate.content.role,
              message: part.text,
              toolCalls: [],
              metadata: {
                ...this.getMetadata(model, metadata),
                done: false,
              },
              ...(candidate.finishReason
                ? {
                    finishReason: this.parseFinishReason(candidate.finishReason),
                  }
                : {}),
            });
          }

          // FUNCTION CALL PART
          if ('functionCall' in part && part.functionCall) {
            // accumulate content (function call comes in 1 chunk fully)
            accumulatedContent.parts[currentPartIndex] = {
              functionCall: part.functionCall,
            };

            // send to the observable
            observable.next({
              id: `${uuid()}`,
              role: candidate.content.role,
              message: '',
              toolCalls: [
                {
                  id: `${uuid()}`,
                  type: 'function',
                  function: {
                    name: part.functionCall.name,
                    arguments: JSON.stringify(part.functionCall.args),
                  },
                },
              ],
              metadata: {
                ...this.getMetadata(model, metadata),
                done: false,
              },
              ...(candidate.finishReason
                ? {
                    finishReason: this.parseFinishReason(candidate.finishReason),
                  }
                : {}),
            });
          }
        }
      }

      // Update chatResult with the latest chunk
      chatResult = {
        ...chunk,
        candidates: [
          {
            ...candidate,
            content: accumulatedContent,
            index: candidate?.index ?? 0,
          },
        ],
      };
    } // end of chunk loop

    if (!chatResult) {
      throw new Error('No data received from the stream.');
    }

    return { chatResult, timeToFirstToken };
  }
}
