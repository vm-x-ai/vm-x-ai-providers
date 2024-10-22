import { Logger } from '@nestjs/common';
import { CompletionResponse } from '@vm-x-ai/completion-provider';
import { Subject } from 'rxjs';
import { DummyLLMProvider } from './index';

describe('Dummy e2e', () => {
  const logger = new Logger();
  const provider = new DummyLLMProvider(logger);
  const connection = {
    connectionId: 'default',
    description: 'default',
    name: 'default',
    provider: 'dummy',
    workspaceEnvironmentId: 'default',
  };
  const model = {
    connectionId: 'default',
    model: 'dummy',
    provider: 'dummy',
  };
  const metadata = {
    primaryModel: true,
    secondaryModel: false,
  };

  it('should return the completion message', async () => {
    const stream$ = new Subject<CompletionResponse>();
    const response = await provider.completion(
      {
        messages: [
          {
            role: 'user',
            content: 'Hi',
            toolCalls: [],
          },
        ],
        resource: 'default',
        stream: false,
        tools: [],
      },
      connection,
      model,
      metadata,
      stream$,
    );

    expect(response.finishReason).toBe('stop');
    expect(response.message).toBeDefined();
  });

  it('should return the completion message with chunks', async () => {
    const stream$ = new Subject<CompletionResponse>();
    let chunkCount = 0;
    stream$.subscribe({
      next: () => {
        chunkCount += 1;
      },
    });

    const response = await provider.completion(
      {
        messages: [
          {
            role: 'user',
            content: 'Hi',
            toolCalls: [],
          },
        ],
        resource: 'default',
        stream: true,
        tools: [],
      },
      connection,
      model,
      metadata,
      stream$,
    );

    stream$.complete();

    expect(response.finishReason).toBe('stop');
    expect(response.message).toBeDefined();
    expect(chunkCount).toBe(22);
  });
});
