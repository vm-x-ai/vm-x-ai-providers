// toolCall examples

// OPENAI FORMAT
const messages1 = [
  { role: 'user', content: 'Hi, can you tell me the delivery date for my order?' },
  { role: 'assistant', content: 'Hi there! I can help with that. Can you please provide your order ID?' },
  { role: 'user', content: 'i think it is order_12345' },
  {
    role: 'assistant',
    content: null,
    tool_calls: [
      {
        id: 'call_123',
        function: {
          arguments: '{"order_id":"order_12345"}', // <-- is a string!!
          name: 'get_delivery_date',
        },
        type: 'function',
      },
    ],
  },
  {
    role: 'tool',
    content: 'order_12345 was delivered on 2020-01-01',
    tool_call_id: 'call_123',
  },
];

// GEMINI FORMAT
const messages2 = [
  { role: 'user', parts: [{ text: 'Hi, can you tell me the delivery date for my order?' }] },
  { role: 'model', parts: [{ text: 'Hi there! I can help with that. Can you please provide your order ID?' }] },
  { role: 'user', parts: [{ text: 'i think it is order_12345' }] },
  {
    role: 'model',
    parts: [
      {
        functionCall: {
          // <-- No ID here
          name: 'get_delivery_date',
          args: { order_id: 'order_12345' },
        },
      },
    ],
  },
  {
    role: 'user',
    parts: [
      {
        functionResponse: {
          name: 'get_delivery_date',
          response: {
            // <-- is an object!!
            orderId: 'order_12345',
            date: '2020-01-01',
          },
        },
      },
    ],
  },
];
