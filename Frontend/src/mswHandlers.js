import { rest } from 'msw';

export const handlers = [
  // Mock the /chat endpoint
  rest.post('/chat', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        message: 'Mocked chat response',
        data: { text: 'Hello, world!' }
      })
    );
  }),

  // Simulate failure response for LLM failure
  rest.post('/chat', (req, res, ctx) => {
    return res(
      ctx.status(500),
      ctx.json({
        success: false,
        error: 'Internal Server Error'
      })
    );
  }),
];
