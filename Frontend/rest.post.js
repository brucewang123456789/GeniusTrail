// src/mocks/rest.js
import { rest } from 'msw';

export const restPost = rest.post('/chat', (req, res, ctx) => {
  return res(
    ctx.status(200),
    ctx.json({
      success: true,
      message: 'Mocked chat response',
      data: { text: 'Hello, world!' }
    })
  );
});
