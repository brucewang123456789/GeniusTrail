// src/mocks/setupServer.js
import { setupServer } from 'msw/node';
import { restPost } from './rest';

const server = setupServer(
  restPost
);

export { server };
