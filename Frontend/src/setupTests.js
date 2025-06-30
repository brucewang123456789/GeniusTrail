import { server } from 'msw/node';
import { handlers } from './mswHandlers';

// Setup handlers for mock API
server.use(...handlers);

// Establish API mocking before tests
beforeAll(() => server.listen());

// Reset after each test to ensure no test interference
afterEach(() => server.resetHandlers());

// Clean up after the tests are done
afterAll(() => server.close());
