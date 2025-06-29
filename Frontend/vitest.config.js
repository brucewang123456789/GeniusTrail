import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['src/**/*.{test,spec}.{js,jsx}'],
  },
  resolve: {
    alias: {
      '@components': resolve(__dirname, 'src/components'),  // Resolved alias for Vitest
    },
  },
});
