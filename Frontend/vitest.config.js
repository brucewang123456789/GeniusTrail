import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['src/**/*.{test,spec}.{js,jsx}'], // 确保匹配测试文件
  },
  resolve: {
    alias: {
      '@components': resolve(__dirname, 'src/components'),
      '@src': resolve(__dirname, 'src'),
      '@root': resolve(__dirname),
    },
  },
});