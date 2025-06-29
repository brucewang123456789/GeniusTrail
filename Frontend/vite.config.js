import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';  // Import path to resolve aliases

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@components': path.resolve(__dirname, 'src/components'),  // Fixed alias resolution
    },
  },
  build: {
    rollupOptions: {
      input: 'Frontend/index.html',
    },
  },
});
