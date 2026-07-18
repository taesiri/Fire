import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        app: 'index.html',
        capture: 'capture.html',
      },
    },
  },
});
