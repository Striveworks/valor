import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig(() => {
  return {
    build: {
      outDir: 'build',
    },
    define: {
      'process.env': {},
    },
    plugins: [react()],
  };
});
