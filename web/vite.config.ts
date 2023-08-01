import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react()],
    server: {
      host: true,
      port: 3000,
    },
  };
});
