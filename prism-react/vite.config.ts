import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': new URL('./src', import.meta.url).pathname,
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api/v1': {
        target: 'http://127.0.0.1:8005',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8001',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
