import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI wrapper (server/api.py) runs on 8000. Streamlit already holds 8501/8502,
// so those are deliberately avoided.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
