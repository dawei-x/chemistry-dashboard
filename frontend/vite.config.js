import path from "path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import eslint from "vite-plugin-eslint"
import tailwindcss from "@tailwindcss/vite"


export default defineConfig(() => {
    return {
        plugins: [
            react(), 
            eslint(), 
            tailwindcss()
        ],
        server: {
            proxy: {
                "/api": {  // Flask backend proxy
                    target: "http://127.0.0.1:5000",
                    changeOrigin: true,
                },
                "/socket.io": {  // Websocket proxy
                    target: "http://127.0.0.1:5000",
                    ws: true,
                    changeOrigin: true,
                },
                "/audio_socket": {  // Audio processing proxy
                    target: "http://127.0.0.1:9000",
                    ws: true,
                    changeOrigin: true,
                },
                "/video_socket": {  // Video processing proxy
                    target: "http://127.0.0.1:9003",
                    ws: true,
                    changeOrigin: true,
                },
            },
        },
        resolve: {
            alias: {
                "@": path.resolve(__dirname, "./src"),
                "@assets": path.resolve(__dirname, "./src/assets"),
                "@components": path.resolve(__dirname, "./src/components"),
                "@icons": path.resolve(__dirname, "./src/Icons"),
            },
        },
        build: {
            outDir: "build",
            sourcemap: true,
        },
    }
})
