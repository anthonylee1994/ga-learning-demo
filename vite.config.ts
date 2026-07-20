import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import {defineConfig} from "vite";

export default defineConfig({
    plugins: [react(), tailwindcss()],
    server: {
        host: "127.0.0.1",
        port: 5173,
        proxy: {
            "/api": "http://127.0.0.1:3001",
        },
    },
    build: {
        // Snake/breaker workers embed matter-js; those entry chunks stay large by design.
        chunkSizeWarningLimit: 700,
        rollupOptions: {
            output: {
                manualChunks(id) {
                    if (!id.includes("node_modules")) {
                        return;
                    }
                    if (/node_modules[/\\](?:react-dom|react-router(?:-dom)?|scheduler)[/\\]/.test(id) || /node_modules[/\\]react[/\\]/.test(id)) {
                        return "react-vendor";
                    }
                    if (id.includes("@heroui") || id.includes("tailwind-variants")) {
                        return "heroui";
                    }
                    if (id.includes("recharts") || id.includes("d3-") || id.includes("@reduxjs")) {
                        return "charts";
                    }
                    if (id.includes("lucide-react")) {
                        return "icons";
                    }
                },
            },
        },
    },
    worker: {
        format: "es",
    },
});
