import react from "@vitejs/plugin-react";
import {fileURLToPath} from "node:url";
import {defineConfig} from "vitest/config";

export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "brain.js": fileURLToPath(new URL("./node_modules/brain.js/dist/browser.js", import.meta.url)),
        },
    },
    test: {
        environment: "jsdom",
        exclude: ["e2e/**", "node_modules/**", "dist/**"],
        globals: true,
        setupFiles: "./src/test/setup.ts",
        coverage: {
            reporter: ["text", "html"],
        },
    },
});
