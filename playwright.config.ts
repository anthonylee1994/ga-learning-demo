import {defineConfig, devices} from "@playwright/test";

export default defineConfig({
    testDir: "./e2e",
    timeout: 30_000,
    use: {
        baseURL: "http://127.0.0.1:5173",
        trace: "retain-on-failure",
    },
    projects: [
        {name: "desktop", use: {...devices["Desktop Chrome"]}},
        {name: "mobile", use: {...devices["Pixel 7"]}},
    ],
    webServer: {
        command: "pnpm dev",
        url: "http://127.0.0.1:5173",
        reuseExistingServer: true,
    },
});
