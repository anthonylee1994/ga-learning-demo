import {createApp} from "../server/app.js";

/**
 * Vercel serverless entry for all `/api/*` routes.
 * Local production still uses `server/index.ts` + `express.static(dist)`.
 */
const app = createApp();

export default app;
