import express from "express";
import {existsSync} from "node:fs";
import path from "node:path";
import YahooFinance from "yahoo-finance2";

interface CachedValue {
    expiresAt: number;
    payload: MarketDataPayload;
}

interface MarketDataPayload {
    symbol: string;
    currency: string;
    timezone: string;
    fetchedAt: string;
    points: Array<{
        date: string;
        open: number;
        high: number;
        low: number;
        close: number;
        adjClose: number | null;
        volume: number;
    }>;
}

const CACHE_TTL = 15 * 60 * 1000;
const SYMBOL_PATTERN = /^[A-Z0-9.^=-]{1,15}$/;
const yahooFinance = new YahooFinance({suppressNotices: ["yahooSurvey"]});
const cache = new Map<string, CachedValue>();

export function createApp(): express.Express {
    const app = express();
    app.disable("x-powered-by");

    app.get("/api/health", (_request, response) => {
        response.json({ok: true});
    });

    app.get("/api/market-data", async (request, response) => {
        const symbol = String(request.query.symbol ?? "QQQ")
            .trim()
            .toUpperCase();
        // 15 years keeps recent market regimes while retaining enough data for walk-forward validation.
        const range = String(request.query.range ?? "15y");
        const interval = String(request.query.interval ?? "1d");
        if (!SYMBOL_PATTERN.test(symbol) || (range !== "15y" && range !== "max") || interval !== "1d") {
            response.status(400).json({error: "Ticker、range 或 interval 格式無效。"});
            return;
        }

        const cacheKey = `${symbol}:${range}:${interval}`;
        const cached = cache.get(cacheKey);
        if (cached && cached.expiresAt > Date.now()) {
            response.json(cached.payload);
            return;
        }

        try {
            const period2 = new Date();
            const period1 = new Date(period2);
            if (range === "15y") {
                period1.setFullYear(period1.getFullYear() - 15);
            } else {
                period1.setFullYear(1900, 0, 1);
            }
            const result = await yahooFinance.chart(symbol, {
                period1,
                period2,
                interval: "1d",
                return: "array",
            });
            const points = result.quotes.flatMap(quote => {
                if (quote.open === null || quote.high === null || quote.low === null || quote.close === null || quote.volume === null) {
                    return [];
                }
                // Align OHLC to adjClose so overnight/intraday PnL matches dividend/split-adjusted equity.
                const close = quote.close;
                const adjClose = quote.adjclose ?? close;
                const factor = close !== 0 ? adjClose / close : 1;
                return [
                    {
                        date: quote.date.toISOString().slice(0, 10),
                        open: quote.open * factor,
                        high: quote.high * factor,
                        low: quote.low * factor,
                        close: adjClose,
                        adjClose,
                        volume: quote.volume,
                    },
                ];
            });

            if (points.length < 100) {
                response.status(404).json({error: `${symbol} 冇足夠歷史數據。`});
                return;
            }

            const payload: MarketDataPayload = {
                symbol: result.meta.symbol,
                currency: result.meta.currency,
                timezone: result.meta.exchangeTimezoneName,
                fetchedAt: new Date().toISOString(),
                points,
            };
            cache.set(cacheKey, {expiresAt: Date.now() + CACHE_TTL, payload});
            response.json(payload);
        } catch (error) {
            const message = error instanceof Error ? error.message : "Yahoo Finance request failed";
            const status = /not found|no data|invalid/i.test(message) ? 404 : 502;
            response.status(status).json({error: status === 404 ? `${symbol} 冇可用數據。` : "暫時連唔到 Yahoo Finance。"});
        }
    });

    // On Vercel, static assets come from `dist` via CDN; express.static is ignored.
    // Keep this path for local `pnpm start` after `pnpm build`.
    if (!process.env.VERCEL) {
        const distPath = path.resolve(process.cwd(), "dist");
        if (existsSync(distPath)) {
            app.use(express.static(distPath));
            app.use((_request, response) => response.sendFile(path.join(distPath, "index.html")));
        }
    }

    return app;
}
