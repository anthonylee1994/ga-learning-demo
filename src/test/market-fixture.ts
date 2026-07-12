import type {MarketDataPoint} from "../lib/types";

export function createMarketData(length: number): MarketDataPoint[] {
    return Array.from({length}, (_, index) => {
        const close = 100 + index * 0.18 + Math.sin(index / 5) * 3;
        return {
            date: new Date(Date.UTC(2020, 0, index + 1)).toISOString().slice(0, 10),
            open: close - 0.4,
            high: close + 1.2,
            low: close - 1.1,
            close,
            adjClose: close,
            volume: 1_000_000 + Math.sin(index / 3) * 120_000 + index * 1_000,
        };
    });
}
