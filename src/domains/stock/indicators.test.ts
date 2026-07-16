import {createMarketData} from "../../test/marketFixture";
import {calculateIndicators, getIndicatorWarmup, splitIndicators} from "./indicators";
import {DEFAULT_INDICATOR_PARAMETERS} from "./strategyGenome";

describe("stock indicators", () => {
    it("calculates every requested signal after warm-up", () => {
        const points = createMarketData(120);
        const rows = calculateIndicators(points);
        expect(rows).toHaveLength(points.length - getIndicatorWarmup(DEFAULT_INDICATOR_PARAMETERS));
        expect(rows[0]).toEqual(
            expect.objectContaining({
                smaFast: expect.any(Number),
                smaSlow: expect.any(Number),
                williamsR: expect.any(Number),
                roc: expect.any(Number),
                rsi: expect.any(Number),
                macd: expect.any(Number),
                bollingerPercentB: expect.any(Number),
                volatility: expect.any(Number),
                volumeZScore: expect.any(Number),
                nDayHigh: expect.any(Number),
                newHighRatio: expect.any(Number),
                nDayLow: expect.any(Number),
                newLowRatio: expect.any(Number),
            })
        );
        expect(rows.every(row => Number.isFinite(row.rsi) && Number.isFinite(row.volumeZScore) && Number.isFinite(row.newHighRatio) && Number.isFinite(row.newLowRatio))).toBe(true);
        expect(rows.every(row => row.newHighRatio > 0 && row.newHighRatio <= 1 + 1e-9)).toBe(true);
        expect(rows.every(row => row.newLowRatio > 0 && row.newLowRatio <= 1 + 1e-9)).toBe(true);
    });

    it("changes indicator output when the GA parameters change", () => {
        const points = createMarketData(240);
        const baseline = calculateIndicators(points, DEFAULT_INDICATOR_PARAMETERS);
        const optimized = calculateIndicators(points, {
            ...DEFAULT_INDICATOR_PARAMETERS,
            rsiPeriod: 7,
            rocPeriod: 5,
            bollingerPeriod: 12,
            bollingerMultiplier: 3,
            volatilityPeriod: 10,
            volumeZScorePeriod: 10,
            newHighPeriod: 15,
        });
        expect(optimized.at(-1)?.rsi).not.toBe(baseline.at(-1)?.rsi);
        expect(optimized.at(-1)?.roc).not.toBe(baseline.at(-1)?.roc);
        expect(optimized.at(-1)?.bollingerUpper).not.toBe(baseline.at(-1)?.bollingerUpper);
        expect(optimized.at(-1)?.volatility).not.toBe(baseline.at(-1)?.volatility);
        expect(optimized.at(-1)?.volumeZScore).not.toBe(baseline.at(-1)?.volumeZScore);
    });

    it("marks a rising series near the N-day high", () => {
        const points = createMarketData(120);
        const rows = calculateIndicators(points, {...DEFAULT_INDICATOR_PARAMETERS, newHighPeriod: 20});
        const last = rows.at(-1);
        expect(last).toBeDefined();
        expect(last!.newHighRatio).toBeGreaterThan(0.95);
        expect(last!.nDayHigh).toBeGreaterThanOrEqual(last!.close);
    });

    it("marks a falling trough near the N-day low", () => {
        const points = createMarketData(120).map((point, index) => {
            // Steady decline so the last bar sits near the lookback low.
            const price = 200 - index;
            return {...point, open: price, high: price + 1, low: price - 1, close: price};
        });
        const rows = calculateIndicators(points, {...DEFAULT_INDICATOR_PARAMETERS, newLowPeriod: 20});
        const last = rows.at(-1);
        expect(last).toBeDefined();
        expect(last!.newLowRatio).toBeGreaterThan(0.95);
        expect(last!.nDayLow).toBeLessThanOrEqual(last!.close);
    });

    it("forgets an older peak when the N-day lookback is shorter", () => {
        const points = createMarketData(200);
        // Spike ~50 bars before the end: long (80) still sees it, short (20) does not.
        const spikeIndex = points.length - 50;
        points[spikeIndex] = {...points[spikeIndex], high: 500, close: 400};
        const longLookback = calculateIndicators(points, {...DEFAULT_INDICATOR_PARAMETERS, newHighPeriod: 80});
        const shortLookback = calculateIndicators(points, {...DEFAULT_INDICATOR_PARAMETERS, newHighPeriod: 20});
        expect(longLookback.at(-1)?.nDayHigh).toBe(500);
        expect(shortLookback.at(-1)?.nDayHigh).toBeLessThan(500);
        expect(shortLookback.at(-1)?.newHighRatio).toBeGreaterThan(longLookback.at(-1)?.newHighRatio ?? 0);
    });

    it("forgets an older trough when the N-day low lookback is shorter", () => {
        const points = createMarketData(200);
        const troughIndex = points.length - 50;
        points[troughIndex] = {...points[troughIndex], low: 1, close: 5};
        const longLookback = calculateIndicators(points, {...DEFAULT_INDICATOR_PARAMETERS, newLowPeriod: 80});
        const shortLookback = calculateIndicators(points, {...DEFAULT_INDICATOR_PARAMETERS, newLowPeriod: 20});
        expect(longLookback.at(-1)?.nDayLow).toBe(1);
        expect(shortLookback.at(-1)?.nDayLow).toBeGreaterThan(1);
        expect(shortLookback.at(-1)?.newLowRatio).toBeGreaterThan(longLookback.at(-1)?.newLowRatio ?? 0);
    });

    it("does not leak a future price into earlier indicators", () => {
        const points = createMarketData(120);
        const original = calculateIndicators(points);
        const changed = points.map((point, index) => (index === points.length - 1 ? {...point, close: point.close * 4} : point));
        const recalculated = calculateIndicators(changed);
        expect(recalculated.slice(0, -1)).toEqual(original.slice(0, -1));
        expect(recalculated.at(-1)).not.toEqual(original.at(-1));
    });

    it("uses a chronological 80/20 split with one bridge row", () => {
        const rows = calculateIndicators(createMarketData(150));
        const result = splitIndicators(rows);
        expect(result.splitIndex).toBe(Math.floor(rows.length * 0.6));
        expect(result.test[0]).toEqual(rows[result.splitIndex - 1]);
    });
});
