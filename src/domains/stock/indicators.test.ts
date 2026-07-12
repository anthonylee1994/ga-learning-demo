import {createMarketData} from "../../test/market-fixture";
import {calculateIndicators, splitIndicators} from "./indicators";
import {DEFAULT_INDICATOR_PARAMETERS} from "./strategy-genome";

describe("stock indicators", () => {
    it("calculates every requested signal after warm-up", () => {
        const rows = calculateIndicators(createMarketData(120));
        expect(rows).toHaveLength(70);
        expect(rows[0].features).toHaveLength(13);
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
            })
        );
        expect(rows.every(row => row.features.every(Number.isFinite))).toBe(true);
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
        });
        expect(optimized.at(-1)?.rsi).not.toBe(baseline.at(-1)?.rsi);
        expect(optimized.at(-1)?.roc).not.toBe(baseline.at(-1)?.roc);
        expect(optimized.at(-1)?.bollingerUpper).not.toBe(baseline.at(-1)?.bollingerUpper);
        expect(optimized.at(-1)?.volatility).not.toBe(baseline.at(-1)?.volatility);
        expect(optimized.at(-1)?.volumeZScore).not.toBe(baseline.at(-1)?.volumeZScore);
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
        expect(result.splitIndex).toBe(Math.floor(rows.length * 0.8));
        expect(result.test[0]).toEqual(rows[result.splitIndex - 1]);
    });
});
