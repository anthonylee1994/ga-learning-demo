import {createMarketData} from "../../test/market-fixture";
import {calculateIndicators, splitIndicators} from "./indicators";

describe("stock indicators", () => {
    it("calculates every requested signal after warm-up", () => {
        const rows = calculateIndicators(createMarketData(120));
        expect(rows).toHaveLength(70);
        expect(rows[0].features).toHaveLength(13);
        expect(rows[0]).toEqual(
            expect.objectContaining({
                sma20: expect.any(Number),
                sma50: expect.any(Number),
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
