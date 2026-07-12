import {calculateGeneCount} from "../../lib/neural-network";
import {createMarketData} from "../../test/market-fixture";
import {createTradingReplay, evaluateStockGenome, STOCK_TOPOLOGY} from "./simulation";

describe("stock simulation", () => {
    const genome = Array(calculateGeneCount(STOCK_TOPOLOGY)).fill(0);
    const points = createMarketData(180);

    it("returns a finite fitness", () => {
        expect(evaluateStockGenome(genome, points)).toEqual(expect.any(Number));
        expect(Number.isFinite(evaluateStockGenome(genome, points))).toBe(true);
    });

    it("creates train and test equity data", () => {
        const replay = createTradingReplay(genome, points);
        expect(replay.points.length).toBeGreaterThan(100);
        expect(replay.points.some(point => point.segment === "train")).toBe(true);
        expect(replay.points.some(point => point.segment === "test")).toBe(true);
        expect(replay.benchmarkReturn).toBeGreaterThan(0);
    });
});
