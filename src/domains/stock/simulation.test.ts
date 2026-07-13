import type {IndicatorSnapshot} from "../../lib/types";
import {createMarketData} from "../../test/marketFixture";
import {createTradingReplay, decidePosition, evaluateStockGenome} from "./simulation";
import {DEFAULT_STRATEGY_RULES, STOCK_GENE_COUNT} from "./strategyGenome";

describe("stock simulation", () => {
    const genome = Array(STOCK_GENE_COUNT).fill(0);
    const points = createMarketData(300);

    it("returns a finite fitness", () => {
        expect(evaluateStockGenome(genome, points)).toEqual(expect.any(Number));
        expect(Number.isFinite(evaluateStockGenome(genome, points))).toBe(true);
    });

    it("creates train and test equity data with rule parameters", () => {
        const replay = createTradingReplay(genome, points);
        expect(replay.points.length).toBeGreaterThan(100);
        expect(replay.points.some(point => point.segment === "train")).toBe(true);
        expect(replay.points.some(point => point.segment === "test")).toBe(true);
        expect(replay.benchmarkReturn).toBeGreaterThan(0);
        expect(replay.optimizedParameters).toEqual(expect.objectContaining({rsiPeriod: expect.any(Number), bollingerMultiplier: expect.any(Number)}));
        expect(replay.optimizedRules).toEqual(
            expect.objectContaining({
                rsiBuy: expect.any(Number),
                minBuySignals: expect.any(Number),
                useTrendFilter: expect.any(Boolean),
            })
        );
    });

    it("decides long/cash from multi-signal voting rules without a neural network", () => {
        const bullish: IndicatorSnapshot = {
            date: "2020-01-01",
            close: 100,
            smaFast: 105,
            smaSlow: 100,
            williamsR: -90,
            roc: 0.05,
            rsi: 25,
            macd: 1,
            macdSignal: 0,
            macdHistogram: 1,
            bollingerUpper: 110,
            bollingerLower: 90,
            bollingerPercentB: 0.1,
            bollingerBandwidth: 0.2,
            volatility: 0.2,
            volumeZScore: 0,
        };
        const rules = {...DEFAULT_STRATEGY_RULES, minBuySignals: 3, minSellSignals: 3, useTrendFilter: true};
        expect(decidePosition(bullish, rules, 0)).toBe(1);
        expect(decidePosition({...bullish, smaFast: 90, smaSlow: 100, rsi: 80, williamsR: -10, roc: -0.05, bollingerPercentB: 0.9, macd: -1, macdSignal: 0}, rules, 1)).toBe(0);
    });
});
