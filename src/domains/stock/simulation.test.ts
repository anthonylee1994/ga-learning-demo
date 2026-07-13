import type {IndicatorSnapshot} from "../../lib/types";
import {createMarketData} from "../../test/marketFixture";
import {createTradingReplay, decidePosition, evaluateStockGenome} from "./simulation";
import {createStockSeedGenomes, DEFAULT_STRATEGY_RULES, STOCK_GENE_COUNT} from "./strategyGenome";

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
                strategyStyle: expect.stringMatching(/trend|mean_reversion|hybrid/),
            })
        );
    });

    it("decides long/cash from style-scoped multi-signal rules without a neural network", () => {
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
            volumeZScore: 1,
        };
        const trendRules = {...DEFAULT_STRATEGY_RULES, strategyStyle: "trend" as const, minBuySignals: 2, minSellSignals: 2};
        expect(decidePosition(bullish, trendRules, 0)).toBe(1);
        expect(decidePosition({...bullish, smaFast: 90, smaSlow: 100, macd: -1, macdSignal: 0, roc: -0.05, volumeZScore: -1}, trendRules, 1)).toBe(0);

        const reversionRules = {...DEFAULT_STRATEGY_RULES, strategyStyle: "mean_reversion" as const, minBuySignals: 2, minSellSignals: 2};
        expect(decidePosition(bullish, reversionRules, 0)).toBe(1);
        expect(decidePosition({...bullish, rsi: 80, williamsR: -10, bollingerPercentB: 0.9}, reversionRules, 1)).toBe(0);
    });

    it("prefers coherent seed strategies over pure cash on a rising market", () => {
        const rising = createMarketData(400);
        const seeds = createStockSeedGenomes();
        const cashGenome = Array.from({length: STOCK_GENE_COUNT}, () => 2);
        const cashFitness = evaluateStockGenome(cashGenome, rising);
        const bestSeed = Math.max(...seeds.map(seed => evaluateStockGenome(seed, rising)));
        expect(bestSeed).toBeGreaterThan(cashFitness);
    });

    it("blocks trend entries when volatility is extreme", () => {
        const hot: IndicatorSnapshot = {
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
            volatility: 0.9,
            volumeZScore: 2,
        };
        const trendRules = {...DEFAULT_STRATEGY_RULES, strategyStyle: "trend" as const, minBuySignals: 1};
        expect(decidePosition(hot, trendRules, 0)).toBe(0);
    });
});
