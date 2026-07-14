import {createMarketData} from "../../test/marketFixture";
import {
    ablateIndicatorMasks,
    buildNetworkFeatures,
    createTradingReplay,
    decidePositionFromNetwork,
    decidePositionFromRules,
    evaluateStockGenome,
    getIndicatorColumns,
    positionBeforeDate,
} from "./simulation";
import {
    ALL_INDICATOR_MASKS_ON,
    createStockSeedGenomes,
    decodeStockGenome,
    DEFAULT_INDICATOR_PARAMETERS,
    encodeMask,
    STOCK_GENE_COUNT,
    STOCK_MASK_GENE_COUNT,
    STOCK_PARAMETER_GENE_COUNT,
    STOCK_TOPOLOGY,
    withMaskOverride,
} from "./strategyGenome";

describe("stock simulation", () => {
    const genome = Array(STOCK_GENE_COUNT).fill(0);
    // Zero genome → all masks off; turn all masks on so fitness path is usable.
    for (let index = 0; index < STOCK_MASK_GENE_COUNT; index += 1) {
        genome[STOCK_PARAMETER_GENE_COUNT + index] = encodeMask(true);
    }
    const points = createMarketData(300);

    it("returns a finite fitness", () => {
        expect(evaluateStockGenome(genome, points)).toEqual(expect.any(Number));
        expect(Number.isFinite(evaluateStockGenome(genome, points))).toBe(true);
    });

    it("penalizes empty indicator masks hard", () => {
        const empty = Array(STOCK_GENE_COUNT).fill(0);
        for (let index = 0; index < STOCK_MASK_GENE_COUNT; index += 1) {
            empty[STOCK_PARAMETER_GENE_COUNT + index] = encodeMask(false);
        }
        expect(evaluateStockGenome(empty, points)).toBeLessThan(-50);
    });

    it("creates train and test equity data with network-backed decisions", () => {
        const replay = createTradingReplay(genome, points);
        expect(replay.points.length).toBeGreaterThan(100);
        expect(replay.points.some(point => point.segment === "train")).toBe(true);
        expect(replay.points.some(point => point.segment === "test")).toBe(true);
        expect(replay.benchmarkReturn).toBeGreaterThan(0);
        expect(replay.optimizedParameters).toEqual(
            expect.objectContaining({
                rsiPeriod: expect.any(Number),
                rsiBuyThreshold: expect.any(Number),
                rsiSellThreshold: expect.any(Number),
                williamsBuyThreshold: expect.any(Number),
                williamsSellThreshold: expect.any(Number),
                bollingerMultiplier: expect.any(Number),
            })
        );
        expect(replay.indicatorMasks).toEqual(
            expect.objectContaining({
                sma: true,
                rsi: true,
                macd: true,
            })
        );
    });

    it("maps network outputs to long/cash without shorting", () => {
        expect(decidePositionFromNetwork([0.9, 0.1, 0.0], 0)).toBe(1);
        expect(decidePositionFromNetwork([0.1, 0.9, 0.2], 1)).toBe(1);
        expect(decidePositionFromNetwork([0.1, 0.2, 0.9], 1)).toBe(0);
        expect(decidePositionFromNetwork([0.1, 0.9, 0.2], 0)).toBe(0);
    });

    it("tracks long/cash position from the trade log before a date", () => {
        expect(
            positionBeforeDate(
                [
                    {date: "2020-01-02", action: "buy", price: 100},
                    {date: "2020-01-10", action: "sell", price: 110},
                ],
                "2020-01-05"
            )
        ).toBe(1);
        expect(
            positionBeforeDate(
                [
                    {date: "2020-01-02", action: "buy", price: 100},
                    {date: "2020-01-10", action: "sell", price: 110},
                ],
                "2020-01-10"
            )
        ).toBe(0);
        expect(positionBeforeDate([], "2020-01-01")).toBe(0);
    });

    it("builds a fixed-size feature vector from indicator columns", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const features = buildNetworkFeatures(columns, Math.min(50, columns.length - 1), 1, DEFAULT_INDICATOR_PARAMETERS);
        expect(features).toHaveLength(STOCK_TOPOLOGY.inputSize);
        features.forEach(value => {
            expect(Number.isFinite(value)).toBe(true);
            expect(value).toBeGreaterThanOrEqual(-1);
            expect(value).toBeLessThanOrEqual(1);
        });
        expect(features[12]).toBe(1);
    });

    it("zeroes features for masked-off indicator families", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const index = Math.min(50, columns.length - 1);
        const masks = {...ALL_INDICATOR_MASKS_ON, rsi: false, williams: false, roc: false};
        const features = buildNetworkFeatures(columns, index, 0, DEFAULT_INDICATOR_PARAMETERS, masks);
        expect(features[3]).toBe(0);
        expect(features[4]).toBe(0);
        expect(features[5]).toBe(0);
        expect(features[13]).toBe(0);
        expect(features[14]).toBe(0);
        expect(features[15]).toBe(0);
        expect(features[16]).toBe(0);
        expect(features[12]).toBe(-1);
    });

    it("feeds tuned RSI and Williams threshold distances into the network", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const index = Math.min(50, columns.length - 1);
        columns.rsi[index] = 20;
        columns.williamsR[index] = -90;
        const oversold = buildNetworkFeatures(columns, index, 0, DEFAULT_INDICATOR_PARAMETERS);
        expect(oversold[13]).toBeGreaterThan(0);
        expect(oversold[14]).toBeLessThan(0);
        expect(oversold[15]).toBeGreaterThan(0);
        expect(oversold[16]).toBeLessThan(0);

        columns.rsi[index] = 80;
        columns.williamsR[index] = -10;
        const overbought = buildNetworkFeatures(columns, index, 1, DEFAULT_INDICATOR_PARAMETERS);
        expect(overbought[13]).toBeLessThan(0);
        expect(overbought[14]).toBeGreaterThan(0);
        expect(overbought[15]).toBeLessThan(0);
        expect(overbought[16]).toBeGreaterThan(0);
    });

    it("uses tuned RSI and Williams thresholds in rule mode", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const index = Math.min(50, columns.length - 1);
        columns.smaFast[index] = 0;
        columns.smaSlow[index] = 1;
        columns.macd[index] = 0;
        columns.macdSignal[index] = 1;
        columns.rsi[index] = DEFAULT_INDICATOR_PARAMETERS.rsiBuyThreshold;
        columns.williamsR[index] = DEFAULT_INDICATOR_PARAMETERS.williamsBuyThreshold;
        expect(decidePositionFromRules(columns, index, 0, DEFAULT_INDICATOR_PARAMETERS)).toBe(1);

        columns.rsi[index] = DEFAULT_INDICATOR_PARAMETERS.rsiSellThreshold;
        columns.williamsR[index] = -50;
        expect(decidePositionFromRules(columns, index, 1, DEFAULT_INDICATOR_PARAMETERS)).toBe(0);
    });

    it("ignores disabled rule-mode vote families", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const index = Math.min(50, columns.length - 1);
        columns.smaFast[index] = 2;
        columns.smaSlow[index] = 1;
        columns.macd[index] = 2;
        columns.macdSignal[index] = 1;
        columns.rsi[index] = 80;
        columns.williamsR[index] = -10;
        // Only SMA enabled → majority of 1 needs the trend vote.
        expect(
            decidePositionFromRules(columns, index, 0, DEFAULT_INDICATOR_PARAMETERS, {
                ...ALL_INDICATOR_MASKS_ON,
                sma: true,
                macd: false,
                rsi: false,
                williams: false,
            })
        ).toBe(1);
        expect(
            decidePositionFromRules(columns, index, 0, DEFAULT_INDICATOR_PARAMETERS, {
                ...ALL_INDICATOR_MASKS_ON,
                sma: false,
                macd: false,
                rsi: false,
                williams: false,
            })
        ).toBe(0);
    });

    it("scores seed genomes finitely on a rising market", () => {
        const rising = createMarketData(400);
        const seeds = createStockSeedGenomes();
        seeds.forEach(seed => {
            const fitness = evaluateStockGenome(seed, rising);
            expect(Number.isFinite(fitness)).toBe(true);
            expect(decodeStockGenome(seed).networkGenome.length).toBeGreaterThan(0);
        });
    });

    it("ranks buy-biased seed above near-cash empty-mask policy on a rising market", () => {
        const rising = createMarketData(500);
        const buySeed = createStockSeedGenomes()[0];
        const empty = Array(STOCK_GENE_COUNT).fill(0);
        for (let index = 0; index < STOCK_MASK_GENE_COUNT; index += 1) {
            empty[STOCK_PARAMETER_GENE_COUNT + index] = encodeMask(false);
        }
        const buyFitness = evaluateStockGenome(buySeed, rising, true);
        const emptyFitness = evaluateStockGenome(empty, rising, true);
        const buyReplay = createTradingReplay(buySeed, rising, true);
        expect(buyFitness).toBeGreaterThan(emptyFitness);
        // Fixture is strongly upward — buy-biased seed must actually enter and capture return.
        expect(buyReplay.trades.some(trade => trade.action === "buy")).toBe(true);
        expect(buyReplay.trainReturn).toBeGreaterThan(0.15);
    });

    it("ablates enabled masks and ranks by fitness drop", () => {
        const seed = createStockSeedGenomes()[0];
        const result = ablateIndicatorMasks(seed, points, true);
        expect(result.rows).toHaveLength(STOCK_MASK_GENE_COUNT);
        expect(result.activeCount).toBe(STOCK_MASK_GENE_COUNT);
        expect(Number.isFinite(result.baselineFitness)).toBe(true);
        result.rows.forEach(row => {
            expect(row.enabled).toBe(true);
            expect(Number.isFinite(row.fitnessDrop)).toBe(true);
        });
        // Sparse genome: ablation of disabled masks reports enabled=false.
        const sparse = withMaskOverride(withMaskOverride(seed, "roc", false), "volume", false);
        const sparseAblation = ablateIndicatorMasks(sparse, points, true);
        expect(sparseAblation.rows.find(row => row.id === "roc")?.enabled).toBe(false);
        expect(sparseAblation.rows.find(row => row.id === "volume")?.enabled).toBe(false);
        expect(sparseAblation.activeCount).toBe(STOCK_MASK_GENE_COUNT - 2);
    });
});
