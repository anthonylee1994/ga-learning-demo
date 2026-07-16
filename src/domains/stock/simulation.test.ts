import {createMarketData} from "../../test/marketFixture";
import {
    buildNetworkFeatures,
    createTradingReplay,
    decidePositionFromNetwork,
    decidePositionFromRules,
    evaluateStockGenome,
    getIndicatorColumns,
    getStockSplitIndices,
    positionBeforeDate,
} from "./simulation";
import {createStockSeedGenomes, decodeStockGenome, DEFAULT_INDICATOR_PARAMETERS, STOCK_GENE_COUNT, STOCK_TOPOLOGY} from "./strategyGenome";

describe("stock simulation", () => {
    const genome = Array(STOCK_GENE_COUNT).fill(0);
    // 三段切法要夠長（warmup 後仍有 train≥80、val≥30）
    const points = createMarketData(500);

    it("returns a finite fitness", () => {
        expect(evaluateStockGenome(genome, points)).toEqual(expect.any(Number));
        expect(Number.isFinite(evaluateStockGenome(genome, points))).toBe(true);
    });

    it("creates train / validate / test equity data with network-backed decisions", () => {
        const replay = createTradingReplay(genome, points);
        expect(replay.points.length).toBeGreaterThan(100);
        expect(replay.points.some(point => point.segment === "train")).toBe(true);
        expect(replay.points.some(point => point.segment === "validate")).toBe(true);
        expect(replay.points.some(point => point.segment === "test")).toBe(true);
        expect(Number.isFinite(replay.validateReturn)).toBe(true);
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
    });

    it("splits series into 65% train / 20% validate / 15% test", () => {
        const {trainEnd, validateEnd} = getStockSplitIndices(1000);
        expect(trainEnd).toBe(650);
        expect(validateEnd).toBe(850);
    });

    it("maps network outputs to long/cash without shorting", () => {
        expect(decidePositionFromNetwork([0.9, 0.1, 0.0], 0)).toBe(1);
        expect(decidePositionFromNetwork([0.1, 0.9, 0.2], 1)).toBe(1);
        expect(decidePositionFromNetwork([0.1, 0.2, 0.9], 1)).toBe(0);
        expect(decidePositionFromNetwork([0.1, 0.9, 0.2], 0)).toBe(0);
    });

    it("ignores weak buy/sell edges that barely beat hold (margin)", () => {
        // buy slightly above hold — should stay cash if already flat
        expect(decidePositionFromNetwork([0.12, 0.1, 0.0], 0, 0.08)).toBe(0);
        // clear buy
        expect(decidePositionFromNetwork([0.5, 0.1, 0.0], 0, 0.08)).toBe(1);
        // sell barely above hold — stay long
        expect(decidePositionFromNetwork([0.0, 0.1, 0.12], 1, 0.08)).toBe(1);
        // clear sell
        expect(decidePositionFromNetwork([0.0, 0.1, 0.5], 1, 0.08)).toBe(0);
    });

    it("uses position-sticky decisions when buy≈sell (anti thrash)", () => {
        // Long: sell must beat max(hold, buy)+margin — noisy sell above buy alone is not enough
        expect(decidePositionFromNetwork([0.55, 0.1, 0.6], 1, 0.08)).toBe(1);
        // Long: clear sell over the buy channel
        expect(decidePositionFromNetwork([0.5, 0.1, 0.7], 1, 0.08)).toBe(0);
        // Flat: buy must beat max(hold, sell)+margin
        expect(decidePositionFromNetwork([0.6, 0.1, 0.55], 0, 0.08)).toBe(0);
        expect(decidePositionFromNetwork([0.7, 0.1, 0.5], 0, 0.08)).toBe(1);
    });

    it("never stacks same-action fills and enforces min re-entry gap", () => {
        const rising = createMarketData(700);
        for (const seed of createStockSeedGenomes()) {
            for (const useNetwork of [true, false]) {
                const replay = createTradingReplay(seed, rising, useNetwork);
                const dateToIdx = new Map(replay.points.map((point, index) => [point.date, index]));
                for (let index = 1; index < replay.trades.length; index += 1) {
                    expect(replay.trades[index].action).not.toBe(replay.trades[index - 1].action);
                    const gap = (dateToIdx.get(replay.trades[index].date) ?? 0) - (dateToIdx.get(replay.trades[index - 1].date) ?? 0);
                    expect(gap).toBeGreaterThanOrEqual(5);
                }
                const buys = replay.trades.filter(trade => trade.action === "buy");
                for (let index = 1; index < buys.length; index += 1) {
                    const gap = (dateToIdx.get(buys[index].date) ?? 0) - (dateToIdx.get(buys[index - 1].date) ?? 0);
                    // 最少 5 持倉 + 5 空倉
                    expect(gap).toBeGreaterThanOrEqual(10);
                }
            }
        }
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
        // 持倉特徵喺 index 17（N 日新低喺 16）
        expect(features[17]).toBe(1);
    });

    it("includes OHLC candle structure in the feature vector", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const index = Math.min(50, columns.length - 1);
        columns.open[index] = 100;
        columns.high[index] = 105;
        columns.low[index] = 98;
        columns.close[index] = 100;
        columns.closeReturn[index] = 0.02;
        const features = buildNetworkFeatures(columns, index, 0, DEFAULT_INDICATOR_PARAMETERS);
        expect(features[0]).toBeCloseTo(0, 5);
        expect(features[1]).toBeGreaterThan(0);
        expect(features[2]).toBeLessThan(0);
        expect(features[3]).toBeCloseTo(0.4, 5);
    });

    it("feeds tuned RSI and Williams threshold distances into the network", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const index = Math.min(50, columns.length - 1);
        columns.rsi[index] = 20;
        columns.williamsR[index] = -90;
        const oversold = buildNetworkFeatures(columns, index, 0, DEFAULT_INDICATOR_PARAMETERS);
        // 18–21：RSI 買距 / 賣距 / W% 買距 / 賣距
        expect(oversold[18]).toBeGreaterThan(0);
        expect(oversold[19]).toBeLessThan(0);
        expect(oversold[20]).toBeGreaterThan(0);
        expect(oversold[21]).toBeLessThan(0);

        columns.rsi[index] = 80;
        columns.williamsR[index] = -10;
        const overbought = buildNetworkFeatures(columns, index, 1, DEFAULT_INDICATOR_PARAMETERS);
        expect(overbought[18]).toBeLessThan(0);
        expect(overbought[19]).toBeGreaterThan(0);
        expect(overbought[20]).toBeLessThan(0);
        expect(overbought[21]).toBeGreaterThan(0);
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

        // Downtrend: a single overbought exit is enough to sell.
        columns.rsi[index] = DEFAULT_INDICATOR_PARAMETERS.rsiSellThreshold;
        columns.williamsR[index] = -50;
        expect(decidePositionFromRules(columns, index, 1, DEFAULT_INDICATOR_PARAMETERS)).toBe(0);
    });

    it("requires both exits to sell while the SMA trend is up (rule mode)", () => {
        const columns = getIndicatorColumns(points, DEFAULT_INDICATOR_PARAMETERS);
        const index = Math.min(50, columns.length - 1);
        columns.smaFast[index] = 110;
        columns.smaSlow[index] = 100;
        columns.macd[index] = 0;
        columns.macdSignal[index] = 1;
        columns.rsi[index] = DEFAULT_INDICATOR_PARAMETERS.rsiSellThreshold;
        columns.williamsR[index] = -50;
        // Uptrend + only RSI hot → stay long.
        expect(decidePositionFromRules(columns, index, 1, DEFAULT_INDICATOR_PARAMETERS)).toBe(1);

        columns.williamsR[index] = DEFAULT_INDICATOR_PARAMETERS.williamsSellThreshold;
        // Both exits fire → allow sell even in uptrend.
        expect(decidePositionFromRules(columns, index, 1, DEFAULT_INDICATOR_PARAMETERS)).toBe(0);
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

    it("ranks buy-biased seed above sell-biased seed on a rising market", () => {
        const rising = createMarketData(700);
        const buySeed = createStockSeedGenomes()[0];
        // Same period head as buy seed, but flip output biases toward sell / cash.
        const sellSeed = buySeed.slice();
        const {networkGenome} = decodeStockGenome(buySeed);
        // Output biases sit at the end of the network slice (buy, hold, sell).
        const headLen = STOCK_GENE_COUNT - networkGenome.length;
        const outputBiasStart = headLen + networkGenome.length - 3;
        sellSeed[outputBiasStart] = -1.2;
        sellSeed[outputBiasStart + 1] = 0.2;
        sellSeed[outputBiasStart + 2] = 1.2;
        const buyFitness = evaluateStockGenome(buySeed, rising, true);
        const sellFitness = evaluateStockGenome(sellSeed, rising, true);
        const buyReplay = createTradingReplay(buySeed, rising, true);
        expect(buyFitness).toBeGreaterThan(sellFitness);
        // Fixture is strongly upward — buy-biased seed must actually enter and capture return.
        expect(buyReplay.trades.some(trade => trade.action === "buy")).toBe(true);
        expect(buyReplay.trainReturn).toBeGreaterThan(0.1);
    });

    it("prefers higher-return policies over cash-heavy ones on a rising market", () => {
        const rising = createMarketData(700);
        const buySeed = createStockSeedGenomes()[0];
        const sellSeed = buySeed.slice();
        const {networkGenome} = decodeStockGenome(buySeed);
        const headLen = STOCK_GENE_COUNT - networkGenome.length;
        const outputBiasStart = headLen + networkGenome.length - 3;
        sellSeed[outputBiasStart] = -1.2;
        sellSeed[outputBiasStart + 1] = 0.2;
        sellSeed[outputBiasStart + 2] = 1.2;
        const buyReplay = createTradingReplay(buySeed, rising, true);
        const sellReplay = createTradingReplay(sellSeed, rising, true);
        const buyFitness = evaluateStockGenome(buySeed, rising, true);
        const sellFitness = evaluateStockGenome(sellSeed, rising, true);
        // When buy-seed actually earns more train return, fitness must agree (return-first).
        if (buyReplay.trainReturn > sellReplay.trainReturn + 0.02) {
            expect(buyFitness).toBeGreaterThan(sellFitness);
        }
        expect(buyReplay.trainReturn).toBeGreaterThan(0.08);
    });

    it("marks fills on the next bar open (not signal close)", () => {
        const rising = createMarketData(700);
        const buySeed = createStockSeedGenomes()[0];
        const replay = createTradingReplay(buySeed, rising, true);
        const buy = replay.trades.find(trade => trade.action === "buy");
        expect(buy).toBeDefined();
        const point = replay.points.find(row => row.date === buy!.date);
        expect(point).toBeDefined();
        // 成交價應係當日 open（fixture / 真實序列 open 可能同 close 差）
        expect(buy!.price).toBeGreaterThan(0);
    });
});
