import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import {createPineScript, createNetworkDecisionLines, decodeLayers, evaluatePineNetwork} from "./pineScript";
import {createStockSeedGenomes, decodeStockGenome, STOCK_GENE_COUNT, STOCK_NETWORK_GENE_COUNT, STOCK_TOPOLOGY} from "./strategyGenome";

describe("Pine Script export", () => {
    it("embeds optimized parameters, multi-layer network, and long/flat orders", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => Math.sin(index) * 0.4);
        const {parameters} = decodeStockGenome(genome);
        const script = createPineScript(genome, "QQQ");

        expect(script).toContain("//@version=6");
        expect(script).toContain('strategy("EvoLab QQQ Evolved Strategy"');
        expect(script).toContain("default_qty_type=strategy.percent_of_equity");
        expect(script).toContain("commission_value=0.15");
        expect(script).toContain("process_orders_on_close=false");
        expect(script).toContain(`rsiPeriod = ${parameters.rsiPeriod}`);
        expect(script).toContain(`rsiBuyThreshold = ${parameters.rsiBuyThreshold}`);
        expect(script).toContain(`rsiSellThreshold = ${parameters.rsiSellThreshold}`);
        expect(script).toContain(`bollingerPeriod = ${parameters.bollingerPeriod}`);
        expect(script).toContain(`williamsPeriod = ${parameters.williamsPeriod}`);
        expect(script).toContain(`williamsBuyThreshold = ${parameters.williamsBuyThreshold}`);
        expect(script).toContain(`williamsSellThreshold = ${parameters.williamsSellThreshold}`);
        expect(script).toContain(`volatilityPeriod = ${parameters.volatilityPeriod}`);
        expect(script).toContain(`volumeZScorePeriod = ${parameters.volumeZScorePeriod}`);
        expect(script).toContain(`newHighPeriod = ${parameters.newHighPeriod}`);
        expect(script).toContain(`newLowPeriod = ${parameters.newLowPeriod}`);
        expect(script).not.toContain("useSma");
        expect(script).not.toContain("useRsi");
        expect(script).toContain("nDayHigh = ta.highest(high, newHighPeriod)");
        expect(script).toContain("nDayLow = ta.lowest(low, newLowPeriod)");
        expect(script).toContain("tanh(value) =>");
        expect(script).toContain("outBuy");
        expect(script).toContain("outHold");
        expect(script).toContain("outSell");
        expect(script).toContain("h1_0 = tanh(");
        // Second hidden layer must appear when topology has two hidden layers.
        expect(STOCK_TOPOLOGY.hiddenLayers).toEqual([10, 5]);
        expect(script).toContain("h2_0 = tanh(");
        expect(script).toContain("h2_4 = tanh(");
        expect(script).toContain("outBuy = tanh(");
        // Output must read from h2, not skip straight from h1 (regression for two-layer head).
        expect(script).toMatch(/outBuy = tanh\([^)]*h2_0/);
        expect(script).toContain(`f${STOCK_TOPOLOGY.inputSize - 1}`);
        expect(script).toContain("clamp((close / smaFast");
        expect(script).toContain("clamp((close / smaSlow");
        // NN 特徵唔再餵 K 線結構（開高低收）；volatility 仍可用 dailyReturn = close/close[1]
        expect(script).not.toContain("f0 = clamp((open / close");
        expect(script).not.toContain("f1 = clamp((high / close");
        expect(script).not.toContain("f2 = clamp((low / close");
        expect(script).not.toContain("f3 = clamp((close / close[1]");
        expect(script).toContain("18 → 10 → 5 → 3");
        expect(script).toContain('strategy.entry("Long", strategy.long)');
        expect(script).toContain('strategy.close("Long")');
        expect(script).not.toContain('strategy.entry("Short"');
        expect(script).not.toContain('strategy.close("Short")');
        // Parity with browser sim: margin + sticky stay (no min-hold hard lock)
        expect(script).toContain("actionMargin = 0.08");
        expect(script).toContain("stayIfFlat = math.max(outHold, outSell)");
        expect(script).toContain("stayIfLong = math.max(outHold, outBuy)");
        expect(script).toContain("flatPos < 1");
        expect(script).toContain("flatPos > 0");
        expect(script).not.toContain("minBarsLong");
        expect(script).not.toContain("barsInState");
    });

    it("decodes the full network genome into hidden×2 + output layers", () => {
        const genome = createStockSeedGenomes()[0];
        const {networkGenome} = decodeStockGenome(genome);
        expect(networkGenome).toHaveLength(STOCK_NETWORK_GENE_COUNT);
        const layers = decodeLayers(networkGenome);
        expect(layers).toHaveLength(3);
        expect(layers[0].biases).toHaveLength(10);
        expect(layers[0].weights[0]).toHaveLength(18);
        expect(layers[1].biases).toHaveLength(5);
        expect(layers[1].weights[0]).toHaveLength(10);
        expect(layers[2].biases).toHaveLength(3);
        expect(layers[2].weights[0]).toHaveLength(5);
    });

    it("matches NeuralNetworkAdapter forward pass (parity with in-app decisions)", () => {
        const genome = createStockSeedGenomes()[0];
        const {networkGenome} = decodeStockGenome(genome);
        const adapter = new NeuralNetworkAdapter(STOCK_TOPOLOGY);
        const inputs = [
            Array.from({length: STOCK_TOPOLOGY.inputSize}, (_, index) => Math.sin(index + 0.3) * 0.7),
            Array.from({length: STOCK_TOPOLOGY.inputSize}, (_, index) => Math.cos(index * 0.4) * 0.5),
            Array(STOCK_TOPOLOGY.inputSize).fill(0),
            Array(STOCK_TOPOLOGY.inputSize).fill(0.9),
        ];
        for (const input of inputs) {
            const pineOut = evaluatePineNetwork(networkGenome, input);
            const appOut = adapter.run(networkGenome, input);
            expect(pineOut).toHaveLength(3);
            expect(appOut).toHaveLength(3);
            pineOut.forEach((value, index) => {
                expect(value).toBeCloseTo(appOut[index], 5);
            });
            expect(argMax(pineOut)).toBe(argMax(appOut));
        }
    });

    it("emits every hidden layer in createNetworkDecisionLines", () => {
        const genome = createStockSeedGenomes()[0];
        const {networkGenome} = decodeStockGenome(genome);
        const lines = createNetworkDecisionLines(networkGenome).join("\n");
        expect(lines).toContain("h1_9 = tanh(");
        expect(lines).toContain("h2_4 = tanh(");
        expect(lines).toContain("outSell = tanh(");
        // h2 nodes must depend on h1 activations.
        expect(lines).toMatch(/h2_0 = tanh\([^)]*h1_0/);
        expect(lines).toMatch(/outHold = tanh\([^)]*h2_0/);
    });

    it("exports threshold-based rule mode without the neural network", () => {
        const genome = createStockSeedGenomes()[0];
        const script = createPineScript(genome, "QQQ", false);
        expect(script).toContain("rsiBuyVote = rsi <= rsiBuyThreshold");
        expect(script).toContain("williamsBuyVote = williamsR <= williamsBuyThreshold");
        expect(script).toContain("neededVotes");
        expect(script).not.toContain("useRsi");
        expect(script).not.toContain("h1_0 = tanh(");
        expect(script).not.toContain("h2_0 = tanh(");
    });
});
