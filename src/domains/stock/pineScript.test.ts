import {createPineScript} from "./pineScript";
import {createStockSeedGenomes, decodeStockGenome, STOCK_GENE_COUNT, STOCK_TOPOLOGY} from "./strategyGenome";

describe("Pine Script export", () => {
    it("embeds optimized parameters, masks, network layers, and long-only orders", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => Math.sin(index) * 0.4);
        const {parameters, masks} = decodeStockGenome(genome);
        const script = createPineScript(genome, "QQQ");

        expect(script).toContain("//@version=6");
        expect(script).toContain('strategy("EvoLab QQQ Evolved Strategy"');
        expect(script).toContain("default_qty_type=strategy.percent_of_equity");
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
        expect(script).toContain(`useSma = ${masks.sma}`);
        expect(script).toContain(`useRsi = ${masks.rsi}`);
        expect(script).toContain("nDayHigh = ta.highest(high, newHighPeriod)");
        expect(script).toContain("tanh(value) =>");
        expect(script).toContain("outBuy");
        expect(script).toContain("outHold");
        expect(script).toContain("outSell");
        expect(script).toContain("h1_0 = tanh(");
        expect(script).toContain(`f${STOCK_TOPOLOGY.inputSize - 1}`);
        expect(script).toContain("useSma ? clamp");
        expect(script).toContain('strategy.entry("Long", strategy.long)');
        expect(script).toContain('strategy.close("Long")');
        expect(script).not.toContain("strategy.short");
    });

    it("exports threshold-based rule mode without the neural network", () => {
        const genome = createStockSeedGenomes()[0];
        const script = createPineScript(genome, "QQQ", false);
        expect(script).toContain("rsiBuyVote = useRsi and rsi <= rsiBuyThreshold");
        expect(script).toContain("williamsBuyVote = useWilliams and williamsR <= williamsBuyThreshold");
        expect(script).toContain("neededVotes");
        expect(script).toContain("useSma = true");
        expect(script).not.toContain("h1_0 = tanh(");
    });
});
