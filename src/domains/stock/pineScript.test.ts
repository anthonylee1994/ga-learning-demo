import {createPineScript} from "./pineScript";
import {decodeStockGenome, STOCK_GENE_COUNT, STOCK_TOPOLOGY} from "./strategyGenome";

describe("Pine Script export", () => {
    it("embeds optimized parameters, network layers, and long-only orders", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => Math.sin(index) * 0.4);
        const {parameters} = decodeStockGenome(genome);
        const script = createPineScript(genome, "QQQ");

        expect(script).toContain("//@version=6");
        expect(script).toContain('strategy("EvoLab QQQ Evolved Strategy"');
        expect(script).toContain("default_qty_type=strategy.percent_of_equity");
        expect(script).toContain(`rsiPeriod = ${parameters.rsiPeriod}`);
        expect(script).toContain(`bollingerPeriod = ${parameters.bollingerPeriod}`);
        expect(script).toContain(`williamsPeriod = ${parameters.williamsPeriod}`);
        expect(script).toContain(`volatilityPeriod = ${parameters.volatilityPeriod}`);
        expect(script).toContain(`volumeZScorePeriod = ${parameters.volumeZScorePeriod}`);
        expect(script).toContain("tanh(value) =>");
        expect(script).toContain("outBuy");
        expect(script).toContain("outHold");
        expect(script).toContain("outSell");
        expect(script).toContain("h1_0 = tanh(");
        expect(script).toContain(`f${STOCK_TOPOLOGY.inputSize - 1}`);
        expect(script).toContain('strategy.entry("Long", strategy.long)');
        expect(script).toContain('strategy.close("Long")');
        expect(script).not.toContain("strategy.short");
    });
});
