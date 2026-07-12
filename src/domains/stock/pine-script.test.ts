import {createPineScript} from "./pine-script";
import {decodeStockGenome, STOCK_GENE_COUNT} from "./strategy-genome";

describe("Pine Script export", () => {
    it("embeds optimized parameters, neural weights, and long-only orders", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => Math.sin(index) * 0.4);
        const parameters = decodeStockGenome(genome).parameters;
        const script = createPineScript(genome, "QQQ");

        expect(script).toContain("//@version=6");
        expect(script).toContain('strategy("EvoLab QQQ Neuroevolution"');
        expect(script).toContain(`rsiPeriod = ${parameters.rsiPeriod}`);
        expect(script).toContain(`bollingerPeriod = ${parameters.bollingerPeriod}`);
        expect(script).toContain(`williamsPeriod = ${parameters.williamsPeriod}`);
        expect(script).toContain("tanh(value) =>");
        expect(script).toContain("exponent = math.exp(2.0 * limited)");
        expect(script).toContain("h1_0 = tanh(");
        expect(script).not.toContain("math.tanh");
        expect(script).toContain('strategy.entry("Long", strategy.long)');
        expect(script).toContain('strategy.close("Long")');
        expect(script).not.toContain("strategy.short");
    });
});
