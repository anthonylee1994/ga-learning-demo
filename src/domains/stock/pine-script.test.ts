import {createPineScript} from "./pine-script";
import {decodeStockGenome, STOCK_GENE_COUNT} from "./strategy-genome";

describe("Pine Script export", () => {
    it("embeds optimized parameters, rule thresholds, and long-only orders", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => Math.sin(index) * 0.4);
        const {parameters, rules} = decodeStockGenome(genome);
        const script = createPineScript(genome, "QQQ");

        expect(script).toContain("//@version=6");
        expect(script).toContain('strategy("EvoLab QQQ Evolved Strategy"');
        expect(script).toContain("default_qty_type=strategy.percent_of_equity");
        expect(script).toContain(`rsiPeriod = ${parameters.rsiPeriod}`);
        expect(script).toContain(`bollingerPeriod = ${parameters.bollingerPeriod}`);
        expect(script).toContain(`williamsPeriod = ${parameters.williamsPeriod}`);
        expect(script).toContain(`volatilityPeriod = ${parameters.volatilityPeriod}`);
        expect(script).toContain(`volumeZScorePeriod = ${parameters.volumeZScorePeriod}`);
        expect(script).toContain(`rsiBuy = ${rules.rsiBuy}`);
        expect(script).toContain(`rsiSell = ${rules.rsiSell}`);
        expect(script).toContain(`minBuySignals = ${rules.minBuySignals}`);
        expect(script).toContain(`minSellSignals = ${rules.minSellSignals}`);
        expect(script).toContain(`useTrendFilter = ${rules.useTrendFilter}`);
        expect(script).toContain("buySignals = 0");
        expect(script).toContain("sellSignals = 0");
        expect(script).not.toContain("tanh(value) =>");
        expect(script).not.toContain("h1_0 = tanh(");
        expect(script).not.toContain("outBuy");
        expect(script).toContain('strategy.entry("Long", strategy.long)');
        expect(script).toContain('strategy.close("Long")');
        expect(script).not.toContain("strategy.short");
    });
});
