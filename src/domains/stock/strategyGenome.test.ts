import {createStockSeedGenomes, decodeStockGenome, encodeGene, STOCK_GENE_COUNT, STOCK_PARAMETER_GENE_COUNT, STOCK_RULE_GENE_COUNT, STRATEGY_STYLES} from "./strategyGenome";

describe("stock strategy genome", () => {
    it("decodes bounded indicator parameters and rule thresholds", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => (index % 3) - 1);
        const decoded = decodeStockGenome(genome);
        expect(STOCK_GENE_COUNT).toBe(STOCK_PARAMETER_GENE_COUNT + STOCK_RULE_GENE_COUNT);
        expect(decoded.parameters.smaFastPeriod).toBeGreaterThanOrEqual(5);
        expect(decoded.parameters.smaSlowPeriod).toBeGreaterThan(decoded.parameters.smaFastPeriod);
        expect(decoded.parameters.rsiPeriod).toBeGreaterThanOrEqual(5);
        expect(decoded.parameters.rsiPeriod).toBeLessThanOrEqual(40);
        expect(decoded.parameters.bollingerMultiplier).toBeGreaterThanOrEqual(1);
        expect(decoded.parameters.bollingerMultiplier).toBeLessThanOrEqual(3.5);
        expect(decoded.parameters.macdSlowPeriod).toBeGreaterThan(decoded.parameters.macdFastPeriod);
        expect(decoded.parameters.volatilityPeriod).toBeGreaterThanOrEqual(10);
        expect(decoded.parameters.volatilityPeriod).toBeLessThanOrEqual(60);
        expect(decoded.parameters.volumeZScorePeriod).toBeGreaterThanOrEqual(10);
        expect(decoded.parameters.volumeZScorePeriod).toBeLessThanOrEqual(60);
        expect(decoded.rules.rsiBuy).toBeGreaterThanOrEqual(15);
        expect(decoded.rules.rsiSell).toBeGreaterThan(decoded.rules.rsiBuy);
        expect(decoded.rules.williamsSell).toBeGreaterThan(decoded.rules.williamsBuy);
        expect(decoded.rules.bollingerSell).toBeGreaterThan(decoded.rules.bollingerBuy);
        expect(decoded.rules.minBuySignals).toBeGreaterThanOrEqual(1);
        expect(decoded.rules.minBuySignals).toBeLessThanOrEqual(5);
        expect(decoded.rules.minSellSignals).toBeGreaterThanOrEqual(1);
        expect(decoded.rules.minSellSignals).toBeLessThanOrEqual(5);
        expect(STRATEGY_STYLES).toContain(decoded.rules.strategyStyle);
    });

    it("rejects genomes with the wrong length", () => {
        expect(() => decodeStockGenome(Array(STOCK_PARAMETER_GENE_COUNT).fill(0))).toThrow(/Stock genome length/);
        expect(() => decodeStockGenome(Array(100).fill(0))).toThrow(/Stock genome length/);
    });

    it("seeds classic strategies with valid gene length and styles", () => {
        const seeds = createStockSeedGenomes();
        expect(seeds.length).toBeGreaterThanOrEqual(3);
        const styles = new Set(seeds.map(genome => decodeStockGenome(genome).rules.strategyStyle));
        expect(styles.has("trend")).toBe(true);
        expect(styles.has("mean_reversion")).toBe(true);
        seeds.forEach(genome => {
            expect(genome).toHaveLength(STOCK_GENE_COUNT);
            expect(() => decodeStockGenome(genome)).not.toThrow();
        });
    });

    it("round-trips encodeGene through tanh decode range", () => {
        const gene = encodeGene(20, 5, 40);
        const normalized = (Math.tanh(gene) + 1) / 2;
        const value = 5 + normalized * (40 - 5);
        expect(value).toBeCloseTo(20, 5);
    });
});
