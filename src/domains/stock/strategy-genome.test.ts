import {decodeStockGenome, STOCK_GENE_COUNT, STOCK_NETWORK_GENE_COUNT} from "./strategy-genome";

describe("stock strategy genome", () => {
    it("decodes bounded indicator parameters and preserves the Brain.js weights", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => (index % 3) - 1);
        const decoded = decodeStockGenome(genome);
        expect(decoded.networkGenome).toHaveLength(STOCK_NETWORK_GENE_COUNT);
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
    });

    it("rejects legacy network-only genomes", () => {
        expect(() => decodeStockGenome(Array(STOCK_NETWORK_GENE_COUNT).fill(0))).toThrow(/Stock genome length/);
    });
});
