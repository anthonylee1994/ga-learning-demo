import {decodeStockGenome, STOCK_DECISION_GENE_COUNT, STOCK_FEATURE_COUNT, STOCK_GENE_COUNT} from "./strategy-genome";

describe("stock strategy genome", () => {
    it("decodes bounded indicator parameters and a linear signal policy", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => (index % 3) - 1);
        const decoded = decodeStockGenome(genome);
        expect(decoded.strategy.weights).toHaveLength(STOCK_FEATURE_COUNT);
        decoded.strategy.weights.forEach(weight => {
            expect(weight).toBeGreaterThanOrEqual(-2);
            expect(weight).toBeLessThanOrEqual(2);
        });
        expect(decoded.strategy.enterThreshold).toBeGreaterThanOrEqual(0);
        expect(decoded.strategy.enterThreshold).toBeLessThanOrEqual(0.8);
        expect(decoded.strategy.exitThreshold).toBeGreaterThanOrEqual(-0.8);
        expect(decoded.strategy.exitThreshold).toBeLessThanOrEqual(0);
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

    it("rejects genomes that omit the decision genes", () => {
        expect(() => decodeStockGenome(Array(STOCK_GENE_COUNT - STOCK_DECISION_GENE_COUNT).fill(0))).toThrow(/Stock genome length/);
    });
});
