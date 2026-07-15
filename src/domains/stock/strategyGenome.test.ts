import {calculateGeneCount} from "../../lib/neuralNetwork";
import {
    createStockSeedGenomes,
    decodeStockGenome,
    encodeGene,
    STOCK_GENE_COUNT,
    STOCK_HEAD_GENE_COUNT,
    STOCK_MUTATION_PROFILE,
    STOCK_NETWORK_GENE_COUNT,
    STOCK_PARAMETER_GENE_COUNT,
    STOCK_TOPOLOGY,
} from "./strategyGenome";

describe("stock strategy genome", () => {
    it("decodes indicator parameters and network weights", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => (index % 7) * 0.05 - 0.15);
        const decoded = decodeStockGenome(genome);
        expect(STOCK_GENE_COUNT).toBe(STOCK_HEAD_GENE_COUNT + STOCK_NETWORK_GENE_COUNT);
        expect(STOCK_HEAD_GENE_COUNT).toBe(STOCK_PARAMETER_GENE_COUNT);
        expect(STOCK_NETWORK_GENE_COUNT).toBe(calculateGeneCount(STOCK_TOPOLOGY));
        expect(STOCK_TOPOLOGY.hiddenLayers.length).toBeGreaterThanOrEqual(1);
        expect(STOCK_NETWORK_GENE_COUNT).toBeLessThan(400);
        expect(STOCK_MUTATION_PROFILE.headGeneCount).toBe(STOCK_HEAD_GENE_COUNT);
        expect(STOCK_MUTATION_PROFILE.headRateMultiplier).toBeGreaterThan(STOCK_MUTATION_PROFILE.tailRateMultiplier);
        expect(decoded.parameters.smaFastPeriod).toBeGreaterThanOrEqual(5);
        expect(decoded.parameters.smaSlowPeriod).toBeGreaterThan(decoded.parameters.smaFastPeriod);
        expect(decoded.parameters.rsiPeriod).toBeGreaterThanOrEqual(5);
        expect(decoded.parameters.rsiPeriod).toBeLessThanOrEqual(40);
        expect(decoded.parameters.rsiBuyThreshold).toBeGreaterThanOrEqual(10);
        expect(decoded.parameters.rsiBuyThreshold).toBeLessThanOrEqual(45);
        expect(decoded.parameters.rsiSellThreshold).toBeGreaterThanOrEqual(55);
        expect(decoded.parameters.rsiSellThreshold).toBeLessThanOrEqual(90);
        expect(decoded.parameters.rsiBuyThreshold).toBeLessThan(decoded.parameters.rsiSellThreshold);
        expect(decoded.parameters.williamsBuyThreshold).toBeGreaterThanOrEqual(-95);
        expect(decoded.parameters.williamsBuyThreshold).toBeLessThanOrEqual(-55);
        expect(decoded.parameters.williamsSellThreshold).toBeGreaterThanOrEqual(-45);
        expect(decoded.parameters.williamsSellThreshold).toBeLessThanOrEqual(-5);
        expect(decoded.parameters.williamsBuyThreshold).toBeLessThan(decoded.parameters.williamsSellThreshold);
        expect(decoded.parameters.bollingerMultiplier).toBeGreaterThanOrEqual(1);
        expect(decoded.parameters.bollingerMultiplier).toBeLessThanOrEqual(3.5);
        expect(decoded.parameters.macdSlowPeriod).toBeGreaterThan(decoded.parameters.macdFastPeriod);
        expect(decoded.parameters.newHighPeriod).toBeGreaterThanOrEqual(10);
        expect(decoded.parameters.newHighPeriod).toBeLessThanOrEqual(120);
        expect(decoded.parameters.newLowPeriod).toBeGreaterThanOrEqual(10);
        expect(decoded.parameters.newLowPeriod).toBeLessThanOrEqual(120);
        expect(decoded.networkGenome).toHaveLength(STOCK_NETWORK_GENE_COUNT);
    });

    it("rejects genomes with the wrong length", () => {
        expect(() => decodeStockGenome(Array(STOCK_PARAMETER_GENE_COUNT).fill(0))).toThrow(/Stock genome length/);
        expect(() => decodeStockGenome(Array(100).fill(0))).toThrow(/Stock genome length/);
    });

    it("seeds classic indicator setups with full genome length", () => {
        const seeds = createStockSeedGenomes();
        expect(seeds.length).toBeGreaterThanOrEqual(3);
        seeds.forEach(genome => {
            expect(genome).toHaveLength(STOCK_GENE_COUNT);
            const decoded = decodeStockGenome(genome);
            expect(decoded.networkGenome.length).toBe(STOCK_NETWORK_GENE_COUNT);
        });
    });

    it("round-trips encodeGene through tanh decode range", () => {
        const gene = encodeGene(20, 5, 40);
        const normalized = (Math.tanh(gene) + 1) / 2;
        const value = 5 + normalized * (40 - 5);
        expect(value).toBeCloseTo(20, 5);
    });
});
