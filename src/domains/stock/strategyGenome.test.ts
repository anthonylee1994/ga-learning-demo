import {calculateGeneCount} from "../../lib/neuralNetwork";
import {
    countActiveMasks,
    createStockSeedGenomes,
    decodeMask,
    decodeStockGenome,
    encodeGene,
    encodeMask,
    STOCK_GENE_COUNT,
    STOCK_HEAD_GENE_COUNT,
    STOCK_MASK_GENE_COUNT,
    STOCK_MUTATION_PROFILE,
    STOCK_NETWORK_GENE_COUNT,
    STOCK_PARAMETER_GENE_COUNT,
    STOCK_TOPOLOGY,
    withMaskOverride,
} from "./strategyGenome";

describe("stock strategy genome", () => {
    it("decodes indicator parameters, masks, and network weights", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => (index % 7) * 0.05 - 0.15);
        // Force first three masks on, rest off for a deterministic decode check.
        for (let index = 0; index < STOCK_MASK_GENE_COUNT; index += 1) {
            genome[STOCK_PARAMETER_GENE_COUNT + index] = index < 3 ? 1.2 : -1.2;
        }
        const decoded = decodeStockGenome(genome);
        expect(STOCK_GENE_COUNT).toBe(STOCK_HEAD_GENE_COUNT + STOCK_NETWORK_GENE_COUNT);
        expect(STOCK_HEAD_GENE_COUNT).toBe(STOCK_PARAMETER_GENE_COUNT + STOCK_MASK_GENE_COUNT);
        expect(STOCK_MASK_GENE_COUNT).toBe(9);
        expect(STOCK_NETWORK_GENE_COUNT).toBe(calculateGeneCount(STOCK_TOPOLOGY));
        expect(STOCK_TOPOLOGY.hiddenLayers).toEqual([10]);
        expect(STOCK_NETWORK_GENE_COUNT).toBeLessThan(250);
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
        expect(decoded.networkGenome).toHaveLength(STOCK_NETWORK_GENE_COUNT);
        expect(decoded.masks.sma).toBe(true);
        expect(decoded.masks.williams).toBe(true);
        expect(decoded.masks.roc).toBe(true);
        expect(decoded.masks.rsi).toBe(false);
        expect(countActiveMasks(decoded.masks)).toBe(3);
    });

    it("rejects genomes with the wrong length", () => {
        expect(() => decodeStockGenome(Array(STOCK_PARAMETER_GENE_COUNT).fill(0))).toThrow(/Stock genome length/);
        expect(() => decodeStockGenome(Array(100).fill(0))).toThrow(/Stock genome length/);
    });

    it("seeds classic indicator setups with full genome length and varied masks", () => {
        const seeds = createStockSeedGenomes();
        expect(seeds.length).toBeGreaterThanOrEqual(3);
        const activeCounts = new Set<number>();
        seeds.forEach(genome => {
            expect(genome).toHaveLength(STOCK_GENE_COUNT);
            const decoded = decodeStockGenome(genome);
            expect(decoded.networkGenome.length).toBe(STOCK_NETWORK_GENE_COUNT);
            activeCounts.add(countActiveMasks(decoded.masks));
        });
        // At least one dense and one sparse prior.
        expect(Math.max(...activeCounts)).toBe(STOCK_MASK_GENE_COUNT);
        expect(Math.min(...activeCounts)).toBeLessThan(STOCK_MASK_GENE_COUNT);
    });

    it("round-trips encodeGene through tanh decode range", () => {
        const gene = encodeGene(20, 5, 40);
        const normalized = (Math.tanh(gene) + 1) / 2;
        const value = 5 + normalized * (40 - 5);
        expect(value).toBeCloseTo(20, 5);
    });

    it("encodes and overrides mask genes", () => {
        expect(decodeMask(encodeMask(true))).toBe(true);
        expect(decodeMask(encodeMask(false))).toBe(false);
        const genome = createStockSeedGenomes()[0].slice();
        const off = withMaskOverride(genome, "rsi", false);
        expect(decodeStockGenome(off).masks.rsi).toBe(false);
        const on = withMaskOverride(off, "rsi", true);
        expect(decodeStockGenome(on).masks.rsi).toBe(true);
    });
});
