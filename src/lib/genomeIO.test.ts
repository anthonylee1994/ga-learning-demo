import {calculateGeneCount} from "./neuralNetwork";
import {buildGenomeFile, parseGenomeFile} from "./genomeIO";

const SNAKE_TOPOLOGY = {inputSize: 10, hiddenLayers: [12], outputSize: 3};

describe("genomeIO", () => {
    const geneCount = calculateGeneCount(SNAKE_TOPOLOGY);
    const genome = Array.from({length: geneCount}, (_, index) => Math.sin(index * 0.2) * 0.5);

    it("round-trips a versioned snake genome file", () => {
        const file = buildGenomeFile({
            topic: "snake",
            topology: SNAKE_TOPOLOGY,
            genome,
            fitness: 12.5,
            score: 3,
            steps: 140,
            exportedAt: "2026-01-01T00:00:00.000Z",
        });
        expect(file.format).toBe("evolab-genome");
        expect(file.version).toBe(1);
        expect(file.genome).toHaveLength(geneCount);

        const parsed = parseGenomeFile(file, {topic: "snake", topology: SNAKE_TOPOLOGY});
        expect(parsed).toEqual(genome);
    });

    it("accepts a bare number array", () => {
        expect(parseGenomeFile(genome, {topic: "snake", topology: SNAKE_TOPOLOGY})).toEqual(genome);
    });

    it("rejects wrong length", () => {
        expect(() => parseGenomeFile([1, 2, 3], {topic: "snake", topology: SNAKE_TOPOLOGY})).toThrow(/長度/);
    });

    it("rejects another lab topic", () => {
        const file = buildGenomeFile({topic: "breaker", topology: SNAKE_TOPOLOGY, genome});
        expect(() => parseGenomeFile(file, {topic: "snake", topology: SNAKE_TOPOLOGY})).toThrow(/breaker/);
    });

    it("rejects mismatched topology", () => {
        const file = buildGenomeFile({
            topic: "snake",
            topology: {inputSize: 4, hiddenLayers: [2], outputSize: 2},
            genome: Array.from({length: calculateGeneCount({inputSize: 4, hiddenLayers: [2], outputSize: 2})}, () => 0),
        });
        // Force snake topic so only topology fails.
        file.topic = "snake";
        expect(() => parseGenomeFile(file, {topic: "snake", topology: SNAKE_TOPOLOGY})).toThrow(/Topology/);
    });

    it("supports composite geneCount larger than the NN topology (stock-style)", () => {
        const topology = {inputSize: 17, hiddenLayers: [4], outputSize: 3};
        const networkGenes = calculateGeneCount(topology);
        const compositeCount = networkGenes + 17;
        const composite = Array.from({length: compositeCount}, (_, index) => index * 0.01);
        const file = buildGenomeFile({
            topic: "stock",
            topology,
            geneCount: compositeCount,
            genome: composite,
            fitness: 1.2,
        });
        expect(file.geneCount).toBe(compositeCount);
        expect(parseGenomeFile(file, {topic: "stock", topology, geneCount: compositeCount})).toEqual(composite);
        expect(() => parseGenomeFile(file, {topic: "stock", topology})).toThrow(/長度/);
    });

    it("allows stock ↔ stock-mc genome interchange", () => {
        const topology = {inputSize: 2, hiddenLayers: [2], outputSize: 1};
        const composite = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        const file = buildGenomeFile({
            topic: "stock",
            topology,
            geneCount: composite.length,
            genome: composite,
        });
        expect(parseGenomeFile(file, {topic: "stock-mc", topology, geneCount: composite.length})).toEqual(composite);

        const mcFile = buildGenomeFile({
            topic: "stock-mc",
            topology,
            geneCount: composite.length,
            genome: composite,
        });
        expect(parseGenomeFile(mcFile, {topic: "stock", topology, geneCount: composite.length})).toEqual(composite);
    });
});
