import {calculateGeneCount, NeuralNetworkAdapter} from "./neuralNetwork";

describe("NeuralNetworkAdapter", () => {
    const topology = {inputSize: 3, hiddenLayers: [4], outputSize: 2};

    it("calculates weights and biases for every non-input layer", () => {
        expect(calculateGeneCount(topology)).toBe(26);
    });

    it("runs a deterministic Brain.js network from a genome", () => {
        const adapter = new NeuralNetworkAdapter(topology);
        const genome = Array.from({length: adapter.geneCount}, (_, index) => (index - 10) / 20);
        const runner = adapter.createRunner(genome);
        expect(runner([0.2, -0.4, 0.7])).toEqual(runner([0.2, -0.4, 0.7]));
        expect(runner([0.2, -0.4, 0.7])).toHaveLength(2);
    });

    it("rejects an invalid genome", () => {
        const adapter = new NeuralNetworkAdapter(topology);
        expect(() => adapter.run([1, 2], [0, 0, 0])).toThrow(/Genome length/);
    });
});
