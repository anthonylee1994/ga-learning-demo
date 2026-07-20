import {calculateGeneCount, createForwardRunner, forwardWithActivations, inspectGenome, NeuralNetworkAdapter} from "./neuralNetwork";

describe("NeuralNetworkAdapter", () => {
    const topology = {inputSize: 3, hiddenLayers: [4], outputSize: 2};

    it("calculates weights and biases for every non-input layer", () => {
        expect(calculateGeneCount(topology)).toBe(26);
    });

    it("runs a deterministic pure forward network from a genome", () => {
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

    it("matches pure forward activations with TensorFlow.js outputs", () => {
        const adapter = new NeuralNetworkAdapter(topology);
        const genome = Array.from({length: adapter.geneCount}, (_, index) => Math.sin(index * 0.37) * 0.8);
        const input = [0.2, -0.4, 0.7];
        const tfOut = adapter.createTfjsRunner(genome)(input);
        const pureOut = createForwardRunner(genome, topology)(input);
        const detailed = forwardWithActivations(genome, topology, input);
        expect(detailed.outputs).toHaveLength(2);
        expect(detailed.activations).toHaveLength(3);
        expect(detailed.activations[0]).toEqual(input);
        for (let index = 0; index < tfOut.length; index += 1) {
            expect(detailed.outputs[index]).toBeCloseTo(tfOut[index], 5);
            expect(pureOut[index]).toBeCloseTo(tfOut[index], 5);
        }
        expect(detailed.decision).toBe(tfOut[0] >= tfOut[1] ? 0 : 1);
    });

    it("inspects genome into layer matrices with the right shapes", () => {
        const genome = Array.from({length: calculateGeneCount(topology)}, (_, index) => index * 0.01);
        const inspection = inspectGenome(genome, topology);
        expect(inspection.sizes).toEqual([3, 4, 2]);
        expect(inspection.layers).toHaveLength(2);
        expect(inspection.layers[0].biases).toHaveLength(4);
        expect(inspection.layers[0].weights).toHaveLength(4);
        expect(inspection.layers[0].weights[0]).toHaveLength(3);
        expect(inspection.layers[1].biases).toHaveLength(2);
        expect(inspection.layers[1].weights[0]).toHaveLength(4);
    });
});
