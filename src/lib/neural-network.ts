import {NeuralNetwork} from "brain.js";
import type {Genome, NetworkTopology} from "./types";

export class NeuralNetworkAdapter {
    readonly topology: NetworkTopology;
    readonly geneCount: number;
    private network: NeuralNetwork<number[], number[]> | null = null;

    constructor(topology: NetworkTopology) {
        this.topology = topology;
        this.geneCount = calculateGeneCount(topology);
    }

    run(genome: Genome, input: number[]): number[] {
        return this.createRunner(genome)(input);
    }

    createRunner(genome: Genome): (input: number[]) => number[] {
        if (genome.length !== this.geneCount) {
            throw new Error(`Genome length ${genome.length} does not match ${this.geneCount}`);
        }

        // Reuse one network instance per adapter — avoid allocating a fresh
        // brain.js graph on every genome evaluation (was a major GC pressure source).
        if (!this.network) {
            this.network = new NeuralNetwork<number[], number[]>({
                inputSize: this.topology.inputSize,
                hiddenLayers: this.topology.hiddenLayers,
                outputSize: this.topology.outputSize,
                activation: "tanh",
            });
            this.network.initialize();
        }

        applyGenome(this.network, genome);
        const network = this.network;
        return (input: number[]) => {
            if (input.length !== this.topology.inputSize) {
                throw new Error(`Input length ${input.length} does not match ${this.topology.inputSize}`);
            }
            return Array.from(network.run(input));
        };
    }
}

export function calculateGeneCount(topology: NetworkTopology): number {
    const sizes = [topology.inputSize, ...topology.hiddenLayers, topology.outputSize];
    let count = 0;
    for (let layer = 1; layer < sizes.length; layer += 1) {
        count += sizes[layer];
        count += sizes[layer] * sizes[layer - 1];
    }
    return count;
}

function applyGenome(network: NeuralNetwork<number[], number[]>, genome: Genome): void {
    let cursor = 0;
    for (let layer = 1; layer < network.sizes.length; layer += 1) {
        for (let node = 0; node < network.sizes[layer]; node += 1) {
            network.biases[layer][node] = genome[cursor];
            cursor += 1;
        }
        for (let node = 0; node < network.sizes[layer]; node += 1) {
            for (let input = 0; input < network.sizes[layer - 1]; input += 1) {
                network.weights[layer][node][input] = genome[cursor];
                cursor += 1;
            }
        }
    }
}

export function argMax(values: number[]): number {
    return values.reduce((bestIndex, value, index) => (value > values[bestIndex] ? index : bestIndex), 0);
}
