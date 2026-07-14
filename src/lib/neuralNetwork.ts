import {NeuralNetwork} from "brain.js";
import type {Genome, NetworkTopology} from "./types";

export interface LayerParams {
    /** Bias per neuron in this layer. */
    biases: number[];
    /** weights[node][prevNode] — incoming edges into each neuron. */
    weights: number[][];
}

export interface NetworkInspection {
    sizes: number[];
    /** One entry per non-input layer (hidden…output). */
    layers: LayerParams[];
}

export interface NetworkForwardPass {
    /** Activations for every layer, including the input layer at index 0. */
    activations: number[][];
    outputs: number[];
    /** argMax of outputs. */
    decision: number;
}

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

    /**
     * Pure forward pass that also returns per-layer activations.
     * Used by UI visualizers — keeps brain.js graph out of the animation path.
     */
    runDetailed(genome: Genome, input: number[]): NetworkForwardPass {
        return forwardWithActivations(genome, this.topology, input);
    }

    inspect(genome: Genome): NetworkInspection {
        return inspectGenome(genome, this.topology);
    }

    createRunner(genome: Genome): (input: number[]) => number[] {
        // Pure JS forward is faster than brain.js on the stock hot path (thousands of
        // bar-steps × population × generations) and matches tanh stack used elsewhere.
        return createForwardRunner(genome, this.topology);
    }

    /** brain.js path kept for adapters that need a live Network instance. */
    createBrainRunner(genome: Genome): (input: number[]) => number[] {
        if (genome.length !== this.geneCount) {
            throw new Error(`Genome length ${genome.length} does not match ${this.geneCount}`);
        }
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

/**
 * Zero-allocation-friendly pure forward pass (tanh every layer).
 * Decode weights once; reuse layer buffers across calls — critical for stock fitness.
 */
export function createForwardRunner(genome: Genome, topology: NetworkTopology): (input: number[]) => number[] {
    const inspection = inspectGenome(genome, topology);
    const buffers = inspection.layers.map(layer => new Array<number>(layer.biases.length));
    const inputSize = topology.inputSize;

    return (input: number[]) => {
        if (input.length !== inputSize) {
            throw new Error(`Input length ${input.length} does not match ${inputSize}`);
        }
        let prev = input;
        for (let layerIndex = 0; layerIndex < inspection.layers.length; layerIndex += 1) {
            const {biases, weights} = inspection.layers[layerIndex];
            const next = buffers[layerIndex];
            for (let node = 0; node < biases.length; node += 1) {
                let sum = biases[node];
                const inbound = weights[node];
                for (let prevNode = 0; prevNode < inbound.length; prevNode += 1) {
                    sum += inbound[prevNode] * prev[prevNode];
                }
                next[node] = Math.tanh(sum);
            }
            prev = next;
        }
        return prev;
    };
}

export function calculateGeneCount(topology: NetworkTopology): number {
    const sizes = layerSizes(topology);
    let count = 0;
    for (let layer = 1; layer < sizes.length; layer += 1) {
        count += sizes[layer];
        count += sizes[layer] * sizes[layer - 1];
    }
    return count;
}

export function layerSizes(topology: NetworkTopology): number[] {
    return [topology.inputSize, ...topology.hiddenLayers, topology.outputSize];
}

/** Decode a flat genome into per-layer biases + weight matrices. */
export function inspectGenome(genome: Genome, topology: NetworkTopology): NetworkInspection {
    const sizes = layerSizes(topology);
    const expected = calculateGeneCount(topology);
    if (genome.length !== expected) {
        throw new Error(`Genome length ${genome.length} does not match ${expected}`);
    }

    const layers: LayerParams[] = [];
    let cursor = 0;
    for (let layer = 1; layer < sizes.length; layer += 1) {
        const nodeCount = sizes[layer];
        const prevCount = sizes[layer - 1];
        const biases = genome.slice(cursor, cursor + nodeCount);
        cursor += nodeCount;
        const weights: number[][] = [];
        for (let node = 0; node < nodeCount; node += 1) {
            weights.push(genome.slice(cursor, cursor + prevCount));
            cursor += prevCount;
        }
        layers.push({biases, weights});
    }

    return {sizes, layers};
}

/**
 * Feed-forward with tanh, matching brain.js layout used by `applyGenome`.
 * Returns activations for every layer so the UI can light up nodes live.
 */
export function forwardWithActivations(genome: Genome, topology: NetworkTopology, input: number[]): NetworkForwardPass {
    if (input.length !== topology.inputSize) {
        throw new Error(`Input length ${input.length} does not match ${topology.inputSize}`);
    }

    const inspection = inspectGenome(genome, topology);
    const activations: number[][] = [Array.from(input)];

    for (let layerIndex = 0; layerIndex < inspection.layers.length; layerIndex += 1) {
        const prev = activations[layerIndex];
        const {biases, weights} = inspection.layers[layerIndex];
        const next = biases.map((bias, node) => {
            let sum = bias;
            const inbound = weights[node];
            for (let prevNode = 0; prevNode < prev.length; prevNode += 1) {
                sum += inbound[prevNode] * prev[prevNode];
            }
            return Math.tanh(sum);
        });
        activations.push(next);
    }

    const outputs = activations[activations.length - 1];
    return {activations, outputs, decision: argMax(outputs)};
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
