import {calculateGeneCount} from "../../lib/neuralNetwork";
import type {Genome, NetworkTopology, OptimizedIndicatorParameters} from "../../lib/types";

/**
 * Normalized indicator features + current position → buy / hold / sell.
 * Kept compact so GA can search the weight space (snake/breaker use similar scale).
 */
export const STOCK_TOPOLOGY: NetworkTopology = {
    inputSize: 12,
    hiddenLayers: [10],
    outputSize: 3,
};

/** Indicator period genes (0–11) + brain.js weights/biases. */
export const STOCK_PARAMETER_GENE_COUNT = 12;
export const STOCK_NETWORK_GENE_COUNT = calculateGeneCount(STOCK_TOPOLOGY);
export const STOCK_GENE_COUNT = STOCK_PARAMETER_GENE_COUNT + STOCK_NETWORK_GENE_COUNT;

export const DEFAULT_INDICATOR_PARAMETERS: OptimizedIndicatorParameters = {
    smaFastPeriod: 20,
    smaSlowPeriod: 50,
    williamsPeriod: 14,
    rocPeriod: 12,
    rsiPeriod: 14,
    macdFastPeriod: 12,
    macdSlowPeriod: 26,
    macdSignalPeriod: 9,
    bollingerPeriod: 20,
    bollingerMultiplier: 2,
    volatilityPeriod: 20,
    volumeZScorePeriod: 20,
};

export interface DecodedStockGenome {
    parameters: OptimizedIndicatorParameters;
    networkGenome: Genome;
}

export function decodeStockGenome(genome: Genome): DecodedStockGenome {
    if (genome.length !== STOCK_GENE_COUNT) {
        throw new Error(`Stock genome length ${genome.length} does not match ${STOCK_GENE_COUNT}`);
    }

    const value = (index: number, min: number, max: number) => decodeInteger(genome[index], min, max);
    const smaFastPeriod = value(0, 5, 40);
    const smaSlowPeriod = Math.max(smaFastPeriod + 5, value(1, 30, 200));
    const macdFastPeriod = value(5, 5, 18);
    const macdSlowPeriod = Math.max(macdFastPeriod + 3, value(6, 20, 50));

    return {
        parameters: {
            smaFastPeriod,
            smaSlowPeriod,
            williamsPeriod: value(2, 5, 40),
            rocPeriod: value(3, 3, 40),
            rsiPeriod: value(4, 5, 40),
            macdFastPeriod,
            macdSlowPeriod,
            macdSignalPeriod: value(7, 3, 15),
            bollingerPeriod: value(8, 10, 60),
            bollingerMultiplier: decodeFloat(genome[9], 1, 3.5, 2),
            volatilityPeriod: value(10, 10, 60),
            volumeZScorePeriod: value(11, 10, 60),
        },
        networkGenome: genome.slice(STOCK_PARAMETER_GENE_COUNT),
    };
}

/**
 * Inverse of decodeFloat: map a real parameter into the raw gene space (approx. artanh).
 * Used to seed the population with known-good classic indicator settings.
 */
export function encodeGene(value: number, min: number, max: number): number {
    const span = max - min;
    if (span <= 0) {
        return 0;
    }
    const normalized = Math.min(0.999, Math.max(0.001, (value - min) / span));
    const centered = 2 * normalized - 1;
    return 0.5 * Math.log((1 + centered) / (1 - centered));
}

/**
 * Classic indicator setups + lightly biased networks so evolution starts from
 * coherent long/cash priors instead of pure noise weights.
 */
export function createStockSeedGenomes(): Genome[] {
    return [
        seedGenome(
            {smaFast: 10, smaSlow: 40, williams: 14, roc: 12, rsi: 14, macdFast: 12, macdSlow: 26, macdSignal: 9, bollinger: 20, bollingerMult: 2, volatility: 20, volumeZ: 20},
            {buyBias: 1.2, holdBias: 0.2, sellBias: -0.8, weightScale: 0.05}
        ),
        seedGenome(
            {smaFast: 20, smaSlow: 50, williams: 14, roc: 12, rsi: 14, macdFast: 12, macdSlow: 26, macdSignal: 9, bollinger: 20, bollingerMult: 2, volatility: 20, volumeZ: 20},
            {buyBias: 0.4, holdBias: 0.1, sellBias: -0.2, weightScale: 0.12}
        ),
        seedGenome(
            {smaFast: 12, smaSlow: 36, williams: 14, roc: 10, rsi: 14, macdFast: 8, macdSlow: 21, macdSignal: 5, bollinger: 20, bollingerMult: 2.2, volatility: 14, volumeZ: 20},
            {buyBias: 0.1, holdBias: 0.3, sellBias: 0.1, weightScale: 0.18}
        ),
        seedGenome(
            {smaFast: 30, smaSlow: 100, williams: 21, roc: 20, rsi: 21, macdFast: 12, macdSlow: 26, macdSignal: 9, bollinger: 30, bollingerMult: 2.5, volatility: 30, volumeZ: 30},
            {buyBias: 0.8, holdBias: 0, sellBias: -0.4, weightScale: 0.08}
        ),
    ];
}

export function describeStockNetwork(): string {
    const hidden = STOCK_TOPOLOGY.hiddenLayers.join(" → ");
    return `${STOCK_TOPOLOGY.inputSize} → ${hidden} → ${STOCK_TOPOLOGY.outputSize} (${STOCK_NETWORK_GENE_COUNT} weights)`;
}

function decodeInteger(gene: number, min: number, max: number): number {
    return Math.round(decodeFloat(gene, min, max));
}

function decodeFloat(gene: number, min: number, max: number, decimals = 6): number {
    const normalized = (Math.tanh(gene) + 1) / 2;
    const value = min + normalized * (max - min);
    return Number(value.toFixed(decimals));
}

function seedGenome(
    periods: {
        smaFast: number;
        smaSlow: number;
        williams: number;
        roc: number;
        rsi: number;
        macdFast: number;
        macdSlow: number;
        macdSignal: number;
        bollinger: number;
        bollingerMult: number;
        volatility: number;
        volumeZ: number;
    },
    network: {buyBias: number; holdBias: number; sellBias: number; weightScale: number}
): Genome {
    const genome = Array.from({length: STOCK_GENE_COUNT}, () => 0);
    genome[0] = encodeGene(periods.smaFast, 5, 40);
    genome[1] = encodeGene(periods.smaSlow, 30, 200);
    genome[2] = encodeGene(periods.williams, 5, 40);
    genome[3] = encodeGene(periods.roc, 3, 40);
    genome[4] = encodeGene(periods.rsi, 5, 40);
    genome[5] = encodeGene(periods.macdFast, 5, 18);
    genome[6] = encodeGene(periods.macdSlow, 20, 50);
    genome[7] = encodeGene(periods.macdSignal, 3, 15);
    genome[8] = encodeGene(periods.bollinger, 10, 60);
    genome[9] = encodeGene(periods.bollingerMult, 1, 3.5);
    genome[10] = encodeGene(periods.volatility, 10, 60);
    genome[11] = encodeGene(periods.volumeZ, 10, 60);

    // Deterministic small weights + explicit output biases (buy / hold / sell).
    const hidden = STOCK_TOPOLOGY.hiddenLayers[0];
    const inputSize = STOCK_TOPOLOGY.inputSize;
    for (let index = 0; index < STOCK_NETWORK_GENE_COUNT; index += 1) {
        // Fixed pseudo-random pattern so seeds are reproducible across reloads.
        const wave = Math.sin((index + 1) * 1.7) * network.weightScale;
        genome[STOCK_PARAMETER_GENE_COUNT + index] = wave;
    }
    const outputBiasStart = hidden + hidden * inputSize;
    genome[STOCK_PARAMETER_GENE_COUNT + outputBiasStart] = network.buyBias;
    genome[STOCK_PARAMETER_GENE_COUNT + outputBiasStart + 1] = network.holdBias;
    genome[STOCK_PARAMETER_GENE_COUNT + outputBiasStart + 2] = network.sellBias;
    return genome;
}
