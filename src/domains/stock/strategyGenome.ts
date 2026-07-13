import {calculateGeneCount} from "../../lib/neuralNetwork";
import type {Genome, NetworkTopology, OptimizedIndicatorParameters} from "../../lib/types";

/**
 * Normalized indicator features, tuned threshold distances, and current position → buy / hold / sell.
 * Thin decision head on purpose: the GA is steered to spend most of its search
 * budget on indicator periods (see STOCK_MUTATION_PROFILE), not a fat hidden net.
 */
export const STOCK_TOPOLOGY: NetworkTopology = {
    inputSize: 17,
    hiddenLayers: [4],
    outputSize: 3,
};

/** Indicator period/threshold genes (0–16) + brain.js weights/biases. */
export const STOCK_PARAMETER_GENE_COUNT = 17;
export const STOCK_NETWORK_GENE_COUNT = calculateGeneCount(STOCK_TOPOLOGY);
export const STOCK_GENE_COUNT = STOCK_PARAMETER_GENE_COUNT + STOCK_NETWORK_GENE_COUNT;

/**
 * Indicator-first mutation: period/threshold genes flip ~3× more often / harder than the NN tail.
 * Immigrants re-roll only indicator parameters so a stable decision head is reused across setups.
 */
export const STOCK_MUTATION_PROFILE = {
    headGeneCount: STOCK_PARAMETER_GENE_COUNT,
    headRateMultiplier: 3,
    headScaleMultiplier: 1.5,
    tailRateMultiplier: 0.35,
    tailScaleMultiplier: 0.45,
    immigrantHeadOnly: true,
} as const;

export const DEFAULT_INDICATOR_PARAMETERS: OptimizedIndicatorParameters = {
    smaFastPeriod: 20,
    smaSlowPeriod: 50,
    williamsPeriod: 14,
    williamsBuyThreshold: -80,
    williamsSellThreshold: -20,
    rocPeriod: 12,
    rsiPeriod: 14,
    rsiBuyThreshold: 30,
    rsiSellThreshold: 70,
    macdFastPeriod: 12,
    macdSlowPeriod: 26,
    macdSignalPeriod: 9,
    bollingerPeriod: 20,
    bollingerMultiplier: 2,
    volatilityPeriod: 20,
    volumeZScorePeriod: 20,
    newHighPeriod: 55,
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
            williamsBuyThreshold: value(15, -95, -55),
            williamsSellThreshold: value(16, -45, -5),
            rocPeriod: value(3, 3, 40),
            rsiPeriod: value(4, 5, 40),
            rsiBuyThreshold: value(13, 10, 45),
            rsiSellThreshold: value(14, 55, 90),
            macdFastPeriod,
            macdSlowPeriod,
            macdSignalPeriod: value(7, 3, 15),
            bollingerPeriod: value(8, 10, 60),
            bollingerMultiplier: decodeFloat(genome[9], 1, 3.5, 2),
            volatilityPeriod: value(10, 10, 60),
            volumeZScorePeriod: value(11, 10, 60),
            newHighPeriod: value(12, 10, 120),
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
            {
                smaFast: 10,
                smaSlow: 40,
                williams: 14,
                williamsBuy: -80,
                williamsSell: -20,
                roc: 12,
                rsi: 14,
                rsiBuy: 30,
                rsiSell: 70,
                macdFast: 12,
                macdSlow: 26,
                macdSignal: 9,
                bollinger: 20,
                bollingerMult: 2,
                volatility: 20,
                volumeZ: 20,
                newHigh: 20,
            },
            {buyBias: 1.2, holdBias: 0.2, sellBias: -0.8, weightScale: 0.05}
        ),
        seedGenome(
            {
                smaFast: 20,
                smaSlow: 50,
                williams: 14,
                williamsBuy: -80,
                williamsSell: -20,
                roc: 12,
                rsi: 14,
                rsiBuy: 30,
                rsiSell: 70,
                macdFast: 12,
                macdSlow: 26,
                macdSignal: 9,
                bollinger: 20,
                bollingerMult: 2,
                volatility: 20,
                volumeZ: 20,
                newHigh: 55,
            },
            {buyBias: 0.4, holdBias: 0.1, sellBias: -0.2, weightScale: 0.12}
        ),
        seedGenome(
            {
                smaFast: 12,
                smaSlow: 36,
                williams: 14,
                williamsBuy: -85,
                williamsSell: -15,
                roc: 10,
                rsi: 14,
                rsiBuy: 25,
                rsiSell: 75,
                macdFast: 8,
                macdSlow: 21,
                macdSignal: 5,
                bollinger: 20,
                bollingerMult: 2.2,
                volatility: 14,
                volumeZ: 20,
                newHigh: 40,
            },
            {buyBias: 0.1, holdBias: 0.3, sellBias: 0.1, weightScale: 0.18}
        ),
        seedGenome(
            {
                smaFast: 30,
                smaSlow: 100,
                williams: 21,
                williamsBuy: -75,
                williamsSell: -25,
                roc: 20,
                rsi: 21,
                rsiBuy: 35,
                rsiSell: 65,
                macdFast: 12,
                macdSlow: 26,
                macdSignal: 9,
                bollinger: 30,
                bollingerMult: 2.5,
                volatility: 30,
                volumeZ: 30,
                newHigh: 100,
            },
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
        williamsBuy: number;
        williamsSell: number;
        roc: number;
        rsi: number;
        rsiBuy: number;
        rsiSell: number;
        macdFast: number;
        macdSlow: number;
        macdSignal: number;
        bollinger: number;
        bollingerMult: number;
        volatility: number;
        volumeZ: number;
        newHigh: number;
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
    genome[12] = encodeGene(periods.newHigh, 10, 120);
    genome[13] = encodeGene(periods.rsiBuy, 10, 45);
    genome[14] = encodeGene(periods.rsiSell, 55, 90);
    genome[15] = encodeGene(periods.williamsBuy, -95, -55);
    genome[16] = encodeGene(periods.williamsSell, -45, -5);

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
