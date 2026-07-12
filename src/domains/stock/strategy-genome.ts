import {calculateGeneCount} from "../../lib/neural-network";
import type {Genome, OptimizedIndicatorParameters} from "../../lib/types";

export const STOCK_TOPOLOGY = {
    inputSize: 12,
    hiddenLayers: [16, 8],
    outputSize: 3,
};

export const STOCK_PARAMETER_GENE_COUNT = 10;
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
        },
        networkGenome: genome.slice(STOCK_PARAMETER_GENE_COUNT),
    };
}

function decodeInteger(gene: number, min: number, max: number): number {
    return Math.round(decodeFloat(gene, min, max));
}

function decodeFloat(gene: number, min: number, max: number, decimals = 6): number {
    const normalized = (Math.tanh(gene) + 1) / 2;
    const value = min + normalized * (max - min);
    return Number(value.toFixed(decimals));
}
