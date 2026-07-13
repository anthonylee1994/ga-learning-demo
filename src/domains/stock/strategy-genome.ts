import type {Genome, OptimizedIndicatorParameters, OptimizedStrategyRules} from "../../lib/types";

/** Indicator period genes (0–11) + rule-threshold genes (12–22). Pure GA — no neural weights. */
export const STOCK_PARAMETER_GENE_COUNT = 12;
export const STOCK_RULE_GENE_COUNT = 11;
export const STOCK_GENE_COUNT = STOCK_PARAMETER_GENE_COUNT + STOCK_RULE_GENE_COUNT;

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

export const DEFAULT_STRATEGY_RULES: OptimizedStrategyRules = {
    rsiBuy: 30,
    rsiSell: 70,
    williamsBuy: -80,
    williamsSell: -20,
    rocBuy: 0,
    rocSell: 0,
    bollingerBuy: 0.2,
    bollingerSell: 0.8,
    minBuySignals: 3,
    minSellSignals: 3,
    useTrendFilter: true,
};

export interface DecodedStockGenome {
    parameters: OptimizedIndicatorParameters;
    rules: OptimizedStrategyRules;
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
    const rsiBuy = value(12, 15, 45);
    const rsiSell = Math.max(rsiBuy + 10, value(13, 55, 85));
    const williamsBuy = decodeFloat(genome[14], -95, -50, 2);
    const williamsSell = Math.max(williamsBuy + 5, decodeFloat(genome[15], -50, -5, 2));
    const bollingerBuy = decodeFloat(genome[18], 0, 0.4, 3);
    const bollingerSell = Math.max(bollingerBuy + 0.15, decodeFloat(genome[19], 0.6, 1, 3));

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
        rules: {
            rsiBuy,
            rsiSell,
            williamsBuy,
            williamsSell,
            rocBuy: decodeFloat(genome[16], -0.05, 0.1, 4),
            rocSell: decodeFloat(genome[17], -0.1, 0.05, 4),
            bollingerBuy,
            bollingerSell,
            minBuySignals: value(20, 1, 5),
            minSellSignals: value(21, 1, 5),
            useTrendFilter: Math.tanh(genome[22]) >= 0,
        },
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
