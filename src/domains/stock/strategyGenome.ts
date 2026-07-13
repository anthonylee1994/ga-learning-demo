import type {Genome, OptimizedIndicatorParameters, OptimizedStrategyRules, StrategyStyle} from "../../lib/types";

/** Indicator period genes (0–11) + rule-threshold genes (12–22). Pure GA — no neural weights. */
export const STOCK_PARAMETER_GENE_COUNT = 12;
export const STOCK_RULE_GENE_COUNT = 11;
export const STOCK_GENE_COUNT = STOCK_PARAMETER_GENE_COUNT + STOCK_RULE_GENE_COUNT;

export const STRATEGY_STYLES: StrategyStyle[] = ["trend", "mean_reversion", "hybrid"];

/** Skip new long entries when annualized vol is above this (period is still evolved). */
export const MAX_ENTRY_VOLATILITY = 0.55;
/** Trend/hybrid: volume Z-score above this counts as a breakout confirmation. */
export const VOLUME_Z_CONFIRM = 0.5;

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
    minBuySignals: 2,
    minSellSignals: 2,
    strategyStyle: "trend",
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
    const strategyStyle = STRATEGY_STYLES[value(22, 0, 2)] ?? "trend";
    // Cap required confirmations to signals that style can actually produce.
    const maxSignals = maxSignalsForStyle(strategyStyle);
    const minBuySignals = Math.min(value(20, 1, 5), maxSignals);
    const minSellSignals = Math.min(value(21, 1, 5), maxSignals);

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
            minBuySignals,
            minSellSignals,
            strategyStyle,
        },
    };
}

/**
 * Inverse of decodeFloat: map a real parameter into the raw gene space (approx. artanh).
 * Used to seed the population with known-good classic strategies.
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

/** Classic baselines so evolution starts from strategies that already trade coherently. */
export function createStockSeedGenomes(): Genome[] {
    return [seedBuyAndHold(), seedSmaCross(), seedRsiMeanReversion(), seedHybridSwing()];
}

export function maxSignalsForStyle(style: StrategyStyle): number {
    if (style === "trend") {
        // SMA, MACD, ROC, volume-Z confirmation
        return 4;
    }
    if (style === "mean_reversion") {
        // RSI, Williams, Bollinger %B
        return 3;
    }
    // trend trio + volume + three reversion
    return 7;
}

export function strategyStyleLabel(style: StrategyStyle): string {
    if (style === "trend") {
        return "Trend (SMA/MACD/ROC)";
    }
    if (style === "mean_reversion") {
        return "Mean reversion (RSI/WR/BB)";
    }
    return "Hybrid (all families)";
}

function decodeInteger(gene: number, min: number, max: number): number {
    return Math.round(decodeFloat(gene, min, max));
}

function decodeFloat(gene: number, min: number, max: number, decimals = 6): number {
    const normalized = (Math.tanh(gene) + 1) / 2;
    const value = min + normalized * (max - min);
    return Number(value.toFixed(decimals));
}

function seedBuyAndHold(): Genome {
    // Stay long: trend style, easy entry, hard exit.
    const genome = Array.from({length: STOCK_GENE_COUNT}, () => 0);
    writePeriodGenes(genome, {smaFast: 10, smaSlow: 40, williams: 14, roc: 12, rsi: 14, macdFast: 12, macdSlow: 26, macdSignal: 9, bollinger: 20, bollingerMult: 2, volatility: 20, volumeZ: 20});
    writeRuleGenes(genome, {
        rsiBuy: 40,
        rsiSell: 80,
        williamsBuy: -70,
        williamsSell: -15,
        rocBuy: -0.02,
        rocSell: -0.08,
        bollingerBuy: 0.35,
        bollingerSell: 0.95,
        minBuy: 1,
        minSell: 4,
        style: 0,
    });
    return genome;
}

function seedSmaCross(): Genome {
    const genome = Array.from({length: STOCK_GENE_COUNT}, () => 0);
    writePeriodGenes(genome, {smaFast: 20, smaSlow: 50, williams: 14, roc: 12, rsi: 14, macdFast: 12, macdSlow: 26, macdSignal: 9, bollinger: 20, bollingerMult: 2, volatility: 20, volumeZ: 20});
    writeRuleGenes(genome, {
        rsiBuy: 25,
        rsiSell: 75,
        williamsBuy: -85,
        williamsSell: -15,
        rocBuy: 0,
        rocSell: 0,
        bollingerBuy: 0.15,
        bollingerSell: 0.85,
        minBuy: 2,
        minSell: 2,
        style: 0,
    });
    return genome;
}

function seedRsiMeanReversion(): Genome {
    const genome = Array.from({length: STOCK_GENE_COUNT}, () => 0);
    writePeriodGenes(genome, {smaFast: 10, smaSlow: 30, williams: 14, roc: 10, rsi: 14, macdFast: 12, macdSlow: 26, macdSignal: 9, bollinger: 20, bollingerMult: 2, volatility: 20, volumeZ: 20});
    writeRuleGenes(genome, {
        rsiBuy: 30,
        rsiSell: 70,
        williamsBuy: -80,
        williamsSell: -20,
        rocBuy: 0.02,
        rocSell: -0.02,
        bollingerBuy: 0.2,
        bollingerSell: 0.8,
        minBuy: 2,
        minSell: 2,
        style: 1,
    });
    return genome;
}

function seedHybridSwing(): Genome {
    const genome = Array.from({length: STOCK_GENE_COUNT}, () => 0);
    writePeriodGenes(genome, {smaFast: 15, smaSlow: 45, williams: 14, roc: 12, rsi: 14, macdFast: 12, macdSlow: 26, macdSignal: 9, bollinger: 20, bollingerMult: 2.2, volatility: 20, volumeZ: 20});
    writeRuleGenes(genome, {
        rsiBuy: 35,
        rsiSell: 65,
        williamsBuy: -75,
        williamsSell: -25,
        rocBuy: 0.01,
        rocSell: -0.01,
        bollingerBuy: 0.25,
        bollingerSell: 0.75,
        minBuy: 3,
        minSell: 3,
        style: 2,
    });
    return genome;
}

function writePeriodGenes(
    genome: Genome,
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
    }
): void {
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
}

function writeRuleGenes(
    genome: Genome,
    rules: {
        rsiBuy: number;
        rsiSell: number;
        williamsBuy: number;
        williamsSell: number;
        rocBuy: number;
        rocSell: number;
        bollingerBuy: number;
        bollingerSell: number;
        minBuy: number;
        minSell: number;
        style: number;
    }
): void {
    genome[12] = encodeGene(rules.rsiBuy, 15, 45);
    genome[13] = encodeGene(rules.rsiSell, 55, 85);
    genome[14] = encodeGene(rules.williamsBuy, -95, -50);
    genome[15] = encodeGene(rules.williamsSell, -50, -5);
    genome[16] = encodeGene(rules.rocBuy, -0.05, 0.1);
    genome[17] = encodeGene(rules.rocSell, -0.1, 0.05);
    genome[18] = encodeGene(rules.bollingerBuy, 0, 0.4);
    genome[19] = encodeGene(rules.bollingerSell, 0.6, 1);
    genome[20] = encodeGene(rules.minBuy, 1, 5);
    genome[21] = encodeGene(rules.minSell, 1, 5);
    genome[22] = encodeGene(rules.style, 0, 2);
}
