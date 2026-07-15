import {argMax, createForwardRunner} from "../../lib/neuralNetwork";
import type {Genome, MarketDataPoint, OptimizedIndicatorParameters, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import type {IndicatorColumns} from "./indicators";
import {calculateIndicatorColumns, columnsToSnapshots} from "./indicators";
import {decodeStockGenome, STOCK_TOPOLOGY} from "./strategyGenome";

export {STOCK_TOPOLOGY} from "./strategyGenome";

export const STOCK_INPUT_LABELS = [
    "開",
    "高",
    "低",
    "收",
    "快線",
    "慢線",
    "快慢線差",
    "威廉指標",
    "ROC",
    "RSI",
    "MACD",
    "MACD訊",
    "BB通道",
    "波動率",
    "成交量",
    "N日新高",
    "持倉",
    "RSI買距",
    "RSI賣距",
    "W%買距",
    "W%賣距",
] as const;

export const STOCK_OUTPUT_LABELS = ["買入", "持有", "賣出"] as const;

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;
/** Keep a small LRU of indicator series — Float64Array columns, ~1MB per full history entry. */
const MAX_INDICATOR_CACHE = 16;
/**
 * Mild weight decay — keep below return-scale so L2 never prefers cash over investing.
 */
const WEIGHT_L2_PENALTY = 0.55;

/**
 * Cache indicator columns per points-array reference (multi-ticker fitness interleaves
 * several series per genome — a single "current points" slot would thrash every call).
 * WeakMap lets a replaced series' cache be collected with the array itself.
 */
const columnCachesBySeries = new WeakMap<MarketDataPoint[], Map<string, IndicatorColumns>>();

interface SegmentMetrics {
    totalReturn: number;
    sharpe: number;
    maxDrawdown: number;
    endingPosition: number;
    /** Average long exposure in [0, 1] over the segment (cash-only ≈ 0). */
    meanExposure: number;
    /** Mean |Δposition| per bar — thrashy policies sit high (≈ frequent full flips). */
    meanTurnover: number;
}

interface SegmentResult extends SegmentMetrics {
    equityCurve: number[];
    trades: TradeMarker[];
}

export function getIndicatorColumns(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters): IndicatorColumns {
    let seriesCache = columnCachesBySeries.get(points);
    if (!seriesCache) {
        seriesCache = new Map();
        columnCachesBySeries.set(points, seriesCache);
    }
    const key = createIndicatorCacheKey(parameters);
    const cached = seriesCache.get(key);
    if (cached) {
        // Refresh LRU order.
        seriesCache.delete(key);
        seriesCache.set(key, cached);
        return cached;
    }
    const columns = calculateIndicatorColumns(points, parameters);
    if (seriesCache.size >= MAX_INDICATOR_CACHE) {
        const oldest = seriesCache.keys().next().value;
        if (oldest !== undefined) {
            seriesCache.delete(oldest);
        }
    }
    seriesCache.set(key, columns);
    return columns;
}

export function getIndicatorSnapshots(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters) {
    return columnsToSnapshots(points, getIndicatorColumns(points, parameters));
}

export function evaluateStockGenome(genome: Genome, points: MarketDataPoint[], useNetwork = true): number {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const trainLength = Math.max(2, Math.floor(columns.length * 0.8));
    if (trainLength < 100) {
        return -1_000;
    }

    const decide = createPositionDecider(columns, parameters, networkGenome, useNetwork);
    const mid = Math.floor(trainLength / 2);
    // One pass for full + first half (was two full walks); second half still restarts equity.
    const {full, firstHalf} = simulateFullAndFirstHalf(decide, columns, trainLength, mid);
    const secondHalf = simulateColumnMetrics(decide, columns, firstHalf.endingPosition, mid, trainLength);
    const fullScore = scoreSegment(full, columns, 0, trainLength);
    const halfA = scoreSegment(firstHalf, columns, 0, mid + 1);
    const halfB = scoreSegment(secondHalf, columns, mid, trainLength);
    // Soft robust: light floor so one weak half does not erase a strong train return.
    const robustScore = 0.22 * Math.min(halfA, halfB) + 0.78 * ((halfA + halfB) / 2);
    // Rule mode ignores the NN tail entirely — penalizing unused weights would just add noise.
    const regularization = useNetwork ? WEIGHT_L2_PENALTY * meanSquare(networkGenome) : 0;
    // Full-segment excess vs buy-and-hold dominates; halves only guard against one-half flukes.
    return fullScore * 0.9 + robustScore * 0.1 - regularization;
}

/**
 * Excess-return-first segment score, with a stronger absolute-return / participation bias so
 * bull-market demos do not collapse into cash thrash. Main term is still
 * log(strategy equity / benchmark equity); matching buy-and-hold ≈ 0 on excess alone.
 * Absolute return + mean exposure pull the search toward staying invested; mean turnover
 * penalizes fee-bleeding flip-flops that look lucky on a single half-segment.
 */
function scoreSegment(metrics: SegmentMetrics, columns: IndicatorColumns, start: number, end: number): number {
    const first = columns.close[start] || 1;
    const last = columns.close[end - 1] || first;
    const years = Math.max((end - start) / 252, 0.5);
    const benchmarkReturn = last / first - 1;
    const logExcess = Math.log(1 + Math.max(metrics.totalReturn, -0.99)) - Math.log(1 + Math.max(benchmarkReturn, -0.99));
    const annualizedExcess = logExcess / years;

    return annualizedExcess * 250 + logExcess * 40 + metrics.totalReturn * 40 + metrics.meanExposure * 18 + metrics.sharpe * 2 - metrics.maxDrawdown * 8 - metrics.meanTurnover * 35;
}

function meanSquare(genome: Genome): number {
    if (genome.length === 0) {
        return 0;
    }
    let sum = 0;
    for (const weight of genome) {
        sum += weight * weight;
    }
    return sum / genome.length;
}

export function createTradingReplay(genome: Genome, points: MarketDataPoint[], useNetwork = true): TradingReplay {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const decide = createPositionDecider(columns, parameters, networkGenome, useNetwork);
    const splitIndex = Math.max(2, Math.floor(columns.length * 0.8));
    const trainResult = simulateColumnReplay(decide, columns, points, 0, splitIndex, 0);
    const testResult = simulateColumnReplay(decide, columns, points, Math.max(0, splitIndex - 1), columns.length, trainResult.endingPosition);
    const trainCurve = trainResult.equityCurve;
    const testScale = trainCurve.at(-1) ?? STARTING_EQUITY;
    const fullCurve = new Array<number>(columns.length);
    for (let index = 0; index < trainCurve.length; index += 1) {
        fullCurve[index] = trainCurve[index];
    }
    for (let index = 1; index < testResult.equityCurve.length; index += 1) {
        fullCurve[splitIndex - 1 + index] = (testResult.equityCurve[index] / STARTING_EQUITY) * testScale;
    }
    const firstClose = columns.close[0] || 1;
    const lastClose = columns.close[columns.length - 1] || firstClose;
    const tradingPoints: TradingPoint[] = new Array(columns.length);
    for (let index = 0; index < columns.length; index += 1) {
        const close = columns.close[index];
        tradingPoints[index] = {
            date: points[columns.warmup + index].date,
            close,
            strategy: fullCurve[index] ?? fullCurve[fullCurve.length - 1] ?? STARTING_EQUITY,
            benchmark: STARTING_EQUITY * (close / firstClose),
            segment: index < splitIndex ? "train" : "test",
            smaFast: columns.smaFast[index],
            smaSlow: columns.smaSlow[index],
            rsi: columns.rsi[index],
            williamsR: columns.williamsR[index],
            roc: columns.roc[index],
            macd: columns.macd[index],
            macdSignal: columns.macdSignal[index],
            bollingerUpper: columns.bollingerUpper[index],
            bollingerLower: columns.bollingerLower[index],
            volatility: columns.volatility[index],
            volumeZScore: columns.volumeZScore[index],
            nDayHigh: columns.nDayHigh[index],
            newHighRatio: columns.newHighRatio[index],
        };
    }

    return {
        points: tradingPoints,
        trades: [...trainResult.trades, ...testResult.trades],
        trainReturn: trainResult.totalReturn,
        testReturn: testResult.totalReturn,
        benchmarkReturn: columns.length > 1 ? lastClose / firstClose - 1 : 0,
        sharpe: trainResult.sharpe,
        maxDrawdown: trainResult.maxDrawdown,
        optimizedParameters: parameters,
    };
}

/**
 * Map brain.js outputs to long (1) or cash (0).
 * action 0 = buy/long, 1 = hold, 2 = sell/cash.
 */
export function decidePositionFromNetwork(output: number[], position: number): number {
    const action = argMax(output);
    if (action === 0) {
        return 1;
    }
    if (action === 2) {
        return 0;
    }
    return position;
}

/**
 * Reconstruct long/cash position just before `date` from the trade log
 * (used when scrubbing the NN activation preview on the stock lab).
 */
export function positionBeforeDate(trades: TradeMarker[], date: string): number {
    let position = 0;
    for (const trade of trades) {
        if (trade.date > date) {
            break;
        }
        position = trade.action === "buy" ? 1 : 0;
    }
    return position;
}

/**
 * Clamp raw indicator ratios into roughly [-1, 1] so tanh units see consistent scale.
 */
export function buildNetworkFeatures(
    columns: IndicatorColumns,
    index: number,
    position: number,
    parameters: OptimizedIndicatorParameters,
    out: number[] = new Array(STOCK_TOPOLOGY.inputSize)
): number[] {
    const close = Math.max(columns.close[index], 1e-9);
    const open = columns.open[index];
    const high = columns.high[index];
    const low = columns.low[index];
    const smaFast = Math.max(columns.smaFast[index], 1e-9);
    const smaSlow = Math.max(columns.smaSlow[index], 1e-9);
    // 高低開收：K 線結構（相對收盤）+ 收盤日回報，縮放到 roughly [-1, 1]。
    out[0] = clamp((open / close - 1) * 50);
    out[1] = clamp((high / close - 1) * 50);
    out[2] = clamp((low / close - 1) * 50);
    out[3] = clamp(columns.closeReturn[index] * 20);
    out[4] = clamp((close / smaFast - 1) * 10);
    out[5] = clamp((close / smaSlow - 1) * 10);
    out[6] = clamp((smaFast / smaSlow - 1) * 10);
    out[7] = clamp((columns.williamsR[index] + 50) / 50);
    out[8] = clamp(columns.roc[index] * 5);
    out[9] = clamp((columns.rsi[index] - 50) / 50);
    out[10] = clamp((columns.macd[index] / close) * 25);
    out[11] = clamp((columns.macdSignal[index] / close) * 25);
    out[12] = clamp((columns.bollingerPercentB[index] - 0.5) * 2);
    out[13] = clamp(columns.volatility[index] * 5);
    out[14] = clamp(columns.volumeZScore[index] / 3);
    // close / N-day high ≈ 1 at breakout; map ~[0.9, 1.0] into roughly [-1, 1].
    out[15] = clamp((columns.newHighRatio[index] - 0.95) * 20);
    out[16] = position > 0 ? 1 : -1;
    out[17] = clamp((parameters.rsiBuyThreshold - columns.rsi[index]) / 20);
    out[18] = clamp((columns.rsi[index] - parameters.rsiSellThreshold) / 20);
    out[19] = clamp((parameters.williamsBuyThreshold - columns.williamsR[index]) / 25);
    out[20] = clamp((columns.williamsR[index] - parameters.williamsSellThreshold) / 25);
    return out;
}

/**
 * Rule mode: SMA / MACD / RSI / Williams cast buy votes.
 * Buy = majority of the 4 votes (min 2).
 * Sell: RSI / Williams overbought, but in an uptrend (fast SMA > slow) require *both*
 * exit signals — single-indicator "overbought" exits bleed train return on strong bulls.
 */
export function decidePositionFromRules(columns: IndicatorColumns, index: number, position: number, parameters: OptimizedIndicatorParameters): number {
    const hasRsiExit = columns.rsi[index] >= parameters.rsiSellThreshold;
    const hasWilliamsExit = columns.williamsR[index] >= parameters.williamsSellThreshold;
    if (position > 0 && (hasRsiExit || hasWilliamsExit)) {
        const uptrend = columns.smaFast[index] > columns.smaSlow[index];
        if (uptrend) {
            // Strong trend: only exit when both momentum exits fire.
            if (hasRsiExit && hasWilliamsExit) {
                return 0;
            }
        } else if (hasRsiExit || hasWilliamsExit) {
            return 0;
        }
    }

    const votes = [
        columns.smaFast[index] > columns.smaSlow[index] ? 1 : 0,
        columns.macd[index] > columns.macdSignal[index] ? 1 : 0,
        columns.rsi[index] <= parameters.rsiBuyThreshold ? 1 : 0,
        columns.williamsR[index] <= parameters.williamsBuyThreshold ? 1 : 0,
    ];

    const yes = votes.reduce((sum, vote) => sum + vote, 0);
    const needed = Math.max(1, Math.ceil(votes.length / 2));
    if (yes >= needed) {
        return 1;
    }
    return position;
}

type PositionDecider = (index: number, position: number) => number;

function createPositionDecider(columns: IndicatorColumns, parameters: OptimizedIndicatorParameters, networkGenome: Genome, useNetwork: boolean): PositionDecider {
    if (!useNetwork) {
        return (index, position) => decidePositionFromRules(columns, index, position, parameters);
    }
    // Pure forward runner: decode weights once per genome; reuse feature buffer every bar.
    const runNetwork = createForwardRunner(networkGenome, STOCK_TOPOLOGY);
    const features = new Array<number>(STOCK_TOPOLOGY.inputSize);
    return (index, position) => decidePositionFromNetwork(runNetwork(buildNetworkFeatures(columns, index, position, parameters, features)), position);
}

function createIndicatorCacheKey(parameters: OptimizedIndicatorParameters): string {
    return [
        parameters.smaFastPeriod,
        parameters.smaSlowPeriod,
        parameters.williamsPeriod,
        parameters.rocPeriod,
        parameters.rsiPeriod,
        parameters.macdFastPeriod,
        parameters.macdSlowPeriod,
        parameters.macdSignalPeriod,
        parameters.bollingerPeriod,
        parameters.bollingerMultiplier,
        parameters.volatilityPeriod,
        parameters.volumeZScorePeriod,
        parameters.newHighPeriod,
    ].join(":");
}

function simulateColumnMetrics(decide: PositionDecider, columns: IndicatorColumns, startingPosition: number, start: number, end: number): SegmentMetrics {
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    let exposureSum = 0;
    let turnoverSum = 0;

    for (let index = start + 1; index < end; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const turnover = Math.abs(targetPosition - position);
        const priceReturn = columns.close[index] / columns.close[previous] - 1;
        const dailyReturn = position * priceReturn - turnover * TRANSACTION_COST;
        equity *= Math.max(0.01, 1 + dailyReturn);
        returnSum += dailyReturn;
        returnSqSum += dailyReturn * dailyReturn;
        returnCount += 1;
        exposureSum += position;
        turnoverSum += turnover;
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);
        position = targetPosition;
    }

    return {
        totalReturn: equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
        maxDrawdown,
        endingPosition: position,
        meanExposure: returnCount > 0 ? exposureSum / returnCount : 0,
        meanTurnover: returnCount > 0 ? turnoverSum / returnCount : 0,
    };
}

/**
 * Walk the full train window once while also accumulating first-half metrics
 * (same semantics as separate simulateColumnMetrics calls for full + first half).
 */
function simulateFullAndFirstHalf(decide: PositionDecider, columns: IndicatorColumns, trainLength: number, mid: number): {full: SegmentMetrics; firstHalf: SegmentMetrics} {
    let equity = STARTING_EQUITY;
    let position = 0;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    let exposureSum = 0;
    let turnoverSum = 0;

    let halfEquity = STARTING_EQUITY;
    let halfPeak = halfEquity;
    let halfMaxDrawdown = 0;
    let halfReturnSum = 0;
    let halfReturnSqSum = 0;
    let halfReturnCount = 0;
    let halfExposureSum = 0;
    let halfTurnoverSum = 0;
    let halfEndingPosition = 0;
    const halfEnd = mid + 1;

    for (let index = 1; index < trainLength; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const turnover = Math.abs(targetPosition - position);
        const priceReturn = columns.close[index] / columns.close[previous] - 1;
        const dailyReturn = position * priceReturn - turnover * TRANSACTION_COST;
        equity *= Math.max(0.01, 1 + dailyReturn);
        returnSum += dailyReturn;
        returnSqSum += dailyReturn * dailyReturn;
        returnCount += 1;
        exposureSum += position;
        turnoverSum += turnover;
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);

        if (index < halfEnd) {
            halfEquity *= Math.max(0.01, 1 + dailyReturn);
            halfReturnSum += dailyReturn;
            halfReturnSqSum += dailyReturn * dailyReturn;
            halfReturnCount += 1;
            halfExposureSum += position;
            halfTurnoverSum += turnover;
            halfPeak = Math.max(halfPeak, halfEquity);
            halfMaxDrawdown = Math.max(halfMaxDrawdown, halfPeak > 0 ? (halfPeak - halfEquity) / halfPeak : 0);
            halfEndingPosition = targetPosition;
        }

        position = targetPosition;
    }

    return {
        full: {
            totalReturn: equity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
            maxDrawdown,
            endingPosition: position,
            meanExposure: returnCount > 0 ? exposureSum / returnCount : 0,
            meanTurnover: returnCount > 0 ? turnoverSum / returnCount : 0,
        },
        firstHalf: {
            totalReturn: halfEquity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(halfReturnSum, halfReturnSqSum, halfReturnCount),
            maxDrawdown: halfMaxDrawdown,
            endingPosition: halfEndingPosition,
            meanExposure: halfReturnCount > 0 ? halfExposureSum / halfReturnCount : 0,
            meanTurnover: halfReturnCount > 0 ? halfTurnoverSum / halfReturnCount : 0,
        },
    };
}

function simulateColumnReplay(decide: PositionDecider, columns: IndicatorColumns, points: MarketDataPoint[], start: number, end: number, startingPosition: number): SegmentResult {
    const length = Math.max(0, end - start);
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    let exposureSum = 0;
    let turnoverSum = 0;
    const equityCurve = new Array<number>(length);
    if (length > 0) {
        equityCurve[0] = equity;
    }
    const trades: TradeMarker[] = [];

    for (let index = start + 1; index < end; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const turnover = Math.abs(targetPosition - position);
        const priceReturn = columns.close[index] / columns.close[previous] - 1;
        const dailyReturn = position * priceReturn - turnover * TRANSACTION_COST;
        equity *= Math.max(0.01, 1 + dailyReturn);
        returnSum += dailyReturn;
        returnSqSum += dailyReturn * dailyReturn;
        returnCount += 1;
        exposureSum += position;
        turnoverSum += turnover;
        equityCurve[index - start] = equity;
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);

        if (targetPosition !== position) {
            trades.push({
                date: points[columns.warmup + previous].date,
                action: targetPosition > position ? "buy" : "sell",
                price: columns.close[previous],
            });
            position = targetPosition;
        }
    }

    return {
        equityCurve,
        trades,
        totalReturn: equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
        maxDrawdown,
        endingPosition: position,
        meanExposure: returnCount > 0 ? exposureSum / returnCount : 0,
        meanTurnover: returnCount > 0 ? turnoverSum / returnCount : 0,
    };
}

function calculateSharpeFromMoments(returnSum: number, returnSqSum: number, returnCount: number): number {
    if (returnCount < 2) {
        return 0;
    }
    const average = returnSum / returnCount;
    const variance = returnSqSum / returnCount - average * average;
    const deviation = Math.sqrt(Math.max(0, variance));
    return deviation > 1e-9 ? (average / deviation) * Math.sqrt(252) : 0;
}

function clamp(value: number): number {
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.min(1, Math.max(-1, value));
}
