import {argMax, createForwardRunner} from "../../lib/neuralNetwork";
import type {Genome, IndicatorMaskState, MarketDataPoint, OptimizedIndicatorParameters, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import type {IndicatorColumns} from "./indicators";
import {calculateIndicatorColumns, columnsToSnapshots} from "./indicators";
import {ALL_INDICATOR_MASKS_ON, countActiveMasks, decodeStockGenome, INDICATOR_MASK_DEFS, type IndicatorMaskId, STOCK_TOPOLOGY, withMaskOverride} from "./strategyGenome";

export {STOCK_TOPOLOGY} from "./strategyGenome";

export const STOCK_INPUT_LABELS = [
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
 * Soft sparsity: keep feature selection pressure without starving the decision head of inputs.
 * First FREE_MASK_COUNT families free; each extra costs a little fitness.
 */
const FREE_MASK_COUNT = 5;
const SPARSITY_PENALTY_PER_MASK = 0.45;
/** Hard floor when every indicator family is off (only position feature left). */
const EMPTY_MASK_PENALTY = 80;

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
}

interface SegmentResult extends SegmentMetrics {
    equityCurve: number[];
    trades: TradeMarker[];
}

export interface IndicatorAblationRow {
    id: IndicatorMaskId;
    label: string;
    shortLabel: string;
    enabled: boolean;
    /** baselineFitness − ablatedFitness. Positive ⇒ removing hurts ⇒ useful. */
    fitnessDrop: number;
}

export interface IndicatorAblationResult {
    baselineFitness: number;
    activeCount: number;
    rows: IndicatorAblationRow[];
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
    const {parameters, masks, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const trainLength = Math.max(2, Math.floor(columns.length * 0.8));
    if (trainLength < 100) {
        return -1_000;
    }

    const activeCount = countActiveMasks(masks);
    if (activeCount === 0) {
        return -EMPTY_MASK_PENALTY;
    }

    const decide = createPositionDecider(columns, parameters, masks, networkGenome, useNetwork);
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
    const sparsity = SPARSITY_PENALTY_PER_MASK * Math.max(0, activeCount - FREE_MASK_COUNT);
    // Full-segment excess vs buy-and-hold dominates; halves only guard against one-half flukes.
    return fullScore * 0.9 + robustScore * 0.1 - regularization - sparsity;
}

/**
 * Knock out each enabled mask in turn; fitness drop ≈ how useful that family is under the same fitness.
 */
export function ablateIndicatorMasks(genome: Genome, points: MarketDataPoint[], useNetwork = true): IndicatorAblationResult {
    const {masks} = decodeStockGenome(genome);
    const baselineFitness = evaluateStockGenome(genome, points, useNetwork);
    const rows: IndicatorAblationRow[] = INDICATOR_MASK_DEFS.map(def => {
        if (!masks[def.id]) {
            return {
                id: def.id,
                label: def.label,
                shortLabel: def.shortLabel,
                enabled: false,
                fitnessDrop: 0,
            };
        }
        const ablated = withMaskOverride(genome, def.id, false);
        const ablatedFitness = evaluateStockGenome(ablated, points, useNetwork);
        return {
            id: def.id,
            label: def.label,
            shortLabel: def.shortLabel,
            enabled: true,
            fitnessDrop: baselineFitness - ablatedFitness,
        };
    });
    // Useful first: largest positive drop.
    rows.sort((a, b) => {
        if (a.enabled !== b.enabled) {
            return a.enabled ? -1 : 1;
        }
        return b.fitnessDrop - a.fitnessDrop;
    });
    return {
        baselineFitness,
        activeCount: countActiveMasks(masks),
        rows,
    };
}

/**
 * Excess-return-first segment score: the main term is log(strategy equity / benchmark equity),
 * so matching buy-and-hold scores ~0 and fitness can only be earned by actually beating it.
 * Cash through a bull (negative excess) and riding a crash (worse than cash's excess) both
 * penalize themselves — no bespoke idle/lag penalties needed. Light risk terms break ties
 * toward smoother paths; a small absolute-return term keeps participation preferred when
 * two policies tie on excess.
 */
function scoreSegment(metrics: SegmentMetrics, columns: IndicatorColumns, start: number, end: number): number {
    const first = columns.close[start] || 1;
    const last = columns.close[end - 1] || first;
    const years = Math.max((end - start) / 252, 0.5);
    const benchmarkReturn = last / first - 1;
    const logExcess = Math.log(1 + Math.max(metrics.totalReturn, -0.99)) - Math.log(1 + Math.max(benchmarkReturn, -0.99));
    const annualizedExcess = logExcess / years;

    return annualizedExcess * 250 + logExcess * 40 + metrics.totalReturn * 12 + metrics.sharpe * 2 - metrics.maxDrawdown * 8;
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
    const {parameters, masks, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const decide = createPositionDecider(columns, parameters, masks, networkGenome, useNetwork);
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
        indicatorMasks: {...masks},
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
 * Masked-off families force their feature slots to 0 so the decision head cannot use them.
 */
export function buildNetworkFeatures(
    columns: IndicatorColumns,
    index: number,
    position: number,
    parameters: OptimizedIndicatorParameters,
    masks: IndicatorMaskState = ALL_INDICATOR_MASKS_ON,
    out: number[] = new Array(STOCK_TOPOLOGY.inputSize)
): number[] {
    const close = Math.max(columns.close[index], 1e-9);
    const smaFast = Math.max(columns.smaFast[index], 1e-9);
    const smaSlow = Math.max(columns.smaSlow[index], 1e-9);
    out[0] = masks.sma ? clamp((close / smaFast - 1) * 10) : 0;
    out[1] = masks.sma ? clamp((close / smaSlow - 1) * 10) : 0;
    out[2] = masks.sma ? clamp((smaFast / smaSlow - 1) * 10) : 0;
    out[3] = masks.williams ? clamp((columns.williamsR[index] + 50) / 50) : 0;
    out[4] = masks.roc ? clamp(columns.roc[index] * 5) : 0;
    out[5] = masks.rsi ? clamp((columns.rsi[index] - 50) / 50) : 0;
    out[6] = masks.macd ? clamp((columns.macd[index] / close) * 25) : 0;
    out[7] = masks.macd ? clamp((columns.macdSignal[index] / close) * 25) : 0;
    out[8] = masks.bollinger ? clamp((columns.bollingerPercentB[index] - 0.5) * 2) : 0;
    out[9] = masks.volatility ? clamp(columns.volatility[index] * 5) : 0;
    out[10] = masks.volume ? clamp(columns.volumeZScore[index] / 3) : 0;
    // close / N-day high ≈ 1 at breakout; map ~[0.9, 1.0] into roughly [-1, 1].
    out[11] = masks.newHigh ? clamp((columns.newHighRatio[index] - 0.95) * 20) : 0;
    out[12] = position > 0 ? 1 : -1;
    out[13] = masks.rsi ? clamp((parameters.rsiBuyThreshold - columns.rsi[index]) / 20) : 0;
    out[14] = masks.rsi ? clamp((columns.rsi[index] - parameters.rsiSellThreshold) / 20) : 0;
    out[15] = masks.williams ? clamp((parameters.williamsBuyThreshold - columns.williamsR[index]) / 25) : 0;
    out[16] = masks.williams ? clamp((columns.williamsR[index] - parameters.williamsSellThreshold) / 25) : 0;
    return out;
}

/**
 * Rule mode: only enabled families cast votes.
 * Buy = majority of active buy votes (min 1). Sell = any enabled overbought exit, else fail-to-buy majority when long.
 */
export function decidePositionFromRules(
    columns: IndicatorColumns,
    index: number,
    position: number,
    parameters: OptimizedIndicatorParameters,
    masks: IndicatorMaskState = ALL_INDICATOR_MASKS_ON
): number {
    const hasRsiExit = masks.rsi && columns.rsi[index] >= parameters.rsiSellThreshold;
    const hasWilliamsExit = masks.williams && columns.williamsR[index] >= parameters.williamsSellThreshold;
    if (position > 0 && (hasRsiExit || hasWilliamsExit)) {
        return 0;
    }

    const votes: number[] = [];
    if (masks.sma) {
        votes.push(columns.smaFast[index] > columns.smaSlow[index] ? 1 : 0);
    }
    if (masks.macd) {
        votes.push(columns.macd[index] > columns.macdSignal[index] ? 1 : 0);
    }
    if (masks.rsi) {
        votes.push(columns.rsi[index] <= parameters.rsiBuyThreshold ? 1 : 0);
    }
    if (masks.williams) {
        votes.push(columns.williamsR[index] <= parameters.williamsBuyThreshold ? 1 : 0);
    }

    if (votes.length === 0) {
        // No rule-capable families → stay flat (or hold existing if somehow long without exits).
        return position > 0 && !masks.rsi && !masks.williams ? 0 : 0;
    }

    const yes = votes.reduce((sum, vote) => sum + vote, 0);
    const needed = Math.max(1, Math.ceil(votes.length / 2));
    if (yes >= needed) {
        return 1;
    }
    // No dedicated exit indicators: leave when the buy majority fails.
    if (position > 0 && !masks.rsi && !masks.williams) {
        return 0;
    }
    return position;
}

type PositionDecider = (index: number, position: number) => number;

function createPositionDecider(columns: IndicatorColumns, parameters: OptimizedIndicatorParameters, masks: IndicatorMaskState, networkGenome: Genome, useNetwork: boolean): PositionDecider {
    if (!useNetwork) {
        return (index, position) => decidePositionFromRules(columns, index, position, parameters, masks);
    }
    // Pure forward runner: decode weights once per genome; reuse feature buffer every bar.
    const runNetwork = createForwardRunner(networkGenome, STOCK_TOPOLOGY);
    const features = new Array<number>(STOCK_TOPOLOGY.inputSize);
    return (index, position) => decidePositionFromNetwork(runNetwork(buildNetworkFeatures(columns, index, position, parameters, masks, features)), position);
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

    let halfEquity = STARTING_EQUITY;
    let halfPeak = halfEquity;
    let halfMaxDrawdown = 0;
    let halfReturnSum = 0;
    let halfReturnSqSum = 0;
    let halfReturnCount = 0;
    let halfExposureSum = 0;
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
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);

        if (index < halfEnd) {
            halfEquity *= Math.max(0.01, 1 + dailyReturn);
            halfReturnSum += dailyReturn;
            halfReturnSqSum += dailyReturn * dailyReturn;
            halfReturnCount += 1;
            halfExposureSum += position;
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
        },
        firstHalf: {
            totalReturn: halfEquity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(halfReturnSum, halfReturnSqSum, halfReturnCount),
            maxDrawdown: halfMaxDrawdown,
            endingPosition: halfEndingPosition,
            meanExposure: halfReturnCount > 0 ? halfExposureSum / halfReturnCount : 0,
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
