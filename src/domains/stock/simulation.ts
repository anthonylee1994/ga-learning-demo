import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import type {Genome, MarketDataPoint, OptimizedIndicatorParameters, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import type {IndicatorColumns} from "./indicators";
import {calculateIndicatorColumns, columnsToSnapshots} from "./indicators";
import {decodeStockGenome, STOCK_TOPOLOGY} from "./strategyGenome";

export {STOCK_TOPOLOGY} from "./strategyGenome";

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;
/** Keep a small LRU of indicator series — Float64Array columns, ~1MB per full history entry. */
const MAX_INDICATOR_CACHE = 16;
/** Stronger weight decay so fitness gains come from indicator periods, not overweight nets. */
const WEIGHT_L2_PENALTY = 2.5;

/** Shared adapter — avoid allocating a fresh brain.js graph per genome. */
const networkAdapter = new NeuralNetworkAdapter(STOCK_TOPOLOGY);

/**
 * Cache indicator columns by the exact points array reference.
 * evaluate() used to recompute indicators for every genome every generation —
 * with full daily history that was a large allocation storm.
 */
let cachedPoints: MarketDataPoint[] | null = null;
const cachedColumns = new Map<string, IndicatorColumns>();

interface SegmentMetrics {
    totalReturn: number;
    sharpe: number;
    maxDrawdown: number;
    endingPosition: number;
}

interface SegmentResult extends SegmentMetrics {
    equityCurve: number[];
    trades: TradeMarker[];
}

export function getIndicatorColumns(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters): IndicatorColumns {
    if (points !== cachedPoints) {
        cachedPoints = points;
        cachedColumns.clear();
    }
    const key = createIndicatorCacheKey(parameters);
    const cached = cachedColumns.get(key);
    if (cached) {
        // Refresh LRU order.
        cachedColumns.delete(key);
        cachedColumns.set(key, cached);
        return cached;
    }
    const columns = calculateIndicatorColumns(points, parameters);
    if (cachedColumns.size >= MAX_INDICATOR_CACHE) {
        const oldest = cachedColumns.keys().next().value;
        if (oldest !== undefined) {
            cachedColumns.delete(oldest);
        }
    }
    cachedColumns.set(key, columns);
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
    const full = simulateColumnMetrics(decide, columns, 0, 0, trainLength);
    const firstHalf = simulateColumnMetrics(decide, columns, 0, 0, mid + 1);
    const secondHalf = simulateColumnMetrics(decide, columns, firstHalf.endingPosition, mid, trainLength);
    const fullScore = scoreSegment(full, columns, 0, trainLength);
    const robustScore = Math.min(scoreSegment(firstHalf, columns, 0, mid + 1), scoreSegment(secondHalf, columns, mid, trainLength));
    // Rule mode ignores the NN tail entirely — penalizing unused weights would just add noise.
    const regularization = useNetwork ? WEIGHT_L2_PENALTY * meanSquare(networkGenome) : 0;
    return fullScore * 0.7 + robustScore * 0.3 - regularization;
}

/**
 * Prefer absolute risk-adjusted performance (what the equity chart shows as "working"),
 * with a softer excess-vs-B&H term so pure cash still loses on strong bulls.
 */
function scoreSegment(metrics: SegmentMetrics, columns: IndicatorColumns, start: number, end: number): number {
    const first = columns.close[start] || 1;
    const last = columns.close[end - 1] || first;
    const years = Math.max((end - start) / 252, 0.5);
    const strategyCagr = annualize(metrics.totalReturn, years);
    const benchmarkCagr = annualize(last / first - 1, years);
    const excess = strategyCagr - benchmarkCagr;
    const idlePenalty = Math.abs(metrics.totalReturn) < 0.02 ? 8 : 0;
    return strategyCagr * 100 + metrics.sharpe * 15 - metrics.maxDrawdown * 40 + excess * 35 - idlePenalty;
}

function annualize(totalReturn: number, years: number): number {
    return Math.pow(1 + Math.max(totalReturn, -0.99), 1 / years) - 1;
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
    const smaFast = Math.max(columns.smaFast[index], 1e-9);
    const smaSlow = Math.max(columns.smaSlow[index], 1e-9);
    out[0] = clamp((close / smaFast - 1) * 10);
    out[1] = clamp((close / smaSlow - 1) * 10);
    out[2] = clamp((smaFast / smaSlow - 1) * 10);
    out[3] = clamp((columns.williamsR[index] + 50) / 50);
    out[4] = clamp(columns.roc[index] * 5);
    out[5] = clamp((columns.rsi[index] - 50) / 50);
    out[6] = clamp((columns.macd[index] / close) * 25);
    out[7] = clamp((columns.macdSignal[index] / close) * 25);
    out[8] = clamp((columns.bollingerPercentB[index] - 0.5) * 2);
    out[9] = clamp(columns.volatility[index] * 5);
    out[10] = clamp(columns.volumeZScore[index] / 3);
    // close / N-day high ≈ 1 at breakout; map ~[0.9, 1.0] into roughly [-1, 1].
    out[11] = clamp((columns.newHighRatio[index] - 0.95) * 20);
    out[12] = position > 0 ? 1 : -1;
    out[13] = clamp((parameters.rsiBuyThreshold - columns.rsi[index]) / 20);
    out[14] = clamp((columns.rsi[index] - parameters.rsiSellThreshold) / 20);
    out[15] = clamp((parameters.williamsBuyThreshold - columns.williamsR[index]) / 25);
    out[16] = clamp((columns.williamsR[index] - parameters.williamsSellThreshold) / 25);
    return out;
}

/**
 * NN 關咗時嘅 threshold 基準：買入要 trend / MACD / RSI oversold / Williams oversold 四票取二；
 * 持倉後 RSI 或 Williams 任一升穿進化出嚟嘅 sell threshold 就離場。
 */
export function decidePositionFromRules(columns: IndicatorColumns, index: number, position: number, parameters: OptimizedIndicatorParameters): number {
    if (position > 0 && (columns.rsi[index] >= parameters.rsiSellThreshold || columns.williamsR[index] >= parameters.williamsSellThreshold)) {
        return 0;
    }
    const trendVote = columns.smaFast[index] > columns.smaSlow[index] ? 1 : 0;
    const macdVote = columns.macd[index] > columns.macdSignal[index] ? 1 : 0;
    const rsiBuyVote = columns.rsi[index] <= parameters.rsiBuyThreshold ? 1 : 0;
    const williamsBuyVote = columns.williamsR[index] <= parameters.williamsBuyThreshold ? 1 : 0;
    return trendVote + macdVote + rsiBuyVote + williamsBuyVote >= 2 ? 1 : position;
}

type PositionDecider = (index: number, position: number) => number;

function createPositionDecider(columns: IndicatorColumns, parameters: OptimizedIndicatorParameters, networkGenome: Genome, useNetwork: boolean): PositionDecider {
    if (!useNetwork) {
        return (index, position) => decidePositionFromRules(columns, index, position, parameters);
    }
    const runNetwork = networkAdapter.createRunner(networkGenome);
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
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);
        position = targetPosition;
    }

    return {
        totalReturn: equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
        maxDrawdown,
        endingPosition: position,
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
