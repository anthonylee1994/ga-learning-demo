import type {Genome, IndicatorSnapshot, MarketDataPoint, OptimizedIndicatorParameters, OptimizedStrategyRules, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import type {IndicatorColumns} from "./indicators";
import {calculateIndicatorColumns, columnsToSnapshots, splitIndicators} from "./indicators";
import {decodeStockGenome} from "./strategyGenome";

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;
/** Keep a small LRU of indicator series — Float64Array columns, ~1MB per full history entry. */
const MAX_INDICATOR_CACHE = 24;

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
    const key = Object.values(parameters).join(":");
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

export function getIndicatorSnapshots(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters): IndicatorSnapshot[] {
    return columnsToSnapshots(points, getIndicatorColumns(points, parameters));
}

export function evaluateStockGenome(genome: Genome, points: MarketDataPoint[]): number {
    const {parameters, rules} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    // Same boundary as splitIndicators(snapshots) so fitness and replay agree.
    const trainLength = Math.max(2, Math.floor(columns.length * 0.8));
    if (trainLength < 100) {
        return -1_000;
    }

    // Robust fitness: score the two halves of the train window separately and keep the
    // worst one. A single full-history score compounded the benchmark to +several hundred
    // percent, so every "mostly cash" genome collapsed to the same huge negative number
    // (no gradient) and "never sell" became an unbeatable local optimum. Annualizing keeps
    // the scale sane, and min() across regimes punishes one-lucky-bull-run overfits.
    const mid = Math.floor(trainLength / 2);
    const firstHalf = simulateColumnMetrics(rules, columns, 0, 0, mid + 1);
    const secondHalf = simulateColumnMetrics(rules, columns, firstHalf.endingPosition, mid, trainLength);
    return Math.min(scoreSegment(firstHalf, columns, 0, mid + 1), scoreSegment(secondHalf, columns, mid, trainLength));
}

/** Annualized excess return vs buy & hold, rewarded for Sharpe, penalized for drawdown. */
function scoreSegment(metrics: SegmentMetrics, columns: IndicatorColumns, start: number, end: number): number {
    const first = columns.close[start] || 1;
    const last = columns.close[end - 1] || first;
    const years = Math.max((end - start) / 252, 0.5);
    const strategyCagr = annualize(metrics.totalReturn, years);
    const benchmarkCagr = annualize(last / first - 1, years);
    return (strategyCagr - benchmarkCagr) * 100 + metrics.sharpe * 10 - metrics.maxDrawdown * 30;
}

function annualize(totalReturn: number, years: number): number {
    return Math.pow(1 + Math.max(totalReturn, -0.99), 1 / years) - 1;
}

export function createTradingReplay(genome: Genome, points: MarketDataPoint[]): TradingReplay {
    const {parameters, rules} = decodeStockGenome(genome);
    const snapshots = getIndicatorSnapshots(points, parameters);
    const {train, test, splitIndex} = splitIndicators(snapshots);
    const trainResult = simulateSegmentReplay(rules, train, 0);
    const testResult = simulateSegmentReplay(rules, test, trainResult.endingPosition);
    const trainCurve = trainResult.equityCurve;
    const testScale = trainCurve.at(-1) ?? STARTING_EQUITY;
    const normalizedTestCurve = testResult.equityCurve.map(value => (value / STARTING_EQUITY) * testScale);
    const fullCurve = [...trainCurve, ...normalizedTestCurve.slice(1)];
    const firstClose = snapshots[0]?.close ?? 1;
    const tradingPoints: TradingPoint[] = snapshots.map((snapshot, index) => ({
        date: snapshot.date,
        close: snapshot.close,
        strategy: fullCurve[index] ?? fullCurve.at(-1) ?? STARTING_EQUITY,
        benchmark: STARTING_EQUITY * (snapshot.close / firstClose),
        segment: index < splitIndex ? "train" : "test",
        smaFast: snapshot.smaFast,
        smaSlow: snapshot.smaSlow,
        rsi: snapshot.rsi,
        williamsR: snapshot.williamsR,
        roc: snapshot.roc,
        macd: snapshot.macd,
        macdSignal: snapshot.macdSignal,
        bollingerUpper: snapshot.bollingerUpper,
        bollingerLower: snapshot.bollingerLower,
        volatility: snapshot.volatility,
        volumeZScore: snapshot.volumeZScore,
    }));

    return {
        points: tradingPoints,
        trades: [...trainResult.trades, ...testResult.trades],
        trainReturn: trainResult.totalReturn,
        testReturn: testResult.totalReturn,
        benchmarkReturn: snapshots.length > 1 ? snapshots.at(-1)!.close / firstClose - 1 : 0,
        sharpe: trainResult.sharpe,
        maxDrawdown: trainResult.maxDrawdown,
        optimizedParameters: parameters,
        optimizedRules: rules,
    };
}

/** The subset of indicator fields the voting rules actually read. */
export type StrategySignals = Pick<IndicatorSnapshot, "smaFast" | "smaSlow" | "rsi" | "williamsR" | "roc" | "bollingerPercentB" | "macd" | "macdSignal">;

/**
 * Multi-indicator voting rules evolved by GA.
 * buy / hold / sell → target long (1) or cash (0); never short.
 */
export function decidePosition(snapshot: StrategySignals, rules: OptimizedStrategyRules, position: number): number {
    const trendUp = snapshot.smaFast > snapshot.smaSlow;
    let buySignals = 0;
    let sellSignals = 0;

    if (trendUp) {
        buySignals += 1;
    } else {
        sellSignals += 1;
    }
    if (snapshot.rsi < rules.rsiBuy) {
        buySignals += 1;
    }
    if (snapshot.rsi > rules.rsiSell) {
        sellSignals += 1;
    }
    if (snapshot.williamsR < rules.williamsBuy) {
        buySignals += 1;
    }
    if (snapshot.williamsR > rules.williamsSell) {
        sellSignals += 1;
    }
    if (snapshot.roc > rules.rocBuy) {
        buySignals += 1;
    }
    if (snapshot.roc < rules.rocSell) {
        sellSignals += 1;
    }
    if (snapshot.bollingerPercentB < rules.bollingerBuy) {
        buySignals += 1;
    }
    if (snapshot.bollingerPercentB > rules.bollingerSell) {
        sellSignals += 1;
    }
    if (snapshot.macd > snapshot.macdSignal) {
        buySignals += 1;
    } else {
        sellSignals += 1;
    }

    if (position <= 0) {
        const trendOk = !rules.useTrendFilter || trendUp;
        if (trendOk && buySignals >= rules.minBuySignals) {
            return 1;
        }
        return 0;
    }

    if (sellSignals >= rules.minSellSignals) {
        return 0;
    }
    return 1;
}

/**
 * Streaming fitness metrics over columns[start, end) — the hot GA path. Reads Float64Array
 * columns through one reused view object so evaluating a genome allocates nothing per bar.
 */
function simulateColumnMetrics(rules: OptimizedStrategyRules, columns: IndicatorColumns, startingPosition: number, start: number, end: number): SegmentMetrics {
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    const view: StrategySignals = {smaFast: 0, smaSlow: 0, rsi: 0, williamsR: 0, roc: 0, bollingerPercentB: 0, macd: 0, macdSignal: 0};

    for (let index = start + 1; index < end; index += 1) {
        const previous = index - 1;
        view.smaFast = columns.smaFast[previous];
        view.smaSlow = columns.smaSlow[previous];
        view.rsi = columns.rsi[previous];
        view.williamsR = columns.williamsR[previous];
        view.roc = columns.roc[previous];
        view.bollingerPercentB = columns.bollingerPercentB[previous];
        view.macd = columns.macd[previous];
        view.macdSignal = columns.macdSignal[previous];
        const targetPosition = decidePosition(view, rules, position);
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

function simulateSegmentReplay(rules: OptimizedStrategyRules, snapshots: IndicatorSnapshot[], startingPosition: number): SegmentResult {
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    const equityCurve = new Array<number>(snapshots.length);
    equityCurve[0] = equity;
    const returns: number[] = [];
    const trades: TradeMarker[] = [];

    for (let index = 1; index < snapshots.length; index += 1) {
        const previous = snapshots[index - 1];
        const current = snapshots[index];
        const targetPosition = decidePosition(previous, rules, position);
        const turnover = Math.abs(targetPosition - position);
        const priceReturn = current.close / previous.close - 1;
        const dailyReturn = position * priceReturn - turnover * TRANSACTION_COST;
        equity *= Math.max(0.01, 1 + dailyReturn);
        returns.push(dailyReturn);
        equityCurve[index] = equity;

        if (targetPosition !== position) {
            trades.push({
                date: previous.date,
                action: targetPosition > position ? "buy" : "sell",
                price: previous.close,
            });
            position = targetPosition;
        }
    }

    return {
        equityCurve,
        trades,
        totalReturn: equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpe(returns),
        maxDrawdown: calculateMaxDrawdown(equityCurve),
        endingPosition: position,
    };
}

function calculateSharpe(returns: number[]): number {
    if (returns.length < 2) {
        return 0;
    }
    const average = returns.reduce((sum, value) => sum + value, 0) / returns.length;
    const variance = returns.reduce((sum, value) => sum + (value - average) ** 2, 0) / returns.length;
    const deviation = Math.sqrt(variance);
    return deviation > 1e-9 ? (average / deviation) * Math.sqrt(252) : 0;
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

function calculateMaxDrawdown(equityCurve: number[]): number {
    let peak = equityCurve[0] ?? 0;
    let maxDrawdown = 0;
    equityCurve.forEach(equity => {
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);
    });
    return maxDrawdown;
}
