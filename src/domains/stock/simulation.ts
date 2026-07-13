import type {Genome, IndicatorSnapshot, MarketDataPoint, OptimizedIndicatorParameters, OptimizedStrategyRules, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import type {IndicatorColumns} from "./indicators";
import {calculateIndicatorColumns, columnsToSnapshots} from "./indicators";
import {decodeStockGenome, MAX_ENTRY_VOLATILITY, VOLUME_Z_CONFIRM} from "./strategyGenome";

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;
/** Keep a small LRU of indicator series — Float64Array columns, ~1MB per full history entry. */
const MAX_INDICATOR_CACHE = 16;

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
    // Same boundary as createTradingReplay so fitness and UI metrics agree on train window.
    const trainLength = Math.max(2, Math.floor(columns.length * 0.8));
    if (trainLength < 100) {
        return -1_000;
    }

    // Full-train score gives a smooth gradient toward strategies that actually make money.
    // Worst-half score (30%) keeps one-lucky-bull-run overfits from dominating.
    // Old fitness was pure min(excess vs B&H) — on QQQ that crushed every coherent TA
    // strategy into "sit in cash" / random churn with no learnable path to good charts.
    const mid = Math.floor(trainLength / 2);
    const full = simulateColumnMetrics(rules, columns, 0, 0, trainLength);
    const firstHalf = simulateColumnMetrics(rules, columns, 0, 0, mid + 1);
    const secondHalf = simulateColumnMetrics(rules, columns, firstHalf.endingPosition, mid, trainLength);
    const fullScore = scoreSegment(full, columns, 0, trainLength);
    const robustScore = Math.min(scoreSegment(firstHalf, columns, 0, mid + 1), scoreSegment(secondHalf, columns, mid, trainLength));
    return fullScore * 0.7 + robustScore * 0.3;
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
    // Mild churn tax: many flips destroy edge after costs; sharpe already hurts noise,
    // but explicit turnover (approx from return path variance vs position) is heavy —
    // use drawdown + sharpe; add small penalty when total return is near-zero (idle cash).
    const idlePenalty = Math.abs(metrics.totalReturn) < 0.02 ? 8 : 0;
    return strategyCagr * 100 + metrics.sharpe * 15 - metrics.maxDrawdown * 40 + excess * 35 - idlePenalty;
}

function annualize(totalReturn: number, years: number): number {
    return Math.pow(1 + Math.max(totalReturn, -0.99), 1 / years) - 1;
}

export function createTradingReplay(genome: Genome, points: MarketDataPoint[]): TradingReplay {
    const {parameters, rules} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    // Same train/test split as fitness + previous snapshot-based replay (test overlaps last train bar).
    const splitIndex = Math.max(2, Math.floor(columns.length * 0.8));
    const trainResult = simulateColumnReplay(rules, columns, points, 0, splitIndex, 0);
    const testResult = simulateColumnReplay(rules, columns, points, Math.max(0, splitIndex - 1), columns.length, trainResult.endingPosition);
    const trainCurve = trainResult.equityCurve;
    const testScale = trainCurve.at(-1) ?? STARTING_EQUITY;
    const fullCurve = new Array<number>(columns.length);
    for (let index = 0; index < trainCurve.length; index += 1) {
        fullCurve[index] = trainCurve[index];
    }
    // testResult starts at splitIndex-1 (overlap); map equity onto splitIndex..end with scale continuity.
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
        optimizedRules: rules,
    };
}

/** The subset of indicator fields the voting rules actually read. */
export type StrategySignals = Pick<IndicatorSnapshot, "smaFast" | "smaSlow" | "rsi" | "williamsR" | "roc" | "bollingerPercentB" | "macd" | "macdSignal" | "volatility" | "volumeZScore">;

/**
 * Multi-indicator voting rules evolved by GA.
 * buy / hold / sell → target long (1) or cash (0); never short.
 *
 * Styles keep trend-following and mean-reversion from vetoing each other:
 * - trend: SMA, MACD, ROC, volume-Z (directional / breakout)
 * - mean_reversion: RSI, Williams, Bollinger %B (only vote at extremes)
 * - hybrid: both families
 */
export function decidePosition(snapshot: StrategySignals, rules: OptimizedStrategyRules, position: number): number {
    const style = rules.strategyStyle;
    const useTrend = style === "trend" || style === "hybrid";
    const useReversion = style === "mean_reversion" || style === "hybrid";
    const trendUp = snapshot.smaFast > snapshot.smaSlow;
    const macdUp = snapshot.macd > snapshot.macdSignal;
    let buySignals = 0;
    let sellSignals = 0;

    if (useTrend) {
        if (trendUp) {
            buySignals += 1;
        } else {
            sellSignals += 1;
        }
        if (macdUp) {
            buySignals += 1;
        } else {
            sellSignals += 1;
        }
        if (snapshot.roc > rules.rocBuy) {
            buySignals += 1;
        } else if (snapshot.roc < rules.rocSell) {
            sellSignals += 1;
        }
        // Evolved volume-Z period feeds a real confirmation (was previously computed but unused).
        if (snapshot.volumeZScore >= VOLUME_Z_CONFIRM) {
            buySignals += 1;
        }
    }

    if (useReversion) {
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
        if (snapshot.bollingerPercentB < rules.bollingerBuy) {
            buySignals += 1;
        }
        if (snapshot.bollingerPercentB > rules.bollingerSell) {
            sellSignals += 1;
        }
    }

    if (position <= 0) {
        // Vol filter uses evolved volatility lookback — avoid chasing panicked spikes.
        if (snapshot.volatility > MAX_ENTRY_VOLATILITY) {
            return 0;
        }
        // Trend styles only open longs with the tape (SMA fast > slow).
        if (style === "trend" && !trendUp) {
            return 0;
        }
        if (buySignals >= rules.minBuySignals) {
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
    const view: StrategySignals = {
        smaFast: 0,
        smaSlow: 0,
        rsi: 0,
        williamsR: 0,
        roc: 0,
        bollingerPercentB: 0,
        macd: 0,
        macdSignal: 0,
        volatility: 0,
        volumeZScore: 0,
    };

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
        view.volatility = columns.volatility[previous];
        view.volumeZScore = columns.volumeZScore[previous];
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

/**
 * Replay path over columns[start, end) — builds equity curve + trade markers without
 * materializing a full IndicatorSnapshot[] first (that was a multi-MB spike every refresh).
 */
function simulateColumnReplay(rules: OptimizedStrategyRules, columns: IndicatorColumns, points: MarketDataPoint[], start: number, end: number, startingPosition: number): SegmentResult {
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
    const view: StrategySignals = {
        smaFast: 0,
        smaSlow: 0,
        rsi: 0,
        williamsR: 0,
        roc: 0,
        bollingerPercentB: 0,
        macd: 0,
        macdSignal: 0,
        volatility: 0,
        volumeZScore: 0,
    };

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
        view.volatility = columns.volatility[previous];
        view.volumeZScore = columns.volumeZScore[previous];
        const targetPosition = decidePosition(view, rules, position);
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
