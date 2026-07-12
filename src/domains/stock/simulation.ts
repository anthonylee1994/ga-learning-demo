import type {Genome, IndicatorSnapshot, MarketDataPoint, OptimizedIndicatorParameters, StockStrategy, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import {calculateIndicators, splitIndicators} from "./indicators";
import {decodeStockGenome} from "./strategy-genome";

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;

/** Fraction of the training segment used to fit; the remainder is held out as validation. */
const TRAIN_VALIDATION_RATIO = 0.7;
/** L2 penalty on the signal weights — nudges the GA toward simpler, more generalizable policies. */
const WEIGHT_L2_PENALTY = 0.2;

/**
 * Cache indicator snapshots by the exact points array reference.
 * evaluate() used to recompute indicators for every genome every generation —
 * with ~2.5k daily bars that was a large allocation storm.
 */
let cachedPoints: MarketDataPoint[] | null = null;
const cachedSnapshots = new Map<string, IndicatorSnapshot[]>();

interface SegmentResult {
    equityCurve: number[];
    trades: TradeMarker[];
    returns: number[];
    totalReturn: number;
    sharpe: number;
    maxDrawdown: number;
    endingPosition: number;
}

export function getIndicatorSnapshots(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters): IndicatorSnapshot[] {
    if (points !== cachedPoints) {
        cachedPoints = points;
        cachedSnapshots.clear();
    }
    const key = Object.values(parameters).join(":");
    const cached = cachedSnapshots.get(key);
    if (cached) {
        return cached;
    }
    const snapshots = calculateIndicators(points, parameters);
    if (cachedSnapshots.size >= 256) {
        cachedSnapshots.clear();
    }
    cachedSnapshots.set(key, snapshots);
    return snapshots;
}

export function evaluateStockGenome(genome: Genome, points: MarketDataPoint[]): number {
    const {parameters, strategy} = decodeStockGenome(genome);
    const snapshots = getIndicatorSnapshots(points, parameters);
    const {train} = splitIndicators(snapshots);
    if (train.length < 60) {
        return -1_000;
    }

    // Split the training segment into a fit window and a held-out validation window.
    // Selection rewards generalization, not in-sample curve-fitting.
    const splitAt = Math.max(30, Math.floor(train.length * TRAIN_VALIDATION_RATIO));
    const fit = train.slice(0, splitAt);
    const validation = train.slice(splitAt - 1);
    const fitResult = simulateSegment(strategy, fit, 0);
    const validationResult = simulateSegment(strategy, validation, fitResult.endingPosition);
    const fitScore = segmentScore(fitResult, fit);
    const validationScore = segmentScore(validationResult, validation);

    // mean − 0.5·max(0, fit − val): when the genome overfits (fit ≫ val) this collapses to the
    // validation score alone; when it generalizes (val ≥ fit) it keeps the average. Plus L2 on weights.
    const mean = (fitScore + validationScore) / 2;
    const overfitPenalty = Math.max(0, fitScore - validationScore) * 0.5;
    const regularization = WEIGHT_L2_PENALTY * strategy.weights.reduce((sum, weight) => sum + weight * weight, 0);
    return mean - overfitPenalty - regularization;
}

/** Per-window fitness: reward return in excess of buy & hold, reward Sharpe, punish drawdown. */
function segmentScore(result: SegmentResult, snapshots: IndicatorSnapshot[]): number {
    const first = snapshots[0]?.close ?? 1;
    const last = snapshots.at(-1)?.close ?? first;
    const benchmarkReturn = last / first - 1;
    const excessReturn = result.totalReturn - benchmarkReturn;
    return excessReturn * 100 + result.sharpe * 10 - result.maxDrawdown * 40;
}

export function createTradingReplay(genome: Genome, points: MarketDataPoint[]): TradingReplay {
    const {parameters, strategy} = decodeStockGenome(genome);
    const snapshots = getIndicatorSnapshots(points, parameters);
    const {train, test, splitIndex} = splitIndicators(snapshots);
    const trainResult = simulateSegment(strategy, train, 0);
    const testResult = simulateSegment(strategy, test, trainResult.endingPosition);
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
    };
}

function simulateSegment(strategy: StockStrategy, snapshots: IndicatorSnapshot[], startingPosition: number): SegmentResult {
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    const equityCurve = [equity];
    const returns: number[] = [];
    const trades: TradeMarker[] = [];

    for (let index = 1; index < snapshots.length; index += 1) {
        const previous = snapshots[index - 1];
        const current = snapshots[index];
        const score = scoreStrategy(strategy, previous.features);
        // Hysteresis band: enter long on a strong positive score, exit to cash once it drops
        // below the exit threshold. Holding between the thresholds keeps turnover (and cost) low.
        const targetPosition = position < 0.5 ? (score > strategy.enterThreshold ? 1 : 0) : score < strategy.exitThreshold ? 0 : 1;
        const turnover = Math.abs(targetPosition - position);
        const priceReturn = current.close / previous.close - 1;
        const dailyReturn = position * priceReturn - turnover * TRANSACTION_COST;
        equity *= Math.max(0.01, 1 + dailyReturn);
        returns.push(dailyReturn);
        equityCurve.push(equity);

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
        returns,
        totalReturn: equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpe(returns),
        maxDrawdown: calculateMaxDrawdown(equityCurve),
        endingPosition: position,
    };
}

/** Weighted linear signal squashed to [-1, 1]; the GA evolves the weights, bias and thresholds. */
function scoreStrategy(strategy: StockStrategy, features: number[]): number {
    let sum = strategy.bias;
    for (let index = 0; index < strategy.weights.length; index += 1) {
        sum += strategy.weights[index] * (features[index] ?? 0);
    }
    return Math.tanh(sum);
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

function calculateMaxDrawdown(equityCurve: number[]): number {
    let peak = equityCurve[0] ?? 0;
    let maxDrawdown = 0;
    equityCurve.forEach(equity => {
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);
    });
    return maxDrawdown;
}
