import {argMax, NeuralNetworkAdapter} from "../../lib/neural-network";
import type {Genome, IndicatorSnapshot, MarketDataPoint, OptimizedIndicatorParameters, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import {calculateIndicators, splitIndicators} from "./indicators";
import {decodeStockGenome, STOCK_TOPOLOGY} from "./strategy-genome";

export {STOCK_TOPOLOGY} from "./strategy-genome";

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;

/** Number of contiguous sub-periods the training segment is scored across (each a mini out-of-sample window). */
const VALIDATION_FOLDS = 5;
/** Weight-decay coefficient (× mean-square of the network genome) — nudges the GA toward smaller, more generalizable weights. */
const WEIGHT_L2_PENALTY = 8;
/** Trades per year a strategy can make before the churn penalty kicks in. */
const TRADE_ALLOWANCE_PER_YEAR = 12;
/** Fitness deducted per annualized trade beyond the allowance — high-churn policies are usually noise fits. */
const CHURN_PENALTY = 1.5;

/** Shared adapter so evaluate/replay do not allocate a new network graph per genome. */
const networkAdapter = new NeuralNetworkAdapter(STOCK_TOPOLOGY);

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
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const snapshots = getIndicatorSnapshots(points, parameters);
    const {train} = splitIndicators(snapshots);
    if (train.length < VALIDATION_FOLDS * 20) {
        return -1_000;
    }

    // Score the genome across several contiguous sub-periods of the training segment. Each fold is a
    // mini out-of-sample window, so a policy that only curve-fits one regime scores badly on the others.
    const foldSize = Math.floor(train.length / VALIDATION_FOLDS);
    const scores: number[] = [];
    for (let fold = 0; fold < VALIDATION_FOLDS; fold += 1) {
        const start = fold * foldSize;
        const end = fold === VALIDATION_FOLDS - 1 ? train.length : start + foldSize;
        const window = train.slice(start, end);
        scores.push(segmentScore(simulateSegment(networkGenome, window, 0), window));
    }

    // Blend the average with the worst fold so consistency across every sub-period is rewarded, not a
    // lucky fit to one window. Weight the worst fold heavier — one blow-out window must not carry the
    // score. Plus L2 weight decay on the network genome.
    const mean = scores.reduce((sum, value) => sum + value, 0) / scores.length;
    const worst = Math.min(...scores);
    const regularization = WEIGHT_L2_PENALTY * meanSquare(networkGenome);
    return 0.35 * mean + 0.65 * worst - regularization;
}

/** Mean of the squared network weights — the weight-decay term. */
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

/** Per-window fitness: reward return in excess of buy & hold, reward Sharpe, punish drawdown and churn. */
function segmentScore(result: SegmentResult, snapshots: IndicatorSnapshot[]): number {
    const first = snapshots[0]?.close ?? 1;
    const last = snapshots.at(-1)?.close ?? first;
    const benchmarkReturn = last / first - 1;
    const excessReturn = result.totalReturn - benchmarkReturn;
    const years = Math.max(snapshots.length / 252, 1 / 252);
    const tradesPerYear = result.trades.length / years;
    const churnPenalty = Math.max(0, tradesPerYear - TRADE_ALLOWANCE_PER_YEAR) * CHURN_PENALTY;
    return excessReturn * 100 + result.sharpe * 10 - result.maxDrawdown * 40 - churnPenalty;
}

export function createTradingReplay(genome: Genome, points: MarketDataPoint[]): TradingReplay {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const snapshots = getIndicatorSnapshots(points, parameters);
    const {train, test, splitIndex} = splitIndicators(snapshots);
    const trainResult = simulateSegment(networkGenome, train, 0);
    const testResult = simulateSegment(networkGenome, test, trainResult.endingPosition);
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

function simulateSegment(genome: Genome, snapshots: IndicatorSnapshot[], startingPosition: number): SegmentResult {
    const runNetwork = networkAdapter.createRunner(genome);
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    const equityCurve = [equity];
    const returns: number[] = [];
    const trades: TradeMarker[] = [];

    for (let index = 1; index < snapshots.length; index += 1) {
        const previous = snapshots[index - 1];
        const current = snapshots[index];
        const output = runNetwork([...previous.features, position]);
        const action = argMax(output);
        const targetPosition = action === 0 ? 1 : action === 2 ? 0 : position;
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
