import {argMax, NeuralNetworkAdapter} from "../../lib/neural-network";
import type {Genome, IndicatorSnapshot, MarketDataPoint, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import {calculateIndicators, splitIndicators} from "./indicators";

export const STOCK_TOPOLOGY = {
    inputSize: 14,
    hiddenLayers: [16, 8],
    outputSize: 3,
};

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;

/** Shared adapter so evaluate/replay do not allocate a new network graph per genome. */
const networkAdapter = new NeuralNetworkAdapter(STOCK_TOPOLOGY);

/**
 * Cache indicator snapshots by the exact points array reference.
 * evaluate() used to recompute indicators for every genome every generation —
 * with ~2.5k daily bars that was a large allocation storm.
 */
let cachedPoints: MarketDataPoint[] | null = null;
let cachedSnapshots: IndicatorSnapshot[] = [];

interface SegmentResult {
    equityCurve: number[];
    trades: TradeMarker[];
    returns: number[];
    totalReturn: number;
    sharpe: number;
    maxDrawdown: number;
    endingPosition: number;
}

export function getIndicatorSnapshots(points: MarketDataPoint[]): IndicatorSnapshot[] {
    if (points === cachedPoints) {
        return cachedSnapshots;
    }
    cachedPoints = points;
    cachedSnapshots = calculateIndicators(points);
    return cachedSnapshots;
}

export function evaluateStockGenome(genome: Genome, points: MarketDataPoint[]): number {
    const snapshots = getIndicatorSnapshots(points);
    const {train} = splitIndicators(snapshots);
    if (train.length < 30) {
        return -1_000;
    }
    const result = simulateSegment(genome, train, 0);
    return result.totalReturn * 100 + result.sharpe * 8 - result.maxDrawdown * 45;
}

export function createTradingReplay(genome: Genome, points: MarketDataPoint[]): TradingReplay {
    const snapshots = getIndicatorSnapshots(points);
    const {train, test, splitIndex} = splitIndicators(snapshots);
    const trainResult = simulateSegment(genome, train, 0);
    const testResult = simulateSegment(genome, test, trainResult.endingPosition);
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
        sma20: snapshot.sma20,
        sma50: snapshot.sma50,
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
