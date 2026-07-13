import type {Genome, IndicatorSnapshot, MarketDataPoint, OptimizedStrategyRules, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import {calculateIndicators, splitIndicators} from "./indicators";
import {decodeStockGenome} from "./strategy-genome";

const STARTING_EQUITY = 10_000;
const TRANSACTION_COST = 0.001;

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

export function getIndicatorSnapshots(points: MarketDataPoint[], parameters: ReturnType<typeof decodeStockGenome>["parameters"]): IndicatorSnapshot[] {
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
    const {parameters, rules} = decodeStockGenome(genome);
    const snapshots = getIndicatorSnapshots(points, parameters);
    const {train} = splitIndicators(snapshots);
    if (train.length < 100) {
        return -1_000;
    }

    // Fitness: performance (return in excess of buy & hold) minus a max-drawdown penalty.
    const result = simulateSegment(rules, train, 0);
    const first = train[0]?.close ?? 1;
    const last = train.at(-1)?.close ?? first;
    const benchmarkReturn = last / first - 1;
    const excessReturn = result.totalReturn - benchmarkReturn;
    return excessReturn * 100 - result.maxDrawdown * 40;
}

export function createTradingReplay(genome: Genome, points: MarketDataPoint[]): TradingReplay {
    const {parameters, rules} = decodeStockGenome(genome);
    const snapshots = getIndicatorSnapshots(points, parameters);
    const {train, test, splitIndex} = splitIndicators(snapshots);
    const trainResult = simulateSegment(rules, train, 0);
    const testResult = simulateSegment(rules, test, trainResult.endingPosition);
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

/**
 * Multi-indicator voting rules evolved by GA.
 * buy / hold / sell → target long (1) or cash (0); never short.
 */
export function decidePosition(snapshot: IndicatorSnapshot, rules: OptimizedStrategyRules, position: number): number {
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

function simulateSegment(rules: OptimizedStrategyRules, snapshots: IndicatorSnapshot[], startingPosition: number): SegmentResult {
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    const equityCurve = [equity];
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
