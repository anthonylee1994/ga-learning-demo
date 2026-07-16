import type {IndicatorSnapshot, MarketDataPoint, OptimizedIndicatorParameters} from "../../lib/types";
import {DEFAULT_INDICATOR_PARAMETERS} from "./strategyGenome";

const EPSILON = 1e-9;

/**
 * Column-major indicator series aligned to points[warmup..]. The GA fitness path evaluates
 * thousands of genomes per second — Float64Array columns keep that path free of per-bar
 * object allocations (a per-genome IndicatorSnapshot[] was ~MBs of GC churn each).
 */
export interface IndicatorColumns {
    warmup: number;
    length: number;
    open: Float64Array;
    high: Float64Array;
    low: Float64Array;
    close: Float64Array;
    /** close / previous close − 1（收盤日回報，畀 NN 用）。 */
    closeReturn: Float64Array;
    smaFast: Float64Array;
    smaSlow: Float64Array;
    williamsR: Float64Array;
    roc: Float64Array;
    rsi: Float64Array;
    macd: Float64Array;
    macdSignal: Float64Array;
    macdHistogram: Float64Array;
    bollingerUpper: Float64Array;
    bollingerLower: Float64Array;
    bollingerPercentB: Float64Array;
    bollingerBandwidth: Float64Array;
    volatility: Float64Array;
    volumeZScore: Float64Array;
    nDayHigh: Float64Array;
    newHighRatio: Float64Array;
    nDayLow: Float64Array;
    newLowRatio: Float64Array;
}

export function calculateIndicatorColumns(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters = DEFAULT_INDICATOR_PARAMETERS): IndicatorColumns {
    const warmup = getIndicatorWarmup(parameters);
    const length = Math.max(0, points.length - warmup);
    const columns: IndicatorColumns = {
        warmup,
        length,
        open: new Float64Array(length),
        high: new Float64Array(length),
        low: new Float64Array(length),
        close: new Float64Array(length),
        closeReturn: new Float64Array(length),
        smaFast: new Float64Array(length),
        smaSlow: new Float64Array(length),
        williamsR: new Float64Array(length),
        roc: new Float64Array(length),
        rsi: new Float64Array(length),
        macd: new Float64Array(length),
        macdSignal: new Float64Array(length),
        macdHistogram: new Float64Array(length),
        bollingerUpper: new Float64Array(length),
        bollingerLower: new Float64Array(length),
        bollingerPercentB: new Float64Array(length),
        bollingerBandwidth: new Float64Array(length),
        volatility: new Float64Array(length),
        volumeZScore: new Float64Array(length),
        nDayHigh: new Float64Array(length),
        newHighRatio: new Float64Array(length),
        nDayLow: new Float64Array(length),
        newLowRatio: new Float64Array(length),
    };
    if (length === 0) {
        return columns;
    }

    const closes = points.map(point => point.close);
    const volumes = points.map(point => point.volume);
    const emaFast = exponentialMovingAverage(closes, parameters.macdFastPeriod);
    const emaSlow = exponentialMovingAverage(closes, parameters.macdSlowPeriod);
    const macd = closes.map((_, index) => emaFast[index] - emaSlow[index]);
    const macdSignal = exponentialMovingAverage(macd, parameters.macdSignalPeriod);

    for (let index = warmup; index < points.length; index += 1) {
        const point = points[index];
        const cursor = index - warmup;
        const bollingerBasis = meanRange(closes, index - parameters.bollingerPeriod + 1, index + 1);
        const bollingerDeviation = standardDeviationRange(closes, index - parameters.bollingerPeriod + 1, index + 1, bollingerBasis);
        const bollingerUpper = bollingerBasis + bollingerDeviation * parameters.bollingerMultiplier;
        const bollingerLower = bollingerBasis - bollingerDeviation * parameters.bollingerMultiplier;
        const bollingerRange = Math.max(bollingerUpper - bollingerLower, EPSILON);
        const volumeMean = meanRange(volumes, index - parameters.volumeZScorePeriod + 1, index + 1);
        const volumeDeviation = standardDeviationRange(volumes, index - parameters.volumeZScorePeriod + 1, index + 1, volumeMean);
        const nDayHigh = highestHighRange(points, index - parameters.newHighPeriod + 1, index + 1);
        const nDayLow = lowestLowRange(points, index - parameters.newLowPeriod + 1, index + 1);

        const prevClose = index > 0 ? closes[index - 1] : point.open;
        columns.open[cursor] = point.open;
        columns.high[cursor] = point.high;
        columns.low[cursor] = point.low;
        columns.close[cursor] = point.close;
        columns.closeReturn[cursor] = point.close / Math.max(prevClose, EPSILON) - 1;
        columns.smaFast[cursor] = meanRange(closes, index - parameters.smaFastPeriod + 1, index + 1);
        columns.smaSlow[cursor] = meanRange(closes, index - parameters.smaSlowPeriod + 1, index + 1);
        columns.williamsR[cursor] = calculateWilliamsR(points, index, parameters.williamsPeriod);
        columns.roc[cursor] = point.close / closes[index - parameters.rocPeriod] - 1;
        columns.rsi[cursor] = calculateRsi(closes, index, parameters.rsiPeriod);
        columns.macd[cursor] = macd[index];
        columns.macdSignal[cursor] = macdSignal[index];
        columns.macdHistogram[cursor] = macd[index] - macdSignal[index];
        columns.bollingerUpper[cursor] = bollingerUpper;
        columns.bollingerLower[cursor] = bollingerLower;
        columns.bollingerPercentB[cursor] = (point.close - bollingerLower) / bollingerRange;
        columns.bollingerBandwidth[cursor] = bollingerRange / Math.max(bollingerBasis, EPSILON);
        columns.volatility[cursor] = calculateVolatility(closes, index, parameters.volatilityPeriod);
        columns.volumeZScore[cursor] = (point.volume - volumeMean) / Math.max(volumeDeviation, EPSILON);
        columns.nDayHigh[cursor] = nDayHigh;
        columns.newHighRatio[cursor] = point.close / Math.max(nDayHigh, EPSILON);
        columns.nDayLow[cursor] = nDayLow;
        // 對稱 newHighRatio：越接近 N 日低 → ratio 越近 1
        columns.newLowRatio[cursor] = nDayLow / Math.max(point.close, EPSILON);
    }

    return columns;
}

/** Object snapshots for the replay/chart path — built on demand from the column series. */
export function columnsToSnapshots(points: MarketDataPoint[], columns: IndicatorColumns): IndicatorSnapshot[] {
    const snapshots: IndicatorSnapshot[] = new Array(columns.length);
    for (let cursor = 0; cursor < columns.length; cursor += 1) {
        snapshots[cursor] = {
            date: points[columns.warmup + cursor].date,
            close: columns.close[cursor],
            smaFast: columns.smaFast[cursor],
            smaSlow: columns.smaSlow[cursor],
            williamsR: columns.williamsR[cursor],
            roc: columns.roc[cursor],
            rsi: columns.rsi[cursor],
            macd: columns.macd[cursor],
            macdSignal: columns.macdSignal[cursor],
            macdHistogram: columns.macdHistogram[cursor],
            bollingerUpper: columns.bollingerUpper[cursor],
            bollingerLower: columns.bollingerLower[cursor],
            bollingerPercentB: columns.bollingerPercentB[cursor],
            bollingerBandwidth: columns.bollingerBandwidth[cursor],
            volatility: columns.volatility[cursor],
            volumeZScore: columns.volumeZScore[cursor],
            nDayHigh: columns.nDayHigh[cursor],
            newHighRatio: columns.newHighRatio[cursor],
            nDayLow: columns.nDayLow[cursor],
            newLowRatio: columns.newLowRatio[cursor],
        };
    }
    return snapshots;
}

export function calculateIndicators(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters = DEFAULT_INDICATOR_PARAMETERS): IndicatorSnapshot[] {
    return columnsToSnapshots(points, calculateIndicatorColumns(points, parameters));
}

export function getIndicatorWarmup(parameters: OptimizedIndicatorParameters): number {
    return Math.max(
        parameters.smaSlowPeriod,
        parameters.williamsPeriod,
        parameters.rocPeriod,
        parameters.rsiPeriod,
        parameters.macdSlowPeriod + parameters.macdSignalPeriod,
        parameters.bollingerPeriod,
        parameters.volatilityPeriod,
        parameters.volumeZScorePeriod,
        parameters.newHighPeriod,
        parameters.newLowPeriod
    );
}

export function splitIndicators(snapshots: IndicatorSnapshot[], trainRatio = 0.6): {train: IndicatorSnapshot[]; test: IndicatorSnapshot[]; splitIndex: number} {
    const splitIndex = Math.max(2, Math.floor(snapshots.length * trainRatio));
    return {
        train: snapshots.slice(0, splitIndex),
        test: snapshots.slice(splitIndex - 1),
        splitIndex,
    };
}

function exponentialMovingAverage(values: number[], period: number): number[] {
    const multiplier = 2 / (period + 1);
    const result: number[] = new Array(values.length);
    values.forEach((value, index) => {
        result[index] = index === 0 ? value : value * multiplier + result[index - 1] * (1 - multiplier);
    });
    return result;
}

function calculateWilliamsR(points: MarketDataPoint[], index: number, period: number): number {
    let highestHigh = -Infinity;
    let lowestLow = Infinity;
    for (let cursor = index - period + 1; cursor <= index; cursor += 1) {
        const point = points[cursor];
        highestHigh = Math.max(highestHigh, point.high);
        lowestLow = Math.min(lowestLow, point.low);
    }
    return ((highestHigh - points[index].close) / Math.max(highestHigh - lowestLow, EPSILON)) * -100;
}

function calculateRsi(closes: number[], index: number, period: number): number {
    let gains = 0;
    let losses = 0;
    for (let cursor = index - period + 1; cursor <= index; cursor += 1) {
        const change = closes[cursor] - closes[cursor - 1];
        if (change >= 0) {
            gains += change;
        } else {
            losses -= change;
        }
    }
    const averageGain = gains / period;
    const averageLoss = losses / period;
    if (averageLoss < EPSILON) {
        return 100;
    }
    const relativeStrength = averageGain / averageLoss;
    return 100 - 100 / (1 + relativeStrength);
}

function calculateVolatility(closes: number[], index: number, period: number): number {
    let sum = 0;
    let sumSq = 0;
    for (let cursor = index - period + 1; cursor <= index; cursor += 1) {
        const value = closes[cursor] / closes[cursor - 1] - 1;
        sum += value;
        sumSq += value * value;
    }
    const meanValue = sum / period;
    const variance = sumSq / period - meanValue * meanValue;
    return Math.sqrt(Math.max(0, variance)) * Math.sqrt(252);
}

function highestHighRange(points: MarketDataPoint[], start: number, end: number): number {
    let highest = -Infinity;
    for (let index = start; index < end; index += 1) {
        highest = Math.max(highest, points[index].high);
    }
    return highest;
}

function lowestLowRange(points: MarketDataPoint[], start: number, end: number): number {
    let lowest = Infinity;
    for (let index = start; index < end; index += 1) {
        lowest = Math.min(lowest, points[index].low);
    }
    return lowest;
}

function meanRange(values: number[], start: number, end: number): number {
    let sum = 0;
    for (let index = start; index < end; index += 1) {
        sum += values[index];
    }
    return sum / (end - start);
}

function standardDeviationRange(values: number[], start: number, end: number, average: number): number {
    let sumSq = 0;
    for (let index = start; index < end; index += 1) {
        const delta = values[index] - average;
        sumSq += delta * delta;
    }
    return Math.sqrt(sumSq / (end - start));
}
