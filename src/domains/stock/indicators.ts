import type {IndicatorSnapshot, MarketDataPoint, OptimizedIndicatorParameters} from "../../lib/types";
import {DEFAULT_INDICATOR_PARAMETERS} from "./strategyGenome";

const EPSILON = 1e-9;

export function calculateIndicators(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters = DEFAULT_INDICATOR_PARAMETERS): IndicatorSnapshot[] {
    const warmup = getIndicatorWarmup(parameters);
    if (points.length <= warmup) {
        return [];
    }

    const closes = points.map(point => point.close);
    const volumes = points.map(point => point.volume);
    const emaFast = exponentialMovingAverage(closes, parameters.macdFastPeriod);
    const emaSlow = exponentialMovingAverage(closes, parameters.macdSlowPeriod);
    const macd = closes.map((_, index) => emaFast[index] - emaSlow[index]);
    const macdSignal = exponentialMovingAverage(macd, parameters.macdSignalPeriod);
    const snapshots: IndicatorSnapshot[] = [];

    for (let index = warmup; index < points.length; index += 1) {
        const point = points[index];
        const smaFast = meanRange(closes, index - parameters.smaFastPeriod + 1, index + 1);
        const smaSlow = meanRange(closes, index - parameters.smaSlowPeriod + 1, index + 1);
        const bollingerBasis = meanRange(closes, index - parameters.bollingerPeriod + 1, index + 1);
        const bollingerDeviation = standardDeviationRange(closes, index - parameters.bollingerPeriod + 1, index + 1, bollingerBasis);
        const bollingerUpper = bollingerBasis + bollingerDeviation * parameters.bollingerMultiplier;
        const bollingerLower = bollingerBasis - bollingerDeviation * parameters.bollingerMultiplier;
        const bollingerRange = Math.max(bollingerUpper - bollingerLower, EPSILON);
        const bollingerPercentB = (point.close - bollingerLower) / bollingerRange;
        const bollingerBandwidth = bollingerRange / Math.max(bollingerBasis, EPSILON);
        const williamsR = calculateWilliamsR(points, index, parameters.williamsPeriod);
        const roc = point.close / closes[index - parameters.rocPeriod] - 1;
        const rsi = calculateRsi(closes, index, parameters.rsiPeriod);
        const volatility = calculateVolatility(closes, index, parameters.volatilityPeriod);
        const volumeMean = meanRange(volumes, index - parameters.volumeZScorePeriod + 1, index + 1);
        const volumeDeviation = standardDeviationRange(volumes, index - parameters.volumeZScorePeriod + 1, index + 1, volumeMean);
        const volumeZScore = (point.volume - volumeMean) / Math.max(volumeDeviation, EPSILON);
        const macdHistogram = macd[index] - macdSignal[index];

        snapshots.push({
            date: point.date,
            close: point.close,
            smaFast,
            smaSlow,
            williamsR,
            roc,
            rsi,
            macd: macd[index],
            macdSignal: macdSignal[index],
            macdHistogram,
            bollingerUpper,
            bollingerLower,
            bollingerPercentB,
            bollingerBandwidth,
            volatility,
            volumeZScore,
        });
    }

    return snapshots;
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
        parameters.volumeZScorePeriod
    );
}

export function splitIndicators(snapshots: IndicatorSnapshot[], trainRatio = 0.8): {train: IndicatorSnapshot[]; test: IndicatorSnapshot[]; splitIndex: number} {
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
