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

    return points.flatMap((point, index) => {
        if (index < warmup) {
            return [];
        }

        const smaFast = mean(closes.slice(index - parameters.smaFastPeriod + 1, index + 1));
        const smaSlow = mean(closes.slice(index - parameters.smaSlowPeriod + 1, index + 1));
        const bollingerWindow = closes.slice(index - parameters.bollingerPeriod + 1, index + 1);
        const bollingerBasis = mean(bollingerWindow);
        const bollingerDeviation = standardDeviation(bollingerWindow);
        const bollingerUpper = bollingerBasis + bollingerDeviation * parameters.bollingerMultiplier;
        const bollingerLower = bollingerBasis - bollingerDeviation * parameters.bollingerMultiplier;
        const bollingerRange = Math.max(bollingerUpper - bollingerLower, EPSILON);
        const bollingerPercentB = (point.close - bollingerLower) / bollingerRange;
        const bollingerBandwidth = bollingerRange / Math.max(bollingerBasis, EPSILON);
        const williamsR = calculateWilliamsR(points, index, parameters.williamsPeriod);
        const roc = point.close / closes[index - parameters.rocPeriod] - 1;
        const rsi = calculateRsi(closes, index, parameters.rsiPeriod);
        const volatility = calculateVolatility(closes, index, parameters.volatilityPeriod);
        const volumeWindow = volumes.slice(index - parameters.volumeZScorePeriod + 1, index + 1);
        const volumeDeviation = standardDeviation(volumeWindow);
        const volumeZScore = (point.volume - mean(volumeWindow)) / Math.max(volumeDeviation, EPSILON);
        const macdHistogram = macd[index] - macdSignal[index];

        const features = [
            clamp((point.close / smaFast - 1) * 10, -1, 1),
            clamp((point.close / smaSlow - 1) * 10, -1, 1),
            clamp((smaFast / smaSlow - 1) * 10, -1, 1),
            clamp((williamsR + 50) / 50, -1, 1),
            clamp(roc * 5, -1, 1),
            clamp((rsi - 50) / 50, -1, 1),
            clamp((macd[index] / point.close) * 25, -1, 1),
            clamp((macdSignal[index] / point.close) * 25, -1, 1),
            clamp((macdHistogram / point.close) * 50, -1, 1),
            clamp((bollingerPercentB - 0.5) * 2, -1, 1),
            clamp(bollingerBandwidth * 8, -1, 1),
            clamp(volatility * 5, -1, 1),
            clamp(volumeZScore / 3, -1, 1),
        ];

        return [
            {
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
                features,
            },
        ];
    });
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
    const result: number[] = [];
    values.forEach((value, index) => {
        result.push(index === 0 ? value : value * multiplier + result[index - 1] * (1 - multiplier));
    });
    return result;
}

function calculateWilliamsR(points: MarketDataPoint[], index: number, period: number): number {
    const window = points.slice(index - period + 1, index + 1);
    const highestHigh = Math.max(...window.map(point => point.high));
    const lowestLow = Math.min(...window.map(point => point.low));
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
    const returns: number[] = [];
    for (let cursor = index - period + 1; cursor <= index; cursor += 1) {
        returns.push(closes[cursor] / closes[cursor - 1] - 1);
    }
    return standardDeviation(returns) * Math.sqrt(252);
}

function mean(values: number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: number[]): number {
    const average = mean(values);
    return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / values.length);
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, Number.isFinite(value) ? value : 0));
}
