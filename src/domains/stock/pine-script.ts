import type {Genome} from "../../lib/types";
import {decodeStockGenome} from "./strategy-genome";

export function createPineScript(genome: Genome, symbol: string): string {
    const {parameters, strategy} = decodeStockGenome(genome);
    const safeSymbol = symbol.replace(/[^A-Z0-9._-]/gi, "").toUpperCase() || "QQQ";
    const scoreTerms = strategy.weights.map((weight, index) => `f${index} * ${formatNumber(weight)}`);
    const scoreExpression = `tanh(${formatNumber(strategy.bias)} + ${scoreTerms.join(" + ")})`;

    return `//@version=6
strategy("EvoLab ${safeSymbol} Evolved Strategy", overlay=true, initial_capital=10000, pyramiding=0, commission_type=strategy.commission.percent, commission_value=0.1, process_orders_on_close=true)

// Optimized by EvoLab genetic algorithm — indicator parameters and signal weights only, no neural network.
// Validate this strategy on fresh out-of-sample data before considering any use.
smaFastPeriod = ${parameters.smaFastPeriod}
smaSlowPeriod = ${parameters.smaSlowPeriod}
williamsPeriod = ${parameters.williamsPeriod}
rocPeriod = ${parameters.rocPeriod}
rsiPeriod = ${parameters.rsiPeriod}
macdFastPeriod = ${parameters.macdFastPeriod}
macdSlowPeriod = ${parameters.macdSlowPeriod}
macdSignalPeriod = ${parameters.macdSignalPeriod}
bollingerPeriod = ${parameters.bollingerPeriod}
bollingerMultiplier = ${formatNumber(parameters.bollingerMultiplier)}
volatilityPeriod = ${parameters.volatilityPeriod}
volumeZScorePeriod = ${parameters.volumeZScorePeriod}
enterThreshold = ${formatNumber(strategy.enterThreshold)}
exitThreshold = ${formatNumber(strategy.exitThreshold)}

clamp(value) => math.min(1.0, math.max(-1.0, nz(value)))
tanh(value) =>
    limited = math.min(20.0, math.max(-20.0, value))
    exponent = math.exp(2.0 * limited)
    (exponent - 1.0) / (exponent + 1.0)

smaFast = ta.sma(close, smaFastPeriod)
smaSlow = ta.sma(close, smaSlowPeriod)
williamsHigh = ta.highest(high, williamsPeriod)
williamsLow = ta.lowest(low, williamsPeriod)
williamsR = -100.0 * (williamsHigh - close) / math.max(williamsHigh - williamsLow, 0.000000001)
roc = close / close[rocPeriod] - 1.0
priceChange = ta.change(close)
averageGain = ta.sma(math.max(priceChange, 0.0), rsiPeriod)
averageLoss = ta.sma(math.max(-priceChange, 0.0), rsiPeriod)
rsi = averageLoss < 0.000000001 ? 100.0 : 100.0 - 100.0 / (1.0 + averageGain / averageLoss)
macdLine = ta.ema(close, macdFastPeriod) - ta.ema(close, macdSlowPeriod)
macdSignal = ta.ema(macdLine, macdSignalPeriod)
macdHistogram = macdLine - macdSignal
bollingerBasis = ta.sma(close, bollingerPeriod)
bollingerDeviation = ta.stdev(close, bollingerPeriod) * bollingerMultiplier
bollingerUpper = bollingerBasis + bollingerDeviation
bollingerLower = bollingerBasis - bollingerDeviation
bollingerRange = math.max(bollingerUpper - bollingerLower, 0.000000001)
bollingerPercentB = (close - bollingerLower) / bollingerRange
bollingerBandwidth = bollingerRange / math.max(bollingerBasis, 0.000000001)
dailyReturn = close / close[1] - 1.0
volatility = ta.stdev(dailyReturn, volatilityPeriod) * math.sqrt(252.0)
volumeAverage = ta.sma(volume, volumeZScorePeriod)
volumeDeviation = ta.stdev(volume, volumeZScorePeriod)
volumeZScore = (volume - volumeAverage) / math.max(volumeDeviation, 0.000000001)

f0 = clamp((close / smaFast - 1.0) * 10.0)
f1 = clamp((close / smaSlow - 1.0) * 10.0)
f2 = clamp((smaFast / smaSlow - 1.0) * 10.0)
f3 = clamp((williamsR + 50.0) / 50.0)
f4 = clamp(roc * 5.0)
f5 = clamp((rsi - 50.0) / 50.0)
f6 = clamp(macdLine / close * 25.0)
f7 = clamp(macdSignal / close * 25.0)
f8 = clamp(macdHistogram / close * 50.0)
f9 = clamp((bollingerPercentB - 0.5) * 2.0)
f10 = clamp(bollingerBandwidth * 8.0)
f11 = clamp(volatility * 5.0)
f12 = clamp(volumeZScore / 3.0)

score = ${scoreExpression}

ready = not na(smaSlow) and not na(williamsR) and not na(roc) and not na(rsi) and not na(macdSignal) and not na(bollingerUpper) and not na(volatility) and not na(volumeZScore)
buySignal = ready and score > enterThreshold
sellSignal = ready and score < exitThreshold

if buySignal and strategy.position_size <= 0
    strategy.entry("Long", strategy.long)
if sellSignal and strategy.position_size > 0
    strategy.close("Long")

plot(smaFast, "Optimized SMA Fast", color=color.yellow)
plot(smaSlow, "Optimized SMA Slow", color=color.blue)
upperPlot = plot(bollingerUpper, "Optimized BB Upper", color=color.new(color.gray, 35))
lowerPlot = plot(bollingerLower, "Optimized BB Lower", color=color.new(color.gray, 35))
fill(upperPlot, lowerPlot, color=color.new(color.gray, 92))
plotshape(buySignal and strategy.position_size <= 0, title="Buy", style=shape.triangleup, location=location.belowbar, color=color.lime, size=size.tiny)
plotshape(sellSignal and strategy.position_size > 0, title="Sell", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.tiny)
`;
}

function formatNumber(value: number): string {
    return Number(value.toFixed(8)).toString();
}
