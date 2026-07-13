import type {Genome} from "../../lib/types";
import {decodeStockGenome} from "./strategy-genome";

export function createPineScript(genome: Genome, symbol: string): string {
    const {parameters, rules} = decodeStockGenome(genome);
    const safeSymbol = symbol.replace(/[^A-Z0-9._-]/gi, "").toUpperCase() || "QQQ";

    return `//@version=6
strategy("EvoLab ${safeSymbol} Evolved Strategy", overlay=true, initial_capital=1000000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, pyramiding=0, commission_type=strategy.commission.percent, commission_value=0.1, process_orders_on_close=true)

// Optimized by EvoLab genetic algorithm — indicator periods and multi-signal rule thresholds (no neural network).
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

rsiBuy = ${rules.rsiBuy}
rsiSell = ${rules.rsiSell}
williamsBuy = ${formatNumber(rules.williamsBuy)}
williamsSell = ${formatNumber(rules.williamsSell)}
rocBuy = ${formatNumber(rules.rocBuy)}
rocSell = ${formatNumber(rules.rocSell)}
bollingerBuy = ${formatNumber(rules.bollingerBuy)}
bollingerSell = ${formatNumber(rules.bollingerSell)}
minBuySignals = ${rules.minBuySignals}
minSellSignals = ${rules.minSellSignals}
useTrendFilter = ${rules.useTrendFilter}

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
bollingerBasis = ta.sma(close, bollingerPeriod)
bollingerDeviation = ta.stdev(close, bollingerPeriod) * bollingerMultiplier
bollingerUpper = bollingerBasis + bollingerDeviation
bollingerLower = bollingerBasis - bollingerDeviation
bollingerRange = math.max(bollingerUpper - bollingerLower, 0.000000001)
bollingerPercentB = (close - bollingerLower) / bollingerRange

trendUp = smaFast > smaSlow
buySignals = 0
sellSignals = 0
buySignals := buySignals + (trendUp ? 1 : 0)
sellSignals := sellSignals + (trendUp ? 0 : 1)
buySignals := buySignals + (rsi < rsiBuy ? 1 : 0)
sellSignals := sellSignals + (rsi > rsiSell ? 1 : 0)
buySignals := buySignals + (williamsR < williamsBuy ? 1 : 0)
sellSignals := sellSignals + (williamsR > williamsSell ? 1 : 0)
buySignals := buySignals + (roc > rocBuy ? 1 : 0)
sellSignals := sellSignals + (roc < rocSell ? 1 : 0)
buySignals := buySignals + (bollingerPercentB < bollingerBuy ? 1 : 0)
sellSignals := sellSignals + (bollingerPercentB > bollingerSell ? 1 : 0)
buySignals := buySignals + (macdLine > macdSignal ? 1 : 0)
sellSignals := sellSignals + (macdLine > macdSignal ? 0 : 1)

ready = not na(smaSlow) and not na(williamsR) and not na(roc) and not na(rsi) and not na(macdSignal) and not na(bollingerUpper)
trendOk = not useTrendFilter or trendUp
buySignal = ready and trendOk and buySignals >= minBuySignals
sellSignal = ready and sellSignals >= minSellSignals

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
