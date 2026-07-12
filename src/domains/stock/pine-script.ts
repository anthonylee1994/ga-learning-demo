import type {Genome} from "../../lib/types";
import {decodeStockGenome, STOCK_TOPOLOGY} from "./strategy-genome";

interface DenseLayer {
    biases: number[];
    weights: number[][];
}

export function createPineScript(genome: Genome, symbol: string): string {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const layers = decodeLayers(networkGenome);
    const safeSymbol = symbol.replace(/[^A-Z0-9._-]/gi, "").toUpperCase() || "QQQ";
    const inputNames = Array.from({length: STOCK_TOPOLOGY.inputSize}, (_, index) => `f${index}`);
    const hiddenOneNames = layers[0].biases.map((_, index) => `h1_${index}`);
    const hiddenTwoNames = layers[1].biases.map((_, index) => `h2_${index}`);
    const outputNames = ["outBuy", "outHold", "outSell"];
    const networkLines = [
        ...createLayerLines(hiddenOneNames, inputNames, layers[0]),
        ...createLayerLines(hiddenTwoNames, hiddenOneNames, layers[1]),
        ...createLayerLines(outputNames, hiddenTwoNames, layers[2], false),
    ];

    return `//@version=6
strategy("EvoLab ${safeSymbol} Neuroevolution", overlay=true, initial_capital=10000, pyramiding=0, commission_type=strategy.commission.percent, commission_value=0.1, process_orders_on_close=true)

// Optimized by EvoLab genetic algorithm. Training data is not embedded.
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
f11 = strategy.position_size > 0 ? 1.0 : 0.0

${networkLines.join("\n")}

ready = not na(smaSlow) and not na(williamsR) and not na(roc) and not na(rsi) and not na(macdSignal) and not na(bollingerUpper)
buySignal = ready and outBuy >= outHold and outBuy >= outSell
sellSignal = ready and outSell > outBuy and outSell > outHold

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

function decodeLayers(networkGenome: Genome): DenseLayer[] {
    const sizes = [STOCK_TOPOLOGY.inputSize, ...STOCK_TOPOLOGY.hiddenLayers, STOCK_TOPOLOGY.outputSize];
    const layers: DenseLayer[] = [];
    let cursor = 0;
    for (let layerIndex = 1; layerIndex < sizes.length; layerIndex += 1) {
        const biases = networkGenome.slice(cursor, cursor + sizes[layerIndex]);
        cursor += sizes[layerIndex];
        const weights = Array.from({length: sizes[layerIndex]}, () => {
            const row = networkGenome.slice(cursor, cursor + sizes[layerIndex - 1]);
            cursor += sizes[layerIndex - 1];
            return row;
        });
        layers.push({biases, weights});
    }
    return layers;
}

function createLayerLines(names: string[], inputs: string[], layer: DenseLayer, useTanh = true): string[] {
    return names.map((name, nodeIndex) => {
        const terms = inputs.map((input, inputIndex) => `${input} * ${formatNumber(layer.weights[nodeIndex][inputIndex])}`);
        const expression = `${formatNumber(layer.biases[nodeIndex])} + ${terms.join(" + ")}`;
        return `${name} = ${useTanh ? `tanh(${expression})` : expression}`;
    });
}

function formatNumber(value: number): string {
    return Number(value.toFixed(8)).toString();
}
