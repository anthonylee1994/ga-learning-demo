import type {Genome} from "../../lib/types";
import {decodeStockGenome, STOCK_TOPOLOGY} from "./strategyGenome";

interface DenseLayer {
    biases: number[];
    weights: number[][];
}

export function createPineScript(genome: Genome, symbol: string, useNetwork = true): string {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const safeSymbol = symbol.replace(/[^A-Z0-9._-]/gi, "").toUpperCase() || "QQQ";
    const decisionLines = useNetwork ? createNetworkDecisionLines(networkGenome) : createRuleDecisionLines();
    const topologyLabel = [STOCK_TOPOLOGY.inputSize, ...STOCK_TOPOLOGY.hiddenLayers, STOCK_TOPOLOGY.outputSize].join(" → ");

    return `//@version=6
strategy("EvoLab ${safeSymbol} Evolved Strategy", overlay=true, initial_capital=1000000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, pyramiding=0, commission_type=strategy.commission.percent, commission_value=0.1, process_orders_on_close=true)

// Optimized by EvoLab genetic algorithm — indicator periods, thresholds, and Brain.js weights.
// Network topology: ${topologyLabel} (tanh on every layer, matching brain.js).
// Validate this strategy on fresh out-of-sample data before considering any use.
smaFastPeriod = ${parameters.smaFastPeriod}
smaSlowPeriod = ${parameters.smaSlowPeriod}
williamsPeriod = ${parameters.williamsPeriod}
williamsBuyThreshold = ${parameters.williamsBuyThreshold}
williamsSellThreshold = ${parameters.williamsSellThreshold}
rocPeriod = ${parameters.rocPeriod}
rsiPeriod = ${parameters.rsiPeriod}
rsiBuyThreshold = ${parameters.rsiBuyThreshold}
rsiSellThreshold = ${parameters.rsiSellThreshold}
macdFastPeriod = ${parameters.macdFastPeriod}
macdSlowPeriod = ${parameters.macdSlowPeriod}
macdSignalPeriod = ${parameters.macdSignalPeriod}
bollingerPeriod = ${parameters.bollingerPeriod}
bollingerMultiplier = ${formatNumber(parameters.bollingerMultiplier)}
volatilityPeriod = ${parameters.volatilityPeriod}
volumeZScorePeriod = ${parameters.volumeZScorePeriod}
newHighPeriod = ${parameters.newHighPeriod}

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
bollingerBasis = ta.sma(close, bollingerPeriod)
bollingerDeviation = ta.stdev(close, bollingerPeriod) * bollingerMultiplier
bollingerUpper = bollingerBasis + bollingerDeviation
bollingerLower = bollingerBasis - bollingerDeviation
bollingerRange = math.max(bollingerUpper - bollingerLower, 0.000000001)
bollingerPercentB = (close - bollingerLower) / bollingerRange
dailyReturn = close / close[1] - 1.0
volatility = ta.stdev(dailyReturn, volatilityPeriod) * math.sqrt(252.0)
volumeAverage = ta.sma(volume, volumeZScorePeriod)
volumeDeviation = ta.stdev(volume, volumeZScorePeriod)
volumeZScore = (volume - volumeAverage) / math.max(volumeDeviation, 0.000000001)
nDayHigh = ta.highest(high, newHighPeriod)
newHighRatio = close / math.max(nDayHigh, 0.000000001)

ready = not na(smaSlow) and not na(williamsR) and not na(roc) and not na(rsi) and not na(macdSignal) and not na(bollingerUpper) and not na(volatility) and not na(volumeZScore) and not na(nDayHigh)
${decisionLines.join("\n")}

if buySignal and strategy.position_size <= 0
    strategy.entry("Long", strategy.long)
if sellSignal and strategy.position_size > 0
    strategy.close("Long")

plot(smaFast, "Optimized SMA Fast", color=color.yellow)
plot(smaSlow, "Optimized SMA Slow", color=color.blue)
plot(nDayHigh, "N-day High", color=color.fuchsia)
upperPlot = plot(bollingerUpper, "Optimized BB Upper", color=color.new(color.gray, 35))
lowerPlot = plot(bollingerLower, "Optimized BB Lower", color=color.new(color.gray, 35))
fill(upperPlot, lowerPlot, color=color.new(color.gray, 92))
plotshape(buySignal and strategy.position_size <= 0, title="Buy", style=shape.triangleup, location=location.belowbar, color=color.lime, size=size.tiny)
plotshape(sellSignal and strategy.position_size > 0, title="Sell", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.tiny)
`;
}

/**
 * Emit full MLP matching STOCK_TOPOLOGY (supports N hidden layers).
 * Every layer uses tanh — same as brain.js `activation: "tanh"`.
 * argMax of tanh(z) == argMax(z), so decisions match either pre/post activation on the last layer;
 * we still emit tanh on the output so values match the in-app network panel.
 */
export function createNetworkDecisionLines(networkGenome: Genome): string[] {
    const layers = decodeLayers(networkGenome);
    const expectedLayers = STOCK_TOPOLOGY.hiddenLayers.length + 1;
    if (layers.length !== expectedLayers) {
        throw new Error(`Pine export expected ${expectedLayers} dense layers, got ${layers.length}`);
    }

    const inputNames = Array.from({length: STOCK_TOPOLOGY.inputSize}, (_, index) => `f${index}`);
    const layerNames: string[][] = [inputNames];
    for (let hiddenIndex = 0; hiddenIndex < STOCK_TOPOLOGY.hiddenLayers.length; hiddenIndex += 1) {
        const width = STOCK_TOPOLOGY.hiddenLayers[hiddenIndex];
        layerNames.push(Array.from({length: width}, (_, index) => `h${hiddenIndex + 1}_${index}`));
    }
    layerNames.push(["outBuy", "outHold", "outSell"]);

    const lines: string[] = [
        `f0 = clamp((open / close - 1.0) * 50.0)`,
        `f1 = clamp((high / close - 1.0) * 50.0)`,
        `f2 = clamp((low / close - 1.0) * 50.0)`,
        `f3 = clamp((close / close[1] - 1.0) * 20.0)`,
        `f4 = clamp((close / smaFast - 1.0) * 10.0)`,
        `f5 = clamp((close / smaSlow - 1.0) * 10.0)`,
        `f6 = clamp((smaFast / smaSlow - 1.0) * 10.0)`,
        `f7 = clamp((williamsR + 50.0) / 50.0)`,
        `f8 = clamp(roc * 5.0)`,
        `f9 = clamp((rsi - 50.0) / 50.0)`,
        `f10 = clamp(macdLine / close * 25.0)`,
        `f11 = clamp(macdSignal / close * 25.0)`,
        `f12 = clamp((bollingerPercentB - 0.5) * 2.0)`,
        `f13 = clamp(volatility * 5.0)`,
        `f14 = clamp(volumeZScore / 3.0)`,
        `f15 = clamp((newHighRatio - 0.95) * 20.0)`,
        "f16 = strategy.position_size > 0 ? 1.0 : -1.0",
        `f17 = clamp((rsiBuyThreshold - rsi) / 20.0)`,
        `f18 = clamp((rsi - rsiSellThreshold) / 20.0)`,
        `f19 = clamp((williamsBuyThreshold - williamsR) / 25.0)`,
        `f20 = clamp((williamsR - williamsSellThreshold) / 25.0)`,
        "",
    ];

    for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
        const inputs = layerNames[layerIndex];
        const outputs = layerNames[layerIndex + 1];
        const layer = layers[layerIndex];
        if (layer.biases.length !== outputs.length) {
            throw new Error(`Layer ${layerIndex} bias count ${layer.biases.length} != names ${outputs.length}`);
        }
        if (layer.weights.some(row => row.length !== inputs.length)) {
            throw new Error(`Layer ${layerIndex} weight row width does not match previous layer`);
        }
        // tanh on every layer, including output (matches brain.js + NeuralNetworkAdapter).
        lines.push(...createLayerLines(outputs, inputs, layer, true));
        lines.push("");
    }

    lines.push("buySignal = ready and outBuy >= outHold and outBuy >= outSell");
    lines.push("sellSignal = ready and outSell > outBuy and outSell > outHold");
    lines.push(`// topology ${[STOCK_TOPOLOGY.inputSize, ...STOCK_TOPOLOGY.hiddenLayers, STOCK_TOPOLOGY.outputSize].join("→")}`);
    return lines;
}

/** Rule mode（唔用 NN）：threshold 買賣規則，同 simulation.ts decidePositionFromRules 保持一致。 */
function createRuleDecisionLines(): string[] {
    return [
        "trendVote = smaFast > smaSlow ? 1 : 0",
        "macdVote = macdLine > macdSignal ? 1 : 0",
        "rsiBuyVote = rsi <= rsiBuyThreshold ? 1 : 0",
        "williamsBuyVote = williamsR <= williamsBuyThreshold ? 1 : 0",
        "buyVotes = trendVote + macdVote + rsiBuyVote + williamsBuyVote",
        "neededVotes = 2",
        "buySignal = ready and buyVotes >= neededVotes",
        "rsiSell = rsi >= rsiSellThreshold",
        "williamsSell = williamsR >= williamsSellThreshold",
        "uptrend = smaFast > smaSlow",
        // Uptrend: need both exits (matches decidePositionFromRules trend gate).
        "sellSignal = ready and (uptrend ? (rsiSell and williamsSell) : (rsiSell or williamsSell))",
    ];
}

/**
 * Decode network genes into dense layers. Layout matches `inspectGenome` / brain.js applyGenome:
 * for each layer: [biases…] then for each node [weights from previous layer…].
 */
export function decodeLayers(networkGenome: Genome): DenseLayer[] {
    const sizes = [STOCK_TOPOLOGY.inputSize, ...STOCK_TOPOLOGY.hiddenLayers, STOCK_TOPOLOGY.outputSize];
    const layers: DenseLayer[] = [];
    let cursor = 0;
    for (let layerIndex = 1; layerIndex < sizes.length; layerIndex += 1) {
        const nodeCount = sizes[layerIndex];
        const prevCount = sizes[layerIndex - 1];
        const biases = networkGenome.slice(cursor, cursor + nodeCount);
        cursor += nodeCount;
        const weights = Array.from({length: nodeCount}, () => {
            const row = networkGenome.slice(cursor, cursor + prevCount);
            cursor += prevCount;
            return row;
        });
        layers.push({biases, weights});
    }
    if (cursor !== networkGenome.length) {
        throw new Error(`Network genome length ${networkGenome.length} left ${networkGenome.length - cursor} unread genes (topology mismatch)`);
    }
    return layers;
}

/**
 * Evaluate the same affine+tanh stack the Pine export emits (for parity tests).
 * Input is the feature vector already built (matches STOCK_TOPOLOGY.inputSize).
 */
export function evaluatePineNetwork(networkGenome: Genome, input: number[]): number[] {
    const layers = decodeLayers(networkGenome);
    let activations = input.slice();
    for (const layer of layers) {
        const next = layer.biases.map((bias, nodeIndex) => {
            let sum = bias;
            const inbound = layer.weights[nodeIndex];
            for (let prev = 0; prev < inbound.length; prev += 1) {
                sum += inbound[prev] * activations[prev];
            }
            return Math.tanh(sum);
        });
        activations = next;
    }
    return activations;
}

function createLayerLines(names: string[], inputs: string[], layer: DenseLayer, useTanh = true): string[] {
    return names.map((name, nodeIndex) => {
        const terms = inputs.map((input, inputIndex) => `${input} * ${formatNumber(layer.weights[nodeIndex][inputIndex])}`);
        const expression = `${formatNumber(layer.biases[nodeIndex])} + ${terms.join(" + ")}`;
        return `${name} = ${useTanh ? `tanh(${expression})` : expression}`;
    });
}

export function formatNumber(value: number): string {
    return Number(value.toFixed(8)).toString();
}
