import type {Genome, IndicatorMaskState} from "../../lib/types";
import {decodeStockGenome, INDICATOR_MASK_DEFS, STOCK_TOPOLOGY} from "./strategyGenome";

interface DenseLayer {
    biases: number[];
    weights: number[][];
}

export function createPineScript(genome: Genome, symbol: string, useNetwork = true): string {
    const {parameters, masks, networkGenome} = decodeStockGenome(genome);
    const safeSymbol = symbol.replace(/[^A-Z0-9._-]/gi, "").toUpperCase() || "QQQ";
    const decisionLines = useNetwork ? createNetworkDecisionLines(networkGenome, masks) : createRuleDecisionLines(masks);
    const maskComments = INDICATOR_MASK_DEFS.map(def => `//   ${def.id}: ${masks[def.id] ? "ON" : "OFF"}`).join("\n");
    const topologyLabel = [STOCK_TOPOLOGY.inputSize, ...STOCK_TOPOLOGY.hiddenLayers, STOCK_TOPOLOGY.outputSize].join(" → ");

    return `//@version=6
strategy("EvoLab ${safeSymbol} Evolved Strategy", overlay=true, initial_capital=1000000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, pyramiding=0, commission_type=strategy.commission.percent, commission_value=0.1, process_orders_on_close=true)

// Optimized by EvoLab genetic algorithm — indicator periods, on/off masks, thresholds, and Brain.js weights.
// Network topology: ${topologyLabel} (tanh on every layer, matching brain.js).
// Validate this strategy on fresh out-of-sample data before considering any use.
// Indicator masks (feature selection):
${maskComments}
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
useSma = ${masks.sma}
useWilliams = ${masks.williams}
useRoc = ${masks.roc}
useRsi = ${masks.rsi}
useMacd = ${masks.macd}
useBollinger = ${masks.bollinger}
useVolatility = ${masks.volatility}
useVolume = ${masks.volume}
useNewHigh = ${masks.newHigh}

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
export function createNetworkDecisionLines(networkGenome: Genome, masks: IndicatorMaskState): string[] {
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
        `f0 = useSma ? clamp((close / smaFast - 1.0) * 10.0) : 0.0`,
        `f1 = useSma ? clamp((close / smaSlow - 1.0) * 10.0) : 0.0`,
        `f2 = useSma ? clamp((smaFast / smaSlow - 1.0) * 10.0) : 0.0`,
        `f3 = useWilliams ? clamp((williamsR + 50.0) / 50.0) : 0.0`,
        `f4 = useRoc ? clamp(roc * 5.0) : 0.0`,
        `f5 = useRsi ? clamp((rsi - 50.0) / 50.0) : 0.0`,
        `f6 = useMacd ? clamp(macdLine / close * 25.0) : 0.0`,
        `f7 = useMacd ? clamp(macdSignal / close * 25.0) : 0.0`,
        `f8 = useBollinger ? clamp((bollingerPercentB - 0.5) * 2.0) : 0.0`,
        `f9 = useVolatility ? clamp(volatility * 5.0) : 0.0`,
        `f10 = useVolume ? clamp(volumeZScore / 3.0) : 0.0`,
        `f11 = useNewHigh ? clamp((newHighRatio - 0.95) * 20.0) : 0.0`,
        "f12 = strategy.position_size > 0 ? 1.0 : -1.0",
        `f13 = useRsi ? clamp((rsiBuyThreshold - rsi) / 20.0) : 0.0`,
        `f14 = useRsi ? clamp((rsi - rsiSellThreshold) / 20.0) : 0.0`,
        `f15 = useWilliams ? clamp((williamsBuyThreshold - williamsR) / 25.0) : 0.0`,
        `f16 = useWilliams ? clamp((williamsR - williamsSellThreshold) / 25.0) : 0.0`,
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
    lines.push(
        `// topology ${[STOCK_TOPOLOGY.inputSize, ...STOCK_TOPOLOGY.hiddenLayers, STOCK_TOPOLOGY.outputSize].join("→")}; masks sma=${masks.sma} williams=${masks.williams} roc=${masks.roc} rsi=${masks.rsi} macd=${masks.macd}`
    );
    return lines;
}

/** Rule mode（唔用 NN）：threshold 買賣規則，同 simulation.ts decidePositionFromRules 保持一致。 */
function createRuleDecisionLines(masks: IndicatorMaskState): string[] {
    return [
        "trendVote = useSma and smaFast > smaSlow ? 1 : 0",
        "macdVote = useMacd and macdLine > macdSignal ? 1 : 0",
        "rsiBuyVote = useRsi and rsi <= rsiBuyThreshold ? 1 : 0",
        "williamsBuyVote = useWilliams and williamsR <= williamsBuyThreshold ? 1 : 0",
        "activeVotes = (useSma ? 1 : 0) + (useMacd ? 1 : 0) + (useRsi ? 1 : 0) + (useWilliams ? 1 : 0)",
        "buyVotes = trendVote + macdVote + rsiBuyVote + williamsBuyVote",
        "neededVotes = math.max(1, int(math.ceil(activeVotes / 2.0)))",
        "buySignal = ready and activeVotes > 0 and buyVotes >= neededVotes",
        "rsiSell = useRsi and rsi >= rsiSellThreshold",
        "williamsSell = useWilliams and williamsR >= williamsSellThreshold",
        "fallbackSell = not useRsi and not useWilliams and activeVotes > 0 and buyVotes < neededVotes",
        "sellSignal = ready and (rsiSell or williamsSell or fallbackSell)",
        `// rule masks: sma=${masks.sma} macd=${masks.macd} rsi=${masks.rsi} williams=${masks.williams}`,
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
 * Input is the 17-d feature vector already built (masks applied).
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

function formatNumber(value: number): string {
    return Number(value.toFixed(8)).toString();
}
