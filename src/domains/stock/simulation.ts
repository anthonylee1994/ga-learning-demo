/**
 * Stock 交易模擬：genome = 指標參數 +（可選）NN 權重，喺歷史 K 線上做 long / flat。
 *
 * 流程概覽：
 * 1. evaluateStockGenome — GA 評分：前 80% development 內做 expanding walk-forward validation
 * 2. createTradingReplay — UI 曲線：訓練中只跑 development；暫停先揭尾 20% holdout
 * 3. 成交：T 日收市決策 → T+1 開盤成交（overnight 用舊倉、intraday 用新倉）− 換手成本
 *
 * 兩種決策模式：
 * - useNetwork=true：NN 睇 18 維特徵 → 買／持／平倉（持倉 sticky + margin 先轉倉）
 * - useNetwork=false：規則投票（SMA / MACD / RSI / Williams）
 *
 * 倉位 ∈ {0, 1}：做多 或 空倉（賣出 = 平倉落現金，唔做沽空）。
 * 唔再硬 lock 最少持倉日數；thrash 靠 0.15% 成本 + fitness 換手罰（meanTurnover）打壓。
 */
import {createForwardRunner} from "../../lib/neuralNetwork";
import type {Genome, MarketDataPoint, OptimizedIndicatorParameters, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import type {IndicatorColumns} from "./indicators";
import {calculateIndicatorColumns, columnsToSnapshots} from "./indicators";
import {decodeStockGenome, STOCK_TOPOLOGY} from "./strategyGenome";

export {STOCK_TOPOLOGY} from "./strategyGenome";

/** 對應 18 維 NN 輸入（畀 UI activation 標籤；唔含開高低收） */
export const STOCK_INPUT_LABELS = [
    "快線",
    "慢線",
    "快慢線差",
    "威廉指標",
    "ROC",
    "RSI",
    "MACD",
    "MACD訊",
    "BB通道",
    "波動率",
    "成交量",
    "N日新高",
    "N日新低",
    "持倉",
    "RSI買距",
    "RSI賣距",
    "W%買距",
    "W%賣距",
] as const;

/** 對應 3 維輸出 */
export const STOCK_OUTPUT_LABELS = ["買入", "持有", "平倉"] as const;

/** 模擬起始資金 */
const STARTING_EQUITY = 10_000;
/**
 * 單邊換手成本（|Δposition| × 0.15%）。
 * 略高過理想 0.1%，逼策略計埋滑價／佣金，少啲「紙上 thrash」。
 */
const TRANSACTION_COST = 0.0015;
/** 前 80% 用作 development；最後 20% 完全唔入 fitness。 */
const DEVELOPMENT_RATIO = 0.8;
const WALK_FORWARD_INITIAL_TRAIN_DAYS = 3 * 252;
const WALK_FORWARD_VALIDATION_DAYS = 252;
const MIN_WALK_FORWARD_TRAIN_DAYS = 80;
const MIN_VALIDATION_DAYS = 20;
const MIN_HOLDOUT_DAYS = 20;
/** fold 聚合：典型表現為主，同時壓低「平均靚、但有一段爆倉」策略。 */
const FOLD_MEDIAN_WEIGHT = 0.5;
const FOLD_MEAN_WEIGHT = 0.3;
const FOLD_WORST_WEIGHT = 0.2;
/** 每條價格序列最多 cache 幾套指標參數（Float64Array，約 ~1MB／15 年） */
const MAX_INDICATOR_CACHE = 16;
/**
 * NN 權重 L2 衰減係數。要細過 return scale，唔係 GA 寧願 cash 都好過 invest。
 * 夠細先可以令強 buy/hold bias seed 喺 mutation 下生存耐啲。
 */
const WEIGHT_L2_PENALTY = 0.22;
/**
 * NN 模式：轉倉要比「留守」大呢個 tanh 空間 margin。
 * 平手／近平手維持現狀 → 長倉段穩陣啲。
 * 匯出 Pine／Futu 必須同值。
 */
export const STOCK_ACTION_MARGIN = 0.08;

const ACTION_MARGIN = STOCK_ACTION_MARGIN;
/**
 * fitness 換手罰：線性 + 二次。
 * 偶而轉倉（meanTurnover 細）扣少；日日 full flip 二次項爆罰（取代舊 min-hold hard lock）。
 * 略鬆：long↔flat 換手已有 0.15% 成本，唔好再罰死正常進出。
 */
const THRASH_LINEAR = 28;
const THRASH_QUADRATIC = 80;
/** 輸大市（log excess < 0）額外罰，逼冠軍貼住／贏過 buy-and-hold */
const UNDERPERFORM_BENCH_PENALTY = 100;

export interface StockWalkForwardFold {
    trainingEnd: number;
    validationStart: number;
    validationEnd: number;
}

/**
 * Split 先喺原始 K 線定日期，再扣 indicator warm-up 對齊 columns。
 * 因此唔同 genome 揀唔同 lookback，都唔會改變 holdout 開始日期。
 */
export function getStockSplitIndices(length: number, warmup = 0): {developmentEnd: number; holdoutStart: number} {
    const alignedLength = Math.max(0, length - warmup);
    const rawHoldoutStart = Math.floor(length * DEVELOPMENT_RATIO);
    const maximumStart = Math.max(0, alignedLength - 2);
    const minimumStart = Math.min(2, maximumStart);
    const holdoutStart = Math.min(maximumStart, Math.max(minimumStart, rawHoldoutStart - warmup));
    return {developmentEnd: holdoutStart, holdoutStart};
}

/** Expanding train history + chronological validation windows，全部止於 holdout 前。 */
export function getStockWalkForwardFolds(length: number, warmup = 0): StockWalkForwardFold[] {
    const rawDevelopmentEnd = Math.floor(length * DEVELOPMENT_RATIO);
    const minimumRawTrainingEnd = warmup + MIN_WALK_FORWARD_TRAIN_DAYS;
    const latestRawValidationStart = rawDevelopmentEnd - MIN_VALIDATION_DAYS;
    if (minimumRawTrainingEnd > latestRawValidationStart) {
        return [];
    }

    const preferredRawTrainingEnd = warmup + WALK_FORWARD_INITIAL_TRAIN_DAYS;
    const fallbackRawTrainingEnd = Math.max(minimumRawTrainingEnd, Math.floor((warmup + rawDevelopmentEnd) / 2));
    const firstRawValidationStart = Math.min(latestRawValidationStart, preferredRawTrainingEnd <= latestRawValidationStart ? preferredRawTrainingEnd : fallbackRawTrainingEnd);
    const availableDays = rawDevelopmentEnd - firstRawValidationStart;
    const validationDays = Math.min(WALK_FORWARD_VALIDATION_DAYS, Math.max(MIN_VALIDATION_DAYS, Math.floor(availableDays / 4)));
    const folds: StockWalkForwardFold[] = [];

    for (let rawStart = firstRawValidationStart; rawStart < rawDevelopmentEnd; rawStart += validationDays) {
        const rawEnd = Math.min(rawDevelopmentEnd, rawStart + validationDays);
        if (rawEnd - rawStart < MIN_VALIDATION_DAYS && folds.length > 0) {
            folds[folds.length - 1].validationEnd = rawDevelopmentEnd - warmup;
            break;
        }
        folds.push({
            trainingEnd: rawStart - warmup,
            validationStart: rawStart - warmup,
            validationEnd: rawEnd - warmup,
        });
    }
    return folds;
}

function segmentAt(index: number, holdoutStart: number): "development" | "holdout" {
    return index < holdoutStart ? "development" : "holdout";
}

/**
 * 指標欄 cache：key = points 陣列 reference（WeakMap）→ 參數字串 → columns。
 * multi-ticker fitness 會交錯多條序列；用 series reference 做 key 先唔 thrash。
 * WeakMap 令換走嘅 points 可以連 cache 一齊被 GC。
 */
const columnCachesBySeries = new WeakMap<MarketDataPoint[], Map<string, IndicatorColumns>>();

/** 一段 walk 嘅績效摘要（fitness / 曲線共用） */
interface SegmentMetrics {
    totalReturn: number;
    sharpe: number;
    maxDrawdown: number;
    endingPosition: number;
    /** 平均 |position| 曝險 ∈ [0,1]；全程空倉 ≈ 0（做多先算入市） */
    meanExposure: number;
    /** 平均 |Δposition|／bar； thrash 策略偏高（成日 full flip） */
    meanTurnover: number;
}

interface SegmentResult extends SegmentMetrics {
    equityCurve: number[];
    trades: TradeMarker[];
}

/**
 * 攞（或計）指標欄；LRU 最多 MAX_INDICATOR_CACHE 套參數。
 * 同一 points + 同一參數會 hit cache，GA 評多個 genome 時慳大量重算。
 */
export function getIndicatorColumns(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters): IndicatorColumns {
    let seriesCache = columnCachesBySeries.get(points);
    if (!seriesCache) {
        seriesCache = new Map();
        columnCachesBySeries.set(points, seriesCache);
    }
    const key = createIndicatorCacheKey(parameters);
    const cached = seriesCache.get(key);
    if (cached) {
        // 刷新 LRU 次序：delete 再 set = 移到最新
        seriesCache.delete(key);
        seriesCache.set(key, cached);
        return cached;
    }
    const columns = calculateIndicatorColumns(points, parameters);
    if (seriesCache.size >= MAX_INDICATOR_CACHE) {
        const oldest = seriesCache.keys().next().value;
        if (oldest !== undefined) {
            seriesCache.delete(oldest);
        }
    }
    seriesCache.set(key, columns);
    return columns;
}

/** 指標 snapshot 陣列（UI／debug）；底層 columns 一樣走 cache */
export function getIndicatorSnapshots(points: MarketDataPoint[], parameters: OptimizedIndicatorParameters) {
    return columnsToSnapshots(points, getIndicatorColumns(points, parameters));
}

/**
 * GA fitness：只用前 80% development 內嘅 walk-forward validation。
 *
 * 計分結構：
 *   0.50 * fold median  — 典型 validation 表現
 * + 0.30 * fold mean    — 全段平均
 * + 0.20 * worst fold   — 防一段爆倉畀其他升市遮住
 * − L2(network)         — 淨 NN 模式先罰權重
 *
 * 最後 20% holdout 唔行 decision、唔計 score、唔參與揀冠軍。
 */
export function evaluateStockGenome(genome: Genome, points: MarketDataPoint[], useNetwork = true): number {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const {holdoutStart} = getStockSplitIndices(points.length, columns.warmup);
    const folds = getStockWalkForwardFolds(points.length, columns.warmup);
    if (folds.length === 0 || holdoutStart < MIN_WALK_FORWARD_TRAIN_DAYS || columns.length - holdoutStart < MIN_HOLDOUT_DAYS) {
        return -1_000;
    }

    const decide = createPositionDecider(columns, parameters, networkGenome, useNetwork);
    const metrics = simulateWalkForwardValidation(decide, columns, holdoutStart, folds);
    const foldScores = metrics.map((segment, index) => scoreSegment(segment, columns, folds[index].validationStart, folds[index].validationEnd));
    const meanScore = foldScores.reduce((sum, score) => sum + score, 0) / foldScores.length;
    const medianScore = median(foldScores);
    const worstScore = Math.min(...foldScores);
    // rule 模式完全唔用 NN 尾，罰權重淨係加 noise
    const regularization = useNetwork ? WEIGHT_L2_PENALTY * meanSquare(networkGenome) : 0;
    return medianScore * FOLD_MEDIAN_WEIGHT + meanScore * FOLD_MEAN_WEIGHT + worstScore * FOLD_WORST_WEIGHT - regularization;
}

/**
 * 單段 score：以 log 回報 + 相對大盤超額為主，扣 thrash／空倉／輸大市。
 *
 * 調校目標：
 * - 贏／貼 buy-and-hold 先係王道（輸大市有硬罰）
 * - log(1+r) 壓縮極端 train 倍數，唔俾 10,000%+ 假冠軍
 * - 永遠空倉有硬罰；thrash 二次罰
 * - 牛市要有做多曝險
 */
function scoreSegment(metrics: SegmentMetrics, columns: IndicatorColumns, start: number, end: number): number {
    const first = columns.close[Math.max(0, start - 1)] || 1;
    const last = columns.close[end - 1] || first;
    const years = Math.max((end - start) / 252, 0.5);
    const benchmarkReturn = last / first - 1;
    // log 空間：317× → ~5.8，唔再 317 倍線性碾壓
    const logReturn = Math.log(1 + Math.max(metrics.totalReturn, -0.99));
    const logBench = Math.log(1 + Math.max(benchmarkReturn, -0.99));
    const logExcess = logReturn - logBench;
    const annualizedExcess = logExcess / years;
    // 幾乎唔入市：牛市窗口下差過弱 long
    const cashPenalty = metrics.meanExposure < 0.08 ? 50 : metrics.meanExposure < 0.25 ? 18 : 0;
    // 換手：輕轉倉扣少；日日 full flip 二次項重罰
    const t = metrics.meanTurnover;
    const thrashPenalty = t * THRASH_LINEAR + t * t * THRASH_QUADRATIC;
    // 輸大市：罰幅度跟住落後幾多（逼貼住 B&H，唔好淨 train 炫技）
    const underBenchPenalty = logExcess < 0 ? -logExcess * UNDERPERFORM_BENCH_PENALTY : 0;

    return annualizedExcess * 220 + logExcess * 140 + logReturn * 55 + metrics.meanExposure * 30 + metrics.sharpe * 4 - metrics.maxDrawdown * 16 - thrashPenalty - cashPenalty - underBenchPenalty;
}

/** genome 權重均方（L2 用） */
function meanSquare(genome: Genome): number {
    if (genome.length === 0) {
        return 0;
    }
    let sum = 0;
    for (const weight of genome) {
        sum += weight * weight;
    }
    return sum / genome.length;
}

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2 : (sorted[middle] ?? 0);
}

/**
 * UI replay：訓練中 includeHoldout=false，暫停／匯入先用 true 揭封存結果。
 */
export function createTradingReplay(genome: Genome, points: MarketDataPoint[], useNetwork = true, includeHoldout = true): TradingReplay {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const {developmentEnd, holdoutStart} = getStockSplitIndices(points.length, columns.warmup);
    const decide = createPositionDecider(columns, parameters, networkGenome, useNetwork);
    const replayEnd = includeHoldout ? columns.length : developmentEnd;
    const full = simulateColumnReplay(decide, columns, points, 0, replayEnd, 0);
    const equityAt = (index: number) => full.equityCurve[Math.min(Math.max(0, index), full.equityCurve.length - 1)] ?? STARTING_EQUITY;
    const developmentEquity = equityAt(developmentEnd - 1);
    const endEquity = equityAt(replayEnd - 1);
    const firstClose = columns.close[0] || 1;
    const lastClose = columns.close[Math.max(0, replayEnd - 1)] || firstClose;
    const developmentEndClose = columns.close[Math.max(0, developmentEnd - 1)] || firstClose;

    const tradingPoints: TradingPoint[] = new Array(replayEnd);
    for (let index = 0; index < replayEnd; index += 1) {
        const close = columns.close[index];
        tradingPoints[index] = {
            date: points[columns.warmup + index].date,
            close,
            strategy: full.equityCurve[index] ?? endEquity,
            benchmark: STARTING_EQUITY * (close / firstClose),
            segment: segmentAt(index, holdoutStart),
            smaFast: columns.smaFast[index],
            smaSlow: columns.smaSlow[index],
            rsi: columns.rsi[index],
            williamsR: columns.williamsR[index],
            roc: columns.roc[index],
            macd: columns.macd[index],
            macdSignal: columns.macdSignal[index],
            bollingerUpper: columns.bollingerUpper[index],
            bollingerLower: columns.bollingerLower[index],
            volatility: columns.volatility[index],
            volumeZScore: columns.volumeZScore[index],
            nDayHigh: columns.nDayHigh[index],
            newHighRatio: columns.newHighRatio[index],
            nDayLow: columns.nDayLow[index],
            newLowRatio: columns.newLowRatio[index],
        };
    }

    return {
        points: tradingPoints,
        trades: full.trades,
        developmentReturn: developmentEquity / STARTING_EQUITY - 1,
        holdoutReturn: includeHoldout && developmentEquity > 0 ? endEquity / developmentEquity - 1 : null,
        benchmarkReturn: replayEnd > 1 ? lastClose / firstClose - 1 : 0,
        developmentBenchmarkReturn: developmentEndClose / firstClose - 1,
        holdoutBenchmarkReturn: includeHoldout ? lastClose / developmentEndClose - 1 : null,
        sharpe: full.sharpe,
        maxDrawdown: full.maxDrawdown,
        optimizedParameters: parameters,
    };
}

/**
 * 將 NN 輸出映射成 long(1) 或 flat(0)。
 * output[0]=買、[1]=持、[2]=平倉。
 *
 * 持倉 sticky（堵 thrash）：
 * - 已做多：buy 當「留守」；平倉明確贏 max(hold,buy)+margin → 落空倉
 * - 空倉：買明確贏 max(hold,平倉)+margin → 做多（平倉訊號唔開空）
 * 近平手維持原倉。
 */
export function decidePositionFromNetwork(output: number[], position: number, margin = ACTION_MARGIN): number {
    const buy = output[0] ?? 0;
    const hold = output[1] ?? 0;
    const sell = output[2] ?? 0;
    if (position > 0) {
        const stay = Math.max(hold, buy);
        if (sell >= stay + margin) {
            return 0;
        }
        return 1;
    }
    // Flat（或舊 short 當空倉）：淨得買入可開多
    if (buy >= Math.max(hold, sell) + margin) {
        return 1;
    }
    return 0;
}

/**
 * 由 trade log 還原「date 之前」嘅倉位（0|1）。
 * stock lab scrub NN activation 預覽時用。
 */
export function positionBeforeDate(trades: TradeMarker[], date: string): number {
    let position = 0;
    for (const trade of trades) {
        if (trade.date > date) {
            break;
        }
        position = trade.position;
    }
    return position;
}

/**
 * 組 18 維 NN 特徵，大致 clamp 到 [-1, 1]，等 tanh 單元 scale 一致。
 * 可傳入 `out` buffer 重用，避免每 bar new array。
 * 唔再餵開高低收（K 線結構／日回報）；淨指標 + 持倉 + 門檻距離。
 *
 * 特徵分組：
 *   0–2   SMA 快慢／交叉
 *   3–12  動量／波動／量／新高／新低
 *   13    目前持倉（0 空倉 / 1 做多）
 *   14–17 離 genome 解出嘅 RSI／Williams 買賣門檻距離
 */
export function buildNetworkFeatures(
    columns: IndicatorColumns,
    index: number,
    position: number,
    parameters: OptimizedIndicatorParameters,
    out: number[] = new Array(STOCK_TOPOLOGY.inputSize)
): number[] {
    const close = Math.max(columns.close[index], 1e-9);
    const smaFast = Math.max(columns.smaFast[index], 1e-9);
    const smaSlow = Math.max(columns.smaSlow[index], 1e-9);
    out[0] = clamp((close / smaFast - 1) * 10);
    out[1] = clamp((close / smaSlow - 1) * 10);
    out[2] = clamp((smaFast / smaSlow - 1) * 10);
    out[3] = clamp((columns.williamsR[index] + 50) / 50);
    out[4] = clamp(columns.roc[index] * 5);
    out[5] = clamp((columns.rsi[index] - 50) / 50);
    out[6] = clamp((columns.macd[index] / close) * 25);
    out[7] = clamp((columns.macdSignal[index] / close) * 25);
    out[8] = clamp((columns.bollingerPercentB[index] - 0.5) * 2);
    out[9] = clamp(columns.volatility[index] * 5);
    out[10] = clamp(columns.volumeZScore[index] / 3);
    // close / N-day high ≈ 1 係突破；把 ~[0.9, 1.0] 拉開到 roughly [-1, 1]
    out[11] = clamp((columns.newHighRatio[index] - 0.95) * 20);
    // nDayLow / close ≈ 1 係貼近 N 日低（同新高 ratio 對稱）
    out[12] = clamp((columns.newLowRatio[index] - 0.95) * 20);
    out[13] = clamp(position);
    out[14] = clamp((parameters.rsiBuyThreshold - columns.rsi[index]) / 20);
    out[15] = clamp((columns.rsi[index] - parameters.rsiSellThreshold) / 20);
    out[16] = clamp((parameters.williamsBuyThreshold - columns.williamsR[index]) / 25);
    out[17] = clamp((columns.williamsR[index] - parameters.williamsSellThreshold) / 25);
    return out;
}

/**
 * 規則模式：SMA / MACD / RSI / Williams 四票。
 * 買：多數贊成（≥2）→ 做多。
 * 平倉：RSI 或 Williams 超買；上升趨勢要兩個 exit 一齊。
 * 長倉遇平倉訊號 → 落現金；空倉唔開沽空。
 */
export function decidePositionFromRules(columns: IndicatorColumns, index: number, position: number, parameters: OptimizedIndicatorParameters): number {
    const hasRsiExit = columns.rsi[index] >= parameters.rsiSellThreshold;
    const hasWilliamsExit = columns.williamsR[index] >= parameters.williamsSellThreshold;
    const uptrend = columns.smaFast[index] > columns.smaSlow[index];
    const hasExit = uptrend ? hasRsiExit && hasWilliamsExit : hasRsiExit || hasWilliamsExit;

    const votes = [
        columns.smaFast[index] > columns.smaSlow[index] ? 1 : 0,
        columns.macd[index] > columns.macdSignal[index] ? 1 : 0,
        columns.rsi[index] <= parameters.rsiBuyThreshold ? 1 : 0,
        columns.williamsR[index] <= parameters.williamsBuyThreshold ? 1 : 0,
    ];

    const yes = votes.reduce((sum, vote) => sum + vote, 0);
    const needed = Math.max(1, Math.ceil(votes.length / 2));
    const hasBuy = yes >= needed;

    if (position > 0) {
        if (hasExit) {
            return 0;
        }
        return 1;
    }
    if (hasBuy) {
        return 1;
    }
    return 0;
}

type PositionDecider = (index: number, position: number) => number;

/**
 * 工廠：rule 或 NN decider。
 * NN 路徑 decode 權重一次、feature buffer 重用，每 bar 只做 forward。
 */
function createPositionDecider(columns: IndicatorColumns, parameters: OptimizedIndicatorParameters, networkGenome: Genome, useNetwork: boolean): PositionDecider {
    if (!useNetwork) {
        return (index, position) => decidePositionFromRules(columns, index, position, parameters);
    }
    const runNetwork = createForwardRunner(networkGenome, STOCK_TOPOLOGY);
    const features = new Array<number>(STOCK_TOPOLOGY.inputSize);
    return (index, position) => decidePositionFromNetwork(runNetwork(buildNetworkFeatures(columns, index, position, parameters, features)), position);
}

/** 參數 → cache key 字串（只含會影響 columns 嘅 period／multiplier） */
function createIndicatorCacheKey(parameters: OptimizedIndicatorParameters): string {
    return [
        parameters.smaFastPeriod,
        parameters.smaSlowPeriod,
        parameters.williamsPeriod,
        parameters.rocPeriod,
        parameters.rsiPeriod,
        parameters.macdFastPeriod,
        parameters.macdSlowPeriod,
        parameters.macdSignalPeriod,
        parameters.bollingerPeriod,
        parameters.bollingerMultiplier,
        parameters.volatilityPeriod,
        parameters.volumeZScorePeriod,
        parameters.newHighPeriod,
        parameters.newLowPeriod,
    ].join(":");
}

/**
 * 一次由頭行到 development 尾；只喺 validation window 累積 metrics。
 * Holdout index 永遠唔會傳入 decide 或 applyNextOpenDay。
 */
function simulateWalkForwardValidation(decide: PositionDecider, columns: IndicatorColumns, holdoutStart: number, folds: StockWalkForwardFold[]): SegmentMetrics[] {
    let position = 0;
    let foldIndex = 0;
    const accumulators = folds.map(() => createMetricsAccumulator());

    for (let index = 1; index < holdoutStart; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const {dailyReturn, turnover} = applyNextOpenDay(columns, previous, index, position, targetPosition);
        while (foldIndex < folds.length && index >= folds[foldIndex].validationEnd) {
            foldIndex += 1;
        }
        const fold = folds[foldIndex];
        if (fold && index >= fold.validationStart) {
            updateMetricsAccumulator(accumulators[foldIndex], dailyReturn, turnover, targetPosition);
        }
        position = targetPosition;
    }
    return accumulators.map(finishMetricsAccumulator);
}

interface MetricsAccumulator {
    equity: number;
    peak: number;
    maxDrawdown: number;
    returnSum: number;
    returnSqSum: number;
    returnCount: number;
    exposureSum: number;
    turnoverSum: number;
    endingPosition: number;
}

function createMetricsAccumulator(): MetricsAccumulator {
    return {equity: STARTING_EQUITY, peak: STARTING_EQUITY, maxDrawdown: 0, returnSum: 0, returnSqSum: 0, returnCount: 0, exposureSum: 0, turnoverSum: 0, endingPosition: 0};
}

function updateMetricsAccumulator(accumulator: MetricsAccumulator, dailyReturn: number, turnover: number, position: number): void {
    accumulator.equity *= Math.max(0.01, 1 + dailyReturn);
    accumulator.returnSum += dailyReturn;
    accumulator.returnSqSum += dailyReturn * dailyReturn;
    accumulator.returnCount += 1;
    accumulator.exposureSum += Math.abs(position);
    accumulator.turnoverSum += turnover;
    accumulator.peak = Math.max(accumulator.peak, accumulator.equity);
    accumulator.maxDrawdown = Math.max(accumulator.maxDrawdown, accumulator.peak > 0 ? (accumulator.peak - accumulator.equity) / accumulator.peak : 0);
    accumulator.endingPosition = position;
}

function finishMetricsAccumulator(accumulator: MetricsAccumulator): SegmentMetrics {
    return {
        totalReturn: accumulator.equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpeFromMoments(accumulator.returnSum, accumulator.returnSqSum, accumulator.returnCount),
        maxDrawdown: accumulator.maxDrawdown,
        endingPosition: accumulator.endingPosition,
        meanExposure: accumulator.returnCount > 0 ? accumulator.exposureSum / accumulator.returnCount : 0,
        meanTurnover: accumulator.returnCount > 0 ? accumulator.turnoverSum / accumulator.returnCount : 0,
    };
}

/**
 * 同 simulateColumnMetrics，但額外錄 equityCurve 同 trades（UI replay 用）。
 * trade 標記喺成交日（T+1 開盤）嘅 date／open。
 */
function simulateColumnReplay(decide: PositionDecider, columns: IndicatorColumns, points: MarketDataPoint[], start: number, end: number, startingPosition: number): SegmentResult {
    const length = Math.max(0, end - start);
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    let exposureSum = 0;
    let turnoverSum = 0;
    const equityCurve = new Array<number>(length);
    if (length > 0) {
        equityCurve[0] = equity;
    }
    const trades: TradeMarker[] = [];

    for (let index = start + 1; index < end; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const {dailyReturn, turnover} = applyNextOpenDay(columns, previous, index, position, targetPosition);
        equity *= Math.max(0.01, 1 + dailyReturn);
        returnSum += dailyReturn;
        returnSqSum += dailyReturn * dailyReturn;
        returnCount += 1;
        exposureSum += Math.abs(targetPosition);
        turnoverSum += turnover;
        equityCurve[index - start] = equity;
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);

        if (targetPosition !== position) {
            trades.push({
                date: points[columns.warmup + index].date,
                action: targetPosition > position ? "buy" : "sell",
                price: columns.open[index],
                position: targetPosition,
            });
        }
        position = targetPosition;
    }

    return {
        equityCurve,
        trades,
        totalReturn: equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
        maxDrawdown,
        endingPosition: position,
        meanExposure: returnCount > 0 ? exposureSum / returnCount : 0,
        meanTurnover: returnCount > 0 ? turnoverSum / returnCount : 0,
    };
}

/**
 * T 收市訊號 → T+1 開盤成交：
 * - overnight gap：舊倉 × (open_t / close_{t-1} − 1)
 * - 開盤換手成本
 * - intraday：新倉 × (close_t / open_t − 1)
 */
function applyNextOpenDay(columns: IndicatorColumns, previous: number, index: number, position: number, targetPosition: number): {dailyReturn: number; turnover: number} {
    const prevClose = Math.max(columns.close[previous], 1e-9);
    const open = Math.max(columns.open[index], 1e-9);
    const close = Math.max(columns.close[index], 1e-9);
    const overnight = open / prevClose - 1;
    const intraday = close / open - 1;
    const turnover = Math.abs(targetPosition - position);
    const dailyReturn = position * overnight + targetPosition * intraday - turnover * TRANSACTION_COST;
    return {dailyReturn, turnover};
}

/**
 * 年化 Sharpe：用 running moments（mean / variance），× √252。
 * 唔存每日 return 陣列，慳記憶。
 */
function calculateSharpeFromMoments(returnSum: number, returnSqSum: number, returnCount: number): number {
    if (returnCount < 2) {
        return 0;
    }
    const average = returnSum / returnCount;
    const variance = returnSqSum / returnCount - average * average;
    const deviation = Math.sqrt(Math.max(0, variance));
    return deviation > 1e-9 ? (average / deviation) * Math.sqrt(252) : 0;
}

/** 特徵 clamp 到 [-1, 1]；非有限數 → 0 */
function clamp(value: number): number {
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.min(1, Math.max(-1, value));
}
