/**
 * Stock 交易模擬：genome = 指標參數 +（可選）NN 權重，喺歷史 K 線上做 long / short。
 *
 * 流程概覽：
 * 1. evaluateStockGenome — GA 評分：test（尾 40%）為主 + train（60%）輔助
 * 2. createTradingReplay — UI 曲線：train→test 連續模擬
 * 3. 成交：T 日收市決策 → T+1 開盤成交（overnight 用舊倉、intraday 用新倉）− 換手成本
 *
 * 兩種決策模式：
 * - useNetwork=true：NN 睇 18 維特徵 → 買／持／沽空（持倉 sticky + margin 先轉倉）
 * - useNetwork=false：規則投票（SMA / MACD / RSI / Williams）
 *
 * 倉位 ∈ {-1, 0, 1}：開局可空倉；買入 → 做多，沽空 → 做空（唔再平倉落現金）。
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
export const STOCK_OUTPUT_LABELS = ["買入", "持有", "沽空"] as const;

/** 模擬起始資金 */
const STARTING_EQUITY = 10_000;
/**
 * 單邊換手成本（|Δposition| × 0.15%）。
 * 略高過理想 0.1%，逼策略計埋滑價／佣金，少啲「紙上 thrash」。
 */
const TRANSACTION_COST = 0.0015;
/** 訓練段比例（資料切分）；其餘為 test（fitness 主軸） */
const TRAIN_RATIO = 0.6;
/** fitness 權重：test 回報優先於 train（打擊過擬合） */
const FITNESS_TEST_WEIGHT = 0.55;
const FITNESS_TRAIN_WEIGHT = 0.3;
const FITNESS_ROBUST_WEIGHT = 0.15;
/** 每條價格序列最多 cache 幾套指標參數（Float64Array，約 ~1MB／全歷史） */
const MAX_INDICATOR_CACHE = 16;
/**
 * NN 權重 L2 衰減係數。要細過 return scale，唔係 GA 寧願 cash 都好過 invest。
 * 夠細先可以令強 buy/hold bias seed 喺 mutation 下生存耐啲。
 */
const WEIGHT_L2_PENALTY = 0.28;
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
 */
const THRASH_LINEAR = 48;
const THRASH_QUADRATIC = 140;

/** 兩段切法：train（輔助）/ test（主分） */
export function getStockSplitIndices(length: number): {trainEnd: number} {
    const trainEnd = Math.max(2, Math.min(length - 2, Math.floor(length * TRAIN_RATIO)));
    return {trainEnd};
}

function segmentAt(index: number, trainEnd: number): "train" | "test" {
    return index < trainEnd ? "train" : "test";
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
    /** 平均 |position| 曝險 ∈ [0,1]；全程空倉 ≈ 0（做多／做空都算入市） */
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
 * GA fitness：test（尾 40%）為主，train（前 60%）輔助。
 *
 * 計分結構：
 *   0.55 * testScore    — 測試段回報／超額（主軸；可轉移先贏）
 * + 0.30 * trainScore   — 訓練段超額（輔助；唔俾淨背 test 噪音）
 * + 0.15 * robustScore  — train 前後半 soft floor（半段穩健）
 * − L2(network)         — 淨 NN 模式先罰權重
 *
 * thrash 淨靠成本 + scoreSegment 換手罰，唔硬鎖持倉日數。
 */
export function evaluateStockGenome(genome: Genome, points: MarketDataPoint[], useNetwork = true): number {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const {trainEnd} = getStockSplitIndices(columns.length);
    // 訓練太短直接大負分（test 至少留少少）
    if (trainEnd < 80 || columns.length - trainEnd < 20) {
        return -1_000;
    }

    const decide = createPositionDecider(columns, parameters, networkGenome, useNetwork);
    const mid = Math.floor(trainEnd / 2);
    const {full, firstHalf, secondHalf, test} = simulateFitnessWalk(decide, columns, trainEnd, mid);

    const trainScore = scoreSegment(full, columns, 0, trainEnd);
    const halfA = scoreSegment(firstHalf, columns, 0, mid + 1);
    const halfB = scoreSegment(secondHalf, columns, mid, trainEnd);
    const testScore = scoreSegment(test, columns, trainEnd, columns.length);
    // soft robust：輕 floor，弱半段唔會完全抹走強 train return
    const robustScore = 0.35 * Math.min(halfA, halfB) + 0.65 * ((halfA + halfB) / 2);
    // rule 模式完全唔用 NN 尾，罰權重淨係加 noise
    const regularization = useNetwork ? WEIGHT_L2_PENALTY * meanSquare(networkGenome) : 0;
    return testScore * FITNESS_TEST_WEIGHT + trainScore * FITNESS_TRAIN_WEIGHT + robustScore * FITNESS_ROBUST_WEIGHT - regularization;
}

/**
 * 單段 score：excess return 優先，再加參與度／絕對回報，扣 thrash 同「永遠空倉」。
 *
 * 調校目標：可轉移 > 紙上 thrash；
 * - 永遠空倉有硬罰
 * - thrash 用線性+二次罰（取代 min-hold hard lock；配合 0.15% 成本）
 * - 牛市要有曝險（做多）先贏；熊市沽空可貢獻 absolute return
 *
 * 對齊 buy-and-hold 時 excess 部分 ≈ 0。
 */
function scoreSegment(metrics: SegmentMetrics, columns: IndicatorColumns, start: number, end: number): number {
    const first = columns.close[start] || 1;
    const last = columns.close[end - 1] || first;
    const years = Math.max((end - start) / 252, 0.5);
    const benchmarkReturn = last / first - 1;
    // log excess：策略相對大盤；clamp -0.99 防 log 爆
    const logExcess = Math.log(1 + Math.max(metrics.totalReturn, -0.99)) - Math.log(1 + Math.max(benchmarkReturn, -0.99));
    const annualizedExcess = logExcess / years;
    // 幾乎唔入市：牛市窗口下差過弱 long
    const cashPenalty = metrics.meanExposure < 0.08 ? 40 : metrics.meanExposure < 0.2 ? 12 : 0;
    // 換手：輕轉倉扣少；日日 full flip 二次項重罰
    const t = metrics.meanTurnover;
    const thrashPenalty = t * THRASH_LINEAR + t * t * THRASH_QUADRATIC;

    return annualizedExcess * 220 + logExcess * 45 + metrics.totalReturn * 55 + metrics.meanExposure * 24 + metrics.sharpe * 5 - metrics.maxDrawdown * 12 - thrashPenalty - cashPenalty;
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

/**
 * UI 用完整 replay：train→test 一條連續 walk。
 * 段回報用 equity 邊界計；test 績效亦入 evaluateStockGenome。
 */
export function createTradingReplay(genome: Genome, points: MarketDataPoint[], useNetwork = true): TradingReplay {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const {trainEnd} = getStockSplitIndices(columns.length);
    const decide = createPositionDecider(columns, parameters, networkGenome, useNetwork);
    const full = simulateColumnReplay(decide, columns, points, 0, columns.length, 0);
    const equityAt = (index: number) => full.equityCurve[Math.min(Math.max(0, index), full.equityCurve.length - 1)] ?? STARTING_EQUITY;
    const trainEquity = equityAt(trainEnd - 1);
    const endEquity = equityAt(columns.length - 1);
    const firstClose = columns.close[0] || 1;
    const lastClose = columns.close[columns.length - 1] || firstClose;

    const tradingPoints: TradingPoint[] = new Array(columns.length);
    for (let index = 0; index < columns.length; index += 1) {
        const close = columns.close[index];
        tradingPoints[index] = {
            date: points[columns.warmup + index].date,
            close,
            strategy: full.equityCurve[index] ?? endEquity,
            benchmark: STARTING_EQUITY * (close / firstClose),
            segment: segmentAt(index, trainEnd),
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
        trainReturn: trainEquity / STARTING_EQUITY - 1,
        testReturn: trainEquity > 0 ? endEquity / trainEquity - 1 : 0,
        benchmarkReturn: columns.length > 1 ? lastClose / firstClose - 1 : 0,
        sharpe: full.sharpe,
        maxDrawdown: full.maxDrawdown,
        optimizedParameters: parameters,
    };
}

/**
 * 將 NN 輸出映射成 long(1)、flat(0) 或 short(-1)。
 * output[0]=買、[1]=持、[2]=沽空。
 *
 * 持倉 sticky（堵 thrash）：
 * - 已做多：buy 當「留守」；淨得沽空明確贏 max(hold,buy)+margin 先翻空
 * - 已做空：sell 當「留守」；淨得買入明確贏 max(hold,sell)+margin 先翻多
 * - 空倉：買／沽都要明確贏對方通道 + margin；兩邊齊過取較強
 * 近平手維持原倉，減少 buy≈sell 時日日翻。
 */
export function decidePositionFromNetwork(output: number[], position: number, margin = ACTION_MARGIN): number {
    const buy = output[0] ?? 0;
    const hold = output[1] ?? 0;
    const sell = output[2] ?? 0;
    if (position > 0) {
        const stay = Math.max(hold, buy);
        if (sell >= stay + margin) {
            return -1;
        }
        return 1;
    }
    if (position < 0) {
        const stay = Math.max(hold, sell);
        if (buy >= stay + margin) {
            return 1;
        }
        return -1;
    }
    const buyEdge = buy >= Math.max(hold, sell) + margin;
    const sellEdge = sell >= Math.max(hold, buy) + margin;
    if (buyEdge && sellEdge) {
        return buy >= sell ? 1 : -1;
    }
    if (buyEdge) {
        return 1;
    }
    if (sellEdge) {
        return -1;
    }
    return 0;
}

/**
 * 由 trade log 還原「date 之前」嘅倉位（-1|0|1）。
 * stock lab scrub NN activation 預覽時用。
 */
export function positionBeforeDate(trades: TradeMarker[], date: string): number {
    let position = 0;
    for (const trade of trades) {
        if (trade.date > date) {
            break;
        }
        position = trade.action === "buy" ? 1 : -1;
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
 *   13    目前持倉（-1 沽空 / 0 空倉 / 1 做多）
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
 * 沽空：RSI 或 Williams 超買；但上升趨勢（快 SMA > 慢）要兩個 exit 一齊先翻空，
 *     避免單指標「超買」喺強牛市提早翻空。
 * 超買訊號將長倉翻做空（唔再平倉落現金）；空倉亦可直接開空。
 */
export function decidePositionFromRules(columns: IndicatorColumns, index: number, position: number, parameters: OptimizedIndicatorParameters): number {
    const hasRsiExit = columns.rsi[index] >= parameters.rsiSellThreshold;
    const hasWilliamsExit = columns.williamsR[index] >= parameters.williamsSellThreshold;
    if (position >= 0 && (hasRsiExit || hasWilliamsExit)) {
        const uptrend = columns.smaFast[index] > columns.smaSlow[index];
        if (uptrend) {
            // 強趨勢：兩個動量 exit 齊先翻空
            if (hasRsiExit && hasWilliamsExit) {
                return -1;
            }
        } else if (hasRsiExit || hasWilliamsExit) {
            return -1;
        }
    }

    const votes = [
        columns.smaFast[index] > columns.smaSlow[index] ? 1 : 0,
        columns.macd[index] > columns.macdSignal[index] ? 1 : 0,
        columns.rsi[index] <= parameters.rsiBuyThreshold ? 1 : 0,
        columns.williamsR[index] <= parameters.williamsBuyThreshold ? 1 : 0,
    ];

    const yes = votes.reduce((sum, vote) => sum + vote, 0);
    const needed = Math.max(1, Math.ceil(votes.length / 2));
    if (yes >= needed) {
        return 1;
    }
    // 唔夠票：維持現倉（做空會一直持有到買票夠數先翻多）
    return position;
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
 * 一次 walk：train 全段（前後半同時累積）+ test 段。
 * 段回報各自用 STARTING_EQUITY 起計（分數只睇段內 return）；倉位一路由 train 傳落 test。
 */
function simulateFitnessWalk(
    decide: PositionDecider,
    columns: IndicatorColumns,
    trainEnd: number,
    mid: number
): {full: SegmentMetrics; firstHalf: SegmentMetrics; secondHalf: SegmentMetrics; test: SegmentMetrics} {
    let position = 0;
    let trainEndingPosition = 0;

    // train full
    let equity = STARTING_EQUITY;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    let exposureSum = 0;
    let turnoverSum = 0;

    // train first half
    let halfAEquity = STARTING_EQUITY;
    let halfAPeak = halfAEquity;
    let halfAMaxDrawdown = 0;
    let halfAReturnSum = 0;
    let halfAReturnSqSum = 0;
    let halfAReturnCount = 0;
    let halfAExposureSum = 0;
    let halfATurnoverSum = 0;
    let halfAEndingPosition = 0;
    const halfEnd = mid + 1;

    // train second half（連續路徑；equity 獨立起計方便 scoreSegment）
    let halfBEquity = STARTING_EQUITY;
    let halfBPeak = halfBEquity;
    let halfBMaxDrawdown = 0;
    let halfBReturnSum = 0;
    let halfBReturnSqSum = 0;
    let halfBReturnCount = 0;
    let halfBExposureSum = 0;
    let halfBTurnoverSum = 0;
    let halfBEndingPosition = 0;

    // test（倉位接 train；equity 獨立起計方便 scoreSegment）
    let testEquity = STARTING_EQUITY;
    let testPeak = testEquity;
    let testMaxDrawdown = 0;
    let testReturnSum = 0;
    let testReturnSqSum = 0;
    let testReturnCount = 0;
    let testExposureSum = 0;
    let testTurnoverSum = 0;
    let testEndingPosition = 0;

    for (let index = 1; index < columns.length; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const {dailyReturn, turnover} = applyNextOpenDay(columns, previous, index, position, targetPosition);

        if (index < trainEnd) {
            equity *= Math.max(0.01, 1 + dailyReturn);
            returnSum += dailyReturn;
            returnSqSum += dailyReturn * dailyReturn;
            returnCount += 1;
            exposureSum += Math.abs(targetPosition);
            turnoverSum += turnover;
            peak = Math.max(peak, equity);
            maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);
            trainEndingPosition = targetPosition;

            if (index < halfEnd) {
                halfAEquity *= Math.max(0.01, 1 + dailyReturn);
                halfAReturnSum += dailyReturn;
                halfAReturnSqSum += dailyReturn * dailyReturn;
                halfAReturnCount += 1;
                halfAExposureSum += Math.abs(targetPosition);
                halfATurnoverSum += turnover;
                halfAPeak = Math.max(halfAPeak, halfAEquity);
                halfAMaxDrawdown = Math.max(halfAMaxDrawdown, halfAPeak > 0 ? (halfAPeak - halfAEquity) / halfAPeak : 0);
                halfAEndingPosition = targetPosition;
            } else {
                halfBEquity *= Math.max(0.01, 1 + dailyReturn);
                halfBReturnSum += dailyReturn;
                halfBReturnSqSum += dailyReturn * dailyReturn;
                halfBReturnCount += 1;
                halfBExposureSum += Math.abs(targetPosition);
                halfBTurnoverSum += turnover;
                halfBPeak = Math.max(halfBPeak, halfBEquity);
                halfBMaxDrawdown = Math.max(halfBMaxDrawdown, halfBPeak > 0 ? (halfBPeak - halfBEquity) / halfBPeak : 0);
                halfBEndingPosition = targetPosition;
            }
        } else {
            testEquity *= Math.max(0.01, 1 + dailyReturn);
            testReturnSum += dailyReturn;
            testReturnSqSum += dailyReturn * dailyReturn;
            testReturnCount += 1;
            testExposureSum += Math.abs(targetPosition);
            testTurnoverSum += turnover;
            testPeak = Math.max(testPeak, testEquity);
            testMaxDrawdown = Math.max(testMaxDrawdown, testPeak > 0 ? (testPeak - testEquity) / testPeak : 0);
            testEndingPosition = targetPosition;
        }

        position = targetPosition;
    }

    return {
        full: {
            totalReturn: equity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
            maxDrawdown,
            endingPosition: trainEndingPosition,
            meanExposure: returnCount > 0 ? exposureSum / returnCount : 0,
            meanTurnover: returnCount > 0 ? turnoverSum / returnCount : 0,
        },
        firstHalf: {
            totalReturn: halfAEquity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(halfAReturnSum, halfAReturnSqSum, halfAReturnCount),
            maxDrawdown: halfAMaxDrawdown,
            endingPosition: halfAEndingPosition,
            meanExposure: halfAReturnCount > 0 ? halfAExposureSum / halfAReturnCount : 0,
            meanTurnover: halfAReturnCount > 0 ? halfATurnoverSum / halfAReturnCount : 0,
        },
        secondHalf: {
            totalReturn: halfBEquity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(halfBReturnSum, halfBReturnSqSum, halfBReturnCount),
            maxDrawdown: halfBMaxDrawdown,
            endingPosition: halfBEndingPosition,
            meanExposure: halfBReturnCount > 0 ? halfBExposureSum / halfBReturnCount : 0,
            meanTurnover: halfBReturnCount > 0 ? halfBTurnoverSum / halfBReturnCount : 0,
        },
        test: {
            totalReturn: testEquity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(testReturnSum, testReturnSqSum, testReturnCount),
            maxDrawdown: testMaxDrawdown,
            endingPosition: testEndingPosition,
            meanExposure: testReturnCount > 0 ? testExposureSum / testReturnCount : 0,
            meanTurnover: testReturnCount > 0 ? testTurnoverSum / testReturnCount : 0,
        },
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
