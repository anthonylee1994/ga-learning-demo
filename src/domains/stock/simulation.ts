/**
 * Stock 交易模擬：genome = 指標參數 +（可選）NN 權重，喺歷史 K 線上做 long / cash。
 *
 * 流程概覽：
 * 1. evaluateStockGenome — GA 評分：train（65%）+ validate（20%）；純 test（15%）永不入分
 * 2. createTradingReplay — UI 曲線：train→validate→test 連續模擬
 * 3. 成交：T 日收市決策 → T+1 開盤成交（overnight 用舊倉、intraday 用新倉）− 換手成本
 *
 * 兩種決策模式：
 * - useNetwork=true：NN 睇 22 維特徵 → 買／持／賣（持倉 sticky + margin 先轉倉）
 * - useNetwork=false：規則投票（SMA / MACD / RSI / Williams）
 *
 * 兩者外層都包 withHoldCooldown（最少持倉／空倉 5 根 K），防止日日 thrash 俾 fee 食晒。
 */
import {createForwardRunner} from "../../lib/neuralNetwork";
import type {Genome, MarketDataPoint, OptimizedIndicatorParameters, TradeMarker, TradingPoint, TradingReplay} from "../../lib/types";
import type {IndicatorColumns} from "./indicators";
import {calculateIndicatorColumns, columnsToSnapshots} from "./indicators";
import {decodeStockGenome, STOCK_TOPOLOGY} from "./strategyGenome";

export {STOCK_TOPOLOGY} from "./strategyGenome";

/** 對應 22 維 NN 輸入（畀 UI activation 標籤） */
export const STOCK_INPUT_LABELS = [
    "開",
    "高",
    "低",
    "收",
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
export const STOCK_OUTPUT_LABELS = ["買入", "持有", "賣出"] as const;

/** 模擬起始資金 */
const STARTING_EQUITY = 10_000;
/**
 * 單邊換手成本（|Δposition| × 0.15%）。
 * 略高過理想 0.1%，逼策略計埋滑價／佣金，少啲「紙上 thrash」。
 */
const TRANSACTION_COST = 0.0015;
/** 訓練段比例（入 fitness 主力） */
const TRAIN_RATIO = 0.65;
/** 驗證段結尾 = TRAIN+VAL；驗證入 fitness，純 test 係之後 */
const VALIDATE_END_RATIO = 0.85;
/** 每條價格序列最多 cache 幾套指標參數（Float64Array，約 ~1MB／全歷史） */
const MAX_INDICATOR_CACHE = 16;
/**
 * NN 權重 L2 衰減係數。要細過 return scale，唔係 GA 寧願 cash 都好過 invest。
 * 夠細先可以令強 buy/hold bias seed 喺 mutation 下生存耐啲。
 */
const WEIGHT_L2_PENALTY = 0.28;
/**
 * 做多至少持幾多 bar 先准賣。
 * 擋住 multi-day thrash（fee bleed）同「持幾日就翻」嘅 noise。
 * 匯出 Pine／Futu 必須同值。
 */
export const STOCK_MIN_BARS_IN_LONG = 5;
/**
 * 沽出後至少 cash 幾多 bar 先准再買。
 * 專門堵「噚日沽、今日又買返」round-trip。
 * 匯出 Pine／Futu 必須同值。
 */
export const STOCK_MIN_BARS_IN_CASH = 5;
/**
 * NN 模式：轉倉要比「留守」大呢個 tanh 空間 margin。
 * 平手／近平手維持現狀 → 長倉段穩陣啲。
 * 匯出 Pine／Futu 必須同值。
 */
export const STOCK_ACTION_MARGIN = 0.08;

const MIN_BARS_IN_LONG = STOCK_MIN_BARS_IN_LONG;
const MIN_BARS_IN_CASH = STOCK_MIN_BARS_IN_CASH;
const ACTION_MARGIN = STOCK_ACTION_MARGIN;

/** 三段切法：train / validate（fitness）/ test（只展示） */
export function getStockSplitIndices(length: number): {trainEnd: number; validateEnd: number} {
    const trainEnd = Math.max(2, Math.floor(length * TRAIN_RATIO));
    const validateEnd = Math.min(length, Math.max(trainEnd + 2, Math.floor(length * VALIDATE_END_RATIO)));
    return {trainEnd, validateEnd};
}

function segmentAt(index: number, trainEnd: number, validateEnd: number): "train" | "validate" | "test" {
    if (index < trainEnd) {
        return "train";
    }
    if (index < validateEnd) {
        return "validate";
    }
    return "test";
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
    /** 平均 long 曝險 ∈ [0,1]；全程 cash ≈ 0 */
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
 * GA fitness：train（65%）+ validate（20%）入分；純 test（尾 15%）永不入收生。
 *
 * 計分結構：
 *   0.50 * trainScore        — 訓練段超額為主
 * + 0.30 * validateScore     — 驗證段 OOS 壓力（逼可轉移）
 * + 0.12 * robustScore       — train 前後半 soft floor
 * − 0.20 * max(0, train−val) — train 靚、validate 仆 → 罰過擬合
 * − L2(network)              — 淨 NN 模式先罰權重
 */
export function evaluateStockGenome(genome: Genome, points: MarketDataPoint[], useNetwork = true): number {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const {trainEnd, validateEnd} = getStockSplitIndices(columns.length);
    // 資料太短（三段都要有料）直接大負分
    if (trainEnd < 80 || validateEnd - trainEnd < 30) {
        return -1_000;
    }

    const baseDecide = createPositionDecider(columns, parameters, networkGenome, useNetwork);
    const mid = Math.floor(trainEnd / 2);
    // 每次 walk 新 cooldown state；後半段唔可以繼承前半嘅 hold 計數
    const {full, firstHalf} = simulateFullAndFirstHalf(withHoldCooldown(baseDecide), columns, trainEnd, mid);
    const secondHalf = simulateColumnMetrics(withHoldCooldown(baseDecide), columns, firstHalf.endingPosition, mid, trainEnd);
    const validateMetrics = simulateColumnMetrics(withHoldCooldown(baseDecide), columns, full.endingPosition, Math.max(0, trainEnd - 1), validateEnd);

    const trainScore = scoreSegment(full, columns, 0, trainEnd);
    const halfA = scoreSegment(firstHalf, columns, 0, mid + 1);
    const halfB = scoreSegment(secondHalf, columns, mid, trainEnd);
    const validateScore = scoreSegment(validateMetrics, columns, Math.max(0, trainEnd - 1), validateEnd);
    // soft robust：輕 floor，弱半段唔會完全抹走強 train return
    const robustScore = 0.35 * Math.min(halfA, halfB) + 0.65 * ((halfA + halfB) / 2);
    // train 勁過 validate 太多 = 溫習卷過擬合
    const overfitPenalty = Math.max(0, trainScore - validateScore) * 0.2;
    // rule 模式完全唔用 NN 尾，罰權重淨係加 noise
    const regularization = useNetwork ? WEIGHT_L2_PENALTY * meanSquare(networkGenome) : 0;
    return trainScore * 0.5 + validateScore * 0.3 + robustScore * 0.12 - overfitPenalty - regularization;
}

/**
 * 單段 score：excess return 優先，再加參與度／絕對回報，扣 thrash 同「永遠 cash」。
 *
 * 調校目標：可轉移 > 紙上 thrash；
 * - 永遠 cash 有硬罰
 * - thrash 罰比舊版重（配合 0.15% 成本）
 * - 牛市要 long participation 先贏
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
    // 日日 flip 有稅；持幾個星期轉倉問題唔大
    const thrashPenalty = metrics.meanTurnover * 28;

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
 * UI 用完整 replay：train→validate→test 一條連續 walk（cooldown 跨段唔重置）。
 * 段回報用 equity 邊界計；純 test 永不入 fitness。
 */
export function createTradingReplay(genome: Genome, points: MarketDataPoint[], useNetwork = true): TradingReplay {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const columns = getIndicatorColumns(points, parameters);
    const {trainEnd, validateEnd} = getStockSplitIndices(columns.length);
    const decide = withHoldCooldown(createPositionDecider(columns, parameters, networkGenome, useNetwork));
    // 全程一轉：避免 split 處「免費解 thrash lock」
    const full = simulateColumnReplay(decide, columns, points, 0, columns.length, 0);
    const equityAt = (index: number) => full.equityCurve[Math.min(Math.max(0, index), full.equityCurve.length - 1)] ?? STARTING_EQUITY;
    const trainEquity = equityAt(trainEnd - 1);
    const validateEquity = equityAt(validateEnd - 1);
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
            segment: segmentAt(index, trainEnd, validateEnd),
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
        validateReturn: trainEquity > 0 ? validateEquity / trainEquity - 1 : 0,
        testReturn: validateEquity > 0 ? endEquity / validateEquity - 1 : 0,
        benchmarkReturn: columns.length > 1 ? lastClose / firstClose - 1 : 0,
        sharpe: full.sharpe,
        maxDrawdown: full.maxDrawdown,
        optimizedParameters: parameters,
    };
}

/**
 * 將 NN 輸出映射成 long(1) 或 cash(0)。
 * output[0]=買、[1]=持、[2]=賣。
 *
 * 持倉 sticky（堵「噚日買完今日又買返」類 thrash）：
 * - 已做多：buy 當「留守」唔係加倉；淨得 sell 明確贏 max(hold,buy)+margin 先沽
 * - 已空倉：sell 當雜訊；淨得 buy 明確贏 max(hold,sell)+margin 先開
 * 近平手維持原倉，減少 buy≈sell 時日日翻。
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
    const stay = Math.max(hold, sell);
    if (buy >= stay + margin) {
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
        position = trade.action === "buy" ? 1 : 0;
    }
    return position;
}

/**
 * 組 22 維 NN 特徵，大致 clamp 到 [-1, 1]，等 tanh 單元 scale 一致。
 * 可傳入 `out` buffer 重用，避免每 bar new array。
 *
 * 特徵分組：
 *   0–3   K 線結構 + 日回報
 *   4–6   SMA 快慢／交叉
 *   7–16  動量／波動／量／新高／新低
 *   17    目前持倉（±1）
 *   18–21 離 genome 解出嘅 RSI／Williams 買賣門檻距離
 */
export function buildNetworkFeatures(
    columns: IndicatorColumns,
    index: number,
    position: number,
    parameters: OptimizedIndicatorParameters,
    out: number[] = new Array(STOCK_TOPOLOGY.inputSize)
): number[] {
    const close = Math.max(columns.close[index], 1e-9);
    const open = columns.open[index];
    const high = columns.high[index];
    const low = columns.low[index];
    const smaFast = Math.max(columns.smaFast[index], 1e-9);
    const smaSlow = Math.max(columns.smaSlow[index], 1e-9);
    // 高低開收：相對收盤嘅結構 + 收盤日回報
    out[0] = clamp((open / close - 1) * 50);
    out[1] = clamp((high / close - 1) * 50);
    out[2] = clamp((low / close - 1) * 50);
    out[3] = clamp(columns.closeReturn[index] * 20);
    out[4] = clamp((close / smaFast - 1) * 10);
    out[5] = clamp((close / smaSlow - 1) * 10);
    out[6] = clamp((smaFast / smaSlow - 1) * 10);
    out[7] = clamp((columns.williamsR[index] + 50) / 50);
    out[8] = clamp(columns.roc[index] * 5);
    out[9] = clamp((columns.rsi[index] - 50) / 50);
    out[10] = clamp((columns.macd[index] / close) * 25);
    out[11] = clamp((columns.macdSignal[index] / close) * 25);
    out[12] = clamp((columns.bollingerPercentB[index] - 0.5) * 2);
    out[13] = clamp(columns.volatility[index] * 5);
    out[14] = clamp(columns.volumeZScore[index] / 3);
    // close / N-day high ≈ 1 係突破；把 ~[0.9, 1.0] 拉開到 roughly [-1, 1]
    out[15] = clamp((columns.newHighRatio[index] - 0.95) * 20);
    // nDayLow / close ≈ 1 係貼近 N 日低（同新高 ratio 對稱）
    out[16] = clamp((columns.newLowRatio[index] - 0.95) * 20);
    out[17] = position > 0 ? 1 : -1;
    out[18] = clamp((parameters.rsiBuyThreshold - columns.rsi[index]) / 20);
    out[19] = clamp((columns.rsi[index] - parameters.rsiSellThreshold) / 20);
    out[20] = clamp((parameters.williamsBuyThreshold - columns.williamsR[index]) / 25);
    out[21] = clamp((columns.williamsR[index] - parameters.williamsSellThreshold) / 25);
    return out;
}

/**
 * 規則模式：SMA / MACD / RSI / Williams 四票。
 * 買：多數贊成（≥2）。
 * 賣：RSI 或 Williams 超買；但上升趨勢（快 SMA > 慢）要兩個 exit 一齊先沽，
 *     避免單指標「超買」喺強牛市提早離場食少 return。
 */
export function decidePositionFromRules(columns: IndicatorColumns, index: number, position: number, parameters: OptimizedIndicatorParameters): number {
    const hasRsiExit = columns.rsi[index] >= parameters.rsiSellThreshold;
    const hasWilliamsExit = columns.williamsR[index] >= parameters.williamsSellThreshold;
    if (position > 0 && (hasRsiExit || hasWilliamsExit)) {
        const uptrend = columns.smaFast[index] > columns.smaSlow[index];
        if (uptrend) {
            // 強趨勢：兩個動量 exit 齊先離場
            if (hasRsiExit && hasWilliamsExit) {
                return 0;
            }
        } else if (hasRsiExit || hasWilliamsExit) {
            return 0;
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
    // 唔夠票：維持現倉（唔會主動沽，沽淨靠上面 exit 邏輯）
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

/**
 * 有狀態 cooldown 包裝（NN + rule 都用）：
 * - long 至少 MIN_BARS_IN_LONG 先准賣
 * - cash 至少 MIN_BARS_IN_CASH 先准再買（沽後冷靜期）
 *
 * 注意：每次 withHoldCooldown(base) 都係新 closure；
 * evaluate 嘅 second half 要 fresh 一份，唔好共用 full walk 嘅計數。
 */
function withHoldCooldown(decide: PositionDecider): PositionDecider {
    let barsInState = 0;
    let lastPosition = -1;
    return (index, position) => {
        if (position !== lastPosition) {
            barsInState = 0;
            lastPosition = position;
        }
        barsInState += 1;
        const raw = decide(index, position);
        if (raw === position) {
            return position;
        }
        // Long → cash：持倉日數未夠
        if (position > 0 && raw === 0 && barsInState < MIN_BARS_IN_LONG) {
            return position;
        }
        // Cash → long：冷靜期未完
        if (position <= 0 && raw === 1 && barsInState < MIN_BARS_IN_CASH) {
            return position;
        }
        return raw;
    };
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
 * 喺 [start, end) 上 walk 一轉，只回 metrics（唔錄曲線／trades）。
 * 成交：T 收市 decide → T+1 開盤 fill；overnight 舊倉 + intraday 新倉 − 成本。
 */
function simulateColumnMetrics(decide: PositionDecider, columns: IndicatorColumns, startingPosition: number, start: number, end: number): SegmentMetrics {
    let equity = STARTING_EQUITY;
    let position = startingPosition;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    let exposureSum = 0;
    let turnoverSum = 0;

    for (let index = start + 1; index < end; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const {dailyReturn, turnover} = applyNextOpenDay(columns, previous, index, position, targetPosition);
        equity *= Math.max(0.01, 1 + dailyReturn); // floor 1% equity，防歸零
        returnSum += dailyReturn;
        returnSqSum += dailyReturn * dailyReturn;
        returnCount += 1;
        exposureSum += targetPosition;
        turnoverSum += turnover;
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);
        position = targetPosition;
    }

    return {
        totalReturn: equity / STARTING_EQUITY - 1,
        sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
        maxDrawdown,
        endingPosition: position,
        meanExposure: returnCount > 0 ? exposureSum / returnCount : 0,
        meanTurnover: returnCount > 0 ? turnoverSum / returnCount : 0,
    };
}

/**
 * 一次 walk 同時累積 full train + first half metrics。
 * 語意等同分別 call simulateColumnMetrics(full) 同 firstHalf，慳一次 loop。
 */
function simulateFullAndFirstHalf(decide: PositionDecider, columns: IndicatorColumns, trainLength: number, mid: number): {full: SegmentMetrics; firstHalf: SegmentMetrics} {
    let equity = STARTING_EQUITY;
    let position = 0;
    let peak = equity;
    let maxDrawdown = 0;
    let returnSum = 0;
    let returnSqSum = 0;
    let returnCount = 0;
    let exposureSum = 0;
    let turnoverSum = 0;

    // 前半段獨立累積（同 full 共用同一條 dailyReturn 路徑）
    let halfEquity = STARTING_EQUITY;
    let halfPeak = halfEquity;
    let halfMaxDrawdown = 0;
    let halfReturnSum = 0;
    let halfReturnSqSum = 0;
    let halfReturnCount = 0;
    let halfExposureSum = 0;
    let halfTurnoverSum = 0;
    let halfEndingPosition = 0;
    const halfEnd = mid + 1;

    for (let index = 1; index < trainLength; index += 1) {
        const previous = index - 1;
        const targetPosition = decide(previous, position);
        const {dailyReturn, turnover} = applyNextOpenDay(columns, previous, index, position, targetPosition);
        equity *= Math.max(0.01, 1 + dailyReturn);
        returnSum += dailyReturn;
        returnSqSum += dailyReturn * dailyReturn;
        returnCount += 1;
        exposureSum += targetPosition;
        turnoverSum += turnover;
        peak = Math.max(peak, equity);
        maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);

        if (index < halfEnd) {
            halfEquity *= Math.max(0.01, 1 + dailyReturn);
            halfReturnSum += dailyReturn;
            halfReturnSqSum += dailyReturn * dailyReturn;
            halfReturnCount += 1;
            halfExposureSum += targetPosition;
            halfTurnoverSum += turnover;
            halfPeak = Math.max(halfPeak, halfEquity);
            halfMaxDrawdown = Math.max(halfMaxDrawdown, halfPeak > 0 ? (halfPeak - halfEquity) / halfPeak : 0);
            halfEndingPosition = targetPosition;
        }

        position = targetPosition;
    }

    return {
        full: {
            totalReturn: equity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(returnSum, returnSqSum, returnCount),
            maxDrawdown,
            endingPosition: position,
            meanExposure: returnCount > 0 ? exposureSum / returnCount : 0,
            meanTurnover: returnCount > 0 ? turnoverSum / returnCount : 0,
        },
        firstHalf: {
            totalReturn: halfEquity / STARTING_EQUITY - 1,
            sharpe: calculateSharpeFromMoments(halfReturnSum, halfReturnSqSum, halfReturnCount),
            maxDrawdown: halfMaxDrawdown,
            endingPosition: halfEndingPosition,
            meanExposure: halfReturnCount > 0 ? halfExposureSum / halfReturnCount : 0,
            meanTurnover: halfReturnCount > 0 ? halfTurnoverSum / halfReturnCount : 0,
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
        exposureSum += targetPosition;
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
