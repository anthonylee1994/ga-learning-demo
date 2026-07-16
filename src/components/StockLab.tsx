import React from "react";
import {Button, Spinner, Switch} from "@heroui/react";
import {CandlestickChart, Dices, FileDown, TriangleAlert} from "lucide-react";
import {Brush, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import {useEvolutionDemo} from "../hooks/useEvolutionDemo";
import type {GAConfig, Genome, MarketDataPoint, MarketDataResponse, TopicId, TradingPoint, TradingReplay} from "../lib/types";
import {createFutuPythonScript} from "../domains/stock/futuScript";
import {createPineScript} from "../domains/stock/pineScript";
import {
    buildNetworkFeatures,
    createTradingReplay,
    evaluateStockGenome,
    getIndicatorColumns,
    getStockSplitIndices,
    positionBeforeDate,
    STOCK_INPUT_LABELS,
    STOCK_OUTPUT_LABELS,
    STOCK_TOPOLOGY,
} from "../domains/stock/simulation";
import {decodeStockGenome, describeStockNetwork, STOCK_GENE_COUNT, STOCK_HEAD_GENE_COUNT, STOCK_NETWORK_GENE_COUNT, STOCK_PARAMETER_GENE_COUNT} from "../domains/stock/strategyGenome";
import {ApplicationPanel} from "./ApplicationPanel";
import {DemoControls} from "./DemoControls";
import {FitnessChart} from "./FitnessChart";
import {GenomeTransfer} from "./GenomeTransfer";
import {Metrics} from "./Metrics";
import {NetworkPanel} from "./NetworkPanel";
import {DemoShell} from "./SnakeLab";
import {StockPlaybackCanvas, type StockPlaybackDay} from "./StockPlaybackCanvas";

type StockOptimizer = "ga" | "montecarlo";
type StockTopic = Extract<TopicId, "stock" | "stock-mc">;

const GA_DEFAULT_CONFIG: GAConfig = {
    // ~310 genes (indicator head + NN tail) need a real population to search;
    // 36 was mostly elite-neighborhood random walk. Stock sim is cheap enough per genome.
    populationSize: 150,
    // Base rates; stock worker multiplies indicator genes ~3× and NN genes ~0.35×.
    mutationRate: 0.12,
    mutationScale: 0.22,
    eliteRate: 0.08,
    seed: Math.round(Math.random() * 1_000_000),
    // Max speed ⇒ 0ms inter-generation delay (see workerRuntime scheduleNext).
    speed: 5,
    // 預設規則模式：搜尋空間細、較易轉移；要神經演化再開。
    useNeuralNetwork: false,
};

/** 蒙地卡羅：較大批次、較高冠軍附近局部遊走比例。 */
const MC_DEFAULT_CONFIG: GAConfig = {
    populationSize: 40,
    mutationRate: 0.35,
    mutationScale: 0.28,
    eliteRate: 0,
    seed: Math.round(Math.random() * 1_000_000),
    speed: 5,
    useNeuralNetwork: false,
};

type IndicatorView = "price" | "momentum" | "macd" | "risk" | "newHigh" | "newLow";

export const StockLab = React.memo(() => <StockLabView optimizer="ga" />);

/** 同一交易 lab，參數搜尋改用蒙地卡羅隨機抽樣。 */
export const StockMonteCarloLab = React.memo(() => <StockLabView optimizer="montecarlo" />);

const StockLabView = React.memo(({optimizer}: {optimizer: StockOptimizer}) => {
    const isMonteCarlo = optimizer === "montecarlo";
    const topic: StockTopic = isMonteCarlo ? "stock-mc" : "stock";
    const defaultConfig = isMonteCarlo ? MC_DEFAULT_CONFIG : GA_DEFAULT_CONFIG;
    const [tickerInput, setTickerInput] = React.useState("QQQ");
    const [marketData, setMarketData] = React.useState<MarketDataResponse | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [fetchError, setFetchError] = React.useState<string | null>(null);
    const [indicatorView, setIndicatorView] = React.useState<IndicatorView>("price");
    const [transferMessage, setTransferMessage] = React.useState<{type: "status" | "error"; text: string} | null>(null);
    const requestIdRef = React.useRef(0);

    // Pass points only when loaded — useEvolutionDemo resets the worker on every data ref change.
    const trainingData = React.useMemo<MarketDataPoint[] | undefined>(() => {
        if (!marketData?.points.length) {
            return undefined;
        }
        return marketData.points;
    }, [marketData]);

    const demo = useEvolutionDemo<MarketDataPoint[], TradingReplay>({
        topic,
        createWorker: () =>
            isMonteCarlo
                ? new Worker(new URL("../workers/stockMonteCarlo.worker.ts", import.meta.url), {type: "module"})
                : new Worker(new URL("../workers/stock.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: {
            ...defaultConfig,
            seed: Math.round(Math.random() * 1_000_000),
        },
        data: trainingData,
    });

    const load = (symbol: string) => {
        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        setLoading(true);
        setFetchError(null);
        loadMarketData(symbol)
            .then(payload => {
                if (requestId !== requestIdRef.current) {
                    return;
                }
                setMarketData(payload);
                setTickerInput(payload.symbol);
            })
            .catch((error: unknown) => {
                if (requestId === requestIdRef.current) {
                    setFetchError(error instanceof Error ? error.message : "下載失敗。");
                }
            })
            .finally(() => {
                if (requestId === requestIdRef.current) {
                    setLoading(false);
                }
            });
    };

    React.useEffect(() => {
        load("QQQ");
        return () => {
            requestIdRef.current += 1;
        };
    }, []);

    const handleImportGenome = (genome: Genome) => {
        if (!marketData?.points.length) {
            setTransferMessage({type: "error", text: "未有市場數據，無法匯入基因體。"});
            return;
        }
        const nnOn = demo.config.useNeuralNetwork !== false;
        const nextReplay = createTradingReplay(genome, marketData.points, nnOn);
        const fitness = evaluateStockGenome(genome, marketData.points, nnOn);
        demo.loadChampion({genome, replay: nextReplay, fitness});
    };

    const useNetwork = demo.config.useNeuralNetwork !== false;
    const replay = demo.champion?.replay;
    /**
     * Parameters live in the genome; full chart replay is throttled for memory.
     * Decode every generation so the champion panel always tracks the live best genome.
     */
    const decoded = React.useMemo(() => {
        const genome = demo.champion?.genome;
        if (!genome) {
            return null;
        }
        try {
            return decodeStockGenome(genome);
        } catch {
            return null;
        }
    }, [demo.champion?.genome]);
    const parameters = decoded?.parameters ?? replay?.optimizedParameters;
    /**
     * Chart labels must track the *replay* parameters, not the live-decoded genome.
     * Live parameters are a new object every generation and used to blow MarketChart's
     * React.memo — Recharts then redraws ~15y of series several times/sec until the tab OOMs.
     */
    const chartParameters = replay?.optimizedParameters;
    const rawChartData = React.useMemo(() => marketData?.points.map(point => ({date: point.date, close: point.close})) ?? [], [marketData]);
    const chartData = React.useMemo(() => {
        if (!replay) {
            return rawChartData;
        }
        const trades = new Map(replay.trades.map(trade => [trade.date, trade]));
        return replay.points.map(point => {
            const trade = trades.get(point.date);
            return {
                ...point,
                buy: trade?.action === "buy" ? trade.price : null,
                sell: trade?.action === "sell" ? trade.price : null,
            };
        });
    }, [rawChartData, replay]);
    const [marketRange, setMarketRange] = React.useState({startIndex: 0, endIndex: 0});
    React.useEffect(() => {
        if (chartData.length) {
            setMarketRange({startIndex: Math.max(0, chartData.length - 504), endIndex: chartData.length - 1});
        }
    }, [chartData.length, marketData?.symbol]);
    const onMarketRangeChange = React.useCallback((range: {startIndex: number; endIndex: number}) => {
        setMarketRange(current => (current.startIndex === range.startIndex && current.endIndex === range.endIndex ? current : range));
    }, []);
    const validateDate = replay?.points.find(point => point.segment === "validate")?.date;
    const testDate = replay?.points.find(point => point.segment === "test")?.date;
    const metricsExtra = React.useMemo(
        () => [
            {label: "訓練回報", value: replay ? formatPercent(replay.trainReturn) : "—"},
            {label: "驗證回報", value: replay ? formatPercent(replay.validateReturn) : "—"},
            {label: "測試回報", value: replay ? formatPercent(replay.testReturn) : "—"},
            {label: "買入持有", value: replay ? formatPercent(replay.benchmarkReturn) : "—"},
            {label: "最大回撤", value: replay ? formatPercent(-replay.maxDrawdown) : "—"},
        ],
        [replay]
    );

    /** Index into warm-up-aligned indicator columns for NN activation (driven by playback). */
    const [previewIndex, setPreviewIndex] = React.useState(0);
    const [liveDay, setLiveDay] = React.useState<StockPlaybackDay | null>(null);
    const networkGenome = decoded?.networkGenome ?? null;

    const handleDayChange = (day: StockPlaybackDay | null) => {
        setLiveDay(day);
        if (day) {
            setPreviewIndex(day.index);
        }
    };

    React.useEffect(() => {
        if (!demo.champion) {
            setLiveDay(null);
        }
    }, [demo.champion]);

    const networkPreview = React.useMemo(() => {
        if (!useNetwork || !demo.champion?.genome || !marketData?.points.length || !decoded) {
            return null;
        }
        try {
            const columns = getIndicatorColumns(marketData.points, decoded.parameters);
            if (columns.length === 0) {
                return null;
            }
            const index = Math.min(Math.max(0, previewIndex), columns.length - 1);
            const date = marketData.points[columns.warmup + index]?.date ?? "";
            const position = liveDay?.index === index ? liveDay.position : replay ? positionBeforeDate(replay.trades, date) : 0;
            const input = buildNetworkFeatures(columns, index, position, decoded.parameters);
            const {trainEnd, validateEnd} = getStockSplitIndices(columns.length);
            const segmentLabel = index < trainEnd ? "訓練" : index < validateEnd ? "驗證" : "測試";
            return {
                input,
                index,
                date,
                maxIndex: columns.length - 1,
                segment: segmentLabel,
            };
        } catch {
            return null;
        }
    }, [useNetwork, demo.champion?.genome, marketData, decoded, previewIndex, replay, liveDay]);

    return (
        <DemoShell
            accent={isMonteCarlo ? "stock-mc" : "stock"}
            description={
                isMonteCarlo
                    ? "以蒙地卡羅隨機抽樣搜尋交易策略：65% 訓練 + 20% 驗證入分，尾 15% 純測試。次日開盤成交、0.15% 成本。預設規則模式（可開神經網絡）。"
                    : "以遺傳演算法進化指標週期／門檻（可開神經網絡決策頭）。65% 訓練 + 20% 驗證入分，尾 15% 純測試。次日開盤成交、0.15% 成本。"
            }
            icon={isMonteCarlo ? <Dices size={20} strokeWidth={1.5} /> : <CandlestickChart size={20} strokeWidth={1.5} />}
            title={isMonteCarlo ? "股票交易 · 蒙地卡羅" : "股票交易 · 神經演化"}
        >
            <div className="stock-toolbar">
                <label>
                    <span>代號</span>
                    <input
                        aria-label="股票代號"
                        className="ticker-input"
                        maxLength={15}
                        onChange={event => setTickerInput(event.target.value.toUpperCase())}
                        onKeyDown={event => {
                            if (event.key === "Enter") {
                                load(tickerInput);
                            }
                        }}
                        value={tickerInput}
                    />
                </label>
                <span className="disclaimer">只作教育用途，並非投資建議。</span>
            </div>
            {fetchError ? (
                <div className="fetch-error">
                    <TriangleAlert size={18} strokeWidth={1.5} />
                    <span>{fetchError}</span>
                    <Button onPress={() => load(tickerInput)} size="sm" variant="tertiary">
                        重試
                    </Button>
                </div>
            ) : null}
            <div className="workspace-grid">
                <main className="demo-main">
                    <Metrics extra={metricsExtra} generationLabel={isMonteCarlo ? "批次" : "世代"} stats={demo.stats} />
                    <div className="simulation-stage stock-stage">
                        <div className="stage-overlay">
                            <span>{marketData?.symbol ?? "QQQ"} · 逐日重播</span>
                            <span>{replay ? (demo.status === "running" ? (isMonteCarlo ? "冠軍循環重播 · 抽樣中" : "冠軍循環重播 · 進化中") : "冠軍循環重播") : "未有冠軍"}</span>
                        </div>
                        <StockPlaybackCanvas loop onDayChange={handleDayChange} playing={Boolean(replay)} replay={replay} restartKey={demo.showcaseEpoch} speed={demo.config.speed} />
                    </div>
                    {useNetwork ? (
                        <NetworkPanel
                            genome={networkGenome}
                            input={networkPreview?.input ?? null}
                            inputLabels={STOCK_INPUT_LABELS}
                            outputLabels={STOCK_OUTPUT_LABELS}
                            subtitle="只顯示決策頭（週期基因另見下方參數表）。節點亮度跟住下方逐日重播；亦可拖滑桿手動 scrub。"
                            title="股票決策頭"
                            topology={STOCK_TOPOLOGY}
                        >
                            {networkPreview ? (
                                <label className="network-scrub control-field">
                                    <span className="control-label">
                                        <span>網絡預覽日</span>
                                        <strong className="font-mono text-xs">
                                            {networkPreview.date || "—"} · {networkPreview.segment}
                                            {liveDay?.trade ? ` · ${liveDay.trade.action === "buy" ? "買" : "賣"}` : ""}
                                        </strong>
                                    </span>
                                    <input
                                        aria-label="網絡預覽日"
                                        className="range-input"
                                        max={networkPreview.maxIndex}
                                        min={0}
                                        onChange={event => setPreviewIndex(Number(event.target.value))}
                                        step={1}
                                        type="range"
                                        value={Math.min(previewIndex, networkPreview.maxIndex)}
                                    />
                                </label>
                            ) : null}
                        </NetworkPanel>
                    ) : (
                        <section className="network-panel">
                            <div className="panel-heading">
                                <div>
                                    <p className="eyebrow">神經演化大腦</p>
                                    <h3>股票決策頭</h3>
                                </div>
                            </div>
                            <div className="empty-chart network-empty">規則模式開啟中 — 決策用移動平均線 / MACD / RSI / 威廉指標投票，決策頭權重未使用。</div>
                        </section>
                    )}
                    {parameters ? (
                        <section className="optimized-panel">
                            <div className="panel-heading">
                                <div>
                                    <p className="eyebrow">冠軍基因體 · 指標參數</p>
                                    <h3>最佳指標參數</h3>
                                </div>
                                <div className="export-actions">
                                    <Button onPress={() => downloadPineScript(demo.champion!.genome, marketData?.symbol ?? "QQQ", useNetwork)} size="sm" variant="secondary">
                                        <FileDown size={15} strokeWidth={1.5} />
                                        匯出 Pine Script
                                    </Button>
                                    <Button onPress={() => downloadFutuPython(demo.champion!.genome, marketData?.symbol ?? "QQQ", useNetwork)} size="sm" variant="secondary">
                                        <FileDown size={15} strokeWidth={1.5} />
                                        匯出富途 Python
                                    </Button>
                                </div>
                            </div>
                            <div className="parameter-grid">
                                <ParameterValue label="移動平均線" value={`${parameters.smaFastPeriod} / ${parameters.smaSlowPeriod}`} />
                                <ParameterValue label="RSI" value={`${parameters.rsiPeriod} 日 · 買 ≤ ${parameters.rsiBuyThreshold} · 賣 ≥ ${parameters.rsiSellThreshold}`} />
                                <ParameterValue label="保力加通道" value={`${parameters.bollingerPeriod} / ${parameters.bollingerMultiplier.toFixed(2)}σ`} />
                                <ParameterValue label="ROC 週期" value={`${parameters.rocPeriod}`} />
                                <ParameterValue label="威廉指標" value={`${parameters.williamsPeriod} 日 · 買 ≤ ${parameters.williamsBuyThreshold} · 賣 ≥ ${parameters.williamsSellThreshold}`} />
                                <ParameterValue label="MACD" value={`${parameters.macdFastPeriod} / ${parameters.macdSlowPeriod} / ${parameters.macdSignalPeriod}`} />
                                <ParameterValue label="波動率" value={`${parameters.volatilityPeriod}`} />
                                <ParameterValue label="成交量" value={`${parameters.volumeZScorePeriod}`} />
                                <ParameterValue label="N日新高" value={`${parameters.newHighPeriod}`} />
                                <ParameterValue label="N日新低" value={`${parameters.newLowPeriod}`} />
                                <ParameterValue label="Head 基因" value={`${STOCK_HEAD_GENE_COUNT}（週期 / 門檻，突變 ×3）`} />
                                <ParameterValue label="決策頭" value={useNetwork ? describeStockNetwork() : "SMA / MACD / RSI / 威廉票多數；升勢雙過熱先賣，否則單一過熱賣"} />
                                <ParameterValue label="網絡基因" value={useNetwork ? `${STOCK_NETWORK_GENE_COUNT}（突變 ×0.35）` : `${STOCK_NETWORK_GENE_COUNT}（規則模式未使用）`} />
                            </div>
                        </section>
                    ) : null}
                    <section className="market-panel">
                        <div className="panel-heading stock-heading">
                            <div>
                                <p className="eyebrow">{marketData?.symbol ?? "QQQ"} · 日線 · 全段檢視</p>
                                <h3>市場與交易訊號</h3>
                            </div>
                            <select aria-label="技術指標" onChange={event => setIndicatorView(event.target.value as IndicatorView)} value={indicatorView}>
                                <option value="price">移動平均線 + 保力加通道</option>
                                <option value="momentum">RSI + 威廉指標 + ROC</option>
                                <option value="macd">MACD</option>
                                <option value="risk">波動率 + 成交量</option>
                                <option value="newHigh">N 日高</option>
                                <option value="newLow">N 日低</option>
                            </select>
                        </div>
                        <div className="chart-height-lg">
                            {loading ? (
                                <div className="loading-state">
                                    <Spinner color="accent" />
                                    <span>下載 Yahoo Finance 數據...</span>
                                </div>
                            ) : chartData.length ? (
                                <MarketChart
                                    data={chartData}
                                    indicatorView={indicatorView}
                                    marketRange={marketRange}
                                    onRangeChange={onMarketRangeChange}
                                    parameters={chartParameters}
                                    replay={replay}
                                    testDate={testDate}
                                    validateDate={validateDate}
                                />
                            ) : (
                                <div className="empty-chart">未有市場數據。</div>
                            )}
                        </div>
                    </section>
                    <section className="market-panel">
                        <div className="panel-heading">
                            <div>
                                <p className="eyebrow">樣本外檢視</p>
                                <h3>策略 vs 買入持有</h3>
                            </div>
                        </div>
                        <div className="chart-height-md">
                            {replay ? <EquityChart points={replay.points} testDate={testDate} validateDate={validateDate} /> : <div className="empty-chart">訓練出冠軍後會顯示權益曲線。</div>}
                        </div>
                    </section>
                    <FitnessChart eyebrow={isMonteCarlo ? "搜尋訊號" : "演化訊號"} history={demo.history} title={isMonteCarlo ? "批次適應度趨勢" : "適應度趨勢"} />
                    <ApplicationPanel
                        eyebrow={isMonteCarlo ? "蒙地卡羅對應" : "GA 對應"}
                        fitness="train 50% + validate 30% + robust 12% − 過擬合罰 − L2；超額回報主軸；次日開盤成交；0.15% 成本；最少持／空各 8 日；train 靚 validate 仆會扣分；尾 15% test 永不入分"
                        genome={
                            isMonteCarlo
                                ? `${STOCK_PARAMETER_GENE_COUNT} 週期/門檻 + ${STOCK_NETWORK_GENE_COUNT} 決策頭；每批混合全域隨機抽樣 + 冠軍附近局部遊走（局部比例 = 滑桿）；開局有接近買入持有等種子`
                                : `${STOCK_PARAMETER_GENE_COUNT} 週期/門檻（突變 ×3）+ ${STOCK_NETWORK_GENE_COUNT} 決策頭權重（×0.55；${describeStockNetwork()}）；開局有接近買入持有等種子`
                        }
                        genomeLabel={isMonteCarlo ? "參數向量" : "基因體"}
                        inputs="22 維特徵：高低開收 + 全部指標常開（含 N 日新高／新低）+ 持倉狀態。"
                        outputs={useNetwork ? "薄隱藏層取最大 → 買 / 持 / 賣；搜尋主力喺週期 / 門檻" : "SMA / MACD / RSI / 威廉 多數票買入；升勢要 RSI+威廉齊過熱先賣，否則單一過熱賣"}
                        termination={isMonteCarlo ? "65% 訓練 + 20% 驗證入選擇；尾 15% 純測試唔入分；每批保留全域最佳" : "65% 訓練 + 20% 驗證入選擇；尾 15% 純測試唔入分；移民只重抽 head（參數）"}
                        title={isMonteCarlo ? "點樣套用蒙地卡羅優化" : "點樣套用遺傳演算法"}
                    />
                </main>
                <aside className="demo-sidebar">
                    <DemoControls
                        demo={demo}
                        disabled={!trainingData || loading}
                        populationMax={240}
                        labels={
                            isMonteCarlo
                                ? {
                                      title: "蒙地卡羅控制",
                                      description: "每批隨機樣本 + 冠軍附近局部遊走",
                                      populationSize: "每批樣本數",
                                      mutationRate: "局部探索比例",
                                  }
                                : undefined
                        }
                    >
                        <Switch isSelected={useNetwork} onChange={checked => demo.setConfig(current => ({...current, useNeuralNetwork: checked}))}>
                            <Switch.Content>
                                <Switch.Control>
                                    <Switch.Thumb />
                                </Switch.Control>
                                使用神經網絡
                            </Switch.Content>
                        </Switch>
                        <GenomeTransfer
                            disabled={!trainingData || loading}
                            fitness={demo.champion?.fitness}
                            geneCount={STOCK_GENE_COUNT}
                            genome={demo.champion?.genome}
                            onImport={handleImportGenome}
                            onMessage={setTransferMessage}
                            score={replay ? Math.round(((replay.trainReturn + replay.validateReturn) / 2) * 1000) / 10 : undefined}
                            topic={topic}
                            topology={STOCK_TOPOLOGY}
                        />
                        {transferMessage ? <p className={transferMessage.type === "error" ? "error-message" : "status-message"}>{transferMessage.text}</p> : null}
                    </DemoControls>
                </aside>
            </div>
        </DemoShell>
    );
});

type MarketChartDatum = {
    date: string;
    close: number;
    buy?: number | null;
    sell?: number | null;
    smaFast?: number;
    smaSlow?: number;
    rsi?: number;
    williamsR?: number;
    roc?: number;
    macd?: number;
    macdSignal?: number;
    bollingerUpper?: number;
    bollingerLower?: number;
    volatility?: number;
    volumeZScore?: number;
    nDayHigh?: number;
    newHighRatio?: number;
    nDayLow?: number;
    newLowRatio?: number;
};

interface MarketChartProps {
    data: MarketChartDatum[];
    indicatorView: IndicatorView;
    parameters: TradingReplay["optimizedParameters"] | undefined;
    replay: TradingReplay | undefined;
    validateDate: string | undefined;
    testDate: string | undefined;
    marketRange: {startIndex: number; endIndex: number};
    onRangeChange: (range: {startIndex: number; endIndex: number}) => void;
}

/**
 * Heavy 15y market chart. Memoized so the ~8/sec generation ticks (which only touch
 * stats/history) do not force recharts to redraw thousands of points every frame —
 * series data only re-renders when the champion replay actually refreshes.
 */
const MarketChart = React.memo<MarketChartProps>(
    ({data, indicatorView, parameters, replay, validateDate, testDate, marketRange, onRangeChange}) => {
        const hasReplay = replay !== undefined;
        const handleBrushChange = (range: {startIndex?: number; endIndex?: number}) => {
            onRangeChange({startIndex: range.startIndex ?? 0, endIndex: range.endIndex ?? data.length - 1});
        };
        return (
            <ResponsiveContainer height="100%" width="100%">
                <LineChart data={data} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
                    <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" minTickGap={70} stroke="#747b86" tick={{fontSize: 12}} tickLine={false} />
                    <YAxis domain={["auto", "auto"]} stroke="#747b86" tick={{fontSize: 12}} tickFormatter={formatChartValue} tickLine={false} width={58} yAxisId="price" />
                    {(indicatorView === "momentum" || indicatorView === "macd" || indicatorView === "risk" || indicatorView === "newHigh" || indicatorView === "newLow") && hasReplay ? (
                        <YAxis domain={["auto", "auto"]} orientation="right" stroke="#747b86" tick={{fontSize: 12}} tickFormatter={formatChartValue} tickLine={false} width={48} yAxisId="indicator" />
                    ) : null}
                    <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} formatter={formatTooltipValue} />
                    <Legend wrapperStyle={{fontSize: 12}} />
                    <Line dataKey="close" dot={false} isAnimationActive={false} name="收市價" stroke="#dfe3e8" strokeWidth={1.5} type="monotone" yAxisId="price" />
                    {hasReplay ? (
                        <React.Fragment>
                            <Line connectNulls={false} dataKey="buy" dot={BUY_DOT} isAnimationActive={false} name="買入" stroke="none" yAxisId="price" />
                            <Line connectNulls={false} dataKey="sell" dot={SELL_DOT} isAnimationActive={false} name="賣出" stroke="none" yAxisId="price" />
                        </React.Fragment>
                    ) : null}
                    {indicatorView === "price" && hasReplay && parameters ? (
                        <React.Fragment>
                            <Line dataKey="smaFast" dot={false} isAnimationActive={false} name={`SMA${parameters.smaFastPeriod}`} stroke="#e7b955" strokeWidth={1} yAxisId="price" />
                            <Line dataKey="smaSlow" dot={false} isAnimationActive={false} name={`SMA${parameters.smaSlowPeriod}`} stroke="#5da6d9" strokeWidth={1} yAxisId="price" />
                            <Line dataKey="bollingerUpper" dot={false} isAnimationActive={false} name="布林上軌" stroke="#6f7782" strokeDasharray="4 4" strokeWidth={1} yAxisId="price" />
                            <Line dataKey="bollingerLower" dot={false} isAnimationActive={false} name="布林下軌" stroke="#6f7782" strokeDasharray="4 4" strokeWidth={1} yAxisId="price" />
                        </React.Fragment>
                    ) : null}
                    {indicatorView === "newHigh" && hasReplay && parameters ? (
                        <React.Fragment>
                            <Line dataKey="nDayHigh" dot={false} isAnimationActive={false} name={`${parameters.newHighPeriod} 日高`} stroke="#d48bd4" strokeWidth={1.5} yAxisId="price" />
                            <Line dataKey="newHighRatio" dot={false} isAnimationActive={false} name="收市 / N 日高" stroke="#63c6a1" strokeWidth={1} yAxisId="indicator" />
                        </React.Fragment>
                    ) : null}
                    {indicatorView === "newLow" && hasReplay && parameters ? (
                        <React.Fragment>
                            <Line dataKey="nDayLow" dot={false} isAnimationActive={false} name={`${parameters.newLowPeriod} 日低`} stroke="#5da6d9" strokeWidth={1.5} yAxisId="price" />
                            <Line dataKey="newLowRatio" dot={false} isAnimationActive={false} name="N 日低 / 收市" stroke="#e7b955" strokeWidth={1} yAxisId="indicator" />
                        </React.Fragment>
                    ) : null}
                    {indicatorView === "momentum" && hasReplay ? (
                        <React.Fragment>
                            <Line dataKey="rsi" dot={false} isAnimationActive={false} name="RSI" stroke="#63c6a1" strokeWidth={1} yAxisId="indicator" />
                            <Line dataKey="williamsR" dot={false} isAnimationActive={false} name="威廉指標" stroke="#e36f5b" strokeWidth={1} yAxisId="indicator" />
                            <Line dataKey="roc" dot={false} isAnimationActive={false} name="ROC" stroke="#b38bd4" strokeWidth={1} yAxisId="indicator" />
                            {parameters ? (
                                <React.Fragment>
                                    <ReferenceLine
                                        label={{value: `RSI 買 ${parameters.rsiBuyThreshold}`, fill: "#63c6a1", fontSize: 10, position: "insideTopRight"}}
                                        stroke="#63c6a1"
                                        strokeDasharray="4 4"
                                        strokeOpacity={0.55}
                                        y={parameters.rsiBuyThreshold}
                                        yAxisId="indicator"
                                    />
                                    <ReferenceLine
                                        label={{value: `RSI 賣 ${parameters.rsiSellThreshold}`, fill: "#63c6a1", fontSize: 10, position: "insideTopRight"}}
                                        stroke="#63c6a1"
                                        strokeDasharray="4 4"
                                        strokeOpacity={0.55}
                                        y={parameters.rsiSellThreshold}
                                        yAxisId="indicator"
                                    />
                                    <ReferenceLine
                                        label={{value: `W%R 買 ${parameters.williamsBuyThreshold}`, fill: "#e36f5b", fontSize: 10, position: "insideTopRight"}}
                                        stroke="#e36f5b"
                                        strokeDasharray="2 4"
                                        strokeOpacity={0.55}
                                        y={parameters.williamsBuyThreshold}
                                        yAxisId="indicator"
                                    />
                                    <ReferenceLine
                                        label={{value: `W%R 賣 ${parameters.williamsSellThreshold}`, fill: "#e36f5b", fontSize: 10, position: "insideTopRight"}}
                                        stroke="#e36f5b"
                                        strokeDasharray="2 4"
                                        strokeOpacity={0.55}
                                        y={parameters.williamsSellThreshold}
                                        yAxisId="indicator"
                                    />
                                </React.Fragment>
                            ) : null}
                        </React.Fragment>
                    ) : null}
                    {indicatorView === "macd" && hasReplay ? (
                        <React.Fragment>
                            <Line dataKey="macd" dot={false} isAnimationActive={false} name="MACD" stroke="#63c6a1" strokeWidth={1} yAxisId="indicator" />
                            <Line dataKey="macdSignal" dot={false} isAnimationActive={false} name="訊號線" stroke="#e7b955" strokeWidth={1} yAxisId="indicator" />
                        </React.Fragment>
                    ) : null}
                    {indicatorView === "risk" && hasReplay ? (
                        <React.Fragment>
                            <Line dataKey="volatility" dot={false} isAnimationActive={false} name="波動率" stroke="#e36f5b" strokeWidth={1} yAxisId="indicator" />
                            <Line dataKey="volumeZScore" dot={false} isAnimationActive={false} name="成交量" stroke="#5da6d9" strokeWidth={1} yAxisId="indicator" />
                        </React.Fragment>
                    ) : null}
                    {validateDate ? <ReferenceLine label={{value: "驗證", fill: "#5da6d9", fontSize: 12}} stroke="#5da6d9" strokeDasharray="4 4" x={validateDate} /> : null}
                    {testDate ? <ReferenceLine label={{value: "測試", fill: "#e7b955", fontSize: 12}} stroke="#e7b955" strokeDasharray="4 4" x={testDate} /> : null}
                    <Brush
                        ariaLabel="市場日期縮放範圍"
                        className="market-zoom-brush"
                        dataKey="date"
                        endIndex={marketRange.endIndex}
                        fill="#0d1115"
                        gap={Math.max(1, Math.floor(data.length / 1000))}
                        height={28}
                        onChange={handleBrushChange}
                        startIndex={marketRange.startIndex}
                        stroke="#49515b"
                        tickFormatter={formatBrushDate}
                        travellerWidth={10}
                    />
                </LineChart>
            </ResponsiveContainer>
        );
    },
    (prev, next) =>
        prev.data === next.data &&
        prev.indicatorView === next.indicatorView &&
        prev.parameters === next.parameters &&
        prev.replay === next.replay &&
        prev.validateDate === next.validateDate &&
        prev.testDate === next.testDate &&
        prev.marketRange.startIndex === next.marketRange.startIndex &&
        prev.marketRange.endIndex === next.marketRange.endIndex &&
        prev.onRangeChange === next.onRangeChange
);

const BUY_DOT = {fill: "#58d68d", r: 5, strokeWidth: 0} as const;
const SELL_DOT = {fill: "#e36f5b", r: 5, strokeWidth: 0} as const;

interface EquityChartProps {
    points: TradingPoint[];
    validateDate: string | undefined;
    testDate: string | undefined;
}

/** Out-of-sample equity curve. Memoized for the same reason as MarketChart. */
const EquityChart = React.memo<EquityChartProps>(
    ({points, validateDate, testDate}) => (
        <ResponsiveContainer height="100%" width="100%">
            <LineChart data={points} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
                <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" minTickGap={70} stroke="#747b86" tick={{fontSize: 12}} tickLine={false} />
                <YAxis stroke="#747b86" tick={{fontSize: 12}} tickFormatter={formatMoneyAxis} tickLine={false} width={84} />
                <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} formatter={formatMoneyTooltip} />
                <Line dataKey="strategy" dot={false} isAnimationActive={false} name="策略" stroke="#58d68d" strokeWidth={2} type="monotone" />
                <Line dataKey="benchmark" dot={false} isAnimationActive={false} name="買入持有" stroke="#e7b955" strokeWidth={1.5} type="monotone" />
                {validateDate ? <ReferenceLine label={{value: "驗證", fill: "#5da6d9", fontSize: 11}} stroke="#5da6d9" strokeDasharray="4 4" x={validateDate} /> : null}
                {testDate ? <ReferenceLine label={{value: "測試", fill: "#e7b955", fontSize: 11}} stroke="#e7b955" strokeDasharray="4 4" x={testDate} /> : null}
            </LineChart>
        </ResponsiveContainer>
    ),
    (prev, next) => prev.points === next.points && prev.validateDate === next.validateDate && prev.testDate === next.testDate
);

const ParameterValue = React.memo(({label, value}: {label: string; value: string}) => (
    <div>
        <span>{label}</span>
        <strong>{value}</strong>
    </div>
));

async function loadMarketData(symbol: string): Promise<MarketDataResponse> {
    const normalized = symbol.trim().toUpperCase() || "QQQ";
    const response = await fetch(`/api/market-data?symbol=${encodeURIComponent(normalized)}&range=15y&interval=1d`);
    const payload = (await response.json()) as MarketDataResponse | {error: string};
    if (!response.ok || "error" in payload) {
        throw new Error("error" in payload ? payload.error : "下載市場數據失敗。");
    }
    return payload;
}

function formatPercent(value: number): string {
    return new Intl.NumberFormat("zh-HK", {style: "percent", maximumFractionDigits: 1}).format(value);
}

function formatBrushDate(value: string): string {
    return value.slice(0, 7);
}

function formatChartValue(value: number): string {
    return Number.isFinite(value) ? value.toFixed(2) : "";
}

function formatTooltipValue(value: unknown): string {
    if (typeof value === "number" && Number.isFinite(value)) {
        return value.toFixed(2);
    }
    if (value == null) {
        return "—";
    }
    return String(value);
}

const MONEY_FORMAT = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
});

function formatMoney(value: number): string {
    return Number.isFinite(value) ? MONEY_FORMAT.format(value) : "";
}

function formatMoneyAxis(value: number): string {
    return formatMoney(value);
}

function formatMoneyTooltip(value: unknown): string {
    if (typeof value === "number" && Number.isFinite(value)) {
        return formatMoney(value);
    }
    if (value == null) {
        return "—";
    }
    return String(value);
}

function downloadTextFile(filename: string, content: string, mime = "text/plain;charset=utf-8"): void {
    const blob = new Blob([content], {type: mime});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
}

function downloadPineScript(genome: number[], symbol: string, useNetwork: boolean): void {
    downloadTextFile(`${symbol.toLowerCase()}-evolab-strategy.pine`, createPineScript(genome, symbol, useNetwork));
}

function downloadFutuPython(genome: number[], symbol: string, useNetwork: boolean): void {
    downloadTextFile(`${symbol.toLowerCase()}-evolab-strategy-futu.py`, createFutuPythonScript(genome, useNetwork));
}
