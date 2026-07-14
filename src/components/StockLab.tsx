import React from "react";
import {Button, Spinner, Switch} from "@heroui/react";
import {CandlestickChart, FileDown, TriangleAlert} from "lucide-react";
import {Brush, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import {useEvolutionDemo} from "../hooks/useEvolutionDemo";
import type {GAConfig, Genome, MarketDataResponse, TradingPoint, TradingReplay} from "../lib/types";
import {createPineScript} from "../domains/stock/pineScript";
import {
    ablateIndicatorMasks,
    buildNetworkFeatures,
    createTradingReplay,
    evaluateStockGenome,
    getIndicatorColumns,
    positionBeforeDate,
    STOCK_INPUT_LABELS,
    STOCK_OUTPUT_LABELS,
    STOCK_TOPOLOGY,
} from "../domains/stock/simulation";
import {
    countActiveMasks,
    decodeStockGenome,
    describeStockNetwork,
    INDICATOR_MASK_DEFS,
    STOCK_GENE_COUNT,
    STOCK_HEAD_GENE_COUNT,
    STOCK_MASK_GENE_COUNT,
    STOCK_NETWORK_GENE_COUNT,
    STOCK_PARAMETER_GENE_COUNT,
} from "../domains/stock/strategyGenome";
import {ApplicationPanel} from "./ApplicationPanel";
import {DemoControls} from "./DemoControls";
import {FitnessChart} from "./FitnessChart";
import {GenomeTransfer} from "./GenomeTransfer";
import {Metrics} from "./Metrics";
import {NetworkPanel} from "./NetworkPanel";
import {DemoShell} from "./SnakeLab";
import {StockPlaybackCanvas, type StockPlaybackDay} from "./StockPlaybackCanvas";

const DEFAULT_CONFIG: GAConfig = {
    populationSize: 48,
    // Base rates; stock worker multiplies indicator genes ~3× and NN genes ~0.35×.
    mutationRate: 0.12,
    mutationScale: 0.22,
    eliteRate: 0.08,
    seed: Math.round(Math.random() * 1_000_000),
    speed: 3,
    useNeuralNetwork: true,
};

type IndicatorView = "price" | "momentum" | "macd" | "risk" | "newHigh";

export const StockLab = React.memo(() => {
    const [tickerInput, setTickerInput] = React.useState("QQQ");
    const [marketData, setMarketData] = React.useState<MarketDataResponse | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [fetchError, setFetchError] = React.useState<string | null>(null);
    const [indicatorView, setIndicatorView] = React.useState<IndicatorView>("price");
    const [transferMessage, setTransferMessage] = React.useState<{type: "status" | "error"; text: string} | null>(null);
    const requestIdRef = React.useRef(0);
    const demo = useEvolutionDemo<MarketDataResponse["points"], TradingReplay>({
        topic: "stock",
        createWorker: () => new Worker(new URL("../workers/stock.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: {
            ...DEFAULT_CONFIG,
            seed: Math.round(Math.random() * 1_000_000),
        },
        data: marketData?.points,
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
    const masks = decoded?.masks ?? replay?.indicatorMasks;
    /**
     * Ablation is O(masks) full train evals. While evolving, only recompute when the
     * throttled champion showcase refreshes; when paused/idle, follow the live champion genome.
     */
    const ablationTrigger = demo.status === "running" ? demo.showcaseEpoch : demo.champion?.genome;
    const ablation = React.useMemo(() => {
        const genome = demo.champion?.genome;
        if (!genome || !marketData?.points.length) {
            return null;
        }
        try {
            return ablateIndicatorMasks(genome, marketData.points, useNetwork);
        } catch {
            return null;
        }
        // Intentionally omit demo.champion from deps: trigger gates cadence; render provides latest genome.
        // eslint-disable-next-line react-hooks/exhaustive-deps -- ablationTrigger encodes status/genome/showcase
    }, [ablationTrigger, marketData?.points, useNetwork]);
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
    const splitDate = replay?.points.find(point => point.segment === "test")?.date;
    const metricsExtra = React.useMemo(
        () => [
            {label: "訓練回報", value: replay ? formatPercent(replay.trainReturn) : "—"},
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
            const input = buildNetworkFeatures(columns, index, position, decoded.parameters, decoded.masks);
            return {
                input,
                index,
                date,
                maxIndex: columns.length - 1,
                segment: index < Math.floor(columns.length * 0.8) ? "訓練" : "測試",
            };
        } catch {
            return null;
        }
    }, [useNetwork, demo.champion?.genome, marketData, decoded, previewIndex, replay, liveDay]);

    return (
        <DemoShell
            accent="stock"
            description="以遺傳演算法同時進化指標 on/off（feature selection）、週期 / 門檻，同薄 Brain.js 決策頭。Sparsity 懲罰多餘指標；冠軍會做 ablation 睇邊個真正有用。80% 訓練、20% 樣本外。"
            icon={<CandlestickChart size={20} strokeWidth={1.5} />}
            title="股票交易 · 神經演化"
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
                    <Metrics extra={metricsExtra} stats={demo.stats} />
                    <div className="simulation-stage stock-stage">
                        <div className="stage-overlay">
                            <span>{marketData?.symbol ?? "QQQ"} · 逐日重播</span>
                            <span>{replay ? (demo.status === "running" ? "冠軍循環重播 · 進化中" : "冠軍循環重播") : "未有冠軍"}</span>
                        </div>
                        <StockPlaybackCanvas loop onDayChange={handleDayChange} playing={Boolean(replay)} replay={replay} restartKey={demo.showcaseEpoch} speed={demo.config.speed} />
                    </div>
                    {parameters ? (
                        <section className="optimized-panel">
                            <div className="panel-heading">
                                <div>
                                    <p className="eyebrow">冠軍基因體 · 指標選擇 + 參數</p>
                                    <h3>最佳指標參數</h3>
                                </div>
                                <Button onPress={() => downloadPineScript(demo.champion!.genome, marketData?.symbol ?? "QQQ", useNetwork)} size="sm" variant="secondary">
                                    <FileDown size={15} strokeWidth={1.5} />
                                    匯出 Pine Script
                                </Button>
                            </div>
                            {masks ? (
                                <div className="mask-section">
                                    <div className="mask-section-label">
                                        <span>
                                            指標開關（{countActiveMasks(masks)} / {STOCK_MASK_GENE_COUNT} 開）
                                        </span>
                                        <span className="mask-section-hint">前 5 個免罰，之後每個 −0.65 fitness</span>
                                    </div>
                                    <div className="mask-chip-row" role="list" aria-label="指標開關">
                                        {INDICATOR_MASK_DEFS.map(def => {
                                            const on = masks[def.id];
                                            return (
                                                <span className={on ? "mask-chip mask-chip--on" : "mask-chip mask-chip--off"} key={def.id} role="listitem" title={def.label}>
                                                    {def.shortLabel}
                                                    <em>{on ? "ON" : "OFF"}</em>
                                                </span>
                                            );
                                        })}
                                    </div>
                                </div>
                            ) : null}
                            <div className="parameter-grid">
                                <ParameterValue label="移動平均線" value={`${parameters.smaFastPeriod} / ${parameters.smaSlowPeriod}${masks && !masks.sma ? " · 關" : ""}`} />
                                <ParameterValue
                                    label="RSI"
                                    value={`${parameters.rsiPeriod} 日 · 買 ≤ ${parameters.rsiBuyThreshold} · 賣 ≥ ${parameters.rsiSellThreshold}${masks && !masks.rsi ? " · 關" : ""}`}
                                />
                                <ParameterValue label="保力加通道" value={`${parameters.bollingerPeriod} / ${parameters.bollingerMultiplier.toFixed(2)}σ${masks && !masks.bollinger ? " · 關" : ""}`} />
                                <ParameterValue label="ROC 週期" value={`${parameters.rocPeriod}${masks && !masks.roc ? " · 關" : ""}`} />
                                <ParameterValue
                                    label="威廉指標"
                                    value={`${parameters.williamsPeriod} 日 · 買 ≤ ${parameters.williamsBuyThreshold} · 賣 ≥ ${parameters.williamsSellThreshold}${masks && !masks.williams ? " · 關" : ""}`}
                                />
                                <ParameterValue
                                    label="MACD"
                                    value={`${parameters.macdFastPeriod} / ${parameters.macdSlowPeriod} / ${parameters.macdSignalPeriod}${masks && !masks.macd ? " · 關" : ""}`}
                                />
                                <ParameterValue label="波動率" value={`${parameters.volatilityPeriod}${masks && !masks.volatility ? " · 關" : ""}`} />
                                <ParameterValue label="成交量" value={`${parameters.volumeZScorePeriod}${masks && !masks.volume ? " · 關" : ""}`} />
                                <ParameterValue label="N日新高" value={`${parameters.newHighPeriod}${masks && !masks.newHigh ? " · 關" : ""}`} />
                                <ParameterValue label="Head 基因" value={`${STOCK_HEAD_GENE_COUNT}（週期 ${STOCK_PARAMETER_GENE_COUNT} + mask ${STOCK_MASK_GENE_COUNT}，突變 ×3）`} />
                                <ParameterValue label="決策頭" value={useNetwork ? describeStockNetwork() : "啟用票多數；RSI / 威廉指標可賣出"} />
                                <ParameterValue label="網絡基因" value={useNetwork ? `${STOCK_NETWORK_GENE_COUNT}（突變 ×0.35）` : `${STOCK_NETWORK_GENE_COUNT}（規則模式未使用）`} />
                            </div>
                            {ablation ? (
                                <div className="ablation-section">
                                    <div className="mask-section-label">
                                        <span>Ablation · 關掉後 fitness 變化</span>
                                        <span className="mask-section-hint">Δ &gt; 0 = 有用（唔好亂關）；Δ ≤ 0 = 裝飾品</span>
                                    </div>
                                    <div className="ablation-table-wrap">
                                        <table className="ablation-table">
                                            <thead>
                                                <tr>
                                                    <th>指標</th>
                                                    <th>狀態</th>
                                                    <th>Δ fitness</th>
                                                    <th>解讀</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {ablation.rows.map(row => {
                                                    const useful = row.enabled && row.fitnessDrop > 0.15;
                                                    const noise = row.enabled && row.fitnessDrop <= 0;
                                                    let verdict = "未選用";
                                                    if (row.enabled) {
                                                        if (useful) {
                                                            verdict = "有貢獻";
                                                        } else if (noise) {
                                                            verdict = "可關";
                                                        } else {
                                                            verdict = "弱貢獻";
                                                        }
                                                    }
                                                    return (
                                                        <tr className={!row.enabled ? "ablation-row--off" : useful ? "ablation-row--useful" : noise ? "ablation-row--noise" : undefined} key={row.id}>
                                                            <td>
                                                                {row.label}
                                                                <span className="ablation-short">{row.shortLabel}</span>
                                                            </td>
                                                            <td>{row.enabled ? "ON" : "OFF"}</td>
                                                            <td className="font-mono">{row.enabled ? formatFitnessDrop(row.fitnessDrop) : "—"}</td>
                                                            <td>{verdict}</td>
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            ) : null}
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
                                    splitDate={splitDate}
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
                        <div className="chart-height-md">{replay ? <EquityChart points={replay.points} splitDate={splitDate} /> : <div className="empty-chart">訓練出冠軍後會顯示權益曲線。</div>}</div>
                    </section>
                    {useNetwork ? (
                        <NetworkPanel
                            genome={networkGenome}
                            input={networkPreview?.input ?? null}
                            inputLabels={STOCK_INPUT_LABELS}
                            outputLabels={STOCK_OUTPUT_LABELS}
                            subtitle="只顯示決策頭（週期基因另見上方參數表）。節點亮度跟住上方逐日重播；亦可拖滑桿手動 scrub。"
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
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="82% 全段 + 18% soft-robust 半段：年化×130 + 累積回報×55 + Sharpe×10 − 回撤×22 + 超額×22 + 倉位暴露×8 − 閒置/低倉罰 − 輕 L2 − soft sparsity（超過 5 個各 −0.65）；主力係推高訓練回報，唔係避風險躺平"
                        genome={`${STOCK_PARAMETER_GENE_COUNT} 週期/門檻 + ${STOCK_MASK_GENE_COUNT} 個 on/off mask（一齊突變 ×3）+ ${STOCK_NETWORK_GENE_COUNT} 決策頭權重（×0.35；${describeStockNetwork()}）`}
                        inputs="17 維特徵；被 mask 關掉嘅指標家族會強制填 0。持倉狀態永遠開啟。"
                        outputs={
                            useNetwork
                                ? "薄隱藏層取最大 → 買 / 持 / 賣；搜尋主力喺 mask + 週期 / 門檻"
                                : "啟用中嘅 SMA / MACD / RSI / 威廉 多數票買入；RSI / 威廉過熱賣出（兩者都關則買入條件失敗就離場）"
                        }
                        termination="頭 80% 做選擇；尾 20% 唔入訓練；移民只重抽 head（參數 + mask）"
                    />
                </main>
                <aside className="demo-sidebar">
                    <DemoControls demo={demo} disabled={!marketData || loading}>
                        <Switch isSelected={useNetwork} onChange={checked => demo.setConfig(current => ({...current, useNeuralNetwork: checked}))}>
                            <Switch.Content>
                                <Switch.Control>
                                    <Switch.Thumb />
                                </Switch.Control>
                                使用神經網絡
                            </Switch.Content>
                        </Switch>
                        <GenomeTransfer
                            disabled={!marketData || loading}
                            fitness={demo.champion?.fitness}
                            geneCount={STOCK_GENE_COUNT}
                            genome={demo.champion?.genome}
                            onImport={handleImportGenome}
                            onMessage={setTransferMessage}
                            score={replay ? Math.round(replay.trainReturn * 1000) / 10 : undefined}
                            topic="stock"
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
};

interface MarketChartProps {
    data: MarketChartDatum[];
    indicatorView: IndicatorView;
    parameters: TradingReplay["optimizedParameters"] | undefined;
    replay: TradingReplay | undefined;
    splitDate: string | undefined;
    marketRange: {startIndex: number; endIndex: number};
    onRangeChange: (range: {startIndex: number; endIndex: number}) => void;
}

/**
 * Heavy 15y market chart. Memoized so the ~8/sec generation ticks (which only touch
 * stats/history) do not force recharts to redraw thousands of points every frame —
 * series data only re-renders when the champion replay actually refreshes.
 */
const MarketChart = React.memo<MarketChartProps>(
    ({data, indicatorView, parameters, replay, splitDate, marketRange, onRangeChange}) => {
        const hasReplay = replay !== undefined;
        const handleBrushChange = (range: {startIndex?: number; endIndex?: number}) => {
            onRangeChange({startIndex: range.startIndex ?? 0, endIndex: range.endIndex ?? data.length - 1});
        };
        return (
            <ResponsiveContainer height="100%" width="100%">
                <LineChart data={data} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
                    <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" minTickGap={70} stroke="#747b86" tick={{fontSize: 12}} tickLine={false} />
                    <YAxis domain={["auto", "auto"]} stroke="#747b86" tick={{fontSize: 12}} tickLine={false} width={58} yAxisId="price" />
                    {(indicatorView === "momentum" || indicatorView === "macd" || indicatorView === "risk" || indicatorView === "newHigh") && hasReplay ? (
                        <YAxis domain={["auto", "auto"]} orientation="right" stroke="#747b86" tick={{fontSize: 12}} tickLine={false} width={48} yAxisId="indicator" />
                    ) : null}
                    <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} />
                    <Legend wrapperStyle={{fontSize: 12}} />
                    <Line dataKey="close" dot={false} isAnimationActive={false} name="收市" stroke="#dfe3e8" strokeWidth={1.5} type="monotone" yAxisId="price" />
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
                    {splitDate ? <ReferenceLine label={{value: "測試", fill: "#e7b955", fontSize: 12}} stroke="#e7b955" strokeDasharray="4 4" x={splitDate} /> : null}
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
        prev.splitDate === next.splitDate &&
        prev.marketRange.startIndex === next.marketRange.startIndex &&
        prev.marketRange.endIndex === next.marketRange.endIndex &&
        prev.onRangeChange === next.onRangeChange
);

const BUY_DOT = {fill: "#58d68d", r: 5, strokeWidth: 0} as const;
const SELL_DOT = {fill: "#e36f5b", r: 5, strokeWidth: 0} as const;

interface EquityChartProps {
    points: TradingPoint[];
    splitDate: string | undefined;
}

/** Out-of-sample equity curve. Memoized for the same reason as MarketChart. */
const EquityChart = React.memo<EquityChartProps>(
    ({points, splitDate}) => (
        <ResponsiveContainer height="100%" width="100%">
            <LineChart data={points} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
                <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" minTickGap={70} stroke="#747b86" tick={{fontSize: 12}} tickLine={false} />
                <YAxis stroke="#747b86" tick={{fontSize: 12}} tickLine={false} width={64} />
                <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} />
                <Line dataKey="strategy" dot={false} isAnimationActive={false} name="策略" stroke="#58d68d" strokeWidth={2} type="monotone" />
                <Line dataKey="benchmark" dot={false} isAnimationActive={false} name="買入持有" stroke="#e7b955" strokeWidth={1.5} type="monotone" />
                {splitDate ? <ReferenceLine stroke="#e7b955" strokeDasharray="4 4" x={splitDate} /> : null}
            </LineChart>
        </ResponsiveContainer>
    ),
    (prev, next) => prev.points === next.points && prev.splitDate === next.splitDate
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

function formatFitnessDrop(value: number): string {
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}`;
}

function formatBrushDate(value: string): string {
    return value.slice(0, 7);
}

function downloadPineScript(genome: number[], symbol: string, useNetwork: boolean): void {
    const script = createPineScript(genome, symbol, useNetwork);
    const blob = new Blob([script], {type: "text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${symbol.toLowerCase()}-evolab-strategy.pine`;
    anchor.click();
    URL.revokeObjectURL(url);
}
