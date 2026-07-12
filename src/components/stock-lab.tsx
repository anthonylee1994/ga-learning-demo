import React from "react";
import {Button, Chip, Spinner} from "@heroui/react";
import {CandlestickChart, Download, FileDown, TriangleAlert} from "lucide-react";
import {Brush, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import {useEvolutionDemo} from "../hooks/use-evolution-demo";
import type {GAConfig, MarketDataResponse, TradingPoint, TradingReplay} from "../lib/types";
import {createPineScript} from "../domains/stock/pine-script";
import {ApplicationPanel} from "./application-panel";
import {DemoControls} from "./demo-controls";
import {FitnessChart} from "./fitness-chart";
import {Metrics} from "./metrics";
import {DemoShell} from "./snake-lab";

const DEFAULT_CONFIG: GAConfig = {
    populationSize: 12,
    mutationRate: 0.1,
    mutationScale: 0.2,
    eliteRate: 0.08,
    tournamentSize: 4,
    seed: 420,
    speed: 3,
};

type IndicatorView = "price" | "momentum" | "macd" | "risk";

export const StockLab = React.memo(() => {
    const [tickerInput, setTickerInput] = React.useState("QQQ");
    const [marketData, setMarketData] = React.useState<MarketDataResponse | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [fetchError, setFetchError] = React.useState<string | null>(null);
    const [indicatorView, setIndicatorView] = React.useState<IndicatorView>("price");
    const requestIdRef = React.useRef(0);
    const demo = useEvolutionDemo<MarketDataResponse["points"], TradingReplay>({
        topic: "stock",
        createWorker: () => new Worker(new URL("../workers/stock.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: DEFAULT_CONFIG,
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
    const replay = demo.champion?.replay;
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
    const splitDate = replay?.points.find(point => point.segment === "test")?.date;

    return (
        <DemoShell
            accent="stock"
            description="用 QQQ 近 15 年日線歷史進化交易 policy；80% 數據訓練，最後 20% 完全留作 out-of-sample 測試。"
            icon={<CandlestickChart size={20} strokeWidth={1.5} />}
            title="Stock Trading Evolution"
        >
            <div className="stock-toolbar">
                <label>
                    <span>Ticker</span>
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
                <Button isDisabled={!tickerInput.trim()} onPress={() => load(tickerInput)} size="sm" variant="secondary">
                    {loading ? <Spinner color="current" size="sm" /> : <Download size={15} strokeWidth={1.5} />}
                    載入 Ticker
                </Button>
                {marketData ? (
                    <Chip color="success" size="sm" variant="soft">
                        {marketData.symbol} · {marketData.points.length.toLocaleString()} sessions · {marketData.currency}
                    </Chip>
                ) : null}
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
                    <Metrics
                        extra={[
                            {label: "Train return", value: replay ? formatPercent(replay.trainReturn) : "—"},
                            {label: "Test return", value: replay ? formatPercent(replay.testReturn) : "—"},
                            {label: "Buy & hold", value: replay ? formatPercent(replay.benchmarkReturn) : "—"},
                            {label: "Max drawdown", value: replay ? formatPercent(-replay.maxDrawdown) : "—"},
                        ]}
                        stats={demo.stats}
                    />
                    {replay ? (
                        <section className="optimized-panel">
                            <div className="panel-heading">
                                <div>
                                    <p className="eyebrow">Champion genome</p>
                                    <h3>Best indicator parameters</h3>
                                </div>
                                <Button onPress={() => downloadPineScript(demo.champion!.genome, marketData?.symbol ?? "QQQ")} size="sm" variant="secondary">
                                    <FileDown size={15} strokeWidth={1.5} />
                                    匯出 Pine Script
                                </Button>
                            </div>
                            <div className="parameter-grid">
                                <ParameterValue label="SMA" value={`${replay.optimizedParameters.smaFastPeriod} / ${replay.optimizedParameters.smaSlowPeriod}`} />
                                <ParameterValue label="RSI" value={String(replay.optimizedParameters.rsiPeriod)} />
                                <ParameterValue label="Bollinger" value={`${replay.optimizedParameters.bollingerPeriod} / ${replay.optimizedParameters.bollingerMultiplier.toFixed(2)}σ`} />
                                <ParameterValue label="ROC" value={String(replay.optimizedParameters.rocPeriod)} />
                                <ParameterValue label="Williams %R" value={String(replay.optimizedParameters.williamsPeriod)} />
                                <ParameterValue
                                    label="MACD"
                                    value={`${replay.optimizedParameters.macdFastPeriod} / ${replay.optimizedParameters.macdSlowPeriod} / ${replay.optimizedParameters.macdSignalPeriod}`}
                                />
                                <ParameterValue label="Volatility" value={String(replay.optimizedParameters.volatilityPeriod)} />
                                <ParameterValue label="Volume Z" value={String(replay.optimizedParameters.volumeZScorePeriod)} />
                            </div>
                        </section>
                    ) : null}
                    <section className="market-panel">
                        <div className="panel-heading stock-heading">
                            <div>
                                <p className="eyebrow">{marketData?.symbol ?? "QQQ"} · Daily</p>
                                <h3>市場與交易訊號</h3>
                            </div>
                            <select aria-label="技術指標" onChange={event => setIndicatorView(event.target.value as IndicatorView)} value={indicatorView}>
                                <option value="price">SMA + Bollinger</option>
                                <option value="momentum">RSI + Williams %R + ROC</option>
                                <option value="macd">MACD</option>
                                <option value="risk">Volatility + Volume Z</option>
                            </select>
                        </div>
                        <div className="chart-height-lg">
                            {loading ? (
                                <div className="loading-state">
                                    <Spinner color="accent" />
                                    <span>下載 Yahoo Finance 數據...</span>
                                </div>
                            ) : chartData.length ? (
                                <MarketChart data={chartData} indicatorView={indicatorView} marketRange={marketRange} onRangeChange={setMarketRange} replay={replay} splitDate={splitDate} />
                            ) : (
                                <div className="empty-chart">未有市場數據。</div>
                            )}
                        </div>
                    </section>
                    <section className="market-panel">
                        <div className="panel-heading">
                            <div>
                                <p className="eyebrow">Out-of-sample audit</p>
                                <h3>Strategy vs Buy & Hold</h3>
                            </div>
                        </div>
                        <div className="chart-height-md">
                            {replay ? <EquityChart points={replay.points} splitDate={splitDate} /> : <div className="empty-chart">訓練出 champion 後會顯示 equity curve。</div>}
                        </div>
                    </section>
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="每個窗口：超額回報(vs buy&hold) × 100 + Sharpe × 10 − maxDD × 40；fit 同 validation 取 mean − 0.5·overfit gap − weight L2"
                        genome="12 個 indicator parameter genes + Brain.js 14 → 16 → 8 → 3 weights 與 biases"
                        inputs="GA 優化後嘅 SMA、Williams %R、ROC、RSI、MACD、Bollinger、volatility、volume z-score、倉位"
                        outputs="Brain.js network → argMax(buy / hold / sell)：100% long、維持倉位、100% cash"
                        termination="train 內部再切 70% fit / 30% validation，selection 睇 validation 抗 overfit；最後 20% test data 絕不參與 selection"
                    />
                </main>
                <aside className="demo-sidebar">
                    <DemoControls demo={demo} disabled={!marketData || loading} />
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
};

interface MarketChartProps {
    data: MarketChartDatum[];
    indicatorView: IndicatorView;
    replay: TradingReplay | undefined;
    splitDate: string | undefined;
    marketRange: {startIndex: number; endIndex: number};
    onRangeChange: React.Dispatch<React.SetStateAction<{startIndex: number; endIndex: number}>>;
}

/**
 * Heavy 15y market chart. Memoized so the ~8/sec generation ticks (which only touch
 * stats/history) do not force recharts to redraw thousands of points every frame —
 * it only re-renders when the champion replay actually refreshes.
 */
const MarketChart = React.memo<MarketChartProps>(({data, indicatorView, replay, splitDate, marketRange, onRangeChange}) => (
    <ResponsiveContainer height="100%" width="100%">
        <LineChart data={data} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
            <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" minTickGap={70} stroke="#747b86" tick={{fontSize: 10}} tickLine={false} />
            <YAxis domain={["auto", "auto"]} stroke="#747b86" tick={{fontSize: 10}} tickLine={false} width={58} yAxisId="price" />
            {indicatorView !== "price" && replay ? (
                <YAxis domain={["auto", "auto"]} orientation="right" stroke="#747b86" tick={{fontSize: 10}} tickLine={false} width={48} yAxisId="indicator" />
            ) : null}
            <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} />
            <Legend wrapperStyle={{fontSize: 11}} />
            <Line dataKey="close" dot={false} isAnimationActive={false} name="Close" stroke="#dfe3e8" strokeWidth={1.5} type="monotone" yAxisId="price" />
            {replay ? (
                <React.Fragment>
                    <Line connectNulls={false} dataKey="buy" dot={{fill: "#58d68d", r: 2.5, strokeWidth: 0}} isAnimationActive={false} name="Buy" stroke="none" yAxisId="price" />
                    <Line connectNulls={false} dataKey="sell" dot={{fill: "#e36f5b", r: 2.5, strokeWidth: 0}} isAnimationActive={false} name="Sell" stroke="none" yAxisId="price" />
                </React.Fragment>
            ) : null}
            {indicatorView === "price" && replay ? (
                <React.Fragment>
                    <Line dataKey="smaFast" dot={false} isAnimationActive={false} name={`SMA${replay.optimizedParameters.smaFastPeriod}`} stroke="#e7b955" strokeWidth={1} yAxisId="price" />
                    <Line dataKey="smaSlow" dot={false} isAnimationActive={false} name={`SMA${replay.optimizedParameters.smaSlowPeriod}`} stroke="#5da6d9" strokeWidth={1} yAxisId="price" />
                    <Line dataKey="bollingerUpper" dot={false} isAnimationActive={false} name="BB upper" stroke="#6f7782" strokeDasharray="4 4" strokeWidth={1} yAxisId="price" />
                    <Line dataKey="bollingerLower" dot={false} isAnimationActive={false} name="BB lower" stroke="#6f7782" strokeDasharray="4 4" strokeWidth={1} yAxisId="price" />
                </React.Fragment>
            ) : null}
            {indicatorView === "momentum" && replay ? (
                <React.Fragment>
                    <Line dataKey="rsi" dot={false} isAnimationActive={false} name="RSI" stroke="#63c6a1" strokeWidth={1} yAxisId="indicator" />
                    <Line dataKey="williamsR" dot={false} isAnimationActive={false} name="Williams %R" stroke="#e36f5b" strokeWidth={1} yAxisId="indicator" />
                    <Line dataKey="roc" dot={false} isAnimationActive={false} name="ROC" stroke="#b38bd4" strokeWidth={1} yAxisId="indicator" />
                </React.Fragment>
            ) : null}
            {indicatorView === "macd" && replay ? (
                <React.Fragment>
                    <Line dataKey="macd" dot={false} isAnimationActive={false} name="MACD" stroke="#63c6a1" strokeWidth={1} yAxisId="indicator" />
                    <Line dataKey="macdSignal" dot={false} isAnimationActive={false} name="Signal" stroke="#e7b955" strokeWidth={1} yAxisId="indicator" />
                </React.Fragment>
            ) : null}
            {indicatorView === "risk" && replay ? (
                <React.Fragment>
                    <Line dataKey="volatility" dot={false} isAnimationActive={false} name="Volatility" stroke="#e36f5b" strokeWidth={1} yAxisId="indicator" />
                    <Line dataKey="volumeZScore" dot={false} isAnimationActive={false} name="Volume Z" stroke="#5da6d9" strokeWidth={1} yAxisId="indicator" />
                </React.Fragment>
            ) : null}
            {splitDate ? <ReferenceLine label={{value: "TEST", fill: "#e7b955", fontSize: 10}} stroke="#e7b955" strokeDasharray="4 4" x={splitDate} /> : null}
            <Brush
                ariaLabel="市場日期縮放範圍"
                className="market-zoom-brush"
                dataKey="date"
                endIndex={marketRange.endIndex}
                fill="#0d1115"
                gap={Math.max(1, Math.floor(data.length / 1000))}
                height={28}
                onChange={range => onRangeChange({startIndex: range.startIndex ?? 0, endIndex: range.endIndex ?? data.length - 1})}
                startIndex={marketRange.startIndex}
                stroke="#49515b"
                tickFormatter={formatBrushDate}
                travellerWidth={10}
            />
        </LineChart>
    </ResponsiveContainer>
));

interface EquityChartProps {
    points: TradingPoint[];
    splitDate: string | undefined;
}

/** Out-of-sample equity curve. Memoized for the same reason as MarketChart. */
const EquityChart = React.memo<EquityChartProps>(({points, splitDate}) => (
    <ResponsiveContainer height="100%" width="100%">
        <LineChart data={points} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
            <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" minTickGap={70} stroke="#747b86" tick={{fontSize: 10}} tickLine={false} />
            <YAxis stroke="#747b86" tick={{fontSize: 10}} tickLine={false} width={64} />
            <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} />
            <Line dataKey="strategy" dot={false} isAnimationActive={false} name="Strategy" stroke="#58d68d" strokeWidth={2} type="monotone" />
            <Line dataKey="benchmark" dot={false} isAnimationActive={false} name="Buy & hold" stroke="#e7b955" strokeWidth={1.5} type="monotone" />
            {splitDate ? <ReferenceLine stroke="#e7b955" strokeDasharray="4 4" x={splitDate} /> : null}
        </LineChart>
    </ResponsiveContainer>
));

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

function downloadPineScript(genome: number[], symbol: string): void {
    const script = createPineScript(genome, symbol);
    const blob = new Blob([script], {type: "text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${symbol.toLowerCase()}-evolab-strategy.pine`;
    anchor.click();
    URL.revokeObjectURL(url);
}
