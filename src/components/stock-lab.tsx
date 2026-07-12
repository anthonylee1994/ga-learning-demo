import React from "react";
import {Button, Chip, Spinner} from "@heroui/react";
import {CandlestickChart, Download, TriangleAlert} from "lucide-react";
import {CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import {useEvolutionDemo} from "../hooks/use-evolution-demo";
import type {GAConfig, MarketDataResponse, TradingReplay} from "../lib/types";
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

export const StockLab = React.memo(function StockLab() {
    const [tickerInput, setTickerInput] = React.useState("QQQ");
    const [marketData, setMarketData] = React.useState<MarketDataResponse | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [fetchError, setFetchError] = React.useState<string | null>(null);
    const [indicatorView, setIndicatorView] = React.useState<IndicatorView>("price");
    const demo = useEvolutionDemo<MarketDataResponse["points"], TradingReplay>({
        topic: "stock",
        createWorker: () => new Worker(new URL("../workers/stock.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: DEFAULT_CONFIG,
        data: marketData?.points,
    });

    const load = (symbol: string) => {
        setLoading(true);
        setFetchError(null);
        loadMarketData(symbol)
            .then(payload => {
                setMarketData(payload);
                setTickerInput(payload.symbol);
            })
            .catch((error: unknown) => setFetchError(error instanceof Error ? error.message : "下載失敗。"))
            .finally(() => setLoading(false));
    };

    React.useEffect(() => load("QQQ"), []);
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
    const splitDate = replay?.points.find(point => point.segment === "test")?.date;

    return (
        <DemoShell
            accent="stock"
            description="用 QQQ 十年日線進化交易 policy；80% 數據訓練，最後 20% 完全留作 out-of-sample 測試。"
            icon={<CandlestickChart size={20} strokeWidth={1.5} />}
            title="Stock Trading Evolution"
        >
            <div className="stock-toolbar">
                <label>
                    <span>Ticker</span>
                    <input aria-label="股票代號" className="ticker-input" maxLength={15} onChange={event => setTickerInput(event.target.value.toUpperCase())} value={tickerInput} />
                </label>
                <Button isDisabled={loading} isPending={loading} onPress={() => load(tickerInput)} size="sm" variant="secondary">
                    {loading ? <Spinner color="current" size="sm" /> : <Download size={15} strokeWidth={1.5} />}
                    下載 10 年日線
                </Button>
                {marketData ? (
                    <Chip color="success" size="sm" variant="soft">
                        {marketData.points.length.toLocaleString()} sessions · {marketData.currency}
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
                                <ResponsiveContainer height="100%" width="100%">
                                    <LineChart data={chartData} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
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
                                                <Line
                                                    connectNulls={false}
                                                    dataKey="buy"
                                                    dot={{fill: "#58d68d", r: 2.5, strokeWidth: 0}}
                                                    isAnimationActive={false}
                                                    name="Buy"
                                                    stroke="none"
                                                    yAxisId="price"
                                                />
                                                <Line
                                                    connectNulls={false}
                                                    dataKey="sell"
                                                    dot={{fill: "#e36f5b", r: 2.5, strokeWidth: 0}}
                                                    isAnimationActive={false}
                                                    name="Sell"
                                                    stroke="none"
                                                    yAxisId="price"
                                                />
                                            </React.Fragment>
                                        ) : null}
                                        {indicatorView === "price" && replay ? (
                                            <React.Fragment>
                                                <Line dataKey="sma20" dot={false} isAnimationActive={false} name="SMA20" stroke="#e7b955" strokeWidth={1} yAxisId="price" />
                                                <Line dataKey="sma50" dot={false} isAnimationActive={false} name="SMA50" stroke="#5da6d9" strokeWidth={1} yAxisId="price" />
                                                <Line
                                                    dataKey="bollingerUpper"
                                                    dot={false}
                                                    isAnimationActive={false}
                                                    name="BB upper"
                                                    stroke="#6f7782"
                                                    strokeDasharray="4 4"
                                                    strokeWidth={1}
                                                    yAxisId="price"
                                                />
                                                <Line
                                                    dataKey="bollingerLower"
                                                    dot={false}
                                                    isAnimationActive={false}
                                                    name="BB lower"
                                                    stroke="#6f7782"
                                                    strokeDasharray="4 4"
                                                    strokeWidth={1}
                                                    yAxisId="price"
                                                />
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
                                    </LineChart>
                                </ResponsiveContainer>
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
                            {replay ? (
                                <ResponsiveContainer height="100%" width="100%">
                                    <LineChart data={replay.points} margin={{left: 0, right: 14, top: 8, bottom: 0}}>
                                        <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                                        <XAxis dataKey="date" minTickGap={70} stroke="#747b86" tick={{fontSize: 10}} tickLine={false} />
                                        <YAxis stroke="#747b86" tick={{fontSize: 10}} tickLine={false} width={64} />
                                        <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} />
                                        <Line dataKey="strategy" dot={false} isAnimationActive={false} name="Strategy" stroke="#58d68d" strokeWidth={2} type="monotone" />
                                        <Line dataKey="benchmark" dot={false} isAnimationActive={false} name="Buy & hold" stroke="#e7b955" strokeWidth={1.5} type="monotone" />
                                        {splitDate ? <ReferenceLine stroke="#e7b955" strokeDasharray="4 4" x={splitDate} /> : null}
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="empty-chart">訓練出 champion 後會顯示 equity curve。</div>
                            )}
                        </div>
                    </section>
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="Training return × 100 + Sharpe × 8 − max drawdown × 45"
                        genome="Brain.js 14 → 16 → 8 → 3 network 嘅 weights 與 biases"
                        inputs="SMA、Williams %R、ROC、RSI、MACD、Bollinger、volatility、volume z-score、倉位"
                        outputs="100% long、維持倉位、100% cash"
                        termination="由第一個有效 indicator session 跑到 training segment 尾；test data 絕不參與 selection"
                    />
                </main>
                <aside className="demo-sidebar">
                    <DemoControls demo={demo} disabled={!marketData || loading} />
                </aside>
            </div>
        </DemoShell>
    );
});

async function loadMarketData(symbol: string): Promise<MarketDataResponse> {
    const normalized = symbol.trim().toUpperCase() || "QQQ";
    const response = await fetch(`/api/market-data?symbol=${encodeURIComponent(normalized)}&range=10y&interval=1d`);
    const payload = (await response.json()) as MarketDataResponse | {error: string};
    if (!response.ok || "error" in payload) {
        throw new Error("error" in payload ? payload.error : "下載市場數據失敗。");
    }
    return payload;
}

function formatPercent(value: number): string {
    return new Intl.NumberFormat("zh-HK", {style: "percent", maximumFractionDigits: 1}).format(value);
}
