export type TopicId = "theory" | "snake" | "breaker" | "breaker-rainbow" | "flappy" | "stock" | "stock-mc";

export interface GAConfig {
    populationSize: number;
    mutationRate: number;
    mutationScale: number;
    eliteRate: number;
    seed: number;
    speed: number;
    /** Stock lab only: false = 用經典規則投票（SMA / RSI / MACD）代替 NN decision head。 */
    useNeuralNetwork?: boolean;
}

export type Genome = number[];

export interface NetworkTopology {
    inputSize: number;
    hiddenLayers: number[];
    outputSize: number;
}

export interface GenerationStats {
    generation: number;
    bestFitness: number;
    averageFitness: number;
    diversity: number;
}

export interface Champion<TReplay = unknown> {
    genome: Genome;
    fitness: number;
    replay: TReplay;
}

/** Generation payload may omit replay when the champion did not improve (avoids huge postMessage clones). */
export type GenerationChampion<TReplay = unknown> = {
    genome: Genome;
    fitness: number;
    replay?: TReplay;
};

export interface Point {
    x: number;
    y: number;
}

export interface SnakeFrame {
    snake: Point[];
    food: Point;
    score: number;
    step: number;
    terminal?: "collision" | "starved" | "timeout";
}

export interface SnakeReplay {
    frames: SnakeFrame[];
    score: number;
    steps: number;
}

export interface BreakerBrick extends Point {
    id: number;
    active: boolean;
}

export interface BreakerFrame {
    paddleX: number;
    ball: Point;
    /** Ball velocity at capture time — used to rebuild network inputs for live viz. */
    ballVelocity?: Point;
    bricks: BreakerBrick[];
    hits: number;
    step: number;
    terminal?: "lost" | "cleared" | "timeout";
}

export interface BreakerReplay {
    frames: BreakerFrame[];
    bricksCleared: number;
    hits: number;
    steps: number;
}

export interface FlappyPipe {
    x: number;
    /** Gap vertical centre (px). */
    gapY: number;
    gapHeight: number;
    passed: boolean;
}

export interface FlappyFrame {
    birdY: number;
    birdVy: number;
    pipes: FlappyPipe[];
    score: number;
    step: number;
    terminal?: "crash" | "timeout";
}

export interface FlappyReplay {
    frames: FlappyFrame[];
    score: number;
    steps: number;
}

export interface MarketDataPoint {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    adjClose: number | null;
    volume: number;
}

export interface MarketDataResponse {
    symbol: string;
    currency: string;
    timezone: string;
    fetchedAt: string;
    points: MarketDataPoint[];
}

export interface OptimizedIndicatorParameters {
    smaFastPeriod: number;
    smaSlowPeriod: number;
    williamsPeriod: number;
    williamsBuyThreshold: number;
    williamsSellThreshold: number;
    rocPeriod: number;
    rsiPeriod: number;
    rsiBuyThreshold: number;
    rsiSellThreshold: number;
    macdFastPeriod: number;
    macdSlowPeriod: number;
    macdSignalPeriod: number;
    bollingerPeriod: number;
    bollingerMultiplier: number;
    volatilityPeriod: number;
    volumeZScorePeriod: number;
    /** Lookback for N-day highest high (breakout / new-high signal). */
    newHighPeriod: number;
    /** Lookback for N-day lowest low (support / new-low signal). */
    newLowPeriod: number;
}

export interface IndicatorSnapshot {
    date: string;
    close: number;
    smaFast: number;
    smaSlow: number;
    williamsR: number;
    roc: number;
    rsi: number;
    macd: number;
    macdSignal: number;
    macdHistogram: number;
    bollingerUpper: number;
    bollingerLower: number;
    bollingerPercentB: number;
    bollingerBandwidth: number;
    volatility: number;
    volumeZScore: number;
    /** Rolling N-day highest high. */
    nDayHigh: number;
    /** close / nDayHigh — 1.0 means at the N-day high. */
    newHighRatio: number;
    /** Rolling N-day lowest low. */
    nDayLow: number;
    /** nDayLow / close — 1.0 means at the N-day low. */
    newLowRatio: number;
}

export interface TradeMarker {
    date: string;
    action: "buy" | "sell";
    price: number;
    /** Position after the fill: 0 flat / +1 long */
    position: number;
}

export interface TradingPoint {
    date: string;
    close: number;
    strategy: number;
    benchmark: number;
    /** development 參與 walk-forward fitness；holdout 只喺暫停後揭示 */
    segment: "development" | "holdout";
    smaFast: number;
    smaSlow: number;
    rsi: number;
    williamsR: number;
    roc: number;
    macd: number;
    macdSignal: number;
    bollingerUpper: number;
    bollingerLower: number;
    volatility: number;
    volumeZScore: number;
    nDayHigh: number;
    newHighRatio: number;
    nDayLow: number;
    newLowRatio: number;
}

export interface TradingReplay {
    points: TradingPoint[];
    trades: TradeMarker[];
    developmentReturn: number;
    /** 封存 holdout 回報；訓練中未評估所以係 null */
    holdoutReturn: number | null;
    /** 全段買入持有（權益曲線對照） */
    benchmarkReturn: number;
    /** Development 買入持有（同 developmentReturn 對齊） */
    developmentBenchmarkReturn: number;
    /** 封存 holdout 買入持有；訓練中未評估所以係 null */
    holdoutBenchmarkReturn: number | null;
    sharpe: number;
    maxDrawdown: number;
    optimizedParameters: OptimizedIndicatorParameters;
}

export type WorkerCommand<TData = unknown> =
    {type: "start"; config: GAConfig; data?: TData; champion?: Genome} | {type: "pause"} | {type: "reset"} | {type: "update-config"; config: GAConfig} | {type: "set-data"; data: TData};

export type WorkerEvent<TReplay = unknown> =
    | {type: "status"; status: "idle" | "running" | "paused"}
    | {
          type: "generation";
          stats: GenerationStats;
          champion: GenerationChampion<TReplay>;
          /** Pause handler emits a full replay of the best network for showcase playback. */
          reason?: "pause-showcase";
      }
    | {type: "error"; message: string};

export interface PersistedDemoState {
    /** Lab-specific config JSON (GAConfig for evolution labs, RainbowConfig for breaker-rainbow). */
    config: GAConfig;
    champion?: Genome;
    bestFitness?: number;
}

export interface PersistedLabStateV1 {
    version: 1;
    demos: Partial<Record<Exclude<TopicId, "theory">, PersistedDemoState>>;
}
