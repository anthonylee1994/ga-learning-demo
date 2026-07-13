export type TopicId = "theory" | "snake" | "breaker" | "stock";

export interface GAConfig {
    populationSize: number;
    mutationRate: number;
    mutationScale: number;
    eliteRate: number;
    tournamentSize: number;
    seed: number;
    speed: number;
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
    rocPeriod: number;
    rsiPeriod: number;
    macdFastPeriod: number;
    macdSlowPeriod: number;
    macdSignalPeriod: number;
    bollingerPeriod: number;
    bollingerMultiplier: number;
    volatilityPeriod: number;
    volumeZScorePeriod: number;
}

/** Pure rule thresholds evolved by GA (no neural network). */
export interface OptimizedStrategyRules {
    rsiBuy: number;
    rsiSell: number;
    williamsBuy: number;
    williamsSell: number;
    rocBuy: number;
    rocSell: number;
    bollingerBuy: number;
    bollingerSell: number;
    minBuySignals: number;
    minSellSignals: number;
    useTrendFilter: boolean;
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
}

export interface TradeMarker {
    date: string;
    action: "buy" | "sell";
    price: number;
}

export interface TradingPoint {
    date: string;
    close: number;
    strategy: number;
    benchmark: number;
    segment: "train" | "test";
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
}

export interface TradingReplay {
    points: TradingPoint[];
    trades: TradeMarker[];
    trainReturn: number;
    testReturn: number;
    benchmarkReturn: number;
    sharpe: number;
    maxDrawdown: number;
    optimizedParameters: OptimizedIndicatorParameters;
    optimizedRules: OptimizedStrategyRules;
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
    config: GAConfig;
    champion?: Genome;
    bestFitness?: number;
}

export interface PersistedLabStateV1 {
    version: 1;
    demos: Partial<Record<Exclude<TopicId, "theory">, PersistedDemoState>>;
}
