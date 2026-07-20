import React from "react";
import {Button, Card, Chip, Tooltip} from "@heroui/react";
import {Blocks, Pause, Play, RotateCcw} from "lucide-react";
import {createPpoActorReplay, DEFAULT_PPO_CONFIG, type PpoConfig, type PpoUpdateStats} from "../domains/breaker/ppo";
import {BREAKER_INPUT_LABELS, BREAKER_OUTPUT_LABELS, BREAKER_TOPOLOGY, buildBreakerInputFromFrame} from "../domains/breaker/simulation";
import type {BreakerFrame, BreakerReplay, Genome, PersistedLabStateV1} from "../lib/types";
import type {BreakerPpoWorkerCommand, BreakerPpoWorkerEvent} from "../workers/breakerPpo.worker";
import {ApplicationPanel} from "./ApplicationPanel";
import {BreakerCanvas} from "./BreakerCanvas";
import {ControlSlider} from "./DemoControls";
import {GenomeTransfer} from "./GenomeTransfer";
import {NetworkPanel} from "./NetworkPanel";
import {PpoTrainingChart} from "./PpoTrainingChart";
import {DemoShell} from "./SnakeLab";

type TrainingStatus = "idle" | "running" | "paused";

const STORAGE_KEY = "evolab-state-v1";
const PPO_TOPIC = "breaker-ppo" as const;
const PERSIST_DEBOUNCE_MS = 750;

export const BreakerPpoLab = React.memo(() => {
    const stored = React.useMemo(() => readStoredPpoDemo(), []);
    const workerRef = React.useRef<Worker | null>(null);
    const bestReplayUpdateRef = React.useRef(0);
    const persistTimerRef = React.useRef<number | null>(null);
    const skipNextConfigPersistRef = React.useRef(false);
    const workerEpochRef = React.useRef(0);
    const acceptWorkerResultsRef = React.useRef(true);
    const storedChampionRef = React.useRef<Genome | undefined>(stored?.champion);
    const storedBestReturnRef = React.useRef<number | undefined>(stored?.bestFitness);
    const configRef = React.useRef<PpoConfig>(stored?.config ?? {...DEFAULT_PPO_CONFIG, seed: Math.round(Math.random() * 1_000_000)});

    const [config, setConfig] = React.useState<PpoConfig>(() => configRef.current);
    const [status, setStatus] = React.useState<TrainingStatus>(stored?.champion?.length ? "paused" : "idle");
    const [stats, setStats] = React.useState<PpoUpdateStats | null>(() =>
        stored?.champion?.length
            ? {
                  update: 0,
                  bestUpdate: 0,
                  averageReturn: stored.bestFitness ?? 0,
                  bestReturn: stored.bestFitness ?? 0,
                  averageEpisodeLength: 0,
                  policyLoss: 0,
                  valueLoss: 0,
                  entropy: 0,
              }
            : null
    );
    const [history, setHistory] = React.useState<PpoUpdateStats[]>([]);
    const [actorGenome, setActorGenome] = React.useState<Genome | null>(() => (stored?.champion?.length ? [...stored.champion] : null));
    /** 循環重播用：每圈 re-roll 真·隨機場景（同 GA 撞磚頁）。 */
    const [showcaseReplay, setShowcaseReplay] = React.useState<BreakerReplay | null>(null);
    const [showcaseEpoch, setShowcaseEpoch] = React.useState(0);
    const [liveInput, setLiveInput] = React.useState<number[] | null>(null);
    const [error, setError] = React.useState<string | null>(null);
    const [transferMessage, setTransferMessage] = React.useState<{type: "status" | "error"; text: string} | null>(null);
    const actorGenomeRef = React.useRef(actorGenome);

    configRef.current = config;
    actorGenomeRef.current = actorGenome;

    const schedulePersist = React.useCallback((nextConfig: PpoConfig, genome?: Genome, bestReturn?: number) => {
        if (persistTimerRef.current !== null) {
            window.clearTimeout(persistTimerRef.current);
        }
        persistTimerRef.current = window.setTimeout(() => {
            persistTimerRef.current = null;
            writeStoredPpoDemo(nextConfig, genome, bestReturn);
        }, PERSIST_DEBOUNCE_MS) as unknown as number;
    }, []);

    React.useEffect(() => {
        const worker = new Worker(new URL("../workers/breakerPpo.worker.ts", import.meta.url), {type: "module"});
        workerRef.current = worker;
        worker.onmessage = function handleWorkerMessage(event: MessageEvent<BreakerPpoWorkerEvent>) {
            const epochAtReceive = workerEpochRef.current;
            const isStale = () => epochAtReceive !== workerEpochRef.current || !acceptWorkerResultsRef.current;

            if (event.data.type === "update" || event.data.type === "loaded") {
                if (isStale()) {
                    return;
                }
                const result = event.data.result;
                setStats(result.stats);
                setHistory(current => (event.data.type === "loaded" ? [result.stats] : [...current.slice(-119), result.stats]));
                setActorGenome(result.actorGenome);
                storedChampionRef.current = result.actorGenome;
                storedBestReturnRef.current = result.stats.bestReturn;
                if (event.data.type === "loaded" || result.stats.bestUpdate > bestReplayUpdateRef.current) {
                    bestReplayUpdateRef.current = result.stats.bestUpdate;
                    // 新最佳／載入：開一場真隨機 showcase；之後 loop 會再 re-roll。
                    setShowcaseReplay(result.replay);
                    setShowcaseEpoch(value => value + 1);
                }
                if (event.data.type === "loaded") {
                    setStatus(current => (current === "running" ? current : "paused"));
                    setLiveInput(null);
                }
                setError(null);
                schedulePersist(configRef.current, result.actorGenome, result.stats.bestReturn);
                return;
            }
            if (event.data.type === "error") {
                if (isStale()) {
                    return;
                }
                setStatus("paused");
                setError(event.data.message);
            }
        };
        worker.onerror = function handleWorkerError(event) {
            setStatus("paused");
            setError(event.message || "PPO worker 發生錯誤。");
        };

        // Re-hydrate actor into the worker so continue-training keeps the stored weights.
        const restoredGenome = storedChampionRef.current;
        if (restoredGenome?.length) {
            worker.postMessage({type: "load", config: configRef.current, genome: restoredGenome} satisfies BreakerPpoWorkerCommand);
        }

        return () => {
            worker.terminate();
            workerRef.current = null;
            if (persistTimerRef.current !== null) {
                window.clearTimeout(persistTimerRef.current);
                persistTimerRef.current = null;
            }
        };
    }, [schedulePersist]);

    React.useEffect(() => {
        if (skipNextConfigPersistRef.current) {
            skipNextConfigPersistRef.current = false;
            return;
        }
        schedulePersist(config, storedChampionRef.current, storedBestReturnRef.current);
    }, [config, schedulePersist]);

    const postCommand = (command: BreakerPpoWorkerCommand) => {
        workerRef.current?.postMessage(command);
    };

    const start = () => {
        workerEpochRef.current += 1;
        acceptWorkerResultsRef.current = true;
        setStatus("running");
        setError(null);
        postCommand({
            type: "start",
            config,
            genome: actorGenome ?? storedChampionRef.current ?? undefined,
        });
    };

    const pause = () => {
        setStatus("paused");
        postCommand({type: "pause"});
    };

    const reset = () => {
        workerEpochRef.current += 1;
        acceptWorkerResultsRef.current = false;
        postCommand({type: "reset"});
        storedChampionRef.current = undefined;
        storedBestReturnRef.current = undefined;
        setStatus("idle");
        setStats(null);
        setHistory([]);
        setActorGenome(null);
        setShowcaseReplay(null);
        setShowcaseEpoch(0);
        bestReplayUpdateRef.current = 0;
        setLiveInput(null);
        setError(null);
        setTransferMessage(null);
        if (persistTimerRef.current !== null) {
            window.clearTimeout(persistTimerRef.current);
            persistTimerRef.current = null;
        }
        skipNextConfigPersistRef.current = true;
        clearStoredPpoDemo();
        setConfig({...DEFAULT_PPO_CONFIG, seed: Math.round(Math.random() * 1_000_000)});
    };

    const updateConfig = (key: keyof PpoConfig, value: number) => {
        setConfig(current => ({...current, [key]: value}));
    };

    const handleFrameChange = (frame: BreakerFrame | null) => {
        setLiveInput(frame ? buildBreakerInputFromFrame(frame) : null);
    };

    const handleLoop = () => {
        const genome = actorGenomeRef.current;
        if (genome) {
            setShowcaseReplay(createPpoActorReplay(genome, configRef.current.maxSteps));
        }
    };

    const handleImportGenome = (genome: Genome) => {
        workerEpochRef.current += 1;
        acceptWorkerResultsRef.current = true;
        setStatus("paused");
        setError(null);
        postCommand({type: "load", config, genome});
    };

    return (
        <DemoShell
            accent="breaker-ppo"
            description="策略網絡逐場試動作，再用 clipped objective 同 advantage 更新；訓練／評估／重播都用真·隨機場景（同 GA），逼策略學跟波而唔係背死路線。"
            eyebrow="強化學習實驗"
            icon={<Blocks size={20} strokeWidth={1.5} />}
            title="撞磚 · PPO"
        >
            <div className="workspace-grid">
                <main className="demo-main">
                    <PpoMetrics replay={showcaseReplay} stats={stats} />
                    <div className="simulation-stage breaker-stage">
                        <div className="stage-overlay">
                            <span>Matter.js · 60 Hz · 真隨機</span>
                            <span>
                                {showcaseReplay ? (status === "running" ? "最佳策略循環重播 · 訓練中" : "最佳策略循環重播 · 每圈新一場") : status === "running" ? "收集 rollout 中" : "未有策略"}
                            </span>
                        </div>
                        <BreakerCanvas
                            loop
                            onFrameChange={handleFrameChange}
                            onLoop={handleLoop}
                            playing={Boolean(showcaseReplay)}
                            replay={showcaseReplay ?? undefined}
                            restartKey={`${showcaseEpoch}-${showcaseReplay?.steps ?? 0}-${showcaseReplay?.bricksCleared ?? 0}`}
                            speed={config.speed}
                        />
                    </div>
                    <NetworkPanel
                        eyebrow="Actor 策略"
                        genome={actorGenome ?? undefined}
                        input={liveInput}
                        inputLabels={BREAKER_INPUT_LABELS}
                        outputLabels={BREAKER_OUTPUT_LABELS}
                        subtitle="節點亮度跟住策略重播每一格嘅前向運算；熱圖係 actor 權重。"
                        title="PPO 策略網絡"
                        topology={BREAKER_TOPOLOGY}
                    />
                    <PpoTrainingChart history={history} />
                    <ApplicationPanel
                        eyebrow="PPO 對應"
                        fitness="向落緊嚟嘅球移近有即時獎勵；消磚 +2、接球 +3、跌球 -2、全清 +20，並有輕微時間成本"
                        fitnessLabel="獎勵"
                        genome="8 → 12 → 3 actor 策略權重；critic 另外估計每個狀態嘅長期回報"
                        genomeLabel="策略"
                        inputs="擋板/球位置、球速、最近磚塊方向、剩餘比例"
                        outputs="向左、停住、向右嘅機率分佈"
                        termination="球跌出底部、清晒 45 塊磚，或每回合 100,000 步上限；訓練／固定評估用真隨機場景；畫面每圈重抽一場"
                        title="點樣用 PPO 學撞磚"
                    />
                </main>
                <aside className="demo-sidebar">
                    <PpoControls config={config} error={error} onPause={pause} onReset={reset} onStart={start} onUpdate={updateConfig} stats={stats} status={status}>
                        <GenomeTransfer
                            disabled={status === "running"}
                            fitness={stats?.bestReturn}
                            genome={actorGenome}
                            onImport={handleImportGenome}
                            onMessage={setTransferMessage}
                            score={showcaseReplay?.bricksCleared}
                            steps={showcaseReplay?.steps}
                            topic="breaker-ppo"
                            topology={BREAKER_TOPOLOGY}
                        />
                        {transferMessage ? <p className={transferMessage.type === "error" ? "error-message" : "status-message"}>{transferMessage.text}</p> : null}
                    </PpoControls>
                </aside>
            </div>
        </DemoShell>
    );
});

interface PpoMetricsProps {
    replay: BreakerReplay | null;
    stats: PpoUpdateStats | null;
}

const PpoMetrics = React.memo<PpoMetricsProps>(({replay, stats}) => {
    const items = [
        {label: "更新輪次", value: String(stats?.update ?? 0)},
        {label: "固定評估", value: formatMetric(stats?.averageReturn)},
        {label: "歷史最佳", value: formatMetric(stats?.bestReturn)},
        {label: "策略 loss", value: formatMetric(stats?.policyLoss)},
        {label: "清磚數", value: String(replay?.bricksCleared ?? 0)},
        {label: "接球次數", value: String(replay?.hits ?? 0)},
    ];
    return (
        <div className="metrics-grid">
            {items.map(item => (
                <div className="metric" key={item.label}>
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                </div>
            ))}
        </div>
    );
});

interface PpoControlsProps {
    children?: React.ReactNode;
    config: PpoConfig;
    error: string | null;
    onPause: () => void;
    onReset: () => void;
    onStart: () => void;
    onUpdate: (key: keyof PpoConfig, value: number) => void;
    stats: PpoUpdateStats | null;
    status: TrainingStatus;
}

const PpoControls = React.memo<PpoControlsProps>(({children, config, error, onPause, onReset, onStart, onUpdate, stats, status}) => {
    const locked = status === "running";
    return (
        <Card className="control-panel rounded-lg" variant="default">
            <Card.Header className="flex-row items-center justify-between">
                <div>
                    <Card.Title className="text-base">訓練控制</Card.Title>
                    <Card.Description>調整下一輪 rollout 同更新速度</Card.Description>
                </div>
                <Chip color={status === "running" ? "success" : status === "paused" ? "warning" : "default"} size="sm" variant="soft">
                    {status === "running" ? "運行中" : status === "paused" ? "已暫停" : "待機"}
                </Chip>
            </Card.Header>
            <Card.Content className="space-y-5">
                <div className="grid grid-cols-3 gap-2">
                    <Button isDisabled={status === "running"} onPress={onStart} size="sm">
                        <Play size={15} strokeWidth={1.5} />
                        {status === "paused" ? "繼續" : "開始"}
                    </Button>
                    <Button isDisabled={status !== "running"} onPress={onPause} size="sm" variant="secondary">
                        <Pause size={15} strokeWidth={1.5} />
                        暫停
                    </Button>
                    <Tooltip delay={250}>
                        <Button onPress={onReset} size="sm" variant="tertiary">
                            <RotateCcw size={15} strokeWidth={1.5} />
                            重設
                        </Button>
                        <Tooltip.Content showArrow>清除 PPO 策略同訓練記錄。</Tooltip.Content>
                    </Tooltip>
                </div>
                <ControlSlider disabled={locked} label="每輪回合" max={12} min={4} onChange={value => onUpdate("episodesPerUpdate", value)} step={1} value={config.episodesPerUpdate} />
                <ControlSlider disabled={locked} label="學習率" max={0.003} min={0.0002} onChange={value => onUpdate("learningRate", value)} step={0.0002} value={config.learningRate} />
                <ControlSlider disabled={locked} label="播放／訓練速度" max={5} min={1} onChange={value => onUpdate("speed", value)} step={1} value={config.speed} />
                <label className="control-field">
                    <span className="control-label">隨機種子</span>
                    <input
                        aria-label="隨機種子"
                        className="number-input"
                        disabled={locked}
                        min={1}
                        onChange={event => onUpdate("seed", Number(event.target.value) || 1)}
                        type="number"
                        value={config.seed}
                    />
                </label>
                {children}
                <div className="ppo-fixed-grid">
                    <span>
                        clip <strong>{config.clipRatio.toFixed(2)}</strong>
                    </span>
                    <span>
                        gamma <strong>{config.gamma.toFixed(2)}</strong>
                    </span>
                    <span>
                        GAE <strong>{config.gaeLambda.toFixed(2)}</strong>
                    </span>
                    <span>
                        entropy <strong>{stats ? stats.entropy.toFixed(3) : "—"}</strong>
                    </span>
                    <span>
                        value loss <strong>{stats ? stats.valueLoss.toFixed(3) : "—"}</strong>
                    </span>
                    <span>
                        平均步數 <strong>{stats ? Math.round(stats.averageEpisodeLength) : "—"}</strong>
                    </span>
                </div>
                {error ? <p className="error-message">{error}</p> : null}
            </Card.Content>
        </Card>
    );
});

function formatMetric(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(2);
}

interface StoredPpoDemo {
    config: PpoConfig;
    champion?: Genome;
    bestFitness?: number;
}

function readStoredPpoDemo(): StoredPpoDemo | undefined {
    try {
        const state = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as PersistedLabStateV1 | null;
        const demo = state?.version === 1 ? state.demos[PPO_TOPIC] : undefined;
        if (!demo) {
            return undefined;
        }
        return {
            config: mergePpoConfig(demo.config),
            champion: demo.champion,
            bestFitness: demo.bestFitness,
        };
    } catch {
        return undefined;
    }
}

function writeStoredPpoDemo(config: PpoConfig, champion?: Genome, bestFitness?: number): void {
    try {
        const current = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as PersistedLabStateV1 | null;
        const state: PersistedLabStateV1 = current?.version === 1 ? current : {version: 1, demos: {}};
        // Same bag as GA demos; config JSON is lab-specific (PpoConfig here).
        state.demos[PPO_TOPIC] = {config: config as never, champion, bestFitness};
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
        // Private browsing or storage quotas should not stop a training session.
    }
}

function clearStoredPpoDemo(): void {
    try {
        const state = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as PersistedLabStateV1 | null;
        if (state?.version !== 1) {
            localStorage.removeItem(STORAGE_KEY);
            return;
        }
        delete state.demos[PPO_TOPIC];
        if (Object.keys(state.demos).length === 0) {
            localStorage.removeItem(STORAGE_KEY);
            return;
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
        // Storage can be unavailable in private browsing contexts.
    }
}

function mergePpoConfig(raw: unknown): PpoConfig {
    const base = {...DEFAULT_PPO_CONFIG, seed: Math.round(Math.random() * 1_000_000)};
    if (!raw || typeof raw !== "object") {
        return base;
    }
    const source = raw as Partial<PpoConfig>;
    return {
        episodesPerUpdate: typeof source.episodesPerUpdate === "number" ? source.episodesPerUpdate : base.episodesPerUpdate,
        maxSteps: typeof source.maxSteps === "number" ? source.maxSteps : base.maxSteps,
        epochs: typeof source.epochs === "number" ? source.epochs : base.epochs,
        learningRate: typeof source.learningRate === "number" ? source.learningRate : base.learningRate,
        gamma: typeof source.gamma === "number" ? source.gamma : base.gamma,
        gaeLambda: typeof source.gaeLambda === "number" ? source.gaeLambda : base.gaeLambda,
        clipRatio: typeof source.clipRatio === "number" ? source.clipRatio : base.clipRatio,
        entropyCoefficient: typeof source.entropyCoefficient === "number" ? source.entropyCoefficient : base.entropyCoefficient,
        seed: typeof source.seed === "number" ? source.seed : base.seed,
        speed: typeof source.speed === "number" ? source.speed : base.speed,
    };
}
