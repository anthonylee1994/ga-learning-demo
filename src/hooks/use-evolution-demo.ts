import React from "react";
import type {Champion, GAConfig, GenerationStats, PersistedLabStateV1, TopicId, WorkerCommand, WorkerEvent} from "../lib/types";

type DemoTopic = Exclude<TopicId, "theory">;
type TrainingStatus = "idle" | "running" | "paused";

interface EvolutionDemoOptions<TData> {
    topic: DemoTopic;
    createWorker: () => Worker;
    defaultConfig: GAConfig;
    data?: TData;
}

export interface EvolutionDemoState<TReplay> {
    config: GAConfig;
    setConfig: React.Dispatch<React.SetStateAction<GAConfig>>;
    status: TrainingStatus;
    stats: GenerationStats | null;
    history: GenerationStats[];
    champion: Champion<TReplay> | null;
    error: string | null;
    /** Bumps when a pause-showcase champion arrives — canvas should restart from frame 0. */
    showcaseEpoch: number;
    start: () => void;
    pause: () => void;
    reset: () => void;
}

const STORAGE_KEY = "evolab-state-v1";
const PERSIST_DEBOUNCE_MS = 750;

export function useEvolutionDemo<TData, TReplay>(options: EvolutionDemoOptions<TData>): EvolutionDemoState<TReplay> {
    const {topic, defaultConfig, data} = options;
    const stored = React.useMemo(() => readStoredDemo(topic), [topic]);
    const [config, setConfig] = React.useState<GAConfig>(stored?.config ?? options.defaultConfig);
    const [status, setStatus] = React.useState<TrainingStatus>("idle");
    const [stats, setStats] = React.useState<GenerationStats | null>(null);
    const [history, setHistory] = React.useState<GenerationStats[]>([]);
    const [champion, setChampion] = React.useState<Champion<TReplay> | null>(null);
    const [error, setError] = React.useState<string | null>(null);
    const [showcaseEpoch, setShowcaseEpoch] = React.useState(0);
    const workerRef = React.useRef<Worker | null>(null);
    const createWorkerRef = React.useRef(options.createWorker);
    const configRef = React.useRef(config);
    const storedChampionRef = React.useRef(stored?.champion);
    const persistTimerRef = React.useRef<number | null>(null);
    /** Block live training champion thrash while paused / waiting for showcase. */
    const suppressChampionUpdatesRef = React.useRef(false);
    /** Next generation payload with a full replay is the pause showcase snapshot. */
    const awaitingPauseShowcaseRef = React.useRef(false);

    configRef.current = config;

    const schedulePersist = React.useCallback(
        (nextConfig: GAConfig, genome?: number[], bestFitness?: number) => {
            if (persistTimerRef.current !== null) {
                window.clearTimeout(persistTimerRef.current);
            }
            persistTimerRef.current = window.setTimeout(() => {
                persistTimerRef.current = null;
                writeStoredDemo(topic, nextConfig, genome, bestFitness);
            }, PERSIST_DEBOUNCE_MS) as unknown as number;
        },
        [topic]
    );

    React.useEffect(() => {
        const worker = createWorkerRef.current();
        workerRef.current = worker;
        worker.onmessage = (message: MessageEvent<WorkerEvent<TReplay>>) => {
            const event = message.data;
            if (event.type === "status") {
                setStatus(event.status);
                if (event.status === "running") {
                    suppressChampionUpdatesRef.current = false;
                    awaitingPauseShowcaseRef.current = false;
                }
                if (event.status === "paused" && awaitingPauseShowcaseRef.current) {
                    // No champion yet (paused before first generation) — stop waiting.
                    awaitingPauseShowcaseRef.current = false;
                }
            } else if (event.type === "error") {
                setError(event.message);
                setStatus("paused");
                suppressChampionUpdatesRef.current = true;
                awaitingPauseShowcaseRef.current = false;
            } else {
                setStats(event.stats);
                setHistory(current => [...current.slice(-79), event.stats]);

                const isPauseShowcase = event.reason === "pause-showcase" && event.champion.replay !== undefined;
                if (suppressChampionUpdatesRef.current && !isPauseShowcase) {
                    return;
                }

                setChampion(current => {
                    const next = event.champion;
                    storedChampionRef.current = next.genome;
                    if (next.replay !== undefined) {
                        return {genome: next.genome, fitness: next.fitness, replay: next.replay};
                    }
                    if (current) {
                        return {genome: next.genome, fitness: next.fitness, replay: current.replay};
                    }
                    return null;
                });
                schedulePersist(configRef.current, event.champion.genome, event.champion.fitness);

                if (isPauseShowcase) {
                    awaitingPauseShowcaseRef.current = false;
                    suppressChampionUpdatesRef.current = true;
                    setShowcaseEpoch(value => value + 1);
                }
            }
        };
        return () => {
            worker.terminate();
            if (persistTimerRef.current !== null) {
                window.clearTimeout(persistTimerRef.current);
                persistTimerRef.current = null;
            }
        };
    }, [schedulePersist, topic]);

    React.useEffect(() => {
        if (data && workerRef.current) {
            const command: WorkerCommand<TData> = {type: "set-data", data};
            workerRef.current.postMessage(command);
            setStats(null);
            setHistory([]);
            setChampion(null);
        }
    }, [data]);

    React.useEffect(() => {
        schedulePersist(config, storedChampionRef.current);
        if (workerRef.current) {
            const command: WorkerCommand<TData> = {type: "update-config", config};
            workerRef.current.postMessage(command);
        }
    }, [config, schedulePersist, topic]);

    const start = () => {
        setError(null);
        suppressChampionUpdatesRef.current = false;
        awaitingPauseShowcaseRef.current = false;
        const command: WorkerCommand<TData> = {
            type: "start",
            config,
            data,
            champion: champion?.genome ?? storedChampionRef.current,
        };
        workerRef.current?.postMessage(command);
    };
    const pause = () => {
        // Accept one full champion+replay snapshot from the worker (latest optimized net),
        // then ignore further thrash so the showcase can play through until loss.
        awaitingPauseShowcaseRef.current = true;
        suppressChampionUpdatesRef.current = true;
        setStatus("paused");
        workerRef.current?.postMessage({type: "pause"} satisfies WorkerCommand<TData>);
    };
    const reset = () => {
        suppressChampionUpdatesRef.current = false;
        awaitingPauseShowcaseRef.current = false;
        workerRef.current?.postMessage({type: "reset"} satisfies WorkerCommand<TData>);
        storedChampionRef.current = undefined;
        setStatus("idle");
        setStats(null);
        setHistory([]);
        setChampion(null);
        setError(null);
        if (persistTimerRef.current !== null) {
            window.clearTimeout(persistTimerRef.current);
            persistTimerRef.current = null;
        }
        writeStoredDemo(topic, defaultConfig);
        setConfig(defaultConfig);
    };

    return {config, setConfig, status, stats, history, champion, error, showcaseEpoch, start, pause, reset};
}

function readStoredDemo(topic: DemoTopic) {
    try {
        const state = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as PersistedLabStateV1 | null;
        return state?.version === 1 ? state.demos[topic] : undefined;
    } catch {
        return undefined;
    }
}

function writeStoredDemo(topic: DemoTopic, config: GAConfig, champion?: number[], bestFitness?: number): void {
    try {
        const current = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as PersistedLabStateV1 | null;
        const state: PersistedLabStateV1 = current?.version === 1 ? current : {version: 1, demos: {}};
        state.demos[topic] = {config, champion, bestFitness};
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
        // Private browsing or storage quotas should not stop a training session.
    }
}
