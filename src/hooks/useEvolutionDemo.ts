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

export interface LoadChampionPayload<TReplay> {
    genome: number[];
    replay: TReplay;
    fitness?: number;
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
    /**
     * Inject a champion genome + replay (e.g. imported weights).
     * Locks champion updates until the next start() so pause-showcase cannot overwrite it.
     */
    loadChampion: (payload: LoadChampionPayload<TReplay>) => void;
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
    const skipNextConfigPersistRef = React.useRef(false);
    /** Block live training champion thrash while paused / waiting for showcase. */
    const suppressChampionUpdatesRef = React.useRef(false);
    /** Next generation payload with a full replay is the pause showcase snapshot. */
    const awaitingPauseShowcaseRef = React.useRef(false);
    /** Hard-lock champion after import until the user starts training again. */
    const lockChampionRef = React.useRef(false);

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
                    lockChampionRef.current = false;
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
                // Stats/history are cheap; champion replays (stock series) are huge. Keep the
                // generation ticker responsive without blocking paint on structured-clone payloads.
                React.startTransition(() => {
                    setStats(event.stats);
                    setHistory(current => [...current.slice(-79), event.stats]);

                    if (lockChampionRef.current) {
                        return;
                    }

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
                            // Same fitness + same genes → keep the previous object so memoized
                            // chart subtrees (MarketChart / EquityChart) never see a new champion.
                            if (current.fitness === next.fitness && genomesEqual(current.genome, next.genome)) {
                                return current;
                            }
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
                });
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
        if (skipNextConfigPersistRef.current) {
            skipNextConfigPersistRef.current = false;
        } else {
            schedulePersist(config, storedChampionRef.current);
        }
        if (workerRef.current) {
            const command: WorkerCommand<TData> = {type: "update-config", config};
            workerRef.current.postMessage(command);
        }
    }, [config, schedulePersist, topic]);

    const start = () => {
        setError(null);
        suppressChampionUpdatesRef.current = false;
        awaitingPauseShowcaseRef.current = false;
        lockChampionRef.current = false;
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
        lockChampionRef.current = false;
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
        skipNextConfigPersistRef.current = config !== defaultConfig;
        clearStoredDemo(topic);
        setConfig(defaultConfig);
    };
    const loadChampion = (payload: LoadChampionPayload<TReplay>) => {
        // If training is mid-flight, stop accepting worker champion updates (including
        // pause-showcase) so the imported brain is not immediately overwritten.
        if (status === "running") {
            awaitingPauseShowcaseRef.current = false;
            suppressChampionUpdatesRef.current = true;
            workerRef.current?.postMessage({type: "pause"} satisfies WorkerCommand<TData>);
        }
        lockChampionRef.current = true;
        awaitingPauseShowcaseRef.current = false;
        suppressChampionUpdatesRef.current = true;

        const fitness = payload.fitness ?? 0;
        storedChampionRef.current = payload.genome;
        setChampion({genome: payload.genome, fitness, replay: payload.replay});
        setShowcaseEpoch(value => value + 1);
        // Paused so canvas loops the imported champion (idle would freeze playback).
        setStatus("paused");
        setError(null);
        schedulePersist(configRef.current, payload.genome, fitness);
    };

    return {config, setConfig, status, stats, history, champion, error, showcaseEpoch, start, pause, reset, loadChampion};
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

function clearStoredDemo(topic: DemoTopic): void {
    try {
        const state = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as PersistedLabStateV1 | null;
        if (state?.version !== 1) {
            localStorage.removeItem(STORAGE_KEY);
            return;
        }
        delete state.demos[topic];
        if (Object.keys(state.demos).length === 0) {
            localStorage.removeItem(STORAGE_KEY);
            return;
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
        // Storage can be unavailable in private browsing contexts.
    }
}

function genomesEqual(a: number[], b: number[]): boolean {
    if (a === b) {
        return true;
    }
    if (a.length !== b.length) {
        return false;
    }
    for (let index = 0; index < a.length; index += 1) {
        if (a[index] !== b[index]) {
            return false;
        }
    }
    return true;
}
