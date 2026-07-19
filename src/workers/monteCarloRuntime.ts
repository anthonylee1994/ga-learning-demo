/// <reference lib="webworker" />

import {type MutationProfile} from "../lib/ga";
import {runMonteCarloBatch} from "../lib/monteCarlo";
import {createRandom} from "../lib/random";
import type {GAConfig, GenerationStats, Genome, WorkerCommand, WorkerEvent} from "../lib/types";

interface MonteCarloWorkerDefinition<TData, TReplay> {
    geneCount: number;
    requiresData?: boolean;
    /** Minimum batches between full champion replays (stock series are multi-MB). */
    minReplayGenerationGap?: number;
    seedGenomes?: Genome[];
    mutationProfile?: MutationProfile;
    evaluate(genome: Genome, data: TData | undefined, config: GAConfig | null): number;
    createReplay(genome: Genome, data: TData | undefined, config: GAConfig | null, purpose: "progress" | "showcase"): TReplay;
}

/**
 * 蒙地卡羅 worker protocol mirrors the GA worker so useEvolutionDemo works unchanged.
 * Each "generation" is one random-search batch (global draws + local walks).
 */
export function setupMonteCarloWorker<TData, TReplay>(definition: MonteCarloWorkerDefinition<TData, TReplay>): void {
    const scope = self as DedicatedWorkerGlobalScope;
    let config: GAConfig | null = null;
    let generation = 0;
    let running = false;
    let data: TData | undefined;
    let random = createRandom(1);
    let runToken = 0;
    let bestFitnessSeen = Number.NEGATIVE_INFINITY;
    let lastReplayFitness = Number.NEGATIVE_INFINITY;
    let lastReplayGeneration = Number.NEGATIVE_INFINITY;
    let bestGenome: Genome | null = null;
    let lastStats: GenerationStats | null = null;
    let scheduledHandle: ReturnType<typeof scope.setTimeout> | null = null;
    const MIN_REPLAY_GENERATION_GAP = definition.minReplayGenerationGap ?? 5;

    function emit(event: WorkerEvent<TReplay>): void {
        scope.postMessage(event);
    }

    function clearScheduled(): void {
        if (scheduledHandle !== null) {
            scope.clearTimeout(scheduledHandle);
            scheduledHandle = null;
        }
    }

    function scheduleNext(token: number): void {
        const delay = config ? Math.max(0, (5 - config.speed) * 40) : 0;
        clearScheduled();
        scheduledHandle = scope.setTimeout(() => {
            scheduledHandle = null;
            runBatch(token);
        }, delay);
    }

    function resetGatesIfModeChanged(next: GAConfig): void {
        if (config && config.useNeuralNetwork !== next.useNeuralNetwork) {
            bestFitnessSeen = Number.NEGATIVE_INFINITY;
            lastReplayFitness = Number.NEGATIVE_INFINITY;
            lastReplayGeneration = Number.NEGATIVE_INFINITY;
        }
    }

    function resetState(): void {
        generation = 0;
        bestFitnessSeen = Number.NEGATIVE_INFINITY;
        lastReplayFitness = Number.NEGATIVE_INFINITY;
        lastReplayGeneration = Number.NEGATIVE_INFINITY;
        bestGenome = null;
        lastStats = null;
    }

    function runBatch(token: number): void {
        if (!running || !config || token !== runToken) {
            return;
        }
        if (definition.requiresData && !data) {
            running = false;
            emit({type: "error", message: "未有可用訓練數據。"});
            return;
        }

        try {
            const previousBest = bestGenome && bestFitnessSeen !== Number.NEGATIVE_INFINITY ? {genome: bestGenome, fitness: bestFitnessSeen} : null;

            const result = runMonteCarloBatch(definition.geneCount, genome => definition.evaluate(genome, data, config), config, random, previousBest, {
                mutationProfile: definition.mutationProfile,
                seedGenomes: definition.seedGenomes,
                generation,
            });

            if (!running || token !== runToken || !config) {
                return;
            }

            generation += 1;

            if (result.bestFitness > bestFitnessSeen || !bestGenome) {
                bestFitnessSeen = result.bestFitness;
                bestGenome = [...result.bestGenome];
            }

            const REPLAY_IMPROVEMENT_EPS = Math.max(Math.abs(lastReplayFitness) * 0.02, 0.01);
            const LARGE_REPLAY_IMPROVEMENT = Math.max(Math.abs(lastReplayFitness) * 0.12, 1);
            const improved = result.bestFitness >= lastReplayFitness + REPLAY_IMPROVEMENT_EPS;
            const largeJump = result.bestFitness >= lastReplayFitness + LARGE_REPLAY_IMPROVEMENT;
            const gapOk = generation - lastReplayGeneration >= MIN_REPLAY_GENERATION_GAP;
            const shouldRefreshReplay = lastReplayFitness === Number.NEGATIVE_INFINITY || (improved && (gapOk || largeJump));
            if (shouldRefreshReplay) {
                lastReplayFitness = result.bestFitness;
                lastReplayGeneration = generation;
            }

            lastStats = {generation, ...result.stats};
            emit({
                type: "generation",
                stats: lastStats,
                champion: {
                    genome: bestGenome,
                    fitness: bestFitnessSeen,
                    ...(shouldRefreshReplay ? {replay: definition.createReplay(bestGenome, data, config, "progress")} : {}),
                },
            });
            scheduleNext(token);
        } catch (error) {
            running = false;
            emit({type: "error", message: error instanceof Error ? error.message : "蒙地卡羅搜尋失敗。"});
        }
    }

    function emitShowcaseChampion(): void {
        if (!bestGenome) {
            return;
        }
        const fitness = bestFitnessSeen === Number.NEGATIVE_INFINITY ? 0 : bestFitnessSeen;
        const stats: GenerationStats = lastStats ?? {
            generation,
            bestFitness: fitness,
            averageFitness: fitness,
            diversity: 0,
        };
        emit({
            type: "generation",
            reason: "pause-showcase",
            stats,
            champion: {
                genome: bestGenome,
                fitness,
                replay: definition.createReplay(bestGenome, data, config, "showcase"),
            },
        });
        lastReplayFitness = fitness;
        lastReplayGeneration = generation;
    }

    scope.onmessage = (message: MessageEvent<WorkerCommand<TData>>) => {
        const command = message.data;
        if (command.type === "start") {
            runToken += 1;
            clearScheduled();
            resetGatesIfModeChanged(command.config);
            config = command.config;
            data = command.data ?? data;
            // Advance RNG with generation so resume keeps exploring new samples.
            random = createRandom(config.seed + generation * 7919);
            const compatibleChampion = command.champion?.length === definition.geneCount ? command.champion : undefined;
            if (compatibleChampion) {
                bestGenome = [...compatibleChampion];
                // Fitness unknown until first batch re-evaluates.
                if (bestFitnessSeen === Number.NEGATIVE_INFINITY) {
                    bestFitnessSeen = definition.evaluate(compatibleChampion, data, config);
                }
            }
            running = true;
            emit({type: "status", status: "running"});
            runBatch(runToken);
            return;
        }
        if (command.type === "pause") {
            runToken += 1;
            clearScheduled();
            running = false;
            emitShowcaseChampion();
            emit({type: "status", status: "paused"});
            return;
        }
        if (command.type === "reset") {
            runToken += 1;
            clearScheduled();
            running = false;
            resetState();
            emit({type: "status", status: "idle"});
            return;
        }
        if (command.type === "set-data") {
            data = command.data;
            resetState();
            emit({type: "status", status: "idle"});
            return;
        }
        resetGatesIfModeChanged(command.config);
        config = command.config;
    };
}
