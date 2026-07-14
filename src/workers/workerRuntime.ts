/// <reference lib="webworker" />

import {createPopulation, evolvePopulation, type MutationProfile} from "../lib/ga";
import {createRandom} from "../lib/random";
import type {GAConfig, GenerationStats, Genome, WorkerCommand, WorkerEvent} from "../lib/types";

interface WorkerDefinition<TData, TReplay> {
    geneCount: number;
    requiresData?: boolean;
    /**
     * Minimum generations between full champion replays. Stock series / long snake
     * frame buffers are multi-MB structured clones — refresh sparingly.
     */
    minReplayGenerationGap?: number;
    /** Optional classic baselines inserted into a fresh population (e.g. SMA cross). */
    seedGenomes?: Genome[];
    /** Optional head/tail mutation bias (stock lab prioritizes indicator periods). */
    mutationProfile?: MutationProfile;
    evaluate(genome: Genome, data: TData | undefined, config: GAConfig | null): number;
    createReplay(genome: Genome, data: TData | undefined, config: GAConfig | null): TReplay;
}

export function setupEvolutionWorker<TData, TReplay>(definition: WorkerDefinition<TData, TReplay>): void {
    const scope = self as DedicatedWorkerGlobalScope;
    let config: GAConfig | null = null;
    let population: Genome[] = [];
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
    /** Avoid structured-cloning huge replays every generation when fitness crawls upward. */
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
        // speed 1 → ~160ms gap (readable UI); speed 5 → 0ms (train as fast as the worker can).
        // Always go through setTimeout so pause/reset messages can interleave even at full tilt.
        const delay = config ? Math.max(0, (5 - config.speed) * 40) : 0;
        clearScheduled();
        scheduledHandle = scope.setTimeout(() => {
            scheduledHandle = null;
            runGeneration(token);
        }, delay);
    }

    /**
     * Flipping the decision mode (NN ↔ rules) rescales fitness entirely — stale
     * bests would block replay refreshes, so reset the gates but keep the population.
     */
    function resetGatesIfModeChanged(next: GAConfig): void {
        if (config && config.useNeuralNetwork !== next.useNeuralNetwork) {
            bestFitnessSeen = Number.NEGATIVE_INFINITY;
            lastReplayFitness = Number.NEGATIVE_INFINITY;
            lastReplayGeneration = Number.NEGATIVE_INFINITY;
        }
    }

    function resetEvolutionState(): void {
        population = [];
        generation = 0;
        bestFitnessSeen = Number.NEGATIVE_INFINITY;
        lastReplayFitness = Number.NEGATIVE_INFINITY;
        lastReplayGeneration = Number.NEGATIVE_INFINITY;
        bestGenome = null;
        lastStats = null;
    }

    function runGeneration(token: number): void {
        if (!running || !config || token !== runToken) {
            return;
        }
        if (definition.requiresData && !data) {
            running = false;
            emit({type: "error", message: "未有可用訓練數據。"});
            return;
        }

        try {
            const fitnesses = population.map(genome => definition.evaluate(genome, data, config));
            // User may have paused mid-evaluation; keep population as-is and wait for pause handler
            // to emit the showcase champion.
            if (!running || token !== runToken || !config) {
                return;
            }

            const result = evolvePopulation(population, fitnesses, config, random, {
                mutationProfile: definition.mutationProfile,
            });
            population = result.population;
            generation += 1;

            if (!running || token !== runToken) {
                return;
            }

            // Only rebuild champion replay on a meaningful improvement vs the last replay.
            // Tiny survival/noise bumps used to regenerate full replays every few seconds and
            // restart snake/breaker animations while scores looked stuck. Relative to the current
            // fitness scale so it works for both game scores (~hundreds) and Sharpe ratios (~units).
            // Also require a generation gap — full replays are huge (stock series / snake frames)
            // and structured-clone via postMessage is a major memory spike.
            const REPLAY_IMPROVEMENT_EPS = Math.max(Math.abs(lastReplayFitness) * 0.02, 0.01);
            const LARGE_REPLAY_IMPROVEMENT = Math.max(Math.abs(lastReplayFitness) * 0.12, 1);
            if (result.bestFitness > bestFitnessSeen) {
                bestFitnessSeen = result.bestFitness;
                bestGenome = [...result.bestGenome];
            } else if (!bestGenome) {
                bestGenome = [...result.bestGenome];
                bestFitnessSeen = result.bestFitness;
            }

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
                    genome: result.bestGenome,
                    fitness: result.bestFitness,
                    ...(shouldRefreshReplay ? {replay: definition.createReplay(result.bestGenome, data, config)} : {}),
                },
            });
            scheduleNext(token);
        } catch (error) {
            running = false;
            emit({type: "error", message: error instanceof Error ? error.message : "訓練失敗。"});
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
                // Always rebuild so pause shows the true latest optimized network end-to-end.
                replay: definition.createReplay(bestGenome, data, config),
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
            random = createRandom(config.seed + generation * 7919);
            if (population.length !== config.populationSize) {
                const compatibleChampion = command.champion?.length === definition.geneCount ? command.champion : undefined;
                population = createPopulation(config.populationSize, definition.geneCount, random, compatibleChampion);
                // Inject classic baselines after the optional champion slot so GA has coherent
                // starting strategies (stock TA especially needs this — random thresholds thrash).
                if (definition.seedGenomes?.length) {
                    const offset = compatibleChampion ? 1 : 0;
                    definition.seedGenomes.forEach((seed, index) => {
                        const slot = offset + index;
                        if (slot < population.length && seed.length === definition.geneCount) {
                            population[slot] = [...seed];
                        }
                    });
                }
                generation = 0;
                bestFitnessSeen = Number.NEGATIVE_INFINITY;
                lastReplayFitness = Number.NEGATIVE_INFINITY;
                lastReplayGeneration = Number.NEGATIVE_INFINITY;
                bestGenome = compatibleChampion ? [...compatibleChampion] : null;
                lastStats = null;
            }
            running = true;
            emit({type: "status", status: "running"});
            runGeneration(runToken);
            return;
        }
        if (command.type === "pause") {
            runToken += 1;
            clearScheduled();
            running = false;
            // Showcase the latest best network: full replay from kickoff until loss.
            emitShowcaseChampion();
            emit({type: "status", status: "paused"});
            return;
        }
        if (command.type === "reset") {
            runToken += 1;
            clearScheduled();
            running = false;
            resetEvolutionState();
            emit({type: "status", status: "idle"});
            return;
        }
        if (command.type === "set-data") {
            data = command.data;
            resetEvolutionState();
            emit({type: "status", status: "idle"});
            return;
        }
        resetGatesIfModeChanged(command.config);
        config = command.config;
    };
}
