/// <reference lib="webworker" />

import {createPopulation, evolvePopulation} from "../lib/ga";
import {createRandom} from "../lib/random";
import type {GAConfig, GenerationStats, Genome, WorkerCommand, WorkerEvent} from "../lib/types";

interface WorkerDefinition<TData, TReplay> {
    geneCount: number;
    requiresData?: boolean;
    evaluate(genome: Genome, data: TData | undefined): number;
    createReplay(genome: Genome, data: TData | undefined): TReplay;
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
    let bestGenome: Genome | null = null;
    let lastStats: GenerationStats | null = null;
    let scheduledHandle: ReturnType<typeof scope.setTimeout> | null = null;

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
        const delay = config ? Math.max(0, 220 - config.speed * 35) : 0;
        clearScheduled();
        scheduledHandle = scope.setTimeout(() => {
            scheduledHandle = null;
            runGeneration(token);
        }, delay);
    }

    function resetEvolutionState(): void {
        population = [];
        generation = 0;
        bestFitnessSeen = Number.NEGATIVE_INFINITY;
        lastReplayFitness = Number.NEGATIVE_INFINITY;
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
            const fitnesses = population.map(genome => definition.evaluate(genome, data));
            // User may have paused mid-evaluation; keep population as-is and wait for pause handler
            // to emit the showcase champion.
            if (!running || token !== runToken || !config) {
                return;
            }

            const result = evolvePopulation(population, fitnesses, config, random);
            population = result.population;
            generation += 1;

            if (!running || token !== runToken) {
                return;
            }

            // Only rebuild champion replay on a meaningful improvement vs the last replay.
            // Tiny survival/noise bumps used to regenerate full replays every few seconds and
            // restart snake/breaker animations while scores looked stuck.
            const REPLAY_IMPROVEMENT_EPS = 5;
            if (result.bestFitness > bestFitnessSeen) {
                bestFitnessSeen = result.bestFitness;
                bestGenome = [...result.bestGenome];
            } else if (!bestGenome) {
                bestGenome = [...result.bestGenome];
                bestFitnessSeen = result.bestFitness;
            }

            const shouldRefreshReplay = lastReplayFitness === Number.NEGATIVE_INFINITY || result.bestFitness >= lastReplayFitness + REPLAY_IMPROVEMENT_EPS;
            if (shouldRefreshReplay) {
                lastReplayFitness = result.bestFitness;
            }

            lastStats = {generation, ...result.stats};
            emit({
                type: "generation",
                stats: lastStats,
                champion: {
                    genome: result.bestGenome,
                    fitness: result.bestFitness,
                    ...(shouldRefreshReplay ? {replay: definition.createReplay(result.bestGenome, data)} : {}),
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
                replay: definition.createReplay(bestGenome, data),
            },
        });
        lastReplayFitness = fitness;
    }

    scope.onmessage = (message: MessageEvent<WorkerCommand<TData>>) => {
        const command = message.data;
        if (command.type === "start") {
            runToken += 1;
            clearScheduled();
            config = command.config;
            data = command.data ?? data;
            random = createRandom(config.seed + generation * 7919);
            if (population.length !== config.populationSize) {
                population = createPopulation(config.populationSize, definition.geneCount, random, command.champion);
                generation = 0;
                bestFitnessSeen = Number.NEGATIVE_INFINITY;
                lastReplayFitness = Number.NEGATIVE_INFINITY;
                bestGenome = command.champion ? [...command.champion] : null;
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
        config = command.config;
    };
}
