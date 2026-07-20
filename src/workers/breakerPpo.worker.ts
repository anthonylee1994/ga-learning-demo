/// <reference lib="webworker" />

import {createPpoTrainer, disposePpoTrainer, loadPpoActorGenome, trainPpoUpdate, type PpoConfig, type PpoTrainer, type PpoUpdateResult} from "../domains/breaker/ppo";
import type {Genome} from "../lib/types";

export type BreakerPpoWorkerCommand = {type: "start"; config: PpoConfig; genome?: Genome} | {type: "pause"} | {type: "load"; config: PpoConfig; genome: Genome} | {type: "reset"};

export type BreakerPpoWorkerEvent = {type: "update"; result: PpoUpdateResult} | {type: "loaded"; result: PpoUpdateResult} | {type: "paused"} | {type: "reset"} | {type: "error"; message: string};

const scope = self as DedicatedWorkerGlobalScope;
let trainer: PpoTrainer | null = null;
let config: PpoConfig | null = null;
let running = false;
let runToken = 0;
let scheduledHandle: ReturnType<typeof scope.setTimeout> | null = null;
let lastEmittedAt = 0;
let lastEmittedUpdate = 0;
let latestResult: PpoUpdateResult | null = null;

scope.onmessage = function handleMessage(event: MessageEvent<BreakerPpoWorkerCommand>) {
    const command = event.data;
    if (command.type === "start") {
        config = command.config;
        if (!trainer) {
            trainer = createPpoTrainer(config.seed, config.learningRate);
            // Restore persisted / imported actor so refresh + continue keeps the same network.
            if (command.genome?.length) {
                try {
                    latestResult = loadPpoActorGenome(trainer, command.genome, config);
                    lastEmittedUpdate = 0;
                } catch (error) {
                    disposePpoTrainer(trainer);
                    trainer = null;
                    emit({type: "error", message: error instanceof Error ? error.message : "PPO 策略載入失敗。"});
                    return;
                }
            }
        }
        running = true;
        runToken += 1;
        scheduleNext(runToken, 0);
        return;
    }
    if (command.type === "pause") {
        running = false;
        runToken += 1;
        clearScheduled();
        emitLatestResult();
        emit({type: "paused"});
        return;
    }
    if (command.type === "load") {
        loadActor(command.genome, command.config);
        return;
    }
    running = false;
    runToken += 1;
    clearScheduled();
    if (trainer) {
        disposePpoTrainer(trainer);
        trainer = null;
    }
    config = null;
    latestResult = null;
    lastEmittedAt = 0;
    lastEmittedUpdate = 0;
    emit({type: "reset"});
};

function loadActor(genome: Genome, nextConfig: PpoConfig): void {
    running = false;
    runToken += 1;
    clearScheduled();
    try {
        if (trainer) {
            disposePpoTrainer(trainer);
        }
        config = nextConfig;
        trainer = createPpoTrainer(config.seed, config.learningRate);
        const result = loadPpoActorGenome(trainer, genome, config);
        latestResult = result;
        lastEmittedAt = Date.now();
        lastEmittedUpdate = 0;
        emit({type: "loaded", result});
    } catch (error) {
        if (trainer) {
            disposePpoTrainer(trainer);
            trainer = null;
        }
        emit({type: "error", message: error instanceof Error ? error.message : "PPO 策略載入失敗。"});
    }
}

function emit(event: BreakerPpoWorkerEvent): void {
    scope.postMessage(event);
}

function clearScheduled(): void {
    if (scheduledHandle !== null) {
        scope.clearTimeout(scheduledHandle);
        scheduledHandle = null;
    }
}

function scheduleNext(token: number, delay?: number): void {
    clearScheduled();
    scheduledHandle = scope.setTimeout(
        function scheduleTraining() {
            scheduledHandle = null;
            void runUpdate(token);
        },
        delay ?? Math.max(40, (5 - (config?.speed ?? 5)) * 80)
    );
}

async function runUpdate(token: number): Promise<void> {
    if (!running || token !== runToken || !trainer || !config) {
        return;
    }
    try {
        const result = await trainPpoUpdate(trainer, config);
        if (!running || token !== runToken) {
            return;
        }
        latestResult = result;
        if (Date.now() - lastEmittedAt >= 80) {
            emitLatestResult();
        }
        scheduleNext(token);
    } catch (error) {
        running = false;
        emit({type: "error", message: error instanceof Error ? error.message : "PPO 訓練失敗。"});
    }
}

function emitLatestResult(): void {
    if (!latestResult || latestResult.stats.update === lastEmittedUpdate) {
        return;
    }
    emit({type: "update", result: latestResult});
    lastEmittedAt = Date.now();
    lastEmittedUpdate = latestResult.stats.update;
}
