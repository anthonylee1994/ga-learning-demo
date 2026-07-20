/// <reference lib="webworker" />

import {createRainbowTrainer, disposeRainbowTrainer, loadRainbowAgentGenome, trainRainbowUpdate, type RainbowConfig, type RainbowTrainer, type RainbowUpdateResult} from "../domains/breaker/rainbow";
import type {Genome} from "../lib/types";

export type BreakerRainbowWorkerCommand = {type: "start"; config: RainbowConfig; genome?: Genome} | {type: "pause"} | {type: "load"; config: RainbowConfig; genome: Genome} | {type: "reset"};

export type BreakerRainbowWorkerEvent =
    {type: "update"; result: RainbowUpdateResult} | {type: "loaded"; result: RainbowUpdateResult} | {type: "paused"} | {type: "reset"} | {type: "error"; message: string};

const scope = self as DedicatedWorkerGlobalScope;
let trainer: RainbowTrainer | null = null;
let config: RainbowConfig | null = null;
let running = false;
let runToken = 0;
let scheduledHandle: ReturnType<typeof scope.setTimeout> | null = null;
let lastEmittedAt = 0;
let lastEmittedUpdate = 0;
let latestResult: RainbowUpdateResult | null = null;

scope.onmessage = function handleMessage(event: MessageEvent<BreakerRainbowWorkerCommand>) {
    const command = event.data;
    if (command.type === "start") {
        config = command.config;
        if (!trainer) {
            trainer = createRainbowTrainer(config.seed, config.learningRate, config.bufferSize);
            // Restore persisted / imported agent so refresh + continue keeps the same network.
            if (command.genome?.length) {
                try {
                    latestResult = loadRainbowAgentGenome(trainer, command.genome, config);
                    lastEmittedUpdate = 0;
                } catch (error) {
                    disposeRainbowTrainer(trainer);
                    trainer = null;
                    emit({type: "error", message: error instanceof Error ? error.message : "Rainbow 策略載入失敗。"});
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
        loadAgent(command.genome, command.config);
        return;
    }
    running = false;
    runToken += 1;
    clearScheduled();
    if (trainer) {
        disposeRainbowTrainer(trainer);
        trainer = null;
    }
    config = null;
    latestResult = null;
    lastEmittedAt = 0;
    lastEmittedUpdate = 0;
    emit({type: "reset"});
};

function loadAgent(genome: Genome, nextConfig: RainbowConfig): void {
    running = false;
    runToken += 1;
    clearScheduled();
    try {
        if (trainer) {
            disposeRainbowTrainer(trainer);
        }
        config = nextConfig;
        trainer = createRainbowTrainer(config.seed, config.learningRate, config.bufferSize);
        const result = loadRainbowAgentGenome(trainer, genome, config);
        latestResult = result;
        lastEmittedAt = Date.now();
        lastEmittedUpdate = 0;
        emit({type: "loaded", result});
    } catch (error) {
        if (trainer) {
            disposeRainbowTrainer(trainer);
            trainer = null;
        }
        emit({type: "error", message: error instanceof Error ? error.message : "Rainbow 策略載入失敗。"});
    }
}

function emit(event: BreakerRainbowWorkerEvent): void {
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
        const result = await trainRainbowUpdate(trainer, config);
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
        emit({type: "error", message: error instanceof Error ? error.message : "Rainbow 訓練失敗。"});
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
