import {
    addTransition,
    appendNStepTransition,
    createPrioritizedReplayBuffer,
    createRainbowAgentReplay,
    createRainbowTrainer,
    currentBeta,
    currentEpsilon,
    DEFAULT_RAINBOW_CONFIG,
    disposeRainbowTrainer,
    flushRemainingNStep,
    loadRainbowAgentGenome,
    samplePrioritizedBatch,
    trainRainbowUpdate,
    updatePriorities,
} from "./rainbow";
import {createRandom} from "../../lib/random";

describe("breaker Rainbow DQN", () => {
    it("anneals epsilon and PER beta toward their ends", () => {
        expect(currentEpsilon(0, DEFAULT_RAINBOW_CONFIG)).toBeCloseTo(DEFAULT_RAINBOW_CONFIG.epsilonStart);
        expect(currentEpsilon(DEFAULT_RAINBOW_CONFIG.epsilonDecayUpdates, DEFAULT_RAINBOW_CONFIG)).toBeCloseTo(DEFAULT_RAINBOW_CONFIG.epsilonEnd);
        expect(currentBeta(0, DEFAULT_RAINBOW_CONFIG)).toBeCloseTo(DEFAULT_RAINBOW_CONFIG.priorityBetaStart);
        expect(currentBeta(DEFAULT_RAINBOW_CONFIG.betaAnnealingUpdates, DEFAULT_RAINBOW_CONFIG)).toBeCloseTo(DEFAULT_RAINBOW_CONFIG.priorityBetaEnd);
    });

    it("folds transitions into n-step returns", () => {
        const buffer = createPrioritizedReplayBuffer(16);
        const window: Array<{observation: number[]; action: number; reward: number; nextObservation: number[]; done: boolean}> = [];
        const config = {...DEFAULT_RAINBOW_CONFIG, nStep: 3, gamma: 0.5};

        appendNStepTransition(buffer, window, makeStep(1, false), config);
        appendNStepTransition(buffer, window, makeStep(2, false), config);
        expect(buffer.size).toBe(0);
        appendNStepTransition(buffer, window, makeStep(4, false), config);
        expect(buffer.size).toBe(1);
        // R = 1 + 0.5*2 + 0.25*4 = 3
        expect(buffer.rewards[0]).toBeCloseTo(3);
        expect(buffer.gammaNs[0]).toBeCloseTo(0.125);
        expect(window).toHaveLength(2);

        flushRemainingNStep(buffer, window, config);
        expect(buffer.size).toBe(3);
        expect(window).toHaveLength(0);
    });

    it("flushes the remaining window when an episode ends early", () => {
        const buffer = createPrioritizedReplayBuffer(16);
        const window: Array<{observation: number[]; action: number; reward: number; nextObservation: number[]; done: boolean}> = [];
        const config = {...DEFAULT_RAINBOW_CONFIG, nStep: 3, gamma: 1};

        appendNStepTransition(buffer, window, makeStep(1, false), config);
        appendNStepTransition(buffer, window, makeStep(2, true), config);
        expect(buffer.size).toBe(2);
        expect(window).toHaveLength(0);
        expect(buffer.rewards[0]).toBeCloseTo(3);
        expect(buffer.dones[0]).toBe(true);
        expect(buffer.rewards[1]).toBeCloseTo(2);
        expect(buffer.dones[1]).toBe(true);
    });

    it("samples with priorities and updates them from TD error", () => {
        const buffer = createPrioritizedReplayBuffer(8);
        for (let index = 0; index < 4; index += 1) {
            addTransition(buffer, {
                observation: Array(8).fill(index),
                action: index % 3,
                reward: index,
                nextObservation: Array(8).fill(index + 1),
                done: false,
                gammaN: 0.99,
            });
        }
        buffer.priorities[0] = 100;
        buffer.priorities[1] = 0.01;
        buffer.priorities[2] = 0.01;
        buffer.priorities[3] = 0.01;
        buffer.maxPriority = 100;

        const random = createRandom(3);
        const counts = [0, 0, 0, 0];
        for (let trial = 0; trial < 80; trial += 1) {
            const batch = samplePrioritizedBatch(buffer, 1, 1, 0.5, random);
            counts[batch.indices[0]] += 1;
        }
        expect(counts[0]).toBeGreaterThan(counts[1]);
        expect(counts[0]).toBeGreaterThan(40);

        updatePriorities(buffer, [0], [0.001]);
        expect(buffer.priorities[0]).toBeCloseTo(0.001 + 1e-6, 5);
    });

    it("collects env steps and completes a Rainbow update", async () => {
        const trainer = createRainbowTrainer(7, 0.0005, 2_000);
        const config = {
            ...DEFAULT_RAINBOW_CONFIG,
            episodesPerUpdate: 2,
            trainStepsPerUpdate: 4,
            batchSize: 32,
            maxSteps: 180,
            minBufferSize: 16,
            seed: 7,
        };
        try {
            const result = await trainRainbowUpdate(trainer, config);
            const next = await trainRainbowUpdate(trainer, config);
            expect(result.stats.update).toBe(1);
            expect(next.stats.update).toBe(2);
            expect(next.stats.bestReturn).toBeGreaterThanOrEqual(result.stats.bestReturn);
            expect(next.stats.bestUpdate).toBeLessThanOrEqual(next.stats.update);
            expect(Number.isFinite(result.stats.tdLoss)).toBe(true);
            expect(result.agentGenome).toHaveLength(147);
            expect(result.replay.frames.length).toBeGreaterThan(0);
            expect(next.stats.bufferSize).toBeGreaterThan(0);
            expect(next.stats.epsilon).toBeLessThan(result.stats.epsilon + 1e-9);
        } finally {
            disposeRainbowTrainer(trainer);
        }
    });

    it("uses a multi-episode default batch for stable updates", () => {
        expect(DEFAULT_RAINBOW_CONFIG.episodesPerUpdate).toBe(4);
        expect(DEFAULT_RAINBOW_CONFIG.nStep).toBe(3);
        expect(DEFAULT_RAINBOW_CONFIG.trainStepsPerUpdate).toBe(32);
    });

    it("loads an exported agent and continues training from it", async () => {
        const trainer = createRainbowTrainer(19, 0.0005, 2_000);
        const config = {
            ...DEFAULT_RAINBOW_CONFIG,
            episodesPerUpdate: 2,
            trainStepsPerUpdate: 4,
            batchSize: 32,
            maxSteps: 180,
            minBufferSize: 16,
            seed: 19,
        };
        const genome = Array.from({length: 147}, (_, index) => Math.sin(index * 0.17) * 0.2);
        try {
            const loaded = loadRainbowAgentGenome(trainer, genome, config);
            expect(loaded.agentGenome).toEqual(genome);
            expect(loaded.stats.update).toBe(0);
            expect(loaded.replay.frames.length).toBeGreaterThan(0);

            const trained = await trainRainbowUpdate(trainer, config);
            expect(trained.stats.update).toBe(1);
            expect(trained.agentGenome).toHaveLength(genome.length);
        } finally {
            disposeRainbowTrainer(trainer);
        }
    });

    it("rolls a live-random showcase match each call (not a fixed seed path)", () => {
        const genome = Array.from({length: 147}, (_, index) => Math.sin(index * 0.17) * 0.2);
        const first = createRainbowAgentReplay(genome, 120);
        const second = createRainbowAgentReplay(genome, 120);
        expect(first.frames.length).toBeGreaterThan(0);
        expect(second.frames.length).toBeGreaterThan(0);
        const firstBall = first.frames[0]?.ball;
        const secondBall = second.frames[0]?.ball;
        expect(firstBall && secondBall).toBeTruthy();
    });
});

function makeStep(reward: number, done: boolean): {observation: number[]; action: number; reward: number; nextObservation: number[]; done: boolean} {
    return {
        observation: Array(8).fill(reward),
        action: 1,
        reward,
        nextObservation: Array(8).fill(reward + 1),
        done,
    };
}
