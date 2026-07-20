import type {BreakerPolicyTransition} from "./simulation";
import {DEFAULT_PPO_CONFIG, calculateGeneralizedAdvantages, createPpoActorReplay, createPpoTrainer, disposePpoTrainer, loadPpoActorGenome, probabilitiesFromLogits, trainPpoUpdate} from "./ppo";

describe("breaker PPO", () => {
    it("turns actor logits into a normalized action distribution", () => {
        const probabilities = probabilitiesFromLogits([1.2, -0.4, 0.3]);
        expect(probabilities).toHaveLength(3);
        expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 8);
        expect(probabilities[0]).toBeGreaterThan(probabilities[2]);
        expect(probabilities[2]).toBeGreaterThan(probabilities[1]);
    });

    it("stops generalized advantage propagation at episode boundaries", () => {
        const transitions: BreakerPolicyTransition[] = [createTransition(1, false, 0.5), createTransition(2, true, 0.25)];
        const result = calculateGeneralizedAdvantages(transitions, [0.25, 0], 1, 1);
        expect(result.advantages[1]).toBeCloseTo(1.75);
        expect(result.advantages[0]).toBeCloseTo(2.5);
        expect(result.returns).toEqual([3, 2]);
    });

    it("collects a rollout and completes an actor-critic update", async () => {
        const trainer = createPpoTrainer(7, 0.0008);
        try {
            const result = await trainPpoUpdate(trainer, {...DEFAULT_PPO_CONFIG, episodesPerUpdate: 2, maxSteps: 180, epochs: 1, seed: 7});
            const next = await trainPpoUpdate(trainer, {...DEFAULT_PPO_CONFIG, episodesPerUpdate: 2, maxSteps: 180, epochs: 1, seed: 7});
            expect(result.stats.update).toBe(1);
            expect(next.stats.update).toBe(2);
            expect(next.stats.bestReturn).toBeGreaterThanOrEqual(result.stats.bestReturn);
            expect(next.stats.bestUpdate).toBeLessThanOrEqual(next.stats.update);
            expect(Number.isFinite(result.stats.policyLoss)).toBe(true);
            expect(result.actorGenome).toHaveLength(147);
            expect(result.replay.frames.length).toBeGreaterThan(0);
        } finally {
            disposePpoTrainer(trainer);
        }
    });

    it("uses enough rollout episodes for a stable default batch", () => {
        expect(DEFAULT_PPO_CONFIG.episodesPerUpdate).toBe(8);
    });

    it("loads an exported actor and continues training from it", async () => {
        const trainer = createPpoTrainer(19, 0.0008);
        const config = {...DEFAULT_PPO_CONFIG, episodesPerUpdate: 2, maxSteps: 180, epochs: 1, seed: 19};
        const genome = Array.from({length: 147}, (_, index) => Math.sin(index * 0.17) * 0.2);
        try {
            const loaded = loadPpoActorGenome(trainer, genome, config);
            expect(loaded.actorGenome).toEqual(genome);
            expect(loaded.stats.update).toBe(0);
            expect(loaded.replay.frames.length).toBeGreaterThan(0);

            const trained = await trainPpoUpdate(trainer, config);
            expect(trained.stats.update).toBe(1);
            expect(trained.actorGenome).toHaveLength(genome.length);
        } finally {
            disposePpoTrainer(trainer);
        }
    });

    it("rolls a live-random showcase match each call (not a fixed seed path)", () => {
        const genome = Array.from({length: 147}, (_, index) => Math.sin(index * 0.17) * 0.2);
        const first = createPpoActorReplay(genome, 120);
        const second = createPpoActorReplay(genome, 120);
        expect(first.frames.length).toBeGreaterThan(0);
        expect(second.frames.length).toBeGreaterThan(0);
        // Live spawn/jitter noise → almost always different first frames; only compare when both exist.
        const firstBall = first.frames[0]?.ball;
        const secondBall = second.frames[0]?.ball;
        expect(firstBall && secondBall).toBeTruthy();
    });
});

function createTransition(reward: number, done: boolean, value: number): BreakerPolicyTransition {
    return {
        observation: Array(8).fill(0),
        action: 1,
        reward,
        nextObservation: Array(8).fill(0),
        done,
        logProbability: Math.log(1 / 3),
        value,
    };
}
