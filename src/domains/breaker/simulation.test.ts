import {calculateGeneCount} from "../../lib/neural-network";
import {createPopulation, evolvePopulation} from "../../lib/ga";
import {createRandom} from "../../lib/random";
import {BREAKER_TOPOLOGY, createBreakerReplay, evaluateBreakerGenome} from "./simulation";

describe("block breaker simulation", () => {
    const genome = Array(calculateGeneCount(BREAKER_TOPOLOGY)).fill(0);

    it("returns finite fitness from fixed-step Matter.js physics", () => {
        expect(Number.isFinite(evaluateBreakerGenome(genome))).toBe(true);
    });

    it("records a bounded champion replay", () => {
        const replay = createBreakerReplay(genome);
        expect(replay.frames.length).toBeGreaterThan(0);
        expect(replay.steps).toBeLessThanOrEqual(600);
        expect(replay.bricksCleared).toBeGreaterThanOrEqual(0);
        expect(replay.frames.at(-1)?.terminal).toMatch(/lost|cleared|timeout/);
    });

    it("keeps evaluate fitness consistent with the replay launch seed", () => {
        // Replay uses launch 0.86, which is one of the eval launches — a strong genome
        // must not show ~1 brick on screen while scoring many bricks only under a hidden angle.
        const geneCount = calculateGeneCount(BREAKER_TOPOLOGY);
        const random = createRandom(281);
        let population = createPopulation(16, geneCount, random);
        const config = {
            populationSize: 16,
            mutationRate: 0.14,
            mutationScale: 0.26,
            eliteRate: 0.1,
            tournamentSize: 4,
            seed: 281,
            speed: 3,
        };
        let bestGenome = population[0];
        let bestFitness = Number.NEGATIVE_INFINITY;
        for (let generation = 0; generation < 25; generation += 1) {
            const fitnesses = population.map(candidate => evaluateBreakerGenome(candidate));
            const result = evolvePopulation(population, fitnesses, config, random);
            population = result.population;
            if (result.bestFitness > bestFitness) {
                bestFitness = result.bestFitness;
                bestGenome = result.bestGenome;
            }
        }
        const replay = createBreakerReplay(bestGenome);
        // If eval/replay physics diverge, fitness can be high while on-screen clears stay near 1–2.
        expect(bestFitness).toBeGreaterThan(100);
        expect(replay.bricksCleared).toBeGreaterThanOrEqual(2);
    }, 60_000);
});
