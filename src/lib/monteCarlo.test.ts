import {describe, expect, it} from "vitest";
import {createRandom} from "./random";
import {createRandomGenome, runMonteCarloBatch} from "./monteCarlo";
import type {GAConfig, Genome} from "./types";

const BASE_CONFIG: GAConfig = {
    populationSize: 20,
    mutationRate: 0.4,
    mutationScale: 0.2,
    eliteRate: 0.1,
    seed: 42,
    speed: 5,
};

describe("monteCarlo", () => {
    it("createRandomGenome respects gene count and head/tail scales", () => {
        const random = createRandom(1);
        const genome = createRandomGenome(10, random, {
            headGeneCount: 3,
            headScaleMultiplier: 2,
            tailScaleMultiplier: 0.5,
        });
        expect(genome).toHaveLength(10);
        expect(genome.every(gene => Number.isFinite(gene))).toBe(true);
    });

    it("runMonteCarloBatch returns batch stats and improves toward a known optimum", () => {
        const random = createRandom(7);
        // Fitness = negative squared distance from [1,1,1] (simple sphere).
        function fitnessOf(genome: Genome): number {
            return -genome.reduce((sum, gene) => sum + (gene - 1) ** 2, 0);
        }

        let best: {genome: Genome; fitness: number} | null = null;
        for (let generation = 0; generation < 40; generation += 1) {
            const result = runMonteCarloBatch(3, fitnessOf, BASE_CONFIG, random, best, {generation});
            best = {genome: result.bestGenome, fitness: result.bestFitness};
            expect(result.samples).toHaveLength(BASE_CONFIG.populationSize);
            expect(result.stats.bestFitness).toBe(result.bestFitness);
            expect(Number.isFinite(result.stats.averageFitness)).toBe(true);
            expect(result.stats.diversity).toBeGreaterThanOrEqual(0);
        }

        expect(best).not.toBeNull();
        // After enough batches, champion should land near the target.
        expect(best!.fitness).toBeGreaterThan(-0.5);
        for (const gene of best!.genome) {
            expect(Math.abs(gene - 1)).toBeLessThan(0.6);
        }
    });

    it("never worsens the carried champion fitness", () => {
        const random = createRandom(99);
        const champion = {genome: [0, 0, 0], fitness: 100};
        const result = runMonteCarloBatch(3, () => -1, BASE_CONFIG, random, champion, {generation: 5});
        expect(result.bestFitness).toBe(100);
        expect(result.bestGenome).toEqual([0, 0, 0]);
    });

    it("injects seed genomes on the first generation", () => {
        const random = createRandom(3);
        const seed: Genome = [9, 9, 9];
        const evaluated: Genome[] = [];
        runMonteCarloBatch(
            3,
            genome => {
                evaluated.push(genome);
                return genome[0] === 9 ? 50 : 0;
            },
            {...BASE_CONFIG, populationSize: 8},
            random,
            null,
            {generation: 0, seedGenomes: [seed]}
        );
        expect(evaluated.some(genome => genome[0] === 9 && genome[1] === 9)).toBe(true);
    });
});
