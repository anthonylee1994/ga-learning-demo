import {createPopulation, evolvePopulation, mutateGenome, uniformCrossover} from "./ga";
import {createRandom} from "./random";
import type {GAConfig} from "./types";

const CONFIG: GAConfig = {
    populationSize: 10,
    mutationRate: 0.2,
    mutationScale: 0.3,
    eliteRate: 0.1,
    tournamentSize: 3,
    seed: 1,
    speed: 1,
};

describe("genetic algorithm", () => {
    it("creates reproducible populations from a seed", () => {
        const first = createPopulation(4, 6, createRandom(42));
        const second = createPopulation(4, 6, createRandom(42));
        expect(first).toEqual(second);
    });

    it("preserves genome length through crossover and mutation", () => {
        const parentA = [1, 1, 1, 1];
        const parentB = [-1, -1, -1, -1];
        const random = createRandom(19);
        const child = uniformCrossover(parentA, parentB, random);
        const mutated = mutateGenome(child, 1, 0.2, random);
        expect(child).toHaveLength(4);
        expect(mutated).toHaveLength(4);
        expect(child.every(gene => gene === 1 || gene === -1)).toBe(true);
        expect(mutated).not.toEqual(child);
    });

    it("keeps the best genome as an elite", () => {
        const population = Array.from({length: 10}, (_, index) => [index, index + 0.5]);
        const result = evolvePopulation(
            population,
            population.map(genome => genome[0]),
            CONFIG,
            createRandom(7)
        );
        expect(result.bestGenome).toEqual([9, 9.5]);
        expect(result.population[0]).toEqual([9, 9.5]);
        expect(result.population).toHaveLength(10);
        expect(result.stats.diversity).toBeGreaterThan(0);
    });

    it("mutates head genes more than the tail when a profile is set", () => {
        const genome = Array.from({length: 20}, () => 0);
        const random = createRandom(99);
        let headChanges = 0;
        let tailChanges = 0;
        for (let trial = 0; trial < 200; trial += 1) {
            const mutated = mutateGenome(genome, 0.2, 0.3, random, {
                headGeneCount: 5,
                headRateMultiplier: 3,
                tailRateMultiplier: 0.2,
            });
            for (let index = 0; index < mutated.length; index += 1) {
                if (mutated[index] === genome[index]) {
                    continue;
                }
                if (index < 5) {
                    headChanges += 1;
                } else {
                    tailChanges += 1;
                }
            }
        }
        expect(headChanges).toBeGreaterThan(tailChanges);
    });

    it("re-rolls only head genes for head-only immigrants", () => {
        const population = Array.from({length: 8}, (_, index) => Array.from({length: 6}, (__, gene) => index * 10 + gene));
        const fitnesses = population.map((_, index) => index);
        const result = evolvePopulation(population, fitnesses, CONFIG, createRandom(3), {
            mutationProfile: {headGeneCount: 2, immigrantHeadOnly: true},
        });
        const immigrant = result.population[result.population.length - 1];
        const elite = result.bestGenome;
        // Tail genes should match the elite template; head genes were re-sampled.
        expect(immigrant.slice(2)).toEqual(elite.slice(2));
        expect(immigrant.slice(0, 2)).not.toEqual(elite.slice(0, 2));
    });
});
