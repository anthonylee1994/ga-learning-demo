import type {GAConfig, GenerationStats, Genome} from "./types";
import type {RandomSource} from "./random";

export interface EvolutionResult {
    population: Genome[];
    bestGenome: Genome;
    bestFitness: number;
    stats: Omit<GenerationStats, "generation">;
}

export function createPopulation(size: number, geneCount: number, random: RandomSource, champion?: Genome): Genome[] {
    const population = Array.from({length: size}, () => Array.from({length: geneCount}, () => random.gaussian() * 0.55));

    if (champion?.length === geneCount && population.length > 0) {
        population[0] = [...champion];
    }

    return population;
}

export function uniformCrossover(parentA: Genome, parentB: Genome, random: RandomSource): Genome {
    return parentA.map((gene, index) => (random.next() < 0.5 ? gene : parentB[index]));
}

export function mutateGenome(genome: Genome, mutationRate: number, mutationScale: number, random: RandomSource): Genome {
    return genome.map(gene => (random.next() < mutationRate ? gene + random.gaussian() * mutationScale : gene));
}

export function calculateDiversity(population: Genome[]): number {
    if (population.length < 2 || population[0].length === 0) {
        return 0;
    }

    let totalVariance = 0;
    const geneCount = population[0].length;
    for (let geneIndex = 0; geneIndex < geneCount; geneIndex += 1) {
        const mean = population.reduce((sum, genome) => sum + genome[geneIndex], 0) / population.length;
        totalVariance += population.reduce((sum, genome) => sum + (genome[geneIndex] - mean) ** 2, 0) / population.length;
    }

    return Math.sqrt(totalVariance / geneCount);
}

export function evolvePopulation(population: Genome[], fitnesses: number[], config: GAConfig, random: RandomSource): EvolutionResult {
    const ranked = population.map((genome, index) => ({genome, fitness: fitnesses[index]})).sort((a, b) => b.fitness - a.fitness);
    const eliteCount = Math.max(1, Math.floor(population.length * config.eliteRate));
    const nextPopulation = ranked.slice(0, eliteCount).map(({genome}) => [...genome]);

    while (nextPopulation.length < population.length) {
        const parentA = tournamentSelect(ranked, config.tournamentSize, random);
        const parentB = tournamentSelect(ranked, config.tournamentSize, random);
        const child = uniformCrossover(parentA, parentB, random);
        nextPopulation.push(mutateGenome(child, config.mutationRate, config.mutationScale, random));
    }

    const averageFitness = fitnesses.reduce((sum, value) => sum + value, 0) / fitnesses.length;
    return {
        population: nextPopulation,
        bestGenome: [...ranked[0].genome],
        bestFitness: ranked[0].fitness,
        stats: {
            bestFitness: ranked[0].fitness,
            averageFitness,
            diversity: calculateDiversity(population),
        },
    };
}

function tournamentSelect(ranked: Array<{genome: Genome; fitness: number}>, tournamentSize: number, random: RandomSource): Genome {
    let best = ranked[random.integer(0, ranked.length - 1)];
    for (let index = 1; index < tournamentSize; index += 1) {
        const candidate = ranked[random.integer(0, ranked.length - 1)];
        if (candidate.fitness > best.fitness) {
            best = candidate;
        }
    }
    return best.genome;
}
