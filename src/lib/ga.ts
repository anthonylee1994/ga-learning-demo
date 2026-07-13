import type {GAConfig, GenerationStats, Genome} from "./types";
import type {RandomSource} from "./random";

export interface EvolutionResult {
    population: Genome[];
    bestGenome: Genome;
    bestFitness: number;
    stats: Omit<GenerationStats, "generation">;
}

/**
 * Optional head/tail mutation bias. Stock lab uses this so indicator parameter genes
 * explore more aggressively than the thin NN decision-head weights.
 */
export interface MutationProfile {
    /** Genes [0, headGeneCount) are the "head" (e.g. indicator periods). */
    headGeneCount: number;
    /** Multiplier on base mutation rate for head genes. */
    headRateMultiplier?: number;
    /** Multiplier on base mutation scale for head genes. */
    headScaleMultiplier?: number;
    /** Multiplier on base mutation rate for tail genes (e.g. NN weights). */
    tailRateMultiplier?: number;
    /** Multiplier on base mutation scale for tail genes. */
    tailScaleMultiplier?: number;
    /**
     * When true, the random immigrant re-rolls only head genes and keeps the
     * elite network tail — forces period exploration without thrashing weights.
     */
    immigrantHeadOnly?: boolean;
}

export interface EvolveOptions {
    mutationProfile?: MutationProfile;
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

/** Share of mutations that redraw the gene from scratch instead of perturbing it. Keeps
 * tanh-decoded genes (e.g. the stock indicator parameters) mutable after they saturate,
 * and lets a converged population escape a locked-in local optimum. */
const RESET_MUTATION_SHARE = 0.2;

export function mutateGenome(genome: Genome, mutationRate: number, mutationScale: number, random: RandomSource, profile?: MutationProfile): Genome {
    const headCount = profile?.headGeneCount ?? 0;
    const headRate = mutationRate * (profile?.headRateMultiplier ?? 1);
    const headScale = mutationScale * (profile?.headScaleMultiplier ?? 1);
    const tailRate = mutationRate * (profile?.tailRateMultiplier ?? 1);
    const tailScale = mutationScale * (profile?.tailScaleMultiplier ?? 1);

    return genome.map((gene, index) => {
        const inHead = profile !== undefined && index < headCount;
        const rate = inHead ? headRate : tailRate;
        const scale = inHead ? headScale : tailScale;
        if (random.next() >= rate) {
            return gene;
        }
        if (random.next() < RESET_MUTATION_SHARE) {
            return random.gaussian() * 0.55;
        }
        return gene + random.gaussian() * scale;
    });
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

export function evolvePopulation(population: Genome[], fitnesses: number[], config: GAConfig, random: RandomSource, options?: EvolveOptions): EvolutionResult {
    const ranked = population.map((genome, index) => ({genome, fitness: fitnesses[index]})).sort((a, b) => b.fitness - a.fitness);
    const eliteCount = Math.max(1, Math.floor(population.length * config.eliteRate));
    const nextPopulation = ranked.slice(0, eliteCount).map(({genome}) => [...genome]);
    const profile = options?.mutationProfile;

    while (nextPopulation.length < population.length) {
        const parentA = rouletteWheelSelect(ranked, random);
        const parentB = rouletteWheelSelect(ranked, random);
        const child = uniformCrossover(parentA, parentB, random);
        nextPopulation.push(mutateGenome(child, config.mutationRate, config.mutationScale, random, profile));
    }

    // Random immigrant: one slot per generation is a fresh genome, so a converged population
    // never fully stops exploring (e.g. new indicator-parameter combinations in the stock lab).
    if (nextPopulation.length > 2) {
        const geneCount = population[0]?.length ?? 0;
        const headCount = profile?.headGeneCount ?? 0;
        if (profile?.immigrantHeadOnly && headCount > 0 && ranked[0]) {
            const template = ranked[0].genome;
            nextPopulation[nextPopulation.length - 1] = template.map((gene, index) => (index < headCount ? random.gaussian() * 0.55 : gene));
        } else {
            nextPopulation[nextPopulation.length - 1] = Array.from({length: geneCount}, () => random.gaussian() * 0.55);
        }
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

export function rouletteWheelSelect(candidates: Array<{genome: Genome; fitness: number}>, random: RandomSource): Genome {
    let fitnessScale = 1;
    for (const candidate of candidates) {
        fitnessScale = Math.max(fitnessScale, Math.abs(candidate.fitness));
    }

    const scaledFitnesses = candidates.map(candidate => candidate.fitness / fitnessScale);
    const minimumFitness = Math.min(...scaledFitnesses);
    const offset = minimumFitness < 0 ? -minimumFitness : 0;
    const weights = scaledFitnesses.map(fitness => fitness + offset);
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);

    if (totalWeight <= 0) {
        return candidates[random.integer(0, candidates.length - 1)].genome;
    }

    let threshold = random.next() * totalWeight;
    for (let index = 0; index < candidates.length; index += 1) {
        threshold -= weights[index];
        if (threshold < 0) {
            return candidates[index].genome;
        }
    }

    return candidates[candidates.length - 1].genome;
}
