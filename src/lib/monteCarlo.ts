import {calculateDiversity, mutateGenome, type MutationProfile} from "./ga";
import type {RandomSource} from "./random";
import type {GAConfig, Genome} from "./types";

export interface MonteCarloBatchResult {
    /** Candidates evaluated this batch (for diversity). */
    samples: Genome[];
    bestGenome: Genome;
    bestFitness: number;
    stats: {
        bestFitness: number;
        averageFitness: number;
        diversity: number;
    };
}

export interface MonteCarloOptions {
    mutationProfile?: MutationProfile;
    /**
     * Fraction of each batch drawn by local Gaussian walk around the current
     * champion. The rest are pure global random draws (classic 蒙地卡羅).
     * Uses GAConfig.mutationRate as the local share (clamped to [0.05, 0.95]).
     */
    localShare?: number;
    /** Optional classic baselines injected into the first batch only. */
    seedGenomes?: Genome[];
    /** Generation index (0-based before increment) — seeds only apply when 0. */
    generation?: number;
}

/**
 * One 蒙地卡羅 step: evaluate a fresh batch of random / local-walk genomes.
 * Keeps the global champion across batches (caller passes previous best).
 *
 * Config mapping (reuses GAConfig sliders so the UI stays familiar):
 * - populationSize → samples per batch
 * - mutationRate   → local exploration share around the champion
 * - mutationScale  → local walk step size
 */
export function runMonteCarloBatch(
    geneCount: number,
    fitnessesOf: (genome: Genome) => number,
    config: GAConfig,
    random: RandomSource,
    previousBest: {genome: Genome; fitness: number} | null,
    options?: MonteCarloOptions
): MonteCarloBatchResult {
    const batchSize = Math.max(4, config.populationSize);
    const localShare = clamp(options?.localShare ?? config.mutationRate, 0.05, 0.95);
    const profile = options?.mutationProfile;
    const samples: Genome[] = [];

    // First batch: inject classic baselines so MC is not pure noise on day 0.
    if ((options?.generation ?? 0) === 0 && options?.seedGenomes?.length) {
        for (const seed of options.seedGenomes) {
            if (seed.length === geneCount && samples.length < batchSize) {
                samples.push([...seed]);
            }
        }
    }
    if (previousBest?.genome.length === geneCount && samples.length < batchSize) {
        // Always re-evaluate the incumbent so fitness scale changes (NN toggle) stay honest.
        samples.push([...previousBest.genome]);
    }

    while (samples.length < batchSize) {
        if (previousBest && random.next() < localShare) {
            // Local 蒙地卡羅 walk: perturb champion (head genes harder via profile).
            samples.push(mutateGenome(previousBest.genome, Math.min(1, config.mutationRate * 2.5), config.mutationScale, random, profile));
        } else {
            samples.push(createRandomGenome(geneCount, random, profile));
        }
    }

    const fitnesses = samples.map(genome => fitnessesOf(genome));
    let bestIndex = 0;
    for (let index = 1; index < fitnesses.length; index += 1) {
        if (fitnesses[index] > fitnesses[bestIndex]) {
            bestIndex = index;
        }
    }

    let bestGenome = samples[bestIndex];
    let bestFitness = fitnesses[bestIndex];
    if (previousBest && previousBest.fitness > bestFitness) {
        bestGenome = previousBest.genome;
        bestFitness = previousBest.fitness;
    }

    const averageFitness = fitnesses.reduce((sum, value) => sum + value, 0) / fitnesses.length;
    return {
        samples,
        bestGenome: [...bestGenome],
        bestFitness,
        stats: {
            bestFitness,
            averageFitness,
            diversity: calculateDiversity(samples),
        },
    };
}

/**
 * Pure global sample, optionally with head/tail scale bias so indicator genes
 * explore a wider range than the thin NN tail (same spirit as stock GA profile).
 */
export function createRandomGenome(geneCount: number, random: RandomSource, profile?: MutationProfile): Genome {
    const headCount = profile?.headGeneCount ?? 0;
    const headScale = 0.55 * (profile?.headScaleMultiplier ?? 1);
    const tailScale = 0.55 * (profile?.tailScaleMultiplier ?? 1);
    return Array.from({length: geneCount}, (_, index) => {
        const inHead = profile !== undefined && index < headCount;
        return random.gaussian() * (inHead ? headScale : tailScale);
    });
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}
