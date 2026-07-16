/// <reference lib="webworker" />

import {createFlappyReplay, evaluateFlappyGenome, FLAPPY_TOPOLOGY} from "../domains/flappy/simulation";
import {calculateGeneCount} from "../lib/neuralNetwork";
import {setupEvolutionWorker} from "./workerRuntime";

setupEvolutionWorker({
    geneCount: calculateGeneCount(FLAPPY_TOPOLOGY),
    evaluate(genome) {
        return evaluateFlappyGenome(genome);
    },
    createReplay(genome) {
        return createFlappyReplay(genome);
    },
});
