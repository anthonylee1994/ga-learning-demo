/// <reference lib="webworker" />

import {createSnakeReplay, evaluateSnakeGenome, SNAKE_TOPOLOGY} from "../domains/snake/simulation";
import {calculateGeneCount} from "../lib/neuralNetwork";
import {setupEvolutionWorker} from "./workerRuntime";

setupEvolutionWorker({
    geneCount: calculateGeneCount(SNAKE_TOPOLOGY),
    evaluate(genome) {
        return evaluateSnakeGenome(genome);
    },
    createReplay(genome) {
        return createSnakeReplay(genome);
    },
});
