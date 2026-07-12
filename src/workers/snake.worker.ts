/// <reference lib="webworker" />

import {createSnakeReplay, evaluateSnakeGenome, SNAKE_TOPOLOGY} from "../domains/snake/simulation";
import {calculateGeneCount} from "../lib/neural-network";
import {setupEvolutionWorker} from "./worker-runtime";

setupEvolutionWorker({
    geneCount: calculateGeneCount(SNAKE_TOPOLOGY),
    evaluate(genome) {
        return evaluateSnakeGenome(genome);
    },
    createReplay(genome) {
        return createSnakeReplay(genome);
    },
});
