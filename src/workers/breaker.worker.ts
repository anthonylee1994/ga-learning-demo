/// <reference lib="webworker" />

import {BREAKER_TOPOLOGY, createBreakerReplay, evaluateBreakerGenome} from "../domains/breaker/simulation";
import {calculateGeneCount} from "../lib/neuralNetwork";
import {setupEvolutionWorker} from "./workerRuntime";

setupEvolutionWorker({
    geneCount: calculateGeneCount(BREAKER_TOPOLOGY),
    evaluate(genome) {
        return evaluateBreakerGenome(genome);
    },
    createReplay(genome) {
        return createBreakerReplay(genome);
    },
});
