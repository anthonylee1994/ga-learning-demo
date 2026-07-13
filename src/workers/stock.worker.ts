/// <reference lib="webworker" />

import {createTradingReplay, evaluateStockGenome} from "../domains/stock/simulation";
import {createStockSeedGenomes, STOCK_GENE_COUNT, STOCK_MUTATION_PROFILE} from "../domains/stock/strategyGenome";
import type {MarketDataPoint} from "../lib/types";
import {setupEvolutionWorker} from "./workerRuntime";

setupEvolutionWorker<MarketDataPoint[], ReturnType<typeof createTradingReplay>>({
    geneCount: STOCK_GENE_COUNT,
    requiresData: true,
    // Full 15y trading replay is ~MB per postMessage; refresh far less often than snake/breaker.
    minReplayGenerationGap: 20,
    seedGenomes: createStockSeedGenomes(),
    // Period genes explore hard; NN decision head mutates gently.
    mutationProfile: {...STOCK_MUTATION_PROFILE},
    evaluate(genome, data) {
        return evaluateStockGenome(genome, data ?? []);
    },
    createReplay(genome, data) {
        return createTradingReplay(genome, data ?? []);
    },
});
