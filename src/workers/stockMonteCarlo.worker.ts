/// <reference lib="webworker" />

import {createTradingReplay, evaluateStockGenome} from "../domains/stock/simulation";
import {createStockSeedGenomes, STOCK_GENE_COUNT, STOCK_MUTATION_PROFILE} from "../domains/stock/strategyGenome";
import type {MarketDataPoint} from "../lib/types";
import {setupMonteCarloWorker} from "./monteCarloRuntime";

setupMonteCarloWorker<MarketDataPoint[], ReturnType<typeof createTradingReplay>>({
    geneCount: STOCK_GENE_COUNT,
    requiresData: true,
    minReplayGenerationGap: 20,
    seedGenomes: createStockSeedGenomes(),
    mutationProfile: {...STOCK_MUTATION_PROFILE},
    evaluate(genome, data, config) {
        return evaluateStockGenome(genome, data ?? [], config?.useNeuralNetwork !== false);
    },
    createReplay(genome, data, config) {
        return createTradingReplay(genome, data ?? [], config?.useNeuralNetwork !== false);
    },
});
