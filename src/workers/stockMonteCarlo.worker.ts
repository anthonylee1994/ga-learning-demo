/// <reference lib="webworker" />

import {createTradingReplay, evaluateStockGenomeMulti} from "../domains/stock/simulation";
import {createStockSeedGenomes, STOCK_GENE_COUNT, STOCK_MUTATION_PROFILE} from "../domains/stock/strategyGenome";
import type {StockTrainingData} from "../lib/types";
import {setupMonteCarloWorker} from "./monteCarloRuntime";

setupMonteCarloWorker<StockTrainingData, ReturnType<typeof createTradingReplay>>({
    geneCount: STOCK_GENE_COUNT,
    requiresData: true,
    minReplayGenerationGap: 20,
    seedGenomes: createStockSeedGenomes(),
    mutationProfile: {...STOCK_MUTATION_PROFILE},
    // Fitness spans primary + auxiliary tickers (anti-overfit); replay stays on the primary chart.
    evaluate(genome, data, config) {
        return evaluateStockGenomeMulti(genome, data ?? {primary: [], auxiliary: []}, config?.useNeuralNetwork !== false);
    },
    createReplay(genome, data, config) {
        return createTradingReplay(genome, data?.primary ?? [], config?.useNeuralNetwork !== false);
    },
});
