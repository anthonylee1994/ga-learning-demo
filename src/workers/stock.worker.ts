/// <reference lib="webworker" />

import {createTradingReplay, evaluateStockGenome} from "../domains/stock/simulation";
import {STOCK_GENE_COUNT} from "../domains/stock/strategyGenome";
import type {MarketDataPoint} from "../lib/types";
import {setupEvolutionWorker} from "./workerRuntime";

setupEvolutionWorker<MarketDataPoint[], ReturnType<typeof createTradingReplay>>({
    geneCount: STOCK_GENE_COUNT,
    requiresData: true,
    evaluate(genome, data) {
        return evaluateStockGenome(genome, data ?? []);
    },
    createReplay(genome, data) {
        return createTradingReplay(genome, data ?? []);
    },
});
