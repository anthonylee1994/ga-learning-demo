/// <reference lib="webworker" />

import {createTradingReplay, evaluateStockGenome, STOCK_TOPOLOGY} from "../domains/stock/simulation";
import {calculateGeneCount} from "../lib/neural-network";
import type {MarketDataPoint} from "../lib/types";
import {setupEvolutionWorker} from "./worker-runtime";

setupEvolutionWorker<MarketDataPoint[], ReturnType<typeof createTradingReplay>>({
    geneCount: calculateGeneCount(STOCK_TOPOLOGY),
    requiresData: true,
    evaluate(genome, data) {
        return evaluateStockGenome(genome, data ?? []);
    },
    createReplay(genome, data) {
        return createTradingReplay(genome, data ?? []);
    },
});
