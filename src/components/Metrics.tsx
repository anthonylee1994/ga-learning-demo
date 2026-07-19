import React from "react";
import type {GenerationStats} from "../lib/types";

export interface MetricItem {
    label: string;
    value: string;
    /** Highlight beat/miss vs benchmark (or other signed signal). */
    tone?: "good" | "bad" | "neutral";
}

interface MetricsProps {
    stats: GenerationStats | null;
    extra?: MetricItem[];
    /** Override for the generation counter label (e.g. 蒙地卡羅 → 批次). */
    generationLabel?: string;
    bestFitnessLabel?: string;
    averageFitnessLabel?: string;
}

export const Metrics = React.memo<MetricsProps>(({stats, extra = [], generationLabel = "世代", bestFitnessLabel = "最佳適應度", averageFitnessLabel = "平均適應度"}) => {
    const values: MetricItem[] = [
        {label: generationLabel, value: stats ? String(stats.generation) : "0"},
        {label: bestFitnessLabel, value: stats ? formatNumber(stats.bestFitness) : "—"},
        {label: averageFitnessLabel, value: stats ? formatNumber(stats.averageFitness) : "—"},
        {label: "多樣性", value: stats ? stats.diversity.toFixed(3) : "—"},
        ...extra,
    ];
    return (
        <div className="metrics-grid">
            {values.map(item => (
                <div className={item.tone ? `metric metric--${item.tone}` : "metric"} key={item.label}>
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                </div>
            ))}
        </div>
    );
});

function formatNumber(value: number): string {
    return new Intl.NumberFormat("zh-HK", {maximumFractionDigits: 1}).format(value);
}
