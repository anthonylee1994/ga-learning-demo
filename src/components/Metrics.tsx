import React from "react";
import type {GenerationStats} from "../lib/types";

interface MetricsProps {
    stats: GenerationStats | null;
    extra?: Array<{label: string; value: string}>;
}

export const Metrics = React.memo<MetricsProps>(({stats, extra = []}) => {
    const values = [
        {label: "世代", value: stats ? String(stats.generation) : "0"},
        {label: "最佳適應度", value: stats ? formatNumber(stats.bestFitness) : "—"},
        {label: "平均適應度", value: stats ? formatNumber(stats.averageFitness) : "—"},
        {label: "多樣性", value: stats ? stats.diversity.toFixed(3) : "—"},
        ...extra,
    ];
    return (
        <div className="metrics-grid">
            {values.map(item => (
                <div className="metric" key={item.label}>
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
