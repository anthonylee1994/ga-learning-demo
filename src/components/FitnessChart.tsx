import React from "react";
import {CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import type {GenerationStats} from "../lib/types";

interface Props {
    history: GenerationStats[];
    /** When true, plot population diversity on a second Y axis. */
    showDiversity?: boolean;
    eyebrow?: string;
    title?: string;
    emptyMessage?: string;
}

export const FitnessChart = React.memo<Props>(({history, showDiversity = true, eyebrow = "演化訊號", title = "適應度趨勢", emptyMessage = "開始訓練後，呢度會顯示每一代嘅適應度。"}) => {
    return (
        <div className="chart-panel">
            <div className="panel-heading">
                <div>
                    <p className="eyebrow">{eyebrow}</p>
                    <h3>{title}</h3>
                </div>
                <div className="legend-row">
                    <span className="legend-dot best" />
                    最佳
                    <span className="legend-dot avg" />
                    平均
                    {showDiversity ? (
                        <React.Fragment>
                            <span className="legend-dot diversity" />
                            多樣性
                        </React.Fragment>
                    ) : null}
                </div>
            </div>
            <div className="chart-height-sm">
                {history.length > 1 ? (
                    <ResponsiveContainer height="100%" width="100%">
                        <LineChart data={history} margin={{left: -18, right: showDiversity ? 4 : 8, top: 8, bottom: 0}}>
                            <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="generation" stroke="#747b86" tick={{fontSize: 12}} tickLine={false} />
                            <YAxis stroke="#747b86" tick={{fontSize: 12}} tickFormatter={formatChartValue} tickLine={false} width={54} yAxisId="fitness" />
                            {showDiversity ? (
                                <YAxis orientation="right" stroke="#6b7a8d" tick={{fontSize: 11}} tickFormatter={formatChartValue} tickLine={false} width={40} yAxisId="diversity" />
                            ) : null}
                            <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} formatter={formatTooltipValue} />
                            <Line dataKey="bestFitness" dot={false} isAnimationActive={false} stroke="#58d68d" strokeWidth={2} type="monotone" yAxisId="fitness" />
                            <Line dataKey="averageFitness" dot={false} isAnimationActive={false} stroke="#e7b955" strokeWidth={1.5} type="monotone" yAxisId="fitness" />
                            {showDiversity ? (
                                <Line dataKey="diversity" dot={false} isAnimationActive={false} stroke="#7aa2c9" strokeDasharray="4 3" strokeWidth={1.4} type="monotone" yAxisId="diversity" />
                            ) : null}
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="empty-chart">{emptyMessage}</div>
                )}
            </div>
        </div>
    );
});

function formatChartValue(value: number): string {
    return Number.isFinite(value) ? value.toFixed(2) : "";
}

function formatTooltipValue(value: unknown): string {
    if (typeof value === "number" && Number.isFinite(value)) {
        return value.toFixed(2);
    }
    if (value == null) {
        return "—";
    }
    return String(value);
}
