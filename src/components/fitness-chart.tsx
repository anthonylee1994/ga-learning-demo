import React from "react";
import {CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import type {GenerationStats} from "../lib/types";

interface Props {
    history: GenerationStats[];
}

export const FitnessChart = React.memo<Props>(({history}) => {
    return (
        <div className="chart-panel">
            <div className="panel-heading">
                <div>
                    <p className="eyebrow">Evolution signal</p>
                    <h3>Fitness 趨勢</h3>
                </div>
                <div className="legend-row">
                    <span className="legend-dot best" />
                    最佳 <span className="legend-dot avg" />
                    平均
                </div>
            </div>
            <div className="chart-height-sm">
                {history.length > 1 ? (
                    <ResponsiveContainer height="100%" width="100%">
                        <LineChart data={history} margin={{left: -18, right: 8, top: 8, bottom: 0}}>
                            <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="generation" stroke="#747b86" tick={{fontSize: 11}} tickLine={false} />
                            <YAxis stroke="#747b86" tick={{fontSize: 11}} tickLine={false} width={54} />
                            <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} />
                            <Line dataKey="bestFitness" dot={false} isAnimationActive={false} stroke="#58d68d" strokeWidth={2} type="monotone" />
                            <Line dataKey="averageFitness" dot={false} isAnimationActive={false} stroke="#e7b955" strokeWidth={1.5} type="monotone" />
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="empty-chart">開始訓練後，呢度會顯示每一代嘅 fitness。</div>
                )}
            </div>
        </div>
    );
});
