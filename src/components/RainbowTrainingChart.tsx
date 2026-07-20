import React from "react";
import {CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import type {RainbowUpdateStats} from "../domains/breaker/rainbow";

interface Props {
    history: RainbowUpdateStats[];
}

export const RainbowTrainingChart = React.memo<Props>(({history}) => {
    return (
        <div className="chart-panel">
            <div className="panel-heading">
                <div>
                    <p className="eyebrow">訓練訊號</p>
                    <h3>回報與 TD 損失</h3>
                </div>
                <div className="legend-row">
                    <span className="legend-dot best" />
                    歷史最佳
                    <span className="legend-dot avg" />
                    固定評估
                    <span className="legend-dot rainbow-loss" />
                    TD loss
                </div>
            </div>
            <div className="chart-height-sm">
                {history.length > 1 ? (
                    <ResponsiveContainer height="100%" width="100%">
                        <LineChart data={history} margin={{left: -18, right: 4, top: 8, bottom: 0}}>
                            <CartesianGrid stroke="#252a31" strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="update" stroke="#747b86" tick={{fontSize: 12}} tickLine={false} />
                            <YAxis stroke="#747b86" tick={{fontSize: 12}} tickFormatter={formatChartValue} tickLine={false} width={54} yAxisId="return" />
                            <YAxis orientation="right" stroke="#7aa2c9" tick={{fontSize: 11}} tickFormatter={formatChartValue} tickLine={false} width={40} yAxisId="loss" />
                            <Tooltip contentStyle={{background: "#15191f", border: "1px solid #303640", borderRadius: 8}} formatter={formatTooltipValue} />
                            <Line dataKey="bestReturn" dot={false} isAnimationActive={false} name="歷史最佳" stroke="#58d68d" strokeWidth={2} type="monotone" yAxisId="return" />
                            <Line dataKey="averageReturn" dot={false} isAnimationActive={false} name="固定評估" stroke="#e7b955" strokeWidth={1.5} type="monotone" yAxisId="return" />
                            <Line dataKey="tdLoss" dot={false} isAnimationActive={false} name="TD loss" stroke="#7aa2c9" strokeDasharray="4 3" strokeWidth={1.4} type="monotone" yAxisId="loss" />
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="empty-chart">開始訓練後，呢度會顯示每輪 Rainbow 更新嘅回報同 TD loss。</div>
                )}
            </div>
        </div>
    );
});

function formatChartValue(value: number): string {
    return Number.isFinite(value) ? value.toFixed(2) : "";
}

function formatTooltipValue(value: unknown): string {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : String(value ?? "—");
}
