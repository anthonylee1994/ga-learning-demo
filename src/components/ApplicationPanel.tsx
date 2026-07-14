import React from "react";

interface ApplicationPanelProps {
    genome: string;
    inputs: string;
    outputs: string;
    fitness: string;
    termination: string;
    eyebrow?: string;
    title?: string;
    /** First row label — defaults to 基因體. */
    genomeLabel?: string;
}

export const ApplicationPanel = React.memo<ApplicationPanelProps>(props => {
    return (
        <section className="application-panel">
            <div className="panel-heading">
                <div>
                    <p className="eyebrow">{props.eyebrow ?? "GA 對應"}</p>
                    <h3>{props.title ?? "點樣套用遺傳演算法"}</h3>
                </div>
            </div>
            <dl className="mapping-grid">
                <div>
                    <dt>{props.genomeLabel ?? "基因體"}</dt>
                    <dd>{props.genome}</dd>
                </div>
                <div>
                    <dt>輸入</dt>
                    <dd>{props.inputs}</dd>
                </div>
                <div>
                    <dt>輸出</dt>
                    <dd>{props.outputs}</dd>
                </div>
                <div>
                    <dt>適應度</dt>
                    <dd>{props.fitness}</dd>
                </div>
                <div>
                    <dt>停止條件</dt>
                    <dd>{props.termination}</dd>
                </div>
            </dl>
        </section>
    );
});
