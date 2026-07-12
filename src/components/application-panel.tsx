import React from "react";

interface ApplicationPanelProps {
    genome: string;
    inputs: string;
    outputs: string;
    fitness: string;
    termination: string;
}

export const ApplicationPanel = React.memo(function ApplicationPanel(props: ApplicationPanelProps) {
    return (
        <section className="application-panel">
            <div className="panel-heading">
                <div>
                    <p className="eyebrow">GA mapping</p>
                    <h3>點樣套用 Genetic Algorithm</h3>
                </div>
            </div>
            <dl className="mapping-grid">
                <div>
                    <dt>Genome</dt>
                    <dd>{props.genome}</dd>
                </div>
                <div>
                    <dt>Inputs</dt>
                    <dd>{props.inputs}</dd>
                </div>
                <div>
                    <dt>Outputs</dt>
                    <dd>{props.outputs}</dd>
                </div>
                <div>
                    <dt>Fitness</dt>
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
