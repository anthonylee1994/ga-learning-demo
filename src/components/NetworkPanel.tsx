import React from "react";
import type {Genome, NetworkTopology} from "../lib/types";
import {forwardWithActivations, inspectGenome, layerSizes, type NetworkForwardPass, type NetworkInspection} from "../lib/neuralNetwork";

interface NetworkPanelProps {
    topology: NetworkTopology;
    genome?: Genome | null;
    /** Current input vector for live activation; omit for static weight view. */
    input?: number[] | null;
    inputLabels?: readonly string[];
    outputLabels?: readonly string[];
    title?: string;
    subtitle?: string;
}

export const NetworkPanel = React.memo<NetworkPanelProps>(
    ({topology, genome, input = null, inputLabels, outputLabels, title = "Neural network", subtitle = "Champion topology · weights · live activations"}) => {
        const inspection = React.useMemo(() => {
            if (!genome || genome.length === 0) {
                return null;
            }
            try {
                return inspectGenome(genome, topology);
            } catch {
                return null;
            }
        }, [genome, topology]);

        const forward = React.useMemo((): NetworkForwardPass | null => {
            if (!genome || !input || input.length !== topology.inputSize) {
                return null;
            }
            try {
                return forwardWithActivations(genome, topology, input);
            } catch {
                return null;
            }
        }, [genome, input, topology]);

        const sizes = layerSizes(topology);
        const decision = forward?.decision ?? null;
        const decisionLabel = decision !== null ? (outputLabels?.[decision] ?? `out ${decision}`) : null;

        return (
            <section className="network-panel">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">Neuroevolution brain</p>
                        <h3>{title}</h3>
                    </div>
                    <div className="network-meta">
                        <span>
                            {sizes.join(" → ")} · {inspection ? countGenes(inspection) : calculateExpectedGenes(topology)} genes
                        </span>
                        {decisionLabel ? (
                            <strong className="network-decision" data-decision={decisionLabel}>
                                決策：{decisionLabel}
                            </strong>
                        ) : null}
                    </div>
                </div>
                <p className="section-copy">{subtitle}</p>

                {!inspection ? (
                    <div className="empty-chart network-empty">開始訓練後，呢度會顯示 champion network 結構同 weights。</div>
                ) : (
                    <React.Fragment>
                        <TopologyGraph activations={forward?.activations ?? null} decision={decision} inputLabels={inputLabels} inspection={inspection} outputLabels={outputLabels} />
                        <WeightHeatmaps inspection={inspection} inputLabels={inputLabels} outputLabels={outputLabels} />
                        {forward ? <ActivationBars activations={forward} inputLabels={inputLabels} outputLabels={outputLabels} /> : null}
                    </React.Fragment>
                )}
            </section>
        );
    }
);

interface TopologyGraphProps {
    inspection: NetworkInspection;
    activations: number[][] | null;
    decision: number | null;
    inputLabels?: readonly string[];
    outputLabels?: readonly string[];
}

const TopologyGraph = React.memo<TopologyGraphProps>(({inspection, activations, decision, inputLabels, outputLabels}) => {
    const {sizes} = inspection;
    const width = 640;
    const height = 220;
    const padX = 48;
    const padY = 28;
    const layerGap = sizes.length > 1 ? (width - padX * 2) / (sizes.length - 1) : 0;

    const positions = sizes.map((count, layer) => {
        const x = padX + layer * layerGap;
        const span = height - padY * 2;
        return Array.from({length: count}, (_, node) => {
            const y = count === 1 ? height / 2 : padY + (node * span) / (count - 1);
            return {x, y};
        });
    });

    // Cap edge count for readability — full dense nets get too noisy.
    const maxEdges = 180;
    const edges: Array<{x1: number; y1: number; x2: number; y2: number; weight: number; key: string}> = [];
    for (let layer = 0; layer < inspection.layers.length; layer += 1) {
        const params = inspection.layers[layer];
        for (let node = 0; node < params.weights.length; node += 1) {
            for (let prev = 0; prev < params.weights[node].length; prev += 1) {
                edges.push({
                    x1: positions[layer][prev].x,
                    y1: positions[layer][prev].y,
                    x2: positions[layer + 1][node].x,
                    y2: positions[layer + 1][node].y,
                    weight: params.weights[node][prev],
                    key: `${layer}-${prev}-${node}`,
                });
            }
        }
    }
    const stride = Math.max(1, Math.ceil(edges.length / maxEdges));
    const drawnEdges = edges.filter((_, index) => index % stride === 0);

    const layerNames = sizes.map((_, index) => {
        if (index === 0) {
            return "Input";
        }
        if (index === sizes.length - 1) {
            return "Output";
        }
        return `Hidden ${index}`;
    });

    return (
        <div className="topology-wrap">
            <svg aria-label="Neural network topology" className="topology-svg" role="img" viewBox={`0 0 ${width} ${height}`}>
                {drawnEdges.map(edge => {
                    const magnitude = Math.min(1, Math.abs(edge.weight) / 1.4);
                    const stroke = edge.weight >= 0 ? `rgba(88, 214, 141, ${0.12 + magnitude * 0.55})` : `rgba(240, 139, 123, ${0.12 + magnitude * 0.55})`;
                    return <line key={edge.key} stroke={stroke} strokeWidth={0.8 + magnitude * 1.6} x1={edge.x1} x2={edge.x2} y1={edge.y1} y2={edge.y2} />;
                })}
                {positions.map((layerNodes, layer) =>
                    layerNodes.map((point, node) => {
                        const activation = activations?.[layer]?.[node];
                        const isDecision = layer === sizes.length - 1 && decision === node;
                        const fill = activationColor(activation, layer === 0);
                        const label = layer === 0 ? (inputLabels?.[node] ?? `in${node}`) : layer === sizes.length - 1 ? (outputLabels?.[node] ?? `out${node}`) : `h${node}`;
                        return (
                            <g key={`n-${layer}-${node}`}>
                                <circle
                                    className={isDecision ? "topology-node decision" : "topology-node"}
                                    cx={point.x}
                                    cy={point.y}
                                    fill={fill}
                                    r={isDecision ? 9 : 7}
                                    stroke={isDecision ? "#b7f5ca" : "#3a424c"}
                                    strokeWidth={isDecision ? 2 : 1}
                                >
                                    <title>
                                        {label}
                                        {activation !== undefined ? ` · act ${activation.toFixed(2)}` : ""}
                                    </title>
                                </circle>
                            </g>
                        );
                    })
                )}
                {layerNames.map((name, index) => (
                    <text fill="#6e7781" fontFamily="Inter, sans-serif" fontSize="11" key={name} textAnchor="middle" x={padX + index * layerGap} y={height - 6}>
                        {name}
                    </text>
                ))}
            </svg>
            <div className="topology-legend">
                <span>
                    <i className="edge-pos" />正 weight
                </span>
                <span>
                    <i className="edge-neg" />負 weight
                </span>
                <span>
                    <i className="node-live" />
                    節點亮度 = activation
                </span>
            </div>
        </div>
    );
});

interface WeightHeatmapsProps {
    inspection: NetworkInspection;
    inputLabels?: readonly string[];
    outputLabels?: readonly string[];
}

const WeightHeatmaps = React.memo<WeightHeatmapsProps>(({inspection, inputLabels, outputLabels}) => {
    return (
        <div className="weight-heatmaps">
            {inspection.layers.map((layer, layerIndex) => {
                const isOutput = layerIndex === inspection.layers.length - 1;
                const rowLabels = isOutput ? outputLabels : undefined;
                const colLabels = layerIndex === 0 ? inputLabels : undefined;
                const title = isOutput ? "Output weights" : `Hidden ${layerIndex + 1} weights`;
                return <WeightMatrix colLabels={colLabels} key={`w-${layerIndex}`} matrix={layer.weights} rowLabels={rowLabels} title={title} />;
            })}
        </div>
    );
});

interface WeightMatrixProps {
    title: string;
    matrix: number[][];
    rowLabels?: readonly string[];
    colLabels?: readonly string[];
}

const WeightMatrix = React.memo<WeightMatrixProps>(({title, matrix, rowLabels, colLabels}) => {
    if (matrix.length === 0 || matrix[0].length === 0) {
        return null;
    }
    const cols = matrix[0].length;
    const showColLabels = Boolean(colLabels) && cols <= 12;
    const showRowLabels = Boolean(rowLabels);

    return (
        <div className="weight-matrix">
            <div className="weight-matrix-title">{title}</div>
            <div className="weight-matrix-scroll">
                <div
                    className="weight-grid"
                    style={{
                        gridTemplateColumns: `${showRowLabels ? "minmax(44px, auto) " : ""}repeat(${cols}, minmax(10px, 1fr))`,
                    }}
                >
                    {showColLabels ? (
                        <React.Fragment>
                            {showRowLabels ? <span className="weight-corner" /> : null}
                            {colLabels!.map((label, index) => (
                                <span className="weight-col-label" key={`c-${index}`} title={label}>
                                    {shortLabel(label)}
                                </span>
                            ))}
                        </React.Fragment>
                    ) : null}
                    {matrix.map((row, rowIndex) => (
                        <React.Fragment key={`r-${rowIndex}`}>
                            {showRowLabels ? (
                                <span className="weight-row-label" title={rowLabels?.[rowIndex] ?? `n${rowIndex}`}>
                                    {shortLabel(rowLabels?.[rowIndex] ?? `n${rowIndex}`)}
                                </span>
                            ) : null}
                            {row.map((value, colIndex) => (
                                <span className="weight-cell" key={`c-${rowIndex}-${colIndex}`} style={{background: weightColor(value)}} title={`${value.toFixed(3)}`}>
                                    {matrix.length * cols <= 48 ? value.toFixed(1) : ""}
                                </span>
                            ))}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </div>
    );
});

interface ActivationBarsProps {
    activations: NetworkForwardPass;
    inputLabels?: readonly string[];
    outputLabels?: readonly string[];
}

const ActivationBars = React.memo<ActivationBarsProps>(({activations, inputLabels, outputLabels}) => {
    const outputs = activations.outputs;
    const decision = activations.decision;
    const inputs = activations.activations[0] ?? [];

    return (
        <div className="activation-bars">
            <div className="activation-group">
                <div className="weight-matrix-title">Input activations</div>
                <div className="activation-list">
                    {inputs.map((value, index) => (
                        <div className="activation-row" key={`in-${index}`}>
                            <span title={inputLabels?.[index] ?? `in${index}`}>{shortLabel(inputLabels?.[index] ?? `in${index}`)}</span>
                            <div className="activation-track">
                                <i style={{width: `${Math.abs(value) * 50}%`, marginLeft: value >= 0 ? "50%" : `${50 - Math.abs(value) * 50}%`, background: value >= 0 ? "#58d68d" : "#f08b7b"}} />
                            </div>
                            <code>{value.toFixed(2)}</code>
                        </div>
                    ))}
                </div>
            </div>
            <div className="activation-group">
                <div className="weight-matrix-title">Output activations</div>
                <div className="activation-list">
                    {outputs.map((value, index) => {
                        const active = index === decision;
                        return (
                            <div className={`activation-row ${active ? "active" : ""}`} key={`out-${index}`}>
                                <span title={outputLabels?.[index] ?? `out${index}`}>{shortLabel(outputLabels?.[index] ?? `out${index}`)}</span>
                                <div className="activation-track">
                                    <i style={{width: `${((value + 1) / 2) * 100}%`, background: active ? "#b7f5ca" : "#58d68d"}} />
                                </div>
                                <code>{value.toFixed(2)}</code>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
});

function activationColor(activation: number | undefined, isInput: boolean): string {
    if (activation === undefined) {
        return isInput ? "#1a2229" : "#1c242c";
    }
    // tanh-ish / signed inputs: map [-1,1] → green intensity
    const t = Math.max(0, Math.min(1, (activation + 1) / 2));
    const g = Math.round(80 + t * 140);
    const r = Math.round(40 + (1 - t) * 50);
    const b = Math.round(70 + t * 40);
    return `rgb(${r}, ${g}, ${b})`;
}

function weightColor(value: number): string {
    const magnitude = Math.min(1, Math.abs(value) / 1.6);
    if (value >= 0) {
        return `rgba(88, 214, 141, ${0.08 + magnitude * 0.72})`;
    }
    return `rgba(240, 139, 123, ${0.08 + magnitude * 0.72})`;
}

function shortLabel(label: string): string {
    return label.length > 5 ? `${label.slice(0, 4)}…` : label;
}

function countGenes(inspection: NetworkInspection): number {
    return inspection.layers.reduce((sum, layer) => sum + layer.biases.length + layer.weights.reduce((inner, row) => inner + row.length, 0), 0);
}

function calculateExpectedGenes(topology: NetworkTopology): number {
    const sizes = layerSizes(topology);
    let count = 0;
    for (let layer = 1; layer < sizes.length; layer += 1) {
        count += sizes[layer] + sizes[layer] * sizes[layer - 1];
    }
    return count;
}
