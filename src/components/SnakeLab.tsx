import React from "react";
import {Dna} from "lucide-react";
import {buildSnakeInputFromFrame, createSnakeReplay, evaluateSnakeGenome, SNAKE_INPUT_LABELS, SNAKE_OUTPUT_LABELS, SNAKE_TOPOLOGY} from "../domains/snake/simulation";
import {useEvolutionDemo} from "../hooks/useEvolutionDemo";
import type {GAConfig, Genome, SnakeFrame, SnakeReplay} from "../lib/types";
import {ApplicationPanel} from "./ApplicationPanel";
import {DemoControls} from "./DemoControls";
import {FitnessChart} from "./FitnessChart";
import {GenomeTransfer} from "./GenomeTransfer";
import {Metrics} from "./Metrics";
import {NetworkPanel} from "./NetworkPanel";
import {SnakeCanvas} from "./SnakeCanvas";

const DEFAULT_CONFIG: GAConfig = {
    populationSize: 36,
    mutationRate: 0.12,
    mutationScale: 0.24,
    eliteRate: 0.08,
    seed: Math.round(Math.random() * 1_000_000),
    speed: 5,
};

export const SnakeLab = React.memo(() => {
    const demo = useEvolutionDemo<undefined, SnakeReplay>({
        topic: "snake",
        createWorker: () => new Worker(new URL("../workers/snake.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: {
            ...DEFAULT_CONFIG,
            seed: Math.round(Math.random() * 1_000_000),
        },
    });
    const [liveInput, setLiveInput] = React.useState<number[] | null>(null);
    const [transferMessage, setTransferMessage] = React.useState<{type: "status" | "error"; text: string} | null>(null);

    const handleFrameChange = (frame: SnakeFrame | null) => {
        if (!frame || frame.snake.length === 0) {
            setLiveInput(null);
            return;
        }
        setLiveInput(buildSnakeInputFromFrame(frame));
    };

    const handleImportGenome = (genome: Genome) => {
        const replay = createSnakeReplay(genome);
        const fitness = evaluateSnakeGenome(genome);
        demo.loadChampion({genome, replay, fitness});
    };

    // Drop stale activations when the champion genome is cleared / replaced without frames.
    React.useEffect(() => {
        if (!demo.champion) {
            setLiveInput(null);
        }
    }, [demo.champion]);

    return (
        <DemoShell accent="snake" description="一個 10 → 12 → 3 嘅 Brain.js 網絡，靠食嘢、存活同靠近食物獎勵學識轉彎。" icon={<Dna size={20} strokeWidth={1.5} />} title="貪食蛇 · 神經演化">
            <div className="workspace-grid">
                <main className="demo-main">
                    <Metrics
                        extra={[
                            {label: "冠軍分數", value: String(demo.champion?.replay.score ?? 0)},
                            {label: "存活步數", value: String(demo.champion?.replay.steps ?? 0)},
                        ]}
                        stats={demo.stats}
                    />
                    <div className="simulation-stage snake-stage">
                        <div className="stage-overlay">
                            <span>20 × 20 格</span>
                            <span>{demo.champion?.replay ? (demo.status === "running" ? "冠軍循環重播 · 進化中" : "冠軍循環重播") : "未有冠軍"}</span>
                        </div>
                        <SnakeCanvas
                            loop
                            onFrameChange={handleFrameChange}
                            playing={Boolean(demo.champion?.replay)}
                            replay={demo.champion?.replay}
                            restartKey={demo.showcaseEpoch}
                            speed={demo.config.speed}
                        />
                    </div>
                    <NetworkPanel
                        genome={demo.champion?.genome}
                        input={liveInput}
                        inputLabels={SNAKE_INPUT_LABELS}
                        outputLabels={SNAKE_OUTPUT_LABELS}
                        subtitle="節點亮度跟住重播每一格嘅前向運算；熱圖係冠軍權重。"
                        title="貪食蛇網絡"
                        topology={SNAKE_TOPOLOGY}
                    />
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="食物平方獎勵 + 存活步數 + 接近食物塑造"
                        genome="Brain.js 10 → 12 → 3 網絡嘅所有權重同偏差"
                        inputs="前、左、右危險；食物相對位置；方向 one-hot；身體長度"
                        outputs="左轉、直行、右轉"
                        termination="撞牆、撞自己、長時間食唔到食物，或步數上限"
                    />
                </main>
                <aside className="demo-sidebar">
                    <DemoControls demo={demo}>
                        <GenomeTransfer
                            fitness={demo.champion?.fitness}
                            genome={demo.champion?.genome}
                            onImport={handleImportGenome}
                            onMessage={setTransferMessage}
                            score={demo.champion?.replay.score}
                            steps={demo.champion?.replay.steps}
                            topic="snake"
                            topology={SNAKE_TOPOLOGY}
                        />
                        {transferMessage ? <p className={transferMessage.type === "error" ? "error-message" : "status-message"}>{transferMessage.text}</p> : null}
                    </DemoControls>
                </aside>
            </div>
        </DemoShell>
    );
});

interface DemoShellProps {
    title: string;
    description: string;
    accent: string;
    icon: React.ReactNode;
    children: React.ReactNode;
}

export const DemoShell = React.memo<DemoShellProps>(props => {
    return (
        <div className={`demo-view accent-${props.accent}`}>
            <header className="demo-header">
                <div className="demo-icon">{props.icon}</div>
                <div>
                    <p className="eyebrow">即時演化實驗</p>
                    <h2>{props.title}</h2>
                    <p>{props.description}</p>
                </div>
            </header>
            {props.children}
        </div>
    );
});
