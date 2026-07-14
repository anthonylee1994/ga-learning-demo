import React from "react";
import {Blocks} from "lucide-react";
import {BREAKER_INPUT_LABELS, BREAKER_OUTPUT_LABELS, BREAKER_TOPOLOGY, buildBreakerInputFromFrame, createBreakerReplay, evaluateBreakerGenome} from "../domains/breaker/simulation";
import {useEvolutionDemo} from "../hooks/useEvolutionDemo";
import type {BreakerFrame, BreakerReplay, GAConfig, Genome} from "../lib/types";
import {ApplicationPanel} from "./ApplicationPanel";
import {BreakerCanvas} from "./BreakerCanvas";
import {DemoControls} from "./DemoControls";
import {FitnessChart} from "./FitnessChart";
import {GenomeTransfer} from "./GenomeTransfer";
import {Metrics} from "./Metrics";
import {NetworkPanel} from "./NetworkPanel";
import {DemoShell} from "./SnakeLab";

const DEFAULT_CONFIG: GAConfig = {
    populationSize: 12,
    mutationRate: 0.14,
    mutationScale: 0.26,
    eliteRate: 0.1,
    seed: Math.round(Math.random() * 1_000_000),
    speed: 5,
};

export const BreakerLab = React.memo(() => {
    const demo = useEvolutionDemo<undefined, BreakerReplay>({
        topic: "breaker",
        createWorker: () => new Worker(new URL("../workers/breaker.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: {
            ...DEFAULT_CONFIG,
            seed: Math.round(Math.random() * 1_000_000),
        },
    });
    const [liveInput, setLiveInput] = React.useState<number[] | null>(null);
    const [transferMessage, setTransferMessage] = React.useState<{type: "status" | "error"; text: string} | null>(null);

    const handleFrameChange = (frame: BreakerFrame | null) => {
        if (!frame) {
            setLiveInput(null);
            return;
        }
        setLiveInput(buildBreakerInputFromFrame(frame));
    };

    const handleImportGenome = (genome: Genome) => {
        const replay = createBreakerReplay(genome);
        const fitness = evaluateBreakerGenome(genome);
        demo.loadChampion({genome, replay, fitness});
    };

    React.useEffect(() => {
        if (!demo.champion) {
            setLiveInput(null);
        }
    }, [demo.champion]);

    return (
        <DemoShell
            accent="breaker"
            description="Matter.js 固定 timestep 重播物理世界；AI 要從球速同位置預測下一次接球點。"
            icon={<Blocks size={20} strokeWidth={1.5} />}
            title="Block Breaker Evolution"
        >
            <div className="workspace-grid">
                <main className="demo-main">
                    <Metrics
                        extra={[
                            {label: "Bricks cleared", value: String(demo.champion?.replay.bricksCleared ?? 0)},
                            {label: "Paddle hits", value: String(demo.champion?.replay.hits ?? 0)},
                        ]}
                        stats={demo.stats}
                    />
                    <div className="simulation-stage breaker-stage">
                        <div className="stage-overlay">
                            <span>Matter.js · 60 Hz</span>
                            <span>{demo.status === "paused" ? "暫停 · 最新 champion 玩到輸再重開" : "Champion replay"}</span>
                        </div>
                        <BreakerCanvas
                            loop={demo.status === "paused"}
                            onFrameChange={handleFrameChange}
                            playing={demo.status === "running" || demo.status === "paused"}
                            replay={demo.champion?.replay}
                            restartKey={demo.showcaseEpoch}
                            speed={demo.config.speed}
                        />
                    </div>
                    <NetworkPanel
                        genome={demo.champion?.genome}
                        input={liveInput}
                        inputLabels={BREAKER_INPUT_LABELS}
                        outputLabels={BREAKER_OUTPUT_LABELS}
                        subtitle="節點亮度跟住 replay 每一 frame 嘅 forward pass；heatmap 係 champion weights。"
                        title="Breaker network"
                        topology={BREAKER_TOPOLOGY}
                    />
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="清除 bricks + 回球次數 + 存活時間 + clear bonus"
                        genome="Brain.js 8 → 12 → 3 network 嘅所有 weights 與 biases"
                        inputs="Paddle/ball 位置、ball velocity、最近 brick 方向、剩餘比例"
                        outputs="向左、停低、向右"
                        termination="Ball 跌出底部、清晒 45 塊 bricks，或 600 physics steps"
                    />
                </main>
                <aside className="demo-sidebar">
                    <DemoControls demo={demo}>
                        <GenomeTransfer
                            fitness={demo.champion?.fitness}
                            genome={demo.champion?.genome}
                            onImport={handleImportGenome}
                            onMessage={setTransferMessage}
                            score={demo.champion?.replay.bricksCleared}
                            steps={demo.champion?.replay.steps}
                            topic="breaker"
                            topology={BREAKER_TOPOLOGY}
                        />
                        {transferMessage ? <p className={transferMessage.type === "error" ? "error-message" : "status-message"}>{transferMessage.text}</p> : null}
                    </DemoControls>
                </aside>
            </div>
        </DemoShell>
    );
});
