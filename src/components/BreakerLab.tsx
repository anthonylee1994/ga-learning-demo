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
        <DemoShell accent="breaker" description="Matter.js 固定時間步重播物理世界；AI 要從球速同位置預測下一次接球點。" icon={<Blocks size={20} strokeWidth={1.5} />} title="撞磚 · 神經演化">
            <div className="workspace-grid">
                <main className="demo-main">
                    <Metrics
                        extra={[
                            {label: "清磚數", value: String(demo.champion?.replay.bricksCleared ?? 0)},
                            {label: "接球次數", value: String(demo.champion?.replay.hits ?? 0)},
                        ]}
                        stats={demo.stats}
                    />
                    <div className="simulation-stage breaker-stage">
                        <div className="stage-overlay">
                            <span>Matter.js · 60 Hz</span>
                            <span>{demo.champion?.replay ? (demo.status === "running" ? "冠軍循環重播 · 進化中" : "冠軍循環重播") : "未有冠軍"}</span>
                        </div>
                        <BreakerCanvas
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
                        inputLabels={BREAKER_INPUT_LABELS}
                        outputLabels={BREAKER_OUTPUT_LABELS}
                        subtitle="節點亮度跟住重播每一格嘅前向運算；熱圖係冠軍權重。"
                        title="撞磚網絡"
                        topology={BREAKER_TOPOLOGY}
                    />
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="清磚分數 + 接球次數 + 存活時間 + 全清獎勵"
                        genome="Brain.js 8 → 12 → 3 網絡嘅所有權重同偏差"
                        inputs="擋板/球位置、球速、最近磚塊方向、剩餘比例"
                        outputs="向左、停住、向右"
                        termination="球跌出底部、清晒 45 塊磚，或物理步數上限"
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
