import React from "react";
import {Bird} from "lucide-react";
import {buildFlappyInputFromFrame, createFlappyReplay, evaluateFlappyGenome, FLAPPY_INPUT_LABELS, FLAPPY_OUTPUT_LABELS, FLAPPY_TOPOLOGY} from "../domains/flappy/simulation";
import {useEvolutionDemo} from "../hooks/useEvolutionDemo";
import type {FlappyFrame, FlappyReplay, GAConfig, Genome} from "../lib/types";
import {ApplicationPanel} from "./ApplicationPanel";
import {DemoControls} from "./DemoControls";
import {FitnessChart} from "./FitnessChart";
import {FlappyCanvas} from "./FlappyCanvas";
import {GenomeTransfer} from "./GenomeTransfer";
import {Metrics} from "./Metrics";
import {NetworkPanel} from "./NetworkPanel";
import {DemoShell} from "./SnakeLab";

const DEFAULT_CONFIG: GAConfig = {
    populationSize: 40,
    mutationRate: 0.13,
    mutationScale: 0.24,
    eliteRate: 0.08,
    seed: Math.round(Math.random() * 1_000_000),
    speed: 5,
};

export const FlappyLab = React.memo(() => {
    const demo = useEvolutionDemo<undefined, FlappyReplay>({
        topic: "flappy",
        createWorker: () => new Worker(new URL("../workers/flappy.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: {
            ...DEFAULT_CONFIG,
            seed: Math.round(Math.random() * 1_000_000),
        },
        restoreChampion: genome => ({
            replay: createFlappyReplay(genome),
            fitness: evaluateFlappyGenome(genome),
        }),
    });
    const [liveInput, setLiveInput] = React.useState<number[] | null>(null);
    const [transferMessage, setTransferMessage] = React.useState<{type: "status" | "error"; text: string} | null>(null);

    const handleFrameChange = (frame: FlappyFrame | null) => {
        if (!frame) {
            setLiveInput(null);
            return;
        }
        setLiveInput(buildFlappyInputFromFrame(frame));
    };

    const handleImportGenome = (genome: Genome) => {
        const replay = createFlappyReplay(genome);
        const fitness = evaluateFlappyGenome(genome);
        demo.loadChampion({genome, replay, fitness});
    };

    React.useEffect(() => {
        if (!demo.champion) {
            setLiveInput(null);
        }
    }, [demo.champion]);

    return (
        <DemoShell
            accent="flappy"
            description="一個 6 → 10 → 2 嘅 Brain.js 網絡，靠通過水管數、存活同靠近縫心 shaping 學識拍翼。"
            icon={<Bird size={20} strokeWidth={1.5} />}
            title="Flappy Bird · 神經演化"
        >
            <div className="workspace-grid">
                <main className="demo-main">
                    <Metrics
                        extra={[
                            {label: "通過水管", value: String(demo.champion?.replay.score ?? 0)},
                            {label: "存活步數", value: String(demo.champion?.replay.steps ?? 0)},
                        ]}
                        stats={demo.stats}
                    />
                    <div className="simulation-stage flappy-stage">
                        <div className="stage-overlay">
                            <span>重力 · 水管滾動</span>
                            <span>{demo.champion?.replay ? (demo.status === "running" ? "冠軍循環重播 · 進化中" : "冠軍循環重播") : "未有冠軍"}</span>
                        </div>
                        <FlappyCanvas
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
                        inputLabels={FLAPPY_INPUT_LABELS}
                        outputLabels={FLAPPY_OUTPUT_LABELS}
                        subtitle="節點亮度跟住重播每一幀嘅前向運算；熱圖係冠軍權重。"
                        title="Flappy 網絡"
                        topology={FLAPPY_TOPOLOGY}
                    />
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="通過水管² + 存活步數 + 靠近縫心 shaping"
                        genome="Brain.js 6 → 10 → 2 網絡嘅所有權重同偏差"
                        inputs="鳥 y / vy、下條管距離、縫隙上沿／下沿、相對縫心"
                        outputs="拍翼、滑翔"
                        termination="撞地／撞頂／撞管，或步數上限；fitness = 3 個固定 seed 平均"
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
                            topic="flappy"
                            topology={FLAPPY_TOPOLOGY}
                        />
                        {transferMessage ? <p className={transferMessage.type === "error" ? "error-message" : "status-message"}>{transferMessage.text}</p> : null}
                    </DemoControls>
                </aside>
            </div>
        </DemoShell>
    );
});
