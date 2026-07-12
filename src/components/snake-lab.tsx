import React from "react";
import {Dna} from "lucide-react";
import {useEvolutionDemo} from "../hooks/use-evolution-demo";
import type {GAConfig, SnakeReplay} from "../lib/types";
import {ApplicationPanel} from "./application-panel";
import {DemoControls} from "./demo-controls";
import {FitnessChart} from "./fitness-chart";
import {Metrics} from "./metrics";
import {SnakeCanvas} from "./snake-canvas";

const DEFAULT_CONFIG: GAConfig = {
    populationSize: 36,
    mutationRate: 0.12,
    mutationScale: 0.24,
    eliteRate: 0.08,
    tournamentSize: 4,
    seed: 137,
    speed: 3,
};

export const SnakeLab = React.memo(() => {
    const demo = useEvolutionDemo<undefined, SnakeReplay>({
        topic: "snake",
        createWorker: () => new Worker(new URL("../workers/snake.worker.ts", import.meta.url), {type: "module"}),
        defaultConfig: DEFAULT_CONFIG,
    });
    return (
        <DemoShell accent="snake" description="一個 10 → 12 → 3 嘅 Brain.js network，靠食物、存活同距離 shaping 學識轉彎。" icon={<Dna size={20} strokeWidth={1.5} />} title="Snake Neuroevolution">
            <div className="workspace-grid">
                <main className="demo-main">
                    <Metrics
                        extra={[
                            {label: "Champion score", value: String(demo.champion?.replay.score ?? 0)},
                            {label: "Survival steps", value: String(demo.champion?.replay.steps ?? 0)},
                        ]}
                        stats={demo.stats}
                    />
                    <div className="simulation-stage snake-stage">
                        <div className="stage-overlay">
                            <span>20 × 20 grid</span>
                            <span>{demo.status === "paused" ? "暫停 · 最新 champion 玩到輸再重開" : "Champion replay"}</span>
                        </div>
                        <SnakeCanvas
                            loop={demo.status === "paused"}
                            playing={demo.status === "running" || demo.status === "paused"}
                            replay={demo.champion?.replay}
                            restartKey={demo.showcaseEpoch}
                            speed={demo.config.speed}
                        />
                    </div>
                    <FitnessChart history={demo.history} />
                    <ApplicationPanel
                        fitness="食物平方獎勵 + 存活步數 + 接近食物 shaping"
                        genome="Brain.js 10 → 12 → 3 network 嘅所有 weights 與 biases"
                        inputs="前、左、右危險；食物相對位置；方向 one-hot；身體長度"
                        outputs="左轉、直行、右轉"
                        termination="撞牆、撞自己、長時間食唔到食物，或 360 steps"
                    />
                </main>
                <aside className="demo-sidebar">
                    <DemoControls demo={demo} />
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

export const DemoShell = React.memo<DemoShellProps>((props) => {
    return (
        <div className={`demo-view accent-${props.accent}`}>
            <header className="demo-header">
                <div className="demo-icon">{props.icon}</div>
                <div>
                    <p className="eyebrow">Live evolution experiment</p>
                    <h2>{props.title}</h2>
                    <p>{props.description}</p>
                </div>
            </header>
            {props.children}
        </div>
    );
});
