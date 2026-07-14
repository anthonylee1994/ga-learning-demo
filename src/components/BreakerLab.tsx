import React from "react";
import {Blocks} from "lucide-react";
import {useEvolutionDemo} from "../hooks/useEvolutionDemo";
import type {BreakerReplay, GAConfig} from "../lib/types";
import {ApplicationPanel} from "./ApplicationPanel";
import {BreakerCanvas} from "./BreakerCanvas";
import {DemoControls} from "./DemoControls";
import {FitnessChart} from "./FitnessChart";
import {Metrics} from "./Metrics";
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
                            playing={demo.status === "running" || demo.status === "paused"}
                            replay={demo.champion?.replay}
                            restartKey={demo.showcaseEpoch}
                            speed={demo.config.speed}
                        />
                    </div>
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
                    <DemoControls demo={demo} />
                </aside>
            </div>
        </DemoShell>
    );
});
