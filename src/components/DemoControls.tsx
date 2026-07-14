import React from "react";
import {Button, Card, Chip, Tooltip} from "@heroui/react";
import {Info, Pause, Play, RotateCcw} from "lucide-react";
import type {EvolutionDemoState} from "../hooks/useEvolutionDemo";

/** Only the fields DemoControls actually touches — keeps labs free of loadChampion variance issues. */
type DemoControlsDemo = Pick<EvolutionDemoState<unknown>, "config" | "setConfig" | "status" | "error" | "start" | "pause" | "reset">;

interface DemoControlsProps {
    demo: DemoControlsDemo;
    disabled?: boolean;
    /** Lab-specific extra controls (e.g. stock lab's NN toggle)，排喺隨機種子之後。 */
    children?: React.ReactNode;
}

export const DemoControls = React.memo<DemoControlsProps>(({demo, disabled, children}) => {
    const updateNumber = (key: "populationSize" | "mutationRate" | "speed", value: number) => {
        demo.setConfig(current => ({...current, [key]: value}));
    };

    return (
        <Card className="control-panel rounded-lg" variant="default">
            <Card.Header className="flex-row items-center justify-between">
                <div>
                    <Card.Title className="text-base">訓練控制</Card.Title>
                    <Card.Description>調整下一代嘅演化壓力</Card.Description>
                </div>
                <Chip color={demo.status === "running" ? "success" : demo.status === "paused" ? "warning" : "default"} size="sm" variant="soft">
                    {demo.status === "running" ? "運行中" : demo.status === "paused" ? "已暫停" : "待機"}
                </Chip>
            </Card.Header>
            <Card.Content className="space-y-5">
                <div className="grid grid-cols-3 gap-2">
                    <Button isDisabled={disabled || demo.status === "running"} onPress={demo.start} size="sm">
                        <Play size={15} strokeWidth={1.5} />
                        開始
                    </Button>
                    <Button isDisabled={demo.status !== "running"} onPress={demo.pause} size="sm" variant="secondary">
                        <Pause size={15} strokeWidth={1.5} />
                        暫停
                    </Button>
                    <Tooltip delay={250}>
                        <Button onPress={demo.reset} size="sm" variant="tertiary">
                            <RotateCcw size={15} strokeWidth={1.5} />
                            重設
                        </Button>
                        <Tooltip.Content showArrow>清除目前世代，但保留瀏覽器其他實驗資料。</Tooltip.Content>
                    </Tooltip>
                </div>

                <ControlSlider
                    description="個體越多，搜尋範圍越廣，但每一代需要更多運算。"
                    label="族群大小"
                    max={80}
                    min={12}
                    onChange={value => updateNumber("populationSize", value)}
                    step={2}
                    value={demo.config.populationSize}
                />
                <ControlSlider
                    description="太低容易早熟收斂；太高會破壞已學到嘅結構。"
                    label="突變率"
                    max={0.4}
                    min={0.01}
                    onChange={value => updateNumber("mutationRate", value)}
                    step={0.01}
                    value={demo.config.mutationRate}
                />
                <ControlSlider
                    description="調高會縮短代與代之間嘅等待；拉滿（5）= 唔再刻意等，盡量全速訓練。"
                    label="播放速度"
                    max={5}
                    min={1}
                    onChange={value => updateNumber("speed", value)}
                    step={1}
                    value={demo.config.speed}
                />
                <label className="control-field">
                    <span className="control-label">隨機種子</span>
                    <input
                        aria-label="隨機種子"
                        className="number-input"
                        min={1}
                        onChange={event => demo.setConfig(current => ({...current, seed: Number(event.target.value) || 1}))}
                        type="number"
                        value={demo.config.seed}
                    />
                </label>
                {children}
                {demo.error ? <p className="error-message">{demo.error}</p> : null}
            </Card.Content>
        </Card>
    );
});

interface ControlSliderProps {
    label: string;
    description: string;
    min: number;
    max: number;
    step: number;
    value: number;
    onChange: (value: number) => void;
}

const ControlSlider = React.memo<ControlSliderProps>(props => {
    return (
        <label className="control-field">
            <span className="control-label">
                <span>{props.label}</span>
                <span className="text-foreground inline-flex items-center gap-1 font-mono text-xs">
                    {props.step < 1 ? props.value.toFixed(2) : props.value}
                    <Tooltip delay={250}>
                        <span aria-label={`${props.label} 說明`} className="help-icon" role="button" tabIndex={0}>
                            <Info size={13} strokeWidth={1.5} />
                        </span>
                        <Tooltip.Content showArrow className="max-w-60 text-xs">
                            {props.description}
                        </Tooltip.Content>
                    </Tooltip>
                </span>
            </span>
            <input
                aria-label={props.label}
                className="range-input"
                max={props.max}
                min={props.min}
                onChange={event => props.onChange(Number(event.target.value))}
                step={props.step}
                type="range"
                value={props.value}
            />
        </label>
    );
});
