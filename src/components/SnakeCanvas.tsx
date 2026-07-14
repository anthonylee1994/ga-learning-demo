import React from "react";
import type {SnakeFrame, SnakeReplay} from "../lib/types";

const TERMINAL_HOLD_MS = 900;

interface Props {
    replay?: SnakeReplay;
    speed: number;
    /** Advance frames when true. */
    playing?: boolean;
    /** After the final frame (game lost / finished), restart from 0. If false, freeze on last frame. */
    loop?: boolean;
    /** Change to force restart from frame 0 (e.g. pause showcase of latest champion). */
    restartKey?: number | string;
    /** Fires whenever the visible frame changes (for live network activation). */
    onFrameChange?: (frame: SnakeFrame | null, frameIndex: number) => void;
}

export const SnakeCanvas = React.memo<Props>(
    ({
        replay,
        speed,
        playing = true,
        loop = true,
        restartKey = 0,
        onFrameChange,
    }: {
        replay?: SnakeReplay;
        speed: number;
        /** Advance frames when true. */
        playing?: boolean;
        /** After the final frame (game lost / finished), restart from 0. If false, freeze on last frame. */
        loop?: boolean;
        /** Change to force restart from frame 0 (e.g. pause showcase of latest champion). */
        restartKey?: number | string;
        onFrameChange?: (frame: SnakeFrame | null, frameIndex: number) => void;
    }) => {
        const canvasRef = React.useRef<HTMLCanvasElement>(null);
        const [frameIndex, setFrameIndex] = React.useState(0);
        const onFrameChangeRef = React.useRef(onFrameChange);
        onFrameChangeRef.current = onFrameChange;

        React.useEffect(() => {
            setFrameIndex(0);
        }, [replay, restartKey]);

        React.useEffect(() => {
            if (!playing || !replay?.frames.length) {
                return;
            }
            const frameMs = Math.max(12, 80 - speed * 14);
            const currentFrame = replay.frames[frameIndex];
            const delay = currentFrame?.terminal && loop ? TERMINAL_HOLD_MS : frameMs;
            const timer = window.setTimeout(() => {
                setFrameIndex(current => {
                    const last = replay.frames.length - 1;
                    if (last < 0) {
                        return 0;
                    }
                    if (current >= last) {
                        return loop ? 0 : last;
                    }
                    return current + 1;
                });
            }, delay);
            return () => window.clearTimeout(timer);
        }, [frameIndex, replay, speed, playing, loop]);

        React.useEffect(() => {
            const frame = replay?.frames[frameIndex] ?? null;
            onFrameChangeRef.current?.(frame, frameIndex);
        }, [frameIndex, replay]);

        React.useEffect(() => {
            const canvas = canvasRef.current;
            const context = canvas?.getContext("2d");
            if (!canvas || !context) {
                return;
            }
            const size = canvas.width;
            const cell = size / 20;
            context.fillStyle = "#0a0d0f";
            context.fillRect(0, 0, size, size);
            context.strokeStyle = "#151a1e";
            context.lineWidth = 1;
            for (let index = 0; index <= 20; index += 1) {
                context.beginPath();
                context.moveTo(index * cell, 0);
                context.lineTo(index * cell, size);
                context.stroke();
                context.beginPath();
                context.moveTo(0, index * cell);
                context.lineTo(size, index * cell);
                context.stroke();
            }

            const frame = replay?.frames[frameIndex];
            if (!frame) {
                context.fillStyle = "#78818b";
                context.font = "500 18px Inter, sans-serif";
                context.textAlign = "center";
                context.fillText("等待第一個冠軍", size / 2, size / 2);
                return;
            }
            context.fillStyle = "#ef6262";
            context.fillRect(frame.food.x * cell + 5, frame.food.y * cell + 5, cell - 10, cell - 10);
            frame.snake.forEach((part, index) => {
                context.fillStyle = index === 0 ? "#b7f5ca" : `rgba(88, 214, 141, ${Math.max(0.38, 1 - index * 0.035)})`;
                context.fillRect(part.x * cell + 2, part.y * cell + 2, cell - 4, cell - 4);
            });
            if (frame.terminal) {
                drawTerminalOverlay(context, size, frame.terminal === "collision" ? "遊戲結束" : "本局完結");
            }
        }, [frameIndex, replay]);

        return (
            <canvas
                aria-label="貪食蛇冠軍重播"
                className="simulation-canvas square"
                data-frame-index={frameIndex}
                data-loop={loop}
                data-playing={playing}
                data-terminal={replay?.frames[frameIndex]?.terminal ?? ""}
                height={600}
                ref={canvasRef}
                width={600}
            />
        );
    }
);

function drawTerminalOverlay(context: CanvasRenderingContext2D, size: number, label: string): void {
    context.fillStyle = "rgba(5, 8, 10, 0.72)";
    context.fillRect(0, size / 2 - 38, size, 76);
    context.fillStyle = "#f1f3f5";
    context.font = "600 22px Inter, sans-serif";
    context.textAlign = "center";
    context.fillText(label, size / 2, size / 2 + 7);
}
