import React from "react";
import type {BreakerReplay} from "../lib/types";

const TERMINAL_HOLD_MS = 900;

interface Props {
    replay?: BreakerReplay;
    speed: number;
    /** Advance frames when true. */
    playing?: boolean;
    /** After the final frame (game lost / finished), restart from 0. If false, freeze on last frame. */
    loop?: boolean;
    /** Change to force restart from frame 0 (e.g. pause showcase of latest champion). */
    restartKey?: number | string;
}

export const BreakerCanvas = React.memo<Props>(({ replay, speed, playing = true, loop = true, restartKey = 0 }) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    const [frameIndex, setFrameIndex] = React.useState(0);

    React.useEffect(() => {
        setFrameIndex(0);
    }, [replay, restartKey]);

    React.useEffect(() => {
        if (!playing || !replay?.frames.length) {
            return;
        }
        const frameMs = Math.max(24, 110 - speed * 15);
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
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d");
        if (!canvas || !context) {
            return;
        }
        context.fillStyle = "#0a0d0f";
        context.fillRect(0, 0, 560, 420);
        context.strokeStyle = "#252b32";
        context.strokeRect(0.5, 0.5, 559, 419);
        const frame = replay?.frames[frameIndex];
        if (!frame) {
            context.fillStyle = "#78818b";
            context.font = "500 17px Inter, sans-serif";
            context.textAlign = "center";
            context.fillText("等待第一個 champion", 280, 210);
            return;
        }
        const brickColors = ["#e36f5b", "#e7b955", "#63c6a1", "#5da6d9", "#b38bd4"];
        frame.bricks.forEach(brick => {
            if (!brick.active) {
                return;
            }
            context.fillStyle = brickColors[Math.floor((brick.y - 50) / 25)] ?? "#e7b955";
            context.fillRect(brick.x - 26, brick.y - 8, 52, 16);
        });
        context.fillStyle = "#d9dde3";
        context.fillRect(frame.paddleX - 46, 386, 92, 12);
        context.beginPath();
        context.arc(frame.ball.x, frame.ball.y, 7, 0, Math.PI * 2);
        context.fillStyle = "#ffffff";
        context.fill();
        if (frame.terminal) {
            drawTerminalOverlay(context, frame.terminal === "cleared" ? "ROUND COMPLETE" : "GAME OVER");
        }
    }, [frameIndex, replay]);

    return (
        <canvas
            aria-label="Block Breaker champion replay"
            className="simulation-canvas breaker"
            data-frame-index={frameIndex}
            data-loop={loop}
            data-playing={playing}
            data-terminal={replay?.frames[frameIndex]?.terminal ?? ""}
            height={420}
            ref={canvasRef}
            width={560}
        />
    );
});

function drawTerminalOverlay(context: CanvasRenderingContext2D, label: string): void {
    context.fillStyle = "rgba(5, 8, 10, 0.72)";
    context.fillRect(0, 172, 560, 76);
    context.fillStyle = "#f1f3f5";
    context.font = "600 20px Inter, sans-serif";
    context.textAlign = "center";
    context.fillText(label, 280, 217);
}
