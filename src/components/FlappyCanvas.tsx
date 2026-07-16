import React from "react";
import birdSpriteUrl from "../assets/flappy-bird.png";
import {FLAPPY_HEIGHT, FLAPPY_WIDTH} from "../domains/flappy/simulation";
import type {FlappyFrame, FlappyReplay} from "../lib/types";

const TERMINAL_HOLD_MS = 900;
/** 同 simulation 碰撞盒一致 */
const BIRD_X = 78;
const PIPE_WIDTH = 48;
const GROUND_H = 56;
/** 畫面上雀嘅顯示闊（邏輯 px）；跟住參考 sprite 比例 */
const BIRD_DRAW_W = 42;
const BIRD_DRAW_H = 30;

const birdSprite = new Image();
birdSprite.src = birdSpriteUrl;

/** 經典 Flappy 配色（參考原版） */
const SKY = "#4EC0CA";
const CLOUD = "#FFFFFF";
const CLOUD_EDGE = "#E8F6F8";
const CITY = "#D5E99A";
const CITY_EDGE = "#C5DB88";
const BUSH = "#73BF2E";
const BUSH_DARK = "#5DA322";
const GROUND = "#DED895";
const GROUND_EDGE = "#E5C65A";
const GRASS = "#73BF2E";
const PIPE = "#73BF2E";
const PIPE_LIGHT = "#96E05A";
const PIPE_DARK = "#548C2F";
const PIPE_LIP = "#5DA322";

interface Props {
    replay?: FlappyReplay;
    speed: number;
    playing?: boolean;
    loop?: boolean;
    restartKey?: number | string;
    onFrameChange?: (frame: FlappyFrame | null, frameIndex: number) => void;
}

export const FlappyCanvas = React.memo<Props>(({replay, speed, playing = true, loop = true, restartKey = 0, onFrameChange}) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    const [frameIndex, setFrameIndex] = React.useState(0);
    const [spriteReady, setSpriteReady] = React.useState(() => birdSprite.complete && birdSprite.naturalWidth > 0);
    const onFrameChangeRef = React.useRef(onFrameChange);
    onFrameChangeRef.current = onFrameChange;

    React.useEffect(() => {
        if (birdSprite.complete && birdSprite.naturalWidth > 0) {
            setSpriteReady(true);
            return;
        }
        const onLoad = () => setSpriteReady(true);
        birdSprite.addEventListener("load", onLoad);
        return () => birdSprite.removeEventListener("load", onLoad);
    }, []);

    React.useEffect(() => {
        setFrameIndex(0);
    }, [replay, restartKey]);

    React.useEffect(() => {
        if (!playing || !replay?.frames.length) {
            return;
        }
        const frameMs = Math.max(12, 72 - speed * 12);
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

        const width = canvas.width;
        const height = canvas.height;
        const sx = width / FLAPPY_WIDTH;
        const sy = height / FLAPPY_HEIGHT;
        // 背景 parallax：用 frame step 做輕微滾動
        const scroll = ((replay?.frames[frameIndex]?.step ?? 0) * 0.6) % FLAPPY_WIDTH;

        context.imageSmoothingEnabled = false;
        drawBackground(context, width, height, sx, sy, scroll);

        const frame = replay?.frames[frameIndex];
        if (frame) {
            for (const pipe of frame.pipes) {
                drawPipe(context, pipe.x * sx, (pipe.gapY - pipe.gapHeight / 2) * sy, (pipe.gapY + pipe.gapHeight / 2) * sy, PIPE_WIDTH * sx, height, sy);
            }
        }

        // 前景草 + 地面：有冇冠軍都要畫，唔好 idle 時缺一塊地
        drawBushes(context, width, height, sx, sy, scroll * 1.15, 1, 0.42);
        drawGround(context, width, height, sy);

        if (!frame) {
            context.fillStyle = "rgba(20, 40, 48, 0.45)";
            context.fillRect(0, height / 2 - 36, width, 72);
            context.fillStyle = "#FFFFFF";
            context.font = "600 18px Inter, sans-serif";
            context.textAlign = "center";
            context.fillText("等待第一個冠軍", width / 2, height / 2 + 6);
            return;
        }

        // 雀（跟用戶提供嘅像素 design sprite）
        drawBird(context, BIRD_X * sx, frame.birdY * sy, Math.min(sx, sy), frame.birdVy);

        // 分數（白字 + 深色描邊，原版感覺）
        drawScore(context, width, frame.score, sy);

        if (frame.terminal) {
            context.fillStyle = "rgba(0, 0, 0, 0.45)";
            context.fillRect(0, height / 2 - 40, width, 80);
            context.fillStyle = "#FFFFFF";
            context.font = "700 24px Inter, sans-serif";
            context.textAlign = "center";
            context.strokeStyle = "rgba(0,0,0,0.55)";
            context.lineWidth = 4;
            const label = frame.terminal === "crash" ? "撞到了" : "本局完結";
            context.strokeText(label, width / 2, height / 2 + 8);
            context.fillText(label, width / 2, height / 2 + 8);
        }
    }, [frameIndex, replay, spriteReady]);

    return (
        <canvas
            aria-label="Flappy Bird 冠軍重播"
            className="simulation-canvas flappy"
            data-frame-index={frameIndex}
            data-loop={loop}
            data-playing={playing}
            data-terminal={replay?.frames[frameIndex]?.terminal ?? ""}
            height={FLAPPY_HEIGHT}
            ref={canvasRef}
            width={FLAPPY_WIDTH}
        />
    );
});

function drawBackground(context: CanvasRenderingContext2D, width: number, height: number, sx: number, sy: number, scroll: number): void {
    context.fillStyle = SKY;
    context.fillRect(0, 0, width, height);

    // 雲
    const clouds = [
        {x: 40, y: 48, s: 1},
        {x: 180, y: 72, s: 0.75},
        {x: 280, y: 40, s: 1.1},
    ];
    for (const cloud of clouds) {
        const cx = (((cloud.x - scroll * 0.25 + FLAPPY_WIDTH * 2) % (FLAPPY_WIDTH + 80)) - 40) * sx;
        drawCloud(context, cx, cloud.y * sy, cloud.s * sx);
    }

    const groundTop = height - GROUND_H * sy;

    // 遠景草丘先畫（矮一截），之後再畫樓，避免蓋住樓身
    drawBushes(context, width, height, sx, sy, scroll * 0.55, 0.75, 0.55);

    // 城市剪影：貼地 + 加高，畫喺草上面
    const buildings = [48, 72, 40, 86, 58, 78, 44, 66, 82, 52, 70, 42, 76, 56];
    let bx = -((scroll * 0.35) % 36) * sx;
    for (const h of buildings) {
        const bw = 26 * sx;
        const bh = h * sy;
        const top = groundTop - bh;
        context.fillStyle = CITY;
        context.fillRect(bx, top, bw - 2 * sx, bh);
        context.fillStyle = CITY_EDGE;
        context.fillRect(bx, top, bw - 2 * sx, 3 * sy);
        bx += 28 * sx;
    }
}

function drawCloud(context: CanvasRenderingContext2D, x: number, y: number, scale: number): void {
    context.fillStyle = CLOUD;
    const r = 14 * scale;
    context.beginPath();
    context.arc(x, y, r * 1.1, 0, Math.PI * 2);
    context.arc(x + r * 1.2, y - r * 0.2, r * 1.35, 0, Math.PI * 2);
    context.arc(x + r * 2.4, y, r, 0, Math.PI * 2);
    context.arc(x + r * 1.1, y + r * 0.35, r * 0.95, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = CLOUD_EDGE;
    context.beginPath();
    context.ellipse(x + r * 1.1, y + r * 0.55, r * 1.6, r * 0.35, 0, 0, Math.PI * 2);
    context.fill();
}

function drawBushes(context: CanvasRenderingContext2D, _width: number, height: number, sx: number, sy: number, scroll: number, alpha = 1, sizeScale = 1): void {
    const baseY = height - GROUND_H * sy;
    context.save();
    context.globalAlpha = alpha;
    const offset = -((scroll * 0.9) % 48) * sx;
    const s = sizeScale;
    for (let i = -1; i < 12; i += 1) {
        const x = offset + i * 48 * sx;
        context.fillStyle = BUSH_DARK;
        context.beginPath();
        context.ellipse(x + 10 * sx, baseY, 22 * sx * s, 16 * sy * s, 0, 0, Math.PI * 2);
        context.ellipse(x + 28 * sx, baseY - 4 * sy * s, 20 * sx * s, 18 * sy * s, 0, 0, Math.PI * 2);
        context.ellipse(x + 44 * sx, baseY, 18 * sx * s, 14 * sy * s, 0, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = BUSH;
        context.beginPath();
        context.ellipse(x + 12 * sx, baseY - 2 * sy * s, 18 * sx * s, 13 * sy * s, 0, 0, Math.PI * 2);
        context.ellipse(x + 30 * sx, baseY - 6 * sy * s, 16 * sx * s, 14 * sy * s, 0, 0, Math.PI * 2);
        context.fill();
    }
    context.restore();
}

function drawGround(context: CanvasRenderingContext2D, width: number, height: number, sy: number): void {
    const gy = height - GROUND_H * sy;
    context.fillStyle = GRASS;
    context.fillRect(0, gy, width, 8 * sy);
    context.fillStyle = GROUND_EDGE;
    context.fillRect(0, gy + 8 * sy, width, 4 * sy);
    context.fillStyle = GROUND;
    context.fillRect(0, gy + 12 * sy, width, GROUND_H * sy);
    // 泥面點點
    context.fillStyle = "rgba(180, 150, 70, 0.35)";
    for (let i = 0; i < 18; i += 1) {
        context.fillRect((i * 37 + 8) % width, gy + (18 + (i % 5) * 6) * sy, 6, 3 * sy);
    }
}

function drawPipe(context: CanvasRenderingContext2D, x: number, gapTop: number, gapBottom: number, pw: number, height: number, sy: number): void {
    const lipH = 16 * sy;
    const lipPad = 4;
    const bodyW = pw;
    const lipW = pw + lipPad * 2;

    // 上管身
    fillPipeBody(context, x, 0, bodyW, Math.max(0, gapTop - lipH));
    // 上管口
    fillPipeLip(context, x - lipPad, gapTop - lipH, lipW, lipH);

    // 下管身（唔蓋地面）
    const groundY = height - GROUND_H * sy;
    const bodyTop = gapBottom + lipH;
    fillPipeBody(context, x, bodyTop, bodyW, Math.max(0, groundY - bodyTop));
    // 下管口
    fillPipeLip(context, x - lipPad, gapBottom, lipW, lipH);
}

function fillPipeBody(context: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    if (h <= 0) {
        return;
    }
    context.fillStyle = PIPE;
    context.fillRect(x, y, w, h);
    // 高光條
    context.fillStyle = PIPE_LIGHT;
    context.fillRect(x + 4, y, 6, h);
    context.fillStyle = PIPE_DARK;
    context.fillRect(x + w - 5, y, 4, h);
    context.strokeStyle = PIPE_DARK;
    context.lineWidth = 2;
    context.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

function fillPipeLip(context: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    context.fillStyle = PIPE_LIP;
    context.fillRect(x, y, w, h);
    context.fillStyle = PIPE_LIGHT;
    context.fillRect(x + 5, y + 2, 7, h - 4);
    context.fillStyle = PIPE_DARK;
    context.fillRect(x + w - 7, y + 2, 5, h - 4);
    context.strokeStyle = PIPE_DARK;
    context.lineWidth = 2;
    context.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

/**
 * 用戶提供嘅像素黃雀 sprite：深紫 outline、黃身、白眼、紅咀、左翼。
 * 跟住 vy 旋轉；imageSmoothing 關咗保持像素感。
 */
function drawBird(context: CanvasRenderingContext2D, x: number, y: number, scale: number, birdVy: number): void {
    const tilt = Math.max(-0.55, Math.min(0.9, birdVy * 0.09));
    const dw = BIRD_DRAW_W * scale;
    const dh = BIRD_DRAW_H * scale;

    context.save();
    context.translate(x, y);
    context.rotate(tilt);
    context.imageSmoothingEnabled = false;

    if (birdSprite.complete && birdSprite.naturalWidth > 0) {
        context.drawImage(birdSprite, -dw / 2, -dh / 2, dw, dh);
    } else {
        // sprite 未 load 前嘅 fallback（同樣 palette）
        context.fillStyle = "#F7D031";
        context.beginPath();
        context.ellipse(0, 0, dw * 0.42, dh * 0.38, 0, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = "#E8F8E0";
        context.beginPath();
        context.ellipse(dw * 0.12, -dh * 0.05, dw * 0.16, dh * 0.18, 0, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = "#D80000";
        context.fillRect(dw * 0.18, dh * 0.02, dw * 0.28, dh * 0.18);
    }

    context.restore();
}

function drawScore(context: CanvasRenderingContext2D, width: number, score: number, sy: number): void {
    const text = String(score);
    context.font = "800 36px Inter, ui-sans-serif, system-ui, sans-serif";
    context.textAlign = "center";
    context.lineJoin = "round";
    context.strokeStyle = "rgba(0,0,0,0.55)";
    context.lineWidth = 6;
    context.strokeText(text, width / 2, 48 * sy);
    context.fillStyle = "#FFFFFF";
    context.fillText(text, width / 2, 48 * sy);
}
