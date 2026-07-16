import React from "react";
import digit0 from "../assets/flappy-bird-sprites/0.png";
import digit1 from "../assets/flappy-bird-sprites/1.png";
import digit2 from "../assets/flappy-bird-sprites/2.png";
import digit3 from "../assets/flappy-bird-sprites/3.png";
import digit4 from "../assets/flappy-bird-sprites/4.png";
import digit5 from "../assets/flappy-bird-sprites/5.png";
import digit6 from "../assets/flappy-bird-sprites/6.png";
import digit7 from "../assets/flappy-bird-sprites/7.png";
import digit8 from "../assets/flappy-bird-sprites/8.png";
import digit9 from "../assets/flappy-bird-sprites/9.png";
import backgroundDayUrl from "../assets/flappy-bird-sprites/background-day.png";
import baseUrl from "../assets/flappy-bird-sprites/base.png";
import gameoverUrl from "../assets/flappy-bird-sprites/gameover.png";
import messageUrl from "../assets/flappy-bird-sprites/message.png";
import pipeGreenUrl from "../assets/flappy-bird-sprites/pipe-green.png";
import birdDownUrl from "../assets/flappy-bird-sprites/yellowbird-downflap.png";
import birdMidUrl from "../assets/flappy-bird-sprites/yellowbird-midflap.png";
import birdUpUrl from "../assets/flappy-bird-sprites/yellowbird-upflap.png";
import {FLAPPY_HEIGHT, FLAPPY_WIDTH} from "../domains/flappy/simulation";
import type {FlappyFrame, FlappyReplay} from "../lib/types";

const TERMINAL_HOLD_MS = 900;
/** 同 simulation 碰撞盒一致 */
const BIRD_X = 78;
const PIPE_WIDTH = 48;
/** 地面顯示高度（邏輯 px）— base sprite 縮放到呢度 */
const GROUND_H = 96;
/** 雀顯示尺寸（邏輯 px；原版 34×24） */
const BIRD_DRAW_W = 38;
const BIRD_DRAW_H = 27;
/** 分數數字高度（邏輯 px） */
const DIGIT_H = 36;

const digitUrls = [digit0, digit1, digit2, digit3, digit4, digit5, digit6, digit7, digit8, digit9] as const;
const birdUrls = [birdUpUrl, birdMidUrl, birdDownUrl] as const;

const sprites = {
    background: loadImage(backgroundDayUrl),
    base: loadImage(baseUrl),
    pipe: loadImage(pipeGreenUrl),
    gameover: loadImage(gameoverUrl),
    message: loadImage(messageUrl),
    digits: digitUrls.map(loadImage),
    birds: birdUrls.map(loadImage),
};

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
    const [spritesReady, setSpritesReady] = React.useState(() => allSpritesReady());
    const onFrameChangeRef = React.useRef(onFrameChange);
    onFrameChangeRef.current = onFrameChange;

    React.useEffect(() => {
        if (allSpritesReady()) {
            setSpritesReady(true);
            return;
        }
        const images = collectImages();
        const onLoad = () => {
            if (allSpritesReady()) {
                setSpritesReady(true);
            }
        };
        for (const image of images) {
            image.addEventListener("load", onLoad);
            image.addEventListener("error", onLoad);
        }
        return () => {
            for (const image of images) {
                image.removeEventListener("load", onLoad);
                image.removeEventListener("error", onLoad);
            }
        };
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
        const step = replay?.frames[frameIndex]?.step ?? 0;
        // 地面滾動速度跟 simulation 水管速度大致一致
        const scroll = (step * 2.6) % (sprites.base.naturalWidth || 336);

        context.imageSmoothingEnabled = false;

        drawBackground(context, width, height);

        const frame = replay?.frames[frameIndex];
        if (frame) {
            for (const pipe of frame.pipes) {
                drawPipe(context, pipe.x * sx, (pipe.gapY - pipe.gapHeight / 2) * sy, (pipe.gapY + pipe.gapHeight / 2) * sy, PIPE_WIDTH * sx, height, sy);
            }
        }

        // base 要蓋住水管下半截
        drawBase(context, width, height, sx, sy, scroll);

        if (!frame) {
            drawIdle(context, width, height, sy);
            return;
        }

        drawBird(context, BIRD_X * sx, frame.birdY * sy, Math.min(sx, sy), frame.birdVy, frame.step);
        drawScore(context, width, frame.score, sy);

        if (frame.terminal) {
            drawGameOver(context, width, height, sx);
        }
    }, [frameIndex, replay, spritesReady]);

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

function loadImage(src: string): HTMLImageElement {
    const image = new Image();
    image.src = src;
    return image;
}

function collectImages(): HTMLImageElement[] {
    return [sprites.background, sprites.base, sprites.pipe, sprites.gameover, sprites.message, ...sprites.digits, ...sprites.birds];
}

function allSpritesReady(): boolean {
    return collectImages().every(image => image.complete && image.naturalWidth > 0);
}

function isReady(image: HTMLImageElement): boolean {
    return image.complete && image.naturalWidth > 0;
}

function drawBackground(context: CanvasRenderingContext2D, width: number, height: number): void {
    if (isReady(sprites.background)) {
        context.drawImage(sprites.background, 0, 0, width, height);
        return;
    }
    context.fillStyle = "#4EC0CA";
    context.fillRect(0, 0, width, height);
}

/**
 * 原版 base 係橫向 tile；用 scroll 做出滾動感。
 */
function drawBase(context: CanvasRenderingContext2D, width: number, height: number, sx: number, sy: number, scroll: number): void {
    const baseH = GROUND_H * sy;
    const baseY = height - baseH;
    if (!isReady(sprites.base)) {
        context.fillStyle = "#DED895";
        context.fillRect(0, baseY, width, baseH);
        return;
    }

    // 按高度等比縮放，闊度跟住比例
    const srcW = sprites.base.naturalWidth;
    const srcH = sprites.base.naturalHeight;
    const tileW = (srcW / srcH) * baseH;
    const offset = -((scroll * sx) % tileW);
    for (let x = offset; x < width + tileW; x += tileW) {
        context.drawImage(sprites.base, x, baseY, tileW, baseH);
    }
}

/**
 * pipe-green 原圖：管口喺頂、管身向下。
 * 下管直接畫；上管垂直翻轉。
 */
function drawPipe(context: CanvasRenderingContext2D, x: number, gapTop: number, gapBottom: number, pw: number, height: number, sy: number): void {
    const groundY = height - GROUND_H * sy;

    if (!isReady(sprites.pipe)) {
        context.fillStyle = "#73BF2E";
        context.fillRect(x, 0, pw, Math.max(0, gapTop));
        context.fillRect(x, gapBottom, pw, Math.max(0, groundY - gapBottom));
        return;
    }

    const srcW = sprites.pipe.naturalWidth;
    const srcH = sprites.pipe.naturalHeight;

    // 下管：從 gapBottom 向下畫，高度最多去到地面
    const bottomH = Math.max(0, groundY - gapBottom);
    if (bottomH > 0) {
        const srcDrawH = Math.min(srcH, (bottomH / pw) * srcW);
        context.drawImage(sprites.pipe, 0, 0, srcW, srcDrawH, x, gapBottom, pw, bottomH);
    }

    // 上管：翻轉後管口朝下貼 gapTop
    const topH = Math.max(0, gapTop);
    if (topH > 0) {
        const srcDrawH = Math.min(srcH, (topH / pw) * srcW);
        context.save();
        context.translate(x, gapTop);
        context.scale(1, -1);
        context.drawImage(sprites.pipe, 0, 0, srcW, srcDrawH, 0, 0, pw, topH);
        context.restore();
    }
}

function drawBird(context: CanvasRenderingContext2D, x: number, y: number, scale: number, birdVy: number, step: number): void {
    const tilt = Math.max(-0.55, Math.min(0.9, birdVy * 0.09));
    const dw = BIRD_DRAW_W * scale;
    const dh = BIRD_DRAW_H * scale;
    // 拍翼循環：每 4 幀換一張
    const flapIndex = Math.floor(step / 4) % sprites.birds.length;
    const bird = sprites.birds[flapIndex] ?? sprites.birds[1];

    context.save();
    context.translate(x, y);
    context.rotate(tilt);
    context.imageSmoothingEnabled = false;

    if (bird && isReady(bird)) {
        context.drawImage(bird, -dw / 2, -dh / 2, dw, dh);
    } else {
        context.fillStyle = "#F7D031";
        context.beginPath();
        context.ellipse(0, 0, dw * 0.42, dh * 0.38, 0, 0, Math.PI * 2);
        context.fill();
    }

    context.restore();
}

function drawScore(context: CanvasRenderingContext2D, width: number, score: number, sy: number): void {
    const digits = String(Math.max(0, Math.floor(score)))
        .split("")
        .map(ch => Number(ch));
    const digitH = DIGIT_H * sy;
    const gaps = 2 * sy;
    let totalW = 0;
    const widths: number[] = [];

    for (const d of digits) {
        const img = sprites.digits[d];
        if (img && isReady(img)) {
            const w = (img.naturalWidth / img.naturalHeight) * digitH;
            widths.push(w);
            totalW += w;
        } else {
            widths.push(digitH * 0.66);
            totalW += digitH * 0.66;
        }
    }
    totalW += gaps * Math.max(0, digits.length - 1);

    let x = (width - totalW) / 2;
    const y = 40 * sy;
    for (let i = 0; i < digits.length; i += 1) {
        const img = sprites.digits[digits[i] ?? 0];
        const w = widths[i] ?? digitH * 0.66;
        if (img && isReady(img)) {
            context.drawImage(img, x, y, w, digitH);
        } else {
            context.fillStyle = "#FFFFFF";
            context.font = `800 ${Math.round(digitH)}px Inter, sans-serif`;
            context.textAlign = "left";
            context.fillText(String(digits[i]), x, y + digitH * 0.85);
        }
        x += w + gaps;
    }
}

function drawGameOver(context: CanvasRenderingContext2D, width: number, height: number, sx: number): void {
    context.fillStyle = "rgba(0, 0, 0, 0.28)";
    context.fillRect(0, 0, width, height);

    if (isReady(sprites.gameover)) {
        const srcW = sprites.gameover.naturalWidth;
        const srcH = sprites.gameover.naturalHeight;
        const drawW = Math.min(width * 0.72, srcW * sx * 1.15);
        const drawH = (srcH / srcW) * drawW;
        context.drawImage(sprites.gameover, (width - drawW) / 2, height * 0.38 - drawH / 2, drawW, drawH);
        return;
    }

    context.fillStyle = "#FFFFFF";
    context.font = "700 24px Inter, sans-serif";
    context.textAlign = "center";
    context.fillText("GAME OVER", width / 2, height / 2);
}

function drawIdle(context: CanvasRenderingContext2D, width: number, height: number, sy: number): void {
    if (isReady(sprites.message)) {
        const srcW = sprites.message.naturalWidth;
        const srcH = sprites.message.naturalHeight;
        const drawH = Math.min(height * 0.48, srcH * sy * 1.05);
        const drawW = (srcW / srcH) * drawH;
        context.drawImage(sprites.message, (width - drawW) / 2, (height - drawH) / 2 - 12 * sy, drawW, drawH);
        return;
    }

    context.fillStyle = "rgba(20, 40, 48, 0.45)";
    context.fillRect(0, height / 2 - 36, width, 72);
    context.fillStyle = "#FFFFFF";
    context.font = "600 18px Inter, sans-serif";
    context.textAlign = "center";
    context.fillText("等待第一個冠軍", width / 2, height / 2 + 6);
}
