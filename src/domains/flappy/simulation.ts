/**
 * Flappy Bird 模擬：固定步長物理，神經網絡決定「拍翼 / 唔拍」。
 *
 * 流程：
 * 1. evaluateFlappyGenome — 多個固定 seed 各飛一場，fitness 取平均
 * 2. createFlappyReplay — UI 重播用固定 seed
 * 3. simulateFlappy — 重力 + 水管滾動 + 碰撞
 */
import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import {createRandom} from "../../lib/random";
import type {FlappyFrame, FlappyPipe, FlappyReplay, Genome, NetworkTopology} from "../../lib/types";

/** 畫面尺寸（邏輯 px；canvas 會 scale） */
export const FLAPPY_WIDTH = 360;
export const FLAPPY_HEIGHT = 520;

/** NN：6 觀測 → 10 hidden → 2 動作（拍翼 / 滑翔） */
export const FLAPPY_TOPOLOGY: NetworkTopology = {
    inputSize: 6,
    hiddenLayers: [10],
    outputSize: 2,
};

export const FLAPPY_INPUT_LABELS = ["鳥 y", "鳥 vy", "下管距離", "縫隙上沿", "縫隙下沿", "相對縫心"] as const;
export const FLAPPY_OUTPUT_LABELS = ["拍翼", "滑翔"] as const;

const BIRD_X = 78;
const BIRD_RADIUS = 12;
const PIPE_WIDTH = 48;
const PIPE_GAP = 128;
const PIPE_SPEED = 2.6;
const GRAVITY = 0.38;
const FLAP_VELOCITY = -6.4;
const MAX_FALL = 9.5;
const PIPE_SPAWN_GAP = 168;
/** 一場最長步數，防止無限飛 */
const MAX_STEPS = 8_000;
const MAX_REPLAY_FRAMES = 2_400;

const networkAdapter = new NeuralNetworkAdapter(FLAPPY_TOPOLOGY);

interface FlappyResult {
    fitness: number;
    replay: FlappyReplay;
}

/**
 * GA 評分：三個固定 seed 平均，減少某一組水管幸運。
 */
export function evaluateFlappyGenome(genome: Genome): number {
    return [41, 277, 809].reduce((sum, seed) => sum + simulateFlappy(genome, seed, false).fitness, 0) / 3;
}

export function createFlappyReplay(genome: Genome): FlappyReplay {
    return simulateFlappy(genome, 41, true).replay;
}

/**
 * 由 frame 重建 NN 輸入（同 simulate 一致嘅正規化）。
 */
export function buildFlappyInputFromFrame(frame: FlappyFrame): number[] {
    return buildInput(frame.birdY, frame.birdVy, frame.pipes);
}

function simulateFlappy(genome: Genome, seed: number, record: boolean): FlappyResult {
    const random = createRandom(seed);
    const runNetwork = networkAdapter.createRunner(genome);

    let birdY = FLAPPY_HEIGHT * 0.45;
    let birdVy = 0;
    let score = 0;
    let shaping = 0;
    let executedSteps = 0;
    let terminal: NonNullable<FlappyFrame["terminal"]> = "timeout";
    const pipes: FlappyPipe[] = [spawnPipe(FLAPPY_WIDTH + 40, () => random.next())];
    const frames: FlappyFrame[] = [];
    const frameStride = record ? Math.max(1, Math.ceil(MAX_STEPS / MAX_REPLAY_FRAMES)) : 1;

    for (let step = 0; step < MAX_STEPS; step += 1) {
        executedSteps = step + 1;

        if (record && step % frameStride === 0 && frames.length < MAX_REPLAY_FRAMES) {
            frames.push(snapshotFrame(birdY, birdVy, pipes, score, step));
        }

        const input = buildInput(birdY, birdVy, pipes);
        const action = argMax(runNetwork(input));
        if (action === 0) {
            birdVy = FLAP_VELOCITY;
        }

        birdVy = Math.min(MAX_FALL, birdVy + GRAVITY);
        birdY += birdVy;

        for (const pipe of pipes) {
            pipe.x -= PIPE_SPEED;
        }

        // 計分：鳥通過管右沿
        for (const pipe of pipes) {
            if (!pipe.passed && pipe.x + PIPE_WIDTH < BIRD_X - BIRD_RADIUS) {
                pipe.passed = true;
                score += 1;
            }
        }

        // 清走離場水管；必要時加新管
        while (pipes.length > 0 && pipes[0].x + PIPE_WIDTH < -8) {
            pipes.shift();
        }
        const rightmost = pipes.reduce((max, pipe) => Math.max(max, pipe.x), -Infinity);
        if (rightmost < FLAPPY_WIDTH - PIPE_SPAWN_GAP) {
            pipes.push(spawnPipe(FLAPPY_WIDTH + 20, () => random.next()));
        }

        // 靠近縫心 shaping（早期梯度）
        const next = nextPipe(pipes);
        if (next) {
            const gapCenter = next.gapY;
            const dist = Math.abs(birdY - gapCenter) / FLAPPY_HEIGHT;
            shaping += (0.5 - dist) * 0.12;
        }

        if (isCrashed(birdY, pipes)) {
            terminal = "crash";
            break;
        }
    }

    if (record) {
        frames.push(snapshotFrame(birdY, birdVy, pipes, score, executedSteps, terminal));
    }

    // score² 主導；存活有少少分；shaping 輔導飛近縫
    const fitness = score * score * 220 + score * 80 + executedSteps * 0.35 + shaping;
    return {
        fitness,
        replay: {frames, score, steps: executedSteps},
    };
}

function buildInput(birdY: number, birdVy: number, pipes: FlappyPipe[]): number[] {
    const pipe = nextPipe(pipes);
    const gapHalf = (pipe?.gapHeight ?? PIPE_GAP) / 2;
    const gapTop = pipe ? pipe.gapY - gapHalf : FLAPPY_HEIGHT * 0.35;
    const gapBottom = pipe ? pipe.gapY + gapHalf : FLAPPY_HEIGHT * 0.65;
    const pipeDistance = pipe ? (pipe.x - BIRD_X) / FLAPPY_WIDTH : 1;
    const gapCenter = pipe ? pipe.gapY : FLAPPY_HEIGHT * 0.5;

    return [
        (birdY / FLAPPY_HEIGHT) * 2 - 1,
        Math.max(-1, Math.min(1, birdVy / MAX_FALL)),
        Math.max(-1, Math.min(1, pipeDistance * 2 - 1)),
        (gapTop / FLAPPY_HEIGHT) * 2 - 1,
        (gapBottom / FLAPPY_HEIGHT) * 2 - 1,
        Math.max(-1, Math.min(1, (birdY - gapCenter) / (FLAPPY_HEIGHT * 0.5))),
    ];
}

function nextPipe(pipes: FlappyPipe[]): FlappyPipe | undefined {
    return pipes.find(pipe => pipe.x + PIPE_WIDTH >= BIRD_X - BIRD_RADIUS - 4) ?? pipes[0];
}

function spawnPipe(x: number, float: () => number): FlappyPipe {
    const margin = 70;
    const gapY = margin + PIPE_GAP / 2 + float() * (FLAPPY_HEIGHT - margin * 2 - PIPE_GAP);
    return {x, gapY, gapHeight: PIPE_GAP, passed: false};
}

function isCrashed(birdY: number, pipes: FlappyPipe[]): boolean {
    if (birdY - BIRD_RADIUS <= 0 || birdY + BIRD_RADIUS >= FLAPPY_HEIGHT) {
        return true;
    }

    const top = birdY - BIRD_RADIUS;
    const bottom = birdY + BIRD_RADIUS;
    const left = BIRD_X - BIRD_RADIUS;
    const right = BIRD_X + BIRD_RADIUS;

    for (const pipe of pipes) {
        const pipeLeft = pipe.x;
        const pipeRight = pipe.x + PIPE_WIDTH;
        if (right < pipeLeft || left > pipeRight) {
            continue;
        }
        const gapHalf = pipe.gapHeight / 2;
        const gapTop = pipe.gapY - gapHalf;
        const gapBottom = pipe.gapY + gapHalf;
        if (top < gapTop || bottom > gapBottom) {
            return true;
        }
    }
    return false;
}

function snapshotFrame(birdY: number, birdVy: number, pipes: FlappyPipe[], score: number, step: number, terminal?: FlappyFrame["terminal"]): FlappyFrame {
    return {
        birdY,
        birdVy,
        pipes: pipes.map(pipe => ({...pipe})),
        score,
        step,
        terminal,
    };
}
