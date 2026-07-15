/**
 * Snake 模擬：20×20 格、神經網絡控制「相對轉向」。
 *
 * 流程概覽：
 * 1. evaluateSnakeGenome — GA 評分：兩個固定 seed 各打一場，取平均 fitness
 * 2. createSnakeReplay — UI 重播：固定 seed 137，錄 frames 畀畫面畫
 * 3. simulateSnake — 真正跑一場：感知 → 左/直/右 → 撞牆/咬自己/餓死就停
 *
 * 設計重點：
 * - 動作係相對方向（左轉／直行／右轉），唔係絕對上下左右，NN 細啲易學
 * - shaping reward：靠近食物 +0.16、離遠 −0.2，early generation 先有梯度
 * - 太耐唔食會死（starved），防止傻傻轉圈刷 steps 分
 */
import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import {createRandom} from "../../lib/random";
import type {Genome, Point, SnakeFrame, SnakeReplay} from "../../lib/types";

/** NN 結構：10 個觀測 → 12 hidden → 3 個動作（左轉 / 直行 / 右轉） */
export const SNAKE_TOPOLOGY = {
    inputSize: 10,
    hiddenLayers: [12],
    outputSize: 3,
};

/** 地圖邊長（格） */
const GRID_SIZE = 20;
/** 一場最多步數；長命 champion 唔可以令 worker / postMessage OOM */
const MAX_STEPS = 100_000;
/** Replay 最多幀數（UI 唔需要每個 training step） */
const MAX_REPLAY_FRAMES = MAX_STEPS * 2;

/**
 * 四個絕對方向，index 對應 one-hot 輸入：
 * 0 上、1 右、2 下、3 左（順時針）
 */
const DIRECTIONS: Point[] = [
    {x: 0, y: -1},
    {x: 1, y: 0},
    {x: 0, y: 1},
    {x: -1, y: 0},
];

/** 共用 NN adapter：evaluate / replay 都用同一個 graph，唔使每個 genome new 一次 */
const networkAdapter = new NeuralNetworkAdapter(SNAKE_TOPOLOGY);

interface SnakeResult {
    fitness: number;
    replay: SnakeReplay;
}

/**
 * GA 用：同一個 genome 用兩個固定 seed 各打一場，fitness 取平均。
 * 多 seed 減少「某一局食物位置幸運」嘅 noise。
 */
export function evaluateSnakeGenome(genome: Genome): number {
    return [137, 911].reduce((sum, seed) => sum + simulateSnake(genome, seed, false).fitness, 0) / 2;
}

/**
 * UI 用：固定 seed 137 錄 replay（結果可重現，方便 debug）。
 */
export function createSnakeReplay(genome: Genome): SnakeReplay {
    return simulateSnake(genome, 137, true).replay;
}

/** 對應 10 維輸入（畀 UI 顯示 activation 標籤） */
export const SNAKE_INPUT_LABELS = ["前危險", "左危險", "右危險", "食物 Δx", "食物 Δy", "向上", "向右", "向下", "向左", "身長"] as const;

/** 對應 3 維輸出：相對轉向 */
export const SNAKE_OUTPUT_LABELS = ["左轉", "直行", "右轉"] as const;

/**
 * 由已錄 frame 重建 NN 輸入。
 * UI 顯示 live activation 時用：唔使重跑成場，只讀 snapshot。
 * 正規化方式要同 simulateSnake 入面嘅 input 完全一致。
 *
 * 輸入（危險用 ±1，方向 one-hot ±1）：
 *   [0–2]  前／左／右一格會唔會撞（危險 = 1，安全 = -1）
 *   [3–4]  食物相對頭嘅 Δx、Δy（÷ GRID_SIZE）
 *   [5–8]  目前朝向 one-hot
 *   [9]    身長 / 全圖格數
 */
export function buildSnakeInputFromFrame(frame: SnakeFrame): number[] {
    const snake = frame.snake;
    const head = snake[0];
    const directionIndex = directionIndexFromSnake(snake);
    // 相對左／右：index 順時針 +1 右、−1（+3）左
    const leftDirection = (directionIndex + 3) % 4;
    const rightDirection = (directionIndex + 1) % 4;
    return [
        isDanger(head, DIRECTIONS[directionIndex], snake) ? 1 : -1,
        isDanger(head, DIRECTIONS[leftDirection], snake) ? 1 : -1,
        isDanger(head, DIRECTIONS[rightDirection], snake) ? 1 : -1,
        (frame.food.x - head.x) / GRID_SIZE,
        (frame.food.y - head.y) / GRID_SIZE,
        directionIndex === 0 ? 1 : -1,
        directionIndex === 1 ? 1 : -1,
        directionIndex === 2 ? 1 : -1,
        directionIndex === 3 ? 1 : -1,
        snake.length / (GRID_SIZE * GRID_SIZE),
    ];
}

/**
 * 由蛇身還原目前朝向：頭 − 第二節 = 移動向量。
 * 長度 < 2 或對唔上 → 預設向右 (1)。
 */
function directionIndexFromSnake(snake: Point[]): number {
    if (snake.length < 2) {
        return 1;
    }
    const dx = snake[0].x - snake[1].x;
    const dy = snake[0].y - snake[1].y;
    for (let index = 0; index < DIRECTIONS.length; index += 1) {
        if (DIRECTIONS[index].x === dx && DIRECTIONS[index].y === dy) {
            return index;
        }
    }
    return 1;
}

/**
 * 跑完整一場 Snake。
 *
 * @param genome 神經網絡權重
 * @param seed   RNG seed（食物生成位置；evaluate 用多 seed 平均）
 * @param record true = 抽樣存 frames 做 replay；false = 淨計 fitness
 */
function simulateSnake(genome: Genome, seed: number, record: boolean): SnakeResult {
    const random = createRandom(seed);
    const runNetwork = networkAdapter.createRunner(genome);

    // 開局：頭喺 (10,10)，向右伸兩格身體
    const snake: Point[] = [
        {x: 10, y: 10},
        {x: 9, y: 10},
        {x: 8, y: 10},
    ];
    let directionIndex = 1; // 向右
    let food = createFood(snake, random.integer.bind(random));
    let score = 0; // 食到幾多粒
    let stepsSinceFood = 0; // 距離上次食嘢幾多步（餓死計時）
    let shapingReward = 0; // 靠近／遠離食物嘅 dense reward
    let executedSteps = 0;
    let terminal: NonNullable<SnakeFrame["terminal"]> = "timeout";
    const frames: SnakeFrame[] = [];
    const frameStride = record ? Math.max(1, Math.ceil(MAX_STEPS / MAX_REPLAY_FRAMES)) : 1;

    for (let step = 0; step < MAX_STEPS; step += 1) {
        executedSteps = step + 1;

        // 錄決策前狀態（UI 先見到「想點郁」嗰刻嘅場）
        if (record && step % frameStride === 0 && frames.length < MAX_REPLAY_FRAMES) {
            frames.push({snake: snake.map(part => ({...part})), food: {...food}, score, step});
        }

        // ── 感知 ────────────────────────────────────────────────────
        const head = snake[0];
        const leftDirection = (directionIndex + 3) % 4;
        const rightDirection = (directionIndex + 1) % 4;
        const input = [
            isDanger(head, DIRECTIONS[directionIndex], snake) ? 1 : -1,
            isDanger(head, DIRECTIONS[leftDirection], snake) ? 1 : -1,
            isDanger(head, DIRECTIONS[rightDirection], snake) ? 1 : -1,
            (food.x - head.x) / GRID_SIZE,
            (food.y - head.y) / GRID_SIZE,
            directionIndex === 0 ? 1 : -1,
            directionIndex === 1 ? 1 : -1,
            directionIndex === 2 ? 1 : -1,
            directionIndex === 3 ? 1 : -1,
            snake.length / (GRID_SIZE * GRID_SIZE),
        ];

        // ── 決策：0 左轉、1 直行、2 右轉 ────────────────────────────
        const action = argMax(runNetwork(input));
        directionIndex = action === 0 ? leftDirection : action === 2 ? rightDirection : directionIndex;
        const movement = DIRECTIONS[directionIndex];
        const nextHead = {x: head.x + movement.x, y: head.y + movement.y};

        // 撞牆或咬自己 → 即死
        if (isCollision(nextHead, snake)) {
            terminal = "collision";
            break;
        }

        // Dense shaping：靠近食物有獎、離遠有罰（比淨等 score 密好多訊號）
        const previousDistance = manhattanDistance(head, food);
        const nextDistance = manhattanDistance(nextHead, food);
        shapingReward += previousDistance > nextDistance ? 0.16 : -0.2;

        // 前進：頭 unshift；食到就唔 pop（變長），否則 pop 尾
        // in-place mutate，避免每步 O(length) spread 新 array
        snake.unshift(nextHead);
        stepsSinceFood += 1;

        if (nextHead.x === food.x && nextHead.y === food.y) {
            score += 1;
            stepsSinceFood = 0;
            food = createFood(snake, random.integer.bind(random));
        } else {
            snake.pop();
        }

        // 太耐唔食：base 110 步 + 每分 16 步 buffer（越長越准多啲時間找食）
        // 防止轉圈刷 steps 分、又唔真係玩
        if (stepsSinceFood > 110 + score * 16) {
            terminal = "starved";
            break;
        }
        // 跑到 MAX_STEPS → terminal 保持 "timeout"
    }

    // 最後一幀帶 terminal，方便 UI 顯示結局
    if (record) {
        frames.push({
            snake: snake.map(part => ({...part})),
            food: {...food},
            score,
            step: executedSteps,
            terminal,
        });
    }

    const steps = executedSteps;
    // fitness：食越多越好（二次項鼓勵高分）、行耐有少少分、shaping 累積
    // score² * 180 主導；steps * 0.28 鼓勵生存但遠唔夠靠耍步數贏
    return {
        fitness: score * score * 180 + score * 90 + steps * 0.28 + shapingReward,
        replay: {frames, score, steps},
    };
}

/** 隨機放食物，唔可以疊喺蛇身上；試盡全圖都唔得就 fallback (0,0) */
function createFood(snake: Point[], integer: (min: number, max: number) => number): Point {
    for (let attempt = 0; attempt < GRID_SIZE * GRID_SIZE; attempt += 1) {
        const food = {x: integer(0, GRID_SIZE - 1), y: integer(0, GRID_SIZE - 1)};
        if (!snake.some(part => part.x === food.x && part.y === food.y)) {
            return food;
        }
    }
    return {x: 0, y: 0};
}

/** 頭沿 direction 走一格會唔會撞 */
function isDanger(head: Point, direction: Point, snake: Point[]): boolean {
    return isCollision({x: head.x + direction.x, y: head.y + direction.y}, snake);
}

/** 出界或撞到自己身體（含頭） */
function isCollision(point: Point, snake: Point[]): boolean {
    return point.x < 0 || point.x >= GRID_SIZE || point.y < 0 || point.y >= GRID_SIZE || snake.some(part => part.x === point.x && part.y === point.y);
}

function manhattanDistance(a: Point, b: Point): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}
