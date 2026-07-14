import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import {createRandom} from "../../lib/random";
import type {Genome, Point, SnakeFrame, SnakeReplay} from "../../lib/types";

export const SNAKE_TOPOLOGY = {
    inputSize: 10,
    hiddenLayers: [12],
    outputSize: 3,
};

const GRID_SIZE = 20;
/** Hard cap so long-lived champions cannot run the worker / postMessage out of memory. */
const MAX_STEPS = 100_000;
/** Cap recorded frames (UI only needs a short playback, not every training step). */
const MAX_REPLAY_FRAMES = MAX_STEPS * 2;

const DIRECTIONS: Point[] = [
    {x: 0, y: -1},
    {x: 1, y: 0},
    {x: 0, y: 1},
    {x: -1, y: 0},
];

/** Shared adapter so evaluate/replay do not allocate a new network graph per genome. */
const networkAdapter = new NeuralNetworkAdapter(SNAKE_TOPOLOGY);

interface SnakeResult {
    fitness: number;
    replay: SnakeReplay;
}

export function evaluateSnakeGenome(genome: Genome): number {
    return [137, 911].reduce((sum, seed) => sum + simulateSnake(genome, seed, false).fitness, 0) / 2;
}

export function createSnakeReplay(genome: Genome): SnakeReplay {
    return simulateSnake(genome, 137, true).replay;
}

export const SNAKE_INPUT_LABELS = ["前危險", "左危險", "右危險", "食物 Δx", "食物 Δy", "向上", "向右", "向下", "向左", "身長"] as const;

export const SNAKE_OUTPUT_LABELS = ["左轉", "直行", "右轉"] as const;

/**
 * Rebuild the network input vector from a recorded frame so the UI can show
 * live activations without re-simulating the whole game.
 */
export function buildSnakeInputFromFrame(frame: SnakeFrame): number[] {
    const snake = frame.snake;
    const head = snake[0];
    const directionIndex = directionIndexFromSnake(snake);
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

function simulateSnake(genome: Genome, seed: number, record: boolean): SnakeResult {
    const random = createRandom(seed);
    const runNetwork = networkAdapter.createRunner(genome);
    const snake: Point[] = [
        {x: 10, y: 10},
        {x: 9, y: 10},
        {x: 8, y: 10},
    ];
    let directionIndex = 1;
    let food = createFood(snake, random.integer.bind(random));
    let score = 0;
    let stepsSinceFood = 0;
    let shapingReward = 0;
    let executedSteps = 0;
    let terminal: NonNullable<SnakeFrame["terminal"]> = "timeout";
    const frames: SnakeFrame[] = [];
    const frameStride = record ? Math.max(1, Math.ceil(MAX_STEPS / MAX_REPLAY_FRAMES)) : 1;

    for (let step = 0; step < MAX_STEPS; step += 1) {
        executedSteps = step + 1;
        if (record && step % frameStride === 0 && frames.length < MAX_REPLAY_FRAMES) {
            frames.push({snake: snake.map(part => ({...part})), food: {...food}, score, step});
        }

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
        const action = argMax(runNetwork(input));
        directionIndex = action === 0 ? leftDirection : action === 2 ? rightDirection : directionIndex;
        const movement = DIRECTIONS[directionIndex];
        const nextHead = {x: head.x + movement.x, y: head.y + movement.y};
        if (isCollision(nextHead, snake)) {
            terminal = "collision";
            break;
        }

        const previousDistance = manhattanDistance(head, food);
        const nextDistance = manhattanDistance(nextHead, food);
        shapingReward += previousDistance > nextDistance ? 0.16 : -0.2;
        // Mutate in place — avoid O(length) spread allocation every step.
        snake.unshift(nextHead);
        stepsSinceFood += 1;

        if (nextHead.x === food.x && nextHead.y === food.y) {
            score += 1;
            stepsSinceFood = 0;
            food = createFood(snake, random.integer.bind(random));
        } else {
            snake.pop();
        }

        if (stepsSinceFood > 110 + score * 16) {
            terminal = "starved";
            break;
        }
    }

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
    return {
        fitness: score * score * 180 + score * 90 + steps * 0.28 + shapingReward,
        replay: {frames, score, steps},
    };
}

function createFood(snake: Point[], integer: (min: number, max: number) => number): Point {
    for (let attempt = 0; attempt < GRID_SIZE * GRID_SIZE; attempt += 1) {
        const food = {x: integer(0, GRID_SIZE - 1), y: integer(0, GRID_SIZE - 1)};
        if (!snake.some(part => part.x === food.x && part.y === food.y)) {
            return food;
        }
    }
    return {x: 0, y: 0};
}

function isDanger(head: Point, direction: Point, snake: Point[]): boolean {
    return isCollision({x: head.x + direction.x, y: head.y + direction.y}, snake);
}

function isCollision(point: Point, snake: Point[]): boolean {
    return point.x < 0 || point.x >= GRID_SIZE || point.y < 0 || point.y >= GRID_SIZE || snake.some(part => part.x === point.x && part.y === point.y);
}

function manhattanDistance(a: Point, b: Point): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}
