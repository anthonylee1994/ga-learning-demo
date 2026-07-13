import Matter from "matter-js";
import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import type {BreakerBrick, BreakerFrame, BreakerReplay, Genome} from "../../lib/types";

export const BREAKER_TOPOLOGY = {
    inputSize: 8,
    hiddenLayers: [12],
    outputSize: 3,
};

const WIDTH = 560;
const HEIGHT = 420;
const MAX_STEPS = 9999;
const PADDLE_WIDTH = 92;
/** Canonical launch used for champion replay AND as one of the eval seeds. */
const REPLAY_LAUNCH = 0.86;
/**
 * Multiple fixed launches so fitness is not a lucky single angle.
 * Must include REPLAY_LAUNCH so the on-screen game matches what was trained.
 */
const EVAL_LAUNCHES = [0.72, 0.86, 1.0] as const;

/** Shared adapter so evaluate/replay do not allocate a new network graph per genome. */
const networkAdapter = new NeuralNetworkAdapter(BREAKER_TOPOLOGY);

interface BreakerResult {
    fitness: number;
    replay: BreakerReplay;
}

export function evaluateBreakerGenome(genome: Genome): number {
    return EVAL_LAUNCHES.reduce((sum, launch) => sum + simulateBreaker(genome, launch, false).fitness, 0) / EVAL_LAUNCHES.length;
}

export function createBreakerReplay(genome: Genome): BreakerReplay {
    return simulateBreaker(genome, REPLAY_LAUNCH, true).replay;
}

function simulateBreaker(genome: Genome, xVelocityFactor: number, record: boolean): BreakerResult {
    const engine = Matter.Engine.create({gravity: {x: 0, y: 0}});
    const runNetwork = networkAdapter.createRunner(genome);
    const paddle = Matter.Bodies.rectangle(WIDTH / 2, HEIGHT - 28, PADDLE_WIDTH, 12, {
        isStatic: true,
        label: "paddle",
        restitution: 1,
    });
    const ball = Matter.Bodies.circle(WIDTH / 2, HEIGHT - 58, 7, {
        label: "ball",
        restitution: 1,
        friction: 0,
        frictionAir: 0,
        inertia: Infinity,
    });
    const walls = [
        Matter.Bodies.rectangle(-6, HEIGHT / 2, 12, HEIGHT, {isStatic: true, label: "wall"}),
        Matter.Bodies.rectangle(WIDTH + 6, HEIGHT / 2, 12, HEIGHT, {isStatic: true, label: "wall"}),
        Matter.Bodies.rectangle(WIDTH / 2, -6, WIDTH, 12, {isStatic: true, label: "wall"}),
    ];
    const brickBodies: Matter.Body[] = [];
    const bricks = new Map<number, BreakerBrick>();
    let brickId = 0;
    for (let row = 0; row < 5; row += 1) {
        for (let column = 0; column < 9; column += 1) {
            const x = 42 + column * 59.5;
            const y = 50 + row * 25;
            const brick = Matter.Bodies.rectangle(x, y, 52, 16, {
                isStatic: true,
                label: `brick:${brickId}`,
                restitution: 1,
            });
            brickBodies.push(brick);
            bricks.set(brickId, {id: brickId, x, y, active: true});
            brickId += 1;
        }
    }

    Matter.Composite.add(engine.world, [paddle, ball, ...walls, ...brickBodies]);
    Matter.Body.setVelocity(ball, {x: 4.4 * xVelocityFactor, y: -4.4});
    let hits = 0;
    let bricksCleared = 0;
    let executedSteps = 0;
    let trackingReward = 0;
    let terminal: NonNullable<BreakerFrame["terminal"]> = "timeout";
    const frames: BreakerFrame[] = [];

    const onCollisionStart = (event: Matter.IEventCollision<Matter.Engine>) => {
        event.pairs.forEach(pair => {
            const labels = [pair.bodyA.label, pair.bodyB.label];
            if (labels.includes("paddle") && labels.includes("ball")) {
                hits += 1;
                const offset = (ball.position.x - paddle.position.x) / (PADDLE_WIDTH / 2);
                Matter.Body.setVelocity(ball, {x: offset * 5.2, y: -Math.abs(ball.velocity.y || 4.4)});
            }
            const brickBody = [pair.bodyA, pair.bodyB].find(body => body.label.startsWith("brick:"));
            if (brickBody) {
                const id = Number(brickBody.label.split(":")[1]);
                const brick = bricks.get(id);
                if (brick?.active) {
                    brick.active = false;
                    bricksCleared += 1;
                    Matter.Composite.remove(engine.world, brickBody);
                }
            }
        });
    };

    Matter.Events.on(engine, "collisionStart", onCollisionStart);

    try {
        for (let step = 0; step < MAX_STEPS; step += 1) {
            executedSteps = step + 1;
            const activeBricks = Array.from(bricks.values()).filter(brick => brick.active);
            const nearestBrick = activeBricks.reduce(
                (nearest, brick) => (Math.abs(brick.x - ball.position.x) < Math.abs(nearest.x - ball.position.x) ? brick : nearest),
                activeBricks[0] ?? {id: -1, x: ball.position.x, y: 0, active: false}
            );
            const input = [
                (paddle.position.x / WIDTH) * 2 - 1,
                (ball.position.x / WIDTH) * 2 - 1,
                (ball.position.y / HEIGHT) * 2 - 1,
                clamp(ball.velocity.x / 7, -1, 1),
                clamp(ball.velocity.y / 7, -1, 1),
                clamp((nearestBrick.x - ball.position.x) / WIDTH, -1, 1),
                clamp((nearestBrick.y - ball.position.y) / HEIGHT, -1, 1),
                activeBricks.length / bricks.size,
            ];
            const action = argMax(runNetwork(input));
            const movement = action === 0 ? -7 : action === 2 ? 7 : 0;
            Matter.Body.setPosition(paddle, {
                x: clamp(paddle.position.x + movement, PADDLE_WIDTH / 2, WIDTH - PADDLE_WIDTH / 2),
                y: paddle.position.y,
            });

            // Dense shaping: stay under the ball — makes early generations learn to track.
            if (ball.position.y > HEIGHT * 0.45) {
                const align = 1 - Math.min(1, Math.abs(paddle.position.x - ball.position.x) / (PADDLE_WIDTH * 1.5));
                trackingReward += align * 0.08;
            }

            Matter.Engine.update(engine, 1000 / 60);

            const speed = Math.hypot(ball.velocity.x, ball.velocity.y);
            if (speed > 0 && (speed < 4 || speed > 7)) {
                const targetSpeed = clamp(speed, 4, 7);
                Matter.Body.setVelocity(ball, {
                    x: (ball.velocity.x / speed) * targetSpeed,
                    y: (ball.velocity.y / speed) * targetSpeed,
                });
            }

            if (record && step % 4 === 0) {
                frames.push({
                    paddleX: paddle.position.x,
                    ball: {x: ball.position.x, y: ball.position.y},
                    bricks: Array.from(bricks.values()).map(brick => ({...brick})),
                    hits,
                    step,
                });
            }

            if (ball.position.y > HEIGHT + 20) {
                terminal = "lost";
                break;
            }
            if (bricksCleared === bricks.size) {
                terminal = "cleared";
                break;
            }
        }

        if (record) {
            frames.push({
                paddleX: paddle.position.x,
                ball: {x: ball.position.x, y: ball.position.y},
                bricks: Array.from(bricks.values()).map(brick => ({...brick})),
                hits,
                step: executedSteps,
                terminal,
            });
        }
    } finally {
        // Matter 0.20 Engine.clear only clears pairs/detector — world bodies & events must be released manually.
        Matter.Events.off(engine, "collisionStart", onCollisionStart);
        Matter.Composite.clear(engine.world, false);
        Matter.Engine.clear(engine);
    }

    const clearBonus = bricksCleared === bricks.size ? 2_000 : 0;
    return {
        // Bricks dominate; steps are a tiny tie-breaker so micro-improvements don't thrash the UI.
        fitness: bricksCleared * 110 + hits * 9 + trackingReward + executedSteps * 0.01 + clearBonus,
        replay: {frames, bricksCleared, hits, steps: executedSteps},
    };
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}
