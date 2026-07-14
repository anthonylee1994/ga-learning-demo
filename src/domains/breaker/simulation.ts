import Matter from "matter-js";
import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import type {RandomSource} from "../../lib/random";
import type {BreakerBrick, BreakerFrame, BreakerReplay, Genome} from "../../lib/types";

export const BREAKER_TOPOLOGY = {
    inputSize: 8,
    hiddenLayers: [12],
    outputSize: 3,
};

const WIDTH = 560;
const HEIGHT = 420;
/**
 * Cap physics steps so a stuck rally cannot freeze the worker.
 * ~60s at 60 Hz is already a very long breakout rally for this arena.
 */
const MAX_STEPS = 100_000;
/** UI playback only needs a short highlight reel (not one frame per physics step). */
const MAX_REPLAY_FRAMES = MAX_STEPS * 2;
const PADDLE_WIDTH = 92;
/** Keep a meaningful vertical component so the ball cannot infinite-loop left↔right off the walls. */
const MIN_BALL_VY = 2.4;
const MIN_BALL_SPEED = 4;
const MAX_BALL_SPEED = 7;

/**
 * Base launch angles for multi-match fitness. Each match draws fresh Math.random noise
 * so the agent must track the ball — not memorize a fixed serve path.
 */
const EVAL_LAUNCHES = [0.58, 0.72, 0.86, 1.0, 1.18] as const;

/** Shared adapter so evaluate/replay do not allocate a new network graph per genome. */
const networkAdapter = new NeuralNetworkAdapter(BREAKER_TOPOLOGY);

interface BreakerResult {
    fitness: number;
    replay: BreakerReplay;
}

/** Live (non-seeded) RNG — every match looks different to the user. */
function createLiveRandom(): RandomSource {
    let spare: number | null = null;
    return {
        next: () => Math.random(),
        integer(min: number, max: number) {
            return Math.floor(Math.random() * (max - min + 1)) + min;
        },
        gaussian() {
            if (spare !== null) {
                const value = spare;
                spare = null;
                return value;
            }
            const u = Math.max(Math.random(), Number.EPSILON);
            const v = Math.max(Math.random(), Number.EPSILON);
            const magnitude = Math.sqrt(-2 * Math.log(u));
            spare = magnitude * Math.sin(2 * Math.PI * v);
            return magnitude * Math.cos(2 * Math.PI * v);
        },
    };
}

export function evaluateBreakerGenome(genome: Genome): number {
    return EVAL_LAUNCHES.reduce((sum, launch) => sum + simulateBreaker(genome, launch, false, createLiveRandom()).fitness, 0) / EVAL_LAUNCHES.length;
}

export function createBreakerReplay(genome: Genome): BreakerReplay {
    // Fresh random each call so loop / re-roll shows a different version of the champion.
    const launch = EVAL_LAUNCHES[Math.floor(Math.random() * EVAL_LAUNCHES.length)];
    return simulateBreaker(genome, launch, true, createLiveRandom()).replay;
}

export const BREAKER_INPUT_LABELS = ["板 X", "球 X", "球 Y", "速 X", "速 Y", "磚 Δx", "磚 Δy", "剩餘磚"] as const;

export const BREAKER_OUTPUT_LABELS = ["向左", "停住", "向右"] as const;

/**
 * Rebuild the network input vector from a recorded frame so the UI can show
 * live activations without re-running Matter.js.
 */
export function buildBreakerInputFromFrame(frame: BreakerFrame): number[] {
    const activeBricks = frame.bricks.filter(brick => brick.active);
    const nearestBrick = activeBricks.reduce(
        (nearest, brick) => (Math.abs(brick.x - frame.ball.x) < Math.abs(nearest.x - frame.ball.x) ? brick : nearest),
        activeBricks[0] ?? {id: -1, x: frame.ball.x, y: 0, active: false}
    );
    const velocity = frame.ballVelocity ?? {x: 0, y: 0};
    const totalBricks = Math.max(1, frame.bricks.length);
    return [
        (frame.paddleX / WIDTH) * 2 - 1,
        (frame.ball.x / WIDTH) * 2 - 1,
        (frame.ball.y / HEIGHT) * 2 - 1,
        clamp(velocity.x / 7, -1, 1),
        clamp(velocity.y / 7, -1, 1),
        clamp((nearestBrick.x - frame.ball.x) / WIDTH, -1, 1),
        clamp((nearestBrick.y - frame.ball.y) / HEIGHT, -1, 1),
        activeBricks.length / totalBricks,
    ];
}

function simulateBreaker(genome: Genome, xVelocityFactor: number, record: boolean, random: RandomSource): BreakerResult {
    const engine = Matter.Engine.create({gravity: {x: 0, y: 0}});
    const runNetwork = networkAdapter.createRunner(genome);
    // Scenario-specific paddle start — cannot memorize one centre-only serve recovery.
    const paddleStartX = clamp(WIDTH / 2 + (random.next() - 0.5) * 56, PADDLE_WIDTH / 2, WIDTH - PADDLE_WIDTH / 2);
    const paddle = Matter.Bodies.rectangle(paddleStartX, HEIGHT - 28, PADDLE_WIDTH, 12, {
        isStatic: true,
        label: "paddle",
        restitution: 1,
    });
    // Ball spawn offset so the first bounce geometry differs per scenario.
    const ballStartX = clamp(WIDTH / 2 + (random.next() - 0.5) * 36, 24, WIDTH - 24);
    const ball = Matter.Bodies.circle(ballStartX, HEIGHT - 58 - random.next() * 6, 7, {
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
    // Slight whole-grid shift per scenario (still fully on-screen) so brick hit order changes.
    const gridShiftX = (random.next() - 0.5) * 10;
    const gridShiftY = (random.next() - 0.5) * 6;
    for (let row = 0; row < 5; row += 1) {
        for (let column = 0; column < 9; column += 1) {
            const x = 42 + column * 59.5 + gridShiftX;
            const y = 50 + row * 25 + gridShiftY;
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
    // Base launch + noticeable jitter (still deterministic via RandomSource seed).
    const launchVx = 4.4 * xVelocityFactor + (random.next() - 0.5) * 1.05;
    const launchVy = -4.4 + (random.next() - 0.5) * 0.55;
    Matter.Body.setVelocity(ball, {x: launchVx, y: launchVy});
    normalizeBallVelocity(ball);
    let hits = 0;
    let bricksCleared = 0;
    let executedSteps = 0;
    let trackingReward = 0;
    let terminal: NonNullable<BreakerFrame["terminal"]> = "timeout";
    const frames: BreakerFrame[] = [];
    const frameStride = record ? Math.max(4, Math.ceil(MAX_STEPS / MAX_REPLAY_FRAMES)) : 4;

    const onCollisionStart = (event: Matter.IEventCollision<Matter.Engine>) => {
        event.pairs.forEach(pair => {
            const labels = [pair.bodyA.label, pair.bodyB.label];
            if (labels.includes("paddle") && labels.includes("ball")) {
                hits += 1;
                const offset = (ball.position.x - paddle.position.x) / (PADDLE_WIDTH / 2);
                // Contact angle + real spin noise — same offset does not always yield same rebound.
                const spin = (random.next() - 0.5) * 1.15;
                const bounceX = offset * 5.2 + spin;
                const bounceY = -Math.max(MIN_BALL_VY, Math.abs(ball.velocity.y) || 4.4) - random.next() * 0.55;
                Matter.Body.setVelocity(ball, {x: bounceX, y: bounceY});
                normalizeBallVelocity(ball);
            }
            const brickBody = [pair.bodyA, pair.bodyB].find(body => body.label.startsWith("brick:"));
            if (brickBody) {
                const id = Number(brickBody.label.split(":")[1]);
                const brick = bricks.get(id);
                if (brick?.active) {
                    brick.active = false;
                    bricksCleared += 1;
                    Matter.Composite.remove(engine.world, brickBody);
                    // Stronger post-brick scatter so the next approach is not a fixed script.
                    nudgeBallVelocity(ball, random, 0.55);
                }
            }
            // Side / ceiling wall: tangent jitter (keeps |vy| floor via normalize).
            if (labels.includes("wall") && labels.includes("ball")) {
                nudgeBallVelocity(ball, random, 0.32);
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

            // Wall/brick bounces can collapse |vy| → endless side-to-side loop until MAX_STEPS.
            normalizeBallVelocity(ball);

            if (record && step % frameStride === 0 && frames.length < MAX_REPLAY_FRAMES) {
                frames.push({
                    paddleX: paddle.position.x,
                    ball: {x: ball.position.x, y: ball.position.y},
                    ballVelocity: {x: ball.velocity.x, y: ball.velocity.y},
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
                ballVelocity: {x: ball.velocity.x, y: ball.velocity.y},
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

    const totalBricks = bricks.size;
    const clearedAll = bricksCleared === totalBricks;
    const clearBonus = clearedAll ? 2_000 : 0;
    /**
     * Brick-first + speed. Faster full clears score much higher; raw paddle hits are not paid
     * (and wasteful juggling is penalized) so the agent cannot farm rallies instead of clearing.
     */
    // Allow ~2 defensive hits per brick cleared (plus a small free buffer); extras cost score.
    const allowedHits = bricksCleared * 2 + 4;
    const wasteHits = Math.max(0, hits - allowedHits);
    const wastePenalty = wasteHits * 6;
    // Pace: bricks cleared per step — rewards quick progress even before a full clear.
    const paceBonus = executedSteps > 0 ? (bricksCleared / executedSteps) * 1_400 : 0;
    // Full clear: remaining time is a big deal (quadratic so "much faster" >> "slightly faster").
    const finishRatio = clearedAll ? Math.max(0, 1 - executedSteps / MAX_STEPS) : 0;
    const clearSpeedBonus = finishRatio * finishRatio * 2_200 + finishRatio * 500;
    // Soft shape paddle under the ball while learning — cap so it never rivals a single brick.
    const trackingCap = bricksCleared * 8 + 12;
    const trackingScore = Math.min(trackingReward, trackingCap);

    return {
        fitness: bricksCleared * 140 + clearBonus + clearSpeedBonus + paceBonus + trackingScore - wastePenalty,
        replay: {frames, bricksCleared, hits, steps: executedSteps},
    };
}

/** Apply a small random delta to velocity, then re-normalize speed / |vy|. */
function nudgeBallVelocity(ball: Matter.Body, random: RandomSource, scale: number): void {
    Matter.Body.setVelocity(ball, {
        x: ball.velocity.x + (random.next() - 0.5) * scale * 2,
        y: ball.velocity.y + (random.next() - 0.5) * scale,
    });
    normalizeBallVelocity(ball);
}

/**
 * Clamp ball speed into [MIN, MAX] and guarantee a minimum |vertical| component.
 * Without this, Matter.js wall/brick reflections can leave vy≈0 and the ball
 * orbits left↔right forever (UI looks stuck in an infinite loop).
 */
function normalizeBallVelocity(ball: Matter.Body): void {
    let vx = ball.velocity.x;
    let vy = ball.velocity.y;

    if (!Number.isFinite(vx) || !Number.isFinite(vy)) {
        Matter.Body.setVelocity(ball, {x: 3.2, y: -4.4});
        return;
    }

    if (Math.abs(vy) < MIN_BALL_VY) {
        // Keep prior sign when possible; if flat, push away from the nearer vertical edge of play.
        const sign = Math.abs(vy) > 1e-6 ? Math.sign(vy) : ball.position.y < HEIGHT * 0.45 ? 1 : -1;
        vy = sign * MIN_BALL_VY;
    }

    let speed = Math.hypot(vx, vy);
    if (speed < 1e-6) {
        Matter.Body.setVelocity(ball, {x: 3.2, y: -4.4});
        return;
    }

    const target = clamp(speed, MIN_BALL_SPEED, MAX_BALL_SPEED);
    vx = (vx / speed) * target;
    vy = (vy / speed) * target;

    // Renormalizing can shrink |vy| again when |vx| dominates — re-assert floor once.
    if (Math.abs(vy) < MIN_BALL_VY) {
        const sign = Math.sign(vy) || (ball.position.y < HEIGHT * 0.45 ? 1 : -1);
        vy = sign * MIN_BALL_VY;
        speed = Math.hypot(vx, vy);
        const retarget = clamp(speed, MIN_BALL_SPEED, MAX_BALL_SPEED);
        vx = (vx / speed) * retarget;
        vy = (vy / speed) * retarget;
    }

    Matter.Body.setVelocity(ball, {x: vx, y: vy});
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}
