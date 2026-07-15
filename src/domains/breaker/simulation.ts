/**
 * Breaker（打磚塊）模擬：用 Matter.js 做物理，神經網絡控制擋板。
 *
 * 流程概覽：
 * 1. evaluateBreakerGenome — GA 評分：同一個 genome 打多場（唔同發射角），取平均 fitness
 * 2. createBreakerReplay — UI 重播：錄低每幾步一幀，畀畫面畫返出嚟
 * 3. simulateBreaker — 真正跑一場：擺場 → 每步問 NN 移板 → 計分
 *
 * 設計重點：每場有隨機出生點／發射／反彈 jitter，逼 agent 學「跟波」而唔係背死一條路。
 */
import Matter from "matter-js";
import {argMax, NeuralNetworkAdapter} from "../../lib/neuralNetwork";
import type {RandomSource} from "../../lib/random";
import type {BreakerBrick, BreakerFrame, BreakerReplay, Genome} from "../../lib/types";

/** NN 結構：8 個觀測 → 12 hidden → 3 個動作（左 / 停 / 右） */
export const BREAKER_TOPOLOGY = {
    inputSize: 8,
    hiddenLayers: [12],
    outputSize: 3,
};

/** 遊戲場地闊度（像素） */
const WIDTH = 560;
/** 遊戲場地高度（像素） */
const HEIGHT = 420;
/**
 * 一場最多跑幾多個物理 step，防止卡死 rally 令 worker 凍死。
 * 60 Hz 計大概 ~28 分鐘先 timeout，對呢個 arena 已經長過龍。
 */
const MAX_STEPS = 100_000;
/**
 * Replay 最多存幾多幀。
 * 唔會每 step 都錄（太肥），用 frameStride 抽樣。
 */
const MAX_REPLAY_FRAMES = MAX_STEPS * 2;
const PADDLE_WIDTH = 92;
/** 球垂直速度下限：避免撞牆後 vy≈0，左右無限彈 */
const MIN_BALL_VY = 2.4;
/** 球速下限：太慢會「黏住」難打 */
const MIN_BALL_SPEED = 4;
/** 球速上限：太快 NN 跟唔切 */
const MAX_BALL_SPEED = 7;

/**
 * 評分用嘅一組基準發射水平係數（乘落 launchVx）。
 * 每場再加 random noise，所以 agent 要真正跟波，背固定 serve 冇用。
 */
const EVAL_LAUNCHES = [0.58, 0.72, 0.86, 1.0, 1.18] as const;

/** 共用 NN adapter：evaluate / replay 都用同一個 graph，唔使每個 genome new 一次 */
const networkAdapter = new NeuralNetworkAdapter(BREAKER_TOPOLOGY);

interface BreakerResult {
    fitness: number;
    replay: BreakerReplay;
}

/**
 * 即場 RNG（唔 seed）。
 * gaussian 用 Box–Muller，spare 存第二個樣本，慳一次 random。
 * （而家 mainly 用 next()；gaussian 係為咗符合 RandomSource interface）
 */
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
            // Box–Muller：兩個均勻 random → 一對標準正態
            const u = Math.max(Math.random(), Number.EPSILON);
            const v = Math.max(Math.random(), Number.EPSILON);
            const magnitude = Math.sqrt(-2 * Math.log(u));
            spare = magnitude * Math.sin(2 * Math.PI * v);
            return magnitude * Math.cos(2 * Math.PI * v);
        },
    };
}

/**
 * GA 用：同一個 genome 用 5 個唔同發射角各打一場，fitness 取平均。
 * 多場平均可以減低「幸運一場高分」嘅 noise。
 */
export function evaluateBreakerGenome(genome: Genome): number {
    return EVAL_LAUNCHES.reduce((sum, launch) => sum + simulateBreaker(genome, launch, false, createLiveRandom()).fitness, 0) / EVAL_LAUNCHES.length;
}

/**
 * UI 用：隨機抽一個發射角，錄低整場 replay frames 畀畫面播。
 * 每次 call 都重新 random，所以 loop / re-roll 會見到唔同版本。
 */
export function createBreakerReplay(genome: Genome): BreakerReplay {
    const launch = EVAL_LAUNCHES[Math.floor(Math.random() * EVAL_LAUNCHES.length)];
    return simulateBreaker(genome, launch, true, createLiveRandom()).replay;
}

/** 對應 8 維輸入（畀 UI 顯示 activation 標籤） */
export const BREAKER_INPUT_LABELS = ["板 X", "球 X", "球 Y", "速 X", "速 Y", "磚 Δx", "磚 Δy", "剩餘磚"] as const;

/** 對應 3 維輸出 */
export const BREAKER_OUTPUT_LABELS = ["向左", "停住", "向右"] as const;

/**
 * 由已錄嘅 frame 重建 NN 輸入向量。
 * UI 顯示 live activation 時用：唔使再跑 Matter.js，只讀 snapshot 計返 8 個數。
 * 正規化方式要同 simulateBreaker 入面嘅 input 完全一致。
 *
 * 輸入維度（大致 ∈ [-1, 1]）：
 *   [0] 板 X 正規化到 [-1, 1]
 *   [1] 球 X
 *   [2] 球 Y
 *   [3] 球速 X / 7
 *   [4] 球速 Y / 7
 *   [5] 最近磚相對球嘅 Δx
 *   [6] 最近磚相對球嘅 Δy
 *   [7] 剩餘磚比例（0～1）
 */
export function buildBreakerInputFromFrame(frame: BreakerFrame): number[] {
    const activeBricks = frame.bricks.filter(brick => brick.active);
    // 最近磚：用 |Δx| 揀（同模擬 loop 一致），冇磚就假一個喺球頭頂
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

/**
 * 跑完整一場 Breaker。
 *
 * @param genome          神經網絡權重
 * @param xVelocityFactor 基準水平發射係數（來自 EVAL_LAUNCHES）
 * @param record          true = 抽樣存 frames 做 replay；false = 淨係計 fitness（快）
 * @param random          亂數源（出生點、jitter、spin 都靠佢）
 */
function simulateBreaker(genome: Genome, xVelocityFactor: number, record: boolean, random: RandomSource): BreakerResult {
    // ── 1. 建立無重力物理世界 ──────────────────────────────────────────
    const engine = Matter.Engine.create({gravity: {x: 0, y: 0}});
    const runNetwork = networkAdapter.createRunner(genome);

    // 擋板起始 X 有隨機偏移：唔可以淨係背「中間接 serve」
    const paddleStartX = clamp(WIDTH / 2 + (random.next() - 0.5) * 56, PADDLE_WIDTH / 2, WIDTH - PADDLE_WIDTH / 2);
    const paddle = Matter.Bodies.rectangle(paddleStartX, HEIGHT - 28, PADDLE_WIDTH, 12, {
        isStatic: true, // 我哋自己 setPosition 移板，唔靠力
        label: "paddle",
        restitution: 1,
    });

    // 球出生位置微移，令第一下彈幾何每場唔同
    const ballStartX = clamp(WIDTH / 2 + (random.next() - 0.5) * 36, 24, WIDTH - 24);
    const ball = Matter.Bodies.circle(ballStartX, HEIGHT - 58 - random.next() * 6, 7, {
        label: "ball",
        restitution: 1,
        friction: 0,
        frictionAir: 0,
        inertia: Infinity, // 唔轉，淨係平移
    });

    // 左、右、頂牆；底部冇牆 → 跌出 HEIGHT 就 lost
    const walls = [
        Matter.Bodies.rectangle(-6, HEIGHT / 2, 12, HEIGHT, {isStatic: true, label: "wall"}),
        Matter.Bodies.rectangle(WIDTH + 6, HEIGHT / 2, 12, HEIGHT, {isStatic: true, label: "wall"}),
        Matter.Bodies.rectangle(WIDTH / 2, -6, WIDTH, 12, {isStatic: true, label: "wall"}),
    ];

    // ── 2. 磚塊網格 5 行 × 9 列 ───────────────────────────────────────
    const brickBodies: Matter.Body[] = [];
    const bricks = new Map<number, BreakerBrick>(); // id → 邏輯狀態（active / 座標）
    let brickId = 0;
    // 成個 grid 輕微平移，改變打磚次序，仍然保持喺畫面內
    const gridShiftX = (random.next() - 0.5) * 10;
    const gridShiftY = (random.next() - 0.5) * 6;
    for (let row = 0; row < 5; row += 1) {
        for (let column = 0; column < 9; column += 1) {
            const x = 42 + column * 59.5 + gridShiftX;
            const y = 50 + row * 25 + gridShiftY;
            const brick = Matter.Bodies.rectangle(x, y, 52, 16, {
                isStatic: true,
                label: `brick:${brickId}`, // collision 時靠 label 解 id
                restitution: 1,
            });
            brickBodies.push(brick);
            bricks.set(brickId, {id: brickId, x, y, active: true});
            brickId += 1;
        }
    }

    Matter.Composite.add(engine.world, [paddle, ball, ...walls, ...brickBodies]);

    // ── 3. 發球 ───────────────────────────────────────────────────────
    // 基準方向 + jitter；之後 normalize 保證速度 / |vy| 喺合理範圍
    const launchVx = 4.4 * xVelocityFactor + (random.next() - 0.5) * 1.05;
    const launchVy = -4.4 + (random.next() - 0.5) * 0.55; // 負 = 向上
    Matter.Body.setVelocity(ball, {x: launchVx, y: launchVy});
    normalizeBallVelocity(ball);

    let hits = 0; // 擋板成功接波次數
    let bricksCleared = 0;
    let executedSteps = 0;
    let trackingReward = 0; // 跟波 shaping reward 累積
    let terminal: NonNullable<BreakerFrame["terminal"]> = "timeout";
    const frames: BreakerFrame[] = [];
    // 抽樣間隔：保證 frames 唔超過 MAX_REPLAY_FRAMES；唔 record 時 stride 唔重要
    const frameStride = record ? Math.max(4, Math.ceil(MAX_STEPS / MAX_REPLAY_FRAMES)) : 4;

    // ── 4. 碰撞回調 ───────────────────────────────────────────────────
    const onCollisionStart = (event: Matter.IEventCollision<Matter.Engine>) => {
        event.pairs.forEach(pair => {
            const labels = [pair.bodyA.label, pair.bodyB.label];

            // 板接波：用球喺板上嘅偏移決定反彈角，再加 spin noise
            if (labels.includes("paddle") && labels.includes("ball")) {
                hits += 1;
                // offset ∈ 大約 [-1, 1]：球打左邊 → 負（向左彈），打右邊 → 正
                const offset = (ball.position.x - paddle.position.x) / (PADDLE_WIDTH / 2);
                const spin = (random.next() - 0.5) * 1.15;
                const bounceX = offset * 5.2 + spin;
                // 一定向上彈；保留最小 |vy|
                const bounceY = -Math.max(MIN_BALL_VY, Math.abs(ball.velocity.y) || 4.4) - random.next() * 0.55;
                Matter.Body.setVelocity(ball, {x: bounceX, y: bounceY});
                normalizeBallVelocity(ball);
            }

            // 撞磚：消磚、移走 body、加 scatter 令下一球軌唔好預測
            const brickBody = [pair.bodyA, pair.bodyB].find(body => body.label.startsWith("brick:"));
            if (brickBody) {
                const id = Number(brickBody.label.split(":")[1]);
                const brick = bricks.get(id);
                if (brick?.active) {
                    brick.active = false;
                    bricksCleared += 1;
                    Matter.Composite.remove(engine.world, brickBody);
                    nudgeBallVelocity(ball, random, 0.55);
                }
            }

            // 撞牆：細 jitter，防止永遠同一條反射角
            if (labels.includes("wall") && labels.includes("ball")) {
                nudgeBallVelocity(ball, random, 0.32);
            }
        });
    };

    Matter.Events.on(engine, "collisionStart", onCollisionStart);

    try {
        // ── 5. 主 loop：感知 → 決策 → 移板 → 物理 → 檢查結束 ─────────
        for (let step = 0; step < MAX_STEPS; step += 1) {
            executedSteps = step + 1;

            // 5a. 組 NN 輸入（同 buildBreakerInputFromFrame 同一套正規化）
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

            // 5b. NN 揀動作：0 左、1 停、2 右；每 step 移 7px
            const action = argMax(runNetwork(input));
            const movement = action === 0 ? -7 : action === 2 ? 7 : 0;
            Matter.Body.setPosition(paddle, {
                x: clamp(paddle.position.x + movement, PADDLE_WIDTH / 2, WIDTH - PADDLE_WIDTH / 2),
                y: paddle.position.y,
            });

            // 5c. Dense shaping：球落半場時，板越對正球 reward 越高
            //     幫 early generation 學「跟住球行」，唔使等消到磚先有訊號
            if (ball.position.y > HEIGHT * 0.45) {
                const align = 1 - Math.min(1, Math.abs(paddle.position.x - ball.position.x) / (PADDLE_WIDTH * 1.5));
                trackingReward += align * 0.08;
            }

            // 5d. 推進物理一幀（~16.67ms @ 60Hz）
            Matter.Engine.update(engine, 1000 / 60);

            // 撞完牆／磚後 Matter 可能令 vy≈0 → 強制修正
            normalizeBallVelocity(ball);

            // 5e. 抽樣錄 frame（只喺 record 模式）
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

            // 5f. 結束條件
            if (ball.position.y > HEIGHT + 20) {
                terminal = "lost"; // 球跌出場
                break;
            }
            if (bricksCleared === bricks.size) {
                terminal = "cleared"; // 全清
                break;
            }
            // 否則跑到 MAX_STEPS → terminal 保持 "timeout"
        }

        // 最後一幀帶 terminal，方便 UI 顯示結局
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
        // Matter 0.20 Engine.clear 淨係清 pairs/detector，
        // world bodies 同 event listener 要自己放，唔係 worker 會漏 memory
        Matter.Events.off(engine, "collisionStart", onCollisionStart);
        Matter.Composite.clear(engine.world, false);
        Matter.Engine.clear(engine);
    }

    // ── 6. Fitness 計分 ───────────────────────────────────────────────
    // 哲學：優先消磚 + 快；接波本身唔加分，無用嘅「耍波」仲要扣分。
    //
    // fitness =
    //   bricksCleared * 140      // 每磚底分
    // + clearBonus (2000)        // 全清大獎
    // + clearSpeedBonus          // 全清越快越高（二次項，差少少時間差好遠）
    // + paceBonus                // 途中消磚效率（未全清都有進度分）
    // + trackingScore            // 跟波 shaping（有 cap，唔可以蓋過消磚）
    // − wastePenalty             // 超出合理接波次數就罰
    //
    const totalBricks = bricks.size;
    const clearedAll = bricksCleared === totalBricks;
    const clearBonus = clearedAll ? 2_000 : 0;

    // 大約每消 1 磚准 2 下防守接波 + 4 下 buffer；多出嚟當 waste
    const allowedHits = bricksCleared * 2 + 4;
    const wasteHits = Math.max(0, hits - allowedHits);
    const wastePenalty = wasteHits * 6;

    // 消磚 / step：鼓勵快進度，唔使等全清
    const paceBonus = executedSteps > 0 ? (bricksCleared / executedSteps) * 1_400 : 0;

    // finishRatio = 剩低時間比例；平方後「快好多」>>「快少少」
    const finishRatio = clearedAll ? Math.max(0, 1 - executedSteps / MAX_STEPS) : 0;
    const clearSpeedBonus = finishRatio * finishRatio * 2_200 + finishRatio * 500;

    // tracking 有硬 cap：最多 ≈ 每磚 8 分 + 12，永遠rival 唔到一粒磚嘅 140
    const trackingCap = bricksCleared * 8 + 12;
    const trackingScore = Math.min(trackingReward, trackingCap);

    return {
        fitness: bricksCleared * 140 + clearBonus + clearSpeedBonus + paceBonus + trackingScore - wastePenalty,
        replay: {frames, bricksCleared, hits, steps: executedSteps},
    };
}

/** 對速度加細隨機偏移，再 normalize（牆／磚反彈後用） */
function nudgeBallVelocity(ball: Matter.Body, random: RandomSource, scale: number): void {
    Matter.Body.setVelocity(ball, {
        x: ball.velocity.x + (random.next() - 0.5) * scale * 2, // 水平 jitter 大啲
        y: ball.velocity.y + (random.next() - 0.5) * scale,
    });
    normalizeBallVelocity(ball);
}

/**
 * 強制球速 ∈ [MIN_BALL_SPEED, MAX_BALL_SPEED]，同 |vy| ≥ MIN_BALL_VY。
 *
 * 點解要：Matter.js 牆／磚反射有時會令 vy≈0，球就左右無限彈到 MAX_STEPS，
 * UI 好似卡死。呢度係 safety net。
 */
function normalizeBallVelocity(ball: Matter.Body): void {
    let vx = ball.velocity.x;
    let vy = ball.velocity.y;

    // NaN / Infinity → 重置成合理初始速度
    if (!Number.isFinite(vx) || !Number.isFinite(vy)) {
        Matter.Body.setVelocity(ball, {x: 3.2, y: -4.4});
        return;
    }

    // |vy| 太細：盡量保留原本方向；完全平底就睇球喺上定下半場推開
    if (Math.abs(vy) < MIN_BALL_VY) {
        const sign = Math.abs(vy) > 1e-6 ? Math.sign(vy) : ball.position.y < HEIGHT * 0.45 ? 1 : -1;
        vy = sign * MIN_BALL_VY;
    }

    let speed = Math.hypot(vx, vy);
    if (speed < 1e-6) {
        Matter.Body.setVelocity(ball, {x: 3.2, y: -4.4});
        return;
    }

    // 等比例縮放到目標速度
    const target = clamp(speed, MIN_BALL_SPEED, MAX_BALL_SPEED);
    vx = (vx / speed) * target;
    vy = (vy / speed) * target;

    // 縮放後 |vx| 好大時可能再壓扁 |vy|，所以再 assert 一次 floor
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
