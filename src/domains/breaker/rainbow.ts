import * as tf from "@tensorflow/tfjs";
import {argMax, calculateGeneCount, inspectGenome} from "../../lib/neuralNetwork";
import {createRandom, type RandomSource} from "../../lib/random";
import type {BreakerReplay, Genome, NetworkTopology} from "../../lib/types";
import {BREAKER_TOPOLOGY, runBreakerPolicy} from "./simulation";

/** 同 GA evaluate 一樣嘅一組基準發射角；每場再加 live random noise。 */
const EVAL_LAUNCHES = [0.58, 0.72, 0.86, 1, 1.18] as const;

/**
 * Rainbow DQN（教學版）組合：
 * - Double DQN：online 揀 action、target 估 Q
 * - Prioritized Experience Replay（proportional）
 * - Multi-step returns（n-step）
 * - Target network soft update
 *
 * 略過 C51 / Noisy Nets / Dueling，保持 8→12→3 同 GA 撞磚權重互換。
 */

export interface RainbowConfig {
    episodesPerUpdate: number;
    trainStepsPerUpdate: number;
    batchSize: number;
    maxSteps: number;
    learningRate: number;
    gamma: number;
    nStep: number;
    bufferSize: number;
    minBufferSize: number;
    tau: number;
    priorityAlpha: number;
    priorityBetaStart: number;
    priorityBetaEnd: number;
    betaAnnealingUpdates: number;
    epsilonStart: number;
    epsilonEnd: number;
    epsilonDecayUpdates: number;
    seed: number;
    speed: number;
}

export interface RainbowUpdateStats {
    update: number;
    bestUpdate: number;
    averageReturn: number;
    bestReturn: number;
    averageEpisodeLength: number;
    tdLoss: number;
    meanTdError: number;
    epsilon: number;
    bufferSize: number;
    beta: number;
}

export interface RainbowUpdateResult {
    stats: RainbowUpdateStats;
    agentGenome: Genome;
    replay: BreakerReplay;
}

export interface RainbowTrainer {
    online: tf.LayersModel;
    target: tf.LayersModel;
    optimizer: tf.Optimizer;
    learningRate: number;
    random: RandomSource;
    buffer: PrioritizedReplayBuffer;
    update: number;
    totalEnvSteps: number;
    bestAgentGenome: Genome | null;
    bestEvaluationReturn: number;
    bestReplay: BreakerReplay | null;
    bestUpdate: number;
}

interface PolicyEvaluation {
    averageReturn: number;
    averageEpisodeLength: number;
    replay: BreakerReplay;
}

interface NStepWindowItem {
    observation: number[];
    action: number;
    reward: number;
    nextObservation: number[];
    done: boolean;
}

export const DEFAULT_RAINBOW_CONFIG: RainbowConfig = {
    episodesPerUpdate: 4,
    trainStepsPerUpdate: 32,
    batchSize: 64,
    maxSteps: 100_000,
    learningRate: 0.0005,
    gamma: 0.99,
    nStep: 3,
    bufferSize: 20_000,
    minBufferSize: 512,
    tau: 0.01,
    priorityAlpha: 0.6,
    priorityBetaStart: 0.4,
    priorityBetaEnd: 1,
    betaAnnealingUpdates: 300,
    epsilonStart: 1,
    epsilonEnd: 0.05,
    epsilonDecayUpdates: 250,
    seed: Math.round(Math.random() * 1_000_000),
    speed: 5,
};

export function createRainbowTrainer(seed: number, learningRate: number, bufferSize = DEFAULT_RAINBOW_CONFIG.bufferSize): RainbowTrainer {
    const online = buildModel(BREAKER_TOPOLOGY, seed);
    const target = buildModel(BREAKER_TOPOLOGY, seed + 1);
    softUpdateTarget(online, target, 1);
    return {
        online,
        target,
        optimizer: tf.train.adam(learningRate),
        learningRate,
        random: createRandom(seed),
        buffer: createPrioritizedReplayBuffer(bufferSize),
        update: 0,
        totalEnvSteps: 0,
        bestAgentGenome: null,
        bestEvaluationReturn: Number.NEGATIVE_INFINITY,
        bestReplay: null,
        bestUpdate: 0,
    };
}

export async function trainRainbowUpdate(trainer: RainbowTrainer, config: RainbowConfig): Promise<RainbowUpdateResult> {
    await tf.ready();
    refreshOptimizer(trainer, config.learningRate);
    ensureBufferCapacity(trainer, config.bufferSize);

    const agentGenome = modelToGenome(trainer.online, BREAKER_TOPOLOGY);
    const qRunner = createLinearOutputRunner(agentGenome, BREAKER_TOPOLOGY);
    const epsilon = currentEpsilon(trainer.update, config);

    for (let episode = 0; episode < config.episodesPerUpdate; episode += 1) {
        // 場景用 live Math.random（同 GA）；trainer.random 只負責 ε-greedy 抽樣。
        const window: NStepWindowItem[] = [];
        const result = runBreakerPolicy(
            function selectAction(observation) {
                if (trainer.random.next() < epsilon) {
                    return {action: trainer.random.integer(0, BREAKER_TOPOLOGY.outputSize - 1)};
                }
                return {action: argMax(qRunner(observation))};
            },
            {maxSteps: config.maxSteps}
        );

        result.transitions.forEach(function pushTransition(transition) {
            trainer.totalEnvSteps += 1;
            appendNStepTransition(
                trainer.buffer,
                window,
                {
                    observation: transition.observation,
                    action: transition.action,
                    reward: transition.reward,
                    nextObservation: transition.nextObservation,
                    done: transition.done,
                },
                config
            );
        });
        flushRemainingNStep(trainer.buffer, window, config);
    }

    const beta = currentBeta(trainer.update, config);
    let tdLoss = 0;
    let meanTdError = 0;
    if (trainer.buffer.size >= Math.min(config.minBufferSize, config.batchSize)) {
        const losses = optimize(trainer, config, beta);
        tdLoss = losses.tdLoss;
        meanTdError = losses.meanTdError;
        softUpdateTarget(trainer.online, trainer.target, config.tau);
    }

    trainer.update += 1;
    const updatedGenome = modelToGenome(trainer.online, BREAKER_TOPOLOGY);
    const evaluation = evaluatePolicy(updatedGenome, config);
    if (evaluation.averageReturn > trainer.bestEvaluationReturn || !trainer.bestAgentGenome || !trainer.bestReplay) {
        trainer.bestEvaluationReturn = evaluation.averageReturn;
        trainer.bestAgentGenome = [...updatedGenome];
        trainer.bestReplay = evaluation.replay;
        trainer.bestUpdate = trainer.update;
    }

    return {
        agentGenome: [...trainer.bestAgentGenome],
        replay: trainer.bestReplay,
        stats: {
            update: trainer.update,
            bestUpdate: trainer.bestUpdate,
            averageReturn: evaluation.averageReturn,
            bestReturn: trainer.bestEvaluationReturn,
            averageEpisodeLength: evaluation.averageEpisodeLength,
            tdLoss,
            meanTdError,
            epsilon,
            bufferSize: trainer.buffer.size,
            beta,
        },
    };
}

export function loadRainbowAgentGenome(trainer: RainbowTrainer, genome: Genome, config: RainbowConfig): RainbowUpdateResult {
    applyGenomeToModel(trainer.online, genome, BREAKER_TOPOLOGY);
    softUpdateTarget(trainer.online, trainer.target, 1);
    trainer.optimizer.dispose();
    trainer.optimizer = tf.train.adam(config.learningRate);
    trainer.learningRate = config.learningRate;
    trainer.update = 0;
    trainer.totalEnvSteps = 0;
    ensureBufferCapacity(trainer, config.bufferSize);
    trainer.buffer = createPrioritizedReplayBuffer(config.bufferSize);

    const evaluation = evaluatePolicy(genome, config);
    trainer.bestAgentGenome = [...genome];
    trainer.bestEvaluationReturn = evaluation.averageReturn;
    trainer.bestReplay = evaluation.replay;
    trainer.bestUpdate = 0;

    return {
        agentGenome: [...genome],
        replay: evaluation.replay,
        stats: {
            update: 0,
            bestUpdate: 0,
            averageReturn: evaluation.averageReturn,
            bestReturn: evaluation.averageReturn,
            averageEpisodeLength: evaluation.averageEpisodeLength,
            tdLoss: 0,
            meanTdError: 0,
            epsilon: currentEpsilon(0, config),
            bufferSize: 0,
            beta: currentBeta(0, config),
        },
    };
}

/**
 * UI 用：用 greedy Q-network 跑一場真·隨機場景並錄 replay。
 * 每次 call 都重新抽發球／板位／磚位，loop 重播會見到唔同版本。
 */
export function createRainbowAgentReplay(genome: Genome, maxSteps = DEFAULT_RAINBOW_CONFIG.maxSteps): BreakerReplay {
    const qRunner = createLinearOutputRunner(genome, BREAKER_TOPOLOGY);
    return runBreakerPolicy(
        function selectShowcaseAction(observation) {
            return {action: argMax(qRunner(observation))};
        },
        {maxSteps, record: true}
    ).replay;
}

export function disposeRainbowTrainer(trainer: RainbowTrainer): void {
    trainer.online.dispose();
    trainer.target.dispose();
    trainer.optimizer.dispose();
}

export function currentEpsilon(update: number, config: RainbowConfig): number {
    if (config.epsilonDecayUpdates <= 0) {
        return config.epsilonEnd;
    }
    const progress = Math.min(1, update / config.epsilonDecayUpdates);
    return config.epsilonStart + (config.epsilonEnd - config.epsilonStart) * progress;
}

export function currentBeta(update: number, config: RainbowConfig): number {
    if (config.betaAnnealingUpdates <= 0) {
        return config.priorityBetaEnd;
    }
    const progress = Math.min(1, update / config.betaAnnealingUpdates);
    return config.priorityBetaStart + (config.priorityBetaEnd - config.priorityBetaStart) * progress;
}

/** 推入一步；滿 n-step 或遇到 done 就摺成 (s0, a0, R_n, s_n, done) 寫入 PER。 */
export function appendNStepTransition(buffer: PrioritizedReplayBuffer, window: NStepWindowItem[], item: NStepWindowItem, config: RainbowConfig): void {
    window.push(item);
    while (window.length >= config.nStep) {
        emitNStepTransition(buffer, window, config.nStep, config.gamma);
        window.shift();
    }
    if (item.done) {
        flushRemainingNStep(buffer, window, config);
    }
}

/** episode 完結時清晒未滿 n 嘅尾巴。 */
export function flushRemainingNStep(buffer: PrioritizedReplayBuffer, window: NStepWindowItem[], config: RainbowConfig): void {
    while (window.length > 0) {
        emitNStepTransition(buffer, window, window.length, config.gamma);
        window.shift();
    }
}

function emitNStepTransition(buffer: PrioritizedReplayBuffer, window: NStepWindowItem[], horizon: number, gamma: number): void {
    const first = window[0];
    let discountedReturn = 0;
    let gammaPower = 1;
    let lastIndex = 0;
    for (let offset = 0; offset < horizon; offset += 1) {
        const item = window[offset];
        discountedReturn += gammaPower * item.reward;
        gammaPower *= gamma;
        lastIndex = offset;
        if (item.done) {
            break;
        }
    }
    const last = window[lastIndex];
    addTransition(buffer, {
        observation: first.observation,
        action: first.action,
        reward: discountedReturn,
        nextObservation: last.nextObservation,
        done: last.done,
        gammaN: Math.pow(gamma, lastIndex + 1),
    });
}

// ── Prioritized Experience Replay ──────────────────────────────────────────

export interface StoredTransition {
    observation: number[];
    action: number;
    reward: number;
    nextObservation: number[];
    done: boolean;
    /** γ^n for the n-step bootstrap term. */
    gammaN: number;
}

export interface PrioritizedReplayBuffer {
    capacity: number;
    size: number;
    position: number;
    maxPriority: number;
    observations: number[][];
    actions: number[];
    rewards: number[];
    nextObservations: number[][];
    dones: boolean[];
    gammaNs: number[];
    priorities: Float64Array;
}

export function createPrioritizedReplayBuffer(capacity: number): PrioritizedReplayBuffer {
    return {
        capacity,
        size: 0,
        position: 0,
        maxPriority: 1,
        observations: new Array(capacity),
        actions: new Array(capacity),
        rewards: new Array(capacity),
        nextObservations: new Array(capacity),
        dones: new Array(capacity),
        gammaNs: new Array(capacity),
        priorities: new Float64Array(capacity),
    };
}

export function addTransition(buffer: PrioritizedReplayBuffer, transition: StoredTransition): void {
    const index = buffer.position;
    buffer.observations[index] = transition.observation;
    buffer.actions[index] = transition.action;
    buffer.rewards[index] = transition.reward;
    buffer.nextObservations[index] = transition.nextObservation;
    buffer.dones[index] = transition.done;
    buffer.gammaNs[index] = transition.gammaN;
    buffer.priorities[index] = buffer.maxPriority;
    buffer.position = (buffer.position + 1) % buffer.capacity;
    buffer.size = Math.min(buffer.size + 1, buffer.capacity);
}

export function samplePrioritizedBatch(
    buffer: PrioritizedReplayBuffer,
    batchSize: number,
    alpha: number,
    beta: number,
    random: RandomSource
): {indices: number[]; weights: number[]; transitions: StoredTransition[]} {
    const size = buffer.size;
    const count = Math.min(batchSize, size);
    const probabilities = new Float64Array(size);
    let total = 0;
    for (let index = 0; index < size; index += 1) {
        const priority = Math.pow(Math.max(buffer.priorities[index], 1e-6), alpha);
        probabilities[index] = priority;
        total += priority;
    }

    const indices: number[] = [];
    const weights: number[] = [];
    const transitions: StoredTransition[] = [];
    let maxWeight = 0;

    for (let sample = 0; sample < count; sample += 1) {
        let threshold = random.next() * total;
        let chosen = size - 1;
        for (let index = 0; index < size; index += 1) {
            threshold -= probabilities[index];
            if (threshold <= 0) {
                chosen = index;
                break;
            }
        }
        const probability = probabilities[chosen] / total;
        const weight = Math.pow(size * probability, -beta);
        indices.push(chosen);
        weights.push(weight);
        maxWeight = Math.max(maxWeight, weight);
        transitions.push({
            observation: buffer.observations[chosen],
            action: buffer.actions[chosen],
            reward: buffer.rewards[chosen],
            nextObservation: buffer.nextObservations[chosen],
            done: buffer.dones[chosen],
            gammaN: buffer.gammaNs[chosen],
        });
    }

    const normalizedWeights = weights.map(function normalizeWeight(weight) {
        return weight / Math.max(maxWeight, 1e-8);
    });
    return {indices, weights: normalizedWeights, transitions};
}

export function updatePriorities(buffer: PrioritizedReplayBuffer, indices: number[], tdErrors: number[]): void {
    for (let index = 0; index < indices.length; index += 1) {
        const priority = Math.abs(tdErrors[index]) + 1e-6;
        buffer.priorities[indices[index]] = priority;
        buffer.maxPriority = Math.max(buffer.maxPriority, priority);
    }
}

// ── Internals ──────────────────────────────────────────────────────────────

function evaluatePolicy(agentGenome: Genome, config: RainbowConfig): PolicyEvaluation {
    const qRunner = createLinearOutputRunner(agentGenome, BREAKER_TOPOLOGY);
    const episodes = EVAL_LAUNCHES.map(function evaluateLaunch(launch) {
        return runBreakerPolicy(
            function selectEvaluationAction(observation) {
                return {action: argMax(qRunner(observation))};
            },
            {
                launch,
                maxSteps: config.maxSteps,
                record: true,
            }
        );
    });
    const bestEpisode = episodes.reduce(function selectBetter(best, episode) {
        return episode.return > best.return ? episode : best;
    });
    return {
        averageReturn: mean(
            episodes.map(function readEvaluationReturn(episode) {
                return episode.return;
            })
        ),
        averageEpisodeLength: mean(
            episodes.map(function readEvaluationLength(episode) {
                return episode.replay.steps;
            })
        ),
        replay: bestEpisode.replay,
    };
}

function optimize(trainer: RainbowTrainer, config: RainbowConfig, beta: number): {tdLoss: number; meanTdError: number} {
    let lastLoss = 0;
    let lastMeanTd = 0;
    const variables = trainer.online.trainableWeights.map(function readVariable(weight) {
        return weight.read() as tf.Variable;
    });

    for (let step = 0; step < config.trainStepsPerUpdate; step += 1) {
        const batch = samplePrioritizedBatch(trainer.buffer, config.batchSize, config.priorityAlpha, beta, trainer.random);
        if (batch.transitions.length === 0) {
            break;
        }

        const observations = tf.tensor2d(
            batch.transitions.map(function readObservation(transition) {
                return transition.observation;
            }),
            [batch.transitions.length, BREAKER_TOPOLOGY.inputSize]
        );
        const nextObservations = tf.tensor2d(
            batch.transitions.map(function readNextObservation(transition) {
                return transition.nextObservation;
            }),
            [batch.transitions.length, BREAKER_TOPOLOGY.inputSize]
        );
        const actions = tf.tensor1d(
            batch.transitions.map(function readAction(transition) {
                return transition.action;
            }),
            "int32"
        );
        const rewards = tf.tensor1d(
            batch.transitions.map(function readReward(transition) {
                return transition.reward;
            })
        );
        const dones = tf.tensor1d(
            batch.transitions.map(function readDone(transition) {
                return transition.done ? 1 : 0;
            })
        );
        const gammaNs = tf.tensor1d(
            batch.transitions.map(function readGammaN(transition) {
                return transition.gammaN;
            })
        );
        const importance = tf.tensor1d(batch.weights);

        // Double DQN targets outside the minimize graph so priorities can reuse the same deltas.
        const targets = tf.tidy(function buildTargets() {
            const nextOnline = trainer.online.apply(nextObservations) as tf.Tensor2D;
            const bestActions = tf.argMax(nextOnline, 1);
            const nextTarget = trainer.target.apply(nextObservations) as tf.Tensor2D;
            const nextBestQ = tf.sum(tf.mul(nextTarget, tf.oneHot(bestActions, BREAKER_TOPOLOGY.outputSize)), 1);
            const bootstrap = tf.mul(tf.mul(gammaNs, nextBestQ), tf.sub(1, dones));
            return tf.add(rewards, bootstrap);
        });

        const selectedQ = tf.tidy(function computeSelectedQ() {
            const qValues = trainer.online.apply(observations) as tf.Tensor2D;
            return tf.sum(tf.mul(qValues, tf.oneHot(actions, BREAKER_TOPOLOGY.outputSize)), 1);
        });
        const tdErrorsTensor = tf.sub(targets, selectedQ);
        const tdErrors = Array.from(tdErrorsTensor.dataSync());
        updatePriorities(trainer.buffer, batch.indices, tdErrors);
        lastMeanTd = mean(tdErrors.map(Math.abs));

        const loss = trainer.optimizer.minimize(
            function calculateTdLoss() {
                return tf.tidy(function tdLossScope() {
                    const qValues = trainer.online.apply(observations, {training: true}) as tf.Tensor2D;
                    const qSelected = tf.sum(tf.mul(qValues, tf.oneHot(actions, BREAKER_TOPOLOGY.outputSize)), 1);
                    // TF.js 對 lessEqual/where 冇梯度，用 importance-weighted MSE。
                    const squared = tf.square(tf.sub(targets, qSelected));
                    return tf.mean(tf.mul(importance, squared));
                });
            },
            true,
            variables
        );

        lastLoss = loss?.dataSync()[0] ?? 0;
        loss?.dispose();
        observations.dispose();
        nextObservations.dispose();
        actions.dispose();
        rewards.dispose();
        dones.dispose();
        gammaNs.dispose();
        importance.dispose();
        targets.dispose();
        selectedQ.dispose();
        tdErrorsTensor.dispose();
    }

    return {tdLoss: lastLoss, meanTdError: lastMeanTd};
}

function softUpdateTarget(online: tf.LayersModel, target: tf.LayersModel, tau: number): void {
    const onlineWeights = online.getWeights();
    const targetWeights = target.getWeights();
    const mixed = onlineWeights.map(function blend(weight, index) {
        return tf.tidy(function blendScope() {
            if (tau >= 1) {
                return tf.clone(weight);
            }
            return tf.add(tf.mul(weight, tau), tf.mul(targetWeights[index], 1 - tau));
        });
    });
    target.setWeights(mixed);
    mixed.forEach(function disposeTensor(tensor) {
        tensor.dispose();
    });
}

function ensureBufferCapacity(trainer: RainbowTrainer, capacity: number): void {
    if (trainer.buffer.capacity === capacity) {
        return;
    }
    // 容量改動時重建 buffer，避免半舊半新嘅 index 錯亂。
    trainer.buffer = createPrioritizedReplayBuffer(capacity);
}

function buildModel(topology: NetworkTopology, seed: number): tf.LayersModel {
    const sizes = [topology.inputSize, ...topology.hiddenLayers, topology.outputSize];
    const model = tf.sequential();
    for (let layer = 1; layer < sizes.length; layer += 1) {
        model.add(
            tf.layers.dense({
                units: sizes[layer],
                ...(layer === 1 ? {inputShape: [sizes[0]]} : {}),
                activation: layer === sizes.length - 1 ? "linear" : "tanh",
                kernelInitializer: tf.initializers.glorotUniform({seed: seed + layer}),
                biasInitializer: "zeros",
                useBias: true,
            })
        );
    }
    return model;
}

function modelToGenome(model: tf.LayersModel, topology: NetworkTopology): Genome {
    const sizes = [topology.inputSize, ...topology.hiddenLayers, topology.outputSize];
    const weights = model.getWeights();
    const genome: number[] = [];
    let tensorIndex = 0;
    for (let layer = 1; layer < sizes.length; layer += 1) {
        const inputSize = sizes[layer - 1];
        const outputSize = sizes[layer];
        const kernel = Array.from(weights[tensorIndex].dataSync());
        const biases = Array.from(weights[tensorIndex + 1].dataSync());
        genome.push(...biases);
        for (let node = 0; node < outputSize; node += 1) {
            for (let input = 0; input < inputSize; input += 1) {
                genome.push(kernel[input * outputSize + node]);
            }
        }
        tensorIndex += 2;
    }
    if (genome.length !== calculateGeneCount(topology)) {
        throw new Error("Rainbow 網絡權重格式錯誤。");
    }
    return genome;
}

function applyGenomeToModel(model: tf.LayersModel, genome: Genome, topology: NetworkTopology): void {
    const inspection = inspectGenome(genome, topology);
    const tensors: tf.Tensor[] = [];
    for (const layer of inspection.layers) {
        const outputSize = layer.biases.length;
        const inputSize = layer.weights[0]?.length ?? 0;
        const kernel = new Float32Array(inputSize * outputSize);
        for (let output = 0; output < outputSize; output += 1) {
            for (let input = 0; input < inputSize; input += 1) {
                kernel[input * outputSize + output] = layer.weights[output][input];
            }
        }
        tensors.push(tf.tensor2d(kernel, [inputSize, outputSize]));
        tensors.push(tf.tensor1d(layer.biases));
    }
    model.setWeights(tensors);
    tensors.forEach(function disposeTensor(tensor) {
        tensor.dispose();
    });
}

function createLinearOutputRunner(genome: Genome, topology: NetworkTopology): (observation: number[]) => number[] {
    const inspection = inspectGenome(genome, topology);
    return function runNetwork(observation) {
        let previous = observation;
        for (let layerIndex = 0; layerIndex < inspection.layers.length; layerIndex += 1) {
            const layer = inspection.layers[layerIndex];
            const isOutput = layerIndex === inspection.layers.length - 1;
            const next = layer.biases.map(function runNode(bias, node) {
                const total = layer.weights[node].reduce(function addWeight(sum, weight, input) {
                    return sum + weight * previous[input];
                }, bias);
                return isOutput ? total : Math.tanh(total);
            });
            previous = next;
        }
        return previous;
    };
}

function refreshOptimizer(trainer: RainbowTrainer, learningRate: number): void {
    if (trainer.learningRate === learningRate) {
        return;
    }
    trainer.optimizer.dispose();
    trainer.optimizer = tf.train.adam(learningRate);
    trainer.learningRate = learningRate;
}

function mean(values: number[]): number {
    if (values.length === 0) {
        return 0;
    }
    return (
        values.reduce(function add(sum, value) {
            return sum + value;
        }, 0) / values.length
    );
}
