import * as tf from "@tensorflow/tfjs";
import {argMax, calculateGeneCount, inspectGenome} from "../../lib/neuralNetwork";
import {createRandom, type RandomSource} from "../../lib/random";
import type {BreakerReplay, Genome, NetworkTopology} from "../../lib/types";
import {BREAKER_TOPOLOGY, runBreakerPolicy, type BreakerPolicyTransition} from "./simulation";

/** 同 GA evaluate 一樣嘅一組基準發射角；每場再加 live random noise。 */
const EVAL_LAUNCHES = [0.58, 0.72, 0.86, 1, 1.18] as const;

const CRITIC_TOPOLOGY: NetworkTopology = {
    inputSize: BREAKER_TOPOLOGY.inputSize,
    hiddenLayers: [...BREAKER_TOPOLOGY.hiddenLayers],
    outputSize: 1,
};

export interface PpoConfig {
    episodesPerUpdate: number;
    maxSteps: number;
    epochs: number;
    learningRate: number;
    gamma: number;
    gaeLambda: number;
    clipRatio: number;
    entropyCoefficient: number;
    seed: number;
    speed: number;
}

export interface PpoUpdateStats {
    update: number;
    bestUpdate: number;
    averageReturn: number;
    bestReturn: number;
    averageEpisodeLength: number;
    policyLoss: number;
    valueLoss: number;
    entropy: number;
}

export interface PpoUpdateResult {
    stats: PpoUpdateStats;
    actorGenome: Genome;
    replay: BreakerReplay;
}

export interface PpoTrainer {
    actor: tf.LayersModel;
    critic: tf.LayersModel;
    actorOptimizer: tf.Optimizer;
    criticOptimizer: tf.Optimizer;
    learningRate: number;
    random: RandomSource;
    update: number;
    bestActorGenome: Genome | null;
    bestEvaluationReturn: number;
    bestReplay: BreakerReplay | null;
    bestUpdate: number;
}

interface TrainingBatch {
    observations: number[][];
    actions: number[];
    oldLogProbabilities: number[];
    advantages: number[];
    returns: number[];
}

interface EpisodeBatch {
    transitions: BreakerPolicyTransition[];
    return: number;
    length: number;
}

interface PolicyEvaluation {
    averageReturn: number;
    averageEpisodeLength: number;
    replay: BreakerReplay;
}

export const DEFAULT_PPO_CONFIG: PpoConfig = {
    episodesPerUpdate: 8,
    maxSteps: 100_000,
    epochs: 4,
    learningRate: 0.0008,
    gamma: 0.99,
    gaeLambda: 0.95,
    clipRatio: 0.2,
    entropyCoefficient: 0.01,
    seed: Math.round(Math.random() * 1_000_000),
    speed: 5,
};

export function createPpoTrainer(seed: number, learningRate: number): PpoTrainer {
    return {
        actor: buildModel(BREAKER_TOPOLOGY, "linear", seed),
        critic: buildModel(CRITIC_TOPOLOGY, "linear", seed + 1),
        actorOptimizer: tf.train.adam(learningRate),
        criticOptimizer: tf.train.adam(learningRate),
        learningRate,
        random: createRandom(seed),
        update: 0,
        bestActorGenome: null,
        bestEvaluationReturn: Number.NEGATIVE_INFINITY,
        bestReplay: null,
        bestUpdate: 0,
    };
}

export async function trainPpoUpdate(trainer: PpoTrainer, config: PpoConfig): Promise<PpoUpdateResult> {
    await tf.ready();
    refreshOptimizers(trainer, config.learningRate);

    const actorGenome = modelToGenome(trainer.actor, BREAKER_TOPOLOGY);
    const criticGenome = modelToGenome(trainer.critic, CRITIC_TOPOLOGY);
    const actorRunner = createLinearOutputRunner(actorGenome, BREAKER_TOPOLOGY);
    const criticRunner = createLinearOutputRunner(criticGenome, CRITIC_TOPOLOGY);
    const episodes: EpisodeBatch[] = [];

    for (let episode = 0; episode < config.episodesPerUpdate; episode += 1) {
        // 場景用 live Math.random（同 GA）：發球／板位／磚位每場唔同，逼策略學跟波。
        // trainer.random 只負責 stochastic action sampling。
        const result = runBreakerPolicy(
            function selectAction(observation) {
                const probabilities = probabilitiesFromLogits(actorRunner(observation));
                const action = sampleAction(probabilities, trainer.random);
                return {
                    action,
                    logProbability: Math.log(Math.max(probabilities[action], 1e-8)),
                    value: criticRunner(observation)[0],
                };
            },
            {
                maxSteps: config.maxSteps,
            }
        );
        episodes.push({transitions: result.transitions, return: result.return, length: result.replay.steps});
    }

    const batch = buildTrainingBatch(episodes, criticRunner, config);
    const losses = optimize(trainer, batch, config);
    trainer.update += 1;
    const updatedActorGenome = modelToGenome(trainer.actor, BREAKER_TOPOLOGY);
    const evaluation = evaluatePolicy(updatedActorGenome, config);
    if (evaluation.averageReturn > trainer.bestEvaluationReturn || !trainer.bestActorGenome || !trainer.bestReplay) {
        trainer.bestEvaluationReturn = evaluation.averageReturn;
        trainer.bestActorGenome = [...updatedActorGenome];
        trainer.bestReplay = evaluation.replay;
        trainer.bestUpdate = trainer.update;
    }

    return {
        actorGenome: [...trainer.bestActorGenome],
        replay: trainer.bestReplay,
        stats: {
            update: trainer.update,
            bestUpdate: trainer.bestUpdate,
            averageReturn: evaluation.averageReturn,
            bestReturn: trainer.bestEvaluationReturn,
            averageEpisodeLength: evaluation.averageEpisodeLength,
            policyLoss: losses.policyLoss,
            valueLoss: losses.valueLoss,
            entropy: losses.entropy,
        },
    };
}

export function loadPpoActorGenome(trainer: PpoTrainer, genome: Genome, config: PpoConfig): PpoUpdateResult {
    applyGenomeToModel(trainer.actor, genome, BREAKER_TOPOLOGY);
    trainer.actorOptimizer.dispose();
    trainer.actorOptimizer = tf.train.adam(config.learningRate);
    trainer.learningRate = config.learningRate;
    trainer.update = 0;

    const evaluation = evaluatePolicy(genome, config);
    trainer.bestActorGenome = [...genome];
    trainer.bestEvaluationReturn = evaluation.averageReturn;
    trainer.bestReplay = evaluation.replay;
    trainer.bestUpdate = 0;

    return {
        actorGenome: [...genome],
        replay: evaluation.replay,
        stats: {
            update: 0,
            bestUpdate: 0,
            averageReturn: evaluation.averageReturn,
            bestReturn: evaluation.averageReturn,
            averageEpisodeLength: evaluation.averageEpisodeLength,
            policyLoss: 0,
            valueLoss: 0,
            entropy: 0,
        },
    };
}

/**
 * UI 用：用 greedy actor 跑一場真·隨機場景並錄 replay。
 * 每次 call 都重新抽發球／板位／磚位，loop 重播會見到唔同版本。
 */
export function createPpoActorReplay(genome: Genome, maxSteps = DEFAULT_PPO_CONFIG.maxSteps): BreakerReplay {
    const actorRunner = createLinearOutputRunner(genome, BREAKER_TOPOLOGY);
    return runBreakerPolicy(
        function selectShowcaseAction(observation) {
            return {action: argMax(actorRunner(observation))};
        },
        {maxSteps, record: true}
    ).replay;
}

function evaluatePolicy(actorGenome: Genome, config: PpoConfig): PolicyEvaluation {
    const actorRunner = createLinearOutputRunner(actorGenome, BREAKER_TOPOLOGY);
    // 同 GA：固定一組基準發射角，但每場用 live random 做出生點／jitter（唔 seed）。
    const episodes = EVAL_LAUNCHES.map(function evaluateLaunch(launch) {
        return runBreakerPolicy(
            function selectEvaluationAction(observation) {
                return {action: argMax(actorRunner(observation))};
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

export function disposePpoTrainer(trainer: PpoTrainer): void {
    trainer.actor.dispose();
    trainer.critic.dispose();
    trainer.actorOptimizer.dispose();
    trainer.criticOptimizer.dispose();
}

export function probabilitiesFromLogits(logits: number[]): number[] {
    const maximum = Math.max(...logits);
    const exponentials = logits.map(function exponentiate(logit) {
        return Math.exp(logit - maximum);
    });
    const total = exponentials.reduce(function add(sum, value) {
        return sum + value;
    }, 0);
    return exponentials.map(function normalize(value) {
        return value / total;
    });
}

export function calculateGeneralizedAdvantages(transitions: BreakerPolicyTransition[], nextValues: number[], gamma: number, gaeLambda: number): {advantages: number[]; returns: number[]} {
    const advantages = new Array<number>(transitions.length);
    const returns = new Array<number>(transitions.length);
    let generalizedAdvantage = 0;

    for (let index = transitions.length - 1; index >= 0; index -= 1) {
        const transition = transitions[index];
        const continuation = transition.done ? 0 : 1;
        const delta = transition.reward + gamma * nextValues[index] * continuation - transition.value;
        generalizedAdvantage = delta + gamma * gaeLambda * continuation * generalizedAdvantage;
        advantages[index] = generalizedAdvantage;
        returns[index] = generalizedAdvantage + transition.value;
    }

    return {advantages, returns};
}

function buildModel(topology: NetworkTopology, outputActivation: "linear" | "tanh", seed: number): tf.LayersModel {
    const sizes = [topology.inputSize, ...topology.hiddenLayers, topology.outputSize];
    const model = tf.sequential();
    for (let layer = 1; layer < sizes.length; layer += 1) {
        model.add(
            tf.layers.dense({
                units: sizes[layer],
                ...(layer === 1 ? {inputShape: [sizes[0]]} : {}),
                activation: layer === sizes.length - 1 ? outputActivation : "tanh",
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
        throw new Error("PPO 網絡權重格式錯誤。");
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

function sampleAction(probabilities: number[], random: RandomSource): number {
    const threshold = random.next();
    let cumulative = 0;
    for (let action = 0; action < probabilities.length; action += 1) {
        cumulative += probabilities[action];
        if (threshold <= cumulative) {
            return action;
        }
    }
    return probabilities.length - 1;
}

function buildTrainingBatch(episodes: EpisodeBatch[], criticRunner: (observation: number[]) => number[], config: PpoConfig): TrainingBatch {
    const observations: number[][] = [];
    const actions: number[] = [];
    const oldLogProbabilities: number[] = [];
    const advantages: number[] = [];
    const returns: number[] = [];

    episodes.forEach(function appendEpisode(episode) {
        const nextValues = episode.transitions.map(function readNextValue(transition) {
            return transition.done ? 0 : criticRunner(transition.nextObservation)[0];
        });
        const calculated = calculateGeneralizedAdvantages(episode.transitions, nextValues, config.gamma, config.gaeLambda);
        episode.transitions.forEach(function appendTransition(transition, index) {
            observations.push(transition.observation);
            actions.push(transition.action);
            oldLogProbabilities.push(transition.logProbability);
            advantages.push(calculated.advantages[index]);
            returns.push(calculated.returns[index]);
        });
    });

    const advantageMean = mean(advantages);
    const variance = mean(
        advantages.map(function squareDifference(value) {
            return (value - advantageMean) ** 2;
        })
    );
    const deviation = Math.sqrt(variance + 1e-8);
    const normalizedAdvantages = advantages.map(function normalizeAdvantage(value) {
        return (value - advantageMean) / deviation;
    });
    return {observations, actions, oldLogProbabilities, advantages: normalizedAdvantages, returns};
}

function optimize(trainer: PpoTrainer, batch: TrainingBatch, config: PpoConfig): {policyLoss: number; valueLoss: number; entropy: number} {
    const observations = tf.tensor2d(batch.observations, [batch.observations.length, BREAKER_TOPOLOGY.inputSize]);
    const actions = tf.oneHot(tf.tensor1d(batch.actions, "int32"), BREAKER_TOPOLOGY.outputSize);
    const oldLogProbabilities = tf.tensor1d(batch.oldLogProbabilities);
    const advantages = tf.tensor1d(batch.advantages);
    const returns = tf.tensor1d(batch.returns);
    const actorVariables = trainer.actor.trainableWeights.map(function readVariable(weight) {
        return weight.read() as tf.Variable;
    });
    const criticVariables = trainer.critic.trainableWeights.map(function readVariable(weight) {
        return weight.read() as tf.Variable;
    });
    let policyLossValue = 0;
    let valueLossValue = 0;

    for (let epoch = 0; epoch < config.epochs; epoch += 1) {
        const policyLoss = trainer.actorOptimizer.minimize(
            function calculatePolicyLoss() {
                return tf.tidy(function policyLossScope() {
                    const logits = trainer.actor.apply(observations, {training: true}) as tf.Tensor2D;
                    const logProbabilities = tf.logSoftmax(logits);
                    const selectedLogProbabilities = tf.sum(tf.mul(logProbabilities, actions), 1);
                    const ratio = tf.exp(tf.sub(selectedLogProbabilities, oldLogProbabilities));
                    const unclipped = tf.mul(ratio, advantages);
                    const clipped = tf.mul(tf.clipByValue(ratio, 1 - config.clipRatio, 1 + config.clipRatio), advantages);
                    const entropy = tf.neg(tf.mean(tf.sum(tf.mul(tf.softmax(logits), logProbabilities), 1)));
                    return tf.sub(tf.neg(tf.mean(tf.minimum(unclipped, clipped))), tf.mul(entropy, config.entropyCoefficient));
                });
            },
            true,
            actorVariables
        );
        const valueLoss = trainer.criticOptimizer.minimize(
            function calculateValueLoss() {
                return tf.tidy(function valueLossScope() {
                    const values = tf.squeeze(trainer.critic.apply(observations, {training: true}) as tf.Tensor2D, [1]);
                    return tf.mul(tf.mean(tf.square(tf.sub(returns, values))), 0.5);
                });
            },
            true,
            criticVariables
        );
        policyLossValue = policyLoss?.dataSync()[0] ?? 0;
        valueLossValue = valueLoss?.dataSync()[0] ?? 0;
        policyLoss?.dispose();
        valueLoss?.dispose();
    }

    const entropy = tf.tidy(function entropyScope() {
        const logits = trainer.actor.apply(observations) as tf.Tensor2D;
        const logProbabilities = tf.logSoftmax(logits);
        return tf.neg(tf.mean(tf.sum(tf.mul(tf.softmax(logits), logProbabilities), 1)));
    });
    const entropyValue = entropy.dataSync()[0];
    entropy.dispose();
    observations.dispose();
    actions.dispose();
    oldLogProbabilities.dispose();
    advantages.dispose();
    returns.dispose();
    return {policyLoss: policyLossValue, valueLoss: valueLossValue, entropy: entropyValue};
}

function refreshOptimizers(trainer: PpoTrainer, learningRate: number): void {
    if (trainer.learningRate === learningRate) {
        return;
    }
    trainer.actorOptimizer.dispose();
    trainer.criticOptimizer.dispose();
    trainer.actorOptimizer = tf.train.adam(learningRate);
    trainer.criticOptimizer = tf.train.adam(learningRate);
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
