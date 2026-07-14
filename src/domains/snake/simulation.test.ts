import {calculateGeneCount, forwardWithActivations} from "../../lib/neuralNetwork";
import {buildSnakeInputFromFrame, createSnakeReplay, evaluateSnakeGenome, SNAKE_TOPOLOGY} from "./simulation";

describe("snake simulation", () => {
    const genome = Array(calculateGeneCount(SNAKE_TOPOLOGY)).fill(0);

    it("produces a finite deterministic fitness", () => {
        expect(evaluateSnakeGenome(genome)).toBe(evaluateSnakeGenome(genome));
        expect(Number.isFinite(evaluateSnakeGenome(genome))).toBe(true);
    });

    it("records champion frames inside the board", () => {
        const replay = createSnakeReplay(genome);
        expect(replay.frames.length).toBeGreaterThan(0);
        expect(replay.frames.length).toBeLessThanOrEqual(901);
        expect(replay.steps).toBeLessThanOrEqual(2_400);
        expect(replay.frames[0].snake.every(point => point.x >= 0 && point.x < 20)).toBe(true);
        expect(replay.frames.at(-1)?.terminal).toMatch(/collision|starved|timeout/);
    });

    it("rebuilds a valid network input from a recorded frame", () => {
        const replay = createSnakeReplay(genome);
        const frame = replay.frames[0];
        const input = buildSnakeInputFromFrame(frame);
        expect(input).toHaveLength(SNAKE_TOPOLOGY.inputSize);
        expect(input.every(value => Number.isFinite(value))).toBe(true);
        const pass = forwardWithActivations(genome, SNAKE_TOPOLOGY, input);
        expect(pass.outputs).toHaveLength(SNAKE_TOPOLOGY.outputSize);
        expect(pass.decision).toBeGreaterThanOrEqual(0);
        expect(pass.decision).toBeLessThan(SNAKE_TOPOLOGY.outputSize);
    });
});
