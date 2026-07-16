import {calculateGeneCount, forwardWithActivations} from "../../lib/neuralNetwork";
import {buildFlappyInputFromFrame, createFlappyReplay, evaluateFlappyGenome, FLAPPY_HEIGHT, FLAPPY_TOPOLOGY, FLAPPY_WIDTH} from "./simulation";

describe("flappy simulation", () => {
    const genome = Array(calculateGeneCount(FLAPPY_TOPOLOGY)).fill(0);

    it("produces a finite deterministic fitness", () => {
        expect(evaluateFlappyGenome(genome)).toBe(evaluateFlappyGenome(genome));
        expect(Number.isFinite(evaluateFlappyGenome(genome))).toBe(true);
    });

    it("records champion frames inside the playfield", () => {
        const replay = createFlappyReplay(genome);
        expect(replay.frames.length).toBeGreaterThan(0);
        expect(replay.steps).toBeLessThanOrEqual(8_000);
        for (const frame of replay.frames) {
            expect(frame.birdY).toBeGreaterThanOrEqual(-20);
            expect(frame.birdY).toBeLessThanOrEqual(FLAPPY_HEIGHT + 20);
            for (const pipe of frame.pipes) {
                expect(pipe.gapY).toBeGreaterThan(0);
                expect(pipe.gapY).toBeLessThan(FLAPPY_HEIGHT);
                // 下管唔好太短：縫隙下沿要留夠距離到畫面底
                expect(pipe.gapY + pipe.gapHeight / 2).toBeLessThanOrEqual(FLAPPY_HEIGHT - 160);
            }
        }
        expect(replay.frames.at(-1)?.terminal).toMatch(/crash|timeout/);
    });

    it("rebuilds a valid network input from a recorded frame", () => {
        const replay = createFlappyReplay(genome);
        const frame = replay.frames[0];
        const input = buildFlappyInputFromFrame(frame);
        expect(input).toHaveLength(FLAPPY_TOPOLOGY.inputSize);
        expect(input.every(value => Number.isFinite(value))).toBe(true);
        const pass = forwardWithActivations(genome, FLAPPY_TOPOLOGY, input);
        expect(pass.outputs).toHaveLength(FLAPPY_TOPOLOGY.outputSize);
        expect(pass.decision).toBeGreaterThanOrEqual(0);
        expect(pass.decision).toBeLessThan(FLAPPY_TOPOLOGY.outputSize);
    });

    it("exposes consistent field dimensions", () => {
        expect(FLAPPY_WIDTH).toBeGreaterThan(200);
        expect(FLAPPY_HEIGHT).toBeGreaterThan(300);
    });
});
