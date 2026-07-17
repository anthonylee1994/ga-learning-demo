import React from "react";
import {renderHook} from "@testing-library/react";
import {useEvolutionDemo} from "./useEvolutionDemo";
import type {GAConfig} from "../lib/types";

const DEFAULT_CONFIG: GAConfig = {
    populationSize: 12,
    mutationRate: 0.1,
    mutationScale: 0.2,
    eliteRate: 0.1,
    seed: 1,
    speed: 3,
};

function createMockWorker() {
    return {
        onmessage: null,
        postMessage: vi.fn(),
        terminate: vi.fn(),
    } as unknown as Worker;
}

describe("useEvolutionDemo", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        localStorage.clear();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("clears only the current page data when reset and does not persist defaults again", () => {
        const storedConfig = {...DEFAULT_CONFIG, populationSize: 48};
        localStorage.setItem(
            "evolab-state-v1",
            JSON.stringify({
                version: 1,
                demos: {
                    snake: {config: storedConfig, champion: [1, 2, 3], bestFitness: 10},
                    breaker: {config: DEFAULT_CONFIG, champion: [4, 5, 6], bestFitness: 20},
                },
            })
        );
        localStorage.setItem("other-state", "value");

        const worker = createMockWorker();
        const {result} = renderHook(() =>
            useEvolutionDemo<undefined, undefined>({
                topic: "snake",
                createWorker: () => worker,
                defaultConfig: DEFAULT_CONFIG,
            })
        );

        expect(result.current.config).toEqual(storedConfig);

        React.act(() => result.current.reset());

        expect(result.current.config).toBe(DEFAULT_CONFIG);
        expect(JSON.parse(localStorage.getItem("evolab-state-v1") ?? "null")).toEqual({
            version: 1,
            demos: {
                breaker: {config: DEFAULT_CONFIG, champion: [4, 5, 6], bestFitness: 20},
            },
        });
        expect(localStorage.getItem("other-state")).toBe("value");
        expect(worker.postMessage).toHaveBeenCalledWith({type: "reset"});

        React.act(() => vi.advanceTimersByTime(1_000));

        const storedState = JSON.parse(localStorage.getItem("evolab-state-v1") ?? "null");
        expect(storedState.demos.snake).toBeUndefined();
        expect(storedState.demos.breaker).toBeDefined();
        expect(localStorage.getItem("other-state")).toBe("value");
    });

    it("restores champion weights from localStorage after refresh", () => {
        const genome = [0.1, 0.2, 0.3];
        localStorage.setItem(
            "evolab-state-v1",
            JSON.stringify({
                version: 1,
                demos: {
                    snake: {config: DEFAULT_CONFIG, champion: genome, bestFitness: 42},
                },
            })
        );

        const worker = createMockWorker();
        const restoreChampion = vi.fn((restoredGenome: number[]) => ({
            replay: {frames: [], score: 1, steps: 2},
            fitness: restoredGenome.length,
        }));

        const {result} = renderHook(() =>
            useEvolutionDemo<undefined, {frames: unknown[]; score: number; steps: number}>({
                topic: "snake",
                createWorker: () => worker,
                defaultConfig: DEFAULT_CONFIG,
                restoreChampion,
            })
        );

        expect(restoreChampion).toHaveBeenCalledWith(genome, undefined, DEFAULT_CONFIG);
        expect(result.current.champion).toEqual({
            genome,
            fitness: 3,
            replay: {frames: [], score: 1, steps: 2},
        });
        expect(result.current.status).toBe("paused");
    });

    it("restores stock champion only after market data is available", () => {
        const genome = [1, 2, 3, 4];
        localStorage.setItem(
            "evolab-state-v1",
            JSON.stringify({
                version: 1,
                demos: {
                    stock: {
                        config: {...DEFAULT_CONFIG, useNeuralNetwork: true},
                        champion: genome,
                        bestFitness: 7,
                    },
                },
            })
        );

        const worker = createMockWorker();
        const restoreChampion = vi.fn((restoredGenome: number[], data: number[] | undefined) => {
            if (!data?.length) {
                return null;
            }
            return {
                replay: {points: data, trades: []},
                fitness: restoredGenome[0],
            };
        });

        const {result, rerender} = renderHook(
            ({data}: {data?: number[]}) =>
                useEvolutionDemo<number[], {points: number[]; trades: unknown[]}>({
                    topic: "stock",
                    createWorker: () => worker,
                    defaultConfig: DEFAULT_CONFIG,
                    data,
                    restoreChampion,
                }),
            {initialProps: {data: undefined as number[] | undefined}}
        );

        expect(result.current.champion).toBeNull();
        expect(restoreChampion).toHaveBeenCalled();

        React.act(() => {
            rerender({data: [10, 20, 30]});
        });

        expect(result.current.champion?.genome).toEqual(genome);
        expect(result.current.champion?.fitness).toBe(1);
        expect(result.current.champion?.replay).toEqual({points: [10, 20, 30], trades: []});
    });

    it("clears restored champion when user clicks reset", () => {
        const genome = [9, 8, 7];
        localStorage.setItem(
            "evolab-state-v1",
            JSON.stringify({
                version: 1,
                demos: {
                    snake: {config: DEFAULT_CONFIG, champion: genome, bestFitness: 5},
                },
            })
        );

        const worker = createMockWorker();
        const {result} = renderHook(() =>
            useEvolutionDemo<undefined, {ok: boolean}>({
                topic: "snake",
                createWorker: () => worker,
                defaultConfig: DEFAULT_CONFIG,
                restoreChampion: () => ({replay: {ok: true}, fitness: 5}),
            })
        );

        expect(result.current.champion).not.toBeNull();

        React.act(() => result.current.reset());

        expect(result.current.champion).toBeNull();
        expect(result.current.status).toBe("idle");
        expect(JSON.parse(localStorage.getItem("evolab-state-v1") ?? "null")?.demos?.snake).toBeUndefined();
    });

    it("ignores late worker champion events after a single reset", () => {
        const worker = createMockWorker() as Worker & {
            onmessage: ((message: MessageEvent) => void) | null;
            postMessage: ReturnType<typeof vi.fn>;
        };

        const {result} = renderHook(() =>
            useEvolutionDemo<undefined, {frames: number[]}>({
                topic: "snake",
                createWorker: () => worker,
                defaultConfig: DEFAULT_CONFIG,
            })
        );

        // Simulate a trained champion, then reset — a queued generation must not revive weights.
        React.act(() => {
            worker.onmessage?.({
                data: {
                    type: "generation",
                    stats: {generation: 3, bestFitness: 12, averageFitness: 4, diversity: 0.5},
                    champion: {genome: [1, 2, 3], fitness: 12, replay: {frames: [1]}},
                },
            } as MessageEvent);
        });

        expect(result.current.champion?.genome).toEqual([1, 2, 3]);

        React.act(() => result.current.reset());
        expect(result.current.champion).toBeNull();

        React.act(() => {
            worker.onmessage?.({
                data: {
                    type: "generation",
                    stats: {generation: 4, bestFitness: 99, averageFitness: 9, diversity: 0.1},
                    champion: {genome: [9, 9, 9], fitness: 99, replay: {frames: [9]}},
                    reason: "pause-showcase",
                },
            } as MessageEvent);
        });

        expect(result.current.champion).toBeNull();
        expect(JSON.parse(localStorage.getItem("evolab-state-v1") ?? "null")?.demos?.snake).toBeUndefined();

        React.act(() => vi.advanceTimersByTime(1_000));
        expect(JSON.parse(localStorage.getItem("evolab-state-v1") ?? "null")?.demos?.snake).toBeUndefined();
    });
});
