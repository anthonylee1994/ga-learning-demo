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

        const worker = {
            onmessage: null,
            postMessage: vi.fn(),
            terminate: vi.fn(),
        } as unknown as Worker;
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
});
