import React from "react";
import {act, render, screen} from "@testing-library/react";
import {BreakerCanvas} from "./breaker-canvas";
import {SnakeCanvas} from "./snake-canvas";
import type {BreakerReplay, SnakeReplay} from "../lib/types";

const snakeReplay: SnakeReplay = {
    frames: [
        {snake: [{x: 1, y: 1}], food: {x: 3, y: 3}, score: 0, step: 0},
        {snake: [{x: 2, y: 1}], food: {x: 3, y: 3}, score: 0, step: 1, terminal: "collision"},
    ],
    score: 0,
    steps: 2,
};

const breakerReplay: BreakerReplay = {
    frames: [
        {paddleX: 200, ball: {x: 200, y: 200}, bricks: [], hits: 0, step: 0},
        {paddleX: 210, ball: {x: 205, y: 205}, bricks: [], hits: 0, step: 1, terminal: "lost"},
    ],
    bricksCleared: 0,
    hits: 0,
    steps: 2,
};

describe("champion replay playback", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        const context = new Proxy(
            {},
            {
                get: () => vi.fn(),
                set: () => true,
            }
        );
        vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as unknown as CanvasRenderingContext2D);
    });

    afterEach(() => {
        vi.restoreAllMocks();
        vi.useRealTimers();
    });

    it("restarts Snake from frame zero only after the champion loses", () => {
        render(
            <React.Fragment>
                <SnakeCanvas loop playing replay={snakeReplay} restartKey={1} speed={5} />
            </React.Fragment>
        );
        const canvas = screen.getByLabelText("Snake champion replay");
        expect(canvas).toHaveAttribute("data-frame-index", "0");

        act(() => vi.advanceTimersByTime(40));
        expect(canvas).toHaveAttribute("data-frame-index", "1");

        act(() => vi.advanceTimersByTime(899));
        expect(canvas).toHaveAttribute("data-frame-index", "1");

        act(() => vi.advanceTimersByTime(1));
        expect(canvas).toHaveAttribute("data-frame-index", "0");
    });

    it("restarts Block Breaker only after the final loss frame", () => {
        render(
            <React.Fragment>
                <BreakerCanvas loop playing replay={breakerReplay} restartKey={1} speed={5} />
            </React.Fragment>
        );
        const canvas = screen.getByLabelText("Block Breaker champion replay");
        expect(canvas).toHaveAttribute("data-frame-index", "0");

        act(() => vi.advanceTimersByTime(35));
        expect(canvas).toHaveAttribute("data-frame-index", "1");

        act(() => vi.advanceTimersByTime(899));
        expect(canvas).toHaveAttribute("data-frame-index", "1");

        act(() => vi.advanceTimersByTime(1));
        expect(canvas).toHaveAttribute("data-frame-index", "0");
    });
});
