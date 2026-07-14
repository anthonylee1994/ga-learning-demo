import React from "react";
import {render, screen} from "@testing-library/react";
import {calculateGeneCount} from "../lib/neuralNetwork";
import {SNAKE_INPUT_LABELS, SNAKE_OUTPUT_LABELS, SNAKE_TOPOLOGY} from "../domains/snake/simulation";
import {NetworkPanel} from "./NetworkPanel";

describe("NetworkPanel", () => {
    it("shows empty state without a champion genome", () => {
        render(
            <React.Fragment>
                <NetworkPanel inputLabels={SNAKE_INPUT_LABELS} outputLabels={SNAKE_OUTPUT_LABELS} topology={SNAKE_TOPOLOGY} />
            </React.Fragment>
        );
        expect(screen.getByText(/開始訓練後/)).toBeInTheDocument();
    });

    it("renders topology and decision when genome + input are provided", () => {
        const genome = Array.from({length: calculateGeneCount(SNAKE_TOPOLOGY)}, (_, index) => Math.sin(index) * 0.4);
        const input = [1, -1, -1, 0.2, -0.1, -1, 1, -1, -1, 0.01];
        render(
            <React.Fragment>
                <NetworkPanel genome={genome} input={input} inputLabels={SNAKE_INPUT_LABELS} outputLabels={SNAKE_OUTPUT_LABELS} title="Snake network" topology={SNAKE_TOPOLOGY} />
            </React.Fragment>
        );
        expect(screen.getByLabelText("Neural network topology")).toBeInTheDocument();
        expect(screen.getByText(/決策：/)).toBeInTheDocument();
        expect(screen.getByText("Output weights")).toBeInTheDocument();
        expect(screen.getByText("Input activations")).toBeInTheDocument();
    });
});
