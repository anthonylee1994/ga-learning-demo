import React from "react";
import {fireEvent, render, screen} from "@testing-library/react";
import {TheoryLab} from "./TheoryLab";

describe("TheoryLab", () => {
    it("shows the full GA loop and updates the mutation visualizer", () => {
        const {container} = render(
            <React.Fragment>
                <TheoryLab />
            </React.Fragment>
        );
        expect(screen.getByText("初始化 Population")).toBeInTheDocument();
        expect(screen.getByText("下一代")).toBeInTheDocument();
        expect(screen.getByText("Neuroevolution")).toBeInTheDocument();
        fireEvent.change(screen.getByLabelText("Theory mutation rate"), {target: {value: "0.6"}});
        expect(screen.getByText("60%")).toBeInTheDocument();
        expect(container.querySelectorAll(".gene.mutated").length).toBeGreaterThan(0);
    });
});
