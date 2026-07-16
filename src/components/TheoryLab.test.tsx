import React from "react";
import {fireEvent, render, screen} from "@testing-library/react";
import {MemoryRouter} from "react-router-dom";
import {TheoryLab} from "./TheoryLab";

function renderTheory() {
    return render(
        <MemoryRouter>
            <React.Fragment>
                <TheoryLab />
            </React.Fragment>
        </MemoryRouter>
    );
}

describe("TheoryLab", () => {
    it("shows the full GA loop and updates the mutation visualizer", () => {
        const {container} = renderTheory();
        expect(screen.getByRole("tab", {name: /初始化族群/})).toBeInTheDocument();
        expect(screen.getByRole("tab", {name: /下一代/})).toBeInTheDocument();
        expect(screen.getByText("神經演化")).toBeInTheDocument();
        fireEvent.change(screen.getByLabelText("突變率"), {target: {value: "0.6"}});
        expect(screen.getByText("60%")).toBeInTheDocument();
        expect(container.querySelectorAll(".gene.mutated").length).toBeGreaterThan(0);
    });

    it("explains project pipeline and demo mappings", () => {
        renderTheory();
        expect(screen.getByText("代碼點樣串起成條鏈")).toBeInTheDocument();
        expect(screen.getByText("src/lib/ga.ts")).toBeInTheDocument();
        expect(screen.getByRole("link", {name: /貪食蛇/})).toHaveAttribute("href", "/snake");
        expect(screen.getByRole("link", {name: /撞磚/})).toHaveAttribute("href", "/breaker");
        expect(screen.getByRole("link", {name: /Flappy Bird/})).toHaveAttribute("href", "/flappy");
        expect(screen.getByRole("link", {name: /股票 \(GA\)/})).toHaveAttribute("href", "/stock");
    });

    it("lets users inspect each generation step", () => {
        renderTheory();
        fireEvent.click(screen.getByRole("tab", {name: /選擇/}));
        expect(screen.getByText("錦標賽抽家長")).toBeInTheDocument();
        expect(screen.getByText("抽 3 個比併做家長")).toBeInTheDocument();
        expect(screen.getByText(/tournamentSelect/)).toBeInTheDocument();
    });

    it("only fully selects 下一代 panel, not 族群, when step 7 is active", () => {
        const {container} = renderTheory();
        fireEvent.click(screen.getByRole("tab", {name: /下一代/}));
        const panels = container.querySelectorAll(".gen-panel");
        expect(panels[0]).not.toHaveClass("is-active");
        expect(panels[0]).toHaveClass("is-loop-target");
        expect(panels[6]).toHaveClass("is-active");
        expect(panels[6]).not.toHaveClass("is-loop-target");
        expect(screen.getByText("循環回到呢度")).toBeInTheDocument();
    });

    it("syncs mutation count with the slider", () => {
        const {container} = renderTheory();
        fireEvent.change(screen.getByLabelText("突變率"), {target: {value: "0.6"}});
        expect(screen.getByText("60%")).toBeInTheDocument();
        const stat = container.querySelector(".genome-lab-stat");
        expect(stat?.textContent).toMatch(/已突變\s*[1-8]/);
        expect(container.querySelectorAll(".gene.mutated").length).toBeGreaterThan(0);
    });
});
