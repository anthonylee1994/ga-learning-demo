import {expect, test} from "@playwright/test";

test.describe.configure({mode: "serial"});
test.setTimeout(120_000);

test("desktop workspace runs all three evolution demos", async ({page}, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Desktop-only full simulation flow");
    const consoleErrors: string[] = [];
    page.on("console", message => {
        if (message.type() === "error") {
            consoleErrors.push(message.text());
        }
    });
    page.on("pageerror", error => consoleErrors.push(error.message));

    await page.goto("/");
    await expect(page.getByRole("heading", {name: "用演化，搜尋一個夠好嘅決策腦"})).toBeVisible();
    await expect(page.getByText("初始化 Population")).toBeVisible();
    await page.screenshot({fullPage: true, path: testInfo.outputPath("theory-desktop.png")});

    await page.getByRole("button", {name: "Snake Game"}).click();
    await page.getByRole("slider", {name: "播放速度"}).fill("5");
    await page.getByRole("button", {name: "開始"}).click();
    await expectGeneration(page);
    const snakeHasGreenPixels = await page.getByLabel("Snake champion replay").evaluate(canvas => {
        const context = (canvas as HTMLCanvasElement).getContext("2d");
        const pixels = context?.getImageData(0, 0, (canvas as HTMLCanvasElement).width, (canvas as HTMLCanvasElement).height).data;
        if (!pixels) return false;
        for (let index = 0; index < pixels.length; index += 16) {
            if (pixels[index + 1] > pixels[index] * 1.25 && pixels[index + 1] > pixels[index + 2] * 1.15) return true;
        }
        return false;
    });
    expect(snakeHasGreenPixels).toBe(true);
    await page.getByRole("button", {name: "暫停"}).click();
    await expect(page.getByText("暫停 · 最新 champion 玩到輸再重開")).toBeVisible();
    const snakeCanvas = page.getByLabel("Snake champion replay");
    await expect(snakeCanvas).toHaveAttribute("data-loop", "true");
    await expectCompleteReplayLoop(snakeCanvas, /collision|starved|timeout/, page);

    await page.getByRole("button", {name: "Block Breaker"}).click();
    await page.getByRole("slider", {name: "播放速度"}).fill("5");
    await page.getByRole("button", {name: "開始"}).click();
    await expectGeneration(page);
    await expect(page.getByLabel("Block Breaker champion replay")).toBeVisible();
    await page.getByRole("button", {name: "暫停"}).click();
    await expect(page.getByText("暫停 · 最新 champion 玩到輸再重開")).toBeVisible();
    const breakerCanvas = page.getByLabel("Block Breaker champion replay");
    await expect(breakerCanvas).toHaveAttribute("data-loop", "true");
    await expectCompleteReplayLoop(breakerCanvas, /lost|cleared|timeout/, page);

    await page.getByRole("button", {name: "Stock Trading"}).click();
    await expect(page.getByText(/sessions · USD/)).toBeVisible({timeout: 30_000});
    await page.getByRole("button", {name: "開始"}).click();
    await expectGeneration(page, 45_000);
    await expect(page.getByText("Strategy vs Buy & Hold")).toBeVisible();
    await page.getByRole("button", {name: "暫停"}).click();
    await page.screenshot({fullPage: true, path: testInfo.outputPath("stock-desktop.png")});

    expect(consoleErrors).toEqual([]);
});

test("mobile workspace keeps navigation and theory readable", async ({page}, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "Mobile-only responsive flow");
    await page.setViewportSize({width: 390, height: 844});
    await page.goto("/");
    await expect(page.getByRole("navigation", {name: "手機實驗主題"})).toBeVisible();
    await expect(page.getByRole("heading", {name: "用演化，搜尋一個夠好嘅決策腦"})).toBeVisible();
    await page.screenshot({fullPage: true, path: testInfo.outputPath("theory-mobile.png")});
});

async function expectGeneration(page: import("@playwright/test").Page, timeout = 30_000): Promise<void> {
    const value = page.locator(".metric").filter({hasText: "Generation"}).locator("strong");
    await expect(value).not.toHaveText("0", {timeout});
}

async function expectCompleteReplayLoop(canvas: import("@playwright/test").Locator, terminal: RegExp, page: import("@playwright/test").Page): Promise<void> {
    await expect(canvas).toHaveAttribute("data-terminal", terminal, {timeout: 30_000});
    const terminalFrame = await canvas.getAttribute("data-frame-index");
    await page.waitForTimeout(400);
    await expect(canvas).toHaveAttribute("data-frame-index", terminalFrame ?? "");
    await expect.poll(async () => Number(await canvas.getAttribute("data-frame-index")), {timeout: 1_500}).toBeLessThan(Number(terminalFrame));
}
