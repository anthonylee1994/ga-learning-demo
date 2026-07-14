import {expect, test} from "@playwright/test";
import {readFile} from "node:fs/promises";

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
    await expect(page).toHaveURL(/\/theory$/);
    await expect(page.getByRole("heading", {name: "用演化，搜尋一個夠好嘅決策腦"})).toBeVisible();
    await expect(page.getByText("初始化族群")).toBeVisible();
    await page.getByRole("slider", {name: "突變率"}).fill("0.6");
    const mutatedGene = page.locator(".gene.mutated").first();
    await expect(mutatedGene).toBeVisible();
    expect(await mutatedGene.evaluate(element => window.getComputedStyle(element).borderTopColor)).toBe("rgb(227, 111, 91)");
    await page.screenshot({fullPage: true, path: testInfo.outputPath("theory-desktop.png")});

    await page.getByRole("button", {name: "貪食蛇"}).click();
    await expect(page).toHaveURL(/\/snake$/);
    await page.getByRole("slider", {name: "播放速度"}).fill("5");
    await page.getByRole("button", {name: "開始"}).click();
    await expectGeneration(page);
    const snakeHasGreenPixels = await page.getByLabel("貪食蛇冠軍重播").evaluate(canvas => {
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
    await expect(page.getByText("暫停 · 最新冠軍玩到輸再重開")).toBeVisible();
    const snakeCanvas = page.getByLabel("貪食蛇冠軍重播");
    await expect(snakeCanvas).toHaveAttribute("data-loop", "true");
    await expectCompleteReplayLoop(snakeCanvas, /collision|starved|timeout/, page);

    await page.getByRole("button", {name: "打磚塊"}).click();
    await expect(page).toHaveURL(/\/breaker$/);
    await page.getByRole("slider", {name: "播放速度"}).fill("5");
    await page.getByRole("button", {name: "開始"}).click();
    await expectGeneration(page);
    await expect(page.getByLabel("打磚塊冠軍重播")).toBeVisible();
    await page.getByRole("button", {name: "暫停"}).click();
    await expect(page.getByText("暫停 · 最新冠軍玩到輸再重開")).toBeVisible();
    const breakerCanvas = page.getByLabel("打磚塊冠軍重播");
    await expect(breakerCanvas).toHaveAttribute("data-loop", "true");
    await expectCompleteReplayLoop(breakerCanvas, /lost|cleared|timeout/, page);

    await page.getByRole("button", {name: "股票交易"}).click();
    await expect(page).toHaveURL(/\/stock$/);
    await expect(page.locator(".market-zoom-brush")).toBeVisible({timeout: 30_000});
    await page.getByRole("textbox", {name: "股票代號"}).fill("AAPL");
    await page.getByRole("textbox", {name: "股票代號"}).press("Enter");
    await expect(page.getByText("AAPL · 日線")).toBeVisible({timeout: 30_000});
    await expect(page.locator(".market-zoom-brush")).toBeVisible();
    await page.getByRole("button", {name: "開始"}).click();
    await expectGeneration(page, 45_000);
    await expect(page.getByRole("heading", {name: "最佳指標參數"})).toBeVisible();
    const marketPanel = page.locator(".market-panel").filter({has: page.getByRole("heading", {name: "市場與交易訊號"})});
    await expect(page.getByText(/買 ≤ \d+ · 賣 ≥ \d+/).first()).toBeVisible();
    await expect(page.getByText(/買 ≤ -\d+ · 賣 ≥ -\d+/)).toBeVisible();
    await page.getByLabel("技術指標").selectOption("momentum");
    await expect(marketPanel.getByText(/RSI 買 \d+/)).toBeVisible();
    await expect(marketPanel.getByText(/RSI 賣 \d+/)).toBeVisible();
    await expect(marketPanel.getByText(/W%R 買 -\d+/)).toBeVisible();
    await expect(marketPanel.getByText(/W%R 賣 -\d+/)).toBeVisible();
    await page.screenshot({fullPage: true, path: testInfo.outputPath("stock-thresholds-desktop.png")});
    await page.getByLabel("技術指標").selectOption("risk");
    await expect(marketPanel.getByText("波動率", {exact: true})).toBeVisible();
    await expect(marketPanel.getByText("成交量", {exact: true})).toBeVisible();
    await expect(page.getByText("策略 vs 買入持有")).toBeVisible();
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", {name: "匯出 Pine Script"}).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe("aapl-evolab-strategy.pine");
    const downloadPath = await download.path();
    expect(downloadPath).not.toBeNull();
    const pineScript = await readFile(downloadPath!, "utf8");
    expect(pineScript).toContain("//@version=6");
    expect(pineScript).toContain("rsiPeriod =");
    expect(pineScript).toContain("rsiBuyThreshold =");
    expect(pineScript).toContain("rsiSellThreshold =");
    expect(pineScript).toContain("williamsBuyThreshold =");
    expect(pineScript).toContain("williamsSellThreshold =");
    expect(pineScript).toContain("bollingerPeriod =");
    expect(pineScript).toContain("h1_0 = tanh(");
    expect(pineScript).not.toContain("math.tanh");
    await page.getByRole("button", {name: "暫停"}).click();
    await page.screenshot({fullPage: true, path: testInfo.outputPath("stock-desktop.png")});

    expect(consoleErrors).toEqual([]);
});

test("routing supports direct links and browser history", async ({page}, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Desktop routing flow");
    await page.goto("/snake");
    await expect(page.getByRole("heading", {name: "貪食蛇 · 神經演化"})).toBeVisible();
    await expect(page.getByRole("button", {name: "貪食蛇"})).toHaveClass(/active/);

    await page.getByRole("button", {name: "演算法原理"}).click();
    await expect(page).toHaveURL(/\/theory$/);
    await page.goBack();
    await expect(page).toHaveURL(/\/snake$/);
    await expect(page.getByRole("heading", {name: "貪食蛇 · 神經演化"})).toBeVisible();
});

test("stock evolution tunes RSI and Williams thresholds", async ({page}, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "Desktop-only stock training flow");
    const consoleErrors: string[] = [];
    page.on("console", message => {
        if (message.type() === "error") {
            consoleErrors.push(message.text());
        }
    });
    page.on("pageerror", error => consoleErrors.push(error.message));

    await page.goto("/stock");
    await expect(page.locator(".market-zoom-brush")).toBeVisible({timeout: 30_000});
    await page.getByRole("button", {name: "開始"}).click();
    await expectGeneration(page, 45_000);
    await expect(page.getByRole("heading", {name: "最佳指標參數"})).toBeVisible();
    await expect(page.getByText(/買 ≤ \d+ · 賣 ≥ \d+/).first()).toBeVisible();
    await expect(page.getByText(/買 ≤ -\d+ · 賣 ≥ -\d+/)).toBeVisible();

    const marketPanel = page.locator(".market-panel").filter({has: page.getByRole("heading", {name: "市場與交易訊號"})});
    await page.getByLabel("技術指標").selectOption("momentum");
    await expect(marketPanel.getByText(/RSI 買 \d+/)).toBeVisible();
    await expect(marketPanel.getByText(/RSI 賣 \d+/)).toBeVisible();
    await expect(marketPanel.getByText(/W%R 買 -\d+/)).toBeVisible();
    await expect(marketPanel.getByText(/W%R 賣 -\d+/)).toBeVisible();
    await page.getByRole("button", {name: "暫停"}).click();
    await page.screenshot({fullPage: true, path: testInfo.outputPath("stock-thresholds-desktop.png")});

    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", {name: "匯出 Pine Script"}).click();
    const download = await downloadPromise;
    const downloadPath = await download.path();
    expect(downloadPath).not.toBeNull();
    const pineScript = await readFile(downloadPath!, "utf8");
    expect(pineScript).toContain("rsiBuyThreshold =");
    expect(pineScript).toContain("rsiSellThreshold =");
    expect(pineScript).toContain("williamsBuyThreshold =");
    expect(pineScript).toContain("williamsSellThreshold =");
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
    const value = page.locator(".metric").filter({hasText: "世代"}).locator("strong");
    await expect(value).not.toHaveText("0", {timeout});
}

async function expectCompleteReplayLoop(canvas: import("@playwright/test").Locator, terminal: RegExp, page: import("@playwright/test").Page): Promise<void> {
    await expect(canvas).toHaveAttribute("data-terminal", terminal, {timeout: 30_000});
    const terminalFrame = await canvas.getAttribute("data-frame-index");
    await page.waitForTimeout(400);
    await expect(canvas).toHaveAttribute("data-frame-index", terminalFrame ?? "");
    await expect.poll(async () => Number(await canvas.getAttribute("data-frame-index")), {timeout: 1_500}).toBeLessThan(Number(terminalFrame));
}
