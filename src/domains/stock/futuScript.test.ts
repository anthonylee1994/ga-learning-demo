import {createFutuPythonScript} from "./futuScript";
import {createStockSeedGenomes, decodeStockGenome, STOCK_GENE_COUNT, STOCK_TOPOLOGY} from "./strategyGenome";

describe("Futu Python export", () => {
    it("embeds optimized parameters, network weights, and state-machine entries", () => {
        const genome = Array.from({length: STOCK_GENE_COUNT}, (_, index) => Math.sin(index) * 0.4);
        const {parameters} = decodeStockGenome(genome);
        const script = createFutuPythonScript(genome);

        expect(script).toContain('indicator("GA", "GA", True)');
        expect(script).toContain(`SMA_FAST = ${parameters.smaFastPeriod}`);
        expect(script).toContain(`SMA_SLOW = ${parameters.smaSlowPeriod}`);
        expect(script).toContain(`RSI_P = ${parameters.rsiPeriod}`);
        expect(script).toContain(`RSI_BUY = ${parameters.rsiBuyThreshold}`);
        expect(script).toContain(`RSI_SELL = ${parameters.rsiSellThreshold}`);
        expect(script).toContain(`WILL_P = ${parameters.williamsPeriod}`);
        expect(script).toContain(`BB_P = ${parameters.bollingerPeriod}`);
        expect(script).toContain(`NEW_H_P = ${parameters.newHighPeriod}`);
        expect(script).toContain(`NEW_L_P = ${parameters.newLowPeriod}`);
        expect(script).toContain("H1 = [");
        expect(script).toContain("H2 = [");
        expect(script).toContain("OUT = [");
        expect(script).toContain("def _dense(inputs, row):");
        expect(script).toContain("feats.append(f17)");
        expect(script).toContain("ACTION_MARGIN = 0.08");
        expect(script).toContain("buy_signal = out_buy >= stay + ACTION_MARGIN");
        expect(script).toContain("sell_signal = out_sell >= stay + ACTION_MARGIN");
        expect(script).not.toContain("MIN_BARS_LONG");
        expect(script).not.toContain("bars_in_state");
        expect(script).toContain("if buy_signal and position <= 0:");
        expect(script).toContain("elif sell_signal and position > 0:");
        expect(script).toContain('plot("SMA Fast", sma_f_seq, Color.yellow)');
        expect(script).toContain('plot("SMA Slow", sma_s_seq, Color.blue)');
        expect(script).toContain('plot("BB Upper", bb_up_seq, Color.gray)');
        expect(script).toContain('plot("N-day High", ndh_seq, Color.limagenta)');
        expect(script).toContain('plot("N-day Low", ndl_seq, Color.cyan)');
        expect(script).toContain("Shape.labelup");
        expect(script).toContain("Shape.labeldown");
        expect(script).toContain("Color.white");
        expect(script).toContain("plot_icon");
        expect(script).toContain("output_parameter");
        // IndicatorParser rejects generator expressions inside calls
        expect(script).not.toContain("all(x == x for");
        expect(script).not.toContain(" for x in window");
        expect(STOCK_TOPOLOGY.hiddenLayers).toEqual([10, 5]);
    });

    it("exports rule mode without neural network weights", () => {
        const genome = createStockSeedGenomes()[0];
        const script = createFutuPythonScript(genome, false);
        expect(script).toContain("H1 = []");
        expect(script).toContain("buy_votes = trend_vote + macd_vote");
        expect(script).toContain("buy_signal = buy_votes >= 2");
        expect(script).not.toContain("feats.append(f17)");
        expect(script).not.toContain("out_buy = _dense");
    });
});
