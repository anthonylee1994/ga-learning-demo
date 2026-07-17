import type {Genome} from "../../lib/types";
import {STOCK_ACTION_MARGIN} from "./simulation";
import {decodeLayers, formatNumber as formatNum} from "./pineScript";
import {decodeStockGenome} from "./strategyGenome";

/**
 * Futu Python indicator export (IndicatorParser-compatible).
 * - No generator expressions / list comprehensions with `for` inside calls
 * - True state machine + f13 position feedback (long/flat; matches Pine next-bar open fills)
 * - Plots SMA / BB / nDayHigh / nDayLow + buy/sell icons
 */
export function createFutuPythonScript(genome: Genome, useNetwork = true): string {
    const {parameters, networkGenome} = decodeStockGenome(genome);
    const weightBlock = useNetwork ? emitNetworkWeights(networkGenome) : "H1 = []\nH2 = []\nOUT = []\n";
    const decisionBlock = useNetwork ? emitNetworkDecisionLoop() : emitRuleDecisionLoop();

    return `import math

indicator("GA", "GA", True)

SMA_FAST = ${parameters.smaFastPeriod}
SMA_SLOW = ${parameters.smaSlowPeriod}
WILL_P = ${parameters.williamsPeriod}
WILL_BUY = ${formatNum(parameters.williamsBuyThreshold)}
WILL_SELL = ${formatNum(parameters.williamsSellThreshold)}
ROC_P = ${parameters.rocPeriod}
RSI_P = ${parameters.rsiPeriod}
RSI_BUY = ${formatNum(parameters.rsiBuyThreshold)}
RSI_SELL = ${formatNum(parameters.rsiSellThreshold)}
MACD_F = ${parameters.macdFastPeriod}
MACD_S = ${parameters.macdSlowPeriod}
MACD_SIG = ${parameters.macdSignalPeriod}
BB_P = ${parameters.bollingerPeriod}
BB_M = ${formatNum(parameters.bollingerMultiplier)}
VLT_P = ${parameters.volatilityPeriod}
VZ_P = ${parameters.volumeZScorePeriod}
NEW_H_P = ${parameters.newHighPeriod}
NEW_L_P = ${parameters.newLowPeriod}

${weightBlock}
NAN = float("nan")


def _is_nan(x):
    if x is None:
        return True
    try:
        return math.isnan(x)
    except Exception:
        return x != x


def _valid(x):
    return not _is_nan(x)


def _window_ok(arr, start, end):
    i = start
    while i < end:
        if _is_nan(arr[i]):
            return False
        i = i + 1
    return True


def _window_sum(arr, start, end):
    s = 0.0
    i = start
    while i < end:
        s = s + arr[i]
        i = i + 1
    return s


def _to_list(series):
    if series is None:
        return []
    out = []
    try:
        data = list(series)
    except Exception:
        return [float(series)]
    for x in data:
        if x is None:
            out.append(NAN)
        else:
            out.append(float(x))
    return out


def _sma(arr, n):
    out = []
    i = 0
    while i < len(arr):
        out.append(NAN)
        i = i + 1
    if n <= 0:
        return out
    i = n - 1
    while i < len(arr):
        start = i - n + 1
        if _window_ok(arr, start, i + 1):
            out[i] = _window_sum(arr, start, i + 1) / float(n)
        i = i + 1
    return out


def _ema(arr, n):
    out = []
    i = 0
    while i < len(arr):
        out.append(NAN)
        i = i + 1
    if n <= 0 or len(arr) == 0:
        return out
    k = 2.0 / (n + 1.0)
    seed = _sma(arr, n)
    started = False
    prev = NAN
    i = 0
    while i < len(arr):
        if (not started) and _valid(seed[i]):
            out[i] = seed[i]
            prev = seed[i]
            started = True
        elif started and _valid(arr[i]):
            prev = arr[i] * k + prev * (1.0 - k)
            out[i] = prev
        i = i + 1
    return out


def _stdev(arr, n):
    out = []
    i = 0
    while i < len(arr):
        out.append(NAN)
        i = i + 1
    if n <= 1:
        return out
    i = n - 1
    while i < len(arr):
        start = i - n + 1
        if _window_ok(arr, start, i + 1):
            m = _window_sum(arr, start, i + 1) / float(n)
            var = 0.0
            j = start
            while j <= i:
                d = arr[j] - m
                var = var + d * d
                j = j + 1
            out[i] = math.sqrt(var / float(n))
        i = i + 1
    return out


def _hhv(arr, n):
    out = []
    i = 0
    while i < len(arr):
        out.append(NAN)
        i = i + 1
    i = n - 1
    while i < len(arr):
        start = i - n + 1
        if _window_ok(arr, start, i + 1):
            m = arr[start]
            j = start + 1
            while j <= i:
                if arr[j] > m:
                    m = arr[j]
                j = j + 1
            out[i] = m
        i = i + 1
    return out


def _llv(arr, n):
    out = []
    i = 0
    while i < len(arr):
        out.append(NAN)
        i = i + 1
    i = n - 1
    while i < len(arr):
        start = i - n + 1
        if _window_ok(arr, start, i + 1):
            m = arr[start]
            j = start + 1
            while j <= i:
                if arr[j] < m:
                    m = arr[j]
                j = j + 1
            out[i] = m
        i = i + 1
    return out


def _clamp(x):
    if _is_nan(x):
        return 0.0
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


def _tanh(x):
    if x > 20.0:
        x = 20.0
    if x < -20.0:
        x = -20.0
    e = math.exp(2.0 * x)
    return (e - 1.0) / (e + 1.0)


def _dense(inputs, row):
    z = row[0]
    j = 0
    while j < len(inputs):
        z = z + row[j + 1] * inputs[j]
        j = j + 1
    return _tanh(z)


def _max2(a, b):
    if a > b:
        return a
    return b


def compute_signals():
    o = _to_list(open())
    h = _to_list(high())
    l = _to_list(low())
    c = _to_list(close())
    v = _to_list(vol())
    n = len(c)
    empty = []
    if n == 0:
        return empty, empty, empty, empty, empty, empty, empty, empty

    sma_f = _sma(c, SMA_FAST)
    sma_s = _sma(c, SMA_SLOW)
    wh = _hhv(h, WILL_P)
    wl = _llv(l, WILL_P)
    macd_fast = _ema(c, MACD_F)
    macd_slow = _ema(c, MACD_S)

    macd_line = []
    i = 0
    while i < n:
        if _valid(macd_fast[i]) and _valid(macd_slow[i]):
            macd_line.append(macd_fast[i] - macd_slow[i])
        else:
            macd_line.append(NAN)
        i = i + 1

    macd_sig = _ema(macd_line, MACD_SIG)
    bb_basis = _sma(c, BB_P)
    bb_std = _stdev(c, BB_P)

    bb_up = []
    bb_lo = []
    i = 0
    while i < n:
        if _valid(bb_basis[i]) and _valid(bb_std[i]):
            bb_up.append(bb_basis[i] + bb_std[i] * BB_M)
            bb_lo.append(bb_basis[i] - bb_std[i] * BB_M)
        else:
            bb_up.append(NAN)
            bb_lo.append(NAN)
        i = i + 1

    vol_ma = _sma(v, VZ_P)
    vol_sd = _stdev(v, VZ_P)
    ndh = _hhv(h, NEW_H_P)
    ndl = _llv(l, NEW_L_P)

    roc = []
    i = 0
    while i < n:
        roc.append(NAN)
        i = i + 1
    i = ROC_P
    while i < n:
        if _valid(c[i]) and _valid(c[i - ROC_P]) and c[i - ROC_P] != 0:
            roc[i] = c[i] / c[i - ROC_P] - 1.0
        i = i + 1

    gain = []
    loss = []
    i = 0
    while i < n:
        gain.append(0.0)
        loss.append(0.0)
        i = i + 1
    i = 1
    while i < n:
        if _valid(c[i]) and _valid(c[i - 1]):
            d = c[i] - c[i - 1]
            if d > 0:
                gain[i] = d
            else:
                loss[i] = -d
        i = i + 1

    avg_gain = _sma(gain, RSI_P)
    avg_loss = _sma(loss, RSI_P)
    rsi = []
    i = 0
    while i < n:
        rsi.append(NAN)
        i = i + 1
    i = 0
    while i < n:
        ag = avg_gain[i]
        al = avg_loss[i]
        if _valid(ag) and _valid(al):
            if al < 1e-9:
                rsi[i] = 100.0
            else:
                rsi[i] = 100.0 - 100.0 / (1.0 + ag / al)
        i = i + 1

    dret = []
    i = 0
    while i < n:
        dret.append(NAN)
        i = i + 1
    i = 1
    while i < n:
        if _valid(c[i]) and _valid(c[i - 1]) and c[i - 1] != 0:
            dret[i] = c[i] / c[i - 1] - 1.0
        i = i + 1

    vlt = _stdev(dret, VLT_P)
    i = 0
    while i < n:
        if _valid(vlt[i]):
            vlt[i] = vlt[i] * math.sqrt(252.0)
        i = i + 1

    enter = []
    exit_ = []
    i = 0
    while i < n:
        enter.append(0)
        exit_.append(0)
        i = i + 1

    position = 0
    ACTION_MARGIN = ${STOCK_ACTION_MARGIN}
    warm = SMA_SLOW
    if WILL_P > warm:
        warm = WILL_P
    if ROC_P > warm:
        warm = ROC_P
    if RSI_P > warm:
        warm = RSI_P
    if MACD_S + MACD_SIG > warm:
        warm = MACD_S + MACD_SIG
    if BB_P > warm:
        warm = BB_P
    if VLT_P > warm:
        warm = VLT_P
    if VZ_P > warm:
        warm = VZ_P
    if NEW_H_P > warm:
        warm = NEW_H_P
    if NEW_L_P > warm:
        warm = NEW_L_P

    i = 0
    while i < n:
        ready = i >= warm
        if ready:
            if not _valid(sma_s[i]):
                ready = False
            if not _valid(wh[i]):
                ready = False
            if not _valid(wl[i]):
                ready = False
            if not _valid(roc[i]):
                ready = False
            if not _valid(rsi[i]):
                ready = False
            if not _valid(macd_sig[i]):
                ready = False
            if not _valid(bb_up[i]):
                ready = False
            if not _valid(bb_lo[i]):
                ready = False
            if not _valid(vlt[i]):
                ready = False
            if not _valid(vol_sd[i]):
                ready = False
            if not _valid(ndh[i]):
                ready = False
            if not _valid(ndl[i]):
                ready = False
            if not _valid(c[i]):
                ready = False
            if not _valid(o[i]):
                ready = False
            if not _valid(h[i]):
                ready = False
            if not _valid(l[i]):
                ready = False
            if not _valid(v[i]):
                ready = False

        if position > 0:
            f13 = 1.0
        else:
            f13 = 0.0

        if not ready:
            i = i + 1
            continue

${decisionBlock}

        # Long / flat only (matches decidePositionFromNetwork / Rules); sell = close to cash.
        if buy_signal and sell_signal and position == 0:
            sell_signal = False
        if buy_signal and position < 1:
            enter[i] = 1
            position = 1
        elif sell_signal and position > 0:
            exit_[i] = 1
            position = 0

        i = i + 1

    return enter, exit_, sma_f, sma_s, bb_up, bb_lo, ndh, ndl


def _nan_to_plot(values):
    out = []
    i = 0
    while i < len(values):
        if _is_nan(values[i]):
            out.append(NAN)
        else:
            out.append(float(values[i]))
        i = i + 1
    return out


def _to_float_seq(values):
    names = [
        "create_series",
        "series",
        "Series",
        "to_series",
        "make_series",
        "array",
        "Array",
    ]
    for name in names:
        if name in globals():
            fn = globals()[name]
            if callable(fn):
                try:
                    return fn(values)
                except Exception:
                    pass

    s = close() * 0.0
    i = 0
    n = len(values)
    while i < n:
        try:
            s[i] = float(values[i])
        except Exception:
            pass
        i = i + 1
    return s


if __name__ == "__main__":
    enter, exit_, sma_f, sma_s, bb_up, bb_lo, ndh, ndl = compute_signals()

    enter_f = []
    exit_f = []
    signal = []
    i = 0
    while i < len(enter):
        enter_f.append(float(enter[i]))
        exit_f.append(float(exit_[i]))
        if enter[i] == 1:
            signal.append(1.0)
        elif exit_[i] == 1:
            signal.append(-1.0)
        else:
            signal.append(0.0)
        i = i + 1

    enter_seq = _to_float_seq(enter_f)
    exit_seq = _to_float_seq(exit_f)
    signal_seq = _to_float_seq(signal)

    sma_f_seq = _to_float_seq(_nan_to_plot(sma_f))
    sma_s_seq = _to_float_seq(_nan_to_plot(sma_s))
    bb_up_seq = _to_float_seq(_nan_to_plot(bb_up))
    bb_lo_seq = _to_float_seq(_nan_to_plot(bb_lo))
    ndh_seq = _to_float_seq(_nan_to_plot(ndh))
    ndl_seq = _to_float_seq(_nan_to_plot(ndl))

    plot("SMA Fast", sma_f_seq, Color.yellow)
    plot("SMA Slow", sma_s_seq, Color.blue)
    plot("BB Upper", bb_up_seq, Color.gray)
    plot("BB Lower", bb_lo_seq, Color.gray)
    plot("N-day High", ndh_seq, Color.limagenta)
    plot("N-day Low", ndl_seq, Color.cyan)

    plot_icon("buy", enter_seq > 0, low(), Shape.arrowup, Color.white, 1, 0, 10)
    plot_icon("sell", exit_seq > 0, high(), Shape.arrowdown, Color.white, 1, 0, 10)

    output_parameter(
        Signal=signal_seq,
        Enter=enter_seq,
        Exit=exit_seq,
        SmaFast=sma_f_seq,
        SmaSlow=sma_s_seq,
        BbUpper=bb_up_seq,
        BbLower=bb_lo_seq,
        NDayHigh=ndh_seq,
        NDayLow=ndl_seq,
    )
`;
}

function emitNetworkWeights(networkGenome: Genome): string {
    const layers = decodeLayers(networkGenome);
    if (layers.length !== 3) {
        throw new Error(`Futu export expected 3 dense layers, got ${layers.length}`);
    }
    const [h1, h2, out] = layers;

    function layerMatrix(layer: {biases: number[]; weights: number[][]}): string {
        const rows = layer.biases.map((bias, nodeIndex) => {
            const cells = [bias, ...layer.weights[nodeIndex]].map(formatNum);
            return `    [${cells.join(", ")}]`;
        });
        return `[\n${rows.join(",\n")}\n]`;
    }

    return `H1 = ${layerMatrix(h1)}

H2 = ${layerMatrix(h2)}

OUT = ${layerMatrix(out)}
`;
}

function emitNetworkDecisionLoop(): string {
    // Indent: body of `while i < n` (8 spaces) after ready check
    return `        wr = -100.0 * (wh[i] - c[i]) / _max2(wh[i] - wl[i], 1e-9)
        bb_rng = _max2(bb_up[i] - bb_lo[i], 1e-9)
        bbpb = (c[i] - bb_lo[i]) / bb_rng
        volz = (v[i] - vol_ma[i]) / _max2(vol_sd[i], 1e-9)
        nhr = c[i] / _max2(ndh[i], 1e-9)

        feats = []
        feats.append(_clamp((c[i] / sma_f[i] - 1.0) * 10.0))
        feats.append(_clamp((c[i] / sma_s[i] - 1.0) * 10.0))
        feats.append(_clamp((sma_f[i] / sma_s[i] - 1.0) * 10.0))
        feats.append(_clamp((wr + 50.0) / 50.0))
        feats.append(_clamp(roc[i] * 5.0))
        feats.append(_clamp((rsi[i] - 50.0) / 50.0))
        feats.append(_clamp(macd_line[i] / c[i] * 25.0))
        feats.append(_clamp(macd_sig[i] / c[i] * 25.0))
        feats.append(_clamp((bbpb - 0.5) * 2.0))
        feats.append(_clamp(vlt[i] * 5.0))
        feats.append(_clamp(volz / 3.0))
        feats.append(_clamp((nhr - 0.95) * 20.0))
        nlr = ndl[i] / _max2(c[i], 1e-9)
        feats.append(_clamp((nlr - 0.95) * 20.0))
        feats.append(f13)
        feats.append(_clamp((RSI_BUY - rsi[i]) / 20.0))
        feats.append(_clamp((rsi[i] - RSI_SELL) / 20.0))
        feats.append(_clamp((WILL_BUY - wr) / 25.0))
        feats.append(_clamp((wr - WILL_SELL) / 25.0))

        h1 = []
        r = 0
        while r < len(H1):
            h1.append(_dense(feats, H1[r]))
            r = r + 1

        h2 = []
        r = 0
        while r < len(H2):
            h2.append(_dense(h1, H2[r]))
            r = r + 1

        out_buy = _dense(h2, OUT[0])
        out_hold = _dense(h2, OUT[1])
        out_sell = _dense(h2, OUT[2])

        # Sticky + margin (matches decidePositionFromNetwork): long / flat; sell = close.
        if position > 0:
            stay = out_hold
            if out_buy > stay:
                stay = out_buy
            buy_signal = False
            sell_signal = out_sell >= stay + ACTION_MARGIN
        else:
            stay_buy = out_hold
            if out_sell > stay_buy:
                stay_buy = out_sell
            buy_signal = out_buy >= stay_buy + ACTION_MARGIN
            sell_signal = False`;
}

function emitRuleDecisionLoop(): string {
    return `        wr = -100.0 * (wh[i] - c[i]) / _max2(wh[i] - wl[i], 1e-9)
        trend_vote = 0
        if sma_f[i] > sma_s[i]:
            trend_vote = 1
        macd_vote = 0
        if macd_line[i] > macd_sig[i]:
            macd_vote = 1
        rsi_buy_vote = 0
        if rsi[i] <= RSI_BUY:
            rsi_buy_vote = 1
        will_buy_vote = 0
        if wr <= WILL_BUY:
            will_buy_vote = 1
        buy_votes = trend_vote + macd_vote + rsi_buy_vote + will_buy_vote
        buy_signal = buy_votes >= 2
        rsi_sell = rsi[i] >= RSI_SELL
        will_sell = wr >= WILL_SELL
        uptrend = sma_f[i] > sma_s[i]
        if uptrend:
            sell_signal = rsi_sell and will_sell
        else:
            sell_signal = rsi_sell or will_sell`;
}
