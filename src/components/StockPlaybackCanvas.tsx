import React from "react";
import type {TradeMarker, TradingPoint, TradingReplay} from "../lib/types";

/** Days visible in the sliding price window. */
const WINDOW_DAYS = 126;
/** Pause on the last day before looping (paused showcase). */
const TERMINAL_HOLD_MS = 1100;
const CANVAS_W = 900;
const CANVAS_H = 360;

export interface StockPlaybackDay {
    index: number;
    point: TradingPoint;
    position: number;
    trade: TradeMarker | null;
}

interface Props {
    replay?: TradingReplay;
    speed: number;
    /** Advance days when true. */
    playing?: boolean;
    /** After the last day, restart from 0. If false, freeze on last day. */
    loop?: boolean;
    /** Change to force restart from day 0 (e.g. pause showcase of latest champion). */
    restartKey?: number | string;
    /** Fires whenever the visible day changes (for live network activation). */
    onDayChange?: (day: StockPlaybackDay | null) => void;
}

/**
 * Game-style continuous trading playback — same role as SnakeCanvas / BreakerCanvas.
 * Draws a sliding price window on canvas so the heavy Recharts market chart stays static.
 */
export const StockPlaybackCanvas = React.memo<Props>(({replay, speed, playing = true, loop = true, restartKey = 0, onDayChange}) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    const [dayIndex, setDayIndex] = React.useState(0);
    const onDayChangeRef = React.useRef(onDayChange);
    onDayChangeRef.current = onDayChange;

    const positions = React.useMemo(() => (replay ? buildPositionSeries(replay) : null), [replay]);
    const tradeByDate = React.useMemo(() => {
        if (!replay) {
            return null;
        }
        return new Map(replay.trades.map(trade => [trade.date, trade]));
    }, [replay]);

    React.useEffect(() => {
        setDayIndex(0);
    }, [replay, restartKey]);

    React.useEffect(() => {
        if (!playing || !replay?.points.length) {
            return;
        }
        // Higher speed → more days per tick + shorter delay (full-history series needs chunking).
        const step = Math.max(1, Math.round(speed * 1.5));
        const frameMs = Math.max(16, 90 - speed * 12);
        const last = replay.points.length - 1;
        const atEnd = dayIndex >= last;
        const delay = atEnd && loop ? TERMINAL_HOLD_MS : frameMs;
        const timer = window.setTimeout(() => {
            setDayIndex(current => {
                if (last < 0) {
                    return 0;
                }
                if (current >= last) {
                    return loop ? 0 : last;
                }
                return Math.min(last, current + step);
            });
        }, delay);
        return () => window.clearTimeout(timer);
    }, [dayIndex, replay, speed, playing, loop]);

    React.useEffect(() => {
        if (!replay?.points.length || !positions || !tradeByDate) {
            onDayChangeRef.current?.(null);
            return;
        }
        const index = Math.min(dayIndex, replay.points.length - 1);
        const point = replay.points[index];
        onDayChangeRef.current?.({
            index,
            point,
            position: positions[index] ?? 0,
            trade: tradeByDate.get(point.date) ?? null,
        });
    }, [dayIndex, replay, positions, tradeByDate]);

    React.useEffect(() => {
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d");
        if (!canvas || !context) {
            return;
        }
        if (!replay?.points.length || !positions || !tradeByDate) {
            drawEmpty(context, canvas.width, canvas.height);
            return;
        }
        const index = Math.min(dayIndex, replay.points.length - 1);
        drawPlayback(context, canvas.width, canvas.height, replay, positions, tradeByDate, index);
    }, [dayIndex, replay, positions, tradeByDate]);

    const index = replay?.points.length ? Math.min(dayIndex, replay.points.length - 1) : 0;
    const point = replay?.points[index];
    const position = positions?.[index] ?? 0;
    const trade = point && tradeByDate ? (tradeByDate.get(point.date) ?? null) : null;
    const progress = replay?.points.length ? ((index + 1) / replay.points.length) * 100 : 0;

    return (
        <div className="stock-playback">
            <canvas aria-label="交易重播畫布" className="simulation-canvas stock-playback-canvas" data-day-index={index} data-position={position} height={CANVAS_H} ref={canvasRef} width={CANVAS_W} />
            <div className="stock-playback-hud" aria-live="polite">
                <HudStat label="日期" value={point?.date ?? "—"} mono />
                <HudStat label="收市" value={point ? formatPrice(point.close) : "—"} mono />
                <HudStat label="持倉" value={position > 0 ? "做多" : "現金"} accent={position > 0 ? "long" : "cash"} />
                <HudStat label="策略權益" value={point ? formatPrice(point.strategy) : "—"} mono />
                <HudStat label="買入持有" value={point ? formatPrice(point.benchmark) : "—"} mono />
                <HudStat label="動作" value={trade ? (trade.action === "buy" ? "買入" : "賣出") : "持有"} accent={trade?.action === "buy" ? "buy" : trade?.action === "sell" ? "sell" : undefined} />
                <HudStat label="區段" value={segmentLabel(point?.segment)} />
                <HudStat label="進度" value={`${index + 1}/${replay?.points.length ?? 0}`} mono />
            </div>
            <div className="stock-playback-progress" aria-hidden>
                <div className="stock-playback-progress-fill" style={{width: `${progress}%`}} />
            </div>
        </div>
    );
});

const HudStat = React.memo(({label, value, mono, accent}: {label: string; value: string; mono?: boolean; accent?: "long" | "cash" | "buy" | "sell"}) => (
    <div className={accent ? `stock-hud-stat stock-hud-stat--${accent}` : "stock-hud-stat"}>
        <span>{label}</span>
        <strong className={mono ? "font-mono" : undefined}>{value}</strong>
    </div>
));

function buildPositionSeries(replay: TradingReplay): number[] {
    const positions = new Array<number>(replay.points.length);
    let position = 0;
    let tradeIndex = 0;
    const trades = replay.trades;
    for (let index = 0; index < replay.points.length; index += 1) {
        const date = replay.points[index].date;
        while (tradeIndex < trades.length && trades[tradeIndex].date <= date) {
            position = trades[tradeIndex].action === "buy" ? 1 : 0;
            tradeIndex += 1;
        }
        positions[index] = position;
    }
    return positions;
}

function drawEmpty(context: CanvasRenderingContext2D, width: number, height: number): void {
    context.fillStyle = "#0a0d0f";
    context.fillRect(0, 0, width, height);
    context.fillStyle = "#4a525c";
    context.font = "14px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.textAlign = "center";
    context.fillText("訓練出冠軍後會喺度逐日重播買賣", width / 2, height / 2);
}

function drawPlayback(context: CanvasRenderingContext2D, width: number, height: number, replay: TradingReplay, positions: number[], tradeByDate: Map<string, TradeMarker>, dayIndex: number): void {
    const padL = 54;
    const padR = 16;
    const padT = 18;
    const padB = 28;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    context.fillStyle = "#0a0d0f";
    context.fillRect(0, 0, width, height);

    const end = dayIndex;
    const start = Math.max(0, end - WINDOW_DAYS + 1);
    const slice = replay.points.slice(start, end + 1);
    if (slice.length < 2) {
        drawEmpty(context, width, height);
        return;
    }

    let minY = Infinity;
    let maxY = -Infinity;
    for (const point of slice) {
        minY = Math.min(minY, point.close, point.bollingerLower || point.close);
        maxY = Math.max(maxY, point.close, point.bollingerUpper || point.close);
    }
    const padY = (maxY - minY) * 0.08 || maxY * 0.02 || 1;
    minY -= padY;
    maxY += padY;
    const spanY = maxY - minY || 1;

    const xAt = (i: number) => padL + (i / (slice.length - 1)) * plotW;
    const yAt = (price: number) => padT + (1 - (price - minY) / spanY) * plotH;

    // Long/cash background strips
    context.save();
    for (let i = 0; i < slice.length; i += 1) {
        const globalIndex = start + i;
        if ((positions[globalIndex] ?? 0) <= 0) {
            continue;
        }
        const x0 = i === 0 ? xAt(0) : (xAt(i - 1) + xAt(i)) / 2;
        const x1 = i === slice.length - 1 ? xAt(i) : (xAt(i) + xAt(i + 1)) / 2;
        context.fillStyle = "rgba(88, 214, 141, 0.07)";
        context.fillRect(x0, padT, Math.max(1, x1 - x0), plotH);
    }
    context.restore();

    // Grid
    context.strokeStyle = "#151a1e";
    context.lineWidth = 1;
    for (let g = 0; g <= 4; g += 1) {
        const y = padT + (g / 4) * plotH;
        context.beginPath();
        context.moveTo(padL, y);
        context.lineTo(padL + plotW, y);
        context.stroke();
    }

    // Bollinger band (faint)
    if (slice.some(p => p.bollingerUpper && p.bollingerLower)) {
        context.beginPath();
        slice.forEach((point, i) => {
            const y = yAt(point.bollingerUpper || point.close);
            if (i === 0) {
                context.moveTo(xAt(i), y);
            } else {
                context.lineTo(xAt(i), y);
            }
        });
        for (let i = slice.length - 1; i >= 0; i -= 1) {
            context.lineTo(xAt(i), yAt(slice[i].bollingerLower || slice[i].close));
        }
        context.closePath();
        context.fillStyle = "rgba(111, 119, 130, 0.08)";
        context.fill();
    }

    // SMA slow / fast
    drawLine(
        context,
        slice.map((p, i) => ({x: xAt(i), y: yAt(p.smaSlow || p.close)})),
        "#5da6d9",
        1
    );
    drawLine(
        context,
        slice.map((p, i) => ({x: xAt(i), y: yAt(p.smaFast || p.close)})),
        "#e7b955",
        1
    );

    // Close price
    drawLine(
        context,
        slice.map((p, i) => ({x: xAt(i), y: yAt(p.close)})),
        "#dfe3e8",
        1.75
    );

    // Trade markers in window
    for (let i = 0; i < slice.length; i += 1) {
        const trade = tradeByDate.get(slice[i].date);
        if (!trade) {
            continue;
        }
        const x = xAt(i);
        const y = yAt(trade.price);
        const isBuy = trade.action === "buy";
        context.fillStyle = isBuy ? "#58d68d" : "#e36f5b";
        context.beginPath();
        if (isBuy) {
            context.moveTo(x, y - 7);
            context.lineTo(x - 6, y + 5);
            context.lineTo(x + 6, y + 5);
        } else {
            context.moveTo(x, y + 7);
            context.lineTo(x - 6, y - 5);
            context.lineTo(x + 6, y - 5);
        }
        context.closePath();
        context.fill();
    }

    // Playhead (current day)
    const headX = xAt(slice.length - 1);
    const headY = yAt(slice[slice.length - 1].close);
    context.strokeStyle = "rgba(231, 185, 85, 0.85)";
    context.lineWidth = 1;
    context.setLineDash([4, 3]);
    context.beginPath();
    context.moveTo(headX, padT);
    context.lineTo(headX, padT + plotH);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#e7b955";
    context.beginPath();
    context.arc(headX, headY, 4.5, 0, Math.PI * 2);
    context.fill();

    // Flash ring on trade day
    const headTrade = tradeByDate.get(slice[slice.length - 1].date);
    if (headTrade) {
        context.strokeStyle = headTrade.action === "buy" ? "#58d68d" : "#e36f5b";
        context.lineWidth = 2;
        context.beginPath();
        context.arc(headX, headY, 11, 0, Math.PI * 2);
        context.stroke();
    }

    // Y labels
    context.fillStyle = "#6e7781";
    context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (let g = 0; g <= 4; g += 1) {
        const price = maxY - (g / 4) * spanY;
        const y = padT + (g / 4) * plotH;
        context.fillText(formatPrice(price), padL - 8, y);
    }

    // X labels
    context.textAlign = "center";
    context.textBaseline = "top";
    const labelCount = Math.min(5, slice.length);
    for (let g = 0; g < labelCount; g += 1) {
        const i = Math.round((g / Math.max(1, labelCount - 1)) * (slice.length - 1));
        context.fillText(slice[i].date.slice(0, 7), xAt(i), padT + plotH + 8);
    }

    // Train / test badge on playhead day
    const segment = slice[slice.length - 1].segment;
    const badgeStyle =
        segment === "test"
            ? {fill: "rgba(231, 185, 85, 0.15)", stroke: "#e7b955", text: "#e7b955", label: "純樣本外測試"}
            : {fill: "rgba(88, 214, 141, 0.12)", stroke: "#58d68d", text: "#58d68d", label: "訓練段"};
    context.fillStyle = badgeStyle.fill;
    context.strokeStyle = badgeStyle.stroke;
    context.lineWidth = 1;
    const badge = badgeStyle.label;
    context.font = "11px ui-sans-serif, system-ui, sans-serif";
    const tw = context.measureText(badge).width;
    const bx = padL + 8;
    const by = padT + 8;
    context.beginPath();
    context.roundRect(bx, by, tw + 14, 20, 4);
    context.fill();
    context.stroke();
    context.fillStyle = badgeStyle.text;
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(badge, bx + 7, by + 10);
}

function segmentLabel(segment: TradingReplay["points"][number]["segment"] | undefined): string {
    if (segment === "test") {
        return "測試";
    }
    return "訓練";
}

function drawLine(context: CanvasRenderingContext2D, points: {x: number; y: number}[], stroke: string, width: number): void {
    if (points.length < 2) {
        return;
    }
    context.beginPath();
    context.strokeStyle = stroke;
    context.lineWidth = width;
    context.lineJoin = "round";
    points.forEach((point, index) => {
        if (index === 0) {
            context.moveTo(point.x, point.y);
        } else {
            context.lineTo(point.x, point.y);
        }
    });
    context.stroke();
}

function formatPrice(value: number): string {
    return new Intl.NumberFormat("en-US", {maximumFractionDigits: 2, minimumFractionDigits: value < 100 ? 2 : 0}).format(value);
}
