import React from "react";
import {Button, Toast} from "@heroui/react";
import {Bird, Blocks, BookOpen, BrainCircuit, CandlestickChart, Dices, Dna, Network} from "lucide-react";
import {Navigate, Route, Routes, useLocation, useNavigate} from "react-router-dom";

const TheoryLab = React.lazy(() => import("./components/TheoryLab").then(module => ({default: module.TheoryLab})));
const SnakeLab = React.lazy(() => import("./components/SnakeLab").then(module => ({default: module.SnakeLab})));
const BreakerLab = React.lazy(() => import("./components/BreakerLab").then(module => ({default: module.BreakerLab})));
const BreakerPpoLab = React.lazy(() => import("./components/BreakerPpoLab").then(module => ({default: module.BreakerPpoLab})));
const FlappyLab = React.lazy(() => import("./components/FlappyLab").then(module => ({default: module.FlappyLab})));
const StockLab = React.lazy(() => import("./components/StockLab").then(module => ({default: module.StockLab})));
const StockMonteCarloLab = React.lazy(() => import("./components/StockLab").then(module => ({default: module.StockMonteCarloLab})));

const NAV_ITEMS = [
    {id: "theory" as const, path: "/theory", label: "演算法原理", icon: BookOpen, color: "neutral"},
    {id: "snake" as const, path: "/snake", label: "貪食蛇", icon: Dna, color: "snake"},
    {id: "breaker" as const, path: "/breaker", label: "撞磚 (GA)", icon: Blocks, color: "breaker"},
    {id: "breaker-ppo" as const, path: "/breaker-ppo", label: "撞磚 (PPO)", icon: BrainCircuit, color: "breaker-ppo"},
    {id: "flappy" as const, path: "/flappy", label: "Flappy Bird", icon: Bird, color: "flappy"},
    {id: "stock" as const, path: "/stock", label: "股票交易 (GA)", icon: CandlestickChart, color: "stock"},
    {id: "stock-mc" as const, path: "/stock-mc", label: "股票交易 (MC)", icon: Dices, color: "stock-mc"},
];

export const App = React.memo(() => {
    const location = useLocation();
    const navigate = useNavigate();
    const activeItem = NAV_ITEMS.find(item => item.path === location.pathname) ?? NAV_ITEMS[0];

    return (
        <div className="app-shell">
            <Toast.Provider placement="top end" />
            <aside className="sidebar">
                <div className="brand-lockup">
                    <div className="brand-mark">
                        <Network size={19} strokeWidth={1.5} />
                    </div>
                    <div>
                        <strong>EvoLab</strong>
                        <span>遺傳演算法實驗室</span>
                    </div>
                </div>
                <nav aria-label="實驗主題" className="sidebar-nav">
                    <p>學習路徑</p>
                    {NAV_ITEMS.map(item => (
                        <Button className={`nav-button nav-${item.color} ${activeItem.id === item.id ? "active" : ""}`} fullWidth key={item.id} onPress={() => navigate(item.path)} variant="tertiary">
                            <item.icon size={17} strokeWidth={1.5} />
                            <span>{item.label}</span>
                            <i />
                        </Button>
                    ))}
                </nav>
            </aside>

            <div className="main-column">
                <header className="topbar">
                    <div className="breadcrumb">
                        <span>實驗室</span>
                        <span>/</span>
                        <strong>{activeItem.label}</strong>
                    </div>
                </header>
                <div className="content-scroll">
                    <React.Suspense fallback={<div className="lab-loading">載入實驗中…</div>}>
                        <Routes>
                            <Route element={<Navigate replace to="/theory" />} path="/" />
                            <Route element={<TheoryLab />} path="/theory" />
                            <Route element={<SnakeLab />} path="/snake" />
                            <Route element={<BreakerLab />} path="/breaker" />
                            <Route element={<BreakerPpoLab />} path="/breaker-ppo" />
                            <Route element={<FlappyLab />} path="/flappy" />
                            <Route element={<StockLab />} path="/stock" />
                            <Route element={<StockMonteCarloLab />} path="/stock-mc" />
                            <Route element={<Navigate replace to="/theory" />} path="*" />
                        </Routes>
                    </React.Suspense>
                </div>
            </div>
        </div>
    );
});
