import React from "react";
import {Button, Chip} from "@heroui/react";
import {Blocks, BookOpen, CandlestickChart, Dna, Github, Network} from "lucide-react";
import {Navigate, Route, Routes, useLocation, useNavigate} from "react-router-dom";
import {BreakerLab} from "./components/breaker-lab";
import {SnakeLab} from "./components/snake-lab";
import {StockLab} from "./components/stock-lab";
import {TheoryLab} from "./components/theory-lab";

const NAV_ITEMS = [
    {id: "theory" as const, path: "/theory", label: "演算法原理", shortLabel: "原理", icon: BookOpen, color: "neutral"},
    {id: "snake" as const, path: "/snake", label: "Snake Game", shortLabel: "Snake", icon: Dna, color: "snake"},
    {id: "breaker" as const, path: "/breaker", label: "Block Breaker", shortLabel: "Breaker", icon: Blocks, color: "breaker"},
    {id: "stock" as const, path: "/stock", label: "Stock Trading", shortLabel: "Trading", icon: CandlestickChart, color: "stock"},
];

export const App = React.memo(() => {
    const location = useLocation();
    const navigate = useNavigate();
    const activeItem = NAV_ITEMS.find(item => item.path === location.pathname) ?? NAV_ITEMS[0];

    return (
        <div className="app-shell">
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
                    <p>LEARNING PATH</p>
                    {NAV_ITEMS.map(item => (
                        <Button className={`nav-button nav-${item.color} ${activeItem.id === item.id ? "active" : ""}`} fullWidth key={item.id} onPress={() => navigate(item.path)} variant="tertiary">
                            <item.icon size={17} strokeWidth={1.5} />
                            <span>{item.label}</span>
                            <i />
                        </Button>
                    ))}
                </nav>
                <div className="sidebar-note">
                    <p>ENGINE</p>
                    <div>
                        <span>Neural network</span>
                        <strong>Brain.js</strong>
                    </div>
                    <div>
                        <span>Search method</span>
                        <strong>Genetic Algorithm</strong>
                    </div>
                    <div>
                        <span>Execution</span>
                        <strong>Web Workers</strong>
                    </div>
                </div>
                <a className="source-link" href="https://github.com/BrainJS/brain.js" rel="noreferrer" target="_blank">
                    <Github size={15} strokeWidth={1.5} /> Brain.js project
                </a>
            </aside>

            <div className="main-column">
                <header className="topbar">
                    <div className="mobile-brand">
                        <Network size={17} strokeWidth={1.5} />
                        <strong>EvoLab</strong>
                    </div>
                    <div className="breadcrumb">
                        <span>實驗室</span>
                        <span>/</span>
                        <strong>{activeItem.label}</strong>
                    </div>
                    <div className="topbar-status">
                        <span className="live-dot" />
                        <span>Local simulation</span>
                        <Chip color="warning" size="sm" variant="soft">
                            brain.js beta
                        </Chip>
                    </div>
                </header>
                <nav aria-label="手機實驗主題" className="mobile-nav">
                    {NAV_ITEMS.map(item => (
                        <Button className={activeItem.id === item.id ? "active" : ""} key={item.id} onPress={() => navigate(item.path)} size="sm" variant="tertiary">
                            <item.icon size={15} strokeWidth={1.5} />
                            {item.shortLabel}
                        </Button>
                    ))}
                </nav>
                <div className="content-scroll">
                    <Routes>
                        <Route element={<Navigate replace to="/theory" />} path="/" />
                        <Route element={<TheoryLab />} path="/theory" />
                        <Route element={<SnakeLab />} path="/snake" />
                        <Route element={<BreakerLab />} path="/breaker" />
                        <Route element={<StockLab />} path="/stock" />
                        <Route element={<Navigate replace to="/theory" />} path="*" />
                    </Routes>
                </div>
            </div>
        </div>
    );
});
