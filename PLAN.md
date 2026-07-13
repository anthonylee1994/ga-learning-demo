# EvoLab 遺傳演算法實驗室

## Summary

建立 Vite + React + TypeScript 單頁學習工作台，以繁體中文展示 Genetic Algorithm 理論，以及 Snake、Block Breaker、Stock Trading 三個真實運行 demo。三者統一使用 `brain.js` neural networks，由 GA 進化 network weights；Stock 預設使用 `QQQ` 十年日線數據。

## Theory Experience

- 工作台加入「演算法原理」主題，同三個實驗並列，唔做獨立 marketing landing page。
- 內容以短章節、流程圖、可操作示例呈現：
    1. **Genome / Chromosome**：一組 Brain.js weights 與 biases 如何代表一個 AI。
    2. **Population**：同一代包含多個不同 neural networks。
    3. **Fitness Function**：用分數衡量個體解決問題嘅能力。
    4. **Selection**：以 roulette wheel selection 按 fitness 比例增加優良個體繁殖機會。
    5. **Crossover**：由兩個 parents 混合 weights，產生 child genome。
    6. **Mutation**：隨機微調 weights，維持 population diversity。
    7. **Elitism**：最佳個體直接保留到下一代。
    8. **Generation Loop**：評估、排序、繁殖、突變、重新評估。
    9. **Exploration vs Exploitation**：mutation 太低會早熟收斂，太高會令學習不穩。
    10. **Neuroevolution**：GA 如何取代 backpropagation，直接搜尋 neural network parameters。
    11. **Overfitting**：特別解釋 Stock training/test split 點解重要。
    12. **限制**：運算成本、fitness design bias、隨機性，以及結果不保證全局最佳。
- 加入互動 crossover/mutation visualizer：
    - 顯示兩個 parent weight arrays。
    - 逐格標示 child 從邊個 parent 繼承。
    - mutation 後高亮被修改嘅 weights。
    - mutation rate slider 即時改變示例。
- 加入完整 GA 流程視覺：
  `初始化 Population → 評估 Fitness → Selection → Crossover → Mutation → Elitism → 下一代`
- 每個 demo 都有「點樣套用 GA」panel，列出 genome、inputs、outputs、fitness、termination condition。
- 理論文案會標明 GA 係 stochastic search，而唔係保證每次得到相同或最佳結果。

## Core Implementation

- 初始化 Vite React project，入口使用 `app.tsx`，遵從 `AGENTS.md` React 規範。
- 使用 HeroUI v3、Tailwind CSS v4、Lucide icons、Recharts、Matter.js、`brain.js`。
- 鎖定 `brain.js@2.0.0-beta.24`，使用 browser bundle及 TypeScript declarations。
- 建立 `NeuralNetworkAdapter`：
    - 固定各 domain 嘅 input、hidden layer、output topology。
    - GA genome 為扁平化 Brain.js weights 及 biases。
    - Adapter 將 genome 載入 Brain.js network，以 `run()` 做 forward inference。
    - 不使用 Brain.js `train()` 或 backpropagation。
- 三個訓練流程放入獨立 Web Workers。
- 共用 typed GA engine：seeded RNG、roulette wheel selection、elitism、uniform crossover、Gaussian mutation及 generation statistics。
- UI 提供開始、暫停、重設、速度、population、mutation rate、seed，以及 generation、best/average fitness、population diversity、champion replay。
- 控制項旁加入簡短 theory tooltip，解釋調高或調低參數嘅影響。
- 使用 versioned `localStorage` 保存設定、seed、champion genome及摘要；reload 後用 champion 建立新 population。

## Demo Behavior

### Snake

- 20×20 canvas，純 AI 觀察模式。
- Brain.js inputs 包括障礙方向、食物相對位置及目前移動方向；outputs 為左轉、直行、右轉。
- 多個固定 food seeds 評估；fitness 結合食物數、存活步數及接近食物獎勵，並設 step cap。
- Theory panel 明確展示：
    - Genome：Snake network weights/biases。
    - Fitness：食物分數、存活及距離 shaping。
    - Failure：撞牆、撞自己或超過 step cap。

### Block Breaker

- Matter.js 固定 timestep 模擬 paddle、ball、walls及 bricks。
- Brain.js inputs 包括 paddle/ball 位置、ball velocity及最近目標；outputs 為左、停、右。
- Fitness 根據清除 bricks、回球次數、存活時間及 clear bonus；使用固定 ball seeds。
- Theory panel 解釋同一 genome 點樣喺多個初始角度評估，避免靠一次好彩取得高 fitness。

### Stock Trading

- 預設 `QQQ`、十年日線，按時間順序分為 80% training、20% out-of-sample test。
- **Period-first evolution**：GA 主力搜尋 technical indicator periods；NN 只係薄 decision head（`13 → 4 → 3`）。period genes mutation ×3、NN weights ×0.35；每代 immigrant 只重抽 periods。
- Inputs 只使用當日及之前嘅 OHLCV：
    - SMA20、SMA50、close-to-SMA distance、SMA spread。
    - Williams %R 14。
    - ROC 12。
    - RSI 14。
    - MACD 12/26/9：line、signal、histogram。
    - Bollinger Bands 20/2：`%B`、bandwidth。
    - 20 日 rolling volatility。
    - 20 日 volume z-score。
    - N 日最高價（new high ratio：close / N-day high）。
    - 目前持倉比例。
- Warm-up 未完成嘅日期不參與訓練或測試；所有 inputs 會正規化及限制極端值。
- Brain.js outputs 為買入、持有、賣出，對應 100% long、保持倉位、100% cash；不做 short selling或 leverage。
- 每次倉位轉換計入 0.1% transaction cost。
- Fitness 只使用 training segment，綜合 total return、Sharpe ratio及 max drawdown；test segment 不參與 selection。
- 圖表顯示 QQQ 價格、買賣點、strategy equity、buy-and-hold benchmark及 train/test 分界。
- Theory panel 解釋 data leakage、overfitting、transaction costs，以及點解 test performance 可能差過 training。
- 可輸入其他 ticker；重設時恢復 `QQQ`。

## UI And Interfaces

- Desktop：左側 navigation、中間 theory/canvas/chart、右側參數及 metrics。
- Mobile：頂部 topic tabs、主內容、可展開控制面板。
- 深色中性介面，以綠、黃、紅區分實驗；無 gradient，panel/card 圓角不超過 8px。
- Express endpoint：`GET /api/market-data?symbol=QQQ&range=10y&interval=1d`
- Response 包含 symbol、currency、timezone、fetchedAt及排序後 OHLCV points；使用短期 memory cache。
- 無效 ticker 回 `400/404`，Yahoo/network 問題回 `502`。
- 共用 types：
    - `GAConfig`, `Genome`, `GenerationStats`, `Champion`
    - `NetworkTopology`, `SerializedBrainGenome`
    - `WorkerCommand`, `WorkerEvent`
    - `MarketDataPoint`, `IndicatorSnapshot`, `TradingResult`
    - `PersistedLabStateV1`
- Production Express server 同時提供 `/api` 及 Vite build。

## Test Plan

- GA tests：selection、elitism、crossover、mutation、seed reproducibility及 population invariants。
- Brain.js adapter tests：genome round-trip、deterministic output、topology weight count及 mutated genome loading。
- Theory visualizer tests：parent inheritance、mutation highlighting、mutation rate boundaries。
- Indicator tests：SMA、Williams %R、ROC、RSI、MACD、Bollinger Bands、volatility及 volume z-score。
- Stock tests：無 future leakage、warm-up、80/20 split、fees、fitness、QQQ default及 benchmark。
- Game tests：Snake collision/food/step cap；Block Breaker collision、scoring及 deterministic physics。
- API tests：mock `yahoo-finance2`，覆蓋成功、validation、空數據、failure及 cache。
- React/Playwright：理論章節、流程圖、tooltips、三個 demo controls、desktop/mobile layout、canvas及 charts。
- TypeScript、lint、tests及 production build全部通過。

## Assumptions

- 使用 `pnpm`、Node 20+；不加入登入、database、cloud persistence或多人功能。
- Brain.js beta dependency 鎖定 exact version，並由 adapter 隔離 API 變動。
- 理論內容以具備基本程式概念但未學過 GA 嘅讀者為目標。
- Yahoo Finance 數據只作教育用途，介面標示非投資建議。
- 網站名稱為「EvoLab 遺傳演算法實驗室」。
