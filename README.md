# EvoLab — Genetic Algorithm 說明

EvoLab 用 **neuroevolution**：唔做 backpropagation，而係用遺傳演算法（GA）直接搜尋 neural network weights（同 Stock 嘅 indicator periods / on-off masks）。三個實驗（Snake、Block Breaker、Stock）共用同一套 GA engine。

核心實作：`src/lib/ga.ts`  
每代循環：`src/workers/workerRuntime.ts`

---

## 一代流程

```
初始化 Population
        ↓
評估 Fitness（每個 genome 跑一次 domain simulate）
        ↓
按 fitness 排序
        ↓
Elitism：保留最好嘅幾個，原封不動帶去下一代
        ↓
其餘空位用 Selection → Crossover → Mutation 填滿
        ↓
Random Immigrant：最後一個 slot 換成新鮮 genome（維持探索）
        ↓
下一代
```

Worker 每一代做嘅事大致係：

1. `population.map(evaluate)` — 計齊所有個體嘅 fitness
2. `evolvePopulation(...)` — 由今代繁殖出下一代
3. 把 best genome / stats /（有需要時）champion replay 傳返 UI

---

## Genome

一個 **genome** 就係一個 `number[]`（扁平化基因）。

| Demo            | Genome 內容                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| Snake / Breaker | Brain.js 嘅 weights + biases                                                                         |
| Stock           | Head = 17 個 period/threshold + 9 個 indicator mask；Tail = 薄 decision head（`17 → 10 → 3`）weights |

初始化時每個 gene 大約係 `gaussian() * 0.55`。可選 champion 會放喺第 0 位；Stock 仲會注入幾組經典 indicator seed（含 dense / sparse mask 先驗）。

### Stock genome 明細

| 區段 | Genes | 解碼                                                                         |
| ---- | ----- | ---------------------------------------------------------------------------- |
| 週期 | 0–16  | SMA / Williams / ROC / RSI / MACD / BB / 波動 / 成交量 / N 日新高 週期與門檻 |
| Mask | 17–25 | 9 個指標家族 on/off（`gene > 0` → ON）                                       |
| NN   | 26…   | Brain.js decision head weights + biases                                      |

**Mask 家族**（feature selection）：移動平均線、威廉指標、ROC、RSI、MACD、保力加通道、波動率、成交量、N 日新高。  
關掉嘅家族喺 NN 輸入會填 0；規則模式只讓 ON 嘅家族投票。持倉狀態（feature 12）永遠開啟。

---

## Selection — Roulette Wheel Selection

每個 genome 嘅 fitness 會轉成輪盤權重，fitness 愈高，抽中做 parent 嘅機會愈大。

```
rouletteWheelSelect(candidates, random):
  將 fitness 轉成非負權重
  隨機抽一個落點
  回傳落點所在區段嘅 genome
```

每次製造一個 child 會轉兩次輪盤，得到 `parentA`、`parentB`。

- fitness 高 → 輪盤區段較大，偏向 exploitation
- fitness 低 → 仍然保留繁殖機會，維持 exploration
- 有負 fitness 時會先平移成非負權重
- 全部權重都係 0 時會退回均勻隨機選擇

同 tournament selection 唔同，roulette wheel selection 唔需要 `tournamentSize` 設定。

---

## Crossover — Uniform Crossover

每個 gene **獨立** 以 50% 機率由父或母繼承：

```
child[i] = random() < 0.5 ? parentA[i] : parentB[i]
```

即係 `uniformCrossover`：唔係單點 / 雙點 crossover，而係逐個 gene 擲銅板。長度同 parents 一樣。

---

## Mutation — Gaussian + Reset

Crossover 之後，每個 child 會經過 `mutateGenome`。

對每個 gene：

1. 以 `mutationRate`（可再乘 profile multiplier）決定是否突變
2. 若突變：

- **20%**（`RESET_MUTATION_SHARE`）：整粒 gene 重抽 → `gaussian() * 0.55`（reset）
    - **80%**：喺原值上加噪聲 → `gene + gaussian() * mutationScale`（perturb）

Reset 用意：避免 `tanh` 解碼嘅 period genes 飽和之後變相「鎖死」，亦令收斂後嘅 population 仍可跳離 local optimum。

### Mutation Profile（Stock 專用）

Stock 用 `STOCK_MUTATION_PROFILE`，把 genome 分成 head / tail：

| 區段                                 | Genes | Rate 倍數 | Scale 倍數 |
| ------------------------------------ | ----- | --------- | ---------- |
| Head（periods + thresholds + masks） | 0–25  | ×3        | ×1.5       |
| Tail（NN weights）                   | 其餘  | ×0.35     | ×0.45      |

即係 **indicator-first evolution**：搜尋預算多數花喺「用邊啲指標 + 點樣調參數」，NN 只係薄 decision head，突變較少。

---

## Elitism

```
eliteCount = max(1, floor(populationSize * eliteRate))
```

排序後最好嘅 `eliteCount` 個 genome **原樣複製**入下一代，唔經 crossover / mutation。保證當代最佳解唔會因為隨機操作而消失。

例如 Stock 預設 `eliteRate: 0.08`、`populationSize: 48` → 約保留 3 個 elite。

---

## Random Immigrant

每代填完之後，**最後一個 slot** 會被換成「移民」：

- 一般情況：完全隨機新 genome
- Stock（`immigrantHeadOnly: true`）：用當代 elite 做模板，**只重抽 head（periods + masks）**，保留 NN tail

咁樣收斂後仍會持續試新 indicator 組合同 on/off 配置，又唔會每次都打亂已經穩定嘅 decision head。

---

## Stock fitness 同 feature selection

Fitness **只睇時間序前 80% training**；最後 20% 係 sample 外測試，**唔入 selection**。

大致組成：

| 項             | 作用                                                    |
| -------------- | ------------------------------------------------------- |
| 全段訓練 score | 70%：CAGR、Sharpe、回撤、相對 buy-and-hold 超額         |
| 半段 robust    | 30%：train 前半 / 後半取較差嗰邊，減低單段運氣          |
| L2（NN on 時） | 懲罰大 weight，避免肥網硬記                             |
| Sparsity       | 前 3 個開着嘅指標家族免費；之後每個 −2.2；全部關 → 硬罰 |

**Ablation（UI）**：冠軍出現後，逐個關 ON 緊嘅 mask，重計 fitness。

- `Δ fitness > 0` → 關掉會差 → 呢個指標對當前策略有貢獻
- `Δ fitness ≤ 0` → 關掉唔差甚至更好 → 傾向裝飾品 / 可關

進化中 ablation 跟 throttled champion showcase 更新，暫停後跟 live 冠軍。

---

## 可調參數（`GAConfig`）

| 參數               | 作用                                       |
| ------------------ | ------------------------------------------ |
| `populationSize`   | 每代個體數                                 |
| `mutationRate`     | 每個 gene 突變機率（再經 profile 調整）    |
| `mutationScale`    | Gaussian 擾動幅度                          |
| `eliteRate`        | 直接保留到下一代嘅比例                     |
| `seed`             | 可重現嘅 RNG seed                          |
| `speed`            | Worker 代與代之間嘅延遲（只影響 UI 節奏）  |
| `useNeuralNetwork` | Stock：開 NN decision head；關則用規則投票 |

---

## 關鍵原始碼

| 功能                                                                          | 位置                                                                               |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `createPopulation` / `uniformCrossover` / `mutateGenome` / `evolvePopulation` | `src/lib/ga.ts`                                                                    |
| Roulette wheel selection                                                      | `src/lib/ga.ts` → `rouletteWheelSelect`                                            |
| Seeded RNG（含 Gaussian）                                                     | `src/lib/random.ts`                                                                |
| 每代 evaluate → evolve 循環                                                   | `src/workers/workerRuntime.ts`                                                     |
| Stock genome 解碼、mask 定義、mutation profile、seeds                         | `src/domains/stock/strategyGenome.ts`                                              |
| Stock fitness、mask 應用、ablation                                            | `src/domains/stock/simulation.ts` → `evaluateStockGenome` / `ablateIndicatorMasks` |
| Pine Script 匯出（含 mask）                                                   | `src/domains/stock/pineScript.ts`                                                  |
| Stock UI（參數、mask chips、ablation 表）                                     | `src/components/StockLab.tsx`                                                      |

---

## 一句總結

每代：**評估 → 保留精英 → roulette wheel 揀父母 → uniform crossover → Gaussian/reset mutation → 塞一個 immigrant**，再重複。  
Stock 額外用 **mask + sparsity** 做指標選擇，再用 **ablation** 解釋冠軍邊啲指標真係有用。  
GA 係 stochastic search，同一 seed 可重現；唔保證全局最佳，亦唔係投資建議。
