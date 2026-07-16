import React from "react";
import {Button, Card, Chip} from "@heroui/react";
import {ArrowRight, Atom, Binary, Braces, Cpu, Dna, FlaskConical, GitBranch, Layers, Network, RefreshCw, ShieldCheck, Shuffle, Sparkles, Target, TrendingUp, Users, Workflow} from "lucide-react";
import {Link} from "react-router-dom";
import {createRandom} from "../lib/random";

const FLOW_STEPS = [
    {
        id: "init",
        label: "初始化族群",
        metaphor: "一班新學生入學",
        title: "隨機出一班候選解",
        body: "每個「學生」身上有一串數字（genome）。Snake / Breaker 係神經網絡權重；Stock 仲包埋指標週期同門檻。",
        project: "createPopulation() · seeded RNG · 可注入上代 champion",
    },
    {
        id: "eval",
        label: "評估適應度",
        metaphor: "全班考同一份卷",
        title: "同一個規則考晒成班",
        body: "每個 genome 入 simulation 跑幾場，攞一個 fitness。分數高 = 較容易生仔傳落去。",
        project: "evaluate*Genome() · Web Worker 跑一代 · 多 seed 取平均",
    },
    {
        id: "select",
        label: "選擇",
        metaphor: "抽 3 個比併做家長",
        title: "錦標賽抽家長",
        body: "每次隨機抽 3 個，最強做 parent。壓力靠排名唔靠分數比例，後期唔會「人人一樣」。",
        project: "tournamentSelect() · TOURNAMENT_SIZE = 3",
    },
    {
        id: "cross",
        label: "交配",
        metaphor: "兩本筆記砌一本",
        title: "均勻雜交",
        body: "子女每一格 50% 抄 A、50% 抄 B——好似兩本溫書各撕一半砌埋。",
        project: "uniformCrossover()",
    },
    {
        id: "mutate",
        label: "突變",
        metaphor: "筆記改亂幾格",
        title: "低機率擾動",
        body: "約 80% 突變係高斯微調，約 20% 整格重寫。Stock 指標基因突變更兇，決策頭較溫和。",
        project: "mutateGenome() · MutationProfile",
    },
    {
        id: "elite",
        label: "菁英保留",
        metaphor: "保送 + 插班生",
        title: "冠軍直接晉級",
        body: "按 eliteRate 保留最強幾個唔改。最後一位塞 random immigrant，防收斂死透（Stock 移民只重抽指標 head）。",
        project: "eliteRate 切片 + immigrant 槽位",
    },
    {
        id: "next",
        label: "下一代",
        metaphor: "新一班再考",
        title: "循環直到你停",
        body: "新 population 再評估。UI 顯示 best / average fitness 同 diversity。",
        project: "evolvePopulation() → WorkerEvent generation",
    },
] as const;

const CONCEPTS = [
    {
        icon: Dna,
        title: "基因體 / 染色體",
        text: "一串數字＝學生筆記。完整描述「見到咩就點做」——通常係網絡權重同偏差。",
        visual: "genome" as const,
    },
    {
        icon: Users,
        title: "族群",
        text: "同一代有幾十個學生一齊試，唔將希望押喺單一解法。",
        visual: "population" as const,
    },
    {
        icon: Target,
        title: "適應度函數",
        text: "考試分數。你寫咩獎勵，AI 就優化咩——錯嘅卷會穩定學出錯行為。",
        visual: "fitness" as const,
    },
    {
        icon: ShieldCheck,
        title: "選擇 + 菁英",
        text: "錦標賽選家長；菁英原封不動晉級，避免一代倒退。",
        visual: "select" as const,
    },
    {
        icon: Shuffle,
        title: "交配",
        text: "兩個父母逐格混合，子女同時繼承兩邊已經有效嘅結構。",
        visual: "crossover" as const,
    },
    {
        icon: Sparkles,
        title: "突變",
        text: "低機率改亂幾格，注入新方向，避免成班人困喺同一個壞習慣。",
        visual: "mutation" as const,
    },
    {
        icon: Network,
        title: "神經演化",
        text: "唔用反向傳播。brain.js 只做 forward；GA 負責改權重。",
        visual: "neuro" as const,
    },
    {
        icon: TrendingUp,
        title: "探索 vs 利用",
        text: "突變太低早熟收斂，太高就破壞成果。diversity 指標反映呢點。",
        visual: "explore" as const,
    },
    {
        icon: Braces,
        title: "過擬合",
        text: "高訓練分 ≠ 識新環境。股票保留 20% 未見過數據做 test。",
        visual: "overfit" as const,
    },
    {
        icon: Atom,
        title: "隨機搜尋",
        text: "GA 係 stochastic。種子可重播，但唔保證全局最佳。",
        visual: "random" as const,
    },
    {
        icon: Binary,
        title: "適應度偏差",
        text: "獎勵「行耐」就會有人轉圈刷步——所以 Snake 有餓死機制。",
        visual: "bias" as const,
    },
    {
        icon: FlaskConical,
        title: "計算限制",
        text: "population × generations × 每場步數 = 真成本。Worker 只係令 UI 唔卡。",
        visual: "cost" as const,
    },
] as const;

const PIPELINE = [
    {icon: Layers, title: "UI 參數", text: "population · mutation · elite · seed · speed"},
    {icon: Cpu, title: "Web Worker", text: "snake / breaker / flappy / stock.worker 跑演化"},
    {icon: GitBranch, title: "ga.ts", text: "create / select / cross / mutate / elite"},
    {icon: Network, title: "NN Adapter", text: "genome → forward run() · 無 train()"},
    {icon: Target, title: "Domain sim", text: "fitness 分數 + champion replay"},
    {icon: Workflow, title: "UI 回傳", text: "metrics · chart · canvas 重播"},
] as const;

const DEMOS = [
    {
        path: "/snake",
        accent: "snake" as const,
        title: "貪食蛇",
        blurb: "唔撞牆、盡量食果",
        topology: "10 → 12 → 3",
        genes: "171 權重",
        genome: "整條網絡權重 / 偏差",
        fitness: "食物²×180 + 存活 + 靠近 shaping",
        note: "兩個固定 seed 平均，減少一場好彩",
        defaults: "族群 36 · 突變 12% · 菁英 8%",
    },
    {
        path: "/breaker",
        accent: "breaker" as const,
        title: "撞磚",
        blurb: "接波、清磚",
        topology: "8 → 12 → 3",
        genes: "147 權重",
        genome: "整條網絡權重 / 偏差",
        fitness: "清磚 + 全清獎 − 多餘接球",
        note: "5 場唔同發射角平均，逼學跟波",
        defaults: "族群 12 · 突變 14% · 菁英保留",
    },
    {
        path: "/flappy",
        accent: "flappy" as const,
        title: "Flappy Bird",
        blurb: "拍翼過水管",
        topology: "6 → 10 → 2",
        genes: "92 權重",
        genome: "整條網絡權重 / 偏差",
        fitness: "過管² + 存活 + 縫心 shaping",
        note: "三個固定 seed 平均，減少水管幸運",
        defaults: "族群 40 · 突變 13% · 菁英 8%",
    },
    {
        path: "/stock",
        accent: "stock" as const,
        title: "股票 (GA)",
        blurb: "幾時買 / 賣、點調指標",
        topology: "22 → 10 → 5 → 3",
        genes: "18 參數 + 303 權重",
        genome: "指標週期/門檻（突變 ×3）+ 薄決策頭",
        fitness: "訓練段超額回報 − 回撤 − 換手",
        note: "80/20 train·test；移民只重抽指標 head",
        defaults: "教學 demo · 非投資建議",
    },
] as const;

const ANALOGY_ROWS = [
    {nature: "一群生物", code: "Population（一組 genome）"},
    {nature: "DNA / 遺傳訊息", code: "Genome（數字陣列）"},
    {nature: "生存能力", code: "Fitness（simulation 分數）"},
    {nature: "繁殖 + 突變", code: "Crossover / Mutate（ga.ts）"},
] as const;

const PARENT_A = [0.82, -0.31, 0.14, 0.67, -0.74, 0.26, 0.51, -0.08];
const PARENT_B = [-0.44, 0.76, 0.39, -0.22, 0.91, -0.58, 0.05, 0.63];

export const TheoryLab = React.memo(() => {
    const [mutationRate, setMutationRate] = React.useState(0.18);
    const [seed, setSeed] = React.useState(42);
    const [activeStep, setActiveStep] = React.useState(0);
    const child = React.useMemo(() => createChild(mutationRate, seed), [mutationRate, seed]);
    const step = FLOW_STEPS[activeStep];
    const mutatedCount = child.filter(gene => gene.mutated).length;

    return (
        <div className="theory-view">
            <section className="theory-intro">
                <div>
                    <p className="eyebrow">遺傳演算法基礎</p>
                    <h2>用演化，搜尋一個夠好嘅決策腦</h2>
                    <p>
                        唔係人手寫「見到紅燈就停」啲死規則，而係：隨機出一班策略 → 睇邊個叻 → 好嘅生仔 → 仔有少少突變 → 重複幾百代。EvoLab 進化嘅係神經網絡權重（股票仲進化指標參數）；
                        <strong className="theory-inline-strong">brain.js 只負責推理，GA 負責改權重</strong>。
                    </p>
                </div>
                <div className="intro-stat-stack">
                    <div className="intro-stat">
                        <span>核心循環</span>
                        <strong>評估 → 演化</strong>
                        <small>隨機搜尋 · 可重播種子</small>
                    </div>
                    <div className="intro-stat intro-stat--muted">
                        <span>記住</span>
                        <strong>唔保證全局最佳</strong>
                        <small>教學 demo · 非投資軟件</small>
                    </div>
                </div>
            </section>

            <section className="theory-analogy">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">直觀對照</p>
                        <h3>自然界 vs 呢個 project</h3>
                    </div>
                </div>
                <div className="analogy-table" role="table" aria-label="自然界同 EvoLab 對照">
                    <div className="analogy-table-head" role="row">
                        <span role="columnheader">自然界</span>
                        <span className="analogy-table-spacer" aria-hidden="true" />
                        <span role="columnheader">EvoLab</span>
                    </div>
                    {ANALOGY_ROWS.map(row => (
                        <div className="analogy-table-row" key={row.code} role="row">
                            <span role="cell">{row.nature}</span>
                            <ArrowRight aria-hidden="true" className="analogy-row-arrow" size={14} strokeWidth={1.5} />
                            <span role="cell">{row.code}</span>
                        </div>
                    ))}
                </div>
            </section>

            <section className="flow-section">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">世代循環</p>
                        <h3>一代係點樣誕生</h3>
                    </div>
                    <Chip color="accent" size="sm" variant="soft">
                        撳步驟睇詳情
                    </Chip>
                </div>
                <p className="section-copy">好似一班人考試完再收生：保送精英 → 抽家長合併筆記 → 加突變 → 塞個插班生 → 新一班再考。</p>
                <div className="flow-track" role="tablist" aria-label="GA 世代步驟">
                    {FLOW_STEPS.map((item, index) => (
                        <React.Fragment key={item.id}>
                            <button
                                aria-selected={activeStep === index}
                                className={`flow-stage ${activeStep === index ? "is-active" : ""}`}
                                onClick={() => setActiveStep(index)}
                                role="tab"
                                type="button"
                            >
                                <span>{String(index + 1).padStart(2, "0")}</span>
                                {item.label}
                            </button>
                            {index < FLOW_STEPS.length - 1 ? <ArrowRight aria-hidden="true" size={16} strokeWidth={1.5} /> : null}
                        </React.Fragment>
                    ))}
                </div>
                <div className="flow-detail">
                    <div className="flow-detail-main">
                        <p className="flow-detail-kicker">
                            步驟 {String(activeStep + 1).padStart(2, "0")} · {step.label}
                        </p>
                        <p className="flow-detail-metaphor">{step.metaphor}</p>
                        <h4>{step.title}</h4>
                        <p>{step.body}</p>
                    </div>
                    <code className="flow-detail-code">{step.project}</code>
                </div>
                <GenerationLoopDiagram activeStep={activeStep} />
            </section>

            <section className="theory-concepts">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">核心概念</p>
                        <h3>圖文速覽</h3>
                    </div>
                </div>
                <div className="theory-grid">
                    {CONCEPTS.map(item => (
                        <Card className="theory-card rounded-lg" key={item.title} variant="default">
                            <div className="theory-card-top">
                                <item.icon aria-hidden="true" size={18} strokeWidth={1.5} />
                                <ConceptGlyph kind={item.visual} />
                            </div>
                            <Card.Header>
                                <Card.Title className="text-sm">{item.title}</Card.Title>
                            </Card.Header>
                            <Card.Content>
                                <p>{item.text}</p>
                            </Card.Content>
                        </Card>
                    ))}
                </div>
            </section>

            <section className="genome-lab">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">互動基因實驗室</p>
                        <h3>交配 + 突變</h3>
                    </div>
                    <div className="genome-lab-actions">
                        <span className="genome-lab-stat">
                            已突變 <strong>{mutatedCount}</strong> / {child.length}
                        </span>
                        <Button isIconOnly onPress={() => setSeed(value => value + 1)} variant="tertiary">
                            <RefreshCw aria-label="重新生成子女" size={16} strokeWidth={1.5} />
                        </Button>
                    </div>
                </div>
                <p className="section-copy">
                    每一格係一個網絡權重。子女先 50/50 繼承（同 <code>uniformCrossover</code>），再按突變率擾動（同 <code>mutateGenome</code>）。撳右上角可重抽繼承圖樣。
                </p>
                <div className="crossover-diagram" aria-hidden="true">
                    <div className="xo-block">
                        <span className="xo-label">父母 A</span>
                        <div className="xo-genes">
                            {PARENT_A.map((_, index) => (
                                <i className="from-a" key={`xa-${index}`} />
                            ))}
                        </div>
                    </div>
                    <div className="xo-block">
                        <span className="xo-label">父母 B</span>
                        <div className="xo-genes">
                            {PARENT_B.map((_, index) => (
                                <i className="from-b" key={`xb-${index}`} />
                            ))}
                        </div>
                    </div>
                    <div className="xo-block xo-block--child">
                        <span className="xo-label">
                            子女 · uniform 50/50
                            {mutatedCount > 0 ? ` · 突變 ${mutatedCount}` : ""}
                        </span>
                        <div className="xo-genes">
                            {child.map((gene, index) => (
                                <i className={`from-${gene.source}${gene.mutated ? "mutated" : ""}`} key={`xc-${index}`} />
                            ))}
                        </div>
                    </div>
                </div>
                <div className="genome-rows">
                    <GenomeRow label="父母 A" values={PARENT_A.map(value => ({value, source: "a" as const, mutated: false}))} />
                    <GenomeRow label="父母 B" values={PARENT_B.map(value => ({value, source: "b" as const, mutated: false}))} />
                    <GenomeRow label="子女" values={child} />
                </div>
                <label className="mutation-control">
                    <span>
                        <span>突變率</span>
                        <strong>{Math.round(mutationRate * 100)}%</strong>
                    </span>
                    <input
                        aria-label="突變率"
                        className="range-input"
                        max={0.6}
                        min={0}
                        onChange={event => setMutationRate(Number(event.target.value))}
                        step={0.01}
                        type="range"
                        value={mutationRate}
                    />
                </label>
                <div className="genome-legend">
                    <span>
                        <i className="gene-a" />
                        父母 A
                    </span>
                    <span>
                        <i className="gene-b" />
                        父母 B
                    </span>
                    <span>
                        <i className="gene-mutated" />
                        已突變
                    </span>
                </div>
            </section>

            <section className="theory-pipeline">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">Project 實踐</p>
                        <h3>代碼點樣串起成條鏈</h3>
                    </div>
                </div>
                <p className="section-copy">三個 demo 共用同一套 GA engine；差喺 genome 解讀同 fitness 定義。訓練喺 Web Worker 入面跑，UI 只收 stats 同 champion replay。</p>
                <div className="pipeline-track">
                    {PIPELINE.map((node, index) => (
                        <React.Fragment key={node.title}>
                            <div className="pipeline-node">
                                <div className="pipeline-node-index">{String(index + 1).padStart(2, "0")}</div>
                                <node.icon aria-hidden="true" size={18} strokeWidth={1.5} />
                                <strong>{node.title}</strong>
                                <span>{node.text}</span>
                            </div>
                            {index < PIPELINE.length - 1 ? <ArrowRight aria-hidden="true" className="pipeline-arrow" size={16} strokeWidth={1.5} /> : null}
                        </React.Fragment>
                    ))}
                </div>
                <div className="pipeline-code-hint">
                    <div>
                        <span>共用</span>
                        <code>src/lib/ga.ts</code>
                        <code>src/lib/neuralNetwork.ts</code>
                        <code>src/hooks/useEvolutionDemo.ts</code>
                    </div>
                    <div>
                        <span>Domain</span>
                        <code>domains/snake</code>
                        <code>domains/breaker</code>
                        <code>domains/stock</code>
                    </div>
                    <div>
                        <span>Workers</span>
                        <code>*.worker.ts</code>
                        <code>workerRuntime.ts</code>
                    </div>
                </div>
            </section>

            <section className="theory-demos">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">實驗列表</p>
                        <h3>同一 GA，唔同問題</h3>
                    </div>
                </div>
                <div className="demo-map-grid">
                    {DEMOS.map(demo => (
                        <Link className={`demo-map-card demo-map-${demo.accent}`} key={demo.path} to={demo.path}>
                            <div className="demo-map-head">
                                <div>
                                    <h4>{demo.title}</h4>
                                    <p className="demo-map-blurb">{demo.blurb}</p>
                                </div>
                                <span>{demo.topology}</span>
                            </div>
                            <dl>
                                <div>
                                    <dt>基因體</dt>
                                    <dd>{demo.genome}</dd>
                                </div>
                                <div>
                                    <dt>規模</dt>
                                    <dd>{demo.genes}</dd>
                                </div>
                                <div>
                                    <dt>適應度</dt>
                                    <dd>{demo.fitness}</dd>
                                </div>
                                <div>
                                    <dt>要點</dt>
                                    <dd>{demo.note}</dd>
                                </div>
                            </dl>
                            <div className="demo-map-foot">
                                <span className="demo-map-defaults">{demo.defaults}</span>
                                <span className="demo-map-cta">
                                    去實驗
                                    <ArrowRight size={14} strokeWidth={1.5} />
                                </span>
                            </div>
                        </Link>
                    ))}
                </div>
            </section>

            <section className="theory-limits">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">限制同心態</p>
                        <h3>學完記得帶住呢幾點</h3>
                    </div>
                </div>
                <div className="limits-grid">
                    <div>
                        <strong>唔保證最佳</strong>
                        <p>GA 係啟發式搜尋。同一 seed 可重播，換 seed 結果可以差好遠。</p>
                    </div>
                    <div>
                        <strong>Fitness 即政策</strong>
                        <p>你獎勵咩，AI 就鑽咩空子。Snake 餓死、Breaker 扣多餘接球，都係堵漏洞。</p>
                    </div>
                    <div>
                        <strong>訓練 ≠ 實盤</strong>
                        <p>股票頁有手續費同 train/test 分割，但仍然係教學 demo，唔係投資建議。</p>
                    </div>
                    <div>
                        <strong>算力換樣本</strong>
                        <p>population × generations × 每場步數 = 真正成本。Worker 只係令畫面唔卡。</p>
                    </div>
                </div>
            </section>
        </div>
    );
});

interface GeneValue {
    value: number;
    source: "a" | "b";
    mutated: boolean;
}

interface GenomeRowProps {
    label: string;
    values: GeneValue[];
}

const GenomeRow = React.memo<GenomeRowProps>(({label, values}) => {
    return (
        <div className="genome-row">
            <span>{label}</span>
            <div>
                {values.map((gene, index) => (
                    <code className={getGeneClassName(gene)} key={`${label}-${index}`}>
                        {gene.value.toFixed(2)}
                    </code>
                ))}
            </div>
        </div>
    );
});

const STEP_PANELS = [
    {
        title: "族群",
        caption: "12 個候選（示意）",
        kind: "population" as const,
    },
    {
        title: "適應度",
        caption: "分數棒形圖",
        kind: "fitness" as const,
    },
    {
        title: "錦標賽",
        caption: "抽 3 個揀最強",
        kind: "tournament" as const,
    },
    {
        title: "交配",
        caption: "A / B 逐格混合",
        kind: "cross" as const,
    },
    {
        title: "突變",
        caption: "微調或整格重寫",
        kind: "mutate" as const,
    },
    {
        title: "菁英 + 移民",
        caption: "保送 + 新血",
        kind: "elite" as const,
    },
    {
        title: "下一代",
        caption: "循環再評估",
        kind: "next" as const,
    },
] as const;

const GenerationLoopDiagram = React.memo<{activeStep: number}>(({activeStep}) => {
    return (
        <div className="gen-loop" aria-hidden="true">
            <div className="gen-loop-panels">
                {STEP_PANELS.map((panel, index) => {
                    const isActive = index === activeStep;
                    const isLoopTarget = activeStep === 6 && index === 0;
                    return (
                        <div className={`gen-panel ${isActive ? "is-active" : ""} ${isLoopTarget ? "is-loop-target" : ""}`} key={panel.kind}>
                            <div className="gen-panel-head">
                                <span>{String(index + 1).padStart(2, "0")}</span>
                                <strong>{panel.title}</strong>
                            </div>
                            <StepMiniVisual kind={panel.kind} />
                            <small>{isLoopTarget ? "循環回到呢度" : panel.caption}</small>
                        </div>
                    );
                })}
            </div>
        </div>
    );
});

const StepMiniVisual = React.memo<{kind: (typeof STEP_PANELS)[number]["kind"]}>(({kind}) => {
    if (kind === "population") {
        return (
            <div className="mini-pop">
                {Array.from({length: 12}, (_, i) => (
                    <i className={i < 3 ? "hot" : undefined} key={i} />
                ))}
            </div>
        );
    }
    if (kind === "fitness") {
        return (
            <div className="mini-bars">
                <i style={{height: "78%"}} />
                <i style={{height: "55%"}} />
                <i style={{height: "40%"}} />
                <i style={{height: "22%"}} />
            </div>
        );
    }
    if (kind === "tournament") {
        return (
            <div className="mini-tourney">
                <i />
                <i className="winner" />
                <i />
            </div>
        );
    }
    if (kind === "cross") {
        return (
            <div className="mini-cross">
                <span className="a" />
                <span className="b" />
                <span className="mix">
                    <em className="a" />
                    <em className="b" />
                    <em className="a" />
                    <em className="b" />
                </span>
            </div>
        );
    }
    if (kind === "mutate") {
        return (
            <div className="mini-mutate">
                <i />
                <i className="hit" />
                <i />
                <i className="hit reset" />
            </div>
        );
    }
    if (kind === "elite") {
        return (
            <div className="mini-elite">
                <i className="elite" />
                <i className="elite" />
                <i />
                <i className="immigrant" />
            </div>
        );
    }
    return (
        <div className="mini-next">
            <i />
            <ArrowRight size={14} strokeWidth={1.5} />
            <i className="next" />
        </div>
    );
});

type ConceptVisual = (typeof CONCEPTS)[number]["visual"];

const ConceptGlyph = React.memo<{kind: ConceptVisual}>(({kind}) => {
    return <div className={`concept-glyph concept-glyph--${kind}`} aria-hidden="true" />;
});

function getGeneClassName(gene: GeneValue): string {
    return ["gene", `gene-${gene.source}`, gene.mutated ? "mutated" : null].filter(Boolean).join(" ");
}

function createChild(mutationRate: number, seed: number): GeneValue[] {
    const random = createRandom(seed);
    return PARENT_A.map((gene, index) => {
        const source = random.next() < 0.5 ? "a" : "b";
        const inherited = source === "a" ? gene : PARENT_B[index];
        const mutated = random.next() < mutationRate;
        return {value: inherited + (mutated ? random.gaussian() * 0.22 : 0), source, mutated};
    });
}
