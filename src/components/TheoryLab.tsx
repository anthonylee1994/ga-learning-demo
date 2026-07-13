import React from "react";
import {Button, Card, Chip} from "@heroui/react";
import {ArrowRight, Atom, Binary, Braces, Dna, FlaskConical, Network, RefreshCw, ShieldCheck, Shuffle, Sparkles, Target, TrendingUp, Users} from "lucide-react";
import {createRandom} from "../lib/random";

const THEORY_ITEMS = [
    {icon: Dna, title: "Genome / Chromosome", text: "每個 genome 都係一組 Brain.js weights 同 biases，完整描述一個 AI 嘅決策方式。"},
    {icon: Users, title: "Population", text: "同一代同時測試多個不同 neural networks，唔將希望押喺單一解法。"},
    {icon: Target, title: "Fitness Function", text: "將生存、回報或任務表現壓縮成分數，決定邊啲個體值得繁殖。"},
    {icon: ShieldCheck, title: "Selection + Elitism", text: "Roulette wheel selection 按 fitness 比例增加強者繁殖機會；elitism 直接保留最優個體，避免倒退。"},
    {icon: Shuffle, title: "Crossover", text: "兩個 parents 逐個 gene 混合，child 同時繼承兩邊已經有效嘅結構。"},
    {icon: Sparkles, title: "Mutation", text: "以低機率擾動 weights，為 population 注入新方向，跳出局部最佳。"},
    {icon: Network, title: "Neuroevolution", text: "唔用 backpropagation；直接以演化搜尋 neural network parameters。"},
    {icon: TrendingUp, title: "Exploration vs Exploitation", text: "mutation 太低會早熟收斂，太高就會不停破壞已有成果。"},
    {icon: Braces, title: "Overfitting", text: "高 training fitness 唔等於識應付新環境；股票實驗會保留 20% unseen data。"},
    {icon: Atom, title: "Stochastic Search", text: "GA 依賴隨機抽樣。Seed 可重播實驗，但演算法唔保證搵到全局最佳。"},
    {icon: Binary, title: "Fitness Bias", text: "AI 只會優化你寫落去嘅分數。錯嘅 fitness 會穩定地學出錯嘅行為。"},
    {icon: FlaskConical, title: "計算限制", text: "每一代都要評估整個 population；更大樣本通常更穩，但亦更慢。"},
] as const;

const FLOW = ["初始化 Population", "評估 Fitness", "Selection", "Crossover", "Mutation", "Elitism", "下一代"];
const PARENT_A = [0.82, -0.31, 0.14, 0.67, -0.74, 0.26, 0.51, -0.08];
const PARENT_B = [-0.44, 0.76, 0.39, -0.22, 0.91, -0.58, 0.05, 0.63];

export const TheoryLab = React.memo(() => {
    const [mutationRate, setMutationRate] = React.useState(0.18);
    const [seed, setSeed] = React.useState(42);
    const child = React.useMemo(() => createChild(mutationRate, seed), [mutationRate, seed]);

    return (
        <div className="theory-view">
            <section className="theory-intro">
                <div>
                    <p className="eyebrow">Genetic algorithm fundamentals</p>
                    <h2>用演化，搜尋一個夠好嘅決策腦</h2>
                    <p>Genetic Algorithm 將候選解當成生物個體：評分、選擇、繁殖、突變，再重複好多代。 呢個係 stochastic search，唔保證每次一樣，亦唔保證搵到全局最佳。</p>
                </div>
                <div className="intro-stat">
                    <span>核心循環</span>
                    <strong>Evaluate → Evolve</strong>
                    <small>Brain.js 負責推理，GA 負責改寫 weights。</small>
                </div>
            </section>

            <section className="flow-section">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">Generation loop</p>
                        <h3>一代係點樣誕生</h3>
                    </div>
                    <Chip color="accent" size="sm" variant="soft">
                        重複直到暫停
                    </Chip>
                </div>
                <div className="flow-track">
                    {FLOW.map((stage, index) => (
                        <React.Fragment key={stage}>
                            <div className="flow-stage">
                                <span>{String(index + 1).padStart(2, "0")}</span>
                                {stage}
                            </div>
                            {index < FLOW.length - 1 ? <ArrowRight aria-hidden="true" size={16} strokeWidth={1.5} /> : null}
                        </React.Fragment>
                    ))}
                </div>
            </section>

            <section className="theory-grid">
                {THEORY_ITEMS.map(item => (
                    <Card className="theory-card rounded-lg" key={item.title} variant="default">
                        <item.icon aria-hidden="true" size={20} strokeWidth={1.5} />
                        <Card.Header>
                            <Card.Title className="text-sm">{item.title}</Card.Title>
                        </Card.Header>
                        <Card.Content>
                            <p>{item.text}</p>
                        </Card.Content>
                    </Card>
                ))}
            </section>

            <section className="genome-lab">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">Interactive genome lab</p>
                        <h3>Crossover + Mutation</h3>
                    </div>
                    <Button isIconOnly onPress={() => setSeed(value => value + 1)} variant="tertiary">
                        <RefreshCw aria-label="重新生成 child" size={16} strokeWidth={1.5} />
                    </Button>
                </div>
                <p className="section-copy">每一格係一個 network weight。Child 先從兩個 parents 繼承，再按 mutation rate 加入細小擾動。</p>
                <div className="genome-rows">
                    <GenomeRow label="Parent A" values={PARENT_A.map(value => ({value, source: "a" as const, mutated: false}))} />
                    <GenomeRow label="Parent B" values={PARENT_B.map(value => ({value, source: "b" as const, mutated: false}))} />
                    <GenomeRow label="Child" values={child} />
                </div>
                <label className="mutation-control">
                    <span>
                        <span>Mutation rate</span>
                        <strong>{Math.round(mutationRate * 100)}%</strong>
                    </span>
                    <input
                        aria-label="Theory mutation rate"
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
                        Parent A
                    </span>
                    <span>
                        <i className="gene-b" />
                        Parent B
                    </span>
                    <span>
                        <i className="gene-mutated" />
                        已突變
                    </span>
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
