import React from "react";
import {Button, Card, Chip} from "@heroui/react";
import {ArrowRight, Atom, Binary, Braces, Dna, FlaskConical, Network, RefreshCw, ShieldCheck, Shuffle, Sparkles, Target, TrendingUp, Users} from "lucide-react";
import {createRandom} from "../lib/random";

const THEORY_ITEMS = [
    {icon: Dna, title: "基因體 / 染色體", text: "每個基因體都係一組 Brain.js 權重同偏差，完整描述一個 AI 嘅決策方式。"},
    {icon: Users, title: "族群", text: "同一代同時測試多個唔同神經網絡，唔將希望押喺單一解法。"},
    {icon: Target, title: "適應度函數", text: "將生存、回報或任務表現壓縮成分數，決定邊啲個體值得繁殖。"},
    {icon: ShieldCheck, title: "選擇 + 菁英保留", text: "輪盤選擇按適應度比例增加強者繁殖機會；菁英保留直接保留最優個體，避免倒退。"},
    {icon: Shuffle, title: "交配", text: "兩個父母逐個基因混合，子女同時繼承兩邊已經有效嘅結構。"},
    {icon: Sparkles, title: "突變", text: "以低機率擾動權重，為族群注入新方向，跳出局部最佳。"},
    {icon: Network, title: "神經演化", text: "唔用反向傳播；直接以演化搜尋神經網絡參數。"},
    {icon: TrendingUp, title: "探索 vs 利用", text: "突變太低會早熟收斂，太高就會不停破壞已有成果。"},
    {icon: Braces, title: "過擬合", text: "高訓練適應度唔等於識應付新環境；股票實驗會保留 20% 未見過數據。"},
    {icon: Atom, title: "隨機搜尋", text: "遺傳演算法依賴隨機抽樣。種子可重播實驗，但演算法唔保證搵到全局最佳。"},
    {icon: Binary, title: "適應度偏差", text: "AI 只會優化你寫落去嘅分數。錯嘅適應度會穩定地學出錯嘅行為。"},
    {icon: FlaskConical, title: "計算限制", text: "每一代都要評估成個族群；更大樣本通常更穩，但亦更慢。"},
] as const;

const FLOW = ["初始化族群", "評估適應度", "選擇", "交配", "突變", "菁英保留", "下一代"];
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
                    <p className="eyebrow">遺傳演算法基礎</p>
                    <h2>用演化，搜尋一個夠好嘅決策腦</h2>
                    <p>遺傳演算法將候選解當成生物個體：評分、選擇、繁殖、突變，再重複好多代。呢個係隨機搜尋，唔保證每次一樣，亦唔保證搵到全局最佳。</p>
                </div>
                <div className="intro-stat">
                    <span>核心循環</span>
                    <strong>評估 → 演化</strong>
                    <small>Brain.js 負責推理，遺傳演算法負責改寫權重。</small>
                </div>
            </section>

            <section className="flow-section">
                <div className="panel-heading">
                    <div>
                        <p className="eyebrow">世代循環</p>
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
                        <p className="eyebrow">互動基因實驗室</p>
                        <h3>交配 + 突變</h3>
                    </div>
                    <Button isIconOnly onPress={() => setSeed(value => value + 1)} variant="tertiary">
                        <RefreshCw aria-label="重新生成子女" size={16} strokeWidth={1.5} />
                    </Button>
                </div>
                <p className="section-copy">每一格係一個網絡權重。子女先從兩個父母繼承，再按突變率加入細小擾動。</p>
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
