import {calculateGeneCount} from "./neuralNetwork";
import type {Genome, NetworkTopology, TopicId} from "./types";

export const EVOLAB_GENOME_FORMAT = "evolab-genome" as const;
export const EVOLAB_GENOME_VERSION = 1 as const;

export type GenomeFileTopic = Exclude<TopicId, "theory">;

export interface EvolabGenomeFileV1 {
    format: typeof EVOLAB_GENOME_FORMAT;
    version: typeof EVOLAB_GENOME_VERSION;
    topic: GenomeFileTopic;
    topology: NetworkTopology;
    /**
     * Explicit length when the flat genome is longer than the NN topology alone
     * (e.g. stock = indicator period genes + decision-head weights).
     */
    geneCount?: number;
    genome: Genome;
    meta?: {
        fitness?: number;
        score?: number;
        steps?: number;
        exportedAt?: string;
    };
}

export interface GenomeExportOptions {
    topic: GenomeFileTopic;
    topology: NetworkTopology;
    genome: Genome;
    /** Override when genome includes non-NN genes (stock indicator periods). */
    geneCount?: number;
    fitness?: number;
    score?: number;
    steps?: number;
    exportedAt?: string;
}

export interface GenomeParseExpectation {
    topic: GenomeFileTopic;
    topology: NetworkTopology;
    /** Override when genome includes non-NN genes (stock indicator periods). */
    geneCount?: number;
}

/**
 * Build a versioned JSON payload for champion weights (biases included in the flat genome).
 */
export function buildGenomeFile(options: GenomeExportOptions): EvolabGenomeFileV1 {
    const geneCount = resolveGeneCount(options.topology, options.geneCount);
    if (options.genome.length !== geneCount) {
        throw new Error(`Genome length ${options.genome.length} does not match expected (${geneCount} genes)`);
    }

    return {
        format: EVOLAB_GENOME_FORMAT,
        version: EVOLAB_GENOME_VERSION,
        topic: options.topic,
        topology: {
            inputSize: options.topology.inputSize,
            hiddenLayers: [...options.topology.hiddenLayers],
            outputSize: options.topology.outputSize,
        },
        geneCount,
        genome: options.genome.map(value => Number(value)),
        meta: {
            fitness: options.fitness,
            score: options.score,
            steps: options.steps,
            exportedAt: options.exportedAt ?? new Date().toISOString(),
        },
    };
}

/**
 * Accept either the full Evolab envelope or a bare number[] for convenience.
 */
export function parseGenomeFile(raw: unknown, expected: GenomeParseExpectation): Genome {
    const geneCount = resolveGeneCount(expected.topology, expected.geneCount);

    if (Array.isArray(raw)) {
        return validateGenomeArray(raw, geneCount);
    }

    if (!raw || typeof raw !== "object") {
        throw new Error("檔案格式無效：需要 JSON object 或 number array。");
    }

    const file = raw as Partial<EvolabGenomeFileV1> & {weights?: unknown; genes?: unknown};

    // Bare `{ genome: number[] }` or legacy aliases.
    const candidate = file.genome ?? file.weights ?? file.genes;
    if (file.format === undefined && Array.isArray(candidate)) {
        return validateGenomeArray(candidate, geneCount);
    }

    if (file.format !== EVOLAB_GENOME_FORMAT) {
        throw new Error(`不支援嘅 format（需要 "${EVOLAB_GENOME_FORMAT}"）。`);
    }
    if (file.version !== EVOLAB_GENOME_VERSION) {
        throw new Error(`不支援嘅 version（需要 ${EVOLAB_GENOME_VERSION}）。`);
    }
    if (file.topic !== undefined && file.topic !== expected.topic && !topicsCompatible(file.topic, expected.topic)) {
        throw new Error(`呢個檔係 ${file.topic} lab 嘅 weights，唔適合 ${expected.topic}。`);
    }
    if (file.topology && !topologiesMatch(file.topology, expected.topology)) {
        throw new Error(`Topology 唔匹配：檔案 ${formatTopology(file.topology)}，而家 lab 係 ${formatTopology(expected.topology)}。`);
    }
    if (!Array.isArray(file.genome)) {
        throw new Error("檔案缺少 genome 陣列。");
    }

    return validateGenomeArray(file.genome, geneCount);
}

export function downloadJsonFile(filename: string, data: unknown): void {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
}

export function readFileText(file: File): Promise<string> {
    return file.text();
}

export function defaultGenomeFilename(topic: GenomeFileTopic, score?: number): string {
    const stamp = new Date()
        .toISOString()
        .slice(0, 19)
        .replace(/[:T]/g, m => (m === "T" ? "_" : m === ":" ? "" : "-"));
    const scorePart = score !== undefined ? `_score${score}` : "";
    return `evolab-${topic}${scorePart}_${stamp}.json`;
}

function resolveGeneCount(topology: NetworkTopology, geneCount?: number): number {
    return geneCount ?? calculateGeneCount(topology);
}

function validateGenomeArray(values: unknown[], geneCount: number): Genome {
    if (values.length !== geneCount) {
        throw new Error(`Genome 長度錯誤：需要 ${geneCount} 個 genes，檔案有 ${values.length} 個。`);
    }
    const genome: Genome = [];
    for (let index = 0; index < values.length; index += 1) {
        const value = values[index];
        if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new Error(`Genome[${index}] 唔係有效數字。`);
        }
        genome.push(value);
    }
    return genome;
}

/** Stock GA and 蒙地卡羅 labs share the same composite genome layout. */
function topicsCompatible(fileTopic: GenomeFileTopic, expectedTopic: GenomeFileTopic): boolean {
    const stockFamily = new Set<GenomeFileTopic>(["stock", "stock-mc"]);
    return stockFamily.has(fileTopic) && stockFamily.has(expectedTopic);
}

function topologiesMatch(a: NetworkTopology, b: NetworkTopology): boolean {
    if (a.inputSize !== b.inputSize || a.outputSize !== b.outputSize) {
        return false;
    }
    if (a.hiddenLayers.length !== b.hiddenLayers.length) {
        return false;
    }
    return a.hiddenLayers.every((size, index) => size === b.hiddenLayers[index]);
}

function formatTopology(topology: NetworkTopology): string {
    return [topology.inputSize, ...topology.hiddenLayers, topology.outputSize].join("→");
}
