import React from "react";
import {Button} from "@heroui/react";
import {Download, Upload} from "lucide-react";
import {buildGenomeFile, defaultGenomeFilename, downloadJsonFile, parseGenomeFile, readFileText, type GenomeFileTopic} from "../lib/genomeIO";
import type {Genome, NetworkTopology} from "../lib/types";

interface GenomeTransferProps {
    topic: GenomeFileTopic;
    topology: NetworkTopology;
    /**
     * Override expected genome length when the flat genome is longer than the NN
     * topology alone (stock = indicator periods + decision-head weights).
     */
    geneCount?: number;
    /** Current champion genome; export disabled when missing. */
    genome?: Genome | null;
    fitness?: number;
    score?: number;
    steps?: number;
    disabled?: boolean;
    onImport: (genome: Genome) => void;
    onMessage?: (message: {type: "status" | "error"; text: string} | null) => void;
}

export const GenomeTransfer = React.memo<GenomeTransferProps>(props => {
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const [busy, setBusy] = React.useState(false);

    const handleExport = () => {
        if (!props.genome?.length) {
            props.onMessage?.({type: "error", text: "未有冠軍權重可以匯出。"});
            return;
        }
        try {
            const file = buildGenomeFile({
                topic: props.topic,
                topology: props.topology,
                geneCount: props.geneCount,
                genome: props.genome,
                fitness: props.fitness,
                score: props.score,
                steps: props.steps,
            });
            downloadJsonFile(defaultGenomeFilename(props.topic, props.score), file);
            props.onMessage?.({type: "status", text: "已匯出冠軍權重。"});
        } catch (error) {
            props.onMessage?.({type: "error", text: error instanceof Error ? error.message : "匯出失敗。"});
        }
    };

    const handleImportClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        // Allow re-importing the same path.
        event.target.value = "";
        if (!file) {
            return;
        }

        setBusy(true);
        props.onMessage?.(null);
        try {
            const text = await readFileText(file);
            let raw: unknown;
            try {
                raw = JSON.parse(text) as unknown;
            } catch {
                throw new Error("JSON 解析失敗，請確認檔案內容。");
            }
            const genome = parseGenomeFile(raw, {topic: props.topic, topology: props.topology, geneCount: props.geneCount});
            props.onImport(genome);
            props.onMessage?.({type: "status", text: `已匯入 ${file.name}（${genome.length} 個基因）。`});
        } catch (error) {
            props.onMessage?.({type: "error", text: error instanceof Error ? error.message : "匯入失敗。"});
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="genome-transfer">
            <div className="genome-transfer-actions">
                <Button isDisabled={props.disabled || !props.genome?.length || busy} onPress={handleExport} size="sm" variant="secondary">
                    <Download size={15} strokeWidth={1.5} />
                    匯出
                </Button>
                <Button isDisabled={props.disabled || busy} onPress={handleImportClick} size="sm" variant="secondary">
                    <Upload size={15} strokeWidth={1.5} />
                    匯入
                </Button>
            </div>
            <input accept="application/json,.json" aria-label="匯入基因體 JSON" className="genome-file-input" onChange={handleFileChange} ref={fileInputRef} type="file" />
        </div>
    );
});
