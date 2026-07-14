import React from "react";
import {Button, Tooltip} from "@heroui/react";
import {Download, Info, Upload} from "lucide-react";
import {buildGenomeFile, defaultGenomeFilename, downloadJsonFile, parseGenomeFile, readFileText, type GenomeFileTopic} from "../lib/genomeIO";
import type {Genome, NetworkTopology} from "../lib/types";

interface GenomeTransferProps {
    topic: GenomeFileTopic;
    topology: NetworkTopology;
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
            props.onMessage?.({type: "error", text: "未有 champion weights 可以 export。"});
            return;
        }
        try {
            const file = buildGenomeFile({
                topic: props.topic,
                topology: props.topology,
                genome: props.genome,
                fitness: props.fitness,
                score: props.score,
                steps: props.steps,
            });
            downloadJsonFile(defaultGenomeFilename(props.topic, props.score), file);
            props.onMessage?.({type: "status", text: "已 export champion weights。"});
        } catch (error) {
            props.onMessage?.({type: "error", text: error instanceof Error ? error.message : "Export 失敗。"});
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
            const genome = parseGenomeFile(raw, {topic: props.topic, topology: props.topology});
            props.onImport(genome);
            props.onMessage?.({type: "status", text: `已 import ${file.name}（${genome.length} genes）。`});
        } catch (error) {
            props.onMessage?.({type: "error", text: error instanceof Error ? error.message : "Import 失敗。"});
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="genome-transfer">
            <div className="control-label">
                <span>Weights I/O</span>
                <Tooltip delay={250}>
                    <span aria-label="Weights I/O 說明" className="help-icon" role="button" tabIndex={0}>
                        <Info size={13} strokeWidth={1.5} />
                    </span>
                    <Tooltip.Content className="max-w-60 text-xs" showArrow>
                        Export / import champion network 嘅扁平 weights 同 biases（JSON）。Import 後會即刻 replay，再撳「開始」會用呢個 genome 做種子繼續演化。
                    </Tooltip.Content>
                </Tooltip>
            </div>
            <div className="genome-transfer-actions">
                <Button isDisabled={props.disabled || !props.genome?.length || busy} onPress={handleExport} size="sm" variant="secondary">
                    <Download size={15} strokeWidth={1.5} />
                    Export
                </Button>
                <Button isDisabled={props.disabled || busy} onPress={handleImportClick} size="sm" variant="secondary">
                    <Upload size={15} strokeWidth={1.5} />
                    Import
                </Button>
            </div>
            <input accept="application/json,.json" aria-label="Import genome JSON" className="genome-file-input" onChange={handleFileChange} ref={fileInputRef} type="file" />
        </div>
    );
});
