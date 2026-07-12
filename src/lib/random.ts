export interface RandomSource {
    next(): number;
    integer(min: number, max: number): number;
    gaussian(): number;
}

export function createRandom(seed: number): RandomSource {
    let state = seed >>> 0;
    let spare: number | null = null;

    function next(): number {
        state += 0x6d2b79f5;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }

    function integer(min: number, max: number): number {
        return Math.floor(next() * (max - min + 1)) + min;
    }

    function gaussian(): number {
        if (spare !== null) {
            const value = spare;
            spare = null;
            return value;
        }

        const u = Math.max(next(), Number.EPSILON);
        const v = Math.max(next(), Number.EPSILON);
        const magnitude = Math.sqrt(-2 * Math.log(u));
        spare = magnitude * Math.sin(2 * Math.PI * v);
        return magnitude * Math.cos(2 * Math.PI * v);
    }

    return {next, integer, gaussian};
}
