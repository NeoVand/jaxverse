// Deterministic randomness for the reward chapter: same seed, same story —
// demos replay identically and the unit tests never flake.

export type Rand = () => number;

/** mulberry32 — tiny, fast, plenty for sampling actions and coin flips. */
export function mulberry32(seed: number): Rand {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
