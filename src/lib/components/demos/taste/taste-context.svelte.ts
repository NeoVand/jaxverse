// The shared bench for the preference chapter.
//
// One judge threads through the whole page: the model the comparator plate
// fits to your clicks is the same model the over-optimization plate attacks
// and the leash plate restrains. Module-level, like the Rook chapter's engine,
// because four plates on one page must see one judge or the argument falls
// apart.
//
// Reactive fields are scalars only. The judge itself is a plain property —
// Float64Arrays have no business inside a deep proxy — so plates watch the
// `version` counter to know when to redraw. Same contract the Rook lab keeps
// with its weight buffers.

import { mulberry32, type Rand } from '$lib/optim-rl/rng';
import {
	createJudge,
	fitJudge,
	preferProb,
	sampleRefGene,
	type Gene,
	type Judge,
	type Pair
} from '$lib/optim-rl/preference';

const SEED = 20_240;
/** How many reference draws the judge standardizes its scores against. */
const REF_POOL = 512;
/** Enough comparisons for the judge to be worth fitting at all. */
export const MIN_PAIRS = 6;
/** The comparator's suggested stopping point — not a limit. */
export const TARGET_PAIRS = 24;

class TasteLab {
	/** Bumped whenever the judge's weights change; plates redraw off this. */
	version = $state(0);
	/** How many verdicts the reader has recorded. */
	count = $state(0);
	/** How many pairs they declined to call. */
	skipped = $state(0);
	/** Mean pairwise loss on the pile the judge was fitted to. */
	loss = $state(Math.LN2);
	/** Share of the reader's own verdicts the judge now reproduces. */
	fit = $state(0.5);
	/**
	 * The honest one. Before each click, the judge — fitted only to everything
	 * that came before — is asked to call the pair. This records whether it
	 * was right. No train/test split needed and none possible at this size:
	 * every judgment is a test case exactly once, on its way to becoming
	 * training data. Statisticians call it prequential validation.
	 */
	calls = $state<boolean[]>([]);

	readonly foresight = $derived(
		this.calls.length === 0
			? null
			: this.calls.reduce((a, b) => a + (b ? 1 : 0), 0) / this.calls.length
	);
	readonly ready = $derived(this.count >= MIN_PAIRS);

	// ── plain, opaque, never templated ──
	judge: Judge;
	pairs: Pair[] = [];
	refPool: Gene[] = [];
	rand: Rand;

	constructor() {
		this.rand = mulberry32(SEED);
		this.judge = createJudge(mulberry32(SEED + 1));
		this.refPool = Array.from({ length: REF_POOL }, () => sampleRefGene(this.rand));
	}

	/** Two fresh candidates, drawn from the reference — never anything the
	 * judge chose, so the pile stays an honest sample of the space. */
	proposal(): [Gene, Gene] {
		return [sampleRefGene(this.rand), sampleRefGene(this.rand)];
	}

	/** Record a verdict, after first asking the judge to guess it. */
	record(winner: Gene, loser: Gene): void {
		if (this.count >= MIN_PAIRS) {
			this.calls = [...this.calls, preferProb(this.judge, winner, loser) > 0.5];
		}
		this.pairs = [...this.pairs, { winner, loser }];
		this.count = this.pairs.length;
		this.refit();
	}

	skip(): void {
		this.skipped += 1;
	}

	refit(): void {
		if (this.pairs.length === 0) return;
		const { loss, fit } = fitJudge(this.judge, this.pairs, { refSample: this.refPool });
		this.loss = loss;
		this.fit = fit;
		this.version += 1;
	}

	reset(): void {
		this.rand = mulberry32(SEED);
		this.judge = createJudge(mulberry32(SEED + 1));
		this.refPool = Array.from({ length: REF_POOL }, () => sampleRefGene(this.rand));
		this.pairs = [];
		this.calls = [];
		this.count = 0;
		this.skipped = 0;
		this.loss = Math.LN2;
		this.fit = 0.5;
		this.version += 1;
	}
}

export const taste = new TasteLab();
