import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

// Numbers the prose states out loud, checked against the files they describe.
//
// Every defect this book has shipped in its writing has been of one kind: a
// sentence that was true when it was written and then stopped being true,
// because the thing it described moved. A caption said ten thousand where its
// own eyebrow said 8,000; a chart drew a baseline four points below the rate
// its own constant held; a chapter said close-hauled was the fastest point of
// sail while its polar said otherwise. None of those are hard to catch. They
// are only hard to REMEMBER to catch, which is what a test is for.
//
// So: a claim is a phrase that must appear verbatim in a chapter, and a
// function that recomputes the number from the source of truth. If the data is
// regenerated and a count shifts, the sentence quoting it fails here rather
// than going quietly wrong on the page.

const page = (slug: string) => readFileSync(`src/routes/${slug}/+page.svelte`, 'utf8');
const bin = (name: string) => {
	const b = readFileSync(`static/data/${name}`);
	return new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2);
};
const json = (name: string) => JSON.parse(readFileSync(`static/data/${name}`, 'utf8'));

/** The corpus as text, normalised the way buildWordCorpus normalises it. */
function corpusWords(): string[] {
	const v = json('text-vocab.json') as { chars: string[] };
	const t = readFileSync('static/data/text-tokens.bin');
	const s = Array.from(t, (i) => v.chars[i]).join('');
	return s
		.toLowerCase()
		.replace(/[^a-z\s]/g, ' ')
		.split(/\s+/)
		.filter(Boolean);
}

/** Games in a token stream: token 0 is the <game> marker that opens each one. */
const games = (name: string) => bin(name).reduce((n, t) => n + (t === 0 ? 1 : 0), 0);

const CLAIMS: { slug: string; says: string; actual: () => number; is: number }[] = [
	// ── rook: the move vocabulary and the two corpora ──
	{
		slug: 'rook',
		says: 'only 1,930 distinct',
		actual: () => json('rook-vocab.json').moves.length,
		is: 1930
	},
	{
		slug: 'rook',
		says: 'vocabulary of 1,931',
		actual: () => json('rook-vocab.json').vocabSize,
		is: 1931
	},
	{
		slug: 'rook',
		says: 'some six thousand games',
		actual: () => games('rook-tokens.bin'),
		is: 6000
	},
	{
		slug: 'rook',
		says: 'we curate 2,381 games',
		actual: () => games('rook-sft-tokens.bin'),
		is: 2381
	},

	// ── digits: the split, which the chapter states three separate times ──
	{
		slug: 'digits',
		says: 'eight thousand of these teach',
		actual: () => json('mnist-meta.json').train,
		is: 8000
	},
	{ slug: 'digits', says: '8,000 form the', actual: () => json('mnist-meta.json').train, is: 8000 },
	{
		slug: 'digits',
		says: 'The 2,000 test digits',
		actual: () => json('mnist-meta.json').test,
		is: 2000
	},
	{
		slug: 'digits',
		says: 'ten thousand digits',
		actual: () => json('mnist-meta.json').train + json('mnist-meta.json').test,
		is: 10000
	},

	// ── language: the corpus, and one word counted by hand in the prose ──
	// one byte per character index — the character vocabulary is tiny, so the
	// stream is not the Uint16 one the chess corpus uses
	{
		slug: 'language',
		says: '1.5 million characters',
		actual: () => readFileSync('static/data/text-tokens.bin').byteLength,
		is: 1497429
	},
	{
		slug: 'language',
		says: 'nine times in three hundred thousand words',
		actual: () => corpusWords().filter((w) => w === 'king').length,
		is: 9
	}
];

describe('numbers the prose states out loud', () => {
	for (const c of CLAIMS) {
		it(`${c.slug}: “${c.says}”`, () => {
			expect(page(c.slug)).toContain(c.says);
			expect(c.actual()).toBe(c.is);
		});
	}
});

describe('the discount plate and the sentence describing it', () => {
	// The figure computes its own labels; the aria-label and the caption spell
	// them out for anyone who cannot see it. Those two got out of step the hour
	// the plate was written — the drawn label rounded to 30 and the sentence
	// said 31 — which is the whole reason this file exists.
	const plate = readFileSync('src/lib/components/demos/reward/Discount.svelte', 'utf8');
	const N = 40;
	const first = (g: number) => Math.pow(g, N - 1);

	it('the sailing chart still runs at the γ the plate calls its own', () => {
		expect(plate).toContain('0.97');
		const chart = readFileSync('src/lib/optim-rl/chart.ts', 'utf8');
		expect(Number(/gamma:\s*([\d.]+)/.exec(chart)?.[1])).toBe(0.97);
	});

	for (const [g, said] of [
		[0.9, '1.6 per cent'],
		[0.97, '30 per cent'],
		[0.99, '68 per cent']
	] as const) {
		it(`γ = ${g} gives the first of forty decisions ${said}`, () => {
			const v = first(g) * 100;
			const drawn = v < 10 ? v.toFixed(1) : String(Math.round(v));
			expect(said.startsWith(drawn)).toBe(true);
			expect(plate).toContain(said);
		});
	}
});

describe('the step-size plate and the sentence describing it', () => {
	// Four runners, and a caption that names each one's step size and the
	// factor it multiplies θ by. Both numbers are hand-typed in the prose and
	// computed nowhere, so changing a runner silently falsifies the sentence —
	// which is exactly what happened when 0.85 became 0.80 and the caption went
	// on saying 0.85 for a further ten minutes.
	const plate = readFileSync('src/lib/components/demos/descent/StepSize.svelte', 'utf8');
	const etas = [...plate.matchAll(/\{ eta: ([\d.]+), tone:/g)].map((m) => Number(m[1]));

	it('has the four runners the caption walks through', () => {
		expect(etas).toEqual([0.15, 0.45, 0.8, 1.05]);
	});

	it('every |1 − 2η| is distinct, or two runs draw one line', () => {
		const mags = etas.map((e) => Math.abs(1 - 2 * e).toFixed(2));
		expect(new Set(mags).size).toBe(etas.length);
	});

	// The caption promises a different sorting on each landscape. Both starting
	// points had to be moved to make those promises true — from −2.05, where
	// both bold steps were thrown out of the frame before either could reach
	// the far valley, and from −1.35, which divided exactly by two of the four
	// step sizes so those runners landed on the kink and stopped. Neither was
	// visible without running it, so it is run here.
	const outcome = (
		f: (t: number) => number,
		df: (t: number) => number,
		t0: number,
		xMax: number,
		eta: number
	) => {
		let t = t0;
		let crossed = false;
		const lim = xMax - 0.05;
		let least = Infinity;
		for (let k = 0; k < 30; k++) {
			if (Math.abs(t) > lim) return { end: 'diverged', least };
			const n = t - eta * df(t);
			if (t < 0 !== n < 0) crossed = true;
			t = n;
			least = Math.min(least, f(t));
		}
		return { end: crossed ? 'crossed' : 'stayed', least };
	};
	const QUAD = (e: number) =>
		outcome(
			(t) => t * t,
			(t) => 2 * t,
			-0.9,
			1.6,
			e
		);
	const WELL = (e: number) =>
		outcome(
			(t) => 0.3 * t ** 4 - t * t + 1 - (0.3 * (5 / 3) ** 2 - 5 / 3 + 1),
			(t) => 1.2 * t ** 3 - 2 * t,
			-1.7,
			2.2,
			e
		);
	const ABS = (e: number) => outcome(Math.abs, Math.sign, -1.31, 1.6, e);

	it('θ²: the two small steps close from one side, 0.80 overshoots, 1.05 leaves', () => {
		expect([0.15, 0.45].map((e) => QUAD(e).end)).toEqual(['stayed', 'stayed']);
		expect(QUAD(0.8).end).toBe('crossed');
		expect(QUAD(1.05).end).toBe('diverged');
	});

	it('the double well: 0.80 reaches the other valley and only 1.05 is lost', () => {
		expect([0.15, 0.45].map((e) => WELL(e).end)).toEqual(['stayed', 'stayed']);
		expect(WELL(0.8).end).toBe('crossed');
		expect(WELL(1.05).end).toBe('diverged');
	});

	it('|θ|: all four stride across the kink and none of them lands on it', () => {
		for (const e of [0.15, 0.45, 0.8, 1.05]) {
			expect(ABS(e).end).toBe('crossed');
			// a runner that hits the kink exactly stops dead, subgradient zero
			expect(ABS(e).least).toBeGreaterThan(1e-3);
		}
	});

	for (const e of [0.15, 0.45, 0.8, 1.05]) {
		it(`the caption's factor for η = ${e} is the one θ² actually applies`, () => {
			const f = 1 - 2 * e;
			// the prose writes a minus sign as U+2212, not a hyphen
			const said = `${f < 0 ? '\u2212' : ''}${Math.abs(f).toFixed(2)}`;
			expect(plate).toContain(said);
		});
	}
});

describe('rounded claims stay in the range that makes them fair', () => {
	it('“three hundred thousand words” is the corpus, to the nearest hundred thousand', () => {
		const n = corpusWords().length;
		expect(page('language')).toContain('three hundred thousand words');
		expect(Math.round(n / 1e5) * 1e5).toBe(300_000);
	});

	it('“a hundred thousand parameters” is the classifier at its default shape', () => {
		// 784 → width → … → 10, at the depth and width digits-context boots with
		const ctx = readFileSync('src/lib/components/demos/digits/digits-context.svelte.ts', 'utf8');
		const depth = Number(/depth = \$state\((\d+)\)/.exec(ctx)?.[1]);
		const width = Number(/width = \$state\((\d+)\)/.exec(ctx)?.[1]);
		const layers = [784, ...Array(depth).fill(width), 10];
		let p = 0;
		for (let i = 0; i < layers.length - 1; i++) p += layers[i] * layers[i + 1] + layers[i + 1];
		expect(page('digits')).toContain('a hundred thousand parameters');
		// "a hundred thousand" is fair for anything that rounds to it in units of 100k
		expect(Math.round(p / 1e5)).toBe(1);
	});
});
