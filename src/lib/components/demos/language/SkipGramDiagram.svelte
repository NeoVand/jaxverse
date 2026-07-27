<script lang="ts">
	// Skip-gram, drawn once and held still: the sentence window that makes the
	// training pairs, the one-hot column that a word starts as, the table row
	// it trades that for, and the push–pull rule whose loss trains the space.
	// Static by design — the plate below runs it live; this is the shape the
	// reader keeps in mind while it does.

	// ── the sentence and its window ──
	const WORDS = [
		{ t: 'the', c: 46 },
		{ t: 'little', c: 89 },
		{ t: 'dog', c: 136, centre: true },
		{ t: 'wagged', c: 186, ctx: true },
		{ t: 'his', c: 233, ctx: true },
		{ t: 'tail', c: 268 }
	];
	const SENT_Y = 30;

	// ── one-hot → table row → dense ──
	const OH = { x: 36, y: 78, w: 16, h: 108, cells: 9, hot: 3 };
	const TB = { x: 106, y: 78, w: 74, h: 108, rows: 12 };
	const HOT_ROW = 3; // the row the one-hot picks
	const ROWY = TB.y + HOT_ROW * (TB.h / TB.rows);
	// the table filled as texture — the same deterministic value noise the
	// walkthrough's MatrixGlyph paints, so "a matrix of learned numbers" reads
	// the same everywhere in the chapter
	const TB_COLS = 8;
	const TB_CW = (TB.w - 4) / TB_COLS;
	const TB_CH = (TB.h - 4) / TB.rows;
	const TB_CELLS = Array.from({ length: TB.rows * TB_COLS }, (_, k) => {
		const j = Math.floor(k / TB_COLS);
		const i = k % TB_COLS;
		const t = Math.abs(Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233 + 7 * 3.71) * 43758.5453) % 1;
		return { i, j, a: 0.1 + t * 0.55 };
	});
	const DN = { x: 216, y0: 118, mid: 132 }; // dense bars sit on this zero line
	// the same signed bars the plate's rail draws — hand-set, illustrative
	const DENSE = [
		0.7, -0.4, 0.55, 0.2, -0.8, 0.35, -0.25, 0.6, -0.5, 0.3, 0.75, -0.35, 0.45, -0.6, 0.25, -0.45
	];

	// ── the rule: two pairs, one pulled, one pushed ──
	const R1 = 76; // the real neighbor's row
	const R2 = 150; // the random word's row
	const cell = (v: number) => Math.max(0.15, Math.min(0.95, Math.abs(v)));
	const VDOG = [0.7, -0.4, 0.55, -0.8];
	const VWAG = [0.6, -0.3, 0.5, -0.7];
	const VSPO = [-0.5, 0.65, -0.2, 0.4];
</script>

<figure class="my-8">
	<svg
		viewBox="0 0 660 232"
		class="mx-auto block w-full max-w-[760px]"
		role="img"
		aria-label="Skip-gram in one picture. Left: in the sentence 'the little dog wagged his tail', a window around the centre word 'dog' marks its neighbors; every centre–neighbor pair is one lesson. Below, the word 'dog' begins as a one-hot column — six hundred slots, a single 1, which says nothing about meaning — and trades it for row 41 of a 600-by-16 embedding table: sixteen learned numbers, the dense vector the plate scatters. Right: the training rule. The dot product of dog's vector with a real neighbor like 'wagged' is pushed through a sigmoid toward 1, pulling the two vectors together; the dot product with a randomly drawn word like 'spoon' is pushed toward 0, pushing those vectors apart. The loss is the surprise at the real neighbor plus the surprise whenever a random word scores."
	>
		<defs>
			<marker
				id="sg-arrow"
				viewBox="0 0 8 8"
				refX="6.5"
				refY="4"
				markerWidth="6"
				markerHeight="6"
				orient="auto-start-reverse"
			>
				<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--ink-3)" />
			</marker>
		</defs>

		<!-- ── the sentence: centre word, window, lessons ── -->
		{#each WORDS as w (w.t)}
			{#if w.centre}
				<rect
					x={w.c - 16}
					y={SENT_Y - 12}
					width="32"
					height="17"
					rx="8.5"
					fill="var(--accent-soft)"
					stroke="var(--accent)"
				/>
			{/if}
			<text
				x={w.c}
				y={SENT_Y}
				text-anchor="middle"
				class="tok"
				class:tok-centre={w.centre}
				class:tok-ctx={w.ctx}>{w.t}</text
			>
		{/each}
		<!-- lessons: arcs from the centre to each word inside the window -->
		{#each [89, 186, 233] as cx (cx)}
			<path
				d="M 136 {SENT_Y - 15} C 136 {SENT_Y - 27}, {cx} {SENT_Y - 27}, {cx} {SENT_Y - 13}"
				fill="none"
				stroke="var(--accent)"
				stroke-width="1"
				opacity="0.55"
				marker-end="url(#sg-arrow)"
			/>
		{/each}
		<!-- the window bracket -->
		<path
			d="M 72 {SENT_Y + 8} L 72 {SENT_Y + 13} L 248 {SENT_Y + 13} L 248 {SENT_Y + 8}"
			fill="none"
			stroke="var(--ink-3)"
			stroke-width="1"
		/>
		<text x="160" y={SENT_Y + 25} text-anchor="middle" class="cap dim">
			the window · every centre–neighbor pair is one lesson
		</text>

		<!-- ── one-hot: 600 slots, a single 1 ── -->
		<rect
			x={OH.x}
			y={OH.y}
			width={OH.w}
			height={OH.h}
			rx="3"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		{#each Array.from({ length: OH.cells }, (_, k) => k) as k (k)}
			{@const cy = OH.y + 6 + k * ((OH.h - 12) / (OH.cells - 1))}
			{#if k === OH.hot}
				<rect x={OH.x + 3} y={cy - 4} width={OH.w - 6} height="8" rx="2" fill="var(--accent)" />
			{:else if k === 6}
				<text x={OH.x + OH.w / 2} y={cy + 3} text-anchor="middle" class="cap dim">⋮</text>
			{:else}
				<rect
					x={OH.x + 3}
					y={cy - 4}
					width={OH.w - 6}
					height="8"
					rx="2"
					fill="none"
					stroke="var(--line)"
					stroke-width="0.8"
				/>
			{/if}
		{/each}
		<text x={OH.x + OH.w / 2} y={OH.y - 8} text-anchor="middle" class="label">one-hot</text>
		<text x={OH.x + OH.w / 2} y={OH.y + OH.h + 13} text-anchor="middle" class="cap">600 slots</text>
		<text x={OH.x + OH.w / 2} y={OH.y + OH.h + 24} text-anchor="middle" class="cap dim">
			a single 1
		</text>
		<text x={OH.x + OH.w / 2} y={OH.y + OH.h + 35} text-anchor="middle" class="cap dim">
			says nothing
		</text>

		<line
			x1={OH.x + OH.w + 6}
			y1={OH.y + OH.h / 2}
			x2={TB.x - 8}
			y2={OH.y + OH.h / 2}
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#sg-arrow)"
		/>

		<!-- ── the embedding table: the one-hot picks a row ── -->
		<rect
			x={TB.x}
			y={TB.y}
			width={TB.w}
			height={TB.h}
			rx="3"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		{#each TB_CELLS as c (c.j * TB_COLS + c.i)}
			<rect
				x={TB.x + 2 + c.i * TB_CW + 0.5}
				y={TB.y + 2 + c.j * TB_CH + 0.5}
				width={TB_CW - 1}
				height={TB_CH - 1}
				rx="1"
				fill={c.j === HOT_ROW ? 'var(--accent)' : 'var(--ink)'}
				opacity={c.j === HOT_ROW ? Math.min(0.95, 0.35 + c.a) : c.a}
			/>
		{/each}
		<rect
			x={TB.x + 1.5}
			y={ROWY + 1}
			width={TB.w - 3}
			height={TB.h / TB.rows - 2}
			rx="1.5"
			fill="none"
			stroke="var(--accent)"
			stroke-width="1"
		/>
		<text x={TB.x + TB.w / 2} y={TB.y - 8} text-anchor="middle" class="label">the table</text>
		<text x={TB.x + TB.w / 2} y={TB.y + TB.h + 13} text-anchor="middle" class="cap">600 × 16</text>
		<text x={TB.x + TB.w / 2} y={TB.y + TB.h + 24} text-anchor="middle" class="cap dim">
			all of it learned
		</text>

		<line
			x1={TB.x + TB.w + 4}
			y1={ROWY + TB.h / TB.rows / 2}
			x2={DN.x - 8}
			y2={DN.mid}
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#sg-arrow)"
		/>

		<!-- ── dense: the row itself, sixteen learned numbers ── -->
		<line
			x1={DN.x}
			y1={DN.mid}
			x2={DN.x + 16 * 5}
			y2={DN.mid}
			stroke="var(--line)"
			stroke-width="1"
		/>
		{#each DENSE as v, k (k)}
			{@const h = Math.abs(v) * 22}
			<rect
				x={DN.x + k * 5 + 1}
				y={v >= 0 ? DN.mid - h : DN.mid}
				width="3"
				height={Math.max(1, h)}
				rx="0.8"
				fill={v >= 0 ? 'var(--warm)' : 'var(--accent)'}
				opacity="0.85"
			/>
		{/each}
		<text x={DN.x + 40} y={DN.y0 - 22} text-anchor="middle" class="label">dense · v(dog)</text>
		<text x={DN.x + 40} y={DN.mid + 42} text-anchor="middle" class="cap">16 learned numbers</text>
		<text x={DN.x + 40} y={DN.mid + 53} text-anchor="middle" class="cap dim">
			the point on the stage
		</text>

		<!-- the fold -->
		<line x1="330" y1="14" x2="330" y2="218" stroke="var(--line-soft)" stroke-width="1" />

		<!-- ── the rule: one real neighbor pulled in, one random word pushed out ── -->
		<text x="495" y="26" text-anchor="middle" class="label">the rule, one pair at a time</text>

		{#each [{ y: R1, u: VWAG, word: 'wagged', target: '→ 1', deed: 'a real neighbor · pull together', good: true }, { y: R2, u: VSPO, word: 'spoon', target: '→ 0', deed: 'one of 5 random words · push apart', good: false }] as row (row.y)}
			<!-- v(dog) -->
			{#each VDOG as v, k (k)}
				<rect
					x={400 + k * 9}
					y={row.y - 7}
					width="7"
					height="14"
					rx="1.5"
					fill="var(--accent)"
					opacity={cell(v)}
				/>
			{/each}
			<text x={400 + 17} y={row.y + 21} text-anchor="middle" class="cap dim">v(dog)</text>
			<text x="446" y={row.y + 4} text-anchor="middle" class="op">·</text>
			<!-- u(other) -->
			{#each row.u as v, k (k)}
				<rect
					x={458 + k * 9}
					y={row.y - 7}
					width="7"
					height="14"
					rx="1.5"
					fill={row.good ? 'var(--warm)' : 'var(--ink-3)'}
					opacity={cell(v)}
				/>
			{/each}
			<text x={458 + 17} y={row.y + 21} text-anchor="middle" class="cap dim">u({row.word})</text>
			<line
				x1="502"
				y1={row.y}
				x2="524"
				y2={row.y}
				stroke="var(--ink-3)"
				stroke-width="1"
				marker-end="url(#sg-arrow)"
			/>
			<rect
				x="530"
				y={row.y - 11}
				width="26"
				height="22"
				rx="11"
				fill="var(--surface)"
				stroke="var(--cat-2)"
			/>
			<text x="543" y={row.y + 3.5} text-anchor="middle" class="pill">σ</text>
			<text
				x="564"
				y={row.y + 4}
				class="verdict"
				style="fill: {row.good ? 'var(--warm)' : 'var(--ink-2)'};">{row.target}</text
			>
			<text x="492" y={row.y + 35} text-anchor="middle" class="cap dim">{row.deed}</text>
		{/each}

		<!-- the loss, written the way the meter reads it -->
		<text x="495" y="205" text-anchor="middle" class="cap">
			the loss · −log σ(v·u₊) − Σ log(1 − σ(v·u₋))
		</text>
		<text x="495" y="217" text-anchor="middle" class="cap dim">
			surprise at the real neighbor, plus whenever a random word scores
		</text>
	</svg>
	<figcaption
		class="mx-auto mt-2 max-w-[600px] text-center font-serif text-[13.5px] text-ink-2 italic"
		style="font-variation-settings: 'opsz' 13;"
	>
		A word arrives as a one-hot column — six hundred slots and a single 1, a representation that
		says nothing about meaning — and leaves as its row of the table: sixteen dense numbers, all
		learned. The only pressure on that row is the rule on the right, applied a million times; the
		geometry in the plate below is what the pressure leaves behind.
	</figcaption>
</figure>

<style>
	.tok {
		font-family: var(--font-mono);
		font-size: 11px;
		fill: var(--ink-2);
	}
	.tok-centre {
		fill: var(--accent);
		font-weight: 600;
	}
	.tok-ctx {
		fill: var(--ink);
	}
	.label {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 12.5px;
		fill: var(--ink-2);
	}
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-2);
	}
	.dim {
		fill: var(--ink-3);
	}
	.op {
		font-family: var(--font-serif);
		font-size: 16px;
		fill: var(--ink);
	}
	.pill {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 13px;
		fill: var(--cat-2);
	}
	.verdict {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
	}
</style>
