<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	// Self-attention, drawn once and held still: four token vectors become
	// queries, keys and values through three learned lenses; every query is
	// scored against every key; the mask strikes out the future; a softmax
	// turns each surviving row into a budget; and the budget buys a blend of
	// the values. The rose outline follows the last token the whole way — its
	// question, its scores, its budget, its readout. Static by design: the
	// walkthrough plate below runs the same circuit with real numbers.

	// ── the tokens: X, one row each (same illustrative cells as the map) ──
	const TOKENS = ['once', '␣upon', '␣a', '␣time'];
	const ROW_Y = [64, 92, 120, 148];
	const LAST = ROW_Y.length - 1;
	const CELLS = [
		[0.85, 0.3, 0.55, 0.2],
		[0.35, 0.75, 0.25, 0.6],
		[0.2, 0.45, 0.8, 0.35],
		[0.6, 0.25, 0.4, 0.85]
	];
	const X0 = 68; // left edge of the X cells
	const XW = 13; // cell pitch

	// the MatrixGlyph value-noise hash — every learned matrix in the chapter
	// wears this texture
	const noise = (i: number, j: number, seed: number) =>
		Math.abs(Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233 + seed * 3.71) * 43758.5453) % 1;

	// ── the three lenses and their outputs ──
	const LENSES = [
		{ key: 'q', name: 'Q', cy: 52, color: 'var(--accent)', seed: 11 },
		{ key: 'k', name: 'K', cy: 106, color: 'var(--cat-8)', seed: 12 },
		{ key: 'v', name: 'V', cy: 160, color: 'var(--cat-6)', seed: 13 }
	];
	const WM = { x: 150, w: 36, h: 32, cols: 6, rows: 5 }; // a weight matrix
	const PM = { x: 214, w: 34, cols: 3 }; // a projected Q/K/V (4 rows, one per token)
	const PROW = 8; // row pitch inside a projection

	// ── scores and attention: two 4×4 grids around a softmax ──
	const G1 = { x: 300, y: 48 }; // raw scores Q·Kᵀ/√dk
	const G2 = { x: 424, y: 48 }; // after the mask and softmax
	const CELL = 14;
	// lower-triangular, illustrative; WEIGHTS rows sum to 1
	const SCORES = [[0.6], [0.8, 0.3], [0.2, 0.5, 0.45], [0.7, 0.25, 0.15, 0.4]];
	const WEIGHTS = [[1], [0.62, 0.38], [0.2, 0.45, 0.35], [0.45, 0.18, 0.12, 0.25]];

	// ── the readout ──
	const BLEND = { x: 512, y: 108 }; // where the budget meets the values
	const ZC = { x: 552, w: 16 }; // the context vector z
	const ZVALS = [0.7, 0.35, 0.85, 0.5];
</script>

<Plate
	id="attention"
	title="Self-attention, held still"
	caption="Follow the rose outline. The last token asks with its query row, is scored against every key, has its future struck out by the mask, receives a budget of exactly 1 from the softmax — and spends it on a blend of the values. That blend is all attention ever outputs."
>
	<svg
		viewBox="0 0 660 222"
		class="mx-auto block w-full max-w-[950px]"
		role="img"
		aria-label="Self-attention in one picture. Left: four tokens — once, upon, a, time — each a vector of numbers. Three learned weight matrices project every token three ways: into a query (ultramarine), a key (violet) and a value (blue-cyan). Centre: a four-by-four score table with query strips along its rows and key strips along its columns, so each cell is the dot product of one query with one key, scaled by the square root of the key dimension; every cell above the diagonal is struck out with negative infinity — the mask, no reading the future — and a softmax turns each surviving row into a budget that sums to one. Right: the last token's budget row is multiplied against the value vectors, producing its context vector z — a weighted blend of what earlier tokens passed along. A rose outline follows the last token through every stage."
	>
		<defs>
			<!-- one arrowhead per wire color, so heads never contradict their lines -->
			{#each [{ id: 'at-arrow', c: 'var(--ink-3)' }, { id: 'at-arrow-q', c: 'var(--accent)' }, { id: 'at-arrow-k', c: 'var(--cat-8)' }, { id: 'at-arrow-v', c: 'var(--cat-6)' }, { id: 'at-arrow-r', c: 'var(--cat-3)' }] as m (m.id)}
				<marker
					id={m.id}
					viewBox="0 0 8 8"
					refX="6.5"
					refY="4"
					markerWidth="6"
					markerHeight="6"
					orient="auto-start-reverse"
				>
					<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill={m.c} />
				</marker>
			{/each}
		</defs>

		<!-- ── the tokens: X ── -->
		<text x={X0 + 2 * XW - 1} y="36" text-anchor="middle" class="label">the tokens · X</text>
		{#each TOKENS as t, i (t)}
			<text x={X0 - 6} y={ROW_Y[i] + 3.5} text-anchor="end" class="tok">{t}</text>
			{#each CELLS[i] as a, ci (ci)}
				<rect
					x={X0 + ci * XW}
					y={ROW_Y[i] - 6}
					width={XW - 2}
					height="12"
					rx="1.5"
					fill="var(--accent)"
					opacity={a}
				/>
			{/each}
		{/each}
		<text x={X0 + 2 * XW - 1} y="168" text-anchor="middle" class="cap dim">vector + position</text>

		<!-- the fan-out: X feeds all three lenses -->
		<line x1={X0 + 4 * XW} y1="106" x2="132" y2="106" stroke="var(--ink-3)" stroke-width="1" />
		<circle cx="133" cy="106" r="2" fill="var(--ink-3)" />
		{#each LENSES as l (l.key)}
			<path
				d="M 133 106 C 139 106, 139 {l.cy}, {WM.x - 4} {l.cy}"
				fill="none"
				stroke="var(--ink-3)"
				stroke-width="1"
				marker-end="url(#at-arrow)"
			/>
		{/each}

		<!-- ── the lenses: W_Q, W_K, W_V, textured because learned ── -->
		{#each LENSES as l (l.key)}
			{@const y0 = l.cy - WM.h / 2}
			<rect
				x={WM.x}
				y={y0}
				width={WM.w}
				height={WM.h}
				rx="3"
				fill="var(--surface)"
				stroke="var(--line)"
			/>
			{#each Array.from({ length: WM.rows * WM.cols }, (_, k) => k) as k (k)}
				{@const j = Math.floor(k / WM.cols)}
				{@const i = k % WM.cols}
				<rect
					x={WM.x + 2 + i * ((WM.w - 4) / WM.cols) + 0.4}
					y={y0 + 2 + j * ((WM.h - 4) / WM.rows) + 0.4}
					width={(WM.w - 4) / WM.cols - 0.8}
					height={(WM.h - 4) / WM.rows - 0.8}
					rx="0.8"
					fill={l.color}
					opacity={0.15 + noise(i, j, l.seed) * 0.6}
				/>
			{/each}
			<text x={WM.x + WM.w / 2} y={y0 - 5} text-anchor="middle" class="wlbl" fill={l.color}>
				W<tspan dy="2.5" font-size="7.5">{l.name}</tspan>
			</text>
			<line
				x1={WM.x + WM.w + 4}
				y1={l.cy}
				x2={PM.x - 5}
				y2={l.cy}
				stroke="var(--ink-3)"
				stroke-width="1"
				marker-end="url(#at-arrow)"
			/>
		{/each}
		<text x={WM.x + WM.w / 2} y="188" text-anchor="middle" class="cap dim">
			three learned lenses
		</text>

		<!-- ── Q, K, V: one row per token, in the lens's color ── -->
		{#each LENSES as l, li (l.key)}
			{@const y0 = l.cy - WM.h / 2}
			<text x={PM.x + PM.w / 2} y={y0 - 5} text-anchor="middle" class="wlbl" fill={l.color}>
				{l.name}
			</text>
			{#each Array.from({ length: 4 }, (_, k) => k) as r (r)}
				{#each Array.from({ length: PM.cols }, (_, c) => c) as c (c)}
					<rect
						x={PM.x + 1 + c * (PM.w / PM.cols)}
						y={y0 + 1 + r * PROW}
						width={PM.w / PM.cols - 2}
						height={PROW - 2}
						rx="1"
						fill={l.color}
						opacity={0.2 + noise(c, r, 21 + li) * 0.65}
					/>
				{/each}
			{/each}
		{/each}
		<!-- the last token's query, outlined in rose -->
		<rect
			x={PM.x - 0.5}
			y={LENSES[0].cy - WM.h / 2 + LAST * PROW - 0.5}
			width={PM.w + 1}
			height={PROW}
			rx="2"
			fill="none"
			stroke="var(--cat-3)"
			stroke-width="1"
		/>

		<!-- Q and K meet in the score table: queries arrive at the rows, keys at
		     the columns, and each cell is where a blue row crosses an orange
		     column — the dot product qᵢ·kⱼ -->
		<path
			d="M {PM.x + PM.w + 4} 52 C 272 52, 274 54, {G1.x - 10} 54"
			fill="none"
			stroke="var(--accent)"
			stroke-width="1"
			opacity="0.75"
			marker-end="url(#at-arrow-q)"
		/>
		<path
			d="M {PM.x + PM.w + 4} 106 C 270 106, 274 109, {G1.x - 10} 109"
			fill="none"
			stroke="var(--cat-8)"
			stroke-width="1"
			opacity="0.75"
			marker-end="url(#at-arrow-k)"
		/>
		{#each Array.from({ length: 4 }, (_, k) => k) as r (r)}
			<rect
				x={G1.x - 6.5}
				y={G1.y + r * CELL + 2.5}
				width="3.5"
				height={CELL - 5}
				rx="1.2"
				fill="var(--accent)"
				opacity="0.8"
			/>
		{/each}
		{#each Array.from({ length: 4 }, (_, k) => k) as c (c)}
			<rect
				x={G1.x + c * CELL + 2.5}
				y={G1.y + 4 * CELL + 3}
				width={CELL - 5}
				height="3.5"
				rx="1.2"
				fill="var(--cat-8)"
				opacity="0.8"
			/>
		{/each}
		{#each Array.from({ length: 4 }, (_, k) => k) as r (r)}
			{#each Array.from({ length: 4 }, (_, k) => k) as c (c)}
				{@const cx = G1.x + c * CELL}
				{@const cy = G1.y + r * CELL}
				{#if c <= r}
					<rect
						x={cx + 1}
						y={cy + 1}
						width={CELL - 2}
						height={CELL - 2}
						rx="1.5"
						fill="var(--accent)"
						opacity={0.15 + SCORES[r][c] * 0.7}
					/>
				{:else if r === 0 && c === 3}
					<rect
						x={cx + 1}
						y={cy + 1}
						width={CELL - 2}
						height={CELL - 2}
						rx="1.5"
						fill="none"
						stroke="var(--line-soft)"
					/>
					<text x={cx + CELL / 2} y={cy + CELL / 2 + 2.5} text-anchor="middle" class="mask">
						−∞
					</text>
				{:else}
					<rect
						x={cx + 1}
						y={cy + 1}
						width={CELL - 2}
						height={CELL - 2}
						rx="1.5"
						fill="none"
						stroke="var(--line-soft)"
					/>
					<line
						x1={cx + 3.5}
						y1={cy + CELL - 3.5}
						x2={cx + CELL - 3.5}
						y2={cy + 3.5}
						stroke="var(--line-soft)"
						stroke-width="1"
					/>
				{/if}
			{/each}
		{/each}
		<rect
			x={G1.x - 1}
			y={G1.y + LAST * CELL - 1}
			width={4 * CELL + 2}
			height={CELL + 2}
			rx="2.5"
			fill="none"
			stroke="var(--cat-3)"
			stroke-width="1"
		/>
		<text x={G1.x + 2 * CELL} y="126" text-anchor="middle" class="label">
			scores · <tspan fill="var(--accent)">Q</tspan><tspan fill="var(--cat-8)">Kᵀ</tspan>/√dₖ
		</text>

		<!-- softmax, row by row -->
		<line
			x1={G1.x + 4 * CELL + 3}
			y1="76"
			x2="366"
			y2="76"
			stroke="var(--ink-3)"
			stroke-width="1"
		/>
		<rect
			x="370"
			y="68"
			width="42"
			height="17"
			rx="8.5"
			fill="var(--surface)"
			stroke="var(--cat-2)"
		/>
		<text x="391" y="79.5" text-anchor="middle" class="pill">softmax</text>
		<line
			x1="416"
			y1="76"
			x2={G2.x - 6}
			y2="76"
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#at-arrow)"
		/>

		<!-- ── attention: each row now a budget of exactly 1 ── -->
		{#each Array.from({ length: 4 }, (_, k) => k) as r (r)}
			{#each Array.from({ length: 4 }, (_, k) => k) as c (c)}
				{@const cx = G2.x + c * CELL}
				{@const cy = G2.y + r * CELL}
				{#if c <= r}
					<rect
						x={cx + 1}
						y={cy + 1}
						width={CELL - 2}
						height={CELL - 2}
						rx="1.5"
						fill="var(--cat-2)"
						opacity={0.1 + WEIGHTS[r][c] * 0.85}
					/>
				{:else}
					<rect
						x={cx + 1}
						y={cy + 1}
						width={CELL - 2}
						height={CELL - 2}
						rx="1.5"
						fill="var(--ink)"
						opacity="0.05"
					/>
					{#if r === 0 && c === 3}
						<text x={cx + CELL / 2} y={cy + CELL / 2 + 2.5} text-anchor="middle" class="mask">
							0
						</text>
					{/if}
				{/if}
			{/each}
		{/each}
		<rect
			x={G2.x - 1}
			y={G2.y + LAST * CELL - 1}
			width={4 * CELL + 2}
			height={CELL + 2}
			rx="2.5"
			fill="none"
			stroke="var(--cat-3)"
			stroke-width="1"
		/>
		<text x={G2.x + 2 * CELL} y="126" text-anchor="middle" class="label">attention</text>

		<!-- ── the readout: the budget buys a blend of the values ── -->
		<path
			d="M {G2.x + 4 * CELL + 2} {G2.y + LAST * CELL + CELL / 2} C 494 97, 500 102, {BLEND.x -
				9} {BLEND.y - 3}"
			fill="none"
			stroke="var(--cat-3)"
			stroke-width="1"
			marker-end="url(#at-arrow-r)"
		/>
		<path
			d="M {PM.x + PM.w + 4} 160 C 330 200, 450 186, {BLEND.x - 7} {BLEND.y + 5}"
			fill="none"
			stroke="var(--cat-6)"
			stroke-width="1"
			opacity="0.75"
			marker-end="url(#at-arrow-v)"
		/>
		<circle cx={BLEND.x} cy={BLEND.y} r="7.5" fill="var(--surface)" stroke="var(--ink-3)" />
		<text x={BLEND.x} y={BLEND.y + 4.5} text-anchor="middle" class="op">·</text>
		<line
			x1={BLEND.x + 9}
			y1={BLEND.y}
			x2={ZC.x - 6}
			y2={BLEND.y}
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#at-arrow)"
		/>

		<text x={ZC.x + ZC.w / 2 + 1} y="70" text-anchor="middle" class="label">z · the readout</text>
		{#each ZVALS as v, i (i)}
			<rect
				x={ZC.x}
				y={84 + i * 12.5}
				width={ZC.w}
				height="11"
				rx="1.5"
				fill="var(--cat-6)"
				opacity={0.25 + v * 0.65}
			/>
		{/each}
		<rect
			x={ZC.x - 2.5}
			y="81.5"
			width={ZC.w + 5}
			height="53"
			rx="3"
			fill="none"
			stroke="var(--cat-3)"
			stroke-width="1"
		/>
		<text x={ZC.x + ZC.w / 2 + 1} y="148" text-anchor="middle" class="cap dim">
			a weighted blend of values
		</text>

		<!-- the equation, written the way the prose dissects it -->
		<text x="330" y="214" text-anchor="middle" class="eq">
			attention(<tspan fill="var(--accent)">Q</tspan>, <tspan fill="var(--cat-8)">K</tspan>,
			<tspan fill="var(--cat-6)">V</tspan>) = <tspan fill="var(--cat-2)">softmax</tspan>(<tspan
				fill="var(--accent)">Q</tspan
			><tspan fill="var(--cat-8)">Kᵀ</tspan>/√dₖ + <tspan class="dim">M</tspan>)·<tspan
				fill="var(--cat-6)">V</tspan
			>
		</text>
	</svg>
</Plate>

<style>
	.tok {
		font-family: var(--font-mono);
		font-size: 10.5px;
		fill: var(--ink);
	}
	.label {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 12.5px;
		fill: var(--ink-2);
	}
	.wlbl {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 12px;
	}
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-2);
	}
	.dim {
		fill: var(--ink-3);
	}
	.mask {
		font-family: var(--font-mono);
		font-size: 7px;
		fill: var(--ink-3);
	}
	.pill {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 9.5px;
		fill: var(--cat-2);
	}
	.op {
		font-family: var(--font-serif);
		font-size: 15px;
		fill: var(--ink);
	}
	.eq {
		font-family: var(--font-mono);
		font-size: 9.5px;
		fill: var(--ink-2);
	}
</style>
