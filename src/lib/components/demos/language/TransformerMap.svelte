<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	// The transformer, drawn once as a map: tokens travel left to right as
	// lanes, attention is the only place lanes touch, and the tail end reuses
	// the scores → softmax → belief grammar of the digits chapter's pipeline.
	// The five numbered stations match the prose and the walkthrough's five stages.
	// Static by design; the walkthrough below runs it with real numbers.

	const TOKENS = ['once', '␣upon', '␣a', '␣time'];
	const LANE_Y = [48, 86, 124, 162];
	const LAST = LANE_Y.length - 1;

	// each token's vector, as four cells of varying intensity (illustrative)
	const CELLS = [
		[0.85, 0.3, 0.55, 0.2],
		[0.35, 0.75, 0.25, 0.6],
		[0.2, 0.45, 0.8, 0.35],
		[0.6, 0.25, 0.4, 0.85]
	];
	const POS = [0.25, 0.45, 0.65, 0.85]; // the added position, one shade per slot

	// causal attention: every lane reads every earlier lane
	const ATT = [
		{ from: 0, to: 1, w: 1.7 },
		{ from: 0, to: 2, w: 0.8 },
		{ from: 1, to: 2, w: 1.2 },
		{ from: 0, to: 3, w: 1.5 },
		{ from: 1, to: 3, w: 0.7 },
		{ from: 2, to: 3, w: 1.0 }
	];
	const A0 = 168; // where the reading curves leave their lane
	const A1 = 238; // where they land

	// the block's walls and furniture
	const BLOCK = { x0: 138, x1: 332, y0: 26, y1: 182 };
	const QKV_X = 152;
	const MLP = { x0: 262, x1: 316 };

	// the tail: scores, softmax, belief (same grammar as the digits pipeline)
	const Z = [-0.8, 1.2, -1.1, 3.0, -0.4, 0.9, -1.4, 0.5]; // eight of 369 scores
	const zLim = Math.max(...Z.map(Math.abs));
	const Z_X = 436;
	const Z_MID = 100;
	const P = [0.04, 0.1, 0.02, 0.62, 0.05, 0.08, 0.02, 0.07];
	const pLim = Math.max(...P);
	const P_X = 582;
	const P_BASE = 124;
	const SLOT = 9;
	const BAR = 5.5;
</script>

<Plate
	id="transformer"
	title="The transformer as a map"
	caption="The whole machine as a map. Each token rides its own lane; attention is the only place lanes touch, and the curves only ever point backwards. Every lane ends in a guess — the map follows the last one, whose belief names the next token."
>
	<svg
		viewBox="0 0 660 216"
		class="mx-auto block w-full max-w-[950px]"
		role="img"
		aria-label="The transformer as a map. Four tokens — once, upon, a, time — each become a vector plus a position, then travel left to right through a block where attention lets each token read the earlier ones and an MLP lets each think alone. The block repeats. The last token's vector becomes one score per vocabulary token, a softmax turns the scores into a belief, and the tallest bar is the prediction: the token 'there'."
	>
		<defs>
			<marker
				id="tm-arrow"
				viewBox="0 0 8 8"
				refX="6.5"
				refY="4"
				markerWidth="6"
				markerHeight="6"
				orient="auto-start-reverse"
			>
				<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--ink-3)" />
			</marker>
			<linearGradient id="tm-fade" x1="0" y1="0" x2="1" y2="0">
				<stop offset="0" stop-color="var(--line)" />
				<stop offset="1" stop-color="var(--line)" stop-opacity="0" />
			</linearGradient>
		</defs>

		<!-- ── the block, and its ghost repeat behind it ── -->
		<rect
			x={BLOCK.x0 + 8}
			y={BLOCK.y0 - 8}
			width={BLOCK.x1 - BLOCK.x0}
			height={BLOCK.y1 - BLOCK.y0}
			rx="9"
			fill="none"
			stroke="var(--line-soft)"
		/>
		<rect
			x={BLOCK.x0}
			y={BLOCK.y0}
			width={BLOCK.x1 - BLOCK.x0}
			height={BLOCK.y1 - BLOCK.y0}
			rx="9"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		<text x={(BLOCK.x0 + BLOCK.x1) / 2} y={BLOCK.y0 - 12} text-anchor="middle" class="label">
			one block · repeated ×2
		</text>

		<!-- ── lanes: the residual stream, one per token ── -->
		{#each LANE_Y as y, i (i)}
			<text x="52" y={y + 3.5} text-anchor="end" class="tok">{TOKENS[i]}</text>
			<!-- the vector: four cells, plus the thin position stripe -->
			{#each CELLS[i] as a, ci (ci)}
				<rect
					x="60"
					y={y - 14 + ci * 7}
					width="11"
					height="6"
					rx="1"
					fill="var(--accent)"
					opacity={a}
				/>
			{/each}
			<rect x="73" y={y - 14} width="4" height="27" rx="1" fill="var(--cat-1)" opacity={POS[i]} />
			<!-- the lane through the block -->
			<line
				x1="81"
				y1={y}
				x2={i === LAST ? 348 : BLOCK.x1}
				y2={y}
				stroke="var(--line)"
				stroke-width="1.2"
			/>
			{#if i !== LAST}
				<line
					x1={BLOCK.x1}
					y1={y}
					x2={BLOCK.x1 + 42}
					y2={y}
					stroke="url(#tm-fade)"
					stroke-width="1.2"
				/>
			{/if}
			<!-- q·k·v: three colored projections per lane. All three are learned,
			     so they take the three learned slots — ultramarine, violet,
			     blue-cyan — exactly as the attention diagram gives Q, K and V -->
			{#each ['var(--accent)', 'var(--cat-8)', 'var(--cat-6)'] as c, ci (c)}
				<rect
					x={QKV_X - 4.5}
					y={y - 9.5 + ci * 6.5}
					width="9"
					height="5.5"
					rx="1.5"
					fill={c}
					opacity="0.8"
				/>
			{/each}
			<!-- the mlp: widen fourfold, squeeze back — thin, wide, thin -->
			{#each [{ dx: 0, h: 9, o: 0.5 }, { dx: 8, h: 19, o: 0.85 }, { dx: 16, h: 9, o: 0.5 }] as b (b.dx)}
				<rect
					x={(MLP.x0 + MLP.x1) / 2 - 10 + b.dx}
					y={y - b.h / 2}
					width="4"
					height={b.h}
					rx="1.3"
					fill="var(--cat-1)"
					opacity={b.o}
				/>
			{/each}
		{/each}

		<!-- ── attention: the only place lanes touch, and only backwards ── -->
		{#each ATT as a (`${a.from}-${a.to}`)}
			<path
				d="M {A0} {LANE_Y[a.from]} C {A0 + 26} {LANE_Y[a.from]}, {A1 - 26} {LANE_Y[
					a.to
				]}, {A1} {LANE_Y[a.to]}"
				fill="none"
				stroke="var(--accent)"
				stroke-width={a.w}
				opacity="0.5"
			/>
		{/each}
		<!-- pickups and landings, so the curves read as reads, not wiring -->
		{#each [0, 1, 2] as i (i)}
			<circle cx={A0} cy={LANE_Y[i]} r="1.8" fill="var(--accent)" opacity="0.45" />
		{/each}
		{#each [1, 2, 3] as i (i)}
			<circle cx={A1} cy={LANE_Y[i]} r="2.3" fill="var(--accent)" opacity="0.75" />
		{/each}

		<!-- ── the tail: the last lane rises into scores, softmax, belief ── -->
		<path
			d="M 348 {LANE_Y[LAST]} C 390 {LANE_Y[LAST]}, 384 {Z_MID}, {Z_X - 12} {Z_MID}"
			fill="none"
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#tm-arrow)"
		/>

		<line
			x1={Z_X - 5}
			y1={Z_MID}
			x2={Z_X + Z.length * SLOT}
			y2={Z_MID}
			stroke="var(--line)"
			stroke-width="1"
		/>
		{#each Z as z, i (i)}
			{@const h = (Math.abs(z) / zLim) * 30}
			<rect
				x={Z_X + i * SLOT + (SLOT - BAR) / 2}
				y={z >= 0 ? Z_MID - h : Z_MID}
				width={BAR}
				height={Math.max(1, h)}
				rx="1"
				fill={z >= 0 ? 'var(--accent)' : 'var(--warm)'}
				opacity="0.85"
			/>
		{/each}
		<text x={Z_X + (Z.length * SLOT) / 2} y="48" text-anchor="middle" class="label">scores</text>
		<text x={Z_X + (Z.length * SLOT) / 2} y="60" text-anchor="middle" class="cap dim"
			>one per token · 369</text
		>

		<rect
			x="516"
			y={Z_MID - 12}
			width="48"
			height="24"
			rx="12"
			fill="var(--surface)"
			stroke="var(--cat-2)"
		/>
		<text x="540" y={Z_MID + 4} text-anchor="middle" class="pill">softmax</text>
		<line
			x1={Z_X + Z.length * SLOT + 4}
			y1={Z_MID}
			x2="512"
			y2={Z_MID}
			stroke="var(--ink-3)"
			stroke-width="1"
		/>
		<line
			x1="568"
			y1={Z_MID}
			x2={P_X - 10}
			y2={Z_MID}
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#tm-arrow)"
		/>

		<line
			x1={P_X - 5}
			y1={P_BASE}
			x2={P_X + P.length * SLOT}
			y2={P_BASE}
			stroke="var(--line)"
			stroke-width="1"
		/>
		{#each P as p, i (i)}
			{@const h = (p / pLim) * 52}
			<rect
				x={P_X + i * SLOT + (SLOT - BAR) / 2}
				y={P_BASE - Math.max(1, h)}
				width={BAR}
				height={Math.max(1, h)}
				rx="1"
				fill="var(--cat-2)"
				opacity={p === pLim ? 0.95 : 0.5}
			/>
		{/each}
		<text x={P_X + (P.length * SLOT) / 2} y="48" text-anchor="middle" class="label">the belief</text
		>
		<text x={P_X + 3 * SLOT + BAR} y={P_BASE - 58} text-anchor="middle" class="verdict">␣there</text
		>

		<!-- ── the five stations, numbered as in the prose and the walkthrough ── -->
		{#each [{ x: 68, t: 'token + position' }, { x: QKV_X, t: 'q·k·v' }, { x: (A0 + A1) / 2, t: 'attention' }, { x: (MLP.x0 + MLP.x1) / 2, t: 'mlp' }, { x: 505, t: 'scores → belief' }] as s, i (i)}
			<circle cx={s.x} cy="196" r="7" fill="none" stroke="var(--ink-3)" stroke-width="1" />
			<text x={s.x} y="199.5" text-anchor="middle" class="num-lbl">{i + 1}</text>
			<text x={s.x} y="212" text-anchor="middle" class="cap dim">{s.t}</text>
		{/each}
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
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-2);
	}
	.dim {
		fill: var(--ink-3);
	}
	.pill {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 12px;
		fill: var(--cat-2);
	}
	.verdict {
		font-family: var(--font-mono);
		font-size: 10.5px;
		font-weight: 600;
		fill: var(--cat-2);
	}
	/* station numbers are chrome, not a semantic color */
	.num-lbl {
		font-family: var(--font-mono);
		font-size: 9px;
		font-weight: 600;
		fill: var(--ink-3);
	}
</style>
