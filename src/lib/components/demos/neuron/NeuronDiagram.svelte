<script lang="ts">
	// The neuron as a circuit, drawn in the three learned slots: the ultramarine
	// edge carries the weight w (thickness ∝ |w|, dashed when negative), the
	// violet edge carries the bias b, and the blue-cyan edge carries the
	// amplitude v. Between them runs the signal — through Σ, through the teal σ
	// (whose disk plots the actual activation currently selected), out as a. When the
	// reader probes the plot, the sampled x and the produced a flow through
	// here too, with a dot riding the little curve inside σ.
	import type { ActivationSpec } from './activations';

	interface Props {
		w: number;
		b: number;
		v: number;
		act: ActivationSpec;
		/** The plot's hover probe: input and output of the whole neuron. */
		probe: { x: number; a: number } | null;
	}

	let { w, b, v, act, probe }: Props = $props();

	// ── layout (viewBox units) ──
	const X = { x: 56, y: 64, r: 20 }; // input node
	const ONE = { x: 56, y: 176, r: 16 }; // the constant +1
	const SUM = { x: 178, y: 120, r: 22 }; // Σ
	const SIG = { x: 282, y: 120, r: 30 }; // σ — the star of the plate
	const OUT_X = 348; // arrow tip

	/** Trim a center-to-center segment back to the two node rims. */
	function rim(a: { x: number; y: number; r: number }, bn: { x: number; y: number; r: number }) {
		const dx = bn.x - a.x;
		const dy = bn.y - a.y;
		const len = Math.hypot(dx, dy) || 1;
		const ux = dx / len;
		const uy = dy / len;
		return {
			x1: a.x + ux * a.r,
			y1: a.y + uy * a.r,
			x2: bn.x - ux * (bn.r + 4.5), // leave room for the arrowhead
			y2: bn.y - uy * (bn.r + 4.5)
		};
	}

	const wEdge = rim(X, SUM);
	const bEdge = rim(ONE, SUM);

	const wWidth = $derived(1.2 + (2.2 * Math.min(Math.abs(w), 6)) / 6);
	const bWidth = $derived(1.2 + (1.6 * Math.min(Math.abs(b), 3)) / 3);
	const vWidth = $derived(1.2 + (1.8 * Math.min(Math.abs(v), 2)) / 2);

	// ── the activation's own curve, fitted inside the σ disk ──
	const ZSPAN = 3;
	const INNER = SIG.r - 6; // half-extent of the little plot
	const inner = $derived.by(() => {
		const n = 49;
		let maxAbs = 0.2;
		const ys: number[] = [];
		for (let i = 0; i < n; i++) {
			const z = -ZSPAN + (2 * ZSPAN * i) / (n - 1);
			const y = act.fn(z);
			ys.push(y);
			if (Math.abs(y) > maxAbs) maxAbs = Math.abs(y);
		}
		const scale = (INNER - 2) / maxAbs;
		let d = '';
		for (let i = 0; i < n; i++) {
			const px = SIG.x + (-ZSPAN + (2 * ZSPAN * i) / (n - 1)) * (INNER / ZSPAN);
			const py = SIG.y - ys[i] * scale;
			d += `${i === 0 ? 'M' : 'L'}${px.toFixed(1)} ${py.toFixed(1)}`;
		}
		return { d, scale };
	});

	// where the probed signal currently sits on σ's curve
	const probeDot = $derived.by(() => {
		if (!probe) return null;
		const z = w * probe.x + b;
		if (Math.abs(z) > ZSPAN) return null;
		return { x: SIG.x + z * (INNER / ZSPAN), y: SIG.y - act.fn(z) * inner.scale };
	});

	const minus = (s: string) => s.replace('-', '−');
	const dash = (val: number) => (val < 0 ? '5 4' : undefined);
</script>

<svg
	viewBox="0 0 380 240"
	class="block w-full max-w-[460px]"
	role="img"
	aria-label="Diagram of one neuron: the input x is scaled by the weight w (ultramarine), the bias b (violet) is added at the sum node, the result passes through the activation {act.label}, is scaled by the amplitude v (blue-cyan), and comes out as a"
>
	<defs>
		<marker
			id="nd-arrow-w"
			viewBox="0 0 8 8"
			refX="7"
			refY="4"
			markerWidth="6.5"
			markerHeight="6.5"
			orient="auto-start-reverse"
		>
			<path d="M0.5 0.5L7.5 4L0.5 7.5Z" fill="var(--accent)" />
		</marker>
		<marker
			id="nd-arrow-b"
			viewBox="0 0 8 8"
			refX="7"
			refY="4"
			markerWidth="6.5"
			markerHeight="6.5"
			orient="auto-start-reverse"
		>
			<path d="M0.5 0.5L7.5 4L0.5 7.5Z" fill="var(--cat-8)" />
		</marker>
		<marker
			id="nd-arrow-sig"
			viewBox="0 0 8 8"
			refX="7"
			refY="4"
			markerWidth="6.5"
			markerHeight="6.5"
			orient="auto-start-reverse"
		>
			<path d="M0.5 0.5L7.5 4L0.5 7.5Z" fill="var(--ink-2)" />
		</marker>
		<marker
			id="nd-arrow-v"
			viewBox="0 0 8 8"
			refX="7"
			refY="4"
			markerWidth="6.5"
			markerHeight="6.5"
			orient="auto-start-reverse"
		>
			<path d="M0.5 0.5L7.5 4L0.5 7.5Z" fill="var(--cat-6)" />
		</marker>
		<clipPath id="nd-sigma-clip">
			<circle cx={SIG.x} cy={SIG.y} r={SIG.r - 2.5} />
		</clipPath>
	</defs>

	<!-- ── edges ── -->
	<line
		x1={wEdge.x1}
		y1={wEdge.y1}
		x2={wEdge.x2}
		y2={wEdge.y2}
		stroke="var(--accent)"
		stroke-width={wWidth}
		stroke-dasharray={dash(w)}
		marker-end="url(#nd-arrow-w)"
	/>
	<line
		x1={bEdge.x1}
		y1={bEdge.y1}
		x2={bEdge.x2}
		y2={bEdge.y2}
		stroke="var(--cat-8)"
		stroke-width={bWidth}
		stroke-dasharray={dash(b)}
		marker-end="url(#nd-arrow-b)"
	/>
	<line
		x1={SUM.x + SUM.r}
		y1={SUM.y}
		x2={SIG.x - SIG.r - 4.5}
		y2={SIG.y}
		stroke="var(--ink-2)"
		stroke-width="1.6"
		marker-end="url(#nd-arrow-sig)"
	/>
	<line
		x1={SIG.x + SIG.r}
		y1={SIG.y}
		x2={OUT_X - 4.5}
		y2={SIG.y}
		stroke="var(--cat-6)"
		stroke-width={vWidth}
		stroke-dasharray={dash(v)}
		marker-end="url(#nd-arrow-v)"
	/>

	<!-- ── nodes ── -->
	<circle
		cx={X.x}
		cy={X.y}
		r={X.r}
		fill="var(--surface-2)"
		stroke="var(--ink-3)"
		stroke-width="1.4"
	/>
	<text
		x={X.x}
		y={X.y + 5.5}
		text-anchor="middle"
		font-size="17"
		fill="var(--ink)"
		style="font-family: var(--font-serif); font-style: italic;">x</text
	>
	{#if probe}
		<text
			x={X.x}
			y={X.y + X.r + 14}
			text-anchor="middle"
			class="num"
			font-size="10"
			fill="var(--ink-2)">{minus(probe.x.toFixed(2))}</text
		>
	{/if}

	<circle
		cx={ONE.x}
		cy={ONE.y}
		r={ONE.r}
		fill="var(--surface-2)"
		stroke="var(--ink-3)"
		stroke-width="1.4"
	/>
	<text
		x={ONE.x}
		y={ONE.y + 3.5}
		text-anchor="middle"
		class="num"
		font-size="11"
		fill="var(--ink-2)">+1</text
	>

	<circle
		cx={SUM.x}
		cy={SUM.y}
		r={SUM.r}
		fill="var(--surface-2)"
		stroke="var(--ink-3)"
		stroke-width="1.4"
	/>
	<text
		x={SUM.x}
		y={SUM.y + 6.5}
		text-anchor="middle"
		font-size="19"
		fill="var(--ink)"
		style="font-family: var(--font-serif);">Σ</text
	>

	<!-- σ: the disk plots the selected activation itself -->
	<circle
		cx={SIG.x}
		cy={SIG.y}
		r={SIG.r}
		fill="var(--surface-2)"
		stroke="var(--ink-2)"
		stroke-width="1.5"
	/>
	<g clip-path="url(#nd-sigma-clip)">
		<line
			x1={SIG.x - SIG.r}
			y1={SIG.y}
			x2={SIG.x + SIG.r}
			y2={SIG.y}
			stroke="var(--line)"
			stroke-width="1"
		/>
		<line
			x1={SIG.x}
			y1={SIG.y - SIG.r}
			x2={SIG.x}
			y2={SIG.y + SIG.r}
			stroke="var(--line)"
			stroke-width="1"
		/>
		<path d={inner.d} fill="none" stroke="var(--cat-2)" stroke-width="1.9" stroke-linecap="round" />
		{#if probeDot}
			<circle
				cx={probeDot.x}
				cy={probeDot.y}
				r="3"
				fill="var(--cat-2)"
				stroke="var(--surface-2)"
				stroke-width="1.2"
			/>
		{/if}
	</g>
	<text x={SIG.x} y={SIG.y + SIG.r + 17} text-anchor="middle" font-size="11.5" fill="var(--ink-3)">
		<tspan style="font-family: var(--font-serif); font-style: italic;">σ</tspan>
		<tspan dx="2" class="num" font-size="10">= {act.code}</tspan>
	</text>

	<!-- ── output ── -->
	<text
		x={OUT_X + 8}
		y={SIG.y + 5.5}
		font-size="17"
		fill="var(--ink)"
		style="font-family: var(--font-serif); font-style: italic;">a</text
	>
	{#if probe}
		<text x={OUT_X + 8} y={SIG.y + 24} class="num" font-size="10" fill="var(--good)"
			>{minus(probe.a.toFixed(2))}</text
		>
	{/if}

	<!-- ── parameter labels, live ── -->
	<text x={112} y={78} text-anchor="middle">
		<tspan
			font-size="14.5"
			fill="var(--accent)"
			style="font-family: var(--font-serif); font-style: italic;">w</tspan
		>
		<tspan dx="3" class="num" font-size="10.5" fill="var(--accent)">{minus(w.toFixed(1))}</tspan>
	</text>
	<text x={112} y={168} text-anchor="middle">
		<tspan
			font-size="14.5"
			fill="var(--cat-8)"
			style="font-family: var(--font-serif); font-style: italic;">b</tspan
		>
		<tspan dx="3" class="num" font-size="10.5" fill="var(--cat-8)">{minus(b.toFixed(2))}</tspan>
	</text>
	<text x={SIG.x + SIG.r + 18} y={SIG.y - 12} text-anchor="middle">
		<tspan
			font-size="14.5"
			fill="var(--cat-6)"
			style="font-family: var(--font-serif); font-style: italic;">v</tspan
		>
		<tspan dx="3" class="num" font-size="10.5" fill="var(--cat-6)">{minus(v.toFixed(2))}</tspan>
	</text>
</svg>
