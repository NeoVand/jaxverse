<script lang="ts">
	import type { LayerWeights } from '$lib/nn/engine';

	// TensorFlow-Playground-style view of the live network: one column per
	// layer, every edge one weight — thickness ∝ |w|, ultramarine positive,
	// vermilion negative. Hidden columns cap at `cap` units to stay readable;
	// the final hidden layer wears the badges (#1…) that its palette tiles share.
	// Every node is hoverable: `hovered` names a node as {col, idx} (col 0 is
	// the input, the last col the output) and lights up its incident edges —
	// the palette highlights the matching tile, and vice versa.
	export interface NodeRef {
		col: number;
		idx: number;
	}

	interface Props {
		weights: LayerWeights[];
		hovered: NodeRef | null;
		onhover: (h: NodeRef | null) => void;
		cap?: number;
		/** #1… badges on the final hidden layer (for palettes keyed by unit). */
		badges?: boolean;
		/** One label floats above a single input node; one per node sits beside them otherwise. */
		inLabels?: string[];
		outLabels?: string[];
		/** Optional per-node colors for the output labels (e.g. class colors). */
		outColors?: string[];
	}

	let {
		weights,
		hovered,
		onhover,
		cap = 16,
		badges = true,
		inLabels = ['x'],
		outLabels = ['f(x)'],
		outColors
	}: Props = $props();

	// Golden-ratio canvas: the network reads as a landscape, not a squeezed square.
	const W = 380;
	const H = 235;
	const PADX = 34;
	const PADY = 24;
	// Per-node output labels sit beside the last column, so reserve room for them.
	const padR = $derived(
		outLabels.length > 1 ? 14 + 11 + 7.2 * Math.max(...outLabels.map((s) => s.length)) : PADX
	);

	interface Edge {
		x1: number;
		y1: number;
		x2: number;
		y2: number;
		val: number;
		fromCol: number;
		fromIdx: number;
		toCol: number;
		toIdx: number;
	}

	const view = $derived.by(() => {
		const sizes = [weights[0].inDim, ...weights.map((l) => l.outDim)];
		const C = sizes.length;
		const shown = sizes.map((s, k) => (k === 0 || k === C - 1 ? s : Math.min(s, cap)));
		const xs = sizes.map((_, k) => PADX + (k * (W - PADX - padR)) / (C - 1));
		const ys = shown.map((m) =>
			Array.from({ length: m }, (_, i) => PADY + ((i + 0.5) * (H - 2 * PADY)) / m)
		);
		const edges: Edge[] = [];
		let wmax = 0.1;
		for (let k = 0; k < weights.length; k++) {
			const layer = weights[k];
			for (let i = 0; i < shown[k]; i++)
				for (let j = 0; j < shown[k + 1]; j++) {
					const val = layer.w[i * layer.outDim + j];
					wmax = Math.max(wmax, Math.abs(val));
					edges.push({
						x1: xs[k],
						y1: ys[k][i],
						x2: xs[k + 1],
						y2: ys[k + 1][j],
						val,
						fromCol: k,
						fromIdx: i,
						toCol: k + 1,
						toIdx: j
					});
				}
		}
		return {
			sizes,
			shown,
			xs,
			ys,
			edges,
			wmax,
			finalHidden: C - 2,
			capped: sizes.slice(1, -1).some((s) => s > cap)
		};
	});

	function edgeOpacity(e: Edge): number {
		const base = 0.2 + (0.55 * Math.abs(e.val)) / view.wmax;
		if (hovered === null) return base;
		const active =
			(e.toCol === hovered.col && e.toIdx === hovered.idx) ||
			(e.fromCol === hovered.col && e.fromIdx === hovered.idx);
		return active ? Math.min(1, base + 0.4) : base * 0.15;
	}
</script>

<svg
	viewBox="0 0 {W} {H}"
	class="block w-full"
	role="img"
	aria-label="The network as nodes and edges: input on the left, hidden units in the middle, output on the right. Edge thickness shows each weight's size."
	onpointerleave={() => onhover(null)}
>
	{#each view.edges as e, i (i)}
		<line
			x1={e.x1}
			y1={e.y1}
			x2={e.x2}
			y2={e.y2}
			stroke={e.val >= 0 ? 'var(--accent)' : 'var(--warm)'}
			stroke-width={0.6 + (2.6 * Math.abs(e.val)) / view.wmax}
			opacity={edgeOpacity(e)}
		/>
	{/each}

	{#each view.xs as x, k (k)}
		{#each view.ys[k] as y, i (i)}
			{@const isHi = hovered !== null && hovered.col === k && hovered.idx === i}
			<circle
				cx={x}
				cy={y}
				r={k === 0 || k === view.sizes.length - 1 ? 6 : 4.5}
				fill={isHi ? 'var(--accent-soft)' : 'var(--surface)'}
				stroke={isHi ? 'var(--accent)' : 'var(--ink-3)'}
				stroke-width={isHi ? 1.5 : 1}
			/>
			{#if badges && k === view.finalHidden}
				<text
					x={x + 8.5}
					y={y + 2.5}
					class="num"
					font-size="8"
					fill={isHi ? 'var(--accent)' : 'var(--ink-3)'}
					stroke="var(--surface)"
					stroke-width="2.5"
					style="paint-order: stroke;">#{i + 1}</text
				>
			{/if}
			<!-- hover-only affordance: the highlight is decorative, all data is visible without it -->
			<circle
				cx={x}
				cy={y}
				r={k === 0 || k === view.sizes.length - 1 ? 8 : 5.5}
				fill="transparent"
				role="presentation"
				aria-hidden="true"
				style="pointer-events: all;"
				onpointerenter={() => onhover({ col: k, idx: i })}
				onpointerleave={() => onhover(null)}
			/>
		{/each}
	{/each}

	{#each inLabels as label, i (i)}
		<text
			x={inLabels.length === 1 ? view.xs[0] : view.xs[0] - 11}
			y={inLabels.length === 1 ? view.ys[0][0] - 12 : (view.ys[0][i] ?? 0) + 3.5}
			text-anchor={inLabels.length === 1 ? 'middle' : 'end'}
			font-size="12"
			fill="var(--ink-2)"
			style="font-family: var(--font-serif); font-style: italic;">{label}</text
		>
	{/each}
	{#each outLabels as label, i (i)}
		<text
			x={outLabels.length === 1 ? view.xs[view.xs.length - 1] : view.xs[view.xs.length - 1] + 11}
			y={outLabels.length === 1
				? view.ys[view.ys.length - 1][0] - 12
				: (view.ys[view.ys.length - 1][i] ?? 0) + 3.5}
			text-anchor={outLabels.length === 1 ? 'middle' : 'start'}
			font-size="12"
			fill={outColors?.[i] ?? 'var(--ink-2)'}
			style="font-family: var(--font-serif); font-style: italic;">{label}</text
		>
	{/each}

	{#if view.capped}
		<text x={W / 2} y={H - 4} text-anchor="middle" class="num" font-size="8.5" fill="var(--ink-3)"
			>showing {cap} of {Math.max(...view.sizes.slice(1, -1))} units per hidden layer</text
		>
	{/if}
</svg>
