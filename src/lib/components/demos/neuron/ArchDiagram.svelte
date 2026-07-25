<script lang="ts">
	import type { LayerWeights } from '$lib/nn/engine';

	// TensorFlow-Playground-style view of the live network: one column per
	// layer, every edge one weight — thickness ∝ |w|, ultramarine positive,
	// vermilion negative. Hidden columns cap at `cap` units to stay readable;
	// the final hidden layer wears the badges (#1…) that its palette rows share,
	// and hovering a badged node highlights its row (wired via `hovered`).
	interface Props {
		weights: LayerWeights[];
		hovered: number | null;
		onhover: (i: number | null) => void;
		cap?: number;
	}

	let { weights, hovered, onhover, cap = 16 }: Props = $props();

	const W = 260;
	const H = 280;
	const PAD = 26;

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
		const xs = sizes.map((_, k) => PAD + (k * (W - 2 * PAD)) / (C - 1));
		const ys = shown.map((m) =>
			Array.from({ length: m }, (_, i) => PAD + ((i + 0.5) * (H - 2 * PAD)) / m)
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
			(e.toCol === view.finalHidden && e.toIdx === hovered) ||
			(e.fromCol === view.finalHidden && e.fromIdx === hovered);
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
			stroke-width={0.5 + (2.2 * Math.abs(e.val)) / view.wmax}
			opacity={edgeOpacity(e)}
		/>
	{/each}

	{#each view.xs as x, k (k)}
		{#each view.ys[k] as y, i (i)}
			<circle
				cx={x}
				cy={y}
				r={k === 0 || k === view.sizes.length - 1 ? 6 : 4.5}
				fill={k === view.finalHidden && hovered === i ? 'var(--accent-soft)' : 'var(--surface)'}
				stroke={k === view.finalHidden && hovered === i ? 'var(--accent)' : 'var(--ink-3)'}
				stroke-width={k === view.finalHidden && hovered === i ? 1.5 : 1}
			/>
			{#if k === view.finalHidden}
				<text
					x={x + 8.5}
					y={y + 2.5}
					class="num"
					font-size="8"
					fill={hovered === i ? 'var(--accent)' : 'var(--ink-3)'}
					stroke="var(--surface)"
					stroke-width="2.5"
					style="paint-order: stroke;">#{i + 1}</text
				>
				<!-- hover-only affordance: the highlight is decorative, all data is visible without it -->
				<circle
					cx={x}
					cy={y}
					r="9"
					fill="transparent"
					role="presentation"
					aria-hidden="true"
					style="pointer-events: all;"
					onpointerenter={() => onhover(i)}
					onpointerleave={() => onhover(null)}
				/>
			{/if}
		{/each}
	{/each}

	<text
		x={view.xs[0]}
		y={view.ys[0][0] - 12}
		text-anchor="middle"
		font-size="12"
		fill="var(--ink-2)"
		style="font-family: var(--font-serif); font-style: italic;">x</text
	>
	<text
		x={view.xs[view.xs.length - 1]}
		y={view.ys[view.ys.length - 1][0] - 12}
		text-anchor="middle"
		font-size="12"
		fill="var(--ink-2)"
		style="font-family: var(--font-serif); font-style: italic;">f(x)</text
	>

	{#if view.capped}
		<text x={W / 2} y={H - 4} text-anchor="middle" class="num" font-size="8.5" fill="var(--ink-3)"
			>showing {cap} of {Math.max(...view.sizes.slice(1, -1))} units per hidden layer</text
		>
	{/if}
</svg>
