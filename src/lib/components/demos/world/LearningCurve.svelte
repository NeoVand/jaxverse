<script lang="ts">
	import type { WorldMetrics } from '$lib/world/engine';
	let { history, regularization }: { history: WorldMetrics[]; regularization: number } = $props();
	let width = $state(650);
	const left = 36;
	const top = 14;
	const baseline = 76;
	const plotWidth = $derived(Math.max(1, width - left - 18));
	// The label and the plotted scale use the same rounded limit. Keep the
	// existing 0.1 floor so small losses do not fill the chart by zooming in.
	const ceiling = $derived(
		Math.ceil(
			Math.max(0.1, ...history.flatMap((m) => [m.predictionLoss, m.regularizer * regularization])) *
				10
		) / 10
	);
	const maxStep = $derived(Math.max(100, history.at(-1)?.step ?? 0));
	function curve(field: 'predictionLoss' | 'regularizer') {
		return history
			.map(
				(m, i) =>
					`${i ? 'L' : 'M'}${left + (m.step / maxStep) * plotWidth},${baseline - ((m[field] * (field === 'regularizer' ? regularization : 1)) / ceiling) * (baseline - top)}`
			)
			.join(' ');
	}
</script>

<div class="curve" bind:clientWidth={width}>
	<div class="legend">
		<span class="prediction">Prediction loss</span><span class="regularization"
			>λ × distribution loss</span
		>
	</div>
	<svg
		viewBox={`0 0 ${width} 100`}
		role="img"
		aria-label="Training objective terms by optimization step"
	>
		<line x1={left} x2={left + plotWidth} y1={baseline} y2={baseline} />
		<text x={left - 8} y={top + 4} text-anchor="end">{ceiling.toFixed(1)}</text><text
			x="28"
			y={baseline + 3}
			text-anchor="end">0</text
		>
		<path d={curve('predictionLoss')} class="prediction-path" /><path
			d={curve('regularizer')}
			class="regularization-path"
		/>
		<text x={left} y="96">0</text><text x={left + plotWidth} y="96" text-anchor="end"
			>step {maxStep}</text
		>
	</svg>
</div>

<style>
	.curve {
		max-width: 660px;
		width: 100%;
		margin: 0 auto;
	}
	.legend {
		display: flex;
		justify-content: center;
		gap: 20px;
		font: 11px var(--font-sans);
	}
	.prediction {
		color: var(--accent);
	}
	.regularization {
		color: var(--cat-2);
	}
	svg {
		width: 100%;
		height: 100px;
		overflow: visible;
	}
	line {
		stroke: var(--line);
		stroke-width: 1;
	}
	path {
		fill: none;
		stroke-width: 1.5;
	}
	.prediction-path {
		stroke: var(--accent);
	}
	.regularization-path {
		stroke: var(--cat-2);
	}
	text {
		fill: var(--ink-3);
		font: 10px var(--font-mono);
	}
</style>
