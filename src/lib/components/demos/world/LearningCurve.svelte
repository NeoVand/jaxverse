<script lang="ts">
	import type { WorldMetrics } from '$lib/world/engine';
	let { history, regularization }: { history: WorldMetrics[]; regularization: number } = $props();
	let width = $state(650);
	const left = 42;
	const top = 14;
	const baseline = 90;
	const floor = 1e-6;
	const plotWidth = $derived(Math.max(1, width - left - 18));
	const latest = $derived(history.at(-1));
	const values = $derived(
		history.flatMap((m) => [m.predictionLoss, m.regularizer * regularization])
	);
	const upper = $derived(Math.ceil(Math.log10(Math.max(0.1, ...values))));
	const lower = $derived(
		Math.min(upper - 1, Math.floor(Math.log10(Math.max(floor, Math.min(0.01, ...values)))))
	);
	const ticks = $derived(Array.from({ length: upper - lower + 1 }, (_, i) => upper - i));
	const maxStep = $derived(Math.max(5000, latest?.step ?? 0));
	function y(value: number) {
		return (
			top + ((upper - Math.log10(Math.max(floor, value))) / (upper - lower)) * (baseline - top)
		);
	}
	function curve(field: 'predictionLoss' | 'regularizer') {
		return history
			.map(
				(m, i) =>
					`${i ? 'L' : 'M'}${left + (m.step / maxStep) * plotWidth},${y(m[field] * (field === 'regularizer' ? regularization : 1))}`
			)
			.join(' ');
	}
</script>

<div class="curve" bind:clientWidth={width}>
	<div class="legend">
		<span class="eyebrow">Training loss · log scale</span>
		<div class="terms">
			<span class="prediction"
				><i></i>Prediction <span class="num">{latest?.predictionLoss.toFixed(4) ?? '—'}</span></span
			>
			<span class="regularization"
				><i></i>λ × distribution
				<span class="num">{latest ? (latest.regularizer * regularization).toFixed(4) : '—'}</span
				></span
			>
		</div>
	</div>
	<svg
		viewBox={`0 0 ${width} 112`}
		role="img"
		aria-label="Training objective terms by optimization step, on a shared logarithmic loss scale"
	>
		{#each ticks as power (power)}
			<line x1={left} x2={left + plotWidth} y1={y(10 ** power)} y2={y(10 ** power)} />
			<text x={left - 8} y={y(10 ** power) + 3} text-anchor="end"
				>{power < -3 ? `1e${power}` : (10 ** power).toFixed(Math.max(0, -power))}</text
			>
		{/each}
		<path d={curve('predictionLoss')} class="prediction-path" />
		<path d={curve('regularizer')} class="regularization-path" />
		{#if !history.length}
			<text x={left + plotWidth / 2} y="56" text-anchor="middle"
				>Press Train to watch both terms change</text
			>
		{/if}
		<text x={left} y="108">0</text><text x={left + plotWidth} y="108" text-anchor="end"
			>step {maxStep.toLocaleString()}</text
		>
	</svg>
</div>

<style>
	.curve {
		width: 100%;
	}
	.legend,
	.terms {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px 20px;
	}
	.legend {
		justify-content: space-between;
		font: 11px/1.5 var(--font-sans);
	}
	.terms > span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	i {
		display: inline-block;
		width: 12px;
		border-top: 1.5px solid currentColor;
	}
	.prediction {
		color: var(--accent);
	}
	.regularization {
		color: var(--cat-2);
	}
	.regularization i {
		border-top-style: dashed;
	}
	svg {
		display: block;
		width: 100%;
		height: 112px;
		overflow: visible;
	}
	line {
		stroke: var(--line-soft);
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
		stroke-dasharray: 4 3;
	}
	text {
		fill: var(--ink-3);
		font: 10px var(--font-mono);
	}
</style>
