<script lang="ts">
	import { armPoints, type ArmState, type Point } from '$lib/world/simulator';

	interface Props {
		candidates: { cost: number; poses: ArmState[] }[];
		start: ArmState;
		goal: ArmState;
		label?: string;
		selectedIndex?: number;
	}
	let {
		candidates,
		start,
		goal,
		label = 'The same possible futures',
		selectedIndex = 0
	}: Props = $props();

	const geometry = (pose: ArmState) => {
		const points = armPoints(pose);
		const screen = (point: Point) => ({ x: 120 + point.x * 112, y: 108 - point.y * 112 });
		return { base: screen(points.base), elbow: screen(points.elbow), tip: screen(points.tip) };
	};
	function skeleton(pose: ArmState) {
		const { base, elbow, tip } = geometry(pose);
		return `M${base.x},${base.y} L${elbow.x},${elbow.y} L${tip.x},${tip.y}`;
	}
	function tipTrail(poses: ArmState[]) {
		return [start, ...poses]
			.map((pose, i) => {
				const { tip } = geometry(pose);
				return `${i ? 'L' : 'M'}${tip.x.toFixed(2)},${tip.y.toFixed(2)}`;
			})
			.join(' ');
	}
	function formatCost(cost: number) {
		if (!Number.isFinite(cost)) return '—';
		return cost !== 0 && Math.abs(cost) < 0.0001 ? cost.toExponential(2) : cost.toFixed(4);
	}

	const finiteCosts = $derived(
		candidates.map((candidate) => candidate.cost).filter(Number.isFinite)
	);
	const scale = $derived(Math.max(0, ...finiteCosts));
	const lowest = $derived(finiteCosts.length ? Math.min(...finiteCosts) : null);
	const futures = $derived(
		candidates.map((candidate, index) => ({
			...candidate,
			id: String.fromCharCode(65 + index),
			selected: index === selectedIndex,
			rank: Number.isFinite(candidate.cost)
				? 1 + finiteCosts.filter((cost) => cost < candidate.cost).length
				: null,
			best: lowest !== null && candidate.cost === lowest,
			endpoint: candidate.poses.at(-1),
			bar: scale > 0 && Number.isFinite(candidate.cost) ? Math.max(0, candidate.cost / scale) : 0
		}))
	);
	const target = $derived(geometry(goal));
</script>

<section class="candidate-futures" aria-label={label}>
	<div class="gallery-head">
		<span class="eyebrow">{label}</span>
		<span class="cost-hint">Predicted cost · lower is preferred</span>
	</div>
	{#if futures.length}
		<div class="futures">
			{#each futures as future (future.id)}
				<article class={['future', { best: future.best }]}>
					<div class="future-head">
						<span class="future-name">Future {future.id}</span>
						<span class="rank">{future.rank ? `#${future.rank}` : 'Unscored'}</span>
					</div>
					<svg
						viewBox="0 0 240 216"
						role="img"
						aria-label={`Future ${future.id}: the starting pose in gray, predicted tip path and final pose in blue, and desired pose in amber. ${future.poses.length} predicted observations.`}
					>
						<circle cx="120" cy="108" r="10" class="mount" />
						<path d={skeleton(start)} class="start" />
						<path d={tipTrail(future.poses)} class="tip-trail" />
						<path d={skeleton(goal)} class="goal" />
						<circle cx={target.elbow.x} cy={target.elbow.y} r="3" class="goal-joint" />
						<circle cx={target.tip.x} cy={target.tip.y} r="3" class="goal-joint" />
						{#if future.endpoint}
							{@const end = geometry(future.endpoint)}
							<path d={skeleton(future.endpoint)} class="prediction" />
							<circle cx={end.elbow.x} cy={end.elbow.y} r="3" class="predicted-joint" />
							<circle cx={end.tip.x} cy={end.tip.y} r="3.5" class="predicted-tip" />
						{/if}
						<circle cx="120" cy="108" r="3" class="pivot" />
					</svg>
					<div class="score">
						<div class="score-label">
							<span class="num">{formatCost(future.cost)}</span>
							<span class="verdict">
								{#if future.selected}<span class="selection-dot" aria-hidden="true"></span>{/if}
								{future.best ? 'Lowest cost' : future.selected ? 'Selected' : 'Alternative'}
							</span>
						</div>
						<div class="cost-track" aria-hidden="true">
							<span style:width={`${future.bar * 100}%`}></span>
						</div>
					</div>
				</article>
			{/each}
		</div>
		<div class="gallery-foot">
			<div class="legend" aria-label="Diagram legend">
				<span><i class="start-key"></i>Start</span>
				<span><i class="prediction-key"></i>Predicted readout</span>
				<span><i class="goal-key"></i>Goal</span>
			</div>
			<span class="bar-scale">Bar scale: 0–{formatCost(scale)}</span>
		</div>
	{:else}
		<p class="empty">Rehearse to compare possible futures.</p>
	{/if}
</section>

<style>
	.candidate-futures {
		max-width: 820px;
		margin: 24px auto 0;
		padding: 0 16px;
	}
	.gallery-head,
	.gallery-foot,
	.future-head,
	.score-label {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 8px 16px;
	}
	.gallery-head,
	.gallery-foot {
		flex-wrap: wrap;
	}
	.cost-hint,
	.gallery-foot,
	.verdict {
		font: 10px var(--font-sans);
		color: var(--ink-3);
	}
	.futures {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 20px;
		margin-top: 14px;
	}
	.future {
		min-width: 0;
		border-top: 2px solid var(--line);
		padding-top: 10px;
	}
	.future.best {
		border-top-color: var(--accent);
	}
	.future-name {
		font: italic 17px var(--font-serif);
		color: var(--ink-2);
	}
	.rank {
		font: 10px var(--font-mono);
		color: var(--ink-3);
	}
	.best .rank,
	.best .verdict {
		color: var(--accent);
	}
	svg {
		display: block;
		width: 100%;
	}
	path {
		fill: none;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.mount {
		fill: none;
		stroke: var(--line);
		stroke-width: 0.8;
	}
	.start {
		stroke: var(--ink-3);
		stroke-width: 2;
		opacity: 0.3;
	}
	.tip-trail {
		stroke: var(--accent);
		stroke-width: 1;
		opacity: 0.45;
	}
	.goal {
		stroke: var(--warm);
		stroke-width: 1.7;
		stroke-dasharray: 4 4;
	}
	.goal-joint {
		fill: var(--surface);
		stroke: var(--warm);
		stroke-width: 1.1;
	}
	.prediction {
		stroke: var(--accent);
		stroke-width: 2;
	}
	.predicted-joint {
		fill: var(--surface);
		stroke: var(--accent);
		stroke-width: 1.4;
	}
	.predicted-tip {
		fill: var(--accent);
	}
	.pivot {
		fill: var(--surface);
		stroke: var(--ink-3);
		stroke-width: 1;
	}
	.score-label > .num {
		font-size: 12px;
		color: var(--ink-2);
	}
	.verdict {
		display: flex;
		align-items: center;
		gap: 4px;
		white-space: nowrap;
	}
	.selection-dot {
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: currentColor;
	}
	.cost-track {
		height: 3px;
		margin-top: 8px;
		background: var(--line-soft);
	}
	.cost-track > span {
		display: block;
		height: 100%;
		background: var(--ink-3);
		opacity: 0.5;
		transition: width 180ms ease;
	}
	.best .cost-track > span {
		background: var(--accent);
		opacity: 0.8;
	}
	.gallery-foot {
		margin-top: 14px;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 14px;
	}
	.legend > span {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.legend i {
		display: inline-block;
		width: 12px;
		border-top: 1.5px solid var(--ink-3);
	}
	.legend .prediction-key {
		border-color: var(--accent);
	}
	.legend .goal-key {
		border-color: var(--warm);
		border-top-style: dashed;
	}
	.bar-scale {
		font-family: var(--font-mono);
	}
	.empty {
		margin: 16px 0 0;
		font: italic 15px var(--font-serif);
		color: var(--ink-3);
	}
	@media (max-width: 540px) {
		.futures {
			grid-template-columns: 1fr;
			gap: 12px;
		}
		.future {
			display: grid;
			grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr);
			grid-template-rows: 1fr 1fr;
			column-gap: 16px;
			padding-top: 0;
		}
		.future-head {
			grid-column: 2;
			grid-row: 1;
			align-self: end;
			padding-bottom: 10px;
		}
		.future > svg {
			grid-column: 1;
			grid-row: 1 / 3;
		}
		.score {
			grid-column: 2;
			grid-row: 2;
			align-self: start;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.cost-track > span {
			transition: none;
		}
	}
</style>
