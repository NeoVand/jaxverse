<script lang="ts">
	import { armPoints, type ArmState, type Action } from '$lib/world/simulator';
	import Sensor from './Sensor.svelte';

	interface Props {
		state: ArmState;
		goal?: ArmState | null;
		trail?: ArmState[];
		forecasts?: ArmState[][];
		ghost?: ArmState | null;
		past?: ArmState[];
		action?: Action | null;
		pixels?: Float32Array | null;
		label?: string;
		compact?: boolean;
	}
	let {
		state: pose,
		goal = null,
		trail = [],
		forecasts = [],
		ghost = null,
		past = [],
		action = null,
		pixels = null,
		label = 'The kinetic instrument',
		compact = false
	}: Props = $props();
	let width = $state(600);
	const height = $derived(compact ? (width < 320 ? 250 : 300) : width < 440 ? 300 : 420);
	const scale = $derived(Math.min(width * 0.46, height * 0.46) / 0.85);
	const cx = $derived(width * 0.5);
	const cy = $derived(height * 0.51);
	type Point = { x: number; y: number };
	function point(p: Point): Point {
		// The sensor's camera looks along +x to the right and +y upward.
		return { x: cx + p.x * scale, y: cy - p.y * scale };
	}
	function geometry(s: ArmState) {
		const p = armPoints(s);
		return { base: point(p.base), elbow: point(p.elbow), tip: point(p.tip) };
	}
	function link(a: Point, b: Point, start: number, end: number) {
		const dx = b.x - a.x,
			dy = b.y - a.y,
			len = Math.hypot(dx, dy);
		const nx = -dy / len,
			ny = dx / len;
		return `${a.x + nx * start},${a.y + ny * start} ${b.x + nx * end},${b.y + ny * end} ${b.x - nx * end},${b.y - ny * end} ${a.x - nx * start},${a.y - ny * start}`;
	}
	function trace(states: ArmState[]) {
		return states
			.map((s, i) => {
				const p = geometry(s).tip;
				return `${i ? 'L' : 'M'}${p.x.toFixed(2)},${p.y.toFixed(2)}`;
			})
			.join(' ');
	}
	function skeleton(s: ArmState) {
		const p = geometry(s);
		return `M${p.base.x},${p.base.y} L${p.elbow.x},${p.elbow.y} L${p.tip.x},${p.tip.y}`;
	}
	function torque(center: Point, radius: number, command: number) {
		const sign = Math.sign(command);
		const start = sign > 0 ? 0.05 : 1.6;
		const end = start + sign * 1.3;
		const a = { x: center.x + radius * Math.cos(start), y: center.y - radius * Math.sin(start) };
		const b = { x: center.x + radius * Math.cos(end), y: center.y - radius * Math.sin(end) };
		const dx = -Math.sin(end) * sign;
		const dy = -Math.cos(end) * sign;
		return {
			path: `M${a.x},${a.y} A${radius},${radius} 0 0 ${sign > 0 ? 0 : 1} ${b.x},${b.y}`,
			head: `${b.x},${b.y} ${b.x - 5 * dx + 2.5 * dy},${b.y - 5 * dy - 2.5 * dx} ${b.x - 5 * dx - 2.5 * dy},${b.y - 5 * dy + 2.5 * dx}`
		};
	}
	const p = $derived(geometry(pose));
	const g = $derived(goal ? geometry(goal) : null);
	const v = $derived(ghost ? geometry(ghost) : null);
	const ticks = Array.from({ length: 12 }, (_, i) => i);
</script>

<div class="instrument" class:compact bind:clientWidth={width}>
	<svg
		viewBox={`0 0 ${width} ${height}`}
		style:height={`${height}px`}
		role="img"
		aria-label={label}
	>
		<circle {cx} {cy} r="26" class="mount-ring" />
		{#each ticks as k (k)}
			<line
				x1={cx + Math.cos((k * Math.PI) / 6) * 29}
				y1={cy + Math.sin((k * Math.PI) / 6) * 29}
				x2={cx + Math.cos((k * Math.PI) / 6) * 33}
				y2={cy + Math.sin((k * Math.PI) / 6) * 33}
				class="mount-ring"
			/>
		{/each}
		{#if trail.length > 1}<path d={trace(trail)} class="motion-trace" />{/if}
		{#each past as state, i (i)}
			<path d={skeleton(state)} class="past-pose" style:opacity={0.08 + (i + 1) * 0.065} />
		{/each}
		{#each forecasts as states, i (i)}
			<path d={trace(states)} class="forecast-trace" style:opacity={i === 0 ? 0.75 : 0.27} />
		{/each}
		{#if goal && g}
			<path d={skeleton(goal)} class="goal-line" />
			<circle cx={g.elbow.x} cy={g.elbow.y} r="5" class="goal-joint" />
			<circle cx={g.tip.x} cy={g.tip.y} r="4" class="goal-joint" />
		{/if}
		{#if ghost && v}
			<path d={skeleton(ghost)} class="ghost-line" />
			<circle cx={v.elbow.x} cy={v.elbow.y} r="4" class="ghost-joint" />
			<circle cx={v.tip.x} cy={v.tip.y} r="3" class="ghost-joint" />
		{/if}
		<circle {cx} {cy} r="18" class="mount" />
		<polygon points={link(p.base, p.elbow, 7, 5.5)} class="arm" />
		<polygon points={link(p.elbow, p.tip, 5.5, 1.5)} class="arm" />
		<circle {cx} {cy} r="9" class="bearing" />
		<circle cx={p.elbow.x} cy={p.elbow.y} r="7" class="bearing" />
		<circle {cx} {cy} r="4" class="bearing-inner" />
		<circle cx={p.elbow.x} cy={p.elbow.y} r="3" class="bearing-inner" />
		<circle {cx} {cy} r="1" class="pin" />
		<circle cx={p.elbow.x} cy={p.elbow.y} r="1" class="pin" />
		<circle cx={p.tip.x} cy={p.tip.y} r="2" class="pin" />
		{#if action}
			{#each [p.base, p.elbow] as joint, i (i)}
				{#if Math.abs(action[i]) > 1e-6}
					{@const arrow = torque(joint, i === 0 ? 23 : 15, action[i])}
					<path d={arrow.path} class="torque-arrow" />
					<polygon points={arrow.head} class="torque-head" />
				{/if}
			{/each}
		{/if}
	</svg>
	{#if pixels}
		<div class="sensor-inset"><Sensor {pixels} /><span>Model’s view</span></div>
	{/if}
</div>

<style>
	.instrument {
		position: relative;
		width: 100%;
		max-width: 760px;
		margin: 0 auto;
		min-width: 0;
	}
	svg {
		width: 100%;
		display: block;
		overflow: visible;
	}
	.mount-ring {
		fill: none;
		stroke: var(--line);
		stroke-width: 0.7;
	}
	.mount {
		fill: var(--paper);
		stroke: var(--line);
		stroke-width: 1;
	}
	.arm,
	.bearing {
		fill: var(--surface-2);
		stroke: var(--ink-2);
		stroke-width: 0.9;
		stroke-linejoin: round;
	}
	.bearing-inner {
		fill: var(--paper);
		stroke: var(--ink-3);
		stroke-width: 0.7;
	}
	.pin {
		fill: var(--ink);
	}
	.motion-trace {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1.25;
		opacity: 0.45;
	}
	.past-pose {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 2;
		stroke-linecap: round;
	}
	.torque-arrow {
		fill: none;
		stroke: var(--warm);
		stroke-width: 1.6;
		stroke-linecap: round;
	}
	.torque-head {
		fill: var(--warm);
	}
	.forecast-trace {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.4;
	}
	.goal-line {
		fill: none;
		stroke: var(--warm);
		stroke-width: 1.3;
		stroke-dasharray: 4 4;
		opacity: 0.8;
	}
	.goal-joint {
		fill: var(--surface);
		stroke: var(--warm);
		stroke-width: 1;
	}
	.ghost-line {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.3;
		opacity: 0.65;
	}
	.ghost-joint {
		fill: var(--surface);
		stroke: var(--accent);
		stroke-width: 1;
		opacity: 0.8;
	}
	.sensor-inset {
		position: absolute;
		bottom: 12px;
		left: 8px;
		display: grid;
		gap: 5px;
		color: var(--ink-3);
		font: 10px var(--font-sans);
	}
	@media (max-width: 440px) {
		.sensor-inset {
			position: static;
			display: flex;
			align-items: center;
			gap: 8px;
			padding-bottom: 8px;
		}
	}
</style>
