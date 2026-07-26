<script lang="ts">
	import { Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';

	// Three 1-D landscapes. On θ² one update is θ ← (1 − 2γ)θ, so |1 − 2γ|
	// decides the fate; the double well adds a second minimum to hop into;
	// |θ| has a kink, so the (sub)gradient step never shrinks near the bottom.
	interface Curve {
		id: 'quad' | 'well' | 'abs';
		label: string;
		f(th: number): number;
		df(th: number): number;
		xMax: number;
		yMax: number;
		/** θ₀ — mid-slope, with frame headroom for diverging bounces. */
		theta0: number;
	}

	const CURVES: Curve[] = [
		{
			id: 'quad',
			label: 'θ²',
			f: (th) => th * th,
			df: (th) => 2 * th,
			xMax: 1.6,
			yMax: 2.56,
			theta0: -0.9
		},
		{
			id: 'well',
			label: '0.3θ⁴ − θ² + 1',
			f: (th) => 0.3 * th ** 4 - th * th + 1,
			df: (th) => 1.2 * th ** 3 - 2 * th,
			xMax: 2.2,
			yMax: 3.2,
			theta0: -2.05
		},
		{
			id: 'abs',
			label: '|θ|',
			f: (th) => Math.abs(th),
			df: (th) => Math.sign(th),
			xMax: 1.6,
			yMax: 1.6,
			theta0: -1.35
		}
	];

	const KEEP = 12; // cobweb depth kept visible

	// plot geometry, viewBox units
	const VW = 640;
	const VH = 260;
	const PAD_L = 30;
	const PAD_R = 14;
	const PAD_T = 14;
	const PAD_B = 24;

	let curveId = $state<Curve['id']>('quad');
	let gamma = $state(0.3);
	let auto = $state(false);
	let k = $state(0);
	let thetas = $state<number[]>([CURVES[0].theta0]);

	const curve = $derived(CURVES.find((c) => c.id === curveId) ?? CURVES[0]);
	const limit = $derived(curve.xMax - 0.05); // |θ| past the frame = diverged
	const theta = $derived(thetas[thetas.length - 1]);
	const diverged = $derived(Math.abs(theta) > limit);

	const px = $derived(
		(th: number) => PAD_L + ((th + curve.xMax) / (2 * curve.xMax)) * (VW - PAD_L - PAD_R)
	);
	const py = $derived((l: number) => VH - PAD_B - (l / curve.yMax) * (VH - PAD_T - PAD_B));

	const curvePath = $derived.by(() => {
		let d = '';
		for (let i = 0; i <= 96; i++) {
			const th = -curve.xMax + (i / 96) * 2 * curve.xMax;
			d += `${i === 0 ? 'M' : 'L'}${px(th).toFixed(1)} ${py(curve.f(th)).toFixed(1)}`;
		}
		return d;
	});

	const xTicks = $derived.by(() => {
		const out: number[] = [];
		for (let v = -Math.floor(curve.xMax); v <= Math.floor(curve.xMax); v++) out.push(v);
		return out;
	});

	const yTicks = $derived.by(() => {
		const out: number[] = [];
		for (let v = 1; v < curve.yMax; v++) out.push(v);
		return out;
	});

	interface Hop {
		x1: number;
		y1: number;
		x2: number;
		y2: number;
		/** 0 = the newest hop; opacity fades as age grows. */
		age: number;
	}

	const hops = $derived.by(() => {
		const out: Hop[] = [];
		for (let i = 1; i < thetas.length; i++) {
			out.push({
				x1: px(thetas[i - 1]),
				y1: py(curve.f(thetas[i - 1])),
				x2: px(thetas[i]),
				y2: py(curve.f(thetas[i])),
				age: thetas.length - 1 - i
			});
		}
		return out;
	});

	const fade = (age: number) => Math.max(0.14, 0.9 * Math.pow(0.8, age));

	/** Arrowhead triangle at the end of a hop, aligned with its direction. */
	function arrowPoints(h: Hop): string {
		const dx = h.x2 - h.x1;
		const dy = h.y2 - h.y1;
		const len = Math.hypot(dx, dy) || 1;
		const ux = dx / len;
		const uy = dy / len;
		const s = 5;
		const bx = h.x2 - ux * s;
		const by = h.y2 - uy * s;
		return `${h.x2},${h.y2} ${bx - uy * s * 0.55},${by + ux * s * 0.55} ${bx + uy * s * 0.55},${by - ux * s * 0.55}`;
	}

	function stepOnce() {
		const cur = thetas[thetas.length - 1];
		if (Math.abs(cur) > limit) return;
		const next = cur - gamma * curve.df(cur);
		thetas = [...thetas, next].slice(-(KEEP + 1));
		k += 1;
		if (Math.abs(next) > limit) auto = false;
	}

	function reset() {
		thetas = [curve.theta0];
		k = 0;
	}

	function selectCurve(id: Curve['id']) {
		if (id === curveId) return;
		curveId = id;
		auto = false;
		thetas = [(CURVES.find((c) => c.id === id) ?? CURVES[0]).theta0];
		k = 0;
	}

	$effect(() => {
		if (!auto) return;
		const id = setInterval(stepOnce, 400);
		return () => clearInterval(id);
	});
</script>

<Plate
	n={2}
	title="Too big, too small"
	caption="Three regimes of one dial on θ²: below γ = 0.5 the ball eases in, near 0.5 it drops almost in one step, past 1.0 every hop bounces it out of the bowl. On the double well a bold γ hops between valleys; on |θ| the slope never softens, so the ball strides across the kink forever."
>
	{#snippet status()}
		<span>
			k = {k} · γ = {gamma.toFixed(2)} · θ = {theta.toFixed(2)}
			{#if diverged}<span style="color: var(--bad);">· diverged</span>{/if}
		</span>
	{/snippet}

	{#snippet actions()}
		<Btn kind={auto ? 'primary' : 'ghost'} onclick={() => (auto = !auto)} disabled={diverged}>
			{#if auto}
				<Pause size={13} aria-hidden="true" /> Auto
			{:else}
				<Play size={13} aria-hidden="true" /> Auto
			{/if}
		</Btn>
		<Btn onclick={stepOnce} disabled={diverged}>
			<StepForward size={13} aria-hidden="true" /> Step
		</Btn>
		<Btn onclick={reset}><RotateCcw size={13} aria-hidden="true" /> Reset</Btn>
	{/snippet}

	<div class="p-4 sm:p-5">
		<div class="mx-auto max-w-[640px]">
			<div class="mb-3 flex flex-wrap items-center gap-1.5" role="group" aria-label="Loss curve">
				{#each CURVES as c (c.id)}
					<button
						class="chip"
						class:chip-on={curveId === c.id}
						aria-pressed={curveId === c.id}
						onclick={() => selectCurve(c.id)}
					>
						{c.label}
					</button>
				{/each}
			</div>

			<svg
				viewBox="0 0 {VW} {VH}"
				class="block h-auto w-full"
				role="img"
				aria-label="A 1-D loss curve with gradient-descent steps hopping along it"
			>
				<!-- hairline scaffolding -->
				<line
					x1={PAD_L}
					y1={py(0)}
					x2={VW - PAD_R}
					y2={py(0)}
					stroke="var(--line)"
					stroke-width="1"
				/>
				{#each yTicks as l (l)}
					<line
						x1={PAD_L}
						y1={py(l)}
						x2={VW - PAD_R}
						y2={py(l)}
						stroke="var(--line-soft)"
						stroke-width="1"
					/>
					<text
						x={PAD_L - 6}
						y={py(l) + 3}
						text-anchor="end"
						class="num"
						style="fill: var(--ink-3); font-size: 10px;">{l}</text
					>
				{/each}
				<line
					x1={px(0)}
					y1={PAD_T}
					x2={px(0)}
					y2={py(0)}
					stroke="var(--line-soft)"
					stroke-width="1"
					stroke-dasharray="2 4"
				/>
				{#each xTicks as th (th)}
					<text
						x={px(th)}
						y={VH - PAD_B + 14}
						text-anchor="middle"
						class="num"
						style="fill: var(--ink-3); font-size: 10px;">{th}</text
					>
				{/each}
				<text
					x={VW - PAD_R}
					y={VH - PAD_B + 14}
					text-anchor="end"
					class="num"
					style="fill: var(--ink-3); font-size: 10px;">θ</text
				>

				<!-- the landscape -->
				<path d={curvePath} fill="none" stroke="var(--ink)" stroke-width="1.5" />

				<!-- the cobweb: hops fade with age, arrowheads show direction -->
				{#each hops as h, i (i)}
					<g opacity={fade(h.age)}>
						<line
							x1={h.x1}
							y1={h.y1}
							x2={h.x2}
							y2={h.y2}
							stroke="var(--accent)"
							stroke-width="1.4"
						/>
						<polygon points={arrowPoints(h)} fill="var(--accent)" />
						<circle cx={h.x1} cy={h.y1} r="2.6" fill="var(--accent)" />
					</g>
				{/each}

				<!-- the ball -->
				{#if !diverged}
					<circle
						cx={px(theta)}
						cy={py(curve.f(theta))}
						r="5.5"
						fill="var(--warm)"
						stroke="var(--paper)"
						stroke-width="1.5"
					/>
				{:else}
					<text
						x={VW - PAD_R}
						y={PAD_T + 6}
						text-anchor="end"
						class="num"
						style="fill: var(--bad); font-size: 10.5px;">diverged — the step outran the bowl</text
					>
				{/if}
			</svg>
		</div>

		<div class="mx-auto mt-3 max-w-[640px]">
			<Slider
				label="step size γ"
				bind:value={gamma}
				min={0.05}
				max={1.15}
				step={0.01}
				format={(v) => v.toFixed(2)}
				tone="accent"
			/>
		</div>
	</div>
</Plate>

<style>
	.chip {
		display: inline-flex;
		align-items: center;
		font-family: var(--font-sans);
		font-size: 10.5px;
		font-weight: 550;
		letter-spacing: 0.09em;
		text-transform: uppercase;
		color: var(--ink-3);
		background: transparent;
		border: 1px solid var(--line-soft);
		border-radius: 999px;
		padding: 3px 10px;
		cursor: pointer;
		transition:
			color 120ms ease,
			border-color 120ms ease,
			background 120ms ease;
	}
	.chip:hover {
		color: var(--ink-2);
		border-color: var(--line);
	}
	.chip-on {
		color: var(--ink);
		border-color: var(--line);
		background: var(--surface-2);
	}
</style>
