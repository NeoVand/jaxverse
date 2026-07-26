<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import NeuronDiagram from './NeuronDiagram.svelte';
	import { PLATE_CHIPS, activationById, type ActivationId } from './activations';

	// One neuron, no engine: the curve v·σ(wx + b) redrawn live so the reader
	// can feel what each of the three knobs does — and what swapping σ itself
	// does — before training any. Hovering the plot probes one x through the
	// circuit on the left.
	let w = $state(3);
	let b = $state(0);
	let v = $state(1);
	let actId = $state<ActivationId>('tanh');
	let probeX = $state<number | null>(null);

	const act = $derived(activationById(actId));
	const probe = $derived(probeX === null ? null : { x: probeX, a: v * act.fn(w * probeX + b) });

	const H = 264;
	const PADX = 16;
	const PADT = 12;
	const PADB = 20;
	const YMAX = 2.6;
	const N = 201;

	let plotW = $state(0);
	const pw = $derived(plotW || 640);

	const xPix = (x: number) => PADX + ((x + 1) / 2) * (pw - 2 * PADX);
	const yPix = (y: number) => PADT + (1 - (y + YMAX) / (2 * YMAX)) * (H - PADT - PADB);

	const curvePath = $derived.by(() => {
		let d = '';
		for (let i = 0; i < N; i++) {
			const x = -1 + (2 * i) / (N - 1);
			const y = v * act.fn(w * x + b);
			d += `${i === 0 ? 'M' : 'L'}${xPix(x).toFixed(1)} ${yPix(y).toFixed(1)}`;
		}
		return d;
	});

	// σ's own asymptotes, scaled by v — where this curve flattens out.
	const satLevels = $derived(
		(act.levels ?? [])
			.map((L) => v * L)
			.filter((y) => Math.abs(y) > 0.06 && Math.abs(y) < YMAX - 0.05)
	);

	// Where the pre-activation crosses zero — the point the bias slides around.
	const mid = $derived(Math.abs(w) > 0.15 ? -b / w : null);

	function onPointerMove(e: PointerEvent) {
		const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const frac = (e.clientX - rect.left - PADX) / Math.max(1, rect.width - 2 * PADX);
		probeX = Math.max(-1, Math.min(1, frac * 2 - 1));
	}

	const minus = (s: string) => s.replace('-', '−');
	const fmt = (x: number, digits: number) => minus(x.toFixed(digits));
</script>

<Plate
	n={1}
	title="One neuron"
	caption="The neuron as a circuit and as a curve. Pick an activation, then move the sliders: the vermilion weight w sets the bend's steepness, the teal bias b slides it along x, and the ultramarine amplitude v stretches its height — the disk around σ always plots the very curve being bent. Hover the plot to push one x through the circuit and watch it come out as a."
>
	{#snippet status()}
		<span>
			a = <span style="color: var(--accent);">{fmt(v, 2)}</span> · {act.code}(<span
				style="color: var(--warm);">{fmt(w, 1)}</span
			>x {b < 0 ? '−' : '+'}
			<span style="color: var(--cat-2);">{Math.abs(b).toFixed(2)}</span>)
		</span>
	{/snippet}

	<div
		class="flex flex-wrap items-center gap-1.5 border-b border-line-soft px-4 py-2.5"
		role="group"
		aria-label="Activation function"
	>
		{#each PLATE_CHIPS as id (id)}
			<button
				class="chip"
				class:chip-on={actId === id}
				aria-pressed={actId === id}
				title={activationById(id).note}
				onclick={() => (actId = id)}
			>
				{activationById(id).label}
			</button>
		{/each}
	</div>

	<div class="grid sm:grid-cols-2">
		<div
			class="flex items-center justify-center border-b border-line-soft px-5 py-5 sm:border-r sm:border-b-0"
		>
			<NeuronDiagram {w} {b} {v} {act} {probe} />
		</div>

		<div class="relative" bind:clientWidth={plotW}>
			<svg
				width="100%"
				height={H}
				viewBox="0 0 {pw} {H}"
				role="img"
				aria-label="The curve v times {act.label} of w x plus b, redrawn as the sliders move"
				onpointermove={onPointerMove}
				onpointerleave={() => (probeX = null)}
			>
				<defs>
					<clipPath id="one-neuron-clip">
						<rect
							x={PADX - 2}
							y={PADT - 1}
							width={pw - 2 * PADX + 4}
							height={H - PADT - PADB + 2}
						/>
					</clipPath>
				</defs>

				<!-- hairline grid -->
				{#each [-2, -1, 1, 2] as gy (gy)}
					<line
						x1={PADX}
						x2={pw - PADX}
						y1={yPix(gy)}
						y2={yPix(gy)}
						stroke="var(--line-soft)"
						stroke-width="1"
					/>
				{/each}
				<line
					x1={PADX}
					x2={pw - PADX}
					y1={yPix(0)}
					y2={yPix(0)}
					stroke="var(--line)"
					stroke-width="1"
				/>
				<line
					x1={xPix(0)}
					x2={xPix(0)}
					y1={PADT}
					y2={H - PADB}
					stroke="var(--line)"
					stroke-width="1"
				/>

				<!-- σ's saturation levels, scaled by v: where the curve flattens out -->
				{#each satLevels as ay (ay)}
					<line
						x1={PADX}
						x2={pw - PADX}
						y1={yPix(ay)}
						y2={yPix(ay)}
						stroke="var(--accent)"
						stroke-width="1"
						stroke-dasharray="3 4"
						opacity="0.3"
					/>
				{/each}

				<!-- the neuron -->
				<g clip-path="url(#one-neuron-clip)">
					<path
						d={curvePath}
						fill="none"
						stroke="var(--accent)"
						stroke-width="2"
						stroke-linejoin="round"
						stroke-linecap="round"
					/>
				</g>

				<!-- the bend's midpoint, x = −b/w, riding the curve -->
				{#if mid !== null && Math.abs(mid) <= 1}
					<circle
						cx={xPix(mid)}
						cy={yPix(v * act.fn(0))}
						r="2.6"
						fill="var(--paper)"
						stroke="var(--ink-2)"
						stroke-width="1.25"
					/>
				{/if}

				<!-- the probe: one x pushed through the circuit -->
				{#if probe}
					<line
						x1={xPix(probe.x)}
						x2={xPix(probe.x)}
						y1={PADT}
						y2={H - PADB}
						stroke="var(--ink-3)"
						stroke-width="1"
						opacity="0.55"
					/>
					<circle
						cx={xPix(probe.x)}
						cy={yPix(Math.max(-YMAX, Math.min(YMAX, probe.a)))}
						r="3.4"
						fill="var(--accent)"
						stroke="var(--paper)"
						stroke-width="1.4"
					/>
					<text x={pw - PADX - 2} y={PADT + 10} text-anchor="end" class="num" font-size="10.5">
						<tspan fill="var(--ink-2)">x {minus(probe.x.toFixed(2))}</tspan>
						<tspan dx="6" fill="var(--accent)">a {minus(probe.a.toFixed(2))}</tspan>
					</text>
				{/if}

				<!-- axis labels -->
				{#each [-1, 0, 1] as gx (gx)}
					<text
						x={xPix(gx)}
						y={H - 6}
						text-anchor="middle"
						class="num"
						font-size="10"
						fill="var(--ink-3)">{gx === 0 ? '0' : gx > 0 ? '+1' : '−1'}</text
					>
				{/each}
				{#each [-2, -1, 1, 2] as gy (gy)}
					<text x={PADX + 3} y={yPix(gy) - 3} class="num" font-size="9.5" fill="var(--ink-3)"
						>{gy > 0 ? `+${gy}` : `−${-gy}`}</text
					>
				{/each}
			</svg>
		</div>
	</div>

	<div class="grid gap-x-6 gap-y-4 border-t border-line-soft px-4 py-4 sm:grid-cols-3">
		<Slider
			label="weight w"
			bind:value={w}
			min={-6}
			max={6}
			step={0.1}
			tone="warm"
			format={(x) => fmt(x, 1)}
		/>
		<Slider
			label="bias b"
			bind:value={b}
			min={-3}
			max={3}
			step={0.05}
			tone="teal"
			format={(x) => fmt(x, 2)}
		/>
		<Slider
			label="amplitude v"
			bind:value={v}
			min={-2}
			max={2}
			step={0.05}
			tone="accent"
			format={(x) => fmt(x, 2)}
		/>
	</div>
</Plate>

<style>
	.chip {
		display: inline-flex;
		align-items: center;
		gap: 6px;
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
