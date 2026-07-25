<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import NeuronDiagram from './NeuronDiagram.svelte';

	// One tanh neuron, no engine: the curve v·tanh(wx + b) redrawn live so the
	// reader can feel what each of the three knobs does before training any.
	let w = $state(3);
	let b = $state(0);
	let v = $state(1);

	const H = 240;
	const PADX = 16;
	const PADT = 12;
	const PADB = 20;
	const YMAX = 2.3;
	const N = 121;

	let plotW = $state(0);
	const pw = $derived(plotW || 640);

	const xPix = (x: number) => PADX + ((x + 1) / 2) * (pw - 2 * PADX);
	const yPix = (y: number) => PADT + (1 - (y + YMAX) / (2 * YMAX)) * (H - PADT - PADB);

	const curvePath = $derived.by(() => {
		let d = '';
		for (let i = 0; i < N; i++) {
			const x = -1 + (2 * i) / (N - 1);
			const y = v * Math.tanh(w * x + b);
			d += `${i === 0 ? 'M' : 'L'}${xPix(x).toFixed(1)} ${yPix(y).toFixed(1)}`;
		}
		return d;
	});

	// Where the step crosses zero — the point the bias slides around.
	const mid = $derived(Math.abs(w) > 0.15 ? -b / w : null);

	const minus = (s: string) => s.replace('-', '−');
	const fmt = (x: number, digits: number) => minus(x.toFixed(digits));
	const formula = $derived(
		`a = ${fmt(v, 2)} · tanh(${fmt(w, 1)}x ${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(2)})`
	);
</script>

<Plate
	n={1}
	title="One neuron"
	caption="The neuron as a circuit and as a curve. The signal flows along the ultramarine path — weighted by w, shifted by b, squashed by σ — and the plot draws v·tanh(wx + b): the vermilion weight w sets the step's steepness (watch its edge thicken), b slides it along x, and the ultramarine output weight v sets its height and sign."
>
	{#snippet status()}
		<span>{formula}</span>
	{/snippet}

	<div class="grid sm:grid-cols-[300px_1fr]">
		<div
			class="flex items-center justify-center border-b border-line-soft px-4 py-4 sm:border-r sm:border-b-0"
		>
			<NeuronDiagram {w} {b} />
		</div>

		<div class="relative" bind:clientWidth={plotW}>
			<svg
				width="100%"
				height={H}
				viewBox="0 0 {pw} {H}"
				role="img"
				aria-label="The curve v times tanh of w x plus b, redrawn as the sliders move"
			>
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

				<!-- the saturation levels ±v: where the step tops out -->
				{#if Math.abs(v) > 0.06}
					{#each [v, -v] as ay (ay)}
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
				{/if}

				<!-- the neuron -->
				<path
					d={curvePath}
					fill="none"
					stroke="var(--accent)"
					stroke-width="2"
					stroke-linejoin="round"
					stroke-linecap="round"
				/>

				<!-- the step's midpoint, x = −b/w -->
				{#if mid !== null && Math.abs(mid) <= 1}
					<circle
						cx={xPix(mid)}
						cy={yPix(0)}
						r="2.6"
						fill="var(--paper)"
						stroke="var(--ink-2)"
						stroke-width="1.25"
					/>
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
			tone="warm"
			format={(x) => fmt(x, 2)}
		/>
		<Slider
			label="output v"
			bind:value={v}
			min={-2}
			max={2}
			step={0.05}
			tone="accent"
			format={(x) => fmt(x, 2)}
		/>
	</div>
</Plate>
