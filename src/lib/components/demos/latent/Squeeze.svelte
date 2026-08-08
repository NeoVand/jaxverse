<script lang="ts">
	// The squeeze. Owns the shared engine's lifecycle: auto-loads it
	// when scrolled near, trains it in chunks, and shows the only feedback the
	// model ever gets — eight held-out digits against their reconstructions.
	import { onDestroy } from 'svelte';
	import { Play, Pause, RotateCcw } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { ACTIVATIONS, LATENT_DIMS, lab, layersFor, type Depth } from './latent-context.svelte';
	import { DIM, TileStrip, readTokens, setupCanvas } from './common';
	import { sparkPath, sparkSpan } from '$lib/viz/spark';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const N_STRIP = 8;

	// non-reactive machinery — the rAF painter reads these directly
	let stripX: Float32Array | null = null; // 8×784 originals, one per class 0–7
	let recon: Float32Array | null = null; // 8×784 latest reconstructions
	let reconVersion = 0;
	let busy = false;
	let want = false;
	let raf = 0;
	const origTiles = new TileStrip();
	const reconTiles = new TileStrip();

	let origCanvas: HTMLCanvasElement | undefined = $state();
	let reconCanvas: HTMLCanvasElement | undefined = $state();

	const archLabel = $derived(layersFor(lab.latentDim, lab.depth).join(' → '));
	const DEPTHS: Depth[] = [1, 2, 3];
	const span = $derived(sparkSpan([lab.lossHist, lab.valHist]));
	const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : '—');

	function pickStrip(): void {
		const m = lab.mnist;
		if (!m) return;
		// first held-out exemplar of each class 0–7 — deterministic, varied
		const idx: number[] = [];
		for (let d = 0; d < N_STRIP; d++) {
			let found = d;
			for (let i = 0; i < m.testY.length; i++)
				if (m.testY[i] === d) {
					found = i;
					break;
				}
			idx.push(found);
		}
		const x = new Float32Array(N_STRIP * DIM);
		for (let k = 0; k < N_STRIP; k++)
			x.set(m.testX.subarray(idx[k] * DIM, (idx[k] + 1) * DIM), k * DIM);
		stripX = x;
	}

	// one re-decode per trained chunk (and once on load / reset / rebuild)
	$effect(() => {
		void lab.tick;
		if (lab.phase !== 'ready') return;
		void refresh();
	});

	async function refresh(): Promise<void> {
		if (busy) {
			want = true;
			return;
		}
		busy = true;
		try {
			if (!stripX) pickStrip();
			if (stripX) {
				recon = await lab.reconstruct(stripX, N_STRIP);
				reconVersion++;
			}
		} catch {
			// engine disposed or rebuilding mid-flight
		}
		busy = false;
		if (want) {
			want = false;
			void refresh();
		}
	}

	// painter: repaints from cached buffers so theme flips are picked up;
	// nothing animates on its own — the pixels change only when training does
	$effect(() => {
		if (lab.phase !== 'ready') return;
		const paint = () => {
			raf = requestAnimationFrame(paint);
			drawRow(origCanvas, origTiles, stripX, 1);
			drawRow(reconCanvas, reconTiles, recon, reconVersion);
		};
		raf = requestAnimationFrame(paint);
		return () => cancelAnimationFrame(raf);
	});

	function drawRow(
		canvas: HTMLCanvasElement | undefined,
		strip: TileStrip,
		data: Float32Array | null,
		version: number
	): void {
		if (!canvas) return;
		const { ctx, W, H } = setupCanvas(canvas);
		if (W < 40 || H < 10) return;
		ctx.clearRect(0, 0, W, H);
		const tk = readTokens(canvas);
		const gap = 10;
		const tile = Math.min(H, (W - gap * (N_STRIP - 1)) / N_STRIP);
		const x0 = (W - (tile * N_STRIP + gap * (N_STRIP - 1))) / 2;
		const y0 = (H - tile) / 2;
		const tiles = strip.ensure(data, N_STRIP, version, tk.surface, tk.ink);
		if (!tiles) {
			// empty slots until the first reconstruction lands
			ctx.strokeStyle = tk.lineSoft;
			ctx.lineWidth = 1;
			for (let i = 0; i < N_STRIP; i++)
				ctx.strokeRect(x0 + i * (tile + gap) + 0.5, y0 + 0.5, tile - 1, tile - 1);
			return;
		}
		ctx.imageSmoothingEnabled = false; // digit pixels stay crisp
		for (let i = 0; i < N_STRIP; i++)
			ctx.drawImage(tiles[i], x0 + i * (tile + gap), y0, tile, tile);
	}

	onDestroy(() => {
		// this plate owns the shared engine; page unmount powers the lab down
		lab.dispose();
	});
</script>

<Plate id="squeeze" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>step {lab.step}</span>
			<span aria-hidden="true">·</span>
			<span>train {fmt(lab.lossNow)}</span>
			<span aria-hidden="true">·</span>
			<span>val {fmt(lab.valLoss)}</span>
			{#if lab.rebuilding}
				<span aria-hidden="true">·</span>
				<span>rebuilding…</span>
			{/if}
		{:else if lab.phase === 'loading'}
			<span>decoding the digit sheets…</span>
		{:else if lab.phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn
			disabled={lab.phase !== 'ready' || lab.rebuilding}
			onclick={() => lab.setTraining(!lab.training)}
		>
			{#if lab.training}
				<Pause size={12} aria-hidden="true" /> Pause
			{:else}
				<Play size={12} aria-hidden="true" /> Train
			{/if}
		</Btn>
		<Btn
			disabled={lab.phase !== 'ready' || lab.rebuilding}
			onclick={() => void lab.resetWeights()}
			title="Fresh random weights"
		>
			<RotateCcw size={12} aria-hidden="true" /> Reset
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'error'}
			<div class="flex flex-wrap items-center gap-3 px-4 py-4">
				<span class="text-[12.5px] text-bad">{lab.errorMsg}</span>
				<Btn onclick={() => void lab.boot()}>Retry</Btn>
			</div>
		{:else if lab.phase !== 'ready'}
			<div class="flex h-[280px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">fetching MNIST (≈1.7 MB, cached) · warming up the worker…</span>
				<span class="text-[12.5px] text-ink-3">
					ten thousand digits, trained in this tab — nothing leaves the page
				</span>
			</div>
		{:else}
			<!-- stage: originals over reconstructions, one tile row each -->
			<div class="flex flex-col gap-3.5 p-4 sm:p-5">
				<div>
					<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
						<span class="eyebrow">input x · eight held-out digits</span>
						<span class="text-[10.5px] text-ink-3">the question</span>
					</div>
					<canvas
						bind:this={origCanvas}
						class="block h-16 w-full sm:h-24"
						aria-label="Eight original held-out digits"
					></canvas>
				</div>
				<div>
					<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
						<span class="eyebrow">reconstruction D(E(x)) · redrawn as it trains</span>
						<span class="text-[10.5px] text-ink-3">and its own answer key</span>
					</div>
					<canvas
						bind:this={reconCanvas}
						class="block h-16 w-full sm:h-24"
						aria-label="The network's current reconstruction of the same eight digits"
					></canvas>
				</div>

				<!-- the shape of the hourglass — every change re-rolls the weights -->
				<div class="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line-soft pt-3">
					<span class="flex items-center gap-1" role="group" aria-label="Bottleneck width">
						<span class="eyebrow mr-1">bottleneck</span>
						{#each LATENT_DIMS as d (d)}
							<button
								class="chip"
								class:chip-on={lab.latentDim === d}
								aria-pressed={lab.latentDim === d}
								disabled={lab.rebuilding}
								onclick={() => void lab.setLatentDim(d)}
							>
								{d}
							</button>
						{/each}
					</span>
					<span class="flex items-center gap-1" role="group" aria-label="Layers each side">
						<span class="eyebrow mr-1">layers each side</span>
						{#each DEPTHS as k (k)}
							<button
								class="chip"
								class:chip-on={lab.depth === k}
								aria-pressed={lab.depth === k}
								disabled={lab.rebuilding}
								onclick={() => void lab.setDepth(k)}
							>
								{k}
							</button>
						{/each}
					</span>
					<span class="flex items-center gap-1" role="group" aria-label="Activation">
						<span class="eyebrow mr-1">bend</span>
						{#each ACTIVATIONS as a (a)}
							<button
								class="chip"
								class:chip-on={lab.activation === a}
								aria-pressed={lab.activation === a}
								disabled={lab.rebuilding}
								onclick={() => void lab.setActivation(a)}
							>
								{a}
							</button>
						{/each}
					</span>
					<span class="ml-auto flex items-center gap-1.5 text-[10.5px]">
						<i
							class="dot"
							style="background: {lab.trained ? 'var(--good)' : 'var(--ink-3)'};"
							aria-hidden="true"
						></i>
						<span style="color: {lab.trained ? 'var(--good)' : 'var(--ink-3)'};">
							{lab.trained ? 'legible — the map below is formed' : 'fog — keep training'}
						</span>
					</span>
				</div>
			</div>

			<!-- slim telemetry strip: the gap between the rows, as a number -->
			<div class="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line-soft px-4 py-2">
				<span class="flex min-w-40 flex-1 items-center gap-1.5">
					<span class="eyebrow shrink-0 text-[9.5px]" style="color: var(--accent);">
						mse · train
					</span>
					<svg
						viewBox="0 0 200 22"
						preserveAspectRatio="none"
						class="block h-[22px] w-full"
						role="img"
						aria-label="reconstruction error on the training rows, log scale"
					>
						<path
							d={sparkPath(lab.lossHist, 200, 22, {
								log: true,
								floor: 1e-6,
								lo: span[0],
								hi: span[1]
							})}
							fill="none"
							stroke="var(--accent)"
							stroke-width="1.4"
							vector-effect="non-scaling-stroke"
						/>
					</svg>
				</span>
				<span class="flex min-w-40 flex-1 items-center gap-1.5">
					<span class="eyebrow shrink-0 text-[9.5px]" style="color: var(--warm);">
						mse · held out
					</span>
					<svg
						viewBox="0 0 200 22"
						preserveAspectRatio="none"
						class="block h-[22px] w-full"
						role="img"
						aria-label="reconstruction error on the held-out digits, log scale"
					>
						<path
							d={sparkPath(lab.valHist, 200, 22, {
								log: true,
								floor: 1e-6,
								lo: span[0],
								hi: span[1]
							})}
							fill="none"
							stroke="var(--warm)"
							stroke-width="1.4"
							vector-effect="non-scaling-stroke"
						/>
					</svg>
				</span>
				<span class="num text-[10.5px] whitespace-nowrap text-ink-3">
					{archLabel} · {lab.paramCount.toLocaleString('en-US')} params · {lab.device} · {lab.msPerStep.toFixed(
						0
					)} ms/step
				</span>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.dot {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 50%;
	}
</style>
