<script lang="ts">
	// Plate I — the squeeze. Owns the shared engine's lifecycle: auto-loads it
	// when scrolled near, trains it in chunks, and shows the only feedback the
	// model ever gets — eight held-out digits against their reconstructions.
	import { onDestroy } from 'svelte';
	import { Play, Pause, RotateCcw } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab, layersFor } from './latent-context.svelte';
	import { DIM, TileStrip, logSpan, readTokens, setupCanvas, sparkPath } from './common';

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

	const archLabel = $derived(layersFor(lab.latentDim).join(' → '));
	const span = $derived(logSpan([lab.lossHist, lab.valHist]));
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

<div class="flex flex-col" use:inview={() => void lab.boot()}>
	<!-- controls -->
	<div
		class="flex min-h-12 flex-wrap items-center gap-x-5 gap-y-3 border-b border-line-soft px-4 py-3"
	>
		{#if lab.phase === 'idle' || lab.phase === 'loading'}
			<span class="eyebrow">warming up — decoding the digit sheets…</span>
			<span class="text-[12.5px] text-ink-3">
				ten thousand digits, trained in this tab — nothing leaves the page
			</span>
		{:else if lab.phase === 'error'}
			<span class="text-[12.5px] text-bad">{lab.errorMsg}</span>
			<Btn onclick={() => void lab.boot()}>Retry</Btn>
		{:else}
			<Btn kind="primary" disabled={lab.rebuilding} onclick={() => lab.setTraining(!lab.training)}>
				{#if lab.training}
					<Pause size={12} aria-hidden="true" /> Pause
				{:else}
					<Play size={12} aria-hidden="true" /> Train
				{/if}
			</Btn>
			<Btn
				disabled={lab.rebuilding}
				onclick={() => void lab.resetWeights()}
				title="Fresh random weights"
			>
				<RotateCcw size={12} aria-hidden="true" /> Reset
			</Btn>
			<span class="flex items-center gap-1" role="group" aria-label="Bottleneck width">
				<span class="eyebrow mr-1">bottleneck</span>
				{#each [2, 3] as d (d)}
					<button
						class="chip"
						class:chip-on={lab.latentDim === d}
						aria-pressed={lab.latentDim === d}
						disabled={lab.rebuilding}
						onclick={() => void lab.setLatentDim(d as 2 | 3)}
					>
						{d}
					</button>
				{/each}
			</span>
			{#if lab.trained}
				<span class="ml-auto font-serif text-[13px] text-good italic">
					legible — the map below is formed
				</span>
			{/if}
		{/if}
	</div>

	{#if lab.phase === 'ready'}
		<!-- stage: originals over reconstructions -->
		<div class="flex flex-col gap-5 px-4 py-5">
			<div>
				<p class="eyebrow mb-2">input x — eight held-out digits</p>
				<canvas
					bind:this={origCanvas}
					class="block h-16 w-full sm:h-20"
					aria-label="Eight original held-out digits"
				></canvas>
			</div>
			<div>
				<p class="eyebrow mb-2">reconstruction D(E(x)) — redrawn as it trains</p>
				<canvas
					bind:this={reconCanvas}
					class="block h-16 w-full sm:h-20"
					aria-label="The network's current reconstruction of the same eight digits"
				></canvas>
			</div>
		</div>

		<!-- telemetry -->
		<div
			class="flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line-soft px-4 py-2.5 text-ink-2"
		>
			<span class="num text-[11.5px]">
				{archLabel} · {lab.paramCount} params · {lab.device}{lab.rebuilding ? ' · rebuilding…' : ''}
			</span>
			<span class="num text-[11.5px]">
				step {lab.step} · train {fmt(lab.lossNow)} · val {fmt(lab.valLoss)} · {lab.msPerStep.toFixed(
					0
				)} ms/step
			</span>
			<span class="ml-auto flex items-center gap-2">
				<svg
					width="160"
					height="30"
					role="img"
					aria-label="Reconstruction error, log scale — train and held-out"
				>
					<path
						d={sparkPath(lab.lossHist, 160, 30, span[0], span[1])}
						fill="none"
						stroke="var(--accent)"
						stroke-width="1.4"
					/>
					<path
						d={sparkPath(lab.valHist, 160, 30, span[0], span[1])}
						fill="none"
						stroke="var(--warm)"
						stroke-width="1.4"
					/>
				</svg>
				<span class="num text-[10.5px]">
					<span style="color: var(--accent)">train</span>
					/
					<span style="color: var(--warm)">val</span>
				</span>
			</span>
		</div>
	{/if}
</div>

<style>
	.chip {
		font-family: var(--font-sans);
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		padding: 3px 9px;
		border-radius: 5px;
		border: 1px solid var(--line);
		color: var(--ink-2);
		background: var(--surface);
		transition: all 100ms ease;
	}
	.chip:hover:not(:disabled) {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.chip:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.chip-on {
		background: var(--ink);
		border-color: var(--ink);
		color: var(--paper);
	}
</style>
