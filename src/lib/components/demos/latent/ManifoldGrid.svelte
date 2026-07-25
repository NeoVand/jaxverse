<script lang="ts">
	// Plate III — the manifold. No data at all: a uniform K×K grid of latent
	// addresses over [−1,1]², each one decoded into a digit tile and butted
	// into a single sheet. Re-decoded every other trained chunk, so the sheet
	// firms up live. With a width-3 bottleneck the grid covers one slice of
	// the cube; a slider moves the slice.
	import { onDestroy } from 'svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab } from './latent-context.svelte';
	import { SIDE, hexRgb, rasterizeDigitAt, readTokens, setupCanvas } from './common';

	const K = 14;

	let sliceZ = $state(0);

	let canvas: HTMLCanvasElement | undefined = $state();

	// non-reactive machinery — the rAF painter reads these directly
	let gridPix: Float32Array | null = null; // K·K × 784
	let gridVersion = 0;
	let busy = false;
	let want = false;
	let raf = 0;
	let lastTick = -2;
	let sliceTimer: ReturnType<typeof setTimeout> | null = null;
	let sheet: HTMLCanvasElement | null = null;
	let sheetKey = '';

	// live refresh: decoding 196 points costs a real dispatch, so while
	// training runs we only re-decode every other chunk
	$effect(() => {
		const tk = lab.tick;
		void lab.latentDim;
		if (lab.phase !== 'ready') return;
		if (lab.training && tk - lastTick < 2) return;
		lastTick = tk;
		void refreshGrid();
	});

	// slice moves re-decode after a short settle (drags fire fast)
	$effect(() => {
		void sliceZ;
		if (lab.phase !== 'ready' || lab.latentDim !== 3) return;
		if (sliceTimer) clearTimeout(sliceTimer);
		sliceTimer = setTimeout(() => {
			sliceTimer = null;
			void refreshGrid();
		}, 120);
	});

	async function refreshGrid(): Promise<void> {
		if (busy) {
			want = true;
			return;
		}
		if (!lab.engine) return;
		busy = true;
		try {
			const d = lab.latentDim;
			const z = new Float32Array(K * K * d);
			for (let gy = 0; gy < K; gy++) {
				for (let gx = 0; gx < K; gx++) {
					const i = gy * K + gx;
					z[i * d] = -1 + (2 * (gx + 0.5)) / K;
					z[i * d + 1] = 1 - (2 * (gy + 0.5)) / K; // row 0 on top, matching the scatter
					if (d === 3) z[i * d + 2] = sliceZ;
				}
			}
			gridPix = await lab.decode(z, K * K, 256);
			gridVersion++;
		} catch {
			// engine disposed or rebuilding mid-flight
		}
		busy = false;
		if (want) {
			want = false;
			void refreshGrid();
		}
	}

	// painter: rebuilds the K·28 × K·28 sheet only when pixels or theme change
	$effect(() => {
		if (lab.phase !== 'ready') return;
		const paint = () => {
			raf = requestAnimationFrame(paint);
			draw();
		};
		raf = requestAnimationFrame(paint);
		return () => cancelAnimationFrame(raf);
	});

	function draw(): void {
		if (!canvas) return;
		const { ctx, W, H } = setupCanvas(canvas);
		if (W < 40) return;
		ctx.clearRect(0, 0, W, H);
		const tk = readTokens(canvas);
		const s = Math.min(W, H);
		const x0 = (W - s) / 2;
		const y0 = (H - s) / 2;
		if (!gridPix) {
			ctx.strokeStyle = tk.lineSoft;
			ctx.lineWidth = 1;
			ctx.strokeRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1);
			return;
		}
		const key = `${gridVersion}|${tk.surface}|${tk.ink}`;
		if (key !== sheetKey) {
			if (!sheet) {
				sheet = document.createElement('canvas');
				sheet.width = K * SIDE;
				sheet.height = K * SIDE;
			}
			const sctx = sheet.getContext('2d')!;
			const img = sctx.createImageData(K * SIDE, K * SIDE);
			const bg = hexRgb(tk.surface);
			const fg = hexRgb(tk.ink);
			for (let i = 0; i < K * K; i++)
				rasterizeDigitAt(
					img,
					(i % K) * SIDE,
					Math.floor(i / K) * SIDE,
					gridPix,
					i * (SIDE * SIDE),
					bg,
					fg
				);
			sctx.putImageData(img, 0, 0);
			sheetKey = key;
		}
		ctx.imageSmoothingEnabled = false; // digit pixels stay crisp
		ctx.drawImage(sheet!, x0, y0, s, s);
		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.strokeRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1);
	}

	onDestroy(() => {
		if (sliceTimer !== null) clearTimeout(sliceTimer);
	});
</script>

<div class="flex flex-col" use:inview={() => void lab.boot()}>
	{#if lab.phase !== 'ready'}
		<div class="flex min-h-64 flex-col items-center justify-center gap-2 px-6 py-16 text-center">
			<span class="eyebrow">
				{lab.phase === 'error' ? 'the engine stalled — retry it at Plate I' : 'warming up…'}
			</span>
			<p class="max-w-md font-serif text-[15px] text-ink-3 italic">
				This sheet is nothing but decoder: {K} × {K} latent addresses, each asked what lives there.
			</p>
		</div>
	{:else}
		<div class="flex flex-col items-center gap-3 px-4 py-5">
			<span class="eyebrow self-start">
				D(z) over the latent square · {K} × {K} addresses
				{#if lab.latentDim === 3}
					· slice z₃ = {sliceZ.toFixed(2)}
				{/if}
			</span>
			<canvas
				bind:this={canvas}
				class="block aspect-square w-full max-w-md"
				aria-label="A grid of decoded digits covering the whole latent square"
			></canvas>
		</div>

		{#if lab.latentDim === 3}
			<div class="border-t border-line-soft px-4 py-3">
				<div class="mx-auto max-w-sm">
					<Slider
						label="slice position z₃"
						bind:value={sliceZ}
						min={-1}
						max={1}
						step={0.02}
						tone="warm"
						format={(v) => v.toFixed(2)}
					/>
				</div>
			</div>
		{/if}
	{/if}
</div>
