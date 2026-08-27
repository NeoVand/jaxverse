<script lang="ts">
	// The manifold. No data at all: a uniform 21×21 grid of latent
	// addresses across the sheet the map is drawn on, each one decoded into a
	// digit tile and butted into a single sheet. Re-decoded every other trained
	// chunk, so it firms up live. Past a width-2 waist the grid covers one slice
	// of the space and a slider moves the slice. The right column answers the
	// question the sheet raises: at the address you are pointing at, does any
	// real digit live nearby, or is the decoder answering from nothing?
	import { onDestroy } from 'svelte';
	import { Play, Pause, RotateCcw } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { plateLabel } from '$lib/data/plates';
	import { lab } from './latent-context.svelte';
	import {
		DIM,
		SIDE,
		TileStrip,
		hexRgb,
		rasterizeDigitAt,
		readTokens,
		setupCanvas
	} from './common';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	/** Odd, so the sheet has a true middle cell — which is where the cursor
	 * starts, so the two read-out tiles are never blank. */
	const K = 21;
	const HOME_CELL = (K * K - 1) / 2;
	/** Closer than a couple of neighbourly hops counts as inhabited. */
	const home = $derived(lab.zNn * 1.8);
	/** The shape of the sheet's coordinates: the waist itself while it is narrow
	 * enough to plot, otherwise its strongest directions. */
	const viewDim = $derived.by(() => {
		void lab.zVersion;
		return lab.viewZ ? lab.viewDim : Math.min(lab.latentDim, 3);
	});
	const projected = $derived.by(() => {
		void lab.zVersion;
		return !!lab.basis;
	});

	let sliceZ = $state(0);
	let hoverCell = $state(HOME_CELL);

	let canvas: HTMLCanvasElement | undefined = $state();
	let cellCanvas: HTMLCanvasElement | undefined = $state();
	let realCanvas: HTMLCanvasElement | undefined = $state();

	// non-reactive machinery — the rAF painter reads these directly
	let gridPix: Float32Array | null = null; // K·K × 784
	let gridVersion = 0;
	let gridCenter: number[] = [0, 0]; // the frame the sheet spanned
	let gridSpan = 1;
	let gridSlice = 0; // and the slice it cut, when the waist is 3 wide
	let busy = false;
	let want = false;
	let raf = 0;
	let lastTick = -2;
	let sliceTimer: ReturnType<typeof setTimeout> | null = null;
	let sheet: HTMLCanvasElement | null = null;
	let sheetKey = '';
	const cellStrip = new TileStrip();
	const realStrip = new TileStrip();

	// live refresh: decoding K² points costs a real dispatch, so while
	// training runs we only re-decode every other chunk
	$effect(() => {
		const tk = lab.tick;
		void lab.latentDim;
		if (lab.phase !== 'ready') return;
		if (lab.training && tk - lastTick < 2) return;
		lastTick = tk;
		void refreshGrid();
		void lab.refreshTestLatents(); // the occupancy readout reads this cache
	});

	// a rebuild or a drifting cloud can leave the slice outside the space
	$effect(() => {
		if (viewDim !== 3) return;
		const c3 = lab.zCenter[2] ?? 0;
		const s = lab.zSpan;
		if (sliceZ < c3 - s || sliceZ > c3 + s) sliceZ = c3;
	});

	// slice moves re-decode after a short settle (drags fire fast)
	$effect(() => {
		void sliceZ;
		if (lab.phase !== 'ready' || viewDim !== 3) return;
		if (sliceTimer) clearTimeout(sliceTimer);
		sliceTimer = setTimeout(() => {
			sliceTimer = null;
			void refreshGrid();
		}, 120);
	});

	/** The address at the centre of cell `i` of a k×k grid covering the frame
	 * `center ± span`, in the sheet's own coordinates. */
	function addressAt(
		i: number,
		k: number,
		vd: number,
		center: number[],
		span: number,
		slice: number
	): number[] {
		const gx = i % k;
		const gy = Math.floor(i / k);
		const z = [
			(center[0] ?? 0) - span + (2 * span * (gx + 0.5)) / k,
			(center[1] ?? 0) + span - (2 * span * (gy + 0.5)) / k
		];
		if (vd === 3) z.push(slice);
		return z;
	}

	async function refreshGrid(): Promise<void> {
		if (busy) {
			want = true;
			return;
		}
		if (!lab.engine) return;
		busy = true;
		// a wide waist is plotted through a projection, and the sheet cannot be
		// laid out until that projection exists
		if (lab.latentDim > 3 && !lab.basis) await lab.refreshTestLatents();
		const vd = lab.viewZ ? lab.viewDim : lab.latentDim;
		const d = lab.basis ? lab.basis.d : lab.latentDim;
		if (d > 3 && !lab.basis) {
			busy = false;
			return;
		}
		const center = [...lab.zCenter];
		const span = lab.zSpan;
		const c3 = center[2] ?? 0;
		const slice = Math.max(c3 - span, Math.min(c3 + span, sliceZ));
		try {
			const z = new Float32Array(K * K * d);
			for (let i = 0; i < K * K; i++) {
				// lay the address out where you can see it, then lift it back into
				// the waist the decoder actually speaks
				z.set(lab.lift(addressAt(i, K, vd, center, span, slice)), i * d);
			}
			gridPix = await lab.decode(z, K * K, 256);
			gridCenter = center;
			gridSpan = span;
			gridSlice = slice;
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

	// ── who lives at this address? ──
	// A linear scan over the 2000 held-out encodings: cheap, and it makes the
	// chapter's claim checkable — the decoder answers everywhere, but only
	// some addresses have ever been visited by a real digit.
	const nearest = $derived.by(() => {
		void lab.zVersion;
		void gridVersion;
		const z = lab.testZ;
		const d = lab.testZd;
		const m = lab.mnist;
		if (!z || !d || !m || hoverCell < 0 || hoverCell >= K * K) return null;
		const view = addressAt(hoverCell, K, lab.viewDim, gridCenter, gridSpan, gridSlice);
		// the comparison happens in the full waist, never in the shadow: a point
		// can look adjacent on a projected map while being nowhere near
		const a = lab.lift(view);
		if (a.length !== d) return null;
		const rows = m.testY.length;
		let best = -1;
		let bestD = Infinity;
		for (let i = 0; i < rows; i++) {
			let dd = 0;
			for (let c = 0; c < d; c++) {
				const t = z[i * d + c] - a[c];
				dd += t * t;
				if (dd >= bestD) break;
			}
			if (dd < bestD) {
				bestD = dd;
				best = i;
			}
		}
		if (best < 0) return null;
		return { idx: best, dist: Math.sqrt(bestD), label: m.testY[best], address: view };
	});

	// the two tiles beside the sheet: the decoder's answer, and its nearest neighbour
	let cellPix: Float32Array | null = null;
	let cellVersion = 0;
	let realPix: Float32Array | null = null;
	let realVersion = 0;

	$effect(() => {
		void gridVersion;
		const i = hoverCell;
		const g = gridPix;
		cellPix = g && i >= 0 && i < K * K ? g.subarray(i * DIM, (i + 1) * DIM) : null;
		cellVersion++;
	});

	$effect(() => {
		const nb = nearest;
		const m = lab.mnist;
		realPix = nb && m ? m.testX.subarray(nb.idx * DIM, (nb.idx + 1) * DIM) : null;
		realVersion++;
	});

	// painter: rebuilds the K·28 × K·28 sheet only when pixels or theme change
	$effect(() => {
		if (lab.phase !== 'ready') return;
		const paint = () => {
			raf = requestAnimationFrame(paint);
			draw();
			drawTile(cellCanvas, cellStrip, cellPix, cellVersion);
			drawTile(realCanvas, realStrip, realPix, realVersion);
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
				rasterizeDigitAt(img, (i % K) * SIDE, Math.floor(i / K) * SIDE, gridPix, i * DIM, bg, fg);
			sctx.putImageData(img, 0, 0);
			sheetKey = key;
		}
		ctx.imageSmoothingEnabled = false; // digit pixels stay crisp
		ctx.drawImage(sheet!, x0, y0, s, s);
		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.strokeRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1);

		if (hoverCell >= 0 && hoverCell < K * K) {
			const cs = s / K;
			const cx = x0 + (hoverCell % K) * cs;
			const cy = y0 + Math.floor(hoverCell / K) * cs;
			const nb = nearest;
			ctx.strokeStyle = nb && nb.dist <= home ? tk.good : tk.warm;
			ctx.lineWidth = 2;
			ctx.strokeRect(cx + 1, cy + 1, cs - 2, cs - 2);
		}
	}

	function drawTile(
		el: HTMLCanvasElement | undefined,
		strip: TileStrip,
		data: Float32Array | null,
		version: number
	): void {
		if (!el) return;
		const { ctx, W, H } = setupCanvas(el);
		if (W < 8) return;
		ctx.clearRect(0, 0, W, H);
		const tk = readTokens(el);
		const tiles = strip.ensure(data, 1, version, tk.surface, tk.ink);
		if (!tiles) return;
		ctx.imageSmoothingEnabled = false;
		const s = Math.min(W, H);
		ctx.drawImage(tiles[0], (W - s) / 2, (H - s) / 2, s, s);
	}

	/** Which cell the pointer is over — the middle one when it is over none, so
	 * the read-out always has something to say. */
	function cellAt(ev: PointerEvent): number {
		if (!canvas) return HOME_CELL;
		const r = canvas.getBoundingClientRect();
		const s = Math.min(r.width, r.height);
		if (s < 8) return HOME_CELL;
		const px = ev.clientX - r.left - (r.width - s) / 2;
		const py = ev.clientY - r.top - (r.height - s) / 2;
		if (px < 0 || py < 0 || px >= s || py >= s) return HOME_CELL;
		const gx = Math.min(K - 1, Math.floor((px / s) * K));
		const gy = Math.min(K - 1, Math.floor((py / s) * K));
		return gy * K + gx;
	}

	const fz = (v: number) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
	const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : '—');
	const addressLabel = $derived.by(() => {
		void gridVersion;
		if (hoverCell < 0) return '—';
		const a = nearest?.address ?? addressAt(hoverCell, K, viewDim, gridCenter, gridSpan, gridSlice);
		return `(${a.map(fz).join(', ')})`;
	});

	onDestroy(() => {
		if (sliceTimer !== null) clearTimeout(sliceTimer);
	});
</script>

<Plate id="manifold" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>step {lab.step}</span>
			<span aria-hidden="true">·</span>
			<span>val {fmt(lab.valLoss)}</span>
			<span aria-hidden="true">·</span>
			<span>{K * K} decodes</span>
		{:else if lab.phase === 'loading'}
			<span>warming up…</span>
		{:else if lab.phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn
			kind={lab.training ? 'ghost' : 'primary'}
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
		{#if lab.phase !== 'ready'}
			<div class="flex h-[280px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">
					{lab.phase === 'error'
						? 'the engine stalled'
						: `warming up the same network as ${plateLabel('latent', 'squeeze').toLowerCase()}…`}
				</span>
				{#if lab.phase === 'error'}
					<span class="mb-2 text-[12.5px] text-bad">{lab.errorMsg}</span>
					<Btn onclick={() => void lab.boot()}>Retry</Btn>
				{:else}
					<span class="text-[12.5px] text-ink-3">
						this sheet is nothing but decoder — every address asked what lives there
					</span>
				{/if}
			</div>
		{:else}
			<div
				class="grid grid-cols-1 gap-px bg-line-soft lg:grid-cols-[minmax(0,520px)_minmax(0,1fr)]"
			>
				<div class="relative bg-surface">
					<span class="eyebrow absolute top-3 left-3 z-10">
						D(z) over the latent square · {K} × {K}
						{#if viewDim === 3}
							· slice {sliceZ.toFixed(2)}
						{/if}
					</span>
					<canvas
						bind:this={canvas}
						class="block aspect-square w-full cursor-crosshair touch-none"
						aria-label="A grid of decoded digits covering the whole latent square"
						onpointermove={(ev) => (hoverCell = cellAt(ev))}
						onpointerdown={(ev) => (hoverCell = cellAt(ev))}
						onpointerleave={() => (hoverCell = HOME_CELL)}
					></canvas>
				</div>

				<div class="flex flex-col gap-3 bg-surface p-4">
					<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
						<span class="eyebrow">who lives at this address?</span>
						<span class="num text-[12px] text-ink">z = {addressLabel}</span>
					</div>

					<div class="mx-auto flex w-full max-w-[196px] flex-col gap-2">
						<div class="relative">
							<canvas
								bind:this={cellCanvas}
								class="block aspect-square w-full rounded border border-line-soft"
								aria-label="The decoder's answer at the address under the cursor"
							></canvas>
							<span class="eyebrow absolute top-2 left-2.5">the decoder's answer</span>
						</div>
						<div class="relative">
							<canvas
								bind:this={realCanvas}
								class="block aspect-square w-full rounded border border-line-soft"
								aria-label="The real held-out digit nearest that address"
							></canvas>
							<span class="eyebrow absolute top-2 left-2.5">nearest real digit</span>
						</div>
					</div>

					<div class="mt-auto flex flex-col gap-2 border-t border-line-soft pt-3">
						{#if nearest}
							{@const inhabited = nearest.dist <= home}
							<div class="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
								<span class="flex items-center gap-2 text-[12.5px]">
									<i
										class="dot"
										style="background: {inhabited ? 'var(--good)' : 'var(--warm)'};"
										aria-hidden="true"
									></i>
									<span style="color: {inhabited ? 'var(--good)' : 'var(--warm)'};">
										{inhabited ? 'inhabited' : 'empty street'}
									</span>
								</span>
								<span class="num text-[10.5px] text-ink-3">
									a {nearest.label} at {nearest.dist.toFixed(2)} · digits sit ≈{lab.zNn.toFixed(2)}
									apart
								</span>
							</div>
						{/if}
						{#if viewDim === 3}
							<Slider
								label={projected ? 'slice along direction 3' : 'slice position z₃'}
								bind:value={sliceZ}
								min={(lab.zCenter[2] ?? 0) - lab.zSpan}
								max={(lab.zCenter[2] ?? 0) + lab.zSpan}
								step={lab.zSpan / 50}
								tone="warm"
								format={(v) => v.toFixed(2)}
							/>
						{/if}
					</div>
				</div>
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
