<script lang="ts">
	// Plate III — the judge. A Bradley–Terry head, fitted to the reader's own
	// clicks, drawn as a field: the learned reward over a two-gene slice of
	// ornament space, with real specimens struck on top of it.
	//
	// The honest readout is `foresight`. Before every click, the judge — knowing
	// only what came before — is asked to call the pair. Nothing here is scored
	// against data it was trained on.
	import { Check, X } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { contours } from 'd3-contour';
	import { readTokens, themePulse, watchTheme, hexRgb } from '$lib/viz/tokens.svelte';
	import { GENES, N_GENES, sampleRefGene, score, type Gene } from '$lib/optim-rl/preference';
	import { mulberry32 } from '$lib/optim-rl/rng';
	import Rosette from './Rosette.svelte';
	import { taste, MIN_PAIRS } from './taste-context.svelte';

	// Arms across, weight down: the two genes whose far corner is the blot, so
	// the map's own geometry sets up the plate that follows.
	let ax = $state(0);
	let ay = $state(3);

	/** Field resolution — 7k forward passes of a 401-parameter net, ~3 ms. */
	const NX = 96;
	const NY = 72;
	/** Backing-store size. Fixed rather than measured: the contours want more
	 * resolution than the field grid, and CSS scales the result down cleanly. */
	const CW = 1024;
	const CH = 768;
	/** Specimens struck over the wash. */
	const SX = 7;
	const SY = 5;

	let canvas: HTMLCanvasElement | undefined = $state();
	let host: HTMLElement | undefined = $state();

	watchTheme();

	/** A gene on the slice: two axes vary, the rest sit at the reference median. */
	function sliceGene(u: number, v: number, out?: Gene): Gene {
		const g = out ?? new Float64Array(N_GENES);
		g.fill(0.5);
		g[ax] = u;
		g[ay] = v;
		return g;
	}

	// Specimens depend on the axes and nothing else — they are the ornaments
	// those coordinates draw, not the judge's opinion of them. Keeping the
	// judge out of this dependency is what stops 800 SVG paths re-rendering on
	// every single click in the plate above.
	const specimens = $derived.by(() => {
		const out: { gene: Gene; x: number; y: number }[] = [];
		for (let j = 0; j < SY; j++) {
			for (let i = 0; i < SX; i++) {
				const u = (i + 0.5) / SX;
				const v = (j + 0.5) / SY;
				out.push({ gene: sliceGene(u, 1 - v), x: (i + 0.5) / SX, y: (j + 0.5) / SY });
			}
		}
		return out;
	});

	/**
	 * The best and worst of two thousand honest draws from the reference.
	 * Note what this is *not*: it never leaves the distribution the comparisons
	 * came from. It is best-of-n, the baseline the next plate has to beat — and
	 * the next plate beats it by leaving.
	 */
	const poll = $derived.by(() => {
		void taste.version;
		if (!taste.ready) return null;
		const rand = mulberry32(31337);
		let best: Gene | null = null;
		let worst: Gene | null = null;
		let bs = -Infinity;
		let ws = Infinity;
		for (let k = 0; k < 2000; k++) {
			const g = sampleRefGene(rand);
			const s = score(taste.judge, g);
			if (s > bs) {
				bs = s;
				best = g;
			}
			if (s < ws) {
				ws = s;
				worst = g;
			}
		}
		return { best: best!, worst: worst!, bs, ws };
	});

	$effect(() => {
		void taste.version;
		void themePulse.tick;
		void ax;
		void ay;
		const cv = canvas;
		if (!cv || !host) return;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		const tk = readTokens(host);

		cv.width = CW;
		cv.height = CH;
		const hi = hexRgb(tk.accent);
		const lo = hexRgb(tk.warm);
		const bg = hexRgb(tk.band);

		ctx.clearRect(0, 0, CW, CH);
		ctx.fillStyle = tk.band;
		ctx.fillRect(0, 0, CW, CH);
		// nothing voted for yet — leave the plate honestly blank
		if (!taste.ready) return;

		const g = new Float64Array(N_GENES);
		const field = new Float64Array(NX * NY);
		let min = Infinity;
		let max = -Infinity;
		for (let j = 0; j < NY; j++) {
			for (let i = 0; i < NX; i++) {
				const s = score(taste.judge, sliceGene((i + 0.5) / NX, 1 - (j + 0.5) / NY, g));
				field[j * NX + i] = s;
				if (s < min) min = s;
				if (s > max) max = s;
			}
		}

		// ── the wash, kept very quiet: it is the ground the specimens are struck
		// on, not the figure. Symmetric about zero, so "better than the average
		// ornament" is the same blue whatever the reader's taste turned out to be.
		const span = Math.max(0.35, Math.max(Math.abs(min), Math.abs(max)));
		const small = document.createElement('canvas');
		small.width = NX;
		small.height = NY;
		const sctx = small.getContext('2d');
		if (!sctx) return;
		const img = sctx.createImageData(NX, NY);
		for (let p = 0; p < NX * NY; p++) {
			const t = Math.max(-1, Math.min(1, field[p] / span));
			const c = t >= 0 ? hi : lo;
			const a = Math.pow(Math.abs(t), 1.1) * 0.17;
			img.data[p * 4] = Math.round(bg[0] + (c[0] - bg[0]) * a);
			img.data[p * 4 + 1] = Math.round(bg[1] + (c[1] - bg[1]) * a);
			img.data[p * 4 + 2] = Math.round(bg[2] + (c[2] - bg[2]) * a);
			img.data[p * 4 + 3] = 255;
		}
		sctx.putImageData(img, 0, 0);
		ctx.imageSmoothingEnabled = true;
		ctx.imageSmoothingQuality = 'high';
		ctx.drawImage(small, 0, 0, NX, NY, 0, 0, CW, CH);

		// ── the contours. A field of preference is a landscape, and this book
		// already taught the reader to read one — these are the same hairline
		// rings the Prologue's loss surface is drawn with, and they carry the
		// structure the wash is too quiet to. The zero ring is drawn heaviest,
		// because standardized scores make it mean something exact: the line
		// between better and worse than an average ornament.
		const LEVELS = 13;
		const thresholds = Array.from(
			{ length: LEVELS },
			(_, k) => min + ((k + 0.5) / LEVELS) * (max - min)
		);
		const rings = contours()
			.size([NX, NY])
			.thresholds([...thresholds, 0].sort((a, b) => a - b))(field as unknown as number[]);

		for (const c of rings) {
			const zero = Math.abs(c.value) < 1e-9;
			ctx.beginPath();
			for (const poly of c.coordinates)
				for (const ring of poly) {
					for (let i = 0; i < ring.length; i++) {
						const X = (ring[i][0] / NX) * CW;
						const Y = (ring[i][1] / NY) * CH;
						if (i === 0) ctx.moveTo(X, Y);
						else ctx.lineTo(X, Y);
					}
					ctx.closePath();
				}
			ctx.strokeStyle = c.value >= 0 ? tk.accent : tk.warm;
			ctx.globalAlpha = zero ? 0.5 : 0.16 + 0.2 * Math.abs(c.value / span);
			ctx.lineWidth = zero ? 2.4 : 1.2;
			ctx.stroke();
		}
		ctx.globalAlpha = 1;
	});

	const calls = $derived(taste.calls.slice(-18));
	const pct = (v: number) => `${Math.round(v * 100)}%`;
</script>

<Plate
	id="judge"
	title="The judge, and a map of your taste"
	live
	caption="Four hundred parameters, fitted by maximum likelihood to nothing but your comparisons — the same Bradley–Terry fit that turns chess results into Elo ratings. The wash is what it learned, drawn across two genes with the other four held at the middle: ultramarine where it expects you to approve, vermilion where it expects you to pass. The specimens struck on top are real ornaments from those coordinates, so you can check the field against the thing it is describing. Change the axes and the same six-dimensional opinion is sliced a different way."
>
	{#snippet status()}
		{#if taste.ready}
			<span>{taste.count} pairs</span>
			<span class="text-ink-3">· loss {taste.loss.toFixed(3)}</span>
		{:else}
			<span class="text-ink-3">waiting for {MIN_PAIRS - taste.count} more</span>
		{/if}
	{/snippet}

	<div class="jd px-4" bind:this={host}>
		<div class="axes">
			{#each ['across', 'down'] as const as which (which)}
				<div class="axrow">
					<span class="axlabel">{which}</span>
					{#each GENES as gene, i (gene.key)}
						<button
							class="chip"
							class:chip-on={(which === 'across' ? ax : ay) === i}
							disabled={(which === 'across' ? ay : ax) === i}
							title={gene.note}
							onclick={() => (which === 'across' ? (ax = i) : (ay = i))}
						>
							{gene.label}
						</button>
					{/each}
				</div>
			{/each}
		</div>

		<div class="body">
			<figure class="map">
				<canvas bind:this={canvas} aria-label="The judge's score across two genes"></canvas>
				{#if taste.ready}
					<div class="specimens" aria-hidden="true">
						{#each specimens as s, i (i)}
							<span style="left:{s.x * 100}%; top:{s.y * 100}%">
								<Rosette gene={s.gene} size={50} frame={false} />
							</span>
						{/each}
					</div>
				{:else}
					<p class="empty font-serif">Judge a few pairs above and this field fills in.</p>
				{/if}
				<figcaption>
					<span>{GENES[ax].label} →</span>
					<span>↑ {GENES[ay].label}</span>
				</figcaption>
			</figure>

			<div class="reads">
				<div class="big">
					<span class="eyebrow">called in advance</span>
					<strong class="num">{taste.foresight === null ? '—' : pct(taste.foresight)}</strong>
					<p>
						Before each click, the judge — fitted only to everything earlier — guessed which
						ornament you were about to choose. This is how often it was right, on comparisons it had
						never seen.
					</p>
				</div>

				{#if calls.length}
					<div class="tally" aria-label="Recent predictions, oldest first">
						{#each calls as ok, i (taste.calls.length - calls.length + i)}
							<span class="mark" class:hit={ok}>
								{#if ok}<Check size={11} />{:else}<X size={11} />{/if}
							</span>
						{/each}
					</div>
				{/if}

				<dl class="stats num">
					<div>
						<dt>comparisons</dt>
						<dd>{taste.count}</dd>
					</div>
					<div>
						<dt>parameters</dt>
						<dd>401</dd>
					</div>
					<div>
						<dt>fits your own verdicts</dt>
						<dd>{pct(taste.fit)}</dd>
					</div>
				</dl>
				{#if taste.ready && taste.fit > 0.95 && taste.count < 40}
					<p class="warn">
						It reproduces nearly every verdict you gave it. With four hundred parameters and
						{taste.count} comparisons, some of that is memory rather than taste — which is exactly the
						gap the next plate walks into.
					</p>
				{/if}
			</div>
		</div>

		{#if poll}
			<div class="picks">
				<div class="pick">
					<span class="eyebrow">its favourite of 2,000 draws</span>
					<Rosette gene={poll.best} size={112} label="The judge's highest-scoring ornament" />
					<span class="num">r = {poll.bs.toFixed(2)}</span>
				</div>
				<p class="gloss font-serif">
					Two thousand ornaments drawn from the same pile your comparisons came from, ranked by the
					judge, best and worst kept. Nothing here left the distribution: this is
					<em>best-of-n</em>, the cheap baseline that costs no gradients at all. Hold on to how good
					it looks. The next plate stops sampling and starts optimizing.
				</p>
				<div class="pick">
					<span class="eyebrow">its least favourite</span>
					<Rosette gene={poll.worst} size={112} label="The judge's lowest-scoring ornament" />
					<span class="num">r = {poll.ws.toFixed(2)}</span>
				</div>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.jd {
		padding-bottom: 0.25rem;
	}
	.axes {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		margin-bottom: 0.9rem;
	}
	.axrow {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.3rem;
	}
	.axlabel {
		font-family: var(--font-sans);
		font-size: 10px;
		letter-spacing: 0.11em;
		text-transform: uppercase;
		color: var(--ink-3);
		width: 3.6rem;
	}
	.body {
		display: grid;
		grid-template-columns: minmax(0, 1.85fr) minmax(0, 1fr);
		gap: 1.25rem;
		align-items: start;
	}
	.map {
		position: relative;
		margin: 0;
		border: 1px solid var(--line-soft);
		border-radius: var(--r-2);
		overflow: hidden;
		background: var(--band);
	}
	.map canvas {
		display: block;
		width: 100%;
		aspect-ratio: 4 / 3;
		height: auto;
		image-rendering: auto;
	}
	.specimens span {
		position: absolute;
		transform: translate(-50%, -50%);
		pointer-events: none;
		/* struck into the map rather than pasted onto it, so the contours read
		   through and the specimens stop looking like stickers */
		opacity: 0.72;
	}
	.empty {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		margin: 0;
		font-style: italic;
		color: var(--ink-3);
		font-size: 15px;
	}
	.map figcaption {
		position: absolute;
		inset: auto 0 0 0;
		display: flex;
		justify-content: space-between;
		padding: 0.3rem 0.5rem;
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--ink-3);
		background: color-mix(in srgb, var(--band) 72%, transparent);
	}
	.reads {
		display: flex;
		flex-direction: column;
		gap: 0.85rem;
	}
	.big strong {
		display: block;
		font-size: 2.6rem;
		line-height: 1.05;
		color: var(--ink);
		margin: 0.15rem 0 0.35rem;
	}
	.big p {
		margin: 0;
		font-size: 12.5px;
		line-height: 1.5;
		color: var(--ink-2);
	}
	.tally {
		display: flex;
		flex-wrap: wrap;
		gap: 2px;
	}
	.mark {
		display: inline-grid;
		place-items: center;
		width: 16px;
		height: 16px;
		border-radius: var(--r-1);
		background: var(--surface-2);
		color: var(--bad);
	}
	.mark.hit {
		color: var(--good);
	}
	.stats {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		margin: 0;
		font-size: 11.5px;
		border-top: 1px solid var(--line-soft);
		padding-top: 0.6rem;
	}
	.stats > div {
		display: flex;
		justify-content: space-between;
		gap: 0.75rem;
	}
	.stats dt {
		color: var(--ink-3);
	}
	.stats dd {
		margin: 0;
		color: var(--ink-2);
	}
	.warn {
		margin: 0;
		font-size: 12px;
		line-height: 1.5;
		color: var(--ink-2);
		border-left: 2px solid var(--warm);
		padding-left: 0.6rem;
	}
	.picks {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		gap: 1.25rem;
		align-items: center;
		margin-top: 1.15rem;
		padding-top: 1rem;
		border-top: 1px solid var(--line-soft);
	}
	.pick {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.3rem;
	}
	.pick .num {
		font-size: 11px;
		color: var(--ink-3);
	}
	.gloss {
		margin: 0;
		font-size: 14.5px;
		line-height: 1.6;
		color: var(--ink-2);
		font-variation-settings: 'opsz' 16;
	}
	@media (max-width: 860px) {
		.body {
			grid-template-columns: minmax(0, 1fr);
		}
		.picks {
			grid-template-columns: 1fr 1fr;
		}
		.gloss {
			grid-column: 1 / -1;
			order: 2;
		}
	}
</style>
