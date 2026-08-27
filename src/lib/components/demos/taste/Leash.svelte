<script lang="ts">
	// Plate V — the leash, in both of its forms.
	//
	// Left: the closed-form answer. Maximizing reward minus β·KL to a reference
	// has an exact solution, π* ∝ π_ref · exp(r/β), and here it is drawn — the
	// reference density along one gene, the exponential tilt the reward applies
	// to it, and the product. Slide β and watch a leash become a noose.
	//
	// Right: the same β handed to the optimizer of the previous plate. Nothing
	// changed but one number in the objective.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import {
		ascendStep,
		createPolicy,
		expectedScore,
		GENES,
		geneFromU,
		klToRef,
		N_GENES,
		REF_SIGMA,
		score,
		type Gene
	} from '$lib/optim-rl/preference';
	import { mulberry32 } from '$lib/optim-rl/rng';
	import Rosette from './Rosette.svelte';
	import { taste } from './taste-context.svelte';

	/** β on a log dial: 0.05 (a loose leash) to 8 (a very tight one). */
	let logBeta = $state(-0.4);
	const beta = $derived(Number(Math.pow(10, logBeta).toPrecision(2)));
	/** Which gene the tilt is drawn along. Weight, because it is the one the
	 * unleashed optimizer usually runs away with. */
	let axis = $state(3);

	const N = 200;
	const W = 480;
	const H = 190;
	const PAD = { l: 16, r: 16, t: 14, b: 20 };

	/** The reference density over one gene: a Gaussian in u, squashed. */
	function refDensity(g: number): number {
		const p = Math.min(1 - 1e-9, Math.max(1e-9, g));
		const u = Math.log(p / (1 - p));
		return Math.exp(-(u * u) / (2 * REF_SIGMA * REF_SIGMA)) / (REF_SIGMA * p * (1 - p));
	}

	const curves = $derived.by(() => {
		void taste.version;
		const gene = new Float64Array(N_GENES).fill(0.5);
		const xs: number[] = [];
		const ref: number[] = [];
		const tilt: number[] = [];
		const star: number[] = [];
		let refSum = 0;
		let starSum = 0;
		let tiltMax = 0;
		for (let i = 0; i < N; i++) {
			const g = (i + 0.5) / N;
			gene[axis] = g;
			const r = taste.ready ? score(taste.judge, gene) : 0;
			const d = refDensity(g);
			const e = Math.exp(r / beta);
			xs.push(g);
			ref.push(d);
			tilt.push(e);
			star.push(d * e);
			refSum += d;
			starSum += d * e;
			if (e > tiltMax) tiltMax = e;
		}
		// Densities are normalized to their own peak so all three fit one frame;
		// the shapes are the content, not the heights.
		const norm = (a: number[]) => {
			const m = Math.max(...a) || 1;
			return a.map((v) => v / m);
		};
		return {
			xs,
			ref: norm(ref),
			tilt: norm(tilt),
			star: norm(star),
			/** How much of π*'s mass sits where π_ref has almost none. */
			drift: star.reduce(
				(a, v, i) => a + (v / starSum) * Math.log(v / starSum / (ref[i] / refSum) + 1e-30),
				0
			)
		};
	});

	const px = (g: number) => PAD.l + g * (W - PAD.l - PAD.r);
	const py = (v: number) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);
	const path = (ys: number[]) =>
		ys
			.map((v, i) => `${i ? 'L' : 'M'} ${px(curves.xs[i]).toFixed(1)} ${py(v).toFixed(1)}`)
			.join(' ');

	interface Run {
		gene: Gene;
		kl: number;
		proxy: number;
	}

	/** One optimizer run to convergence at a given leash length. Deliberately
	 * cheap — 900 steps of 8 samples is ~15 ms, so the β slider stays live
	 * under the hand instead of stuttering. */
	function go(b: number): Run {
		const p = createPolicy();
		const rand = mulberry32(515);
		for (let k = 0; k < 900; k++) ascendStep(taste.judge, p, rand, { beta: b, samples: 8 });
		return {
			gene: geneFromU(Float64Array.from(p.mu)),
			kl: klToRef(p),
			proxy: expectedScore(taste.judge, p, mulberry32(88), 192)
		};
	}

	// The two ends of the triptych never move with β, so they are computed once
	// per judge rather than once per slider tick.
	const ends = $derived.by(() => {
		void taste.version;
		if (!taste.ready) return null;
		const ref = createPolicy();
		return {
			start: {
				gene: geneFromU(Float64Array.from(ref.mu)),
				kl: 0,
				proxy: expectedScore(taste.judge, ref, mulberry32(88), 192)
			} satisfies Run,
			free: go(0)
		};
	});

	const runs = $derived(ends ? { start: ends.start, free: ends.free, leashed: go(beta) } : null);

	const fmtBeta = (v: number) => {
		const b = Math.pow(10, v);
		return b >= 1 ? b.toFixed(1) : b.toFixed(2);
	};
</script>

<Plate
	id="leash"
	title="The leash"
	live
	caption="Maximize reward minus β times the KL back to the reference, and the answer is not approximate — it is π* ∝ π_ref · exp(r/β), the reference tilted exponentially by reward. Left, that formula drawn along one gene: the pale curve is where ornaments came from, the dashed one is the tilt the judge applies, the solid one is the product. Right, the same β handed to the optimizer. Loosen it far enough and the tilt sharpens into a spike on whatever the judge loves most, which is exactly the plate above. Tighten it and the policy never leaves home. In between is the only place anything good happens."
>
	{#snippet status()}
		<span>β = {beta}</span>
		{#if runs}<span class="text-ink-3">· {runs.leashed.kl.toFixed(1)} nats travelled</span>{/if}
	{/snippet}

	<div class="ls px-4">
		<div class="controls">
			<div class="w-56 shrink-0">
				<Slider
					label="β — the leash"
					bind:value={logBeta}
					min={-1.3}
					max={0.9}
					step={0.05}
					format={fmtBeta}
					tone="knob"
				/>
			</div>
			<div class="axpick">
				<span class="axlabel">along</span>
				{#each GENES as gene, i (gene.key)}
					<button
						class="chip"
						class:chip-on={axis === i}
						onclick={() => (axis = i)}
						title={gene.note}
					>
						{gene.label}
					</button>
				{/each}
			</div>
		</div>

		<div class="body">
			<figure class="tilt">
				<svg
					viewBox="0 0 {W} {H}"
					role="img"
					aria-label="The reference density, the reward tilt, and their product"
				>
					<line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} class="axis" />
					<path d={path(curves.ref)} class="ref" fill="none" />
					<path d={path(curves.tilt)} class="tiltline" fill="none" stroke-dasharray="4 3" />
					<path d={path(curves.star)} class="star" fill="none" />
				</svg>
				<figcaption class="num">
					<span><i class="sw ref-sw"></i>π_ref</span>
					<span><i class="sw tilt-sw"></i>exp(r/β)</span>
					<span><i class="sw star-sw"></i>π*</span>
					<span class="ml-auto">{GENES[axis].label} →</span>
				</figcaption>
			</figure>

			<div class="triptych">
				{#if runs}
					<figure>
						<Rosette gene={runs.start.gene} size={96} label="The reference ornament" />
						<figcaption>
							<span class="eyebrow">reference</span>
							<span class="num">0.0 nats · r {runs.start.proxy.toFixed(2)}</span>
						</figcaption>
					</figure>
					<figure class="hero">
						<Rosette gene={runs.leashed.gene} size={116} label="The leashed optimizer's ornament" />
						<figcaption>
							<span class="eyebrow" style="color: var(--cat-1)"
								>leashed · <span class="sym">β</span> = {beta}</span
							>
							<span class="num"
								>{runs.leashed.kl.toFixed(1)} nats · r {runs.leashed.proxy.toFixed(2)}</span
							>
						</figcaption>
					</figure>
					<figure>
						<Rosette gene={runs.free.gene} size={96} label="The unleashed optimizer's ornament" />
						<figcaption>
							<span class="eyebrow" style="color: var(--warm)">unleashed</span>
							<span class="num"
								>{runs.free.kl.toFixed(1)} nats · r {runs.free.proxy.toFixed(2)}</span
							>
						</figcaption>
					</figure>
				{:else}
					<p class="idle font-serif">Fit a judge above and this triptych fills in.</p>
				{/if}
			</div>
		</div>

		<p class="note font-serif">
			Read the ordering once, because it is the whole safety property. The penalty is KL(π ‖ π_ref)
			— averaged under <em>your own</em> policy, so it only ever notices places you actually go, and bites
			hardest exactly where you put mass the reference never would. Write it the other way round and it
			would patrol territory you never visit and shrug at the blot.
		</p>
	</div>
</Plate>

<style>
	.ls {
		padding-bottom: 0.4rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: end;
		gap: 0.75rem 1.5rem;
		margin-bottom: 0.9rem;
	}
	.axpick {
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
	}
	.body {
		display: grid;
		grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
		gap: 1.5rem;
		align-items: center;
	}
	.tilt {
		margin: 0;
	}
	.tilt svg {
		display: block;
		width: 100%;
		height: auto;
	}
	.axis {
		stroke: var(--line);
		stroke-width: 1;
	}
	.ref {
		stroke: var(--ink-3);
		stroke-width: 1.4;
	}
	.tiltline {
		stroke: var(--cat-1);
		stroke-width: 1.4;
	}
	.star {
		stroke: var(--accent);
		stroke-width: 2.1;
	}
	.tilt figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 0.9rem;
		font-size: 10px;
		color: var(--ink-3);
		margin-top: 0.2rem;
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 2px;
		vertical-align: middle;
		margin-right: 0.35rem;
	}
	.ref-sw {
		background: var(--ink-3);
	}
	.tilt-sw {
		background: var(--cat-1);
	}
	.star-sw {
		background: var(--accent);
	}
	.triptych {
		display: grid;
		grid-template-columns: repeat(3, auto);
		justify-content: center;
		align-items: end;
		gap: 0.75rem;
	}
	.triptych figure {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.3rem;
		margin: 0;
	}
	.triptych figcaption {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.1rem;
	}
	.triptych .num {
		font-size: 10px;
		color: var(--ink-3);
	}
	.hero {
		padding: 0.45rem 0.55rem 0.35rem;
		border-radius: var(--r-3);
		background: color-mix(in srgb, var(--cat-1) 8%, transparent);
	}
	.idle {
		grid-column: 1 / -1;
		text-align: center;
		font-style: italic;
		color: var(--ink-3);
		margin: 0;
		padding: 2rem 0;
	}
	.note {
		margin: 1.1rem 0 0;
		padding-top: 0.85rem;
		border-top: 1px solid var(--line-soft);
		font-size: 14.5px;
		line-height: 1.62;
		color: var(--ink-2);
		font-variation-settings: 'opsz' 16;
	}
	@media (max-width: 820px) {
		.body {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
