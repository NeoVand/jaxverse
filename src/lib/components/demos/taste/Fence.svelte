<script lang="ts">
	// Plate VI — the fence. PPO's clipped objective, drawn rather than asserted.
	//
	// The leash of the previous plate is a *penalty*: it prices drift. This is a
	// different instrument for a related fear — the policy generates its own
	// training data, so one step too large poisons every sample that follows.
	// The clip does not forbid the step. It simply stops paying for it, and the
	// asymmetry between the two panels is the entire design.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';

	let eps = $state(0.2);
	let rho = $state(1.35);

	const W = 250;
	const H = 176;
	const PAD = { l: 30, r: 10, t: 14, b: 26 };
	const RHO_MAX = 2;
	const V = 2.1; // vertical half-range of the objective axis

	const x = (r: number) => PAD.l + (r / RHO_MAX) * (W - PAD.l - PAD.r);
	const y = (v: number) => PAD.t + ((V - v) / (2 * V)) * (H - PAD.t - PAD.b);

	const clip = (r: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, r));
	/** PPO's objective for one action, at advantage `adv` and ratio `r`. */
	const L = (r: number, adv: number) => Math.min(r * adv, clip(r, 1 - eps, 1 + eps) * adv);
	/** Its slope in ρ — zero is the whole point, so it is worth reporting. */
	const slope = (r: number, adv: number) => (L(r + 1e-4, adv) - L(r - 1e-4, adv)) / 2e-4;

	const N = 160;
	const curve = (adv: number, f: (r: number) => number) =>
		Array.from({ length: N + 1 }, (_, i) => {
			const r = (i / N) * RHO_MAX;
			return `${i ? 'L' : 'M'} ${x(r).toFixed(1)} ${y(f(r)).toFixed(1)}`;
		}).join(' ');

	const panels = $derived([
		{ adv: 1, title: 'a good action', note: 'Â > 0 — we want it likelier' },
		{ adv: -1, title: 'a bad action', note: 'Â < 0 — we want it rarer' }
	]);
</script>

<Plate
	id="fence"
	title="The fence"
	caption="A second instrument, for a related fear: a policy generates its own training data, so one step too large does not merely cost you an update — it poisons every sample that follows. PPO's answer is to keep computing the honest objective and the fenced one and always take whichever is worse for you. Slide the ratio past a fence and read the slope: on the side where you would be profiting it goes to zero, so the optimizer simply stops pushing. On the side where you would be losing it never flattens, which is the only reason an overshoot can ever be hauled back."
>
	{#snippet status()}
		<span>ε = {eps.toFixed(2)}</span>
		<span class="text-ink-3">· ρ = {rho.toFixed(2)}</span>
	{/snippet}

	<div class="fn px-4">
		<div class="controls">
			<div class="w-48">
				<Slider
					label="ε — the fence"
					bind:value={eps}
					min={0.05}
					max={0.5}
					step={0.01}
					format={(v) => v.toFixed(2)}
					tone="knob"
				/>
			</div>
			<div class="w-48">
				<Slider
					label="ρ — the ratio"
					bind:value={rho}
					min={0.2}
					max={2}
					step={0.01}
					format={(v) => v.toFixed(2)}
					tone="ink"
				/>
			</div>
		</div>

		<div class="grid">
			{#each panels as p (p.adv)}
				<figure>
					<svg
						viewBox="0 0 {W} {H}"
						role="img"
						aria-label="{p.title}: the clipped objective against the probability ratio"
					>
						<!-- the fences -->
						<rect
							x={x(0)}
							y={PAD.t}
							width={x(1 - eps) - x(0)}
							height={H - PAD.t - PAD.b}
							class="out"
						/>
						<rect
							x={x(1 + eps)}
							y={PAD.t}
							width={x(RHO_MAX) - x(1 + eps)}
							height={H - PAD.t - PAD.b}
							class="out"
						/>
						<line x1={x(0)} y1={y(0)} x2={x(RHO_MAX)} y2={y(0)} class="zero" />
						<line x1={x(1)} y1={PAD.t} x2={x(1)} y2={H - PAD.b} class="one" />

						<path
							d={curve(p.adv, (r) => r * p.adv)}
							class="raw"
							fill="none"
							stroke-dasharray="3 3"
						/>
						<path d={curve(p.adv, (r) => L(r, p.adv))} class="fenced" fill="none" />

						<line x1={x(rho)} y1={PAD.t} x2={x(rho)} y2={H - PAD.b} class="marker" />
						<circle cx={x(rho)} cy={y(L(rho, p.adv))} r="3.5" class="knob" />

						<text x={x(1)} y={H - PAD.b + 12} text-anchor="middle" class="tick">1</text>
						<text x={x(1 - eps)} y={H - PAD.b + 12} text-anchor="middle" class="tick">1−ε</text>
						<text x={x(1 + eps)} y={H - PAD.b + 12} text-anchor="middle" class="tick">1+ε</text>
					</svg>
					<figcaption>
						<span class="eyebrow">{p.title}</span>
						<span class="num note">{p.note}</span>
						<span class="num slope" class:dead={Math.abs(slope(rho, p.adv)) < 1e-6}>
							slope {slope(rho, p.adv).toFixed(2)}
							{#if Math.abs(slope(rho, p.adv)) < 1e-6}· nothing left to gain{/if}
						</span>
					</figcaption>
				</figure>
			{/each}
		</div>

		<p class="note-p font-serif">
			Now the question worth carrying into the last chapter: <em>why clip at all?</em> Generate a batch,
			take exactly one gradient step on it, and the ratio is 1 everywhere — the fence is never touched
			and the whole apparatus is dead weight. It earns its keep only when generation is expensive enough
			that you take several steps per batch, and the data goes stale under you. How much clipping machinery
			a method carries is a direct readout of how many updates it takes per generation.
		</p>
	</div>
</Plate>

<style>
	.fn {
		padding-bottom: 0.4rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 0.6rem 2.5rem;
		max-width: 48rem;
		margin: 0 auto 0.75rem;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
		gap: 1rem 2.25rem;
		/* centred rather than hugging the left edge of a 72rem rail — a
		   two-panel figure adrift in that much paper reads as unfinished */
		max-width: 48rem;
		margin: 0 auto;
	}
	figure {
		margin: 0;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
	}
	.out {
		fill: var(--surface-2);
		opacity: 0.75;
	}
	.zero {
		stroke: var(--line);
		stroke-width: 1;
	}
	.one {
		stroke: var(--line);
		stroke-width: 1;
		stroke-dasharray: 2 3;
	}
	.raw {
		stroke: var(--ink-3);
		stroke-width: 1.2;
	}
	.fenced {
		stroke: var(--accent);
		stroke-width: 2.1;
	}
	.marker {
		stroke: var(--warm);
		stroke-width: 1;
		opacity: 0.5;
	}
	.knob {
		fill: var(--warm);
	}
	.tick {
		font-family: var(--font-mono);
		font-size: 9px;
		fill: var(--ink-3);
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.3rem 0.75rem;
		margin-top: 0.15rem;
	}
	.note {
		font-size: 10.5px;
		color: var(--ink-3);
	}
	.slope {
		font-size: 10.5px;
		color: var(--ink-2);
		margin-left: auto;
	}
	.slope.dead {
		color: var(--warm);
	}
	.note-p {
		max-width: 48rem;
		margin: 1.2rem auto 0;
		padding-top: 0.9rem;
		border-top: 1px solid var(--line-soft);
		font-size: 14.5px;
		line-height: 1.62;
		color: var(--ink-2);
		font-variation-settings: 'opsz' 16;
	}
</style>
