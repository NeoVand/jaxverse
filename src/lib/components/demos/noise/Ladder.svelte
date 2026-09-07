<script lang="ts">
	// The forward process, which is the half of diffusion nobody has to learn.
	// Deliberately GPU-free: it is the chapter's first figure, and it is all
	// arithmetic on pixels, so it should work in any browser that got this far.
	import { base } from '$app/paths';
	import { Shuffle } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { loadFashion, type FashionCorpus } from '$lib/diffusion/corpus';
	import { alphaBar, alphaBarLinear } from '$lib/diffusion/model';
	import { mulberry32 } from '$lib/diffusion/runtime';
	import { corrupt, tilesToImageData, blitCrisp } from '$lib/viz/fashion-paint';
	import { readTokens, themePulse, watchTheme } from '$lib/viz/tokens.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const STEPS = 11; // τ from 0 to 1 inclusive
	const RES = 28;
	const DIM = RES * RES;

	let corpus = $state<FashionCorpus | null>(null);
	let phase = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	let schedule = $state<'cosine' | 'linear'>('cosine');
	let klass = $state(8); // a bag: solid, so the ruin is easy to follow
	let nth = $state(0);

	let canvas: HTMLCanvasElement | undefined = $state();

	/** Row indices grouped by label, so the chips can flip between garments. */
	let byClass: number[][] = [];

	const alpha = $derived(schedule === 'cosine' ? alphaBar : alphaBarLinear);
	const names = $derived(corpus?.meta.classes ?? []);
	const pick = $derived(byClass[klass]?.[nth % (byClass[klass]?.length || 1)] ?? 0);

	async function boot() {
		if (phase !== 'idle') return;
		phase = 'loading';
		try {
			const c = await loadFashion(base);
			byClass = Array.from({ length: c.meta.classes.length }, () => []);
			// a few dozen of each is plenty for a shuffle button and keeps the
			// arrays small
			for (let i = 0; i < c.count && i < 4000; i++) byClass[c.labels[i]].push(i);
			corpus = c;
			phase = 'ready';
		} catch (e) {
			phase = 'error';
			void e;
		}
	}

	// Repaint on any change, and on theme flips: the paper the garment is drawn
	// on is a token, so day and night are the same code path.
	$effect(() => {
		void themePulse.tick;
		void pick;
		void schedule;
		if (phase !== 'ready' || !canvas || !corpus) return;
		const tk = readTokens(canvas);
		const src = new Float32Array(DIM);
		const off = pick * DIM;
		for (let i = 0; i < DIM; i++) src[i] = corpus.images[off + i] / 127.5 - 1;

		const strip = new Float32Array(STEPS * DIM);
		for (let s = 0; s < STEPS; s++) {
			const tau = s / (STEPS - 1);
			// the same seed at every level, so what changes across the row is
			// the amount of noise and not which noise
			corrupt(src, strip.subarray(s * DIM, (s + 1) * DIM), Math.sqrt(alpha(tau)), mulberry32(9));
		}
		blitCrisp(
			canvas,
			tilesToImageData(strip, STEPS, RES, {
				columns: STEPS,
				background: tk.band,
				ink: tk.ink,
				gap: 3
			})
		);
	});

	$effect(() => watchTheme());

	// the α̅ curve, drawn small beside the schedule switch. The viewBox is wide
	// and the box is not, so the shape is legible rather than a flat line.
	const curve = (fn: (t: number) => number) => {
		const pts: string[] = [];
		for (let i = 0; i <= 60; i++) {
			const t = i / 60;
			pts.push(`${(t * 200).toFixed(1)},${(22 - fn(t) * 20).toFixed(1)}`);
		}
		return `M ${pts.join(' L ')}`;
	};
</script>

<Plate id="ladder" {title} {caption}>
	{#snippet status()}
		{#if phase === 'ready'}
			<span>{names[klass] ?? ''}</span>
		{:else if phase === 'loading'}
			<span>decoding the garment sheet…</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn disabled={phase !== 'ready'} onclick={() => (nth += 1)} title="Another picture">
			<Shuffle size={12} aria-hidden="true" /> Another
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void boot()}>
		{#if phase === 'error'}
			<div class="px-4 py-8 text-[12.5px] text-bad">The garment sheet did not load.</div>
		{:else if phase !== 'ready'}
			<div class="flex h-[190px] items-center justify-center">
				<span class="eyebrow">fetching the garment sheet (≈5.3 MB, cached)…</span>
			</div>
		{:else}
			<div class="flex flex-col gap-3 p-4 sm:p-5">
				<div class="flex items-baseline justify-between gap-3">
					<span class="eyebrow">the picture</span>
					<span class="eyebrow">pure noise</span>
				</div>
				<canvas
					bind:this={canvas}
					class="block h-14 w-full sm:h-20"
					aria-label="One garment shown at eleven noise levels, intact at the left and indistinguishable from static at the right"
				></canvas>
				<div class="num flex justify-between text-[10px] text-ink-3">
					{#each Array.from({ length: STEPS }, (_, i) => i) as s (s)}
						<span>{(s / (STEPS - 1)).toFixed(1)}</span>
					{/each}
				</div>

				<div class="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line-soft pt-3">
					<span class="flex items-center gap-1" role="group" aria-label="Noise schedule">
						<span class="eyebrow mr-1">schedule</span>
						{#each ['cosine', 'linear'] as const as s (s)}
							<button
								class="chip"
								class:chip-on={schedule === s}
								aria-pressed={schedule === s}
								onclick={() => (schedule = s)}
							>
								{s}
							</button>
						{/each}
					</span>

					<span class="flex shrink-0 items-center gap-1.5">
						<span class="eyebrow shrink-0 text-[9.5px]">
							<span class="sym">ᾱ</span>(<span class="sym">τ</span>)
						</span>
						<svg
							viewBox="0 0 200 24"
							class="block h-[34px] w-[132px]"
							role="img"
							aria-label="How much of the picture survives at each noise level, for both schedules"
						>
							<path
								d={curve(schedule === 'cosine' ? alphaBarLinear : alphaBar)}
								fill="none"
								stroke="var(--ink-3)"
								stroke-width="1"
								stroke-dasharray="3 3"
								vector-effect="non-scaling-stroke"
							/>
							<path
								d={curve(alpha)}
								fill="none"
								stroke="var(--cat-1)"
								stroke-width="1.4"
								vector-effect="non-scaling-stroke"
							/>
						</svg>
					</span>
				</div>

				<div class="flex flex-wrap items-center gap-1" role="group" aria-label="Garment">
					<span class="eyebrow mr-1">garment</span>
					{#each names as name, i (name)}
						<button
							class="chip"
							class:chip-on={klass === i}
							aria-pressed={klass === i}
							onclick={() => {
								klass = i;
								nth = 0;
							}}
						>
							{name}
						</button>
					{/each}
				</div>
			</div>
		{/if}
	</div>
</Plate>
