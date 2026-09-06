<script lang="ts">
	// The forward process, which is the half of diffusion nobody has to learn.
	// Deliberately GPU-free: it is the chapter's first figure, and it is all
	// arithmetic on pixels, so it should work in any browser that got this far.
	import { base } from '$app/paths';
	import { Shuffle } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { loadEmoji, type EmojiCorpus } from '$lib/diffusion/corpus';
	import { alphaBar, alphaBarLinear } from '$lib/diffusion/model';
	import { mulberry32 } from '$lib/diffusion/runtime';
	import { corrupt, tilesToImageData, blitCrisp } from '$lib/viz/emoji-paint';
	import { readTokens, themePulse, watchTheme } from '$lib/viz/tokens.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const STEPS = 11; // τ from 0 to 1 inclusive
	const RES = 32;
	const DIM = 4 * RES * RES;

	let corpus = $state<EmojiCorpus | null>(null);
	let phase = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	let schedule = $state<'cosine' | 'linear'>('cosine');
	let pick = $state(0);
	let style = $state(0);
	let canvas: HTMLCanvasElement | undefined = $state();

	/** Pictures with a lot of ink survive noise longer; start on one of those. */
	const OPENERS = ['🔥', '🐱', '🍕', '🚀', '❤️', '🌈', '🎈', '👻', '🍄', '⭐'];
	let order: number[] = [];

	const alpha = $derived(schedule === 'cosine' ? alphaBar : alphaBarLinear);
	const label = $derived(corpus ? (corpus.meta.emoji[pick]?.name ?? '') : '');

	async function boot() {
		if (phase !== 'idle') return;
		phase = 'loading';
		try {
			corpus = await loadEmoji(base);
			const byCp = new Map(corpus.meta.emoji.map((e, i) => [e.cp, i]));
			const seeds = OPENERS.map((c) => byCp.get(c)).filter((i): i is number => i !== undefined);
			const rest = corpus.meta.emoji.map((_, i) => i).filter((i) => !seeds.includes(i));
			order = [...seeds, ...rest];
			pick = order[0] ?? 0;
			phase = 'ready';
		} catch (e) {
			phase = 'error';
			void e;
		}
	}

	function shuffle() {
		if (!corpus) return;
		const i = order.indexOf(pick);
		pick = order[(i + 1) % Math.min(order.length, 240)];
	}

	// Repaint on any change, and on theme flips: the background the emoji is
	// composited over is a token, so day and night are the same code path.
	$effect(() => {
		void themePulse.tick;
		void pick;
		void style;
		void schedule;
		if (phase !== 'ready' || !canvas || !corpus) return;
		const tk = readTokens(canvas);
		const src = new Float32Array(DIM);
		const off = (style * corpus.count + pick) * DIM;
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
			tilesToImageData(strip, STEPS, RES, { columns: STEPS, background: tk.band, gap: 3 })
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
			<span>{label}</span>
		{:else if phase === 'loading'}
			<span>decoding the emoji sheets…</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn disabled={phase !== 'ready'} onclick={shuffle} title="Another picture">
			<Shuffle size={12} aria-hidden="true" /> Another
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void boot()}>
		{#if phase === 'error'}
			<div class="px-4 py-8 text-[12.5px] text-bad">The emoji sheets did not load.</div>
		{:else if phase !== 'ready'}
			<div class="flex h-[190px] items-center justify-center">
				<span class="eyebrow">fetching the emoji sheets (≈2.3 MB, cached)…</span>
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
					aria-label="One emoji shown at eleven noise levels, intact at the left and indistinguishable from static at the right"
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

					<span class="flex items-center gap-1" role="group" aria-label="Drawing style">
						<span class="eyebrow mr-1">style</span>
						{#each corpus?.meta.sets ?? [] as set, i (set.id)}
							<button
								class="chip"
								class:chip-on={style === i}
								aria-pressed={style === i}
								title={set.credit}
								onclick={() => (style = i)}
							>
								{set.label}
							</button>
						{/each}
					</span>
				</div>
			</div>
		{/if}
	</div>
</Plate>
