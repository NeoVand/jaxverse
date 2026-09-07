<script lang="ts">
	// The walk back, laid out flat: one picture's whole journey from static,
	// sampled at eight points along it, with the two dials that decide how far
	// apart those points are and how much fresh noise is thrown back in.
	import { Play } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/fashion/Tiles.svelte';
	import { NOTHING } from '$lib/diffusion/runtime';
	import { coalesce } from '$lib/diffusion/lab.svelte';
	import { lab } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const STEP_CHOICES = [100, 50, 20, 10, 5];
	const FRAMES = 8;
	const DIM = 28 * 28;

	let steps = $state(50);
	let eta = $state(0);
	let seed = $state(11);
	let strip = $state<Float32Array | null>(null);
	let taus = $state<number[]>([]);
	let ms = $state(0);
	let running = $state(false);

	const draw = coalesce(async () => {
		if (lab.phase === 'no-webgpu') return;
		running = true;
		try {
			const r = await lab.sample(1, {
				steps,
				a: NOTHING,
				guidanceA: 1,
				seed,
				eta,
				trace: 'state',
				fromShipped: lab.hasShipped
			});
			const frames = r.frames ?? [];
			// pick FRAMES evenly spaced moments, always including the last
			const picks = Array.from({ length: FRAMES }, (_, i) =>
				Math.min(frames.length - 1, Math.round((i * (frames.length - 1)) / (FRAMES - 1)))
			);
			const out = new Float32Array(FRAMES * DIM);
			picks.forEach((p, i) => out.set(frames[p], i * DIM));
			strip = out;
			taus = picks.map((p) => 1 - (p + 1) / steps);
			ms = r.ms;
		} catch {
			// disposed mid-flight
		}
		running = false;
	});

	// Redraw when a dial moves. A slider fires far faster than a forty-step
	// walk completes, so the requests are coalesced rather than dropped —
	// otherwise the row can end up disagreeing with its own controls.
	$effect(() => {
		void steps;
		void eta;
		void seed;
		if (lab.phase === 'ready' || lab.phase === 'training') draw();
	});
</script>

<Plate id="walk" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if strip}
			<span>{steps} steps</span>
			<span aria-hidden="true">·</span>
			<span>{ms.toFixed(0)} ms</span>
		{:else if running}
			<span>drawing…</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn
			kind="primary"
			disabled={running || lab.phase === 'no-webgpu'}
			onclick={() => (seed = Math.floor(Math.random() * 100000))}
		>
			<Play size={12} aria-hidden="true" /> Draw another
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					This plate walks a picture out of static on your GPU, which needs WebGPU. What it shows: a
					row that begins as pure noise and resolves, left to right, into one garment.
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-3 p-4 sm:p-5">
				<div class="flex items-baseline justify-between gap-3">
					<span class="eyebrow">pure noise</span>
					<span class="eyebrow">a picture</span>
				</div>
				<Tiles
					pixels={strip}
					count={FRAMES}
					columns={FRAMES}
					class="h-16 sm:h-24"
					label="One sample shown at eight points along its walk back from noise"
				/>
				<div class="num flex justify-between text-[10px] text-ink-3">
					{#each taus as t, i (i)}
						<span><span class="sym">τ</span> {t.toFixed(2)}</span>
					{/each}
				</div>

				<div class="grid gap-x-8 gap-y-3 border-t border-line-soft pt-3 sm:grid-cols-2">
					<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Denoising steps">
						<span class="eyebrow mr-1">steps</span>
						{#each STEP_CHOICES as s (s)}
							<button
								class="chip"
								class:chip-on={steps === s}
								aria-pressed={steps === s}
								disabled={running}
								onclick={() => (steps = s)}
							>
								{s}
							</button>
						{/each}
					</span>
					<Slider
						label="noise put back η"
						bind:value={eta}
						min={0}
						max={1}
						step={0.1}
						tone="knob"
						disabled={running}
						format={(v) => (v === 0 ? '0 · deterministic' : v.toFixed(1))}
					/>
				</div>
			</div>
		{/if}
	</div>
</Plate>
