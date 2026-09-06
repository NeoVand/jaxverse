<script lang="ts">
	// Curvature, shown in pictures rather than asserted.
	//
	// At every rung of the walk, both models are asked the same question and
	// their answers are converted into the finished picture each one implies
	// from where it stands. A straight path means that picture is decided
	// early and barely moves; a curved one means it keeps changing its mind,
	// and a sampler taking long strides along it will land somewhere else.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { Shuffle } from 'lucide-svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/emoji/Tiles.svelte';
	import { NOTHING } from '$lib/diffusion/runtime';
	import { lab, rival } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const FRAMES = 8;
	const STEPS = 48;
	const DIM = 4 * 32 * 32;

	let seed = $state(7);
	let flowStrip = $state<Float32Array | null>(null);
	let epsStrip = $state<Float32Array | null>(null);
	let busy = $state(false);

	async function pull(which: typeof lab, s: number): Promise<Float32Array | null> {
		try {
			const r = await which.sample(1, {
				steps: STEPS,
				a: NOTHING,
				guidanceA: 1,
				seed: s,
				trace: 'endpoint',
				fromShipped: which.hasShipped
			});
			// Skip the frame taken at τ = 1. There the surviving-signal fraction is
			// at its floor, so recovering an implied clean picture divides by
			// about a thousand and saturates whatever the model said. That first
			// frame reports the arithmetic, not the model, for both of them.
			const frames = (r.frames ?? []).slice(1);
			if (!frames.length) return null;
			const out = new Float32Array(FRAMES * DIM);
			for (let i = 0; i < FRAMES; i++) {
				const p = Math.min(frames.length - 1, Math.round((i * (frames.length - 1)) / (FRAMES - 1)));
				out.set(frames[p], i * DIM);
			}
			return out;
		} catch {
			return null;
		}
	}

	async function draw() {
		if (busy) return;
		busy = true;
		flowStrip = await pull(lab, seed);
		epsStrip = await pull(rival, seed);
		busy = false;
	}

	$effect(() => {
		void seed;
		if (lab.phase === 'ready' && rival.phase === 'ready') void draw();
	});

	async function boot() {
		await lab.boot();
		await rival.boot();
	}

	const blocked = $derived(lab.phase === 'no-webgpu' || rival.phase === 'no-webgpu');
</script>

<Plate id="paths" live {title} {caption}>
	{#snippet status()}
		{#if blocked}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>drawing both…</span>
		{:else if flowStrip}
			<span>{STEPS} steps, sampled at {FRAMES}</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn
			kind="primary"
			disabled={busy || blocked}
			onclick={() => (seed = Math.floor(Math.random() * 100000))}
		>
			<Shuffle size={12} aria-hidden="true" /> Another seed
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void boot()}>
		{#if blocked}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					Both models have to run to compare them, and that needs WebGPU. What the plate shows: the
					diffusion model's guess at its own destination changes all the way down the row, while the
					flow model's is recognizable by the second or third frame and only sharpens.
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-3.5 p-4 sm:p-5">
				<div>
					<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
						<span class="eyebrow" style="color: var(--warm);">predict the noise · chapter 9</span>
						<span class="text-[10.5px] text-ink-3">keeps changing its mind</span>
					</div>
					<Tiles
						pixels={epsStrip}
						count={FRAMES}
						columns={FRAMES}
						class="h-14 sm:h-20"
						label="The picture the diffusion model believes it is heading toward, at eight moments along its walk"
					/>
				</div>
				<div>
					<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
						<span class="eyebrow" style="color: var(--accent);"
							>predict the velocity · this chapter</span
						>
						<span class="text-[10.5px] text-ink-3">decides early, then sharpens</span>
					</div>
					<Tiles
						pixels={flowStrip}
						count={FRAMES}
						columns={FRAMES}
						class="h-14 sm:h-20"
						label="The picture the flow model believes it is heading toward, at eight moments along its walk"
					/>
				</div>
				<div class="num flex justify-between border-t border-line-soft pt-2 text-[10px] text-ink-3">
					<span>starting from static</span>
					<span>finished</span>
				</div>
			</div>
		{/if}
	</div>
</Plate>
