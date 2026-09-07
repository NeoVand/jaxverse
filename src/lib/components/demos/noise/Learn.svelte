<script lang="ts">
	// Watch a denoiser learn, from noise, here.
	//
	// Owns the chapter's engine. The top row is drawn from the weights training
	// in this tab; the bottom row is the same eight seeds drawn from the
	// checkpoint that trained overnight. Same architecture, same code, same
	// noise — the only difference is how many steps each one has had, which is
	// the honest way to show a reader what "keep going" buys.
	import { onDestroy } from 'svelte';
	import { Play, Pause, RotateCcw } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/fashion/Tiles.svelte';
	import { sparkPath } from '$lib/viz/spark';
	import { NOTHING } from '$lib/diffusion/runtime';
	import { lab, SHOWN } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const SEED = 4242;
	const STEPS = 40;

	let live = $state<Float32Array | null>(null);
	let shipped = $state<Float32Array | null>(null);
	let busy = false;
	let want = false;

	// One re-draw per trained chunk. Sampling forty denoising steps costs about
	// as much as a training step does, so it rides along rather than competing.
	$effect(() => {
		void lab.tick;
		if (lab.phase === 'idle' || lab.phase === 'loading' || lab.phase === 'no-webgpu') return;
		void refresh();
	});

	async function refresh() {
		if (busy) {
			want = true;
			return;
		}
		busy = true;
		try {
			const r = await lab.sample(SHOWN, {
				steps: STEPS,
				a: NOTHING,
				guidanceA: 1,
				seed: SEED
			});
			live = r.pixels;
			if (lab.hasShipped && !shipped) {
				const s = await lab.sample(SHOWN, {
					steps: STEPS,
					a: NOTHING,
					guidanceA: 1,
					seed: SEED,
					fromShipped: true
				});
				shipped = s.pixels;
			}
		} catch {
			// engine disposed mid-flight
		}
		busy = false;
		if (want) {
			want = false;
			void refresh();
		}
	}

	onDestroy(() => {
		// this plate owns the chapter's engine
		void lab.dispose();
	});
</script>

<Plate id="learn" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'loading'}
			<span>fetching the garment sheets…</span>
		{:else if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if lab.phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{:else if lab.phase !== 'idle'}
			<span>step {lab.step.toLocaleString('en-US')}</span>
			<span aria-hidden="true">·</span>
			<span>loss {Number.isFinite(lab.loss) ? lab.loss.toFixed(4) : '—'}</span>
			{#if lab.stepMs > 0}
				<span aria-hidden="true">·</span>
				<span>{lab.stepMs.toFixed(0)} ms/step</span>
			{/if}
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn
			kind={lab.training ? 'ghost' : 'primary'}
			disabled={lab.phase === 'loading' || lab.phase === 'no-webgpu' || lab.phase === 'error'}
			onclick={() => void lab.toggleTrain()}
		>
			{#if lab.training}
				<Pause size={12} aria-hidden="true" /> Pause
			{:else}
				<Play size={12} aria-hidden="true" /> Train
			{/if}
		</Btn>
		<Btn
			disabled={lab.phase === 'loading' || lab.phase === 'no-webgpu'}
			onclick={() => void lab.reset()}
			title="Fresh random weights"
		>
			<RotateCcw size={12} aria-hidden="true" /> Reset
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					This chapter trains a real image model in your browser, which needs WebGPU — a current
					Chrome or Edge on a desktop will run it. The prose reads fine without it, and the captions
					say what each plate shows.
				</p>
			</div>
		{:else if lab.phase === 'error'}
			<div class="flex flex-wrap items-center gap-3 px-4 py-4">
				<span class="text-[12.5px] text-bad">{lab.error}</span>
				<Btn onclick={() => void lab.boot()}>Retry</Btn>
			</div>
		{:else if lab.phase === 'loading' || lab.phase === 'idle'}
			<div class="flex h-[240px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">fetching 12,000 pictures · warming up the worker…</span>
				<span class="text-[12.5px] text-ink-3">
					a 2.5-million-parameter image model, trained in this tab
				</span>
			</div>
		{:else}
			<div class="flex flex-col gap-3.5 p-4 sm:p-5">
				<div>
					<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
						<span class="eyebrow" style="color: var(--accent);">trained here · this tab</span>
						<span class="text-[10.5px] text-ink-3">
							{lab.step === 0
								? 'untrained — press Train'
								: `${lab.step.toLocaleString('en-US')} steps`}
						</span>
					</div>
					<Tiles
						pixels={live}
						count={SHOWN}
						columns={SHOWN}
						class="h-16 sm:h-24"
						label="Eight pictures drawn by the model training in this tab, from eight fixed seeds"
					/>
				</div>

				{#if lab.hasShipped}
					<div>
						<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
							<span class="eyebrow" style="color: var(--warm);">the same eight seeds · shipped</span
							>
							<span class="text-[10.5px] text-ink-3">same code, already trained</span>
						</div>
						<Tiles
							pixels={shipped}
							count={SHOWN}
							columns={SHOWN}
							class="h-16 sm:h-24"
							label="The same eight seeds drawn by the checkpoint that ships with the book"
						/>
					</div>
				{/if}

				<div class="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line-soft pt-3">
					<span class="flex min-w-40 flex-1 items-center gap-1.5">
						<span class="eyebrow shrink-0 text-[9.5px]" style="color: var(--accent);">
							denoising loss
						</span>
						<svg
							viewBox="0 0 200 22"
							preserveAspectRatio="none"
							class="block h-[22px] w-full"
							role="img"
							aria-label="Training loss, log scale"
						>
							<path
								d={sparkPath(lab.lossHist, 200, 22, { log: true, floor: 1e-4 })}
								fill="none"
								stroke="var(--accent)"
								stroke-width="1.4"
								vector-effect="non-scaling-stroke"
							/>
						</svg>
					</span>
					<span class="num text-[10.5px] whitespace-nowrap text-ink-3">
						{lab.imagesPerSec > 0 ? `${lab.imagesPerSec} img/s · ` : ''}batch 32 · webgpu
					</span>
				</div>
			</div>
		{/if}
	</div>
</Plate>
