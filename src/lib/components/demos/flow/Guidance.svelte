<script lang="ts">
	// The guidance dial, swept rather than slid: the same label and the same
	// four seeds at five strengths, so the two failure modes sit on the page
	// at once — at zero the label is not obeyed, and past about six the
	// pictures go loud, flat and interchangeable.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { Shuffle } from 'lucide-svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/fashion/Tiles.svelte';
	import { lab } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const WEIGHTS = [0, 1, 2, 4, 8];
	const SHOWN = 4;

	let label = $state(9);
	let seed = $state(808);
	let rows = $state<(Float32Array | null)[]>(WEIGHTS.map(() => null));
	let busy = $state(false);

	const names = $derived(lab.info?.classes ?? []);

	async function draw() {
		if (busy || lab.phase !== 'ready') return;
		busy = true;
		const next: (Float32Array | null)[] = WEIGHTS.map(() => null);
		for (let i = 0; i < WEIGHTS.length; i++) {
			try {
				const r = await lab.sample(SHOWN, {
					steps: 20,
					a: { label },
					guidanceA: WEIGHTS[i],
					seed,
					fromShipped: lab.hasShipped
				});
				next[i] = r.pixels;
				rows = [...next];
			} catch {
				break;
			}
		}
		busy = false;
	}

	$effect(() => {
		void seed;
		void label;
		if (lab.phase === 'ready') void draw();
	});
</script>

<Plate id="guidance" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>sweeping…</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn
			kind="primary"
			disabled={busy || lab.phase !== 'ready'}
			onclick={() => (seed = Math.floor(Math.random() * 100000))}
		>
			<Shuffle size={12} aria-hidden="true" /> New seeds
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					This plate needs WebGPU. What it shows: with guidance at zero the model draws whatever it
					likes and the label might as well not be there; by two the label is obeyed; by eight every
					picture is the same over-stated idea of the garment and the variety is gone.
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-2.5 p-4 sm:p-5">
				<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Garment">
					<span class="eyebrow mr-1">garment</span>
					{#each names as name, i (name)}
						<button
							class="chip"
							class:chip-on={label === i}
							aria-pressed={label === i}
							onclick={() => (label = i)}>{name}</button
						>
					{/each}
				</span>
				{#each WEIGHTS as w, i (w)}
					<div class="flex items-center gap-3">
						<span class="num w-8 shrink-0 text-right text-[11px]" style="color: var(--cat-1);">
							{w}
						</span>
						<Tiles
							pixels={rows[i]}
							count={SHOWN}
							columns={SHOWN}
							class="h-12 sm:h-16"
							label="Four garments drawn at guidance strength {w}"
						/>
						<span class="w-28 shrink-0 text-[10.5px] text-ink-3">
							{w === 0
								? 'label ignored'
								: w === 1
									? 'as trained'
									: w === 2
										? 'about right, here'
										: w === 8
											? 'shouting'
											: ''}
						</span>
					</div>
				{/each}
				<span class="eyebrow border-t border-line-soft pt-2 text-ink-3">
					guidance <span class="sym">w</span> down the left · same four seeds in every row
				</span>
			</div>
		{/if}
	</div>
</Plate>
