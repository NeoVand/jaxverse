<script lang="ts">
	// The step-count budget, spent by both models on the same seeds.
	// The whole argument of the chapter is legible in the bottom-left corner
	// of this plate: at two steps one of these rows is still a picture.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { Shuffle } from 'lucide-svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/fashion/Tiles.svelte';
	import { NOTHING } from '$lib/diffusion/runtime';
	import type { DiffusionLab } from '$lib/diffusion/lab.svelte';
	import { lab, rival } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const BUDGETS = [50, 20, 10, 4, 2];
	const SHOWN = 5;

	let seed = $state(31);
	let flowRows = $state<(Float32Array | null)[]>(BUDGETS.map(() => null));
	let epsRows = $state<(Float32Array | null)[]>(BUDGETS.map(() => null));
	let busy = $state(false);

	/** Fill one column, publishing each row as it lands so the reader watches
	 *  the plate build rather than waiting for all ten. */
	async function fill(which: DiffusionLab, publish: (rows: (Float32Array | null)[]) => void) {
		const rows: (Float32Array | null)[] = BUDGETS.map(() => null);
		for (let i = 0; i < BUDGETS.length; i++) {
			try {
				const r = await which.sample(SHOWN, {
					steps: BUDGETS[i],
					a: NOTHING,
					guidanceA: 1,
					seed,
					fromShipped: which.hasShipped
				});
				rows[i] = r.pixels;
				publish([...rows]);
			} catch {
				return;
			}
		}
	}

	async function draw() {
		if (busy) return;
		busy = true;
		await fill(rival, (r) => (epsRows = r));
		await fill(lab, (r) => (flowRows = r));
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

<Plate id="steps" live {title} {caption}>
	{#snippet status()}
		{#if blocked}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>drawing ten rows…</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn
			kind="primary"
			disabled={busy || blocked}
			onclick={() => (seed = Math.floor(Math.random() * 100000))}
		>
			<Shuffle size={12} aria-hidden="true" /> New seeds
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void boot()}>
		{#if blocked}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					This plate needs WebGPU to run both models. What it shows: as the step budget falls from
					fifty to two, the diffusion model degrades into coloured smoke while the flow model is
					still producing a recognizable garment at four steps and something arguable at two.
				</p>
			</div>
		{:else}
			<div class="grid gap-x-6 gap-y-2 p-4 sm:grid-cols-2 sm:p-5">
				<div class="flex flex-col gap-2">
					<span class="eyebrow" style="color: var(--warm);">predict the noise</span>
					{#each BUDGETS as b, i (b)}
						<div class="flex items-center gap-2.5">
							<span class="num w-9 shrink-0 text-right text-[10.5px] text-ink-3">{b}</span>
							<Tiles
								pixels={epsRows[i]}
								count={SHOWN}
								columns={SHOWN}
								class="h-10 sm:h-12"
								label="Five garments drawn by the diffusion model in {b} steps"
							/>
						</div>
					{/each}
				</div>
				<div class="flex flex-col gap-2">
					<span class="eyebrow" style="color: var(--accent);">predict the velocity</span>
					{#each BUDGETS as b, i (b)}
						<div class="flex items-center gap-2.5">
							<span class="num w-9 shrink-0 text-right text-[10.5px] text-ink-3">{b}</span>
							<Tiles
								pixels={flowRows[i]}
								count={SHOWN}
								columns={SHOWN}
								class="h-10 sm:h-12"
								label="Five garments drawn by the flow model in {b} steps"
							/>
						</div>
					{/each}
				</div>
				<span class="eyebrow col-span-full border-t border-line-soft pt-2 text-ink-3">
					steps down the left · same seeds in every row
				</span>
			</div>
		{/if}
	</div>
</Plate>
