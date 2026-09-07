<script lang="ts">
	// Walking the label from one garment to another.
	//
	// The conditioning is a one-hot: during training the model was only ever
	// shown a vector with a single 1 in it. Half of a sneaker plus half of a
	// boot is a question nobody asked it. It answers anyway, and the answer is
	// continuous — which is the evidence that the model learned a space of
	// garments rather than ten separate lookup tables.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { Shuffle } from 'lucide-svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/fashion/Tiles.svelte';
	import { coalesce } from '$lib/diffusion/lab.svelte';
	import { MORPHS, lab } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	/** Columns across the walk, ends included. */
	const MIXES = 7;
	/** Two seeds, so a reader can tell a trend from an accident. */
	const ROWS = 2;

	let pair = $state(0);
	let guidance = $state(2);
	let seed = $state(5150);
	let rows = $state<(Float32Array | null)[]>(Array.from({ length: ROWS }, () => null));
	let busy = $state(false);

	const names = $derived(lab.info?.classes ?? []);
	const from = $derived(MORPHS[pair][0]);
	const to = $derived(MORPHS[pair][1]);

	const draw = coalesce(async () => {
		if (lab.phase !== 'ready') return;
		busy = true;
		try {
			for (let r = 0; r < ROWS; r++) {
				// One call per column: every column is a different conditioning
				// vector, and the sampler draws one condition at a time.
				const strip = new Float32Array(MIXES * 28 * 28);
				for (let i = 0; i < MIXES; i++) {
					const res = await lab.sample(1, {
						steps: 20,
						a: { label: from, other: to, mix: i / (MIXES - 1) },
						guidanceA: guidance,
						seed: seed + r * 1000,
						fromShipped: lab.hasShipped
					});
					strip.set(res.pixels.subarray(0, 28 * 28), i * 28 * 28);
				}
				rows[r] = strip;
				rows = [...rows];
			}
		} catch {
			// disposed mid-flight
		}
		busy = false;
	});

	$effect(() => {
		void pair;
		void guidance;
		void seed;
		if (lab.phase === 'ready') draw();
	});
</script>

<Plate id="morph" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>walking…</span>
		{:else if names.length}
			<span>{names[from]} → {names[to]}</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn
			kind="primary"
			disabled={busy || lab.phase !== 'ready'}
			onclick={() => (pair = (pair + 1) % MORPHS.length)}
		>
			<Shuffle size={12} aria-hidden="true" /> Another pair
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					This plate needs WebGPU. What it shows: as the label slides from one garment to another
					the picture does not cut from one to the other, it travels — a sole grows a shaft, a
					sleeve lengthens into a skirt — through shapes that are in no catalogue.
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-2.5 p-4 sm:p-5">
				<div class="flex items-baseline justify-between gap-3">
					<span class="eyebrow">{names[from] ?? ''}</span>
					<span class="eyebrow">{names[to] ?? ''}</span>
				</div>
				{#each rows as strip, r (r)}
					<Tiles
						pixels={strip}
						count={MIXES}
						columns={MIXES}
						gap={4}
						class="h-14 sm:h-20"
						label="One garment turning into another across seven steps"
					/>
				{/each}
				<div class="num flex justify-between text-[10px] text-ink-3">
					{#each Array.from({ length: MIXES }, (_, i) => i) as i (i)}
						<span>{(i / (MIXES - 1)).toFixed(2)}</span>
					{/each}
				</div>
				<div class="border-t border-line-soft pt-3">
					<Slider
						label="guidance w"
						bind:value={guidance}
						min={0}
						max={6}
						step={0.5}
						tone="knob"
						format={(v) => v.toFixed(1)}
					/>
				</div>
				<span class="eyebrow text-ink-3">
					same noise in each row · only the label vector changes across a row
				</span>
			</div>
		{/if}
	</div>
</Plate>
