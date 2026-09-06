<script lang="ts">
	// One idea, eight hands.
	//
	// Content and style arrive on separate parts of the conditioning vector and
	// are dropped independently during training, so the model has to learn what
	// each contributes on its own. This plate is the test of whether it did:
	// same prompt, same seeds, eight styles down the page.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { Shuffle } from 'lucide-svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/emoji/Tiles.svelte';
	import { makeVocab, parsePrompt } from '$lib/diffusion/corpus';
	import { lab, SUGGESTIONS } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const SHOWN = 4;

	let prompt = $state('smiling cat face');
	let seed = $state(1212);
	let rows = $state<(Float32Array | null)[]>([]);
	let busy = $state(false);
	let pick = 0;

	const vocab = $derived(makeVocab(lab.info?.tags ?? []));
	const sets = $derived(lab.info?.sets ?? []);

	async function draw() {
		if (busy || lab.phase !== 'ready' || !sets.length) return;
		busy = true;
		const tags = parsePrompt(prompt, vocab).tags;
		const next: (Float32Array | null)[] = sets.map(() => null);
		for (let i = 0; i < sets.length; i++) {
			try {
				const r = await lab.sample(SHOWN, {
					steps: 20,
					a: { tags, style: i },
					guidanceA: 2.5,
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
		void prompt;
		void sets;
		if (lab.phase === 'ready') void draw();
	});

	function shuffle() {
		prompt = SUGGESTIONS[pick++ % SUGGESTIONS.length];
		seed = Math.floor(Math.random() * 100000);
	}
</script>

<Plate id="styles" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>drawing eight hands…</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn kind="primary" disabled={busy || lab.phase !== 'ready'} onclick={shuffle}>
			<Shuffle size={12} aria-hidden="true" /> Another idea
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					This plate needs WebGPU. What it shows: one prompt drawn in each of the eight styles the
					model was trained on — glossy, flat, line art, blob, and three renderings of Microsoft's
					set — from the same four seeds, so only the hand changes.
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-2 p-4 sm:p-5">
				<label class="mb-1 flex items-center gap-2">
					<span class="eyebrow shrink-0">prompt</span>
					<input
						class="w-full max-w-64 rounded-[var(--r-2)] border border-line bg-surface px-2.5 py-1 font-serif text-[13.5px] text-ink italic outline-none focus-visible:shadow-[var(--focus-ring)]"
						bind:value={prompt}
						aria-label="What to draw"
					/>
				</label>
				<div class="grid gap-x-6 gap-y-2 sm:grid-cols-2">
					{#each sets as set, i (set.id)}
						<div class="flex items-center gap-2.5">
							<span
								class="w-20 shrink-0 truncate text-[10.5px] text-ink-3"
								title="{set.credit} · {set.license}">{set.label}</span
							>
							<Tiles
								pixels={rows[i] ?? null}
								count={SHOWN}
								columns={SHOWN}
								class="h-11 sm:h-14"
								label="Four emoji drawn in the {set.label} style"
							/>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	</div>
</Plate>
