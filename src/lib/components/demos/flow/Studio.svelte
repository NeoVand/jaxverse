<script lang="ts">
	// The studio. Choose a garment, turn two dials, and the model draws eight of
	// them — none of which exist.
	//
	// There is no text box here and that is the point: the model was trained on
	// ten labels, so the thing a reader steers is exactly the thing the model
	// was taught. A prompt box would promise a vocabulary that is not there.
	import { Sparkles, Shuffle } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/fashion/Tiles.svelte';
	import { coalesce } from '$lib/diffusion/lab.svelte';
	import { lab, SHOWN } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	let label = $state<number | null>(9); // ankle boot: the most legible at 28px
	let guidance = $state(2);
	let steps = $state(20);
	let seed = $state(2026);
	let pixels = $state<Float32Array | null>(null);
	let ms = $state(0);
	let busy = $state(false);

	const names = $derived(lab.info?.classes ?? []);

	const draw = coalesce(async () => {
		if (lab.phase === 'no-webgpu' || !lab.info) return;
		busy = true;
		try {
			const r = await lab.sample(SHOWN, {
				steps,
				a: { label },
				guidanceA: guidance,
				seed,
				fromShipped: lab.hasShipped
			});
			pixels = r.pixels;
			ms = r.ms;
		} catch {
			// disposed mid-flight
		}
		busy = false;
	});

	$effect(() => {
		void label;
		void guidance;
		void steps;
		void seed;
		if (lab.phase === 'ready') draw();
	});
</script>

<Plate id="studio" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>drawing…</span>
		{:else if pixels}
			<span>{SHOWN} pictures</span>
			<span aria-hidden="true">·</span>
			<span>{steps} steps</span>
			<span aria-hidden="true">·</span>
			<span>{ms.toFixed(0)} ms</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn kind="primary" disabled={busy || lab.phase !== 'ready'} onclick={draw}>
			<Sparkles size={12} aria-hidden="true" /> Draw
		</Btn>
		<Btn
			disabled={busy || lab.phase !== 'ready'}
			onclick={() => (seed = Math.floor(Math.random() * 100000))}
			title="The same garment, different noise"
		>
			<Shuffle size={12} aria-hidden="true" /> New noise
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					The studio draws on your own GPU, and that needs WebGPU — a current Chrome or Edge on a
					desktop will run it. Everything the chapter argues is in the prose and the captions.
				</p>
			</div>
		{:else if lab.phase === 'loading' || lab.phase === 'idle'}
			<div class="flex h-[320px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">fetching the model (2.5 MB) and the garment sheet…</span>
				<span class="text-[12.5px] text-ink-3">nothing leaves the page</span>
			</div>
		{:else}
			<div class="grid gap-x-6 gap-y-4 p-4 sm:p-5 lg:grid-cols-[1fr_20rem]">
				<div class="flex flex-col gap-2">
					<Tiles
						{pixels}
						count={SHOWN}
						columns={4}
						gap={4}
						class="h-40 sm:h-56"
						label="Eight garments drawn for the chosen label"
					/>
					<span class="num text-[10px] text-ink-3">
						{label === null
							? 'no label — this is the model drawing whatever it likes'
							: `${names[label] ?? ''} · guidance ${guidance.toFixed(1)}`}
					</span>
				</div>

				<div class="flex flex-col gap-3.5">
					<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Garment">
						<span class="eyebrow mr-1 w-full">garment</span>
						<button
							class="chip"
							class:chip-on={label === null}
							aria-pressed={label === null}
							onclick={() => (label = null)}>anything</button
						>
						{#each names as name, i (name)}
							<button
								class="chip"
								class:chip-on={label === i}
								aria-pressed={label === i}
								onclick={() => (label = i)}>{name}</button
							>
						{/each}
					</span>

					<Slider
						label="guidance w"
						bind:value={guidance}
						min={0}
						max={8}
						step={0.5}
						tone="knob"
						format={(v) => (v === 0 ? '0 · ignore the label' : v.toFixed(1))}
					/>
					<Slider
						label="steps"
						bind:value={steps}
						min={2}
						max={50}
						step={1}
						tone="op"
						format={(v) => String(v)}
					/>
				</div>
			</div>
		{/if}
	</div>
</Plate>
