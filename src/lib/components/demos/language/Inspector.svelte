<script lang="ts">
	// Plate IV — the surprise meter. One inspect() call bills a sentence
	// character by character: background heat is the per-token loss, and the
	// detail strip below holds the distribution the model had just before each
	// reveal. The strip is a fixed part of the layout rather than a floating
	// popover — nothing can clip it, and tapping works the same as hovering.
	import { ScanText } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import LabGate from './LabGate.svelte';
	import type { PerTokenInfo } from '$lib/llm/engine';
	import { scribe, UNIFORM_NATS } from './lab.svelte';

	const DEFAULT_TEXT = 'The little dog was so happy that he wagged his tail.';
	let text = $state(DEFAULT_TEXT);
	let infos = $state<PerTokenInfo[] | null>(null);
	let readStep = $state(0);
	let busy = $state(false);
	let active = $state<number | null>(null);

	const usable = $derived(scribe.phase === 'ready' || scribe.phase === 'training');
	/** The strip shows the hovered character, or the most surprising one at rest. */
	const focus = $derived.by(() => {
		if (!infos) return null;
		if (active !== null && infos[active]) return { i: active, t: infos[active], pinned: true };
		let best = -1;
		let worst = -Infinity;
		for (let i = 0; i < infos.length; i++) {
			const l = infos[i].loss;
			if (l !== undefined && l > worst) {
				worst = l;
				best = i;
			}
		}
		return best >= 0 ? { i: best, t: infos[best], pinned: false } : null;
	});

	async function read() {
		if (busy || !usable) return;
		const ids = scribe.encode(text).slice(0, 96); // the model's whole window
		if (ids.length === 0) return;
		busy = true;
		active = null;
		const atStep = scribe.step;
		const r = await scribe.inspect(ids);
		busy = false;
		if (r) {
			infos = r;
			readStep = atStep;
		}
	}

	const mean = $derived.by(() => {
		if (!infos) return NaN;
		let s = 0;
		let n = 0;
		for (const t of infos)
			if (t.loss !== undefined) {
				s += t.loss;
				n++;
			}
		return n ? s / n : NaN;
	});

	// paper → warm, eased so mid-surprise characters stay readable
	function heat(loss: number | undefined): string {
		if (loss === undefined) return 'transparent';
		const t = Math.min(loss / UNIFORM_NATS, 1);
		return `color-mix(in srgb, var(--warm) ${Math.round(Math.pow(t, 0.75) * 62)}%, transparent)`;
	}

	function glyph(c: string): string {
		return c === ' ' ? '␣' : c === '\n' ? '↵' : c;
	}

	function charLabel(t: PerTokenInfo): string {
		return t.loss === undefined
			? `“${glyph(t.text)}” — first character, nothing precedes it`
			: `“${glyph(t.text)}” — ${t.loss.toFixed(2)} nats of surprise`;
	}
</script>

<Plate
	n={4}
	title="The surprise meter"
	caption="Each character's background is what the model paid to see it — hotter is more surprising. Word-openings run hot and word-endings cool; type a q somewhere and watch its u come almost free. Surprise is where the information lives."
>
	{#snippet status()}
		{#if infos}
			<span>read at step {readStep}</span>
			<span aria-hidden="true">·</span>
			<span>mean {mean.toFixed(2)} nats/char</span>
		{:else if usable}
			<span>scribe at step {scribe.step}</span>
		{:else}
			<span>waiting for the scribe</span>
		{/if}
	{/snippet}

	<div use:inview={() => void scribe.boot()}>
		{#if !usable}
			<LabGate
				note="this plate reads the scribe from the plate above — it boots when you reach it, and starts out surprised by everything"
			/>
		{:else}
			<div class="flex items-end gap-2 px-4 pt-3.5 pb-3">
				<label class="min-w-0 flex-1">
					<span class="eyebrow mb-1 block">sentence</span>
					<input
						class="input"
						maxlength="96"
						bind:value={text}
						onkeydown={(e) => {
							if (e.key === 'Enter') void read();
						}}
					/>
				</label>
				<Btn kind="primary" onclick={() => void read()} disabled={busy}>
					<ScanText size={13} aria-hidden="true" /> Read
				</Btn>
			</div>

			<div class="px-4 pb-4">
				<div
					class="rounded-md border border-line-soft bg-paper px-3 py-3"
					role="group"
					aria-label="the sentence, one button per character, heated by surprise"
					onpointerleave={() => (active = null)}
				>
					{#if infos}
						<div class="flex flex-wrap font-mono text-[14px] leading-[1.9]" style="row-gap: 3px;">
							{#each infos as t, i (i)}
								<button
									type="button"
									class="ch"
									class:ch-free={t.loss === undefined}
									class:ch-on={focus?.pinned && focus.i === i}
									style="background: {heat(t.loss)};"
									aria-label={charLabel(t)}
									onpointerenter={() => (active = i)}
									onfocus={() => (active = i)}
									onclick={() => (active = i)}
									onblur={() => {
										if (active === i) active = null;
									}}>{t.text}</button
								>
							{/each}
						</div>
					{:else}
						<p class="num py-3 text-center text-[11px] text-ink-3">
							{busy ? 'reading…' : 'press read — every character gets a price'}
						</p>
					{/if}
				</div>

				<!-- the detail strip: part of the layout, so it can never be clipped -->
				<div class="mt-2 min-h-24 rounded-md border border-line-soft bg-surface-2 px-3 py-2.5">
					{#if !focus}
						<span class="num text-[11px] text-ink-3">
							hover or tap any character to see the five candidates the model was weighing there
						</span>
					{:else}
						{@const t = focus.t}
						<div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
							<span class="eyebrow" style="color: {focus.pinned ? 'var(--warm)' : 'var(--ink-3)'};">
								{focus.pinned ? 'at the cursor' : 'the most expensive character'}
							</span>
							<span class="num text-[14px] text-ink">“{glyph(t.text)}”</span>
							{#if t.loss !== undefined}
								<span class="num text-[11px] text-ink-2">
									{t.loss.toFixed(2)} nats · entropy {(t.entropy ?? 0).toFixed(2)} nats
								</span>
							{/if}
						</div>
						{#if t.loss === undefined}
							<p class="mt-1 text-[11px] text-ink-3">
								no context yet — the first character is never predicted
							</p>
						{:else}
							<div class="mt-1.5 grid grid-cols-1 gap-x-5 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
								{#each t.topk ?? [] as cand (cand[0])}
									<div class="flex items-center gap-2">
										<span
											class="num w-4 text-[11.5px]"
											style="color: {cand[0] === t.id ? 'var(--warm)' : 'var(--ink)'};"
											>{glyph(scribe.charOf(cand[0]))}</span
										>
										<span class="h-[3px] flex-1 overflow-hidden rounded-full bg-line">
											<span
												class="block h-full"
												style="width: {(cand[1] * 100).toFixed(1)}%; background: {cand[0] === t.id
													? 'var(--warm)'
													: 'var(--accent)'};"
											></span>
										</span>
										<span class="num w-9 text-right text-[10px] text-ink-3"
											>{(cand[1] * 100).toFixed(cand[1] < 0.095 ? 1 : 0)}%</span
										>
									</div>
								{/each}
							</div>
						{/if}
					{/if}
				</div>

				<p class="mt-2 text-[11px] leading-relaxed text-ink-3">
					the vermilion candidate is the character that actually came; a dotted underline marks the
					free first character. characters outside the 69-glyph alphabet are dropped.
					{#if infos && readStep < 200}
						this scribe has barely trained ({readStep} steps), so expect everything to glow — train it
						above and read again.
					{/if}
				</p>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.input {
		width: 100%;
		height: 32px;
		font-family: var(--font-mono);
		font-size: 13px;
		color: var(--ink);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 6px;
		padding: 4px 10px;
	}
	.input:focus {
		outline: none;
		border-color: var(--accent);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
	}
	.ch {
		white-space: pre;
		border-radius: 3px;
		padding: 1px 1.5px;
		color: var(--ink);
		transition: background 120ms ease;
	}
	.ch:hover {
		outline: 1px solid var(--line);
	}
	.ch-on {
		outline: 1.5px solid var(--warm);
	}
	.ch-free {
		border-bottom: 1px dotted var(--ink-3);
	}
</style>
