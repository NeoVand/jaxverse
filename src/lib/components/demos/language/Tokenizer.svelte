<script lang="ts">
	// Plate II — grow a vocabulary. Real BPE over the real corpus (bpe-live.ts):
	// the browser counts every adjacent pair in 1.5M characters, fuses the
	// winner, repeats. Each merge reports its own wall-clock ms, which is the
	// honest answer to "is this prerecorded?". Runs of 300, extendable
	// indefinitely; the history stays scrubbable. Pure CPU, no engine.
	import { onDestroy } from 'svelte';
	import {
		ArrowRight,
		GitMerge,
		Pause,
		Play,
		Plus,
		RotateCcw,
		StepBack,
		StepForward
	} from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { loadCorpus } from '$lib/data/corpus';
	import { BpeTrainer, trainBpe, RUN_MERGES, type MergeRecord } from './bpe-live';
	import { scribe } from './lab.svelte';

	const SENTENCE = 'Once upon a time, the little dragon looked at the moon.';
	const RECENT = 12;
	const REPLAY_MS = 120;

	type Phase = 'idle' | 'loading' | 'running' | 'paused' | 'done' | 'error';
	let phase = $state<Phase>('idle');
	let errorMsg = $state('');
	let merges = $state<MergeRecord[]>([]);
	/** Scrub position: follows the newest merge while training, free afterwards. */
	let k = $state(0);
	let playing = $state(false);
	let originalLen = $state(0);
	let target = $state(RUN_MERGES);
	let lastMs = $state(0);

	let trainer: BpeTrainer | null = null; // deliberately not $state
	let gen = 0;
	let timer: ReturnType<typeof setInterval> | null = null;

	// 70 ms between merges — spaced out for reading, not for the machine
	const PACE_MS = 70;
	const scrubbable = $derived((phase === 'paused' || phase === 'done') && merges.length > 0);

	async function boot() {
		if (phase !== 'idle' && phase !== 'error') return;
		phase = 'loading';
		errorMsg = '';
		const myGen = ++gen;
		try {
			const corpus = await loadCorpus(); // shared cache with every other plate
			if (myGen !== gen) return;
			trainer = new BpeTrainer(corpus.tokens, corpus.vocab);
			originalLen = trainer.originalLen;
			merges = [];
			k = 0;
			target = RUN_MERGES;
			phase = 'running';
			void run();
		} catch (err) {
			if (myGen !== gen) return;
			errorMsg = err instanceof Error ? err.message : String(err);
			phase = 'error';
		}
	}

	async function run() {
		const t = trainer;
		if (!t) return;
		const myGen = gen;
		const status = await trainBpe(
			t,
			{
				maxMerges: target,
				paceMs: () => PACE_MS,
				shouldStop: () => myGen !== gen
			},
			(m) => {
				merges.push(m);
				k = merges.length;
				lastMs = m.ms;
			}
		);
		if (myGen !== gen) return; // paused or unmounted — phase already set
		phase = status === 'stopped' ? 'paused' : 'done';
	}

	function pauseTraining() {
		if (phase !== 'running') return;
		gen++; // the driver notices within one beat
		phase = 'paused';
	}

	function resumeTraining() {
		if (phase !== 'paused' || !trainer) return;
		stopReplay();
		k = merges.length; // snap the scrubber back to live
		if (target <= merges.length) target = merges.length + RUN_MERGES;
		phase = 'running';
		void run();
	}

	/** Another 300 elections — the trainer keeps its sequence, so this simply
	 * continues where it stopped. There is no end of the list to reach. */
	function keepMerging() {
		if (!trainer || phase === 'running' || phase === 'loading') return;
		stopReplay();
		k = merges.length;
		target = merges.length + RUN_MERGES;
		phase = 'running';
		void run();
	}

	// ── merge-history scrubber (paused or done only) ──
	function stopReplay() {
		if (timer) clearInterval(timer);
		timer = null;
		playing = false;
	}
	function toggleReplay() {
		if (playing) {
			stopReplay();
			return;
		}
		if (!scrubbable) return;
		if (k >= merges.length) k = 0; // replay from the top
		playing = true;
		timer = setInterval(() => {
			if (k < merges.length) k += 1;
			if (k >= merges.length) stopReplay();
		}, REPLAY_MS);
	}
	function stepK(d: number) {
		stopReplay();
		k = Math.min(merges.length, Math.max(0, k + d));
	}

	onDestroy(() => {
		gen++;
		stopReplay();
	});

	/** Hand this vocabulary to the scribe. A different embedding table is a
	 * different model, so the scribe rebuilds and starts from step 0 — which is
	 * the honest thing to show: a tokenizer is a decision made before training. */
	let handing = $state(false);
	const scribeMerges = $derived(scribe.kind === 'pieces' ? scribe.mergeCount : -1);
	// every term here must be reactive: `trainer` deliberately is not, and testing
	// it first would short-circuit this derived into having no dependencies at all
	const handable = $derived(
		merges.length > 0 && phase !== 'running' && merges.length !== scribeMerges
	);
	async function handOver() {
		const t = trainer;
		if (!t || handing) return;
		handing = true;
		try {
			await scribe.useVocabulary(t.chars, t.pairs(), t.tokenizedCorpus(), t.originalLen);
		} finally {
			handing = false;
		}
	}

	// ── views at scrub position k ──
	const newest = $derived(k > 0 && k <= merges.length ? merges[k - 1] : null);
	const recent = $derived.by(() => {
		const out: MergeRecord[] = [];
		for (let i = k - 1; i >= Math.max(0, k - RECENT); i--) out.push(merges[i]);
		return out;
	});
	const sentToks = $derived.by(() => {
		void merges.length;
		const t = trainer;
		if (!t) return [];
		return t.encodeAt(SENTENCE, k).map((id) => ({ id, text: t.decode(id) }));
	});
	const sentCpt = $derived(sentToks.length ? SENTENCE.length / sentToks.length : 1);
	const corpusToks = $derived.by(() => {
		void merges.length;
		return trainer ? trainer.corpusLenAt(k) : 0;
	});
	const corpusCpt = $derived(corpusToks > 0 ? originalLen / corpusToks : 1);

	// five categorical tints: sentence chips cycle by position, vocabulary chips
	// by token id (stable per token as the recent list shifts). 28% mix plus a
	// hairline border — the faint 16% wash vanished on dark paper.
	const CATS = [0, 2, 4, 6, 8];
	const wash = (n: number) => `color-mix(in srgb, var(--cat-${CATS[n % 5]}) 28%, transparent)`;
	const edge = (n: number) => `color-mix(in srgb, var(--cat-${CATS[n % 5]}) 45%, var(--line))`;
	const fmt = (n: number) => n.toLocaleString('en-US');
</script>

{#snippet glyphs(text: string)}
	{@const lead = text.startsWith(' ')}
	{@const body = (lead ? text.slice(1) : text).replaceAll('\n', '')}
	{@const nl = text.length - (lead ? 1 : 0) - body.length}
	<!-- a leading space is what marks a word's start, so show it rather than
	     leaving the reader to infer it from the chip's padding -->
	{#if lead}<span class="sp" aria-hidden="true">␣</span>{/if}{#if body}{body}{/if}{#if nl > 0}<span
			class="nl"
			aria-hidden="true">↵</span
		>{/if}
{/snippet}

<Plate
	n={2}
	title="Grow a vocabulary"
	caption="Byte-pair encoding, running for real: the vocabulary is not designed, it is voted for by the corpus, one most-frequent pair at a time. th, ␣the, ing — watch English assemble itself by frequency, and note that a token never spans two words. Nothing here is prerecorded; every count comes from scanning your own copy of the corpus, and the millisecond figure is how long that scan took on this machine. Compression is the score — each merge shortens the corpus by one token per fusion it makes — and the first three hundred merges are exactly the vocabulary the scribe below reads, so keep merging and send it a longer one if you want to see what changes."
>
	{#snippet status()}
		{#if phase === 'idle' || phase === 'loading'}
			<span>{phase === 'idle' ? 'waiting' : 'fetching corpus…'}</span>
		{:else if phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{:else}
			<span>vocab {69 + k}</span>
			<span aria-hidden="true">·</span>
			<span>merge {k}</span>
			{#if lastMs > 0}
				<span aria-hidden="true">·</span>
				<span>{lastMs.toFixed(1)} ms to count 1.5M pairs</span>
			{/if}
		{/if}
	{/snippet}

	{#snippet actions()}
		{#if handable}
			<Btn
				onclick={() => void handOver()}
				disabled={handing}
				title="Rebuild the scribe on this vocabulary"
			>
				<ArrowRight size={12} aria-hidden="true" />
				{handing ? 'Handing over…' : 'Send to the scribe'}
			</Btn>
		{/if}
		{#if phase === 'running'}
			<Btn onclick={pauseTraining}>
				<Pause size={12} aria-hidden="true" /> Pause
			</Btn>
		{:else if phase === 'paused'}
			<Btn onclick={resumeTraining}>
				<Play size={12} aria-hidden="true" /> Resume
			</Btn>
		{:else if phase === 'done'}
			<Btn onclick={keepMerging}>
				<Plus size={12} aria-hidden="true" /> Keep merging
			</Btn>
		{/if}
	{/snippet}

	<div use:inview={() => void boot()}>
		{#if phase === 'idle' || phase === 'loading' || phase === 'error'}
			<div class="flex min-h-56 flex-col items-center justify-center gap-3 px-6 py-8">
				{#if phase === 'error'}
					<p class="num max-w-md text-center text-[12px]" style="color: var(--bad);">
						{errorMsg || 'the corpus failed to load'}
					</p>
					<Btn onclick={() => void boot()}>
						<GitMerge size={13} aria-hidden="true" /> Retry
					</Btn>
				{:else}
					<span class="num text-[12px] text-ink-3">
						{phase === 'idle'
							? 'the tokenizer trains when you reach it'
							: 'fetching the story corpus (about 1.5 MB)…'}
					</span>
				{/if}
			</div>
		{:else}
			<div class="p-4">
				<!-- the fusion on the floor -->
				<div
					class="flex min-h-12 flex-wrap items-center gap-2 rounded-md border border-line-soft bg-paper px-3 py-2.5"
				>
					{#if newest}
						<span class="eyebrow">merge #{newest.index + 1}</span>
						<span class="tok" style="background: {wash(newest.a)}; border-color: {edge(newest.a)};"
							>{@render glyphs(newest.aText)}</span
						>
						<span class="num text-[11px] text-ink-3">+</span>
						<span class="tok" style="background: {wash(newest.b)}; border-color: {edge(newest.b)};"
							>{@render glyphs(newest.bText)}</span
						>
						<span class="num text-[11px] text-ink-3">→</span>
						{#key k}
							<span
								class="tok fused"
								style="background: {wash(
									newest.newId
								)}; border-color: var(--warm); outline: 1.5px solid var(--warm);"
								>{@render glyphs(newest.text)}</span
							>
						{/key}
						<span class="num text-[10.5px] text-ink-3">
							appeared {fmt(newest.count)} times · token {newest.newId} · counted in {newest.ms.toFixed(
								1
							)} ms
						</span>
					{:else}
						<span class="num text-[11px] text-ink-3">
							no merges yet — the vocabulary is the 69 raw characters
						</span>
					{/if}
				</div>

				<!-- the newest vocabulary -->
				<div class="mt-3.5">
					<span class="eyebrow">newest {Math.min(k, RECENT)} tokens · freshest first</span>
					<div class="mt-1.5 flex min-h-8 flex-wrap items-center gap-1.5">
						{#if recent.length === 0}
							<span class="num text-[11px] text-ink-3">—</span>
						{:else}
							{#each recent as m (m.newId)}
								<span
									class="tok"
									style="background: {wash(m.newId)}; border-color: {m.index + 1 === k
										? 'var(--warm)'
										: edge(m.newId)}; {m.index + 1 === k
										? 'outline: 1.5px solid var(--warm);'
										: ''}"
									title="merge #{m.index + 1} · appeared {fmt(m.count)} times"
									>{@render glyphs(m.text)}<span class="tok-n">#{m.index + 1}</span></span
								>
							{/each}
						{/if}
					</div>
				</div>

				<!-- the sentence, retokenized at merge k -->
				<div class="mt-4 rounded-md border border-line-soft bg-paper px-3 py-2.5">
					<span class="eyebrow">the same sentence, with the first {k} merges</span>
					<div class="mt-2 flex flex-wrap items-center" style="row-gap: 5px; gap-x: 2px;">
						{#each sentToks as t, i (i)}
							<span
								class="tok"
								class:fused={newest !== null && t.id === newest.newId}
								style="background: {wash(i)}; border-color: {newest !== null &&
								t.id === newest.newId
									? 'var(--warm)'
									: edge(i)}; {newest !== null && t.id === newest.newId
									? 'outline: 1.5px solid var(--warm);'
									: ''}">{@render glyphs(t.text)}</span
							>
						{/each}
					</div>
					<div class="mt-2.5 flex flex-wrap items-center gap-3">
						<span class="num text-[11px] text-ink-2">
							{SENTENCE.length} chars → <span class="text-ink">{sentToks.length}</span> tokens
						</span>
						<span class="h-1.5 max-w-56 min-w-24 flex-1 overflow-hidden rounded-full bg-line-soft">
							<span
								class="block h-full rounded-full"
								style="width: {Math.min(((sentCpt - 1) / 2.6) * 100, 100).toFixed(
									1
								)}%; background: var(--accent); transition: width 160ms ease;"
							></span>
						</span>
						<span class="num text-[11px] text-ink">{sentCpt.toFixed(2)} chars/token</span>
					</div>
				</div>

				<!-- the whole-corpus score, and the history scrubber -->
				<div class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2.5">
					<p class="num text-[11.5px] text-ink-2">
						whole corpus: {fmt(originalLen)} chars →
						<span class="text-ink">{fmt(corpusToks)}</span>
						tokens ({corpusCpt.toFixed(2)} chars/token) · vocabulary {69 + k}
						{#if scribeMerges >= 0}
							<span class="text-ink-3">
								· the scribe below reads {scribeMerges === merges.length
									? 'exactly this vocabulary'
									: `${scribe.vocabSize} of them (${scribeMerges} merges)`}
							</span>
						{:else}
							<span class="text-ink-3">· the scribe below still reads single characters</span>
						{/if}
					</p>
					{#if scrubbable}
						<span class="ml-auto flex items-center gap-1.5">
							<button
								class="ctl"
								onclick={() => stepK(-merges.length)}
								disabled={k === 0}
								aria-label="rewind to zero merges"
							>
								<RotateCcw size={13} aria-hidden="true" />
							</button>
							<button
								class="ctl"
								onclick={() => stepK(-1)}
								disabled={k === 0}
								aria-label="back one merge"
							>
								<StepBack size={13} aria-hidden="true" />
							</button>
							<button
								class="ctl"
								onclick={toggleReplay}
								aria-label={playing ? 'pause the replay' : 'replay the merges'}
							>
								{#if playing}
									<Pause size={13} aria-hidden="true" />
								{:else}
									<Play size={13} aria-hidden="true" />
								{/if}
							</button>
							<button
								class="ctl"
								onclick={() => stepK(1)}
								disabled={k >= merges.length}
								aria-label="forward one merge"
							>
								<StepForward size={13} aria-hidden="true" />
							</button>
						</span>
						<span class="max-w-72 min-w-40 flex-1">
							<Slider
								label="merges applied"
								bind:value={k}
								min={0}
								max={merges.length}
								step={1}
								format={(v) => `${v} / ${merges.length}`}
							/>
						</span>
					{/if}
				</div>
			</div>
		{/if}
	</div>
</Plate>

<style>
	/* chip language: real whitespace inside (white-space: pre), a hairline
	 * border so chips read on dark paper too, newlines shown as a quiet ↵ */
	.tok {
		font-family: var(--font-mono);
		font-size: 12.5px;
		line-height: 1.4;
		white-space: pre;
		border: 1px solid var(--line);
		border-radius: 4px;
		padding: 2px 3px;
		margin: 0 1px;
		color: var(--ink);
	}
	.tok-n {
		font-size: 9px;
		color: var(--ink-3);
		margin-left: 4px;
	}
	.nl {
		opacity: 0.5;
		font-size: 0.85em;
		user-select: none;
	}
	.sp {
		opacity: 0.4;
		user-select: none;
	}
	.fused {
		animation: pop 260ms ease-out;
	}
	@keyframes pop {
		0% {
			transform: scale(0.6);
		}
		60% {
			transform: scale(1.12);
		}
		100% {
			transform: scale(1);
		}
	}
	.ctl {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 28px;
		height: 28px;
		border: 1px solid var(--line);
		border-radius: 6px;
		color: var(--ink-2);
		background: var(--surface);
		transition:
			border-color 100ms ease,
			color 100ms ease;
	}
	.ctl:hover:not(:disabled) {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.ctl:disabled {
		opacity: 0.4;
		cursor: default;
	}
</style>
