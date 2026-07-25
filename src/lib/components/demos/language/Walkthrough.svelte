<script lang="ts">
	// Plate V — one forward pass, opened up. Five stages down the page, joined by
	// hairline flow lines. Shapes and captions are architecture; every NUMBER is
	// read live from the model trained in Plate III: real attention rows
	// (softmax(QKᵀ/√d + mask)) and the real next-token distribution. The
	// temperature control re-softmaxes the stored logit row client-side — the
	// model is never asked again, which is exactly what temperature is.
	import { ChevronDown, Dices, Repeat, ScanEye } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	// aliased: this file does real arithmetic, and the component would shadow
	// the global Math object
	import MathTex from '$lib/components/ui/Math.svelte';
	import { inview } from '$lib/components/ui/inview';
	import LabGate from './LabGate.svelte';
	import MatrixGlyph from './MatrixGlyph.svelte';
	import { scribe, SCRIBE_CONFIG, type AttentionSnap } from './lab.svelte';

	const DEFAULT_PROMPT = 'Once upon a time there was a tiny cat';
	const MAX_CHARS = 40;
	const { nLayer, nEmbd, nHead } = SCRIBE_CONFIG;
	const headDim = nEmbd / nHead;
	const layerIdx = Array.from({ length: nLayer }, (_, k) => k);
	const headIdx = Array.from({ length: nHead }, (_, k) => k);

	let prompt = $state(DEFAULT_PROMPT);
	let att = $state<AttentionSnap | null>(null);
	let logits = $state<Float32Array | null>(null); // log-probs, as returned
	let toks = $state<string[]>([]);
	let layer = $state(0);
	let head = $state(0);
	let busy = $state(false);
	let ranStep = $state(0);
	let ranPrompt = $state('');
	let temperature = $state(1);
	let hov = $state<{ i: number; j: number } | null>(null);

	const usable = $derived(scribe.phase === 'ready' || scribe.phase === 'training');
	const stale = $derived(att !== null && ranPrompt !== prompt);

	/** One pass: attention rows and the next-token row for the same context. */
	async function runPass(text: string) {
		if (busy || !usable) return;
		const ids = scribe.encode(text).slice(0, MAX_CHARS);
		if (ids.length < 2) return;
		busy = true;
		hov = null;
		const atStep = scribe.step;
		const [a, row] = await Promise.all([scribe.attention(ids), scribe.nextDistribution(ids)]);
		busy = false;
		if (a && row) {
			att = a;
			logits = row;
			toks = ids.map((id) => scribe.charOf(id));
			ranStep = atStep;
			ranPrompt = text;
		}
	}

	// first pass as soon as the engine is ready, so the plate is never empty
	let kicked = false;
	$effect(() => {
		if (usable && !kicked && !att) {
			kicked = true;
			void runPass(prompt);
		}
	});

	const N = $derived(att ? Math.min(att.seqLen, toks.length) : 0);
	const idx = $derived(Array.from({ length: N }, (_, k) => k));

	// layer buffer is [H·S, S] with S = blockSize; row h·S+i is head h, query i
	function w(i: number, j: number): number {
		if (!att) return 0;
		const S = att.blockSize;
		return att.layers[layer][(head * S + i) * S + j];
	}

	const CELL = 16;
	const GUT = 20;
	const PAD = 4;
	const gw = $derived(GUT + N * CELL + PAD);
	const axis = (c: string) => (c === ' ' ? '·' : c);
	// perceptual lift: small-but-real weights stay visible
	const alpha = (v: number) => Math.min(1, Math.pow(Math.max(v, 0), 0.55)) * 0.95;

	/** Re-softmax the STORED log-prob row at the chosen temperature. */
	const dist = $derived.by(() => {
		const lp = logits;
		if (!lp) return [];
		const t = Math.max(0.05, temperature);
		let mx = -Infinity;
		for (let i = 0; i < lp.length; i++) mx = Math.max(mx, lp[i] / t);
		let sum = 0;
		const ps = new Float64Array(lp.length);
		for (let i = 0; i < lp.length; i++) {
			ps[i] = Math.exp(lp[i] / t - mx);
			sum += ps[i];
		}
		const rows: Array<{ id: number; p: number }> = [];
		for (let i = 0; i < lp.length; i++) rows.push({ id: i, p: ps[i] / sum });
		rows.sort((a, b) => b.p - a.p);
		return rows.slice(0, 10);
	});
	const topP = $derived(dist.length ? dist[0].p : 1);

	/** Draw from the displayed distribution, append, and run the pass again. */
	async function drawAndAppend() {
		if (!dist.length) return;
		let r = Math.random();
		let pick = dist[0].id;
		for (const row of dist) {
			r -= row.p;
			if (r <= 0) {
				pick = row.id;
				break;
			}
		}
		const next = prompt + scribe.charOf(pick);
		prompt = next.slice(-MAX_CHARS);
		await runPass(prompt);
	}

	const glyphOf = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c);
</script>

{#snippet flow(label?: string)}
	<div class="flex flex-col items-center py-1" aria-hidden="true">
		<span class="block h-4 w-px" style="background: var(--line);"></span>
		{#if label}
			<span
				class="num rounded-full border border-line-soft bg-surface-2 px-2 py-0.5 text-[9.5px] text-ink-3"
				>{label}</span
			>
			<span class="block h-4 w-px" style="background: var(--line);"></span>
		{:else}
			<ChevronDown size={13} style="color: var(--line);" />
		{/if}
	</div>
{/snippet}

{#snippet stageHead(n: number, title: string, sub: string)}
	<div class="mb-2.5 flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
		<span class="eyebrow" style="color: var(--warm);">stage {n}</span>
		<span class="font-serif text-[15px]" style="font-variation-settings: 'opsz' 14;">{title}</span>
		<span class="num text-[10.5px] text-ink-3">{sub}</span>
	</div>
{/snippet}

<Plate
	n={5}
	title="One forward pass, opened up"
	caption="The five stages a character travels through on its way to a prediction. The shapes are this model's real architecture; the attention rows and the final distribution are real numbers, read from the weights you trained above — not an illustration of a transformer, but a reading of one."
>
	{#snippet status()}
		{#if att}
			<span>{N} tokens</span>
			<span aria-hidden="true">·</span>
			<span>read at step {ranStep}</span>
		{:else if usable}
			<span>{busy ? 'running the pass…' : `scribe at step ${scribe.step}`}</span>
		{:else}
			<span>waiting for the scribe</span>
		{/if}
	{/snippet}

	<div use:inview={() => void scribe.boot()}>
		{#if !usable}
			<LabGate
				note="this plate opens up the scribe from the plate above — it boots when you reach it"
			/>
		{:else}
			<!-- the prompt this pass runs on -->
			<div class="flex items-end gap-2 border-b border-line-soft px-4 pt-3.5 pb-3">
				<label class="min-w-0 flex-1">
					<span class="eyebrow mb-1 block">the context · up to {MAX_CHARS} characters</span>
					<input
						class="input"
						maxlength={MAX_CHARS}
						bind:value={prompt}
						onkeydown={(e) => {
							if (e.key === 'Enter') void runPass(prompt);
						}}
					/>
				</label>
				<Btn kind="primary" onclick={() => void runPass(prompt)} disabled={busy}>
					<ScanEye size={13} aria-hidden="true" />
					{stale ? 'Re-run' : 'Run the pass'}
				</Btn>
			</div>

			{#if !att}
				<div class="flex min-h-56 items-center justify-center px-6 py-8">
					<span class="num text-[11px] text-ink-3">
						{busy ? 'reading the weights…' : 'press run — every number below comes off your GPU'}
					</span>
				</div>
			{:else}
				<div class="px-4 py-4">
					<!-- ── stage 1: tokens → vectors ── -->
					<div class="stage">
						{@render stageHead(1, 'Characters become vectors', 'lookup + position')}
						<div class="flex flex-wrap items-center gap-x-4 gap-y-3">
							<div class="flex min-w-0 flex-1 flex-wrap gap-1">
								{#each idx.slice(0, 24) as i (i)}
									<span class="chip-tok">{axis(toks[i])}</span>
								{/each}
								{#if N > 24}
									<span class="num self-center text-[10px] text-ink-3">+{N - 24} more</span>
								{/if}
							</div>
							<MatrixGlyph
								rows={N}
								cols={nEmbd}
								label="{N} × {nEmbd}"
								w={120}
								h={54}
								tone="accent"
								seed={3}
							/>
						</div>
						<p class="stage-note">
							Each character is one row of a lookup table, plus a vector for <em>where</em> it sits
							— the model has no other way to know order.
							<MathTex tex="x_i = E[t_i] + P[i]" />
						</p>
					</div>

					{@render flow()}

					<!-- ── stage 2: Q, K, V ── -->
					<div class="stage">
						{@render stageHead(2, 'Three questions per vector', 'query, key, value')}
						<div class="flex flex-wrap items-start gap-x-5 gap-y-3">
							<MatrixGlyph
								rows={nEmbd}
								cols={nEmbd}
								label="W_Q · 96 × 96"
								w={78}
								h={52}
								seed={11}
							/>
							<MatrixGlyph
								rows={nEmbd}
								cols={nEmbd}
								label="W_K · 96 × 96"
								w={78}
								h={52}
								seed={12}
							/>
							<MatrixGlyph
								rows={nEmbd}
								cols={nEmbd}
								label="W_V · 96 × 96"
								w={78}
								h={52}
								seed={13}
							/>
							<p class="stage-note min-w-48 flex-1">
								Every vector projects into a <em>query</em> (what am I looking for?), a
								<em>key</em>
								(what do I offer?) and a <em>value</em> (what I'd pass along). Split across
								{nHead} heads, each gets {headDim} dimensions to work in.
								<MathTex tex={'\\mathrm{Attn} = \\sigma\\!\\left(QK^{\\top}/\\sqrt{d}\\right)V'} />
							</p>
						</div>
					</div>

					{@render flow()}

					<!-- ── stage 3: attention, for real ── -->
					<div class="stage stage-live">
						{@render stageHead(3, 'Tokens read each other', 'real softmax rows')}
						<div class="mb-2.5 flex flex-wrap items-center gap-x-4 gap-y-2">
							<div class="flex items-center gap-1.5">
								<span class="eyebrow">layer</span>
								{#each layerIdx as li (li)}
									<button
										class="chip"
										class:chip-on={layer === li}
										aria-pressed={layer === li}
										onclick={() => (layer = li)}>{li + 1}</button
									>
								{/each}
							</div>
							<div class="flex items-center gap-1.5">
								<span class="eyebrow">head</span>
								{#each headIdx as h (h)}
									<button
										class="chip"
										class:chip-on={head === h}
										aria-pressed={head === h}
										onclick={() => (head = h)}>{h + 1}</button
									>
								{/each}
							</div>
						</div>
						<div class="overflow-x-auto">
							<svg
								width={gw}
								height={gw}
								viewBox="0 0 {gw} {gw}"
								class="mx-auto block"
								role="img"
								aria-label="Causal attention matrix: rows are the position being predicted, columns the earlier positions it attends to; darker ultramarine is more attention."
								onpointerleave={() => (hov = null)}
							>
								{#each idx as j (j)}
									<text
										x={GUT + j * CELL + CELL / 2}
										y={GUT - 7}
										text-anchor="middle"
										class="num"
										font-size="9"
										fill={hov?.j === j ? 'var(--ink)' : 'var(--ink-3)'}>{axis(toks[j])}</text
									>
									<text
										x={GUT - 7}
										y={GUT + j * CELL + CELL / 2 + 3}
										text-anchor="end"
										class="num"
										font-size="9"
										fill={hov?.i === j ? 'var(--ink)' : 'var(--ink-3)'}>{axis(toks[j])}</text
									>
								{/each}
								{#each idx as i (i)}
									<rect
										x={GUT}
										y={GUT + i * CELL + 0.5}
										width={(i + 1) * CELL}
										height={CELL - 1}
										fill="var(--surface-2)"
										opacity="0.55"
									/>
								{/each}
								{#each idx as i (i)}
									{#each idx as j (j)}
										{#if j <= i}
											<rect
												x={GUT + j * CELL + 1}
												y={GUT + i * CELL + 1}
												width={CELL - 2}
												height={CELL - 2}
												rx="2.5"
												fill="var(--accent)"
												fill-opacity={alpha(w(i, j))}
												role="presentation"
												onpointerenter={() => (hov = { i, j })}
											>
												<title
													>predicting “{axis(toks[i])}” ({i}) ← “{axis(toks[j])}” ({j}): {(
														w(i, j) * 100
													).toFixed(1)}%</title
												>
											</rect>
										{/if}
									{/each}
								{/each}
							</svg>
						</div>
						<p class="num mt-1.5 min-h-5 text-[11px] text-ink-2">
							{#if hov}
								predicting “{axis(toks[hov.i])}” (position {hov.i}) — this head spends {(
									w(hov.i, hov.j) * 100
								).toFixed(1)}% of its budget on “{axis(toks[hov.j])}” (position {hov.j})
							{:else}
								each row sums to 1 · spaces drawn as · · hover any cell
							{/if}
						</p>
						<p class="stage-note">
							Rows are the position being predicted; columns are where its attention lands, and each
							row is a budget of exactly one spent over the past. Only the lower triangle exists:
							the causal mask hides the future, because a model that could peek at the answer would
							learn nothing by guessing it.
						</p>
					</div>

					{@render flow()}

					<!-- ── stage 4: the MLP ── -->
					<div class="stage">
						{@render stageHead(4, 'Each token thinks alone', 'widen, rectify, narrow')}
						<div class="flex flex-wrap items-center gap-x-4 gap-y-3">
							<MatrixGlyph
								rows={nEmbd}
								cols={4 * nEmbd}
								label="96 → 384"
								w={132}
								h={48}
								tone="warm"
								seed={21}
							/>
							<span class="num text-[11px] text-ink-3">relu</span>
							<MatrixGlyph
								rows={4 * nEmbd}
								cols={nEmbd}
								label="384 → 96"
								w={72}
								h={48}
								tone="warm"
								seed={22}
							/>
							<p class="stage-note min-w-48 flex-1">
								Attention is where tokens talk; this is where each one thinks alone. The vector is
								widened fourfold, rectified — negatives clipped to zero, the only nonlinearity in
								the block — and squeezed back to size.
							</p>
						</div>
					</div>

					{@render flow(`the block above repeats ×${nLayer}`)}

					<div
						class="mb-3 flex items-start gap-2 rounded-md border border-line-soft bg-surface-2 px-3 py-2"
					>
						<Repeat size={13} style="color: var(--warm);" aria-hidden="true" />
						<p class="text-[11.5px] leading-relaxed text-ink-2">
							Stages 2 through 4 are one <em>block</em>, and this model stacks {nLayer} of them — which
							is why the layer chips above offer two choices. Frontier models stack the same block roughly
							a hundred deep, with vectors thousands of numbers wide. Nothing else about the picture changes.
						</p>
					</div>

					{@render flow()}

					<!-- ── stage 5: logits → the next character ── -->
					<div class="stage stage-live">
						{@render stageHead(5, 'A guess, at last', 'real distribution')}
						<div class="flex flex-wrap items-end gap-x-4 gap-y-2.5">
							<div class="w-40">
								<Slider
									label="temperature"
									bind:value={temperature}
									min={0.2}
									max={1.6}
									step={0.05}
									tone="warm"
									format={(v) => v.toFixed(2)}
								/>
							</div>
							<Btn onclick={() => void drawAndAppend()} disabled={busy}>
								<Dices size={13} aria-hidden="true" /> Draw one & continue
							</Btn>
							<span class="num text-[10.5px] text-ink-3">
								top guess {(topP * 100).toFixed(1)}%
							</span>
						</div>
						<div class="mt-3 flex flex-col gap-1">
							{#each dist as row, i (row.id)}
								<div class="flex items-center gap-2">
									<span
										class="num w-5 text-[12px]"
										style="color: {i === 0 ? 'var(--accent)' : 'var(--ink)'};"
										>{glyphOf(scribe.charOf(row.id))}</span
									>
									<span class="h-2 flex-1 overflow-hidden rounded-sm bg-line-soft">
										<span
											class="block h-full rounded-sm"
											style="width: {(row.p * 100).toFixed(2)}%; background: {i === 0
												? 'var(--accent)'
												: 'color-mix(in srgb, var(--accent) 55%, transparent)'}; transition: width 140ms ease;"
										></span>
									</span>
									<span class="num w-11 text-right text-[10px] text-ink-3"
										>{(row.p * 100).toFixed(1)}%</span
									>
								</div>
							{/each}
						</div>
						<p class="stage-note">
							The final vector meets one more matrix — {nEmbd} numbers in, 69 scores out, one per character
							— and a softmax turns those scores into probabilities.
							<MathTex tex={'P(x_{t+1}) = \\sigma(z / T)'} />
							Dragging the temperature never asks the model again; it reshapes the same stored scores,
							flattening them as it rises. Draw one, and the drawn character joins the context: that is
							the whole loop the scribe runs, 160 times per sample.
						</p>
					</div>
				</div>
			{/if}
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
	.stage {
		border: 1px solid var(--line-soft);
		border-radius: 8px;
		background: var(--surface);
		padding: 12px 14px 13px;
	}
	/* the two stages whose contents are measured, not drawn */
	.stage-live {
		border-color: color-mix(in srgb, var(--accent) 32%, var(--line));
	}
	.stage-note {
		margin-top: 8px;
		font-family: var(--font-serif);
		font-size: 13px;
		line-height: 1.6;
		color: var(--ink-2);
		font-variation-settings: 'opsz' 13;
	}
	.stage-note :global(.katex) {
		font-size: 1em;
	}
	.chip-tok {
		font-family: var(--font-mono);
		font-size: 12px;
		line-height: 1.3;
		white-space: pre;
		border: 1px solid var(--line);
		border-radius: 4px;
		padding: 1px 4px;
		color: var(--ink);
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
	.chip {
		min-width: 24px;
		font-family: var(--font-sans);
		font-size: 10.5px;
		font-weight: 520;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--ink-2);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 999px;
		padding: 3px 9px;
		transition:
			color 100ms ease,
			border-color 100ms ease,
			background 100ms ease;
	}
	.chip:hover {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.chip-on {
		color: var(--accent);
		background: var(--accent-soft);
		border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
	}
</style>
