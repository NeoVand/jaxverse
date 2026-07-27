<script lang="ts">
	// Plate V — one forward pass, opened up. Five stages down the page, joined by
	// hairline flow lines. Shapes and captions are architecture; every NUMBER is
	// read live from the model trained in Plate III: real attention rows
	// (softmax(QKᵀ/√d + mask)) and the real next-token distribution. The
	// temperature control re-softmaxes the stored logit row client-side — the
	// model is never asked again, which is exactly what temperature is.
	import { Dices, ScanEye } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	// aliased: this file does real arithmetic, and the component would shadow
	// the global Math object
	import MathTex from '$lib/components/ui/Math.svelte';
	import { inview } from '$lib/components/ui/inview';
	import LabGate from './LabGate.svelte';
	import MatrixGlyph from './MatrixGlyph.svelte';
	import { scribe, SCRIBE_SHAPE, type AttentionSnap } from './lab.svelte';

	const DEFAULT_PROMPT = 'Once upon a time there was a little girl who wanted a cat';
	const MAX_CHARS = 60;
	/** The attention grid is square and labelled on both edges; past this many
	 * tokens the labels stop being readable, so the context is trimmed. */
	const MAX_TOKENS = 26;
	const { nLayer, nEmbd, nHead } = SCRIBE_SHAPE;
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
		const ids = scribe.encode(text).slice(-MAX_TOKENS);
		if (ids.length < 2) return;
		busy = true;
		hov = null;
		const atStep = scribe.step;
		const [a, row] = await Promise.all([scribe.attention(ids), scribe.nextDistribution(ids)]);
		busy = false;
		if (a && row) {
			att = a;
			logits = row;
			toks = ids.map((id) => scribe.textOf(id));
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

	// Word-piece labels need room: the gutter grows with the longest token, and
	// column labels stand upright rather than crowding each other.
	const CELL = 18;
	const PAD = 4;
	const GUT = $derived.by(() => {
		let longest = 1;
		for (const t of toks.slice(0, N)) longest = Math.max(longest, t.trim().length);
		return Math.min(64, 14 + longest * 5.6);
	});
	const gw = $derived(GUT + N * CELL + PAD);
	const gh = $derived(GUT + N * CELL + PAD);
	/** Leading spaces are what distinguish " the" from "the"; show them. */
	const axis = (c: string) => c.replaceAll(' ', '␣').replaceAll('\n', '↵');
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
		const next = prompt + scribe.textOf(pick);
		prompt = next.slice(-MAX_CHARS);
		await runPass(prompt);
	}

	const glyphOf = (c: string) => c.replaceAll(' ', '␣').replaceAll('\n', '↵');
</script>

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
	caption="The five stages a token travels through on its way to a prediction. The shapes are this model's real architecture; the attention rows and the final distribution are real numbers, read from the weights you trained above — not an illustration of a transformer, but a reading of one. Because the vocabulary is word-pieces, the attention grid is words reading words: look for a verb attending to its subject, or a name being carried forward. Stages two through four form one block, and this model stacks two of them; frontier models stack the same block roughly a hundred deep, with vectors thousands of numbers wide, and nothing else about the picture changes."
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

	{#snippet actions()}
		{#if usable}
			<Btn onclick={() => void runPass(prompt)} disabled={busy}>
				<ScanEye size={12} aria-hidden="true" />
				{stale ? 'Re-run' : 'Run the pass'}
			</Btn>
		{/if}
	{/snippet}

	<div use:inview={() => void scribe.boot()}>
		{#if !usable}
			<LabGate
				note="this plate opens up the scribe from the plate above — it boots when you reach it"
			/>
		{:else}
			<!-- the prompt this pass runs on -->
			<div class="border-b border-line-soft px-4 pt-3.5 pb-3">
				<label class="block">
					<span class="eyebrow mb-1 block">
						the context · the last {MAX_TOKENS} tokens of it
					</span>
					<input
						class="input"
						maxlength={MAX_CHARS}
						bind:value={prompt}
						onkeydown={(e) => {
							if (e.key === 'Enter') void runPass(prompt);
						}}
					/>
				</label>
			</div>

			{#if !att}
				<div class="flex min-h-56 items-center justify-center px-6 py-8">
					<span class="num text-[11px] text-ink-3">
						{busy ? 'reading the weights…' : 'press run — every number below comes off your GPU'}
					</span>
				</div>
			{:else}
				<div class="px-4 py-4">
					<!-- ── the machine in one strip: the drawn stations; №3 and №5 run live below ── -->
					<div class="strip">
						<div class="cell">
							<span class="st-n" aria-hidden="true">1</span>
							<MatrixGlyph
								rows={N}
								cols={nEmbd}
								label="{N} × {nEmbd}"
								w={84}
								h={42}
								tone="accent"
								seed={3}
							/>
							<span class="st-t">tokens become vectors</span>
							<span class="st-s">lookup + position</span>
						</div>
						<span class="strip-arrow" aria-hidden="true">→</span>
						<div class="blockwrap">
							<span class="bw-label">one block · this model stacks {nLayer}</span>
							<div class="cell cell-bare">
								<span class="st-n" aria-hidden="true">2</span>
								<div class="flex gap-1.5">
									<MatrixGlyph rows={nEmbd} cols={nEmbd} label="W_Q" w={40} h={38} seed={11} />
									<MatrixGlyph rows={nEmbd} cols={nEmbd} label="W_K" w={40} h={38} seed={12} />
									<MatrixGlyph rows={nEmbd} cols={nEmbd} label="W_V" w={40} h={38} seed={13} />
								</div>
								<span class="st-t">query · key · value</span>
								<span class="st-s">{nHead} heads × {headDim} dims</span>
							</div>
							<span class="strip-arrow" aria-hidden="true">→</span>
							<div class="cell cell-bare cell-live">
								<span class="st-n" aria-hidden="true">3</span>
								<svg width="42" height="38" viewBox="0 0 42 38" aria-hidden="true">
									{#each [0, 1, 2, 3, 4] as ti (ti)}
										{#each [0, 1, 2, 3, 4] as tj (tj)}
											{#if tj <= ti}
												<rect
													x={tj * 8 + 1}
													y={ti * 7.4 + 1}
													width="6.6"
													height="6"
													rx="1.4"
													fill="var(--accent)"
													opacity={0.2 + 0.68 * (((ti * 3 + tj * 5 + 2) % 7) / 6)}
												/>
											{/if}
										{/each}
									{/each}
								</svg>
								<span class="st-t">attention</span>
								<span class="st-s">live · below left</span>
							</div>
							<span class="strip-arrow" aria-hidden="true">→</span>
							<div class="cell cell-bare">
								<span class="st-n" aria-hidden="true">4</span>
								<div class="flex items-center gap-1.5">
									<MatrixGlyph
										rows={nEmbd}
										cols={4 * nEmbd}
										label="{nEmbd} → {4 * nEmbd}"
										w={58}
										h={38}
										tone="warm"
										seed={21}
									/>
									<MatrixGlyph
										rows={4 * nEmbd}
										cols={nEmbd}
										label="{4 * nEmbd} → {nEmbd}"
										w={36}
										h={38}
										tone="warm"
										seed={22}
									/>
								</div>
								<span class="st-t">each token thinks alone</span>
								<span class="st-s">widen · relu · squeeze</span>
							</div>
						</div>
						<span class="strip-arrow" aria-hidden="true">→</span>
						<div class="cell cell-live">
							<span class="st-n" aria-hidden="true">5</span>
							<svg width="44" height="38" viewBox="0 0 44 38" aria-hidden="true">
								{#each [30, 9, 15, 5, 11, 3] as bh, bi (bi)}
									<rect
										x={bi * 7 + 1.5}
										y={35 - bh}
										width="5"
										height={bh}
										rx="1.2"
										fill="var(--accent)"
										opacity={bi === 0 ? 0.9 : 0.4}
									/>
								{/each}
								<line x1="0" y1="35.5" x2="44" y2="35.5" stroke="var(--line)" stroke-width="1" />
							</svg>
							<span class="st-t">the guess</span>
							<span class="st-s">live · below right</span>
						</div>
					</div>

					<div
						class="mt-3 grid items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(264px,324px)]"
					>
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
									height={gh}
									viewBox="0 0 {gw} {gh}"
									class="mx-auto block"
									role="img"
									aria-label="Causal attention matrix: rows are the position being predicted, columns the earlier positions it attends to; darker ultramarine is more attention."
									onpointerleave={() => (hov = null)}
								>
									{#each idx as j (j)}
										<text
											transform="translate({GUT + j * CELL + CELL / 2} {GUT - 6}) rotate(-90)"
											text-anchor="start"
											class="num"
											font-size="9"
											fill={hov?.j === j ? 'var(--ink)' : 'var(--ink-3)'}>{axis(toks[j])}</text
										>
										<text
											x={GUT - 6}
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
									each row sums to 1 · a leading space is drawn ␣ · hover any cell
								{/if}
							</p>
							<p class="stage-note">
								Rows are the position being predicted, columns where its attention lands — and only
								the lower triangle exists, because a model that could peek at the answer would learn
								nothing by guessing it.
								<MathTex tex={'\\mathrm{Attn} = \\sigma\\!\\left(QK^{\\top}/\\sqrt{d}\\right)V'} />
							</p>
						</div>

						<!-- ── stage 5: logits → the next token ── -->
						<div class="stage stage-live flex flex-col">
							{@render stageHead(5, 'A guess, at last', 'real distribution')}
							<Slider
								label="temperature"
								bind:value={temperature}
								min={0.2}
								max={1.6}
								step={0.05}
								tone="warm"
								format={(v) => v.toFixed(2)}
							/>
							<div class="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-2">
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
											class="num w-16 truncate text-[12px]"
											style="color: {i === 0 ? 'var(--accent)' : 'var(--ink)'};"
											>{glyphOf(scribe.textOf(row.id))}</span
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
							<p class="stage-note mt-auto">
								{scribe.vocabSize} scores, softmaxed into a belief.
								<MathTex tex={'P(x_{t+1}) = \\sigma(z / T)'} />
								Temperature reshapes the same stored scores — the model is never asked again. Draw one
								and it joins the context: that loop is all “generating text” has ever meant.
							</p>
						</div>
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
	/* ── the machine strip: drawn stations №1–5, arrows between ── */
	.strip {
		display: flex;
		flex-wrap: wrap;
		align-items: stretch;
		justify-content: center;
		gap: 8px;
	}
	.cell {
		position: relative;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 3px;
		padding: 9px 12px 7px;
		border: 1px solid var(--line-soft);
		border-radius: 8px;
		background: var(--surface);
	}
	.cell-bare {
		border-color: transparent;
		background: none;
		padding: 4px 6px 2px;
	}
	.cell-live {
		border-color: color-mix(in srgb, var(--accent) 32%, var(--line));
		background: var(--surface);
	}
	.blockwrap {
		position: relative;
		display: flex;
		flex-wrap: wrap;
		align-items: stretch;
		justify-content: center;
		gap: 6px;
		padding: 12px 10px 8px;
		border: 1px dashed var(--line);
		border-radius: 10px;
	}
	.bw-label {
		position: absolute;
		top: -7px;
		left: 12px;
		padding: 0 6px;
		background: var(--surface);
		font-family: var(--font-mono);
		font-size: 9px;
		letter-spacing: 0.04em;
		color: var(--ink-3);
		white-space: nowrap;
	}
	.strip-arrow {
		align-self: center;
		color: var(--ink-3);
		font-size: 12px;
	}
	.st-n {
		position: absolute;
		top: -7px;
		left: -7px;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 15px;
		height: 15px;
		border: 1px solid var(--warm);
		border-radius: 50%;
		background: var(--surface);
		color: var(--warm);
		font-family: var(--font-mono);
		font-size: 9px;
		font-weight: 600;
	}
	.st-t {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 12px;
		color: var(--ink-2);
		white-space: nowrap;
	}
	.st-s {
		font-family: var(--font-mono);
		font-size: 9px;
		letter-spacing: 0.03em;
		color: var(--ink-3);
		white-space: nowrap;
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
