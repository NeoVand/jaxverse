<script lang="ts">
	// Plate III — the scribe learns to write. Left column: the descent from
	// ln 69 nats against the dashed loss of knowing nothing, plus the controls
	// that ask it questions. Right column: the desk — the newest sample large,
	// its predecessors archived in two columns beneath at reduced opacity. The
	// same prompt is re-asked after every burst, so the only thing that changes
	// between samples is the weights. That timeline IS the demo.
	import { Feather, Pause, Play, RotateCcw } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import LabGate from './LabGate.svelte';
	import { scribe, AUTO_PROMPT, TRAIN_CHUNK, UNIFORM_NATS } from './lab.svelte';

	let promptText = $state(AUTO_PROMPT);
	let temperature = $state(0.8);

	const usable = $derived(scribe.phase === 'ready' || scribe.phase === 'training');
	const latest = $derived(scribe.samples[0] ?? null);
	const archive = $derived(scribe.samples.slice(1));

	// ── loss chart geometry (dynamic width, undistorted text) ──
	const CH = 190;
	const PADL = 26;
	const PADR = 10;
	const PADT = 18;
	const PADB = 20;
	const YMAX = 4.6;
	let chartW = $state(0);
	const cw = $derived(chartW || 420);
	const xMax = $derived.by(() => {
		const lastVal = scribe.valPoints.length ? scribe.valPoints[scribe.valPoints.length - 1][0] : 0;
		return Math.max(400, scribe.step, lastVal);
	});
	const xPix = (s: number) => PADL + (s / xMax) * (cw - PADL - PADR);
	const yPix = (l: number) => PADT + (1 - Math.min(l, YMAX) / YMAX) * (CH - PADT - PADB);

	const trainPath = $derived.by(() => {
		const pts = scribe.trainLoss;
		if (pts.length < 2) return '';
		// full history since step 1, thinned so the path stays light
		const stride = Math.max(1, Math.ceil(pts.length / 700));
		let d = '';
		for (let k = 0; k < pts.length; k += stride)
			d += `${d ? 'L' : 'M'}${xPix(pts[k][0]).toFixed(1)} ${yPix(pts[k][1]).toFixed(1)}`;
		const last = pts[pts.length - 1];
		d += `L${xPix(last[0]).toFixed(1)} ${yPix(last[1]).toFixed(1)}`;
		return d;
	});

	const xTicks = $derived([0, Math.round(xMax / 2), xMax]);

	const fmtLoss = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '—');
	const fmtTok = (v: number) =>
		v >= 1000 ? `${(v / 1000).toFixed(1)}k tok/s` : `${Math.round(v)} tok/s`;

	function sampleNow() {
		void scribe.sampleNow(promptText, temperature);
	}
</script>

<Plate
	n={3}
	title="Watch it learn to write"
	caption="After every burst of steps the desk re-asks the same prompt at the same temperature, so the only thing that changes between samples is the weights. Step 0 is uniform noise; word-shapes arrive around step 200 and sentence-shaped grammar near 1,500 — that timeline, noise to spelling to syntax, is the demo."
>
	{#snippet status()}
		{#if scribe.phase === 'idle' || scribe.phase === 'loading'}
			<span>{scribe.phase === 'idle' ? 'waiting' : 'loading…'}</span>
		{:else if scribe.phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{:else if scribe.phase === 'no-webgpu'}
			<span>needs webgpu</span>
		{:else}
			<span>{scribe.device} · {scribe.paramCount.toLocaleString('en-US')} params</span>
			<span aria-hidden="true">·</span>
			<span>step {scribe.step}</span>
			<span aria-hidden="true">·</span>
			<span>loss {fmtLoss(scribe.lossNow)}</span>
			<span aria-hidden="true">·</span>
			<span>{fmtTok(scribe.tokPerSec)}</span>
		{/if}
	{/snippet}

	<div use:inview={() => void scribe.boot()}>
		{#if !usable}
			<LabGate
				tall
				note="the scribe boots when you reach it — 1.5 MB of stories and a 243,648-parameter transformer, built on your GPU"
			/>
		{:else}
			<!-- transport -->
			<div
				class="flex flex-wrap items-center gap-x-3 gap-y-2.5 border-b border-line-soft px-4 py-3"
			>
				<Btn kind="primary" onclick={() => scribe.toggle()}>
					{#if scribe.phase === 'training'}
						<Pause size={13} aria-hidden="true" /> Pause
					{:else}
						<Play size={13} aria-hidden="true" /> Train
					{/if}
				</Btn>
				<Btn onclick={() => void scribe.reset()} title="Back to the step-0 weights">
					<RotateCcw size={13} aria-hidden="true" /> Reset
				</Btn>
				<span class="num text-[11px] text-ink-3">{TRAIN_CHUNK} steps per burst, then a sample</span>
				{#if scribe.spoke}
					<span class="ml-auto font-serif text-[13px] italic" style="color: var(--good);">
						it speaks — keep going and listen to the grammar arrive
					</span>
				{/if}
			</div>

			<div class="grid grid-cols-1 gap-px bg-line-soft lg:grid-cols-2">
				<!-- left: the descent, and the questions you can ask -->
				<div class="flex flex-col bg-surface p-4">
					<div class="flex flex-wrap items-baseline justify-between gap-2">
						<span class="eyebrow">loss · nats per character</span>
						<span class="num flex items-center gap-3 text-[10px] text-ink-3">
							<span class="flex items-center gap-1.5">
								<span class="inline-block h-0.5 w-4" style="background: var(--accent);"></span>
								train
							</span>
							<span class="flex items-center gap-1.5">
								<span
									class="inline-block h-[5px] w-[5px] rounded-full"
									style="background: var(--warm);"
								></span>
								held-out
							</span>
						</span>
					</div>
					<div class="mt-2" bind:clientWidth={chartW}>
						<svg
							width="100%"
							height={CH}
							viewBox="0 0 {cw} {CH}"
							class="block"
							role="img"
							aria-label="Training loss per step in ultramarine with held-out loss dots in vermilion. A dashed line marks the uniform-guess loss, natural log of 69, about 4.23 nats — the model starts there."
						>
							{#each [1, 2, 3, 4] as gy (gy)}
								<line
									x1={PADL}
									x2={cw - PADR}
									y1={yPix(gy)}
									y2={yPix(gy)}
									stroke="var(--line-soft)"
									stroke-width="1"
								/>
							{/each}
							{#each [0, 1, 2, 3, 4] as gy (gy)}
								<text
									x={PADL - 6}
									y={yPix(gy) + 3.5}
									text-anchor="end"
									class="num"
									font-size="10"
									fill="var(--ink-3)">{gy}</text
								>
							{/each}
							<line
								x1={PADL}
								x2={cw - PADR}
								y1={CH - PADB}
								y2={CH - PADB}
								stroke="var(--line)"
								stroke-width="1"
							/>
							<line
								x1={PADL}
								x2={cw - PADR}
								y1={yPix(UNIFORM_NATS)}
								y2={yPix(UNIFORM_NATS)}
								stroke="var(--ink-3)"
								stroke-width="1"
								stroke-dasharray="4 3"
							/>
							<text
								x={cw - PADR}
								y={yPix(UNIFORM_NATS) - 5}
								text-anchor="end"
								class="num"
								font-size="10"
								fill="var(--ink-3)">uniform guess · ln 69 ≈ 4.23</text
							>
							{#if trainPath}
								<path
									d={trainPath}
									fill="none"
									stroke="var(--accent)"
									stroke-width="1.4"
									stroke-linejoin="round"
								/>
							{/if}
							{#each scribe.valPoints as pt (pt[0])}
								<circle cx={xPix(pt[0])} cy={yPix(pt[1])} r="2.6" fill="var(--warm)" />
							{/each}
							{#each xTicks as t, ti (t)}
								<text
									x={xPix(t)}
									y={CH - 6}
									text-anchor={ti === 0 ? 'start' : ti === 2 ? 'end' : 'middle'}
									class="num"
									font-size="10"
									fill="var(--ink-3)">{t}</text
								>
							{/each}
						</svg>
					</div>
					<p class="mt-2 text-[11px] leading-relaxed text-ink-3">
						reading the gauge: at loss ℓ the model hesitates among about e<sup>ℓ</sup> characters — 4.23
						means all 69 of them, 1.2 means about three. the dashed line is where knowing nothing lives;
						the model starts exactly there.
					</p>

					<!-- ask it something -->
					<div class="mt-auto border-t border-line-soft pt-3.5">
						<div class="flex flex-wrap items-end gap-x-3 gap-y-2.5">
							<label class="min-w-40 flex-1">
								<span class="eyebrow mb-1 block">prompt</span>
								<input
									class="input"
									maxlength="48"
									bind:value={promptText}
									onkeydown={(e) => {
										if (e.key === 'Enter') sampleNow();
									}}
								/>
							</label>
							<div class="w-28">
								<Slider
									label="temp"
									bind:value={temperature}
									min={0.2}
									max={1.4}
									step={0.05}
									tone="warm"
									format={(v) => v.toFixed(2)}
								/>
							</div>
							<Btn onclick={sampleNow} disabled={scribe.sampling}>
								<Feather size={13} aria-hidden="true" /> Sample now
							</Btn>
						</div>
						<p class="mt-1.5 text-[11px] leading-snug text-ink-3">
							temperature rescales confidence before each draw — 0.2 plays the favorite, 1.4
							gambles. characters outside the model's 69-glyph alphabet are dropped from the prompt.
						</p>
					</div>
				</div>

				<!-- right: the desk -->
				<div class="flex flex-col bg-surface p-4">
					<div class="flex flex-wrap items-baseline justify-between gap-2">
						<span class="eyebrow">the scribe's desk</span>
						<span class="num text-[10px] text-ink-3">auto: “{AUTO_PROMPT}” · temp 0.80</span>
					</div>

					{#if latest}
						{#key latest.id}
							<div class="sample-card mt-2.5 rounded-md border border-line bg-paper px-4 py-3">
								<div
									class="num mb-1.5 flex items-baseline justify-between gap-3 text-[10px] text-ink-3"
								>
									<span style="color: var(--accent);">step {latest.step}</span>
									<span>temp {latest.temperature.toFixed(2)}{latest.auto ? '' : ' · yours'}</span>
								</div>
								<p
									class="font-serif text-[15.5px] leading-[1.62]"
									style="font-variation-settings: 'opsz' 14;"
								>
									<span class="whitespace-pre-wrap text-ink-3">{latest.prompt}</span><span
										class="whitespace-pre-wrap">{latest.text}</span
									>
								</p>
							</div>
						{/key}
					{:else}
						<div class="mt-2.5 rounded-md border border-dashed border-line px-4 py-8 text-center">
							<span class="num text-[11px] text-ink-3">
								{scribe.sampling ? 'listening…' : 'the first sample arrives in a moment'}
							</span>
						</div>
					{/if}

					{#if archive.length}
						<span class="eyebrow mt-3.5 block">earlier drafts</span>
						<div class="mt-1.5 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
							{#each archive as s, i (s.id)}
								<div
									class="rounded-md border border-line-soft bg-paper px-3 py-2"
									style="opacity: {Math.max(0.32, 0.7 - i * 0.14)};"
								>
									<div class="num mb-1 text-[9.5px] text-ink-3">
										step {s.step}{s.auto ? '' : ' · yours'}
									</div>
									<p
										class="font-serif text-[12.5px] leading-[1.55]"
										style="font-variation-settings: 'opsz' 12;"
									>
										<span class="whitespace-pre-wrap text-ink-3">{s.prompt}</span><span
											class="whitespace-pre-wrap">{s.text}</span
										>
									</p>
								</div>
							{/each}
						</div>
					{/if}
				</div>
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
	/* newest sample settles onto the desk; reduced-motion zeroes this globally */
	.sample-card {
		animation: settle 480ms ease both;
	}
	@keyframes settle {
		from {
			opacity: 0;
			transform: translateY(5px);
		}
	}
</style>
