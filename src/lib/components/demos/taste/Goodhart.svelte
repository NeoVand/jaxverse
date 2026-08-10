<script lang="ts">
	// Plate IV — Goodhart, measured rather than asserted.
	//
	// Optimize hard against the judge with no leash, snapshotting the policy at
	// fixed distances from the reference. The proxy score is free to compute and
	// climbs forever. The number anyone actually cares about — whether the
	// reader still likes the result — costs a click each, and turns over
	// somewhere in the middle. Both curves are drawn on the same axis, which is
	// the whole plate.
	import { Play, Plus, RotateCcw } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { progress } from '$lib/data/progress.svelte';
	import {
		ascendStep,
		clonePolicy,
		createPolicy,
		expectedScore,
		geneFromU,
		klToRef,
		samplePolicyGene,
		sampleRefGene,
		type Gene,
		type Policy
	} from '$lib/optim-rl/preference';
	import { mulberry32 } from '$lib/optim-rl/rng';
	import { inkLoad } from './rosette';
	import Rosette from './Rosette.svelte';
	import { taste } from './taste-context.svelte';

	/** Where to photograph the run, in nats travelled from the reference. */
	const STOPS = [0, 0.5, 1, 2, 4, 8, 16];
	/**
	 * Head-to-heads per checkpoint. Two is far too few to pin a win rate down,
	 * which is the honest situation and so is drawn rather than hidden: every
	 * gold point carries a standard-error bar, and the reader can buy a
	 * narrower one a round at a time. That exchange — clicks for certainty — is
	 * the actual reason nobody measures the gold reward often enough.
	 */
	let rounds = $state(2);

	interface Shot {
		kl: number;
		proxy: number;
		gene: Gene;
		policy: Policy;
		ink: number;
		wins: number;
		asked: number;
	}

	let shots = $state<Shot[]>([]);
	let running = $state(false);
	/** Which checkpoint the head-to-head is currently asking about. */
	let quiz = $state<{ shot: number; left: Gene; right: Gene; leftIsPolicy: boolean } | null>(null);

	const started = $derived(shots.length > 0);
	const judged = $derived(shots.reduce((a, s) => a + s.asked, 0));
	const total = $derived((shots.length - 1) * rounds);

	function optimize() {
		if (!taste.ready) return;
		running = true;
		const rand = mulberry32(90210);
		const policy = createPolicy();
		const out: Shot[] = [];
		const snap = (kl: number) => {
			out.push({
				kl,
				proxy: expectedScore(taste.judge, policy, mulberry32(4242), 256),
				gene: geneFromU(Float64Array.from(policy.mu)),
				policy: clonePolicy(policy),
				ink: inkLoad(geneFromU(Float64Array.from(policy.mu))),
				wins: 0,
				asked: 0
			});
		};
		snap(0);
		let next = 1;
		for (let step = 0; step < 6000 && next < STOPS.length; step++) {
			ascendStep(taste.judge, policy, rand, { beta: 0 });
			if (klToRef(policy) >= STOPS[next]) {
				snap(klToRef(policy));
				next++;
			}
		}
		shots = out;
		running = false;
		progress.reach('taste:goodhart');
		ask(1);
	}

	/** Pose one head-to-head: a draw from this checkpoint against the reference. */
	function ask(i: number) {
		if (i <= 0 || i >= shots.length) {
			quiz = null;
			return;
		}
		const seed = 700 + i * 97 + shots[i].asked * 13;
		const rand = mulberry32(seed);
		const a = samplePolicyGene(shots[i].policy, rand);
		const b = sampleRefGene(rand);
		// Sides are shuffled, because a plate that always put the optimized one
		// on the left would measure which way the reader looks first.
		const leftIsPolicy = rand() < 0.5;
		quiz = { shot: i, left: leftIsPolicy ? a : b, right: leftIsPolicy ? b : a, leftIsPolicy };
	}

	function answer(side: 'left' | 'right') {
		if (!quiz) return;
		const chosePolicy = (side === 'left') === quiz.leftIsPolicy;
		const i = quiz.shot;
		shots[i].asked += 1;
		if (chosePolicy) shots[i].wins += 1;
		const next = nextUnfinished(i);
		if (next === null) quiz = null;
		else ask(next);
	}

	/** The first checkpoint still owed a round, starting the sweep at `from`. */
	function nextUnfinished(from: number): number | null {
		for (let i = from; i < shots.length; i++) if (shots[i].asked < rounds) return i;
		for (let i = 1; i < from; i++) if (shots[i].asked < rounds) return i;
		return null;
	}

	/** Buy a narrower error bar: one more head-to-head at every checkpoint. */
	function anotherRound() {
		rounds += 1;
		const next = nextUnfinished(1);
		if (next !== null) ask(next);
	}

	function reset() {
		shots = [];
		quiz = null;
		rounds = 2;
	}

	// ── the chart ──
	const W = 620;
	const H = 200;
	// Two curves in two different units share this frame, so each gets its own
	// labelled scale — vermilion on the left in win rate, ultramarine on the
	// right in standardized reward. Letting them share one axis would make the
	// blue line's *height* look meaningful when only its shape is.
	const PAD = { l: 42, r: 52, t: 14, b: 26 };
	// KL is compressed, because everything interesting happens in the first few
	// nats and the tail is a long flat lie about how much room there is.
	const kx = (kl: number) =>
		PAD.l + (Math.log1p(kl) / Math.log1p(STOPS[STOPS.length - 1])) * (W - PAD.l - PAD.r);

	const proxyRange = $derived.by(() => {
		if (!shots.length) return [0, 1] as const;
		const vs = shots.map((s) => s.proxy);
		const lo = Math.min(...vs, 0);
		const hi = Math.max(...vs);
		return [lo, hi + (hi - lo) * 0.12 || 1] as const;
	});
	const py = $derived((v: number) => {
		const [lo, hi] = proxyRange;
		const t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
		return PAD.t + (1 - t) * (H - PAD.t - PAD.b);
	});
	/** One decimal, with an explicit sign and no negative zero. */
	const signed = (v: number) => {
		const r = Math.abs(v) < 0.05 ? 0 : v;
		return `${r >= 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}`;
	};
	/** Gold is a win rate, so its axis is fixed at 0…1 and needs no scaling. */
	const gy = (v: number) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);

	const proxyPath = $derived(
		shots
			.map((s, i) => `${i ? 'L' : 'M'} ${kx(s.kl).toFixed(1)} ${py(s.proxy).toFixed(1)}`)
			.join(' ')
	);
	/**
	 * A win rate from a handful of rounds, with the honest bar around it —
	 * Wilson's interval at one standard error, which behaves at 0 and 1 where
	 * the textbook formula collapses to a point and lies about it.
	 */
	function wilson(wins: number, n: number): { rate: number; lo: number; hi: number } {
		const p = wins / n;
		const z = 1;
		const d = 1 + (z * z) / n;
		const c = (p + (z * z) / (2 * n)) / d;
		const h = (z / d) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
		return { rate: p, lo: Math.max(0, c - h), hi: Math.min(1, c + h) };
	}

	const goldPts = $derived(
		shots.filter((s) => s.asked > 0).map((s) => ({ ...s, ...wilson(s.wins, s.asked) }))
	);
	const goldPath = $derived(
		goldPts
			.map((s, i) => `${i ? 'L' : 'M'} ${kx(s.kl).toFixed(1)} ${gy(s.rate).toFixed(1)}`)
			.join(' ')
	);
	/**
	 * Where the reader's own verdicts peaked — and only when there is honestly a
	 * peak to report. A maximum that merely *ties* the last checkpoint is not a
	 * turnover, and claiming one would be the plate committing the exact sin it
	 * spends its caption warning about.
	 */
	const peak = $derived.by(() => {
		if (goldPts.length < 3) return null;
		let best = goldPts[0];
		for (const s of goldPts) if (s.rate > best.rate) best = s;
		const last = goldPts[goldPts.length - 1];
		return best.rate > 0.5 && best.rate > last.rate + 1e-6 ? best : null;
	});
</script>

<Plate
	id="goodhart"
	title="What the judge wants, and what you want"
	live
	caption="The optimizer is given your judge and no other instruction. It travels, and every ornament above is the same policy photographed further from where it started — the distance measured in nats of KL, the only honest ruler for “how hard did we optimize”. The blue curve is the judge's own opinion, which costs nothing to compute and never stops rising. The vermilion curve is yours, and each of its points costs you a click. Sides are shuffled every round so the plate is measuring your taste and not which way you look first."
>
	{#snippet status()}
		{#if started}
			<span>{judged}/{total} judged</span>
		{:else if !taste.ready}
			<span class="text-ink-3">judge some pairs first</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		{#if started}
			<Btn onclick={reset}><RotateCcw size={12} /> Reset</Btn>
		{:else}
			<Btn kind="primary" disabled={!taste.ready || running} onclick={optimize}>
				<Play size={12} /> Optimize
			</Btn>
		{/if}
	{/snippet}

	<div class="gh px-4">
		{#if !started}
			<p class="idle font-serif">
				{#if taste.ready}
					Your judge is fitted. Turn an optimizer loose on it.
				{:else}
					This plate needs a judge. Go and disagree with a few ornaments first.
				{/if}
			</p>
		{:else}
			<ol
				class="strip"
				aria-label="The policy, photographed at increasing distance from the reference"
			>
				{#each shots as s, i (i)}
					<li class:peaked={peak === s}>
						<Rosette
							gene={s.gene}
							size={82}
							label={`Typical ornament after ${s.kl.toFixed(1)} nats of optimization`}
						/>
						<span class="num kl">{s.kl === 0 ? 'start' : `${s.kl.toFixed(1)}`}</span>
						<span class="num r">r {s.proxy >= 0 ? '+' : ''}{s.proxy.toFixed(2)}</span>
					</li>
				{/each}
			</ol>
			<p class="axis num">nats travelled from the reference →</p>

			<svg
				class="chart"
				viewBox="0 0 {W} {H}"
				role="img"
				aria-label="Proxy score and your own verdicts against distance travelled"
			>
				<line
					x1={PAD.l}
					y1={gy(0.5)}
					x2={W - PAD.r}
					y2={gy(0.5)}
					class="rule"
					stroke-dasharray="3 4"
				/>
				<!-- left: the reader's win rate, in the colour of the curve it belongs to -->
				<text x={PAD.l - 6} y={gy(0.5) + 3} class="tick gold-tick" text-anchor="end">50%</text>
				<text x={PAD.l - 6} y={PAD.t + 4} class="tick gold-tick" text-anchor="end">100%</text>
				<text x={PAD.l - 6} y={H - PAD.b} class="tick gold-tick" text-anchor="end">0%</text>
				<!-- right: standardized reward, on its own scale -->
				<text x={W - PAD.r + 6} y={PAD.t + 4} class="tick proxy-tick" text-anchor="start">
					r {signed(proxyRange[1])}
				</text>
				<text x={W - PAD.r + 6} y={H - PAD.b} class="tick proxy-tick" text-anchor="start">
					r {signed(proxyRange[0])}
				</text>
				{#each shots as s, i (i)}
					<line x1={kx(s.kl)} y1={PAD.t} x2={kx(s.kl)} y2={H - PAD.b} class="grid" />
					<text x={kx(s.kl)} y={H - PAD.b + 13} class="tick" text-anchor="middle">
						{s.kl === 0 ? '0' : s.kl.toFixed(s.kl < 1 ? 1 : 0)}
					</text>
				{/each}
				{#if peak}
					<line x1={kx(peak.kl)} y1={PAD.t} x2={kx(peak.kl)} y2={H - PAD.b} class="peak" />
				{/if}
				<path d={proxyPath} class="proxy" fill="none" />
				{#each shots as s, i (i)}
					<circle cx={kx(s.kl)} cy={py(s.proxy)} r="3" class="proxy-dot" />
				{/each}
				{#if goldPts.length > 1}
					<path d={goldPath} class="gold" fill="none" />
				{/if}
				{#each goldPts as s, i (i)}
					<line x1={kx(s.kl)} y1={gy(s.lo)} x2={kx(s.kl)} y2={gy(s.hi)} class="gold-bar" />
					<circle cx={kx(s.kl)} cy={gy(s.rate)} r="3.5" class="gold-dot" />
				{/each}
			</svg>

			<div class="legend num">
				<span
					><i class="sw proxy-sw"></i>what your judge says · right axis · free, and always rising</span
				>
				<span><i class="sw gold-sw"></i>what you say · left axis · one click each</span>
				{#if peak}<span class="peaknote">your verdicts peaked at {peak.kl.toFixed(1)} nats</span
					>{/if}
			</div>

			{#if quiz}
				<div class="quiz">
					<p class="ask font-serif">
						Round {shots[quiz.shot].asked + 1} of {rounds} at
						<strong>{shots[quiz.shot].kl.toFixed(1)} nats</strong> — which would you set on a title page?
					</p>
					<div class="duel">
						<button
							class="card"
							onclick={() => answer('left')}
							aria-label="Choose the left ornament"
						>
							<Rosette gene={quiz.left} size={150} />
						</button>
						<button
							class="card"
							onclick={() => answer('right')}
							aria-label="Choose the right ornament"
						>
							<Rosette gene={quiz.right} size={150} />
						</button>
					</div>
				</div>
			{:else if judged > 0}
				<div class="done">
					<p class="font-serif">
						There it is, drawn from {judged} of your judgments. Look at the vermilion bars before the
						line: each is one standard error, and at {rounds} rounds a checkpoint they are wide enough
						to drive a cart through. That is not a flaw in the plate. The blue curve cost nothing and
						is smooth; every vermilion point cost you a click and is still uncertain, and no dashboard
						anywhere marks where the two part company. You can buy a narrower bar, one round at a time
						— which is exactly the trade nobody makes often enough.
					</p>
					<Btn onclick={anotherRound}>
						<Plus size={12} /> another round at every checkpoint
					</Btn>
				</div>
			{/if}
		{/if}
	</div>
</Plate>

<style>
	.gh {
		padding-bottom: 0.5rem;
	}
	.idle {
		text-align: center;
		font-style: italic;
		color: var(--ink-3);
		padding: 2.5rem 0;
		margin: 0;
	}
	.strip {
		display: grid;
		grid-template-columns: repeat(7, minmax(0, 1fr));
		gap: 0.4rem;
		list-style: none;
		margin: 0.2rem 0 0.4rem;
		padding: 0;
	}
	.strip li {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.15rem;
		padding: 0.45rem 0.2rem;
		border-radius: var(--r-2);
	}
	.strip li.peaked {
		background: color-mix(in srgb, var(--warm) 9%, transparent);
	}
	.kl {
		font-size: 11px;
		color: var(--ink-2);
	}
	.r {
		font-size: 10px;
		color: var(--ink-3);
	}
	.axis {
		text-align: center;
		font-size: 10px;
		color: var(--ink-3);
		margin: 0 0 0.9rem;
	}
	.chart {
		display: block;
		width: 100%;
		height: auto;
	}
	.rule,
	.grid {
		stroke: var(--line);
		stroke-width: 1;
	}
	.grid {
		stroke: var(--line-soft);
	}
	.peak {
		stroke: var(--warm);
		stroke-width: 1;
		opacity: 0.4;
	}
	.tick {
		font-family: var(--font-mono);
		font-size: 9px;
		fill: var(--ink-3);
	}
	.gold-tick {
		fill: color-mix(in srgb, var(--warm) 70%, var(--ink-3));
	}
	.proxy-tick {
		fill: color-mix(in srgb, var(--accent) 70%, var(--ink-3));
	}
	.proxy {
		stroke: var(--accent);
		stroke-width: 1.8;
	}
	.proxy-dot {
		fill: var(--accent);
	}
	.gold {
		stroke: var(--warm);
		stroke-width: 1.8;
	}
	.gold-dot {
		fill: var(--warm);
	}
	.gold-bar {
		stroke: var(--warm);
		stroke-width: 1.4;
		opacity: 0.45;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem 1.1rem;
		font-size: 10.5px;
		color: var(--ink-3);
		margin: 0.15rem 0 0;
	}
	.sw {
		display: inline-block;
		width: 14px;
		height: 2px;
		vertical-align: middle;
		margin-right: 0.4rem;
	}
	.proxy-sw {
		background: var(--accent);
	}
	.gold-sw {
		background: var(--warm);
	}
	.peaknote {
		color: var(--warm);
	}
	.quiz {
		margin-top: 1.1rem;
		padding-top: 0.9rem;
		border-top: 1px solid var(--line-soft);
	}
	.ask {
		text-align: center;
		font-size: 15.5px;
		font-style: italic;
		color: var(--ink-2);
		margin: 0 0 0.85rem;
	}
	.ask strong {
		font-style: normal;
		font-family: var(--font-mono);
		font-size: 14px;
		color: var(--ink);
	}
	.duel {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: clamp(0.6rem, 4vw, 2.5rem);
		max-width: 28rem;
		margin: 0 auto;
	}
	.card {
		display: grid;
		place-items: center;
		padding: 0.85rem 0.4rem;
		border: 1px solid var(--line-soft);
		border-radius: var(--r-3);
		background: var(--surface);
		transition:
			border-color 120ms,
			transform 120ms;
	}
	.card:hover {
		border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
		transform: translateY(-2px);
	}
	.card:focus-visible {
		outline: none;
		box-shadow: var(--focus-ring);
	}
	.done {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.75rem;
		margin: 1.1rem 0 0;
		padding-top: 0.9rem;
		border-top: 1px solid var(--line-soft);
	}
	.done p {
		margin: 0;
		font-size: 15px;
		line-height: 1.6;
		color: var(--ink-2);
		font-variation-settings: 'opsz' 16;
	}
	@media (max-width: 720px) {
		.strip {
			grid-template-columns: repeat(4, minmax(0, 1fr));
		}
	}
</style>
