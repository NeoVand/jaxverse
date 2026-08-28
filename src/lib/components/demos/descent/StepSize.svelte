<script lang="ts">
	import { Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';

	// Four step sizes down the same landscape at once, because the thing worth
	// seeing is the DIFFERENCE and a dial can only ever show one at a time.
	// The plate used to be one ball and a slider, which asked the reader to
	// hold the last setting in their head and compare it with this one.
	//
	// On θ² one update is θ ← (1 − 2η)θ, so a single number decides everything:
	//
	//   η = 0.15 → ×0.70   same side, closing slowly
	//   η = 0.45 → ×0.10   same side, almost there in one
	//   η = 0.80 → ×−0.60  crosses the bottom each step, still closing
	//   η = 1.05 → ×−1.10  crosses it and lands further out than it started
	//
	// 0.80 and not 0.85, which was the first choice and wrong: |1 − 2η| is 0.70
	// for both 0.15 and 0.85, so the two ran at exactly the same rate and their
	// two lines on the log chart lay on top of each other. True, and a fine
	// remark, but a chart drawing three lines where the legend promises four
	// reads as a fault before it reads as a coincidence.
	//
	// which is the whole taxonomy: sign says whether it overshoots, magnitude
	// says whether it survives. The four are chosen to sit one in each cell of
	// that table, and the loss chart underneath is where it becomes obvious —
	// on a log axis a constant contraction is a straight line, so each η is a
	// slope, and the diverging one is the slope that points up.
	interface Curve {
		id: 'quad' | 'well' | 'abs';
		label: string;
		f(th: number): number;
		df(th: number): number;
		/** The floor, so the chart can plot height ABOVE the minimum and take
		 *  a logarithm of it. The double well's floor is not zero. */
		fmin: number;
		xMax: number;
		yMax: number;
		theta0: number;
	}

	const CURVES: Curve[] = [
		{
			id: 'quad',
			label: 'θ²',
			f: (th) => th * th,
			df: (th) => 2 * th,
			fmin: 0,
			xMax: 1.6,
			yMax: 2.56,
			theta0: -0.9
		},
		{
			id: 'well',
			label: '0.3θ⁴ − θ² + 1',
			f: (th) => 0.3 * th ** 4 - th * th + 1,
			df: (th) => 1.2 * th ** 3 - 2 * th,
			// 1.2θ³ = 2θ puts the two minima at θ² = 5/3, and the floor there is
			// 1/6 — not zero, which is why the chart plots height ABOVE it
			fmin: 0.3 * (5 / 3) ** 2 - 5 / 3 + 1,
			xMax: 2.2,
			yMax: 3.2,
			// −1.70 and not −2.05. From −2.05 the slope is steep enough that both
			// bold steps are thrown clean out of the frame on their second move,
			// so the plate's whole point on this landscape — that a big step can
			// cross into the OTHER valley — never happened. From here it does:
			// the two small steps settle in the valley they started in, 0.80
			// crosses to the far one, and only 1.05 leaves.
			theta0: -1.7
		},
		{
			id: 'abs',
			label: '|θ|',
			f: (th) => Math.abs(th),
			df: (th) => Math.sign(th),
			fmin: 0,
			xMax: 1.6,
			yMax: 1.6,
			// −1.31 and not −1.35, which divided exactly by both 0.15 and 0.45 —
			// so those two runners landed precisely on the kink, where the
			// subgradient is zero and they stopped dead. Two of the four then
			// reached a floor the caption said none of them could reach, on the
			// one landscape whose entire lesson is that none of them can.
			theta0: -1.31
		}
	];

	/** One runner per regime, cool to warm as the step grows — the ordering is
	 *  the reading, so the legend needs no explaining. */
	const RUNNERS = [
		{ eta: 0.15, tone: 'var(--cat-6)' },
		{ eta: 0.45, tone: 'var(--accent)' },
		{ eta: 0.8, tone: 'var(--cat-1)' },
		{ eta: 1.05, tone: 'var(--warm)' }
	] as const;

	const MAX_STEPS = 30;
	const TRAIL = 2; // hops kept behind each ball — motion, not a cobweb

	// plot geometry, viewBox units
	const VW = 640;
	const VH = 240;
	const PAD_L = 30;
	const PAD_R = 14;
	const PAD_T = 14;
	const PAD_B = 24;

	// the loss chart below it
	const LH = 132;
	const LPAD_T = 12;
	const LPAD_B = 22;
	const FLOOR = 1e-6; // the bottom of the log axis, and the clamp for zero

	let curveId = $state<Curve['id']>('quad');
	let auto = $state(false);
	let k = $state(0);
	let paths = $state<number[][]>(RUNNERS.map(() => [CURVES[0].theta0]));

	const curve = $derived(CURVES.find((c) => c.id === curveId) ?? CURVES[0]);
	const limit = $derived(curve.xMax - 0.05); // |θ| past the frame = diverged
	const gone = $derived(paths.map((p) => Math.abs(p[p.length - 1]) > limit));
	const done = $derived(k >= MAX_STEPS || gone.every(Boolean));

	const px = $derived(
		(th: number) => PAD_L + ((th + curve.xMax) / (2 * curve.xMax)) * (VW - PAD_L - PAD_R)
	);
	const py = $derived((l: number) => VH - PAD_B - (l / curve.yMax) * (VH - PAD_T - PAD_B));

	const curvePath = $derived.by(() => {
		let d = '';
		for (let i = 0; i <= 96; i++) {
			const th = -curve.xMax + (i / 96) * 2 * curve.xMax;
			d += `${i === 0 ? 'M' : 'L'}${px(th).toFixed(1)} ${py(curve.f(th)).toFixed(1)}`;
		}
		return d;
	});

	const xTicks = $derived.by(() => {
		const out: number[] = [];
		for (let v = -Math.floor(curve.xMax); v <= Math.floor(curve.xMax); v++) out.push(v);
		return out;
	});

	// ── the loss chart: height above the floor, on a log axis ──
	const hi = $derived(Math.max(curve.f(curve.theta0) - curve.fmin, 1e-3));
	const lx = $derived((step: number) => PAD_L + (step / MAX_STEPS) * (VW - PAD_L - PAD_R));
	const ly = $derived((v: number) => {
		const t =
			(Math.log10(Math.max(v, FLOOR)) - Math.log10(FLOOR)) / (Math.log10(hi) - Math.log10(FLOOR));
		// clamped at the FLOOR and not at the ceiling, on purpose. A run that is
		// climbing has to be seen to leave: pinning it to the top edge draws a
		// diverging run as a flat line, which is the shape of a run that has
		// settled — the exact opposite reading. The svg clips it instead, so it
		// goes up and off, which is what it is doing.
		return LH - LPAD_B - Math.max(0, t) * (LH - LPAD_T - LPAD_B);
	});
	const decades = $derived.by(() => {
		const out: number[] = [];
		for (let e = Math.ceil(Math.log10(FLOOR)); e <= Math.floor(Math.log10(hi)); e += 2)
			out.push(Math.pow(10, e));
		return out;
	});

	/** Arrowhead at the far end of a hop, so a chord across the bowl reads as a
	 *  step that was taken and not as a line that wandered in. */
	function arrow(x1: number, y1: number, x2: number, y2: number): string {
		const dx = x2 - x1;
		const dy = y2 - y1;
		const len = Math.hypot(dx, dy) || 1;
		const ux = dx / len;
		const uy = dy / len;
		const s = 4.5;
		const bx = x2 - ux * s;
		const by = y2 - uy * s;
		return `${x2},${y2} ${bx - uy * s * 0.55},${by + ux * s * 0.55} ${bx + uy * s * 0.55},${by - ux * s * 0.55}`;
	}

	/** Each runner's loss trace, clipped where it left the frame. */
	const traces = $derived(
		paths.map((p) => {
			let d = '';
			for (let i = 0; i < p.length; i++) {
				if (Math.abs(p[i]) > limit) break;
				d += `${i === 0 ? 'M' : 'L'}${lx(i).toFixed(1)} ${ly(curve.f(p[i]) - curve.fmin).toFixed(1)}`;
			}
			return d;
		})
	);

	function stepOnce() {
		if (k >= MAX_STEPS) return;
		paths = paths.map((p, i) => {
			const cur = p[p.length - 1];
			if (Math.abs(cur) > limit) return p;
			return [...p, cur - RUNNERS[i].eta * curve.df(cur)];
		});
		k += 1;
	}

	function reset() {
		const c = CURVES.find((x) => x.id === curveId) ?? CURVES[0];
		paths = RUNNERS.map(() => [c.theta0]);
		k = 0;
	}

	function selectCurve(id: Curve['id']) {
		if (id === curveId) return;
		curveId = id;
		auto = false;
		const c = CURVES.find((x) => x.id === id) ?? CURVES[0];
		paths = RUNNERS.map(() => [c.theta0]);
		k = 0;
	}

	$effect(() => {
		if (!auto || done) return;
		const id = setInterval(stepOnce, 380);
		return () => clearInterval(id);
	});
</script>

<Plate
	id="stepsize"
	live
	title="Too big, too small"
	caption="Four step sizes let down the same slope together, from the same start, so the difference is the picture rather than something to remember between two settings of a dial. On θ² a step is θ ← (1 − 2η)θ, and that one number decides everything: at η = 0.15 it multiplies θ by 0.70 each time and the ball closes in from one side; at 0.45, by 0.10, and it is essentially there in one move; at 0.80, by −0.60, so it crosses the bottom every step and still closes; at 1.05, by −1.10, and each crossing lands further out than the last. The chart underneath is the same four runs as height above the floor of the valley, on a log axis, where a constant multiplier is a straight line — so each step size is a slope, and the one that diverges is the slope pointing the wrong way. Switch landscapes and the same four steps sort themselves differently. On the double well the two small steps settle into whichever valley they were dropped beside, 0.80 is bold enough to cross into the other one, and only 1.05 leaves the frame — a bigger step is not simply worse, it is a different search. On |θ| the slope never softens as you approach the bottom, so the step never shrinks either: all four stride back and forth across the kink forever, and not one of the four lines ever reaches the floor."
>
	{#snippet status()}
		<span>step {k} of {MAX_STEPS}</span>
	{/snippet}

	{#snippet actions()}
		<Btn kind={auto ? 'ghost' : 'primary'} onclick={() => (auto = !auto)} disabled={done}>
			{#if auto}
				<Pause size={13} aria-hidden="true" /> Auto
			{:else}
				<Play size={13} aria-hidden="true" /> Auto
			{/if}
		</Btn>
		<Btn onclick={stepOnce} disabled={done}>
			<StepForward size={13} aria-hidden="true" /> Step
		</Btn>
		<Btn onclick={reset}><RotateCcw size={13} aria-hidden="true" /> Reset</Btn>
	{/snippet}

	<div class="p-4 sm:p-5">
		<div class="mx-auto max-w-[640px]">
			<div class="mb-3 flex flex-wrap items-center gap-1.5" role="group" aria-label="Loss curve">
				{#each CURVES as c (c.id)}
					<button
						class="chip"
						class:chip-on={curveId === c.id}
						aria-pressed={curveId === c.id}
						onclick={() => selectCurve(c.id)}
					>
						{c.label}
					</button>
				{/each}
			</div>

			<svg
				viewBox="0 0 {VW} {VH}"
				class="block h-auto w-full"
				role="img"
				aria-label="A one-dimensional loss curve with four balls on it, one per step size, all released from the same point. The small steps close in from one side; the largest crosses the bottom each step and climbs away from it."
			>
				<line
					x1={PAD_L}
					y1={py(0)}
					x2={VW - PAD_R}
					y2={py(0)}
					stroke="var(--line)"
					stroke-width="1"
				/>
				<line
					x1={px(0)}
					y1={PAD_T}
					x2={px(0)}
					y2={py(0)}
					stroke="var(--line-soft)"
					stroke-width="1"
					stroke-dasharray="2 4"
				/>
				{#each xTicks as th (th)}
					<text x={px(th)} y={VH - PAD_B + 14} text-anchor="middle" class="tick">{th}</text>
				{/each}
				<text x={VW - PAD_R} y={VH - PAD_B + 14} text-anchor="end" class="tick">θ</text>

				<path d={curvePath} fill="none" stroke="var(--ink)" stroke-width="1.5" />

				<!-- each runner: a short trail for motion, then the ball. Not a
				     cobweb — four cobwebs on one landscape is a thicket, and the
				     rates belong in the chart below where they can be read. -->
				{#each RUNNERS as r, i (r.eta)}
					{@const p = paths[i]}
					{@const inside = p.filter((th) => Math.abs(th) <= limit)}
					{@const tail = inside.slice(-(TRAIL + 1))}
					{#if !gone[i]}
						{#each tail.slice(0, -1) as th, j (j)}
							<g opacity={0.3 + 0.3 * j}>
								<line
									x1={px(th)}
									y1={py(curve.f(th))}
									x2={px(tail[j + 1])}
									y2={py(curve.f(tail[j + 1]))}
									stroke={r.tone}
									stroke-width="1.3"
								/>
								<polygon
									points={arrow(px(th), py(curve.f(th)), px(tail[j + 1]), py(curve.f(tail[j + 1])))}
									fill={r.tone}
								/>
							</g>
						{/each}
						<circle
							cx={px(p[p.length - 1])}
							cy={py(curve.f(p[p.length - 1]))}
							r="5"
							fill={r.tone}
							stroke="var(--surface)"
							stroke-width="1.5"
						/>
					{:else}
						<!-- gone off the frame: its last step is left as one arrow at the
						     edge, and nothing else, so no chord hangs in the air with no
						     ball on the end of it -->
						{#if inside.length}
							<circle
								cx={px(inside[inside.length - 1])}
								cy={py(curve.f(inside[inside.length - 1]))}
								r="3"
								fill="none"
								stroke={r.tone}
								stroke-width="1.4"
								opacity="0.55"
							/>
						{/if}
					{/if}
				{/each}
			</svg>

			<svg
				viewBox="0 0 {VW} {LH}"
				class="mt-1 block h-auto w-full"
				role="img"
				aria-label="Height above the bottom of the valley against step number, on a logarithmic axis, one line per step size. A constant contraction draws a straight line, so a smaller step size is a shallower slope down, and the largest step size draws a line that rises instead."
			>
				{#each decades as d (d)}
					<line
						x1={PAD_L}
						y1={ly(d)}
						x2={VW - PAD_R}
						y2={ly(d)}
						stroke="var(--line-soft)"
						stroke-width="1"
					/>
					<text x={PAD_L - 6} y={ly(d) + 3} text-anchor="end" class="tick"
						>{d >= 1 ? d : `1e${Math.round(Math.log10(d))}`}</text
					>
				{/each}
				<line
					x1={PAD_L}
					y1={ly(FLOOR)}
					x2={VW - PAD_R}
					y2={ly(FLOOR)}
					stroke="var(--line)"
					stroke-width="1"
				/>
				<text x={PAD_L} y={LH - 6} class="tick">step</text>
				<text x={VW - PAD_R} y={LH - 6} text-anchor="end" class="tick">{MAX_STEPS}</text>

				{#each RUNNERS as r, i (r.eta)}
					<path d={traces[i]} fill="none" stroke={r.tone} stroke-width="1.7" opacity="0.95" />
				{/each}
			</svg>

			<div class="mt-2.5 flex flex-wrap justify-center gap-x-5 gap-y-1.5">
				{#each RUNNERS as r, i (r.eta)}
					<span class="num inline-flex items-baseline gap-1.5 text-[11px] text-ink-3">
						<span class="dot" style="background: {r.tone};"></span>
						η = {r.eta.toFixed(2)}
						{#if gone[i]}<span style="color: var(--bad);">diverged</span>{/if}
					</span>
				{/each}
			</div>
		</div>
	</div>
</Plate>

<style>
	.tick {
		font-family: var(--font-mono);
		font-size: 10px;
		fill: var(--ink-3);
	}
	.dot {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 999px;
		align-self: center;
	}
</style>
