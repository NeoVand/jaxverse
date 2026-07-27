<script lang="ts">
	// Plate I — the double pendulum swing-up on a sliding hinge. The links
	// start hanging straight down; a linear softmax policy over hand-crafted
	// gauges learns, via REINFORCE, to pump energy into the stack and catch
	// it upright. The reader's job is to interfere: clicking or dragging
	// pulls the hinge toward the pointer — an outside force the physics
	// feels but the policy never sees. Nothing ever resets on its own; the
	// policy is always driving. Training runs extra headless episodes behind
	// the visible one, so hours of practice pass in minutes.
	import { onDestroy } from 'svelte';
	import { ArrowDownToLine, Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import {
		actionForce,
		createDpoleBaseline,
		createDpoleTheta,
		DECISIONS,
		deliveryBonus,
		DPOLE_ACTIONS,
		DpoleCurriculum,
		dpoleFeatures,
		dpolePolicy,
		dpoleReinforceUpdate,
		dpoleReward,
		hasDroppedOut,
		isDelivery,
		isUpright,
		L1,
		L2,
		N_FEATURES,
		physicsStep,
		resetDpole,
		TH_LIMIT,
		X_LIMIT,
		type DpoleEpisode,
		type DpoleStart,
		type DpoleState
	} from '$lib/optim-rl/dpole';
	import { mulberry32 } from '$lib/optim-rl/rng';

	const SEED = 512;
	const KEEP = 300; // sparkline window
	const SHOVE_GAIN = 10; // newtons per metre of pointer–hinge separation
	const SHOVE_MAX = 24; // enough to out-muscle the policy's 20 N, barely
	const SHOVE_HOLD_MS = 180; // a quick click still lands a real push
	const BURST_MS = 9; // per-frame budget for headless training episodes —
	const BURST_CAP = 45; // the swing-up needs tens of thousands of them

	// ── the learner: non-reactive, mutated in place ──
	let theta = createDpoleTheta();
	let baseline = createDpoleBaseline();
	let curriculum = new DpoleCurriculum();
	let rand = mulberry32(SEED);
	const featBuf = new Float64Array(N_FEATURES);
	const probBuf = new Float64Array(DPOLE_ACTIONS);

	// The live simulation the canvas shows — begins hanging, like every
	// honest episode. Its recorded segments switch kind at the hand-off
	// height, mirroring how headless episodes are cut: swing segments end
	// at delivery (and bank the delivery bonus), catch segments end when
	// the tip drops back out.
	let sim: DpoleState = resetDpole(() => 0.5);
	let liveKind: DpoleStart = 0;
	let liveFeats: number[] = [];
	let liveActions: number[] = [];
	let liveRewards: number[] = [];
	let dispForce = 0; // EMA of the policy's push, for a legible arrow
	let shoveX: number | null = null; // pointer position in world metres
	let shoveNow = 0; // last applied shove, for the arrow
	let shoveUntil = 0; // taps keep pulling for a beat
	let pointerHeld = false;
	let trail: number[] = []; // recent tip positions (world coords, flat x,y)

	// ── reactive state ──
	let ready = $state(false);
	let training = $state(false);
	let lr = $state(0.15);
	let episodes = $state(0);
	let caughtLog = $state<number[]>([]); // ticks caught per brink drill
	let best = $state(0);
	let alphaPct = $state(25); // catch curriculum: % of the raw delivery replayed
	let upright = $state(0); // consecutive live ticks inside the cones
	let canvas: HTMLCanvasElement | undefined = $state();

	const reduced =
		typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

	const mean20 = $derived.by(() => {
		const r = caughtLog.slice(-20);
		return r.length ? r.reduce((a, b) => a + b, 0) / r.length : NaN;
	});
	const catches = $derived(Number.isFinite(mean20) && mean20 >= 50);

	// ── sparkline: ticks caught per episode, the 400-tick ceiling dashed ──
	const SW = 220;
	const SH = 64;
	const sy = (v: number) => SH - 3 - (v / DECISIONS) * (SH - 6);
	const sparkPath = $derived.by(() => {
		if (caughtLog.length < 2) return '';
		return caughtLog
			.map(
				(v, i) =>
					`${i === 0 ? 'M' : 'L'}${((i / (caughtLog.length - 1)) * SW).toFixed(1)} ${sy(v).toFixed(1)}`
			)
			.join(' ');
	});

	function boot() {
		if (ready) return;
		ready = true;
	}

	/** Reset the simulation only — back to hanging, weights untouched. */
	function rest() {
		sim = resetDpole(rand);
		liveKind = 0;
		liveFeats = [];
		liveActions = [];
		liveRewards = [];
		upright = 0;
		trail = [];
		dispForce = 0;
		if (reduced) draw();
	}

	/** A live segment closed — while training, it teaches. Your shoves are
	 * part of the lesson: a disturbance the policy rides out is a window
	 * with high return, so recovering from you is what gets reinforced. */
	function closeSegment(nextKind: DpoleStart) {
		const steps = liveActions.length;
		if (training && steps > 0) {
			const ep: DpoleEpisode = {
				feats: Float64Array.from(liveFeats),
				actions: liveActions,
				rewards: Float64Array.from(liveRewards),
				steps,
				kind: liveKind,
				ret: liveRewards.reduce((a, b) => a + b, 0),
				caught: 0, // unused by the update; the live window has its own counter
				delivered: null
			};
			dpoleReinforceUpdate(theta, baseline, [ep], lr);
			episodes++;
		}
		liveKind = nextKind;
		liveFeats = [];
		liveActions = [];
		liveRewards = [];
	}

	function currentShove(now: number): number {
		if (shoveX === null || (!pointerHeld && now > shoveUntil)) {
			shoveX = pointerHeld ? shoveX : null;
			return 0;
		}
		return Math.max(-SHOVE_MAX, Math.min(SHOVE_MAX, SHOVE_GAIN * (shoveX - sim.x)));
	}

	/** One visible tick: the policy pushes, the reader may be shoving. */
	function stepLive(now: number) {
		dpoleFeatures(sim, featBuf);
		for (const v of featBuf) liveFeats.push(v);
		dpolePolicy(theta, featBuf, probBuf);
		let u = rand();
		let a = 0;
		for (; a < DPOLE_ACTIONS - 1; a++) {
			u -= probBuf[a];
			if (u <= 0) break;
		}
		liveActions.push(a);
		const push = actionForce(a);
		dispForce = dispForce * 0.75 + push * 0.25;
		shoveNow = currentShove(now);
		physicsStep(sim, push + shoveNow);
		liveRewards.push(dpoleReward(sim));
		upright = isUpright(sim) ? upright + 1 : 0;
		pushTrail();
		// cut the segment the same way headless episodes are cut — the
		// stage never blinks, only the ledger turns a page. A fly-through is
		// not a delivery: the segment (and the whisper reward) carries on
		// until the stack arrives slowly enough to catch.
		if (liveKind === 0 && isDelivery(sim)) {
			liveRewards[liveRewards.length - 1] += deliveryBonus(sim);
			curriculum.deliveries.push({ ...sim });
			closeSegment(2);
		} else if (liveKind === 2 && hasDroppedOut(sim)) {
			closeSegment(0);
		} else if (liveActions.length >= DECISIONS) {
			closeSegment(liveKind);
		}
	}

	function pushTrail() {
		const tipX = sim.x + Math.sin(sim.th1) * L1 + Math.sin(sim.th2) * L2;
		const tipY = Math.cos(sim.th1) * L1 + Math.cos(sim.th2) * L2;
		trail.push(tipX, tipY);
		if (trail.length > 96) trail.splice(0, trail.length - 96);
	}

	// Batch headless training through the curriculum: one $state
	// reassignment per burst, not per episode. Only the brink drills land
	// on the sparkline — "given a delivery, how long did it stay caught" is
	// the skill being built; the swing's own exam is the delivery counter.
	function trainBatch(runUntil: (done: number, t0: number) => boolean) {
		const t0 = performance.now();
		const fresh: number[] = [];
		let n = 0;
		do {
			const batch: DpoleEpisode[] = [];
			for (let i = 0; i < 8; i++) batch.push(curriculum.next(theta, rand));
			dpoleReinforceUpdate(theta, baseline, batch, lr);
			n += batch.length;
			for (const ep of batch) if (ep.kind === 2) fresh.push(ep.caught);
		} while (runUntil(n, t0));
		episodes += n;
		alphaPct = Math.round(curriculum.alpha * 100);
		if (fresh.length) {
			for (const c of fresh) if (c > best) best = c;
			caughtLog = [...caughtLog, ...fresh].slice(-KEEP);
		}
	}

	function trainBurst() {
		trainBatch((n) => n < 400);
		draw();
	}

	function resetTheta() {
		theta = createDpoleTheta();
		baseline = createDpoleBaseline();
		curriculum = new DpoleCurriculum();
		rand = mulberry32(SEED);
		episodes = 0;
		caughtLog = [];
		best = 0;
		alphaPct = 25;
		rest();
	}

	// ── painter ──
	let raf = 0;

	$effect(() => {
		if (!canvas || !ready) return;
		if (reduced) {
			draw();
			return;
		}
		const frame = (t: number) => {
			raf = requestAnimationFrame(frame);
			if (training) trainBatch((n, t0) => performance.now() - t0 < BURST_MS && n < BURST_CAP * 8);
			stepLive(t);
			draw();
		};
		raf = requestAnimationFrame(frame);
		return () => cancelAnimationFrame(raf);
	});

	onDestroy(() => {
		// also runs after server prerender — keep it browser-safe
		if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf);
	});

	function tokens(el: Element) {
		const s = getComputedStyle(el);
		const v = (name: string, fb: string) => s.getPropertyValue(name).trim() || fb;
		return {
			ink: v('--ink', '#1d1c18'),
			ink3: v('--ink-3', '#a3a094'),
			line: v('--line', '#e5e2d8'),
			lineSoft: v('--line-soft', '#efede4'),
			accent: v('--accent', '#2b45d8'),
			warm: v('--warm', '#d3541f'),
			amber: v('--cat-1', '#b98a1b'),
			good: v('--good', '#3a7d44'),
			bad: v('--bad', '#bb3a2b')
		};
	}

	function arrow(ctx: CanvasRenderingContext2D, x0: number, y: number, len: number, color: string) {
		if (Math.abs(len) < 3) return;
		const x1 = x0 + len;
		const s = Math.sign(len) * 5;
		ctx.strokeStyle = color;
		ctx.fillStyle = color;
		ctx.lineWidth = 1.6;
		ctx.beginPath();
		ctx.moveTo(x0, y);
		ctx.lineTo(x1, y);
		ctx.stroke();
		ctx.beginPath();
		ctx.moveTo(x1 + s, y);
		ctx.lineTo(x1 - s * 0.8, y - 3.4);
		ctx.lineTo(x1 - s * 0.8, y + 3.4);
		ctx.closePath();
		ctx.fill();
	}

	function draw() {
		if (!canvas) return;
		const dpr = Math.min(devicePixelRatio || 1, 2);
		const W = canvas.clientWidth;
		const H = canvas.clientHeight;
		if (!W || !H) return;
		if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
			canvas.width = W * dpr;
			canvas.height = H * dpr;
		}
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);
		const tk = tokens(canvas);

		// the rail crosses mid-stage: the pendulum reaches above it when
		// caught and dangles below it while swinging — both halves are home
		const railY = H * 0.5;
		const scale = Math.min((W - 48) / (2 * (X_LIMIT + 0.35)), railY - 34);
		const midX = W / 2;
		const px = (wx: number) => midX + wx * scale;
		const reach = (L1 + L2) * scale;
		const caughtNow = isUpright(sim);

		// rail with end stops, centre tick
		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(px(-X_LIMIT - 0.2), railY);
		ctx.lineTo(px(X_LIMIT + 0.2), railY);
		ctx.stroke();
		for (const e of [-X_LIMIT, X_LIMIT]) {
			ctx.beginPath();
			ctx.moveTo(px(e), railY - 6);
			ctx.lineTo(px(e), railY + 6);
			ctx.stroke();
		}
		ctx.strokeStyle = tk.lineSoft;
		ctx.beginPath();
		ctx.moveTo(px(0), railY + 2);
		ctx.lineTo(px(0), railY + 6);
		ctx.stroke();

		const hx = px(sim.x);

		// the ±18° target cones, faint — green while both links are inside
		ctx.strokeStyle = caughtNow ? tk.good : tk.lineSoft;
		ctx.setLineDash([3, 4]);
		for (const sgn of [-1, 1]) {
			ctx.beginPath();
			ctx.moveTo(hx, railY);
			ctx.lineTo(
				hx + Math.sin(sgn * TH_LIMIT) * (reach + 8),
				railY - Math.cos(sgn * TH_LIMIT) * (reach + 8)
			);
			ctx.stroke();
		}
		ctx.setLineDash([]);

		// the tip's recent past — loops while swinging, a knot once caught
		const n = trail.length / 2;
		for (let i = 0; i < n; i++) {
			const a = ((i + 1) / n) * 0.32;
			ctx.fillStyle = tk.amber;
			ctx.globalAlpha = a;
			ctx.beginPath();
			ctx.arc(px(trail[2 * i]), railY - trail[2 * i + 1] * scale, 1.6, 0, Math.PI * 2);
			ctx.fill();
		}
		ctx.globalAlpha = 1;

		// the two links: lower warm, upper amber, point masses at the tips
		const k1x = hx + Math.sin(sim.th1) * L1 * scale;
		const k1y = railY - Math.cos(sim.th1) * L1 * scale;
		const k2x = k1x + Math.sin(sim.th2) * L2 * scale;
		const k2y = k1y - Math.cos(sim.th2) * L2 * scale;
		ctx.lineCap = 'round';
		ctx.strokeStyle = tk.warm;
		ctx.lineWidth = 3.5;
		ctx.beginPath();
		ctx.moveTo(hx, railY);
		ctx.lineTo(k1x, k1y);
		ctx.stroke();
		ctx.strokeStyle = tk.amber;
		ctx.beginPath();
		ctx.moveTo(k1x, k1y);
		ctx.lineTo(k2x, k2y);
		ctx.stroke();
		ctx.fillStyle = tk.warm;
		ctx.beginPath();
		ctx.arc(k1x, k1y, 5, 0, Math.PI * 2);
		ctx.fill();
		ctx.fillStyle = tk.amber;
		ctx.beginPath();
		ctx.arc(k2x, k2y, 6.5, 0, Math.PI * 2);
		ctx.fill();

		// the hinge: a dot riding the rail — the only thing the policy moves
		ctx.fillStyle = tk.ink;
		ctx.beginPath();
		ctx.arc(hx, railY, 7, 0, Math.PI * 2);
		ctx.fill();

		// forces, left and right of the dot: the policy's push (ultramarine)
		// on the rail, the reader's shove (red) just beneath it
		const pf = dispForce * 2.4;
		arrow(ctx, hx + Math.sign(pf) * 11, railY, pf, tk.accent);
		if (Math.abs(shoveNow) > 0.5) {
			arrow(ctx, hx + Math.sign(shoveNow) * 11, railY + 13, shoveNow * 2.8, tk.bad);
			if (shoveX !== null) {
				ctx.fillStyle = tk.bad;
				ctx.globalAlpha = 0.55;
				ctx.beginPath();
				ctx.arc(px(shoveX), railY, 3, 0, Math.PI * 2);
				ctx.fill();
				ctx.globalAlpha = 1;
			}
		}
	}

	// ── pointer: the shove ──
	function worldX(ev: PointerEvent): number {
		if (!canvas) return 0;
		const r = canvas.getBoundingClientRect();
		const scale = Math.min((r.width - 48) / (2 * (X_LIMIT + 0.35)), r.height * 0.5 - 34);
		return (ev.clientX - r.left - r.width / 2) / scale;
	}
	function down(ev: PointerEvent) {
		if (reduced) return;
		pointerHeld = true;
		shoveX = worldX(ev);
		shoveUntil = performance.now() + SHOVE_HOLD_MS;
		(ev.target as Element).setPointerCapture(ev.pointerId);
	}
	function move(ev: PointerEvent) {
		if (pointerHeld) shoveX = worldX(ev);
	}
	function up() {
		pointerHeld = false;
	}
</script>

<Plate
	n={1}
	title="The double pendulum swing-up"
	caption="The links start hanging. Below the hand-off height the policy is paid once, at the moment it delivers the tip up top slowly enough to catch — a fast fly-through counts for nothing; above the hand-off, every calm tick inside the dashed 18° cones pays rent. Practice is a curriculum: swing drills start hanging or mid-tumble (so bleeding off a botched attempt is a practiced move), the rest rehearse the catch from replayed deliveries, eased toward vertical at first, raw as the success rate earns it (the drill difficulty gauge). Press Train, give it a minute or two, and then shove the hinge — the red arrow is you, the blue one is the policy. Knock it clean over and it will swing back up on its own."
>
	{#snippet status()}
		<span>
			ep {episodes} · caught {upright}
			· avg {Number.isFinite(mean20) ? mean20.toFixed(0) : '—'}/{DECISIONS}
			{#if catches}<span style="color: var(--good);">· it catches</span>{/if}
		</span>
	{/snippet}

	{#snippet actions()}
		{#if ready}
			{#if reduced}
				<Btn onclick={trainBurst}>
					<StepForward size={13} aria-hidden="true" /> Train ×400
				</Btn>
			{:else}
				<Btn onclick={() => (training = !training)}>
					{#if training}
						<Pause size={13} aria-hidden="true" /> Training
					{:else}
						<Play size={13} aria-hidden="true" /> Train
					{/if}
				</Btn>
			{/if}
			<Btn onclick={rest} title="Reset the simulation to hanging — the weights are untouched">
				<ArrowDownToLine size={13} aria-hidden="true" /> Rest
			</Btn>
			<Btn onclick={resetTheta} title="Forget the weights: pushes become coin flips">
				<RotateCcw size={13} aria-hidden="true" /> Reset θ
			</Btn>
		{/if}
	{/snippet}

	<div use:inview={boot} class="flex flex-col">
		{#if !ready}
			<div class="px-4 py-10 text-center">
				<span class="eyebrow">warming up…</span>
			</div>
		{:else}
			<!-- stage + rail -->
			<div class="grid grid-cols-1 sm:grid-cols-[1fr_218px]">
				<div class="relative">
					<canvas
						bind:this={canvas}
						class="block aspect-[16/7] w-full touch-none"
						class:cursor-pointer={!reduced}
						aria-label="A hinge sliding on a rail with two pendulum links hanging from it. Click or drag to shove the hinge."
						onpointerdown={down}
						onpointermove={move}
						onpointerup={up}
						onpointercancel={up}
					></canvas>
					{#if upright > 100}
						<p class="absolute right-3 bottom-2 font-serif text-[13px] text-good italic">
							it's up — now try to knock it over
						</p>
					{/if}
				</div>

				<div class="flex flex-col gap-4 border-line-soft px-4 py-4 sm:border-l">
					<div>
						<span class="eyebrow">ticks caught per drill</span>
						<svg
							viewBox="0 0 {SW} {SH}"
							preserveAspectRatio="none"
							class="mt-1.5 h-16 w-full"
							role="img"
							aria-label="Ticks spent caught upright per catch drill; the 400-tick ceiling is dashed"
						>
							<line
								x1="0"
								y1={sy(DECISIONS)}
								x2={SW}
								y2={sy(DECISIONS)}
								stroke="var(--good)"
								stroke-width="1"
								stroke-dasharray="3 3"
							/>
							{#if sparkPath}
								<path d={sparkPath} fill="none" stroke="var(--accent)" stroke-width="1.3" />
							{/if}
						</svg>
						<div class="num mt-1 flex items-center gap-1.5 text-[10.5px] text-ink-3">
							<span aria-hidden="true" style="color: var(--good);">- - -</span>
							ceiling {DECISIONS}
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 sm:grid-cols-1">
						<div>
							<span class="eyebrow">episodes</span>
							<div class="num mt-0.5 text-[15px] text-ink">{episodes}</div>
						</div>
						<div>
							<span class="eyebrow">avg caught</span>
							<div class="num mt-0.5 text-[15px] text-ink">
								{Number.isFinite(mean20) ? mean20.toFixed(0) : '—'}
							</div>
						</div>
						<div>
							<span class="eyebrow">best</span>
							<div class="num mt-0.5 text-[15px] text-ink">{best}</div>
						</div>
						<div>
							<span class="eyebrow" title="How much of a raw delivery the catch drills replay">
								drill difficulty
							</span>
							<div class="num mt-0.5 text-[15px] text-ink">{alphaPct}%</div>
						</div>
					</div>

					<div class="mt-auto flex flex-col gap-3">
						<Slider
							label="learning rate"
							bind:value={lr}
							min={0.01}
							max={0.3}
							step={0.01}
							format={(v) => v.toFixed(2)}
						/>
						<p class="font-serif text-[12.5px] text-ink-3 italic">
							{#if reduced}
								train in bursts and watch the caught-ticks climb.
							{:else if !training && episodes === 0}
								this policy is a coin flip — shove the hinge, then press Train.
							{:else}
								click or drag on the stage to shove the hinge — even mid-training.
							{/if}
						</p>
					</div>
				</div>
			</div>
		{/if}
	</div>
</Plate>
