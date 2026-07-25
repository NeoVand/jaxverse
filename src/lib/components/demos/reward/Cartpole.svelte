<script lang="ts">
	// Plate I — the inverted pendulum on a cart. A linear softmax policy
	// balances it via REINFORCE; the reader's job is to knock it over.
	// Clicking or dragging on the stage pulls the cart toward the pointer —
	// an outside force the physics feels but the policy never sees. Early
	// policies collapse at a tap; a trained one leans into the shove.
	import { onDestroy } from 'svelte';
	import { Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import {
		actionForce,
		CART_ACTIONS,
		cartFeatures,
		cartPolicy,
		cartReinforceUpdate,
		createCartBaseline,
		createCartTheta,
		isFailed,
		MAX_STEPS,
		N_FEATURES,
		physicsStep,
		resetCart,
		runCartEpisode,
		TH_LIMIT,
		X_LIMIT,
		type CartEpisode,
		type CartState
	} from '$lib/optim-rl/cartpole';
	import { mulberry32 } from '$lib/optim-rl/rng';

	const SEED = 512;
	const KEEP = 300; // sparkline window
	const SHOVE_GAIN = 8; // newtons per metre of pointer–cart separation
	const SHOVE_MAX = 15;

	// ── the learner: non-reactive, mutated in place ──
	let theta = createCartTheta();
	let baseline = createCartBaseline();
	let rand = mulberry32(SEED);
	const featBuf = new Float64Array(N_FEATURES);
	const probBuf = new Float64Array(CART_ACTIONS);

	// The live simulation the canvas shows (one episode at a time).
	let sim: CartState = { x: 0, xd: 0, th: 0.02, thd: 0 };
	let liveFeats: number[] = [];
	let liveActions: number[] = [];
	let dispForce = 0; // EMA of the policy's push, for a legible arrow
	let shoveX: number | null = null; // pointer position in world metres
	let shoveNow = 0; // last applied shove, for the arrow
	let holdUntil = 0; // brief freeze after a fall so it reads
	let fellAt = 0; // return of the episode that just ended

	// ── reactive state ──
	let ready = $state(false);
	let training = $state(false);
	let lr = $state(0.05);
	let episodes = $state(0);
	let rets = $state<number[]>([]);
	let best = $state(0);
	let upright = $state(0);
	let canvas: HTMLCanvasElement | undefined = $state();

	const reduced =
		typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

	const mean50 = $derived.by(() => {
		const r = rets.slice(-50);
		return r.length ? r.reduce((a, b) => a + b, 0) / r.length : NaN;
	});
	const holds = $derived(Number.isFinite(mean50) && mean50 >= 450);

	// ── sparkline: fixed scale, the 500-step ceiling dashed ──
	const SW = 220;
	const SH = 64;
	const sy = (v: number) => SH - 3 - (v / MAX_STEPS) * (SH - 6);
	const sparkPath = $derived.by(() => {
		if (rets.length < 2) return '';
		return rets
			.map(
				(v, i) =>
					`${i === 0 ? 'M' : 'L'}${((i / (rets.length - 1)) * SW).toFixed(1)} ${sy(v).toFixed(1)}`
			)
			.join(' ');
	});

	function boot() {
		if (ready) return;
		ready = true;
	}

	function recordReturn(steps: number) {
		episodes++;
		rets = [...rets.slice(-(KEEP - 1)), steps];
		if (steps > best) best = steps;
	}

	function resetSim() {
		sim = resetCart(rand);
		liveFeats = [];
		liveActions = [];
		upright = 0;
	}

	function finishLive(now: number) {
		fellAt = liveActions.length;
		const ep: CartEpisode = {
			feats: Float64Array.from(liveFeats),
			actions: liveActions,
			steps: liveActions.length
		};
		if (training) cartReinforceUpdate(theta, baseline, ep, lr);
		recordReturn(ep.steps);
		holdUntil = ep.steps < MAX_STEPS ? now + 280 : 0;
		resetSim();
	}

	/** One visible physics tick: the policy pushes, the reader may be shoving. */
	function stepSim(now: number) {
		cartFeatures(sim, featBuf);
		for (const v of featBuf) liveFeats.push(v);
		cartPolicy(theta, featBuf, probBuf);
		let u = rand();
		let a = 0;
		for (; a < CART_ACTIONS - 1; a++) {
			u -= probBuf[a];
			if (u <= 0) break;
		}
		liveActions.push(a);
		const push = actionForce(a);
		dispForce = dispForce * 0.75 + push * 0.25;
		shoveNow =
			shoveX === null
				? 0
				: Math.max(-SHOVE_MAX, Math.min(SHOVE_MAX, SHOVE_GAIN * (shoveX - sim.x)));
		physicsStep(sim, push + shoveNow);
		upright = liveActions.length;
		if (isFailed(sim) || liveActions.length >= MAX_STEPS) finishLive(now);
	}

	// Batch the sparkline update: one $state reassignment per episode would be
	// pure churn when a burst finishes many episodes at once.
	function trainBatch(runUntil: (done: number, t0: number) => boolean) {
		const t0 = performance.now();
		const fresh: number[] = [];
		do {
			const ep = runCartEpisode(theta, rand);
			cartReinforceUpdate(theta, baseline, ep, lr);
			fresh.push(ep.steps);
			if (ep.steps > best) best = ep.steps;
		} while (runUntil(fresh.length, t0));
		episodes += fresh.length;
		rets = [...rets, ...fresh].slice(-KEEP);
	}

	function trainBurst() {
		trainBatch((n) => n < 50);
		draw();
	}

	function resetTheta() {
		theta = createCartTheta();
		baseline = createCartBaseline();
		rand = mulberry32(SEED);
		episodes = 0;
		rets = [];
		best = 0;
		dispForce = 0;
		resetSim();
		if (reduced) draw();
	}

	// ── painter ──
	let raf = 0;
	let lastT = 0;

	$effect(() => {
		if (!canvas || !ready) return;
		if (reduced) {
			draw();
			return;
		}
		const frame = (t: number) => {
			raf = requestAnimationFrame(frame);
			lastT = t;
			if (t >= holdUntil) stepSim(t);
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
			paper: v('--paper', '#faf9f5'),
			ink: v('--ink', '#1d1c18'),
			ink2: v('--ink-2', '#605d54'),
			ink3: v('--ink-3', '#a3a094'),
			line: v('--line', '#e5e2d8'),
			lineSoft: v('--line-soft', '#efede4'),
			accent: v('--accent', '#2b45d8'),
			warm: v('--warm', '#d3541f'),
			bad: v('--bad', '#bb3a2b')
		};
	}

	function geom(w: number) {
		const scale = (w - 48) / (2 * (X_LIMIT + 0.35));
		return { scale, midX: w / 2 };
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
		const { scale, midX } = geom(W);
		const px = (wx: number) => midX + wx * scale;
		const groundY = H * 0.74;
		const poleLen = 1.0 * scale; // drawn at full length ℓ = 1 m

		// track with end stops, centre tick
		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(px(-X_LIMIT - 0.25), groundY);
		ctx.lineTo(px(X_LIMIT + 0.25), groundY);
		ctx.stroke();
		for (const e of [-X_LIMIT, X_LIMIT]) {
			ctx.beginPath();
			ctx.moveTo(px(e), groundY - 7);
			ctx.lineTo(px(e), groundY + 7);
			ctx.stroke();
		}
		ctx.strokeStyle = tk.lineSoft;
		ctx.beginPath();
		ctx.moveTo(px(0), groundY + 2);
		ctx.lineTo(px(0), groundY + 7);
		ctx.stroke();

		// cart
		const cartW = Math.max(34, scale * 0.36);
		const cartH = cartW * 0.42;
		const cartX = px(sim.x);
		const cartTop = groundY - cartH;
		ctx.fillStyle = tk.ink;
		ctx.beginPath();
		ctx.roundRect(cartX - cartW / 2, cartTop, cartW, cartH, 4);
		ctx.fill();

		// the ±12° failure cones, faint, from the pivot
		const pivotY = cartTop - 2;
		ctx.strokeStyle = tk.lineSoft;
		ctx.setLineDash([3, 4]);
		for (const sgn of [-1, 1]) {
			ctx.beginPath();
			ctx.moveTo(cartX, pivotY);
			ctx.lineTo(
				cartX + Math.sin(sgn * TH_LIMIT) * (poleLen + 8),
				pivotY - Math.cos(sgn * TH_LIMIT) * (poleLen + 8)
			);
			ctx.stroke();
		}
		ctx.setLineDash([]);

		// pole + ball tip
		const tipX = cartX + Math.sin(sim.th) * poleLen;
		const tipY = pivotY - Math.cos(sim.th) * poleLen;
		ctx.strokeStyle = tk.warm;
		ctx.lineWidth = 4;
		ctx.lineCap = 'round';
		ctx.beginPath();
		ctx.moveTo(cartX, pivotY);
		ctx.lineTo(tipX, tipY);
		ctx.stroke();
		ctx.fillStyle = tk.warm;
		ctx.beginPath();
		ctx.arc(tipX, tipY, 7, 0, Math.PI * 2);
		ctx.fill();

		// forces: the policy's push (accent), the reader's shove (bad)
		arrow(ctx, cartX, groundY - cartH / 2, dispForce * 2.2, tk.accent);
		if (Math.abs(shoveNow) > 0.5 && shoveX !== null) {
			arrow(ctx, cartX, groundY + 14, shoveNow * 2.6, tk.bad);
		}

		// a beat after each fall, say so
		if (lastT < holdUntil) {
			ctx.fillStyle = tk.ink3;
			ctx.font = `10.5px ${getComputedStyle(canvas).getPropertyValue('--font-mono') || 'monospace'}`;
			ctx.textAlign = 'center';
			ctx.fillText(`fell · R ${fellAt}`, cartX, cartTop - poleLen - 16);
		}
	}

	// ── pointer: the shove ──
	function worldX(ev: PointerEvent): number {
		if (!canvas) return 0;
		const r = canvas.getBoundingClientRect();
		const { scale, midX } = geom(r.width);
		return (ev.clientX - r.left - midX) / scale;
	}
	function down(ev: PointerEvent) {
		if (reduced) return;
		shoveX = worldX(ev);
		(ev.target as Element).setPointerCapture(ev.pointerId);
	}
	function move(ev: PointerEvent) {
		if (shoveX !== null) shoveX = worldX(ev);
	}
	function up() {
		shoveX = null;
		shoveNow = 0;
	}
</script>

<Plate
	n={1}
	title="The inverted pendulum"
	caption="Reward is +1 for every step the pole stays inside the dashed 12° cones and the cart on its track — the score only says how long, never which push was wrong. Shove the cart (click or drag the stage) before training, then again after: the difference is the policy."
>
	{#snippet status()}
		<span>
			ep {episodes} · upright {upright} · avg R {Number.isFinite(mean50) ? mean50.toFixed(0) : '—'}
			{#if holds}<span style="color: var(--good);">· it holds</span>{/if}
		</span>
	{/snippet}

	<div use:inview={boot} class="flex flex-col">
		{#if !ready}
			<div class="px-4 py-10 text-center">
				<span class="eyebrow">warming up…</span>
			</div>
		{:else}
			<!-- controls -->
			<div class="flex flex-wrap items-center gap-x-5 gap-y-3 border-b border-line-soft px-4 py-3">
				<div class="flex items-center gap-2">
					{#if reduced}
						<Btn onclick={trainBurst}>
							<StepForward size={13} aria-hidden="true" /> Train ×50 episodes
						</Btn>
					{:else}
						<Btn kind={training ? 'primary' : 'ghost'} onclick={() => (training = !training)}>
							{#if training}
								<Pause size={13} aria-hidden="true" /> Training
							{:else}
								<Play size={13} aria-hidden="true" /> Train
							{/if}
						</Btn>
					{/if}
					<Btn onclick={resetTheta} title="Forget the weights: pushes become coin flips">
						<RotateCcw size={13} aria-hidden="true" /> Reset θ
					</Btn>
				</div>
				<div class="ml-auto w-40 min-w-36">
					<Slider
						label="learning rate"
						bind:value={lr}
						min={0.005}
						max={0.2}
						step={0.005}
						format={(v) => v.toFixed(3)}
					/>
				</div>
			</div>

			<!-- stage + rail -->
			<div class="grid grid-cols-1 sm:grid-cols-[1fr_218px]">
				<div class="relative">
					<canvas
						bind:this={canvas}
						class="block aspect-[16/7] w-full touch-none"
						class:cursor-pointer={!reduced}
						aria-label="A cart on a track balancing a pole. Click or drag to shove the cart."
						onpointerdown={down}
						onpointermove={move}
						onpointerup={up}
						onpointercancel={up}
					></canvas>
					{#if holds}
						<p class="absolute right-3 bottom-2 font-serif text-[13px] text-good italic">
							it holds — now try to knock it over
						</p>
					{/if}
				</div>

				<div class="flex flex-col gap-4 border-line-soft px-4 py-4 sm:border-l">
					<div>
						<span class="eyebrow">episode return</span>
						<svg
							viewBox="0 0 {SW} {SH}"
							preserveAspectRatio="none"
							class="mt-1.5 h-16 w-full"
							role="img"
							aria-label="Steps survived per episode; the 500-step ceiling is dashed"
						>
							<line
								x1="0"
								y1={sy(MAX_STEPS)}
								x2={SW}
								y2={sy(MAX_STEPS)}
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
							ceiling 500
						</div>
					</div>

					<div class="grid grid-cols-3 gap-2 sm:grid-cols-1">
						<div>
							<span class="eyebrow">episodes</span>
							<div class="num mt-0.5 text-[15px] text-ink">{episodes}</div>
						</div>
						<div>
							<span class="eyebrow">mean return</span>
							<div class="num mt-0.5 text-[15px] text-ink">
								{Number.isFinite(mean50) ? mean50.toFixed(0) : '—'}
							</div>
						</div>
						<div>
							<span class="eyebrow">best</span>
							<div class="num mt-0.5 text-[15px] text-ink">{best}</div>
						</div>
					</div>

					<p class="mt-auto font-serif text-[12.5px] text-ink-3 italic">
						{#if reduced}
							train in bursts and watch the returns climb toward 500.
						{:else if !training && episodes === 0}
							this policy is a coin flip — shove the cart, then press Train.
						{:else}
							click or drag on the stage to shove the cart — even mid-training.
						{/if}
					</p>
				</div>
			</div>
		{/if}
	</div>
</Plate>
