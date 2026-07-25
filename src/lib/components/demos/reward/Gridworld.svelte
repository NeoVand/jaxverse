<script lang="ts">
	// Plate II — the gridworld. A tabular softmax policy learns by REINFORCE
	// with return-to-go and a per-state baseline, hundreds of episodes a
	// second, all on the main thread. The canvas shows the world, the policy
	// (four arrows per cell), the value estimate (a wash), and the current
	// episode replayed as a fading trail. An edit mode turns the board into
	// a map editor — cycle cells wall/pit, drag the start and the treasure —
	// while training keeps running, which is the whole point.
	import { onDestroy } from 'svelte';
	import { Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import SpeedChips from '$lib/components/ui/SpeedChips.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { progress } from '$lib/data/progress.svelte';
	import {
		ACTION_DX,
		ACTION_DY,
		cellIndex,
		createBaseline,
		createTheta,
		makeWorld,
		N_ACTIONS,
		optimalReturn,
		policyAt,
		reinforceUpdate,
		runEpisode,
		type Episode
	} from '$lib/optim-rl/gridworld';
	import { mulberry32 } from '$lib/optim-rl/rng';

	const SEED = 2049;
	const BASE_EPS = 40; // learning episodes per second at speed ×1
	const KEEP = 300; // sparkline window

	// ── the learner: non-reactive, mutated in place ──
	const world = makeWorld();
	let theta = createTheta(world);
	let baseline = createBaseline(world);
	let rand = mulberry32(SEED);
	const probBuf = new Float64Array(N_ACTIONS);

	// ── reactive state ──
	let ready = $state(false);
	let playing = $state(false);
	let speed = $state(1);
	let lr = $state(0.1);
	let showPolicy = $state(true);
	let showValue = $state(false);
	let editing = $state(false);
	let episodes = $state(0);
	let returns = $state<number[]>([]);
	let wins = $state<number[]>([]);
	let optReturn = $state(optimalReturn(world));
	let solved = $state(false);
	let lastGoalEp = $state(0);
	let dragging = $state<'goal' | 'start' | null>(null);
	let hoverMarker = $state(false);
	let canvas: HTMLCanvasElement | undefined = $state();

	const reduced =
		typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

	const successRate = $derived.by(() => {
		const w = wins.slice(-50);
		return w.length ? w.reduce((a, b) => a + b, 0) / w.length : 0;
	});
	const mean50 = $derived.by(() => {
		const r = returns.slice(-50);
		return r.length ? r.reduce((a, b) => a + b, 0) / r.length : NaN;
	});
	const stuck = $derived(playing && episodes - lastGoalEp > 400);

	// ── sparkline (SVG, fixed viewBox) ──
	const SW = 220;
	const SH = 64;
	const spark = $derived.by(() => {
		if (returns.length < 2) return { path: '', optY: null as number | null };
		let lo = Number.isFinite(optReturn) ? optReturn : Infinity;
		let hi = Number.isFinite(optReturn) ? optReturn : -Infinity;
		for (const v of returns) {
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
		if (hi - lo < 1e-9) hi = lo + 1e-9;
		const y = (v: number) => SH - 3 - ((v - lo) / (hi - lo)) * (SH - 6);
		const path = returns
			.map(
				(v, i) =>
					`${i === 0 ? 'M' : 'L'}${((i / (returns.length - 1)) * SW).toFixed(1)} ${y(v).toFixed(1)}`
			)
			.join(' ');
		return { path, optY: Number.isFinite(optReturn) ? y(optReturn) : null };
	});

	// ── episode bookkeeping ──
	// The replayed episode: already learned from, drawn as it unfolds.
	let visEp: { path: number[]; ms: number; t: number } | null = null;
	// A finished path drawn statically (paused / reduced-motion stepping).
	let staticPath: number[] | null = null;

	function boot() {
		if (ready) return;
		ready = true;
	}

	function learnOne(): Episode {
		const ep = runEpisode(world, theta, rand);
		reinforceUpdate(world, theta, baseline, ep, lr);
		return ep;
	}

	function checkSolved() {
		if (solved || wins.length < 50) return;
		const w = wins.slice(-50);
		if (w.reduce((a, b) => a + b, 0) / w.length >= 0.9) {
			solved = true;
			progress.reach('reward:solved');
		}
	}

	function record(ep: Episode) {
		episodes++;
		returns = [...returns.slice(-(KEEP - 1)), ep.totalReward];
		wins = [...wins.slice(-(KEEP - 1)), ep.end === 'goal' ? 1 : 0];
		if (ep.end === 'goal') lastGoalEp = episodes;
		checkSolved();
	}

	// One $state write per frame, not per episode — at max speed hundreds of
	// episodes finish in a single frame.
	function learnBatch(budgetMs: number) {
		const t0 = performance.now();
		const r: number[] = [];
		const w: number[] = [];
		do {
			const ep = learnOne();
			r.push(ep.totalReward);
			w.push(ep.end === 'goal' ? 1 : 0);
			if (ep.end === 'goal') lastGoalEp = episodes + r.length;
		} while (performance.now() - t0 < budgetMs);
		episodes += r.length;
		returns = [...returns, ...r].slice(-KEEP);
		wins = [...wins, ...w].slice(-KEEP);
		checkSolved();
	}

	function stepTen() {
		let last: Episode | null = null;
		for (let i = 0; i < 10; i++) {
			last = learnOne();
			record(last);
		}
		visEp = null;
		staticPath = last ? last.path : null;
		if (reduced) draw();
	}

	function resetTheta() {
		playing = false;
		theta = createTheta(world);
		baseline = createBaseline(world);
		rand = mulberry32(SEED);
		episodes = 0;
		returns = [];
		wins = [];
		solved = false;
		lastGoalEp = 0;
		visEp = null;
		staticPath = null;
		if (reduced) draw();
	}

	// ── painter: runs once booted; learning is driven from it while playing ──
	let raf = 0;
	let lastT = 0;
	let headlessAcc = 0;

	$effect(() => {
		if (!canvas || !ready) return;
		if (reduced) {
			draw();
			return;
		}
		const frame = (t: number) => {
			raf = requestAnimationFrame(frame);
			const dt = lastT ? Math.min(100, t - lastT) : 16;
			lastT = t;
			if (playing) {
				if (speed === 0) {
					learnBatch(7);
				} else {
					headlessAcc += (dt / 1000) * BASE_EPS * speed;
					let n = Math.floor(headlessAcc);
					headlessAcc -= n;
					while (n-- > 0) record(learnOne());
				}
				if (visEp) {
					visEp.t += dt;
					if (visEp.t >= visEp.ms) visEp = null;
				}
				if (!visEp) {
					const ep = learnOne();
					record(ep);
					// short rivers replay at ~3/s; long early wanders get up to 1 s
					visEp = { path: ep.path, ms: Math.min(1000, 140 + 16 * ep.steps), t: 0 };
					staticPath = null;
				}
			}
			draw();
		};
		raf = requestAnimationFrame(frame);
		return () => cancelAnimationFrame(raf);
	});

	onDestroy(() => {
		// also runs after server prerender — keep it browser-safe
		if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf);
	});

	// ── drawing ──
	function tokens(el: Element) {
		const s = getComputedStyle(el);
		const v = (name: string, fb: string) => s.getPropertyValue(name).trim() || fb;
		return {
			paper: v('--paper', '#faf9f5'),
			ink: v('--ink', '#1d1c18'),
			ink2: v('--ink-2', '#605d54'),
			ink3: v('--ink-3', '#a3a094'),
			line: v('--line', '#e5e2d8'),
			accent: v('--accent', '#2b45d8'),
			warm: v('--warm', '#d3541f'),
			good: v('--good', '#22774d'),
			bad: v('--bad', '#bb3a2b')
		};
	}

	function geom(w: number, h: number) {
		const pad = 10;
		const cell = Math.min((w - 2 * pad) / world.w, (h - 2 * pad) / world.h);
		return { cell, ox: (w - cell * world.w) / 2, oy: (h - cell * world.h) / 2 };
	}

	let dragCell = -1; // candidate cell while dragging a marker (painter reads it)

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
		const { cell, ox, oy } = geom(W, H);
		const left = (x: number) => ox + x * cell;
		const top = (y: number) => oy + (world.h - 1 - y) * cell;
		const cx = (s: number) => left(s % world.w) + cell / 2;
		const cy = (s: number) => top((s / world.w) | 0) + cell / 2;

		// value wash — the baseline estimate, paper → accent
		if (showValue) {
			let lo = Infinity;
			let hi = -Infinity;
			for (let s = 0; s < baseline.length; s++) {
				if (world.walls.has(s)) continue;
				if (baseline[s] < lo) lo = baseline[s];
				if (baseline[s] > hi) hi = baseline[s];
			}
			if (hi - lo > 1e-6) {
				ctx.fillStyle = tk.accent;
				for (let s = 0; s < baseline.length; s++) {
					if (world.walls.has(s)) continue;
					ctx.globalAlpha = 0.3 * ((baseline[s] - lo) / (hi - lo));
					ctx.fillRect(left(s % world.w), top((s / world.w) | 0), cell, cell);
				}
				ctx.globalAlpha = 1;
			}
		}

		// walls, pits, goal
		ctx.fillStyle = tk.ink3;
		ctx.globalAlpha = 0.3;
		for (const s of world.walls)
			ctx.fillRect(left(s % world.w), top((s / world.w) | 0), cell, cell);
		ctx.globalAlpha = 1;
		for (const s of world.pits) {
			ctx.fillStyle = tk.bad;
			ctx.globalAlpha = 0.12;
			ctx.fillRect(left(s % world.w), top((s / world.w) | 0), cell, cell);
			ctx.globalAlpha = 0.55;
			ctx.beginPath();
			ctx.arc(cx(s), cy(s), cell * 0.15, 0, Math.PI * 2);
			ctx.fill();
			ctx.globalAlpha = 1;
		}
		const drawGoal = (s: number, alpha: number) => {
			ctx.globalAlpha = alpha * 0.16;
			ctx.fillStyle = tk.good;
			ctx.fillRect(left(s % world.w), top((s / world.w) | 0), cell, cell);
			ctx.globalAlpha = alpha;
			const r = cell * 0.2;
			ctx.beginPath();
			ctx.moveTo(cx(s), cy(s) - r);
			ctx.lineTo(cx(s) + r, cy(s));
			ctx.lineTo(cx(s), cy(s) + r);
			ctx.lineTo(cx(s) - r, cy(s));
			ctx.closePath();
			ctx.fill();
			ctx.globalAlpha = 1;
		};
		const drawStart = (s: number, alpha: number) => {
			ctx.globalAlpha = alpha;
			ctx.strokeStyle = tk.ink3;
			ctx.lineWidth = 1.2;
			ctx.beginPath();
			ctx.arc(cx(s), cy(s), cell * 0.13, 0, Math.PI * 2);
			ctx.stroke();
			ctx.globalAlpha = 1;
		};
		drawGoal(world.goal, dragging === 'goal' ? 0.3 : 1);
		if (dragging === 'goal' && dragCell !== -1 && dragCell !== world.goal) drawGoal(dragCell, 0.85);

		// hairline grid
		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		for (let x = 0; x <= world.w; x++) {
			ctx.moveTo(left(x), oy);
			ctx.lineTo(left(x), oy + cell * world.h);
		}
		for (let y = 0; y <= world.h; y++) {
			ctx.moveTo(ox, oy + y * cell);
			ctx.lineTo(ox + cell * world.w, oy + y * cell);
		}
		ctx.stroke();

		// start marker
		drawStart(world.start, dragging === 'start' ? 0.3 : 1);
		if (dragging === 'start' && dragCell !== -1 && dragCell !== world.start)
			drawStart(dragCell, 0.85);

		// policy arrows: opacity ∝ π(a|s); a clear argmax gets accent and heft
		if (showPolicy) {
			for (let s = 0; s < baseline.length; s++) {
				if (world.walls.has(s) || world.pits.has(s) || s === world.goal) continue;
				policyAt(theta, s, probBuf);
				let am = 0;
				for (let a = 1; a < N_ACTIONS; a++) if (probBuf[a] > probBuf[am]) am = a;
				let second = 0;
				for (let a = 0; a < N_ACTIONS; a++)
					if (a !== am && probBuf[a] > second) second = probBuf[a];
				const decisive = probBuf[am] - second > 0.05;
				for (let a = 0; a < N_ACTIONS; a++) {
					const ux = ACTION_DX[a];
					const uy = -ACTION_DY[a]; // canvas y grows downward
					const boss = decisive && a === am;
					const size = (boss ? 0.15 : 0.11) * cell;
					const off = 0.24 * cell;
					const bx = cx(s) + ux * off;
					const by = cy(s) + uy * off;
					ctx.beginPath();
					ctx.moveTo(bx + ux * size, by + uy * size);
					ctx.lineTo(bx - uy * size * 0.62, by + ux * size * 0.62);
					ctx.lineTo(bx + uy * size * 0.62, by - ux * size * 0.62);
					ctx.closePath();
					ctx.fillStyle = boss ? tk.accent : tk.ink2;
					ctx.globalAlpha = Math.min(1, probBuf[a] * 1.2 + 0.04);
					ctx.fill();
				}
				ctx.globalAlpha = 1;
			}
		}

		// the current episode: a warm dot with a fading trail
		const trail = (path: number[], head: number) => {
			ctx.lineWidth = 2;
			ctx.strokeStyle = tk.warm;
			ctx.lineCap = 'round';
			const i0 = Math.max(0, Math.ceil(head) - 16);
			for (let i = i0; i < Math.floor(head); i++) {
				ctx.globalAlpha = Math.max(0, 0.65 - (head - i - 1) * 0.045);
				ctx.beginPath();
				ctx.moveTo(cx(path[i]), cy(path[i]));
				ctx.lineTo(cx(path[i + 1]), cy(path[i + 1]));
				ctx.stroke();
			}
			ctx.globalAlpha = 1;
			const i = Math.min(path.length - 2, Math.floor(head));
			const u = Math.min(1, head - i);
			const dx = cx(path[i]) + (cx(path[i + 1]) - cx(path[i])) * u;
			const dy = cy(path[i]) + (cy(path[i + 1]) - cy(path[i])) * u;
			if (u > 0 && head > i) {
				ctx.globalAlpha = 0.65;
				ctx.beginPath();
				ctx.moveTo(cx(path[i]), cy(path[i]));
				ctx.lineTo(dx, dy);
				ctx.stroke();
				ctx.globalAlpha = 1;
			}
			ctx.beginPath();
			ctx.arc(dx, dy, cell * 0.16, 0, Math.PI * 2);
			ctx.fillStyle = tk.warm;
			ctx.fill();
			ctx.strokeStyle = tk.paper;
			ctx.lineWidth = 1.5;
			ctx.stroke();
		};
		if (visEp && visEp.path.length > 1) {
			trail(visEp.path, Math.min(1, visEp.t / visEp.ms) * (visEp.path.length - 1));
		} else if (staticPath && staticPath.length > 1) {
			ctx.lineWidth = 2;
			ctx.strokeStyle = tk.warm;
			ctx.globalAlpha = 0.3;
			ctx.beginPath();
			for (let i = 0; i < staticPath.length; i++) {
				if (i === 0) ctx.moveTo(cx(staticPath[i]), cy(staticPath[i]));
				else ctx.lineTo(cx(staticPath[i]), cy(staticPath[i]));
			}
			ctx.stroke();
			ctx.globalAlpha = 1;
			const end = staticPath[staticPath.length - 1];
			ctx.beginPath();
			ctx.arc(cx(end), cy(end), cell * 0.16, 0, Math.PI * 2);
			ctx.fillStyle = tk.warm;
			ctx.fill();
		}
	}

	// ── pointer: the map editor ──
	function eventCell(ev: PointerEvent): number {
		if (!canvas) return -1;
		const r = canvas.getBoundingClientRect();
		const { cell, ox, oy } = geom(r.width, r.height);
		const x = Math.floor((ev.clientX - r.left - ox) / cell);
		const y = world.h - 1 - Math.floor((ev.clientY - r.top - oy) / cell);
		if (x < 0 || x >= world.w || y < 0 || y >= world.h) return -1;
		return cellIndex(x, y);
	}

	function validTarget(c: number, marker: 'goal' | 'start'): boolean {
		if (c === -1 || world.walls.has(c) || world.pits.has(c)) return false;
		return marker === 'goal' ? c !== world.start : c !== world.goal;
	}

	/** Any change to the map: recompute the optimum, drop the stale replay. */
	function worldChanged() {
		optReturn = optimalReturn(world);
		visEp = null;
		staticPath = null;
		if (reduced) draw();
	}

	function cycleCell(c: number) {
		if (c === -1 || c === world.start || c === world.goal) return;
		if (world.walls.has(c)) {
			world.walls.delete(c);
			world.pits.add(c);
		} else if (world.pits.has(c)) {
			world.pits.delete(c);
		} else {
			world.walls.add(c);
		}
		worldChanged();
	}

	function down(ev: PointerEvent) {
		if (!editing) return;
		const c = eventCell(ev);
		if (c === world.goal) dragging = 'goal';
		else if (c === world.start) dragging = 'start';
		else {
			cycleCell(c);
			return;
		}
		dragCell = c;
		(ev.target as Element).setPointerCapture(ev.pointerId);
	}
	function move(ev: PointerEvent) {
		if (!editing) {
			hoverMarker = false;
			return;
		}
		const c = eventCell(ev);
		if (dragging) {
			if (validTarget(c, dragging)) dragCell = c;
			if (reduced) draw();
		} else {
			hoverMarker = c === world.goal || c === world.start;
		}
	}
	function up() {
		if (!dragging) return;
		const marker = dragging;
		dragging = null;
		if (validTarget(dragCell, marker)) {
			if (marker === 'goal' && dragCell !== world.goal) {
				world.goal = dragCell;
				worldChanged();
			} else if (marker === 'start' && dragCell !== world.start) {
				world.start = dragCell;
				worldChanged();
			}
		}
		dragCell = -1;
	}
</script>

<Plate
	n={2}
	title="The gridworld"
	caption="No ε dial anywhere: the early policy is nearly uniform, and that is the exploration — arrows sharpen only as one route proves better than usual. Flip on edit and reshape the world mid-training: wall off the river, move the treasure, and watch the arrows unlearn, then re-carve."
>
	{#snippet status()}
		<span>
			ep {episodes} · avg G {Number.isFinite(mean50) ? mean50.toFixed(2) : '—'} · success {(
				successRate * 100
			).toFixed(0)}%
			{#if solved}<span style="color: var(--good);">· solved</span>{/if}
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
					{#if !reduced}
						<Btn kind={playing ? 'primary' : 'ghost'} onclick={() => (playing = !playing)}>
							{#if playing}
								<Pause size={13} aria-hidden="true" /> Pause
							{:else}
								<Play size={13} aria-hidden="true" /> Play
							{/if}
						</Btn>
					{/if}
					<Btn onclick={stepTen}>
						<StepForward size={13} aria-hidden="true" /> Step ×10
					</Btn>
					<Btn onclick={resetTheta} title="Forget everything: uniform policy, zero value">
						<RotateCcw size={13} aria-hidden="true" /> Reset θ
					</Btn>
				</div>
				<span class="flex items-center gap-1" role="group" aria-label="Overlays and editing">
					<button
						class="chip"
						class:chip-on={showPolicy}
						onclick={() => (showPolicy = !showPolicy)}
					>
						policy
					</button>
					<button class="chip" class:chip-on={showValue} onclick={() => (showValue = !showValue)}>
						value
					</button>
					<button class="chip" class:chip-on={editing} onclick={() => (editing = !editing)}>
						edit
					</button>
				</span>
				{#if !reduced}
					<SpeedChips bind:value={speed} />
				{/if}
				<div class="ml-auto w-40 min-w-36">
					<Slider
						label="learning rate"
						bind:value={lr}
						min={0.05}
						max={0.5}
						step={0.01}
						format={(v) => v.toFixed(2)}
					/>
				</div>
			</div>

			<!-- stage + rail -->
			<div class="grid grid-cols-1 sm:grid-cols-[1fr_218px]">
				<div class="relative">
					<canvas
						bind:this={canvas}
						class="block aspect-[4/3] w-full touch-none"
						class:cursor-grab={hoverMarker && !dragging}
						class:cursor-grabbing={dragging !== null}
						class:cursor-pointer={editing && !hoverMarker && !dragging}
						aria-label="An 8 by 6 gridworld: a start ring, a treasure, pits and walls. Arrows in each cell show the policy's action probabilities. In edit mode, clicking a cell cycles it between empty, wall, and pit."
						onpointerdown={down}
						onpointermove={move}
						onpointerup={up}
						onpointercancel={up}
					></canvas>
					{#if solved}
						<p class="absolute right-3 bottom-2 font-serif text-[13px] text-good italic">
							the gradient carved a river to the treasure
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
							aria-label="Return of each recent episode, with the best possible return marked"
						>
							{#if spark.path && spark.optY !== null}
								<line
									x1="0"
									y1={spark.optY}
									x2={SW}
									y2={spark.optY}
									stroke="var(--good)"
									stroke-width="1"
									stroke-dasharray="3 3"
								/>
							{/if}
							{#if spark.path}
								<path d={spark.path} fill="none" stroke="var(--accent)" stroke-width="1.3" />
							{/if}
						</svg>
						<div class="num mt-1 flex items-center gap-1.5 text-[10.5px] text-ink-3">
							<span aria-hidden="true" style="color: var(--good);">- - -</span>
							{#if Number.isFinite(optReturn)}
								optimal {optReturn.toFixed(2)}
							{:else}
								treasure unreachable
							{/if}
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
								{Number.isFinite(mean50) ? mean50.toFixed(2) : '—'}
							</div>
						</div>
						<div>
							<span class="eyebrow">success rate</span>
							<div class="num mt-0.5 text-[15px] text-ink">{(successRate * 100).toFixed(0)}%</div>
						</div>
					</div>

					<p class="mt-auto font-serif text-[12.5px] text-ink-3 italic">
						{#if stuck}
							no treasure for 400 episodes — the policy has gone rigid. Lower the rate, or Reset θ.
						{:else if editing}
							click a cell: empty → wall → pit. Drag the ○ start and the ◆ treasure.
						{:else}
							flip on <span class="not-italic">edit</span> to reshape the world — even while it trains.
						{/if}
					</p>
				</div>
			</div>

			<!-- legend -->
			<div
				class="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-line-soft px-4 py-2 text-[10.5px] text-ink-2"
				style="font-family: var(--font-sans);"
			>
				<span class="flex items-center gap-1.5">
					<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
						<rect width="12" height="12" fill="var(--good)" opacity="0.16" />
						<path d="M6 2.4 L9.6 6 L6 9.6 L2.4 6 Z" fill="var(--good)" />
					</svg>
					treasure +10
				</span>
				<span class="flex items-center gap-1.5">
					<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
						<rect width="12" height="12" fill="var(--bad)" opacity="0.12" />
						<circle cx="6" cy="6" r="2" fill="var(--bad)" opacity="0.55" />
					</svg>
					pit −8
				</span>
				<span class="flex items-center gap-1.5">
					<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
						<rect width="12" height="12" fill="var(--ink-3)" opacity="0.3" />
					</svg>
					wall
				</span>
				<span class="flex items-center gap-1.5">
					<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
						<circle cx="6" cy="6" r="3.4" fill="none" stroke="var(--ink-3)" stroke-width="1.2" />
					</svg>
					start
				</span>
				<span class="flex items-center gap-1.5">
					<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
						<path d="M10 6 L3.5 2.8 L3.5 9.2 Z" fill="var(--accent)" />
					</svg>
					policy π(a|s)
				</span>
				<span class="flex items-center gap-1.5">
					<span
						aria-hidden="true"
						class="inline-block h-3 w-6 rounded-[2px] border border-line-soft"
						style="background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--accent) 30%, transparent));"
					></span>
					value estimate
				</span>
				<span class="flex items-center gap-1.5">
					<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
						<circle cx="6" cy="6" r="3" fill="var(--warm)" />
					</svg>
					agent
				</span>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.chip {
		font-family: var(--font-sans);
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		padding: 3px 9px;
		border-radius: 5px;
		border: 1px solid var(--line);
		color: var(--ink-2);
		background: var(--surface);
		transition: all 100ms ease;
	}
	.chip:hover {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.chip-on {
		background: var(--ink);
		border-color: var(--ink);
		color: var(--paper);
	}
</style>
