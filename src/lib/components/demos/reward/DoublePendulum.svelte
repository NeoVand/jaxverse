<script lang="ts">
	import { readTokens } from '$lib/viz/tokens.svelte';
	// The double pendulum swing-up on a sliding hinge. The links
	// start hanging straight down; a linear softmax policy over hand-crafted
	// gauges learns, via REINFORCE, to pump energy into the stack and catch
	// it upright. The reader's job is to interfere: clicking or dragging
	// pulls the hinge toward the pointer — an outside force the physics
	// feels but the policy never sees. Nothing ever resets on its own; the
	// policy is always driving. Training happens in a pool of Web Worker
	// "practice halls" racing off the main thread: each is an independent
	// learner and the stage always performs with the current champion's θ.
	// Racing, not averaging — and not only for speed: single streams get
	// trapped by unlucky exploration about half the time, and six
	// independent streams make that a 1-in-64 event. The main thread only
	// trains as a fallback (no Worker, or the reduced-motion burst button).
	//
	// This file is also the BROKER for what passes between the halls (the
	// mechanism is documented in $lib/optim-rl/swarm). It hears every hall's
	// discoveries and rebroadcasts them, and once a second it hands every
	// hall the pool's weights, its own rank, and a prior over whom it would
	// make sense to listen to. It decides nothing: each hall's own social
	// learner chooses whether to take anyone's advice, and what comes back in
	// the reports is a learned adjacency matrix trained on rank alone. Both
	// layers are drawn — the swarm puts every hall on the one stage, and the
	// field below is that matrix, live.
	import { onDestroy } from 'svelte';
	import { ArrowDownToLine, Layers, Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import {
		ACTION_FORCES,
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
		isPrimeDelivery,
		isUpright,
		L1,
		L2,
		N_FEATURES,
		N_PARAMS,
		physicsStep,
		resetDpole,
		TH_LIMIT,
		X_LIMIT,
		type DpoleEpisode,
		type DpoleStart,
		type DpoleState
	} from '$lib/optim-rl/dpole';
	import { mulberry32 } from '$lib/optim-rl/rng';
	import { priorLogits, pullStrength, RISE_EMA, SWARM } from '$lib/optim-rl/swarm';
	import {
		camera,
		fieldBox,
		fitness,
		isoLine,
		niceStep,
		isoWindow,
		separate
	} from '$lib/optim-rl/field';
	import PracticeHall from '$lib/optim-rl/dpole.worker?worker';

	const SEED = 512;
	const KEEP = 300; // sparkline window
	const SHOVE_GAIN = 10; // newtons per metre of pointer–hinge separation
	const SHOVE_MAX = 24; // enough to out-muscle the policy's 20 N, barely
	const SHOVE_HOLD_MS = 180; // a quick click still lands a real push
	const BURST_MS = 9; // main-thread fallback: per-frame budget for
	const BURST_CAP = 45; // headless episodes when Workers are unavailable
	const HALL_CAP = 6; // parallel practice halls by default (fewer on small
	// machines). The reader can take it much higher — past the core count the
	// halls simply time-share, each one slower, the pool as a whole broader.
	// That is a real trade and worth being able to feel.
	const HALL_MAX = 28;
	const CANDIDATES = 8; // most halls one hall may consider listening to
	const plannedHalls =
		typeof navigator === 'undefined'
			? HALL_CAP
			: Math.max(1, Math.min(HALL_CAP, (navigator.hardwareConcurrency ?? 4) - 2));

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
	const liveProbs = new Float64Array(DPOLE_ACTIONS); // its live opinion
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
	let halls = $state(0); // practice halls currently training
	let poolSize = $state(0); // halls that exist (they persist through pause)
	let champView = $state(0); // champion index, for the outline
	let hallScoreView = $state<number[]>([]); // per-hall fitness, for the strip
	let shared = $state(0); // discoveries broadcast between halls
	let lastFind = $state<{ hall: number; to: number } | null>(null);
	// the multi-agent view: every hall on the one stage, and the lane below
	// showing what passes between them. One switch, because they are one
	// picture — the swarm and the reason it is a swarm.
	let stageAll = $state(false);
	// how many learners are in the pool. Live: the slider adds and removes
	// halls without disturbing the ones already training.
	let wantHalls = $state(plannedHalls);
	let canvas: HTMLCanvasElement | undefined = $state();
	let webCanvas: HTMLCanvasElement | undefined = $state();
	let miniCanvases: (HTMLCanvasElement | undefined)[] = $state([]);

	// ── the practice-hall pool ──
	// Halls report a few hundred times a second; touching $state that often
	// would thrash the DOM, so results land in these plain buffers and the
	// animation frame flushes them once per paint.
	let pool: Worker[] = [];
	let hallAlpha: number[] = [];
	let hallScore: number[] = [];
	let champIdx = 0;
	let pendingEpisodes = 0;
	let pendingDrills: number[] = [];
	// each hall's freshest θ and a little stage of its own — the race made
	// visible: the main thread steps and draws these; the halls just train
	let hallTheta: Float64Array[] = [];
	let hallSims: DpoleState[] = [];
	let hallTrails: number[][] = [];

	// ── the swarm: state the broker keeps about who is worth listening to ──
	const PULL_MS = 800; // how often weights may seep downhill
	const FIND_COOLDOWN = 250; // ms — a hall may announce this often, at most
	const PULSE_MS = 1300; // how long a shared discovery takes to fly
	let hallSlow: number[] = []; // each hall's own slow fitness average…
	let hallRise: number[] = []; // …and how far it has just pulled above it
	let hallEMAs: { deliverEMA: number; drillEMA: number }[] = [];
	let hallFindAt: number[] = []; // last announcement, for the cooldown
	// row w = hall w's LEARNED social weights over the pool (last entry: the
	// probability it refuses everyone), as reported back by the hall itself
	let attn: Float64Array[] = [];
	let prior: Float64Array[] = [];
	let pullK: number[] = []; // and how hard hall w is being pulled at all
	let pulses: { from: number; t0: number }[] = [];
	let lastMix = 0;

	function spawnHall(w: number) {
		const hall = new PracticeHall();
		hall.onmessage = (e: MessageEvent) => onReport(w, e.data);
		hall.postMessage({ type: 'boot', seed: SEED + 1 + w, lr });
		pool.push(hall);
		hallAlpha.push(0.25);
		hallScore.push(0);
		hallSlow.push(0);
		hallRise.push(0);
		hallEMAs.push({ deliverEMA: 0, drillEMA: 0 });
		hallFindAt.push(0);
		pullK.push(0);
		hallTheta.push(createDpoleTheta());
		hallSims.push(resetDpole(() => 0.5));
		hallTrails.push([]);
	}

	/** Grow or shrink the pool in place. Halls that already exist keep their
	 * weights, their curriculum and their social opinions — the slider adds
	 * colleagues rather than restarting the experiment, and a hall arriving
	 * late arrives at zero, which is worth watching: you can see the pool
	 * decide whether the newcomers are worth listening to. */
	function sizePool(n: number) {
		if (typeof Worker === 'undefined') return;
		while (pool.length > n) {
			const hall = pool.pop();
			hall?.terminate();
			for (const a of [
				hallAlpha,
				hallScore,
				hallSlow,
				hallRise,
				hallFindAt,
				pullK,
				nodeX,
				nodeY,
				showD,
				showC
			])
				a.pop();
			for (const a of [hallEMAs, hallTheta, hallSims, hallTrails, hallPath]) a.pop();
			if (champIdx >= pool.length) champIdx = 0;
		}
		while (pool.length < n) {
			spawnHall(pool.length);
			if (training) {
				pool[pool.length - 1].postMessage({ type: 'lr', lr });
				pool[pool.length - 1].postMessage({ type: 'run' });
			}
		}
		// the drawn rows are sized by the pool, and a resize invalidates them
		attn = pool.map(() => new Float64Array(n + 1));
		prior = pool.map(() => new Float64Array(n));
		pulses = [];
		poolSize = pool.length;
		if (training) halls = pool.length;
	}

	function startPool() {
		if (pool.length || typeof Worker === 'undefined') return;
		sizePool(wantHalls);
	}

	/** A hall finished a round. The best hall's θ IS the stage's policy, and
	 * anything the hall just discovered becomes everyone's to rehearse. */
	function onReport(
		w: number,
		m: {
			type: string;
			theta: Float64Array;
			score: number;
			deliverEMA: number;
			drillEMA: number;
			episodes: number;
			drills: number[];
			alpha: number;
			finds: { state: DpoleState; bonus: number; grad: Float64Array }[];
			social: Float64Array | null;
		}
	) {
		if (m.type !== 'report') return;
		hallScore[w] = m.score;
		hallAlpha[w] = m.alpha;
		hallEMAs[w] = { deliverEMA: m.deliverEMA, drillEMA: m.drillEMA };
		hallTheta[w].set(m.theta);
		// fitness against the hall's own slow average: `rise` is the spike a
		// hall shows in the seconds after it finds something, and it is what
		// the others' attention actually follows
		hallSlow[w] += RISE_EMA * (m.score - hallSlow[w]);
		hallRise[w] = Math.max(0, m.score - hallSlow[w]);
		pendingEpisodes += m.episodes;
		for (const c of m.drills) pendingDrills.push(c);
		// A report is in flight when the learners slider moves, so its social
		// row can be sized for a pool that no longer exists. Writing it anyway
		// either throws or — worse, silently — files the "refuse everyone"
		// probability under some hall's index and draws a ribbon that was never
		// learned.
		if (m.social && m.social.length === attn[w].length) attn[w].set(m.social);
		if (m.finds?.length) announce(w, m.finds);
		// The champion is STICKY: the stage follows one hall's θ until
		// another beats it by a clear margin. Following the argmax naively
		// flapped the stage between near-tied halls dozens of times a
		// second — six competent controllers with six different pumping
		// rhythms, alternating mid-swing, add up to no rhythm at all.
		if (w !== champIdx && m.score > 1.2 * hallScore[champIdx] + 0.02) champIdx = w;
		if (w === champIdx) theta.set(m.theta);
	}

	/** Hall `w` landed a delivery cleaner than anything it had managed. The
	 * broker relays it to every other hall — the state to rehearse from, and
	 * the gradient of the episode that found it, which they step along at
	 * once. Not the finder's weights: those would overwrite five independent
	 * streams with one, and independence is the whole reason there are six. */
	function announce(w: number, finds: { state: DpoleState; bonus: number; grad: Float64Array }[]) {
		const now = performance.now();
		if (now - hallFindAt[w] < FIND_COOLDOWN) return;
		hallFindAt[w] = now;
		const best = finds.reduce((a, b) => (b.bonus > a.bonus ? b : a));
		let sent = 0;
		for (let j = 0; j < pool.length; j++) {
			if (j === w) continue;
			// The state goes to everyone; the GRADIENT only downhill. Measured:
			// six halls all stepping along whoever just found something turned
			// six independent streams into one, and the pool started failing
			// together — three runs in six where nobody learned, against zero
			// for the same code with no sharing at all. The insurance a race
			// buys is exactly its independence, and a signal that biases where
			// a hall searches next spends that insurance. So the rule is the
			// same one the weights follow: anything that steers a search flows
			// only to halls doing worse than the finder, and the leaders keep
			// their own luck.
			const downhill = hallScore[j] < hallScore[w];
			pool[j].postMessage({
				type: 'delivery',
				state: best.state,
				grad: downhill ? best.grad : undefined
			});
			sent++;
		}
		if (!sent) return;
		pulses.push({ from: w, t0: now });
		shared += sent;
		lastFind = { hall: w, to: sent };
		if (pulses.length > 240) pulses.splice(0, pulses.length - 240);
	}

	/** One round of the social layer. The broker does no deciding: it hands
	 * every hall the pool's weights, the prior over whom it would make sense
	 * to listen to, and its own rank — and each hall's own little REINFORCE
	 * learner chooses whether to take anyone's advice at all. What comes back
	 * in the reports is a learned adjacency matrix: who listens to whom, and
	 * how much, trained on nothing but each hall's rank in the race. */
	function mixRound(now: number) {
		const n = pool.length;
		if (n < 2) return;
		lastMix = now;
		let best = 0;
		for (const s of hallScore) if (s > best) best = s;
		const order = Array.from({ length: n }, (_, i) => i).sort(
			(a, b) => hallScore[b] - hallScore[a]
		);
		const rank: number[] = [];
		order.forEach((w, i) => (rank[w] = i));
		for (let w = 0; w < n; w++) {
			priorLogits(hallScore, hallRise, w, prior[w]);
			// Only the halls ahead of this one are candidates, and at most the
			// best few of those. With a pool of thirty that keeps each message
			// a few kilobytes rather than a megabyte, and costs nothing: a hall
			// was never going to take advice from someone behind it.
			const cand = [];
			for (let j = 0; j < n; j++) if (Number.isFinite(prior[w][j])) cand.push(j);
			cand.sort((a, b) => prior[w][b] - prior[w][a]);
			cand.length = Math.min(CANDIDATES, cand.length);
			const K = cand.length;
			const thetas = new Float64Array(K * N_PARAMS);
			const pr = new Float64Array(K);
			const dEMA = new Float64Array(K);
			const cEMA = new Float64Array(K);
			for (let c = 0; c < K; c++) {
				thetas.set(hallTheta[cand[c]], c * N_PARAMS);
				pr[c] = prior[w][cand[c]];
				dEMA[c] = hallEMAs[cand[c]].deliverEMA;
				cEMA[c] = hallEMAs[cand[c]].drillEMA;
			}
			// how much one accepted step is allowed to move this hall: a hall at
			// the front barely moves even when it chooses to listen, because the
			// independence of the leading stream is the pool's whole insurance
			pullK[w] = pullStrength(hallScore[w], best) / SWARM.maxPull;
			pool[w].postMessage({
				type: 'peers',
				thetas,
				cand: Int32Array.from(cand),
				prior: pr,
				deliverEMAs: dEMA,
				drillEMAs: cEMA,
				rank: rank[w],
				n,
				gate: Math.max(0.12, pullK[w]),
				now
			});
		}
	}

	function runPool() {
		startPool();
		for (const hall of pool) {
			hall.postMessage({ type: 'lr', lr });
			hall.postMessage({ type: 'run' });
		}
		halls = pool.length;
	}

	function stopPool() {
		for (const hall of pool) hall.postMessage({ type: 'stop' });
		halls = 0;
	}

	function killPool() {
		for (const hall of pool) hall.terminate();
		pool = [];
		hallAlpha = [];
		hallScore = [];
		hallTheta = [];
		hallSims = [];
		hallTrails = [];
		hallSlow = [];
		hallRise = [];
		hallEMAs = [];
		hallFindAt = [];
		attn = [];
		prior = [];
		pullK = [];
		pulses = [];
		nodeX = [];
		nodeY = [];
		hallPath = [];
		showD = [];
		showC = [];
		trailD = [];
		trailC = [];
		champIdx = 0;
		pendingEpisodes = 0;
		pendingDrills = [];
		halls = 0;
		poolSize = 0;
		champView = 0;
		hallScoreView = [];
		shared = 0;
		lastFind = null;
	}

	/** Flush worker results into reactive state — called once per frame. */
	function flushHalls() {
		if (pendingEpisodes) {
			episodes += pendingEpisodes;
			pendingEpisodes = 0;
		}
		if (pendingDrills.length) {
			for (const c of pendingDrills) if (c > best) best = c;
			caughtLog = [...caughtLog, ...pendingDrills].slice(-KEEP);
			pendingDrills = [];
		}
		if (hallAlpha.length) {
			const a = hallAlpha.reduce((x, y) => x + y, 0) / hallAlpha.length;
			alphaPct = Math.round(a * 100);
		}
		if (champView !== champIdx) champView = champIdx;
		if (hallScore.length) hallScoreView = [...hallScore];
	}

	/** Step and paint the six little stages — each performs its own hall's
	 * current policy, live. No shoves, no resets, no bookkeeping: these are
	 * honest windows into the race, driven by the same physics. */
	function stepMinis() {
		for (let w = 0; w < hallSims.length; w++) {
			const s = hallSims[w];
			dpoleFeatures(s, featBuf);
			dpolePolicy(hallTheta[w], featBuf, probBuf);
			let u = rand();
			let a = 0;
			for (; a < DPOLE_ACTIONS - 1; a++) {
				u -= probBuf[a];
				if (u <= 0) break;
			}
			physicsStep(s, actionForce(a));
			// each hall's own tip trail — six of them woven over one rail is
			// the clearest picture of six policies this demo can draw
			const tr = hallTrails[w] ?? (hallTrails[w] = []);
			tr.push(
				s.x + Math.sin(s.th1) * L1 + Math.sin(s.th2) * L2,
				Math.cos(s.th1) * L1 + Math.cos(s.th2) * L2
			);
			if (tr.length > 200) tr.splice(0, tr.length - 200);
		}
	}

	/** One rig, drawn wherever: the overlay on the big stage and the little
	 * live portraits in the influence lane are the same picture at two
	 * sizes, so a hall is recognisably itself in both. */
	function drawRig(
		ctx: CanvasRenderingContext2D,
		s: DpoleState,
		hx: number,
		railY: number,
		scale: number,
		o: { color: string; width: number; bob: number; alpha: number; good?: string }
	) {
		const k1x = hx + Math.sin(s.th1) * L1 * scale;
		const k1y = railY - Math.cos(s.th1) * L1 * scale;
		const k2x = k1x + Math.sin(s.th2) * L2 * scale;
		const k2y = k1y - Math.cos(s.th2) * L2 * scale;
		// holding is drawn as light: the rig burns green, and the greener it
		// burns the harder it is holding. A rig fighting its way back from a
		// shove dims and re-lights in real time, which is the whole story of
		// this plate in one channel
		const g = o.good ? held(s) : 0;
		const body = g > 0.55 ? (o.good as string) : o.color;
		ctx.lineCap = 'round';
		ctx.lineJoin = 'round';
		ctx.strokeStyle = body;
		ctx.fillStyle = body;
		if (g > 0.02) {
			ctx.shadowColor = o.good as string;
			ctx.shadowBlur = (6 + 22 * g) * (o.bob / 5);
		}
		ctx.globalAlpha = o.alpha;
		ctx.lineWidth = o.width;
		ctx.beginPath();
		ctx.moveTo(hx, railY);
		ctx.lineTo(k1x, k1y);
		ctx.lineTo(k2x, k2y);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(k1x, k1y, o.bob * 0.72, 0, Math.PI * 2);
		ctx.fill();
		ctx.beginPath();
		ctx.arc(k2x, k2y, o.bob, 0, Math.PI * 2);
		ctx.fill();
		ctx.shadowBlur = 0;
		ctx.beginPath();
		ctx.arc(hx, railY, o.bob * 0.8, 0, Math.PI * 2);
		ctx.fill();
		// the halo, on top of the shadow: a soft disc at the tip that swells
		// as the hold firms up
		if (g > 0.02 && o.good) {
			ctx.fillStyle = o.good;
			ctx.globalAlpha = o.alpha * 0.3 * g;
			ctx.beginPath();
			ctx.arc(k2x, k2y, o.bob * (2 + 3 * g), 0, Math.PI * 2);
			ctx.fill();
		}
		ctx.globalAlpha = 1;
	}

	function drawMinis() {
		if (!hallSims.length) return;
		const first = miniCanvases.find(Boolean);
		if (!first) return;
		const tk = tokens(first);
		const now = performance.now();
		const dpr = Math.min(devicePixelRatio || 1, 2);
		for (let w = 0; w < hallSims.length; w++) {
			const c = miniCanvases[w];
			if (!c) continue;
			const W = c.clientWidth;
			const H = c.clientHeight;
			if (!W || !H) continue;
			if (c.width !== W * dpr || c.height !== H * dpr) {
				c.width = W * dpr;
				c.height = H * dpr;
			}
			const ctx = c.getContext('2d');
			if (!ctx) continue;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, W, H);
			const s = hallSims[w];
			const railY = H * 0.5;
			const scale = Math.min((W - 10) / (2 * (X_LIMIT + 0.35)), railY - 4);
			const px = (wx: number) => W / 2 + wx * scale;
			ctx.strokeStyle = tk.lineSoft;
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(px(-X_LIMIT), railY);
			ctx.lineTo(px(X_LIMIT), railY);
			ctx.stroke();
			drawRig(ctx, s, px(s.x), railY, scale, {
				color: hue(tk, w),
				width: 1.9,
				bob: 2.6,
				alpha: 0.95,
				good: tk.good
			});
			// the hall's colour, so one learner is the same learner in the
			// swarm overlay and in the influence view
			ctx.fillStyle = hue(tk, w);
			ctx.beginPath();
			ctx.arc(6, 6, 2.6, 0, Math.PI * 2);
			ctx.fill();
			// and a flash when it has just told the others something
			const age = (now - hallFindAt[w]) / 700;
			if (age >= 0 && age < 1) {
				ctx.globalAlpha = (1 - age) * 0.9;
				ctx.strokeStyle = hue(tk, w);
				ctx.lineWidth = 1.5;
				ctx.strokeRect(1, 1, W - 2, H - 2);
				ctx.globalAlpha = 1;
			}
		}
	}

	/** Each hall's identity colour — one hue per learner, everywhere. */
	function hue(tk: ReturnType<typeof tokens>, w: number): string {
		return tk.cats[w % tk.cats.length];
	}

	/** How well a rig is holding, 0…1 — dead vertical and dead calm is 1,
	 * the edge of the cones (or any real spin) is 0. Graded, not the binary
	 * `isUpright`: the glow is the scoreboard now, and a policy that is
	 * *nearly* holding should look nearly there. Drawn as green light rather
	 * than a pair of dashed cones — the cones told you where the target was,
	 * which the rig itself tells you the moment it starts glowing. */
	function held(s: DpoleState): number {
		const a1 = Math.atan2(Math.sin(s.th1), Math.cos(s.th1));
		const a2 = Math.atan2(Math.sin(s.th2), Math.cos(s.th2));
		const lean = (Math.abs(a1) + Math.abs(a2)) / (2 * TH_LIMIT);
		const spin = (Math.abs(s.th1d) + Math.abs(s.th2d)) / 4;
		return Math.max(0, 1 - lean) * Math.max(0, 1 - spin);
	}

	/** A canvas font string is not CSS and will not resolve var(--font-sans). */
	const FONT = 'Inter, system-ui, sans-serif';

	// ── the field ──
	// Every hall is a living portrait, not a dot: the same rig it is running
	// on the stage above, drawn small, standing at the place two skills have
	// earned it. Across is how often it gets the stack to the top at all; up
	// is how much of the catch window it then holds. Those are not two
	// decorative measurements — they are the two terms of the fitness the
	// pool actually races on, so the diagonals are lines of equal fitness and
	// a hall's rank is legible as "which rung is it standing on".
	//
	// The position is a claim, and keeping it one is the hard part. See
	// $lib/optim-rl/field: overlapping halls are fanned apart ALONG a line of
	// equal fitness and never across one, so the picture can never put a
	// better hall below a worse one. It used to. Replayed on a converged pool
	// the old collision solver drew 8.8% of delivery comparisons and 15.8% of
	// fitness comparisons backwards, by up to 0.037 of delivery rate — enough
	// to see a hall obviously performing well sitting at the bottom of the
	// field, which is what sent this code to be audited.
	let nodeX: number[] = [];
	let nodeY: number[] = [];
	/** The settled skills each hall is drawn at — smoothed in data space, so
	 * that a drawn position is always some real pair of numbers. */
	let showD: number[] = [];
	let showC: number[] = [];
	/** …and a much slower pair, which is what the wake is drawn from. */
	let trailD: number[] = [];
	let trailC: number[] = [];
	/** Each hall's wake, kept in DATA space. Screen space was wrong: the
	 * camera drifts as the pool spreads, so a trail recorded in pixels is a
	 * history of a field that has since moved. */
	let hallPath: number[][] = [];
	let lastPathAt = 0;
	/** The field, once and for all: both axes are rates, both run 0…1. */
	const VIEW = [0, 1, 0, 1];

	function drawLane(now: number) {
		const c = webCanvas;
		if (!c) return;
		const n = hallSims.length;
		const W = c.clientWidth;
		const H = c.clientHeight;
		if (!n || !W || !H) return;
		const dpr = Math.min(devicePixelRatio || 1, 2);
		if (c.width !== W * dpr || c.height !== H * dpr) {
			c.width = W * dpr;
			c.height = H * dpr;
		}
		const ctx = c.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);
		const tk = tokens(c);

		// the portraits shrink as the pool grows, so thirty of them still fit
		const slot = Math.max(9, Math.min(40, H * 0.1, Math.sqrt((W * H) / (n * 18)) / 2));
		const rigScale = slot * 1.25;
		const box = fieldBox(W, H, slot);

		// ── fixed axes, on purpose ──
		// This used to be a camera that tracked the pool's spread, and it was
		// a mistake with a name: if the frame moves, a trail is a path through
		// a space that no longer exists, and the one thing a trail is for —
		// comparing where a hall was with where it is — becomes unreadable.
		// Both skills are already rates in 0…1, so the field IS 0…1, always.
		// A pool that ends up crowded into the top right has not defeated the
		// chart; that is what winning looks like.
		const cam = camera(VIEW, box);

		// ── the rungs: lines of equal fitness ──
		// deliver + ½·catch, the number the halls are ranked by, drawn as the
		// ladder it is. Two halls on one rung are worth the same to the race
		// and got there differently, which is the single most useful thing
		// this picture can say.
		const fLo = fitness(VIEW[0], VIEW[2]);
		const fHi = fitness(VIEW[1], VIEW[3]);
		const step = niceStep((fHi - fLo) / 5);
		ctx.font = `10px ${FONT}`;
		ctx.textBaseline = 'middle';
		let labelled = false;
		for (let f = Math.ceil(fLo / step) * step; f <= fHi + 1e-9; f += step) {
			const seg = isoLine(f, VIEW);
			if (!seg) continue;
			const [ax, ay, bx, by] = seg;
			ctx.strokeStyle = tk.line;
			ctx.setLineDash([3, 5]);
			ctx.lineWidth = 1;
			ctx.globalAlpha = 0.55;
			ctx.beginPath();
			ctx.moveTo(cam.x(ax), cam.y(ay));
			ctx.lineTo(cam.x(bx), cam.y(by));
			ctx.stroke();
			ctx.setLineDash([]);
			// The number rides the LOW end of the rung — down among the
			// delivery axis rather than up the side, where it read as a y tick
			// and had nothing to do with y at all.
			const low = cam.y(ay) > cam.y(by) ? [cam.x(ax), cam.y(ay)] : [cam.x(bx), cam.y(by)];
			const lx = Math.min(W - 30, Math.max(box.left, low[0]));
			const ly = Math.min(H - 8, Math.max(box.top + 6, low[1]));
			ctx.globalAlpha = 0.9;
			ctx.fillStyle = tk.ink3;
			ctx.textAlign = 'left';
			ctx.fillText(f.toFixed(2), lx + 4, ly - 6);
			if (!labelled) {
				labelled = true;
				ctx.globalAlpha = 0.6;
				ctx.fillText('fitness', lx + 4, ly + 7);
			}
			ctx.globalAlpha = 1;
		}

		// the two walls — a frame, not a scale: the numbers on the rungs are
		// the reading, and these only say which way is better
		ctx.strokeStyle = tk.lineSoft;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(box.left - 10, box.top - 10);
		ctx.lineTo(box.left - 10, box.bottom + 10);
		ctx.lineTo(box.right + 8, box.bottom + 10);
		ctx.stroke();

		// ── where each hall stands ──
		// The settling is done HERE, on the two skills, and not on the pixels
		// afterwards. Smoothing pixels was the second way this plot lied: a
		// lerp toward a target is a straight line in screen space, and a
		// straight line in screen space crosses rungs, so a hall on its way
		// somewhere was drawn with a fitness it did not have. Measured live at
		// six halls it put the third-best hall in second place. Smoothing the
		// skills instead keeps every drawn position the exact image of some
		// real pair of numbers — a slightly stale reading, which is honest,
		// rather than a position between two readings, which is not.
		// The follower snaps when it is far behind and eases when it is close.
		//
		// A plain lerp cannot win here. Slow enough to kill the tremble means
		// slow enough to lag, and lag is a fitness error, and a fitness error
		// is the picture drawing two halls in the wrong order — measured live
		// at 2.7% of the pairs far enough apart to see. Making the rate grow
		// with the distance still to cover separates the two jobs: a hall that
		// has genuinely moved arrives within a frame or two, while the last
		// hundredth of tremble is damped hard, because that is the only part
		// worth damping.
		for (let w = 0; w < n; w++) {
			const e = hallEMAs[w] ?? { deliverEMA: 0, drillEMA: 0 };
			if (showD[w] === undefined) {
				showD[w] = e.deliverEMA;
				showC[w] = e.drillEMA;
				continue;
			}
			const gd = e.deliverEMA - showD[w];
			const gc = e.drillEMA - showC[w];
			const rate = Math.min(1, 0.25 + 6 * Math.hypot(gd, gc));
			showD[w] += rate * gd;
			showC[w] += rate * gc;
		}
		const px = Float64Array.from({ length: n }, (_, w) => cam.x(showD[w]));
		const py = Float64Array.from({ length: n }, (_, w) => cam.y(showC[w]));
		// Fanned apart along a rung, never across one. `room` is how much of
		// that rung the canvas actually offers; when the pool outgrows it the
		// halls close up rather than being clamped, because a clamped point
		// stands somewhere it did not earn.
		// How much of a rung the box actually offers, measured rather than
		// guessed: the projection of the drawable rectangle onto the fan's own
		// direction. Past that the halls close up instead of walking off the
		// plate, which is the other way a rig used to end up outside its field.
		// How much of a rung the box offers FROM WHERE THE POOL IS — not the
		// box's own width along that direction, which is an upper bound the fan
		// can overrun, because a trained pool sits in a corner and the corner
		// is where the room runs out. Past it the halls close up rather than
		// walking off the plate.
		let cx = 0;
		let cy = 0;
		for (let w = 0; w < n; w++) {
			cx += px[w] / n;
			cy += py[w] / n;
		}
		separate(px, py, n, 2 * slot + 6, cam.iso, isoWindow(cx, cy, cam.iso, box, box.slack));
		for (let w = 0; w < n; w++) {
			nodeX[w] = px[w];
			nodeY[w] = py[w];
		}

		// ── the wake each hall leaves through the field ──
		// Sampled slowly, in data space, and drawn through the midpoints as
		// one smooth curve: what matters is the SHAPE of a learning history —
		// right first, then up, and the long slide back when a hall loses the
		// catch — not the frame-to-frame tremble of an EMA.
		if (now - lastPathAt > 420) {
			lastPathAt = now;
			for (let w = 0; w < n; w++) {
				const e = hallEMAs[w];
				if (!e) continue;
				// A much slower average than the one the rig stands on. The
				// trail was drawn from the live EMA first and it was a scribble
				// — that EMA turns over in about a second, so fifty seconds of
				// it is a random walk with the shape of the learning buried
				// somewhere inside. Seven seconds of smoothing leaves the arc:
				// right first, then up, and the long slide back when a hall
				// loses the catch.
				trailD[w] =
					trailD[w] === undefined ? e.deliverEMA : trailD[w] + 0.06 * (e.deliverEMA - trailD[w]);
				trailC[w] =
					trailC[w] === undefined ? e.drillEMA : trailC[w] + 0.06 * (e.drillEMA - trailC[w]);
				const p = hallPath[w] ?? (hallPath[w] = []);
				const k = p.length;
				if (k && Math.hypot(trailD[w] - p[k - 2], trailC[w] - p[k - 1]) < 0.003) continue;
				p.push(trailD[w], trailC[w]);
				if (p.length > 72) p.splice(0, p.length - 72);
			}
		}
		for (let w = 0; w < n; w++) {
			const p = hallPath[w];
			const pts = p ? p.length / 2 : 0;
			if (!p || pts < 3) continue;
			const sx = (i: number) => cam.x(p[2 * i]);
			const sy = (i: number) => cam.y(p[2 * i + 1]);
			ctx.strokeStyle = hue(tk, w);
			ctx.lineWidth = 1.1;
			for (let i = 1; i < pts - 1; i++) {
				ctx.globalAlpha = 0.14 * (i / pts) ** 1.5;
				ctx.beginPath();
				ctx.moveTo((sx(i - 1) + sx(i)) / 2, (sy(i - 1) + sy(i)) / 2);
				ctx.quadraticCurveTo(sx(i), sy(i), (sx(i) + sx(i + 1)) / 2, (sy(i) + sy(i + 1)) / 2);
				ctx.stroke();
			}
			// and a last stroke into where it stands now
			ctx.globalAlpha = 0.14;
			ctx.beginPath();
			ctx.moveTo((sx(pts - 2) + sx(pts - 1)) / 2, (sy(pts - 2) + sy(pts - 1)) / 2);
			ctx.quadraticCurveTo(sx(pts - 1), sy(pts - 1), nodeX[w], nodeY[w]);
			ctx.stroke();
		}
		ctx.globalAlpha = 1;

		// ── the social graph ──
		// Not traffic: structure. One ribbon per learned connection, from the
		// hall being listened to into the hall listening, thick where the weight
		// is large and tapering as it arrives. These are the numbers each hall
		// trained on its own rank — the pool's opinion of itself — so the shape
		// changes over minutes, not frames, and a hall that stops paying off
		// visibly loses its incoming ribbons.
		const inAttn = new Array(n).fill(0);
		for (let w = 0; w < n; w++)
			for (let j = 0; j < n; j++) if (j !== w) inAttn[j] += attn[w][j] ?? 0;
		// a hall that just found something flares, and everything it is telling
		// the others brightens with it
		const flare = new Array(n).fill(0);
		for (let p = pulses.length - 1; p >= 0; p--) {
			const t = (now - pulses[p].t0) / PULSE_MS;
			if (t < 0 || t > 1) {
				pulses.splice(p, 1);
				continue;
			}
			if (pulses[p].from < n) flare[pulses[p].from] = Math.max(flare[pulses[p].from], 1 - t);
		}

		const fade = Math.max(0.4, Math.min(1, 10 / n));
		for (let w = 0; w < n; w++) {
			for (let j = 0; j < n; j++) {
				const a = attn[w][j] ?? 0;
				if (j === w || a < 0.05) continue;
				const x0 = nodeX[j];
				const y0 = nodeY[j];
				const x1 = nodeX[w];
				const y1 = nodeY[w];
				const dx = x1 - x0;
				const dy = y1 - y0;
				const len = Math.hypot(dx, dy) || 1;
				// bow it, so the two directions of a mutual pair never overlap
				const bow = Math.min(30, 0.16 * len);
				const cx = (x0 + x1) / 2 - (dy / len) * bow;
				const cy = (y0 + y1) / 2 + (dx / len) * bow;
				const nx = -dy / len;
				const ny = dx / len;
				const wS = 0.8 + 4 * a; // wide where it leaves…
				const wT = 0.7; // …a filament where it lands
				ctx.fillStyle = hue(tk, j);
				// thinner as the pool grows: thirty halls all looking at the same
				// two leaders is a wash at the alpha six halls need
				ctx.globalAlpha = Math.min(0.7, (0.08 + 0.4 * a) * fade * (1 + 1.6 * flare[j]));
				ctx.beginPath();
				ctx.moveTo(x0 + nx * wS, y0 + ny * wS);
				ctx.quadraticCurveTo(cx + nx * wT, cy + ny * wT, x1 + nx * wT, y1 + ny * wT);
				ctx.lineTo(x1 - nx * wT, y1 - ny * wT);
				ctx.quadraticCurveTo(cx - nx * wT, cy - ny * wT, x0 - nx * wS, y0 - ny * wS);
				ctx.closePath();
				ctx.fill();
			}
		}
		ctx.globalAlpha = 1;

		// ── the halls: each its own live rig, standing where it has earned ──
		for (let w = 0; w < n; w++) {
			const x = nodeX[w];
			const y = nodeY[w];
			const s = hallSims[w];
			const champ = w === champView;
			// the aura is ATTENTION RECEIVED — the sum of the pool's learned
			// weights pointing at this hall. Nobody assigns it; it is what the
			// other halls' independent social learners decided.
			const got = inAttn[w] / Math.max(1, n - 1);
			if (got > 0.01) {
				ctx.globalAlpha = 0.1 + 0.22 * got;
				ctx.fillStyle = hue(tk, w);
				ctx.beginPath();
				ctx.arc(x, y, slot * (0.55 + 0.85 * got), 0, Math.PI * 2);
				ctx.fill();
				ctx.globalAlpha = 1;
			}
			// and the shell is REFUSAL: how much this hall has learned to take
			// nobody's advice at all
			const shut = attn[w][n] ?? 0;
			if (shut > 0.08) {
				ctx.strokeStyle = tk.ink3;
				ctx.globalAlpha = 0.15 + 0.5 * shut;
				ctx.lineWidth = 0.8 + 2 * shut;
				ctx.setLineDash([3, 4]);
				ctx.beginPath();
				ctx.arc(x, y, slot * 0.95, 0, Math.PI * 2);
				ctx.stroke();
				ctx.setLineDash([]);
				ctx.globalAlpha = 1;
			}
			if (flare[w] > 0.01) {
				ctx.strokeStyle = hue(tk, w);
				ctx.globalAlpha = 0.75 * flare[w];
				ctx.lineWidth = 1.6;
				ctx.beginPath();
				ctx.arc(x, y, slot * (0.6 + 2 * (1 - flare[w])), 0, Math.PI * 2);
				ctx.stroke();
				ctx.globalAlpha = 1;
			}
			// The champion is the hall whose θ the big rig on the rail above is
			// running. That was previously a one-pixel change of stroke colour,
			// and readers could not find the pendulum they were shoving. It is
			// now an amber ring in the stage's own colour, with the label —
			// the same object, in two places.
			// Vermilion, and not one of the ten categorical hues: `tk.amber` is
			// cats[1], so an amber ring would have been indistinguishable from
			// hall 2's own colour. Warm belongs to no hall, and it is the
			// colour of the big rig's lower link — the same object, marked the
			// same way, in both places.
			if (champ) {
				ctx.strokeStyle = tk.warm;
				ctx.globalAlpha = 0.9;
				ctx.lineWidth = 1.6;
				ctx.beginPath();
				ctx.arc(x, y, slot * 1.18, 0, Math.PI * 2);
				ctx.stroke();
				ctx.globalAlpha = 0.8;
				ctx.fillStyle = tk.warm;
				ctx.font = `9.5px ${FONT}`;
				ctx.textAlign = 'center';
				ctx.fillText('on the rail ↑', x, y - slot * 1.18 - 5);
				ctx.textAlign = 'left';
				ctx.globalAlpha = 1;
			}
			ctx.strokeStyle = champ ? tk.warm : tk.lineSoft;
			ctx.lineWidth = champ ? 1.4 : 1;
			ctx.beginPath();
			ctx.moveTo(x - slot * 0.8, y);
			ctx.lineTo(x + slot * 0.8, y);
			ctx.stroke();
			drawRig(ctx, s, x + (s.x / X_LIMIT) * slot * 0.7, y, rigScale, {
				color: hue(tk, w),
				width: champ ? 2.6 : 2.1,
				bob: champ ? 4 : 3.3,
				alpha: champ ? 1 : 0.85,
				good: tk.good
			});
		}
	}

	function toggleTraining() {
		training = !training;
		if (training) runPool();
		else stopPool();
	}

	// the learners slider grows and shrinks the pool in place
	$effect(() => {
		const n = Math.round(wantHalls);
		if (pool.length && pool.length !== n) sizePool(n);
	});

	// the learning-rate slider steers the halls too
	$effect(() => {
		const v = lr;
		for (const hall of pool) hall.postMessage({ type: 'lr', lr: v });
	});

	// halls run free of requestAnimationFrame, so a hidden tab would keep
	// burning six cores — rest them until the reader comes back
	$effect(() => {
		if (typeof document === 'undefined') return;
		const onVisibility = () => {
			if (!training) return;
			if (document.hidden) stopPool();
			else runPool();
		};
		document.addEventListener('visibilitychange', onVisibility);
		return () => document.removeEventListener('visibilitychange', onVisibility);
	});

	const reduced =
		typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

	const mean20 = $derived.by(() => {
		const r = caughtLog.slice(-20);
		return r.length ? r.reduce((a, b) => a + b, 0) / r.length : NaN;
	});
	const catches = $derived(Number.isFinite(mean20) && mean20 >= 50);

	// ── how the catch drills are going ──
	// This used to be a line of caught-ticks per drill, and it was unreadable
	// — correctly so. The quantity is BIMODAL: a drill either loses the stack
	// in the first second or rides the 400-tick ceiling, and almost nothing
	// lands in between. A line through samples drawn from two far-apart modes
	// is white noise whatever the policy is doing, and its mean is a number
	// describing an outcome that never happens.
	//
	// What is actually improving is the MIX. So the chart is the share of
	// recent drills in each outcome — held to the ceiling, held a while,
	// dropped — stacked to fill the box, over a rolling window. Learning is
	// green rising from the floor, a collapse is green draining away, and
	// nothing in it flickers.
	const SW = 220;
	const SH = 64;
	const WIN = 24; // drills per column
	const HELD = 100; // 2 seconds: held on to it, even if it lost it later
	const outcomes = $derived.by(() => {
		if (caughtLog.length < WIN) return null;
		const cols = caughtLog.length - WIN + 1;
		const top: string[] = []; // boundary above "held to the ceiling"
		const mid: string[] = []; // boundary above "held a while"
		let ceil = 0;
		let some = 0;
		for (let i = 0; i < WIN; i++) {
			if (caughtLog[i] >= DECISIONS) ceil++;
			else if (caughtLog[i] >= HELD) some++;
		}
		for (let c = 0; c < cols; c++) {
			if (c > 0) {
				// slide the window: drop the leaving drill, add the arriving one
				const out = caughtLog[c - 1];
				const inn = caughtLog[c + WIN - 1];
				if (out >= DECISIONS) ceil--;
				else if (out >= HELD) some--;
				if (inn >= DECISIONS) ceil++;
				else if (inn >= HELD) some++;
			}
			const x = ((c / Math.max(1, cols - 1)) * SW).toFixed(1);
			top.push(`${x} ${(SH - (ceil / WIN) * SH).toFixed(1)}`);
			mid.push(`${x} ${(SH - ((ceil + some) / WIN) * SH).toFixed(1)}`);
		}
		return {
			ceiling: `M0 ${SH} L${top.join(' L')} L${SW} ${SH} Z`,
			some: `M${top.join(' L')} L${[...mid].reverse().join(' L')} Z`,
			pct: Math.round((ceil / WIN) * 100)
		};
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
			// the champion performed this segment, so the champion learns
			// from it; the local update matters only for the fallback
			if (pool.length) pool[champIdx].postMessage({ type: 'live', episode: ep });
			else dpoleReinforceUpdate(theta, baseline, [ep], lr);
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
		liveProbs.set(probBuf); // the opinion, not just the sample: drawn
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
			if (isPrimeDelivery(sim)) {
				curriculum.deliveries.push({ ...sim });
				// live deliveries — including ones that happened under a
				// shove — become rehearsable drills in every practice hall
				for (const hall of pool) hall.postMessage({ type: 'delivery', state: { ...sim } });
			}
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
		killPool();
		theta = createDpoleTheta();
		baseline = createDpoleBaseline();
		curriculum = new DpoleCurriculum();
		rand = mulberry32(SEED);
		episodes = 0;
		caughtLog = [];
		best = 0;
		alphaPct = 25;
		rest();
		if (training) runPool();
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
			flushHalls();
			// main-thread fallback only if Workers don't exist at all
			if (training && pool.length === 0)
				trainBatch((n, t0) => performance.now() - t0 < BURST_MS && n < BURST_CAP * 8);
			// the between-agent step rides the frame loop on purpose: a hidden
			// tab stops mixing exactly when it stops painting
			if (training && t - lastMix > PULL_MS) mixRound(t);
			stepLive(t);
			stepMinis();
			draw();
			if (stageAll) drawLane(t);
			else drawMinis();
		};
		raf = requestAnimationFrame(frame);
		return () => cancelAnimationFrame(raf);
	});

	onDestroy(() => {
		// also runs after server prerender — keep it browser-safe
		if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf);
		killPool();
	});

	const tokens = (el: Element) => {
		const t = readTokens(el);
		return { ...t, amber: t.cats[1] };
	};

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

		// No target cones. The rig lights up green as it holds — the target
		// is legible from the thing itself, and a pair of dashed lines through
		// the middle of the figure cost more than they told.

		// The swarm: every hall performing at once on the one rail. They are
		// not six views of one policy — each is a different learner's live
		// run, so early on the stage is six kinds of flailing and later six
		// pumping rhythms drift into one. Trails first, so the woven history
		// sits behind the rigs; the champion last and brightest.
		if (stageAll) {
			const order = [...hallSims.keys()].sort((a, b) =>
				a === champView ? 1 : b === champView ? -1 : 0
			);
			for (const w of order) {
				const tr = hallTrails[w];
				if (!tr) continue;
				const pts = tr.length / 2;
				ctx.fillStyle = hue(tk, w);
				for (let i = 0; i < pts; i++) {
					const f = (i + 1) / pts;
					ctx.globalAlpha = f * f * (w === champView ? 0.5 : 0.34);
					ctx.beginPath();
					ctx.arc(px(tr[2 * i]), railY - tr[2 * i + 1] * scale, 1.1 + 1.4 * f, 0, Math.PI * 2);
					ctx.fill();
				}
			}
			ctx.globalAlpha = 1;
			for (const w of order) {
				const champ = w === champView;
				drawRig(ctx, hallSims[w], px(hallSims[w].x), railY, scale, {
					color: hue(tk, w),
					width: champ ? 3 : 2.4,
					bob: champ ? 5.5 : 4.4,
					alpha: champ ? 0.95 : 0.72,
					good: tk.good
				});
			}
		}

		// the tip's recent past — loops while swinging, a knot once caught
		const n = trail.length / 2;
		for (let i = 0; i < n; i++) {
			const a = ((i + 1) / n) * 0.32;
			ctx.fillStyle = stageAll ? tk.ink3 : tk.amber;
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
		// yours. Among six coloured learners it goes ink: the neutral one on
		// stage is the one the pointer can shove. And as it settles it burns
		// green — the hold drawn as light, brightening and dimming with every
		// correction instead of a badge that flips on at ±18°.
		const g = held(sim);
		const lower = g > 0.55 ? tk.good : stageAll ? tk.ink : tk.warm;
		const upper = g > 0.55 ? tk.good : stageAll ? tk.ink : tk.amber;
		ctx.lineCap = 'round';
		if (g > 0.02) {
			ctx.shadowColor = tk.good;
			ctx.shadowBlur = 8 + 34 * g;
		}
		ctx.strokeStyle = lower;
		ctx.lineWidth = 3.5;
		ctx.beginPath();
		ctx.moveTo(hx, railY);
		ctx.lineTo(k1x, k1y);
		ctx.stroke();
		ctx.strokeStyle = upper;
		ctx.beginPath();
		ctx.moveTo(k1x, k1y);
		ctx.lineTo(k2x, k2y);
		ctx.stroke();
		ctx.fillStyle = lower;
		ctx.beginPath();
		ctx.arc(k1x, k1y, 5, 0, Math.PI * 2);
		ctx.fill();
		ctx.fillStyle = upper;
		ctx.beginPath();
		ctx.arc(k2x, k2y, 6.5, 0, Math.PI * 2);
		ctx.fill();
		ctx.shadowBlur = 0;
		if (g > 0.02) {
			ctx.fillStyle = tk.good;
			ctx.globalAlpha = 0.3 * g;
			ctx.beginPath();
			ctx.arc(k2x, k2y, 12 + 26 * g, 0, Math.PI * 2);
			ctx.fill();
			ctx.globalAlpha = 1;
		}

		// the hinge: a dot riding the rail — the only thing the policy moves
		ctx.fillStyle = g > 0.55 ? tk.good : tk.ink;
		ctx.beginPath();
		ctx.arc(hx, railY, 7, 0, Math.PI * 2);
		ctx.fill();

		// The policy's live opinion, drawn as thrust: one tapered plume per
		// available push, its brightness and thickness the probability the
		// softmax is giving it right now. This is the policy's actual output
		// — not a summary of it — so an undecided policy fans both ways in
		// pale symmetry, a decided one fires a single bright jet, and the
		// catch reads as constant tiny left-right corrections. The plume for
		// "coast" is a ring on the hinge itself.
		for (let a = 0; a < DPOLE_ACTIONS; a++) {
			const p = liveProbs[a];
			if (p < 0.02) continue;
			const f = ACTION_FORCES[a];
			ctx.fillStyle = tk.accent;
			ctx.strokeStyle = tk.accent;
			if (f === 0) {
				ctx.globalAlpha = 0.1 + 0.45 * p;
				ctx.lineWidth = 1.4;
				ctx.beginPath();
				ctx.arc(hx, railY, 11.5, 0, Math.PI * 2);
				ctx.stroke();
				continue;
			}
			const dir = Math.sign(f);
			const x0 = hx + dir * 9;
			const x1 = x0 + f * 46;
			const h = 1.6 + 6.5 * p;
			ctx.globalAlpha = 0.14 + 0.7 * p;
			ctx.beginPath();
			ctx.moveTo(x0, railY - h);
			ctx.lineTo(x1, railY - 0.9);
			ctx.lineTo(x1, railY + 0.9);
			ctx.lineTo(x0, railY + h);
			ctx.closePath();
			ctx.fill();
		}
		ctx.globalAlpha = 1;
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
	id="pendulum"
	live
	title="The double pendulum swing-up"
	caption="A pool of independent learners, one problem, and a second thing being learned on top of it: who to listen to. Each hall is its own policy on its own background thread — press Swarm and they all step onto the one rail, a colour each, with your own rig among them. Discoveries are broadcast: an arrival cleaner than any a hall has managed goes to every other hall as something to practise from, and the gradient behind it goes to the halls doing worse. But influence is not imposed. Every hall runs a second, tiny REINFORCE learner whose actions are &ldquo;take that hall&rsquo;s weights a step&rdquo; and &ldquo;refuse everyone&rdquo;, and whose only reward is whether it climbed the pool&rsquo;s ranking. Imposed instead, influence made the pool fail together — mean fitness 0.33 against 0.82 for six halls sharing nothing — because a hall mid-discovery that keeps being dragged toward the leader never finishes its own idea. The field below stands each hall on the two skills its fitness is made of, so the labelled diagonals are lines of equal fitness and the ribbons are what the halls decided about each other."
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
				<Btn onclick={toggleTraining}>
					{#if training}
						<Pause size={13} aria-hidden="true" /> Training
					{:else}
						<Play size={13} aria-hidden="true" /> Train
					{/if}
				</Btn>
			{/if}
			{#if !reduced}
				<Btn
					onclick={() => (stageAll = !stageAll)}
					pressed={stageAll}
					disabled={poolSize === 0}
					title="Perform every hall on the one stage — six learners, one rail"
				>
					<Layers size={13} aria-hidden="true" /> Swarm
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
			<!-- stage + rail; the race strip docks under the stage so the left
			     column fills the same height as the sidebar — no dead band -->
			<div class="grid grid-cols-1 sm:grid-cols-[1fr_218px]">
				<div class="flex flex-col">
					<!-- The stage. The rig's scale is set by the RAIL's width, so
					     height past 2× its reach is dead band — on desktop the box is
					     cropped to just past that, and the room goes to the field
					     below, which can use it. Narrow screens keep the taller box:
					     there the rig is height-limited, not width-limited. -->
					<div
						class="relative aspect-[16/7] flex-1 sm:aspect-[16/6] {stageAll
							? 'sm:max-h-[320px]'
							: ''}"
					>
						<canvas
							bind:this={canvas}
							class="block h-full w-full touch-none"
							class:cursor-pointer={!reduced}
							aria-label="A hinge sliding on a rail with two pendulum links hanging from it. Click or drag to shove the hinge."
							onpointerdown={down}
							onpointermove={move}
							onpointerup={up}
							onpointercancel={up}
						></canvas>
						{#if upright > 100 && !stageAll}
							<p class="absolute right-3 bottom-2 font-serif text-[13px] text-good italic">
								it's up — now try to knock it over
							</p>
						{/if}
					</div>

					<!-- the pool, two ways: the halls themselves, or what passes
					     between them -->
					{#if !reduced}
						<div class="border-t border-line-soft px-4 pt-2.5 pb-3">
							<span class="eyebrow">
								{#if poolSize === 0}
									the pool — press Train and {wantHalls} independent learners start practising
								{:else if stageAll}
									the swarm — every learner above, and what passes between them below
								{:else}
									the race — each hall its own policy; the stage copies the outlined champion
								{/if}
							</span>

							{#if !stageAll || poolSize === 0}
								<div
									class="mt-2 grid grid-cols-[repeat(auto-fill,minmax(64px,1fr))] gap-1.5 sm:grid-cols-[repeat(auto-fill,minmax(92px,1fr))]"
								>
									{#each Array.from({ length: poolSize || wantHalls }, (_, i) => i) as w (w)}
										{#if poolSize > 0}
											<div
												class="relative overflow-hidden rounded border"
												style="border-color: {w === champView
													? 'var(--accent)'
													: 'var(--line-soft)'};"
											>
												<canvas
													bind:this={miniCanvases[w]}
													class="block aspect-[16/9] w-full"
													aria-label="Practice hall {w + 1}{w === champView
														? ', the current champion'
														: ''}"
												></canvas>
												<span class="num absolute right-1 bottom-0.5 text-[9.5px] text-ink-3">
													{(hallScoreView[w] ?? 0).toFixed(2)}
												</span>
											</div>
										{:else}
											<div
												class="aspect-[16/9] rounded border border-dashed border-line-soft"
											></div>
										{/if}
									{/each}
								</div>
							{:else}
								<div class="relative mt-1">
									<canvas
										bind:this={webCanvas}
										class="block h-[340px] w-full sm:h-[404px] lg:h-[448px]"
										aria-label="Every practice hall drawn as a live pendulum, standing in a field of
										the two skills its fitness is made of: how often it gets the stack to the top at
										all (left to right) and how much of the catch window it then holds (bottom to
										top). The dashed diagonals are lines of equal fitness, labelled — two halls on
										one diagonal are worth the same to the race and got there differently. Each hall
										trails the path it took. Ribbons run from a hall being listened to into the hall
										listening; a halo is attention received; a dashed shell is a hall that has
										learned to take nobody's advice. The warm ring marks the hall whose weights the
										big rig on the rail above is running."
									></canvas>
									<span
										class="num pointer-events-none absolute right-3 bottom-1 text-[10px] text-ink-3"
									>
										delivers →
									</span>
									<span
										class="num pointer-events-none absolute top-1.5 left-2 text-[10px] text-ink-3"
									>
										↑ catches
									</span>
								</div>
								<div
									class="num mt-0.5 flex flex-wrap items-center gap-x-4 gap-y-0.5 text-[10px] text-ink-3"
								>
									<span>diagonals: equal fitness</span>
									<span>warm ring: the rig on the rail</span>
									<span>ribbons: who listens to whom, learned</span>
									<span>halo: attention received</span>
									<span>dashed shell: refusing advice</span>
									{#if lastFind}
										<span style="color: var(--ink-2);">
											hall {lastFind.hall + 1} found a cleaner arrival — shared with {lastFind.to}
										</span>
									{/if}
								</div>
							{/if}
						</div>
					{/if}
				</div>

				<div class="flex flex-col gap-4 border-line-soft px-4 py-4 sm:border-l">
					<div>
						<span class="eyebrow">drill outcomes</span>
						<div class="relative mt-1.5">
							<svg
								viewBox="0 0 {SW} {SH}"
								preserveAspectRatio="none"
								class="block h-16 w-full"
								role="img"
								aria-label="The share of recent catch drills that held the stack to the {DECISIONS}-tick
								ceiling, held it at least two seconds, or dropped it — over a rolling window of {WIN}
								drills. Learning shows as the green band rising."
							>
								<rect x="0" y="0" width={SW} height={SH} fill="var(--surface-2)" opacity="0.5" />
								{#if outcomes}
									<path d={outcomes.some} fill="var(--cat-1)" opacity="0.4" />
									<path d={outcomes.ceiling} fill="var(--good)" opacity="0.75" />
								{/if}
							</svg>
							{#if !outcomes}
								<span
									class="num absolute inset-0 flex items-center justify-center text-[10.5px] text-ink-3"
								>
									{caughtLog.length}/{WIN} drills
								</span>
							{/if}
						</div>
						<div
							class="num mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[10px] text-ink-3"
						>
							<span><span aria-hidden="true" style="color: var(--good);">▬</span> held to 400</span>
							<span><span aria-hidden="true" style="color: var(--cat-1);">▬</span> held 2 s+</span>
							<span>above: dropped</span>
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2 sm:grid-cols-1">
						<div>
							<span class="eyebrow">episodes</span>
							<div class="num mt-0.5 text-[15px] text-ink">
								{episodes}{#if halls > 0}<span class="text-[10.5px] text-ink-3">
										· {halls} threads</span
									>{/if}
							</div>
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
						<div>
							<span
								class="eyebrow"
								title="Clean arrivals one hall discovered and handed to the others to rehearse"
							>
								finds shared
							</span>
							<div class="num mt-0.5 text-[15px] text-ink">{shared}</div>
						</div>
					</div>

					<div class="mt-auto flex flex-col gap-3">
						<Slider
							label="learners"
							tone="knob"
							bind:value={wantHalls}
							min={1}
							max={HALL_MAX}
							step={1}
							format={(v) => `${v}`}
						/>
						<Slider
							label="learning rate"
							tone="knob"
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
