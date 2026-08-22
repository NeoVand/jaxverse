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
	let hallEMAs: { deliverEMA: number; drillEMA: number; rewardEMA: number }[] = [];
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
		hallEMAs.push({ deliverEMA: 0, drillEMA: 0, rewardEMA: 0 });
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
			for (const a of [hallAlpha, hallScore, hallSlow, hallRise, hallFindAt, pullK, nodeX, nodeY])
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
			rewardEMA: number;
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
		hallEMAs[w] = { deliverEMA: m.deliverEMA, drillEMA: m.drillEMA, rewardEMA: m.rewardEMA };
		hallTheta[w].set(m.theta);
		// fitness against the hall's own slow average: `rise` is the spike a
		// hall shows in the seconds after it finds something, and it is what
		// the others' attention actually follows
		hallSlow[w] += RISE_EMA * (m.score - hallSlow[w]);
		hallRise[w] = Math.max(0, m.score - hallSlow[w]);
		pendingEpisodes += m.episodes;
		for (const c of m.drills) pendingDrills.push(c);
		if (m.social) attn[w].set(m.social);
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
			const rEMA = new Float64Array(K);
			const wEMA = new Float64Array(K);
			for (let c = 0; c < K; c++) {
				thetas.set(hallTheta[cand[c]], c * N_PARAMS);
				pr[c] = prior[w][cand[c]];
				dEMA[c] = hallEMAs[cand[c]].deliverEMA;
				rEMA[c] = hallEMAs[cand[c]].drillEMA;
				wEMA[c] = hallEMAs[cand[c]].rewardEMA;
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
				drillEMAs: rEMA,
				rewardEMAs: wEMA,
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
		view = [];
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

	// ── the influence lane ──
	// Every hall is a living portrait, not a dot: the same rig it is running
	// on the stage above, drawn small, standing at the place on the fitness
	// axis it has earned — laggards left, leader right, so the geometry
	// carries the rule. Under them runs the wire. Discoveries race along it
	// as sparks, in the finder's colour, to every other hall at once; the
	// slow curves dipping below it are weights seeping downhill, one curve
	// per hall a laggard is listening to, thickness its attention. Nothing
	// here is decoration — the curves are the rows `mixRound` posted, and
	// the sparks are actual `postMessage`s leaving one worker for five.
	let nodeX: number[] = [];
	let nodeY: number[] = [];
	let hallPath: number[][] = []; // each hall's wake through the skill field
	let lastPathAt = 0;
	let view: number[] = []; // the field's own slow camera: [x0, x1, y0, y1]

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
		const slot = Math.max(9, Math.min(46, H * 0.105, Math.sqrt((W * H) / (n * 16)) / 2));
		const rigScale = slot * 1.25;
		const padL = 44;
		const padR = 34;
		const padT = 40;
		const padB = 34;

		// ── the field: two skills, not one number ──
		// Fitness is a sum, and a sum hides which half a hall has. These are
		// the two halves, and they are learned in a definite order: getting the
		// stack up at all (across) and keeping it there once it arrives (up).
		// Height is REWARD — what the hall is actually paid per tick, averaged
		// over everything it practises. It was the delivery rate first, and
		// that was wrong in a way worth recording: a rate pins at 1.0 the
		// moment a hall can swing up every time, so a hall that then went on
		// to master the catch — the hard half — sat at exactly the same height
		// as one that could only arrive and drop it. Reward never saturates
		// while the policy is still improving, so the summit of the field is
		// the hall standing still at the top of its stack.
		//
		// Across is the delivery rate, which does saturate, and should: once
		// everyone can get it up there, the horizontal axis compresses and
		// the vertical one carries the story.
		//
		// The frame follows the pool. Both skills saturate — six trained halls
		// all deliver nearly every time — so a fixed 0…1 field would end the
		// run as six rigs stacked in one corner with the interesting differences
		// squeezed to nothing. The view tracks the swarm's own spread instead,
		// slowly enough to feel like drift rather than a jump, and the dashed
		// lines of equal fitness keep the absolute ranking legible underneath.
		let dxLo = Infinity;
		let dxHi = -Infinity;
		let dyLo = Infinity;
		let dyHi = -Infinity;
		for (let w = 0; w < n; w++) {
			const e = hallEMAs[w] ?? { deliverEMA: 0, drillEMA: 0 };
			dxLo = Math.min(dxLo, e.deliverEMA);
			dxHi = Math.max(dxHi, e.deliverEMA);
			dyLo = Math.min(dyLo, e.rewardEMA);
			dyHi = Math.max(dyHi, e.rewardEMA);
		}
		const frame = (lo: number, hi: number, minSpan: number): [number, number] => {
			const c = (lo + hi) / 2;
			const half = Math.max(minSpan, (hi - lo) * 0.72);
			return [c - half, c + half];
		};
		const want = [...frame(dxLo, dxHi, 0.12), ...frame(dyLo, dyHi, 0.25)];
		for (let i = 0; i < 4; i++)
			view[i] = view[i] === undefined ? want[i] : view[i] + 0.02 * (want[i] - view[i]);
		const fx = (u: number) =>
			padL + ((u - view[0]) / Math.max(1e-6, view[1] - view[0])) * (W - padL - padR);
		const fy = (v: number) =>
			H - padB - ((v - view[2]) / Math.max(1e-6, view[3] - view[2])) * (H - padT - padB);

		// faint rules across the reward axis: no numbers, because the camera
		// moves — they are there to make a hall's climb legible against
		// something, and to show the pool spreading vertically as it learns
		ctx.strokeStyle = tk.lineSoft;
		ctx.lineWidth = 1;
		ctx.setLineDash([2, 6]);
		ctx.globalAlpha = 0.75;
		for (let i = 1; i <= 4; i++) {
			const y = fy(view[2] + ((view[3] - view[2]) * i) / 5);
			ctx.beginPath();
			ctx.moveTo(padL - 4, y);
			ctx.lineTo(W - padR + 16, y);
			ctx.stroke();
		}
		ctx.setLineDash([]);
		ctx.globalAlpha = 1;

		// the two walls of the field — a frame, not a scale: the numbers on
		// these axes move with the camera, and it is the pool's shape that is
		// being read, not a reading off a ruler
		ctx.strokeStyle = tk.lineSoft;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(padL - 10, padT - 16);
		ctx.lineTo(padL - 10, H - padB + 8);
		ctx.lineTo(W - padR + 16, H - padB + 8);
		ctx.stroke();

		// where each hall stands
		const tx: number[] = [];
		const ty: number[] = [];
		for (let w = 0; w < n; w++) {
			const e = hallEMAs[w] ?? { deliverEMA: 0, drillEMA: 0 };
			tx[w] = fx(e.deliverEMA);
			ty[w] = fy(e.rewardEMA);
		}
		// a few rounds of gentle repulsion so two halls with the same skills
		// stand beside each other rather than inside each other
		const minD = 2 * slot + 8;
		for (let it = 0; it < 4; it++) {
			for (let a = 0; a < n; a++)
				for (let b = a + 1; b < n; b++) {
					let dx = tx[b] - tx[a];
					let dy = ty[b] - ty[a];
					const d = Math.hypot(dx, dy);
					if (d >= minD) continue;
					if (d < 0.01) {
						dx = (a - b) * 0.7;
						dy = 1;
					}
					const k = ((minD - d) / (d || 1)) * 0.5;
					tx[a] -= dx * k;
					ty[a] -= dy * k;
					tx[b] += dx * k;
					ty[b] += dy * k;
				}
			for (let w = 0; w < n; w++) {
				tx[w] = Math.max(slot + 4, Math.min(W - slot - 4, tx[w]));
				ty[w] = Math.max(padT - 6, Math.min(H - 10, ty[w]));
			}
		}
		for (let w = 0; w < n; w++) {
			nodeX[w] = nodeX[w] === undefined ? tx[w] : nodeX[w] + 0.07 * (tx[w] - nodeX[w]);
			nodeY[w] = nodeY[w] === undefined ? ty[w] : nodeY[w] + 0.07 * (ty[w] - nodeY[w]);
		}

		// ── the wake each hall leaves through the field ──
		// Sampled slowly and only when it has actually moved, then drawn
		// through the midpoints as one smooth curve: what matters is the
		// SHAPE of a learning history — right first, then up, and the long
		// slide back when a hall loses the catch — not the frame-to-frame
		// tremble of an EMA.
		if (now - lastPathAt > 420) {
			lastPathAt = now;
			for (let w = 0; w < n; w++) {
				const p = hallPath[w] ?? (hallPath[w] = []);
				const k = p.length;
				if (k && Math.hypot(nodeX[w] - p[k - 2], nodeY[w] - p[k - 1]) < 3) continue;
				p.push(nodeX[w], nodeY[w]);
				if (p.length > 120) p.splice(0, p.length - 120);
			}
		}
		for (let w = 0; w < n; w++) {
			const p = hallPath[w];
			const pts = p ? p.length / 2 : 0;
			if (!p || pts < 3) continue;
			ctx.strokeStyle = hue(tk, w);
			ctx.lineWidth = 1.3;
			for (let i = 1; i < pts - 1; i++) {
				ctx.globalAlpha = 0.34 * (i / pts) ** 1.4;
				ctx.beginPath();
				ctx.moveTo((p[2 * i - 2] + p[2 * i]) / 2, (p[2 * i - 1] + p[2 * i + 1]) / 2);
				ctx.quadraticCurveTo(
					p[2 * i],
					p[2 * i + 1],
					(p[2 * i] + p[2 * i + 2]) / 2,
					(p[2 * i + 1] + p[2 * i + 3]) / 2
				);
				ctx.stroke();
			}
			// and a last stroke into where it stands now
			ctx.globalAlpha = 0.34;
			ctx.beginPath();
			ctx.moveTo((p[2 * pts - 4] + p[2 * pts - 2]) / 2, (p[2 * pts - 3] + p[2 * pts - 1]) / 2);
			ctx.quadraticCurveTo(p[2 * pts - 2], p[2 * pts - 1], nodeX[w], nodeY[w]);
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
				const px = -dy / len;
				const py = dx / len;
				const wS = 0.8 + 4 * a; // wide where it leaves…
				const wT = 0.7; // …a filament where it lands
				ctx.fillStyle = hue(tk, j);
				// thinner as the pool grows: thirty halls all looking at the same
				// two leaders is a wash at the alpha six halls need
				ctx.globalAlpha = Math.min(0.7, (0.08 + 0.4 * a) * fade * (1 + 1.6 * flare[j]));
				ctx.beginPath();
				ctx.moveTo(x0 + px * wS, y0 + py * wS);
				ctx.quadraticCurveTo(cx + px * wT, cy + py * wT, x1 + px * wT, y1 + py * wT);
				ctx.lineTo(x1 - px * wT, y1 - py * wT);
				ctx.quadraticCurveTo(cx - px * wT, cy - py * wT, x0 - px * wS, y0 - py * wS);
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
			// weights pointing at this hall. Nobody assigns it; it is what five
			// independent social learners decided, and the hall paying off best is
			// the one wearing the most light.
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
			ctx.strokeStyle = champ ? tk.accent : tk.lineSoft;
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
	caption="A pool of independent learners, one problem, and a second thing being learned on top of it: who to listen to. Each small stage below is its own policy on its own background thread — press Swarm and they all step onto the one rail, a colour each, weaving as many ideas about how to pump as there are halls, with your own rig in white among them. Discoveries are broadcast: when any hall lands an arrival cleaner than it has ever managed, that state goes to every other hall as something to practise from, and the gradient of the episode that found it goes to the halls behind. But whether to take anyone’s advice is not imposed — every hall runs a second, tiny REINFORCE learner whose actions are &ldquo;take that hall’s weights a step&rdquo; and &ldquo;refuse everyone&rdquo;, and whose only reward is whether it climbed the pool’s ranking afterwards. The field underneath draws what they decided: each hall stands where its delivery rate (across) and its reward per tick (up) put it, the ribbons are the weights they trained on each other, the halo is attention received — the hall paying off best wears the most light — and a dashed shell is a hall that has learned to take nobody’s advice at all. It usually learns that first at the front, which is the honest answer: a race is insurance, and influence spends it. Take the learners slider up and the pool becomes a crowd, arriving at zero and deciding for itself whom to follow — then shove the hinge, and watch every one of them fight back."
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
										class="block h-[300px] w-full sm:h-[272px]"
										aria-label="Every practice hall as a live pendulum, standing in a field of two
										skills: how often it gets the stack to the top at all (left to right) and how much
										reward it earns per tick (bottom to top). Faint diagonals are lines of equal fitness. Each
										hall trails the path it took to get where it stands. Sparks flying between them are
										one hall's discovery being handed to all the others; the slow curves are laggards
										drifting toward the weights of the halls ahead."
									></canvas>
									<span
										class="num pointer-events-none absolute bottom-1 left-11 text-[10px] text-ink-3"
									>
										delivers →
									</span>
									<span
										class="num pointer-events-none absolute top-1.5 left-2 text-[10px] text-ink-3"
									>
										↑ reward
									</span>
								</div>
								<div
									class="num mt-0.5 flex flex-wrap items-center gap-x-4 gap-y-0.5 text-[10px] text-ink-3"
								>
									<span>ribbons: who listens to whom, learned</span>
									<span>halo: attention received</span>
									<span>dashed shell: refusing advice</span>
									<span>trails: how each hall got here</span>
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
