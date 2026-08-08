// The shared bench for chapter 7: ONE WorkerEngine threads through the whole
// chapter — the weights that pretraining builds are the weights fine-tuning
// bends and RLVR sharpens. Module-level on purpose: five plates live on one
// page and must see one model. The page disposes it on leave.

import { base } from '$app/paths';
import { detectWebGPU } from '$lib/nn/engine';
import { WorkerEngine } from '$lib/llm/worker-engine';
import type { TrainStepMetrics } from '$lib/llm/engine';
import {
	loadRook,
	loadRookManifest,
	loadRookWaypoint,
	type RookData,
	type RookManifest
} from '$lib/data/rook';
import { buildProbeSet, probeLegalRate } from './chess-eval';

export type LabPhase = 'idle' | 'loading' | 'ready' | 'error' | 'no-webgpu';
/** What the resident weights have been through, for honest status lines. */
export type LabStage = 'pretrained' | 'fine-tuned' | 'reinforced';
/** Which corpus the worker currently samples training batches from. */
export type LabCorpus = 'random' | 'sft';
export type LabLoop = '' | 'pretrain' | 'sft' | 'rlvr' | 'arena';
/** The arena's contestants: the frozen pretraining waypoint plus whatever
 * stage snapshots exist. 'current' is always the resident weights. */
export type ArenaStage = 'pretrained' | 'fine-tuned' | 'reinforced';

const SEED = 7;
/** The offline run's rate — pretraining uses it. */
export const LR_PRETRAIN = 1.2e-3;
/** Fine-tuning runs gentler, as fine-tuning does. */
export const LR_SFT = 3e-4;
/** REINFORCE has no KL leash here, so the step must be small — at pretraining
 * rates the policy collapses into repeated tokens within a dozen updates
 * (measured, not theorized). */
export const LR_RL = 1e-4;
const LR = LR_PRETRAIN;

class RookLab {
	phase = $state<LabPhase>('idle');
	stage = $state<LabStage>('pretrained');
	corpus = $state<LabCorpus>('random');
	/** Manifest step of the waypoint the resident weights descend from. */
	waypointStep = $state(2600);
	/** Optimizer steps taken here since that waypoint was loaded (SFT and RLVR
	 * updates included — the worker counts every parameter update). */
	liveSteps = $state(0);
	/** Which plate currently owns a long-running loop. */
	busy = $state<LabLoop>('');
	error = $state('');
	/** Which stage snapshots the arena can field (the buffers themselves are
	 * plain fields below — opaque, never templated). */
	hasSnapshot = $state<{ 'fine-tuned': boolean; reinforced: boolean }>({
		'fine-tuned': false,
		reinforced: false
	});

	readonly weightsLabel = $derived(
		`w${this.waypointStep}${this.liveSteps > 0 ? ` +${this.liveSteps}` : ''}`
	);

	/** Worker handle — a plain field on purpose (opaque, never templated). */
	engine: WorkerEngine | null = null;
	data: RookData | null = null;
	manifest: RookManifest | null = null;
	sftTokens: Uint16Array | null = null;
	/** Position of every <game> token in the pretraining stream (RLVR prefixes). */
	gameStarts: number[] = [];
	/** Fixed real positions for the legal-move probe — the same measurement the
	 * manifest recorded per waypoint, so gauge and chips speak the same number. */
	probeSet: number[][] = [];
	/** The same probe drawn from the GREEDY corpus — the in-distribution check
	 * for the fine-tuned model (drift away from random play is measured
	 * separately, by validation loss on the old corpus). */
	sftProbeSet: number[][] = [];
	/** Generation counter — bumping it cancels every in-flight loop. */
	gen = 0;
	/** Weight snapshots taken as the SFT and RLVR loops pause — what the
	 * arena replays. Kept across dispose() like the other caches: a returning
	 * reader should not lose the students they trained. */
	private stageSnapshots: { 'fine-tuned': ArrayBuffer | null; reinforced: ArrayBuffer | null } = {
		'fine-tuned': null,
		reinforced: null
	};

	private powering: Promise<void> | null = null;
	private inflightTrain: Promise<void> | null = null;

	/** Idempotent power-on; every plate's gate calls this. */
	power(): Promise<void> {
		if (this.phase === 'ready') return Promise.resolve();
		this.powering ??= this.doPower().finally(() => {
			this.powering = null;
		});
		return this.powering;
	}

	private async doPower(): Promise<void> {
		this.phase = 'loading';
		this.error = '';
		try {
			if (!(await detectWebGPU())) {
				this.phase = 'no-webgpu';
				return;
			}
			const [data, manifest] = await Promise.all([loadRook(), loadRookManifest()]);
			this.data = data;
			this.manifest = manifest;
			const starts: number[] = [];
			for (let i = 0; i < data.tokens.length; i++) if (data.tokens[i] === 0) starts.push(i);
			this.gameStarts = starts;
			this.probeSet = buildProbeSet(data.tokens, starts, 32);
			// Start from the last pretraining waypoint so Play/SFT/RLVR are alive
			// immediately; the pretraining plate's time machine can still rewind to step 0.
			const wp = manifest.waypoints[manifest.waypoints.length - 1];
			const flat = await loadRookWaypoint(wp.file, manifest);
			const engine = new WorkerEngine({
				tokenData: data.tokens,
				seed: SEED,
				lr: LR,
				decode: (ids) => ids.map((id) => data.decode(id)).join(' '),
				decodeOne: (id) => data.decode(id),
				stopToken: 0
			});
			this.engine = engine;
			// loadRookWaypoint caches the dequantized array; init transfers its
			// buffer to the worker — hand over a copy so the cache stays alive.
			await engine.init({ name: 'rook', ...manifest.config }, flat.slice().buffer as ArrayBuffer);
			this.waypointStep = wp.step;
			this.liveSteps = 0;
			this.stage = 'pretrained';
			this.corpus = 'random';
			this.lrNow = LR;
			this.phase = 'ready';
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
			this.phase = 'error';
		}
	}

	/** Cancel every loop and wait until the worker is truly quiet: bump the
	 * generation, abort any in-flight train chunk, and await its promise. */
	async quiesce(): Promise<number> {
		this.gen++;
		const g = this.gen;
		if (this.engine) {
			try {
				await this.engine.stop();
			} catch {
				/* worker already gone */
			}
		}
		if (this.inflightTrain) {
			try {
				await this.inflightTrain;
			} catch {
				/* surfaced to its own loop */
			}
		}
		if (this.gen === g) this.busy = '';
		return g;
	}

	/** Claim the engine for a long-running loop (stops whoever had it). */
	async beginLoop(kind: Exclude<LabLoop, ''>): Promise<number> {
		const g = await this.quiesce();
		if (this.gen === g) this.busy = kind;
		return g;
	}

	endLoop(g: number): void {
		if (this.gen === g) this.busy = '';
	}

	/** All train chunks go through here so quiesce() can await the last one. */
	async trainChunk(steps: number, onMetrics: (m: TrainStepMetrics) => void): Promise<void> {
		if (!this.engine) return;
		const p = this.engine.train(steps, onMetrics);
		this.inflightTrain = p;
		try {
			await p;
		} finally {
			if (this.inflightTrain === p) this.inflightTrain = null;
		}
	}

	/** Photograph the resident weights as a stage artifact for the arena.
	 * Called by the SFT and RLVR plates as their loops pause. */
	async captureStage(stage: 'fine-tuned' | 'reinforced'): Promise<void> {
		if (!this.engine) return;
		try {
			this.stageSnapshots[stage] = await this.engine.exportCheckpoint();
			this.hasSnapshot = { ...this.hasSnapshot, [stage]: true };
		} catch {
			/* export raced a dispose — the arena simply won't list this stage */
		}
	}

	/** The arena's one trick: ask several checkpoints for their next-token
	 * distribution over the SAME context. There is only one engine, so this
	 * swaps weights in place and restores the resident set afterwards. The
	 * whole swap-query-restore is registered as the in-flight op that
	 * quiesce() awaits — otherwise a loop starting mid-compare would inherit
	 * an engine wearing some contestant's weights. */
	async compareStages(
		tokens: number[],
		stages: ArenaStage[]
	): Promise<Partial<Record<ArenaStage, Float32Array>> | null> {
		if (!this.engine || !this.manifest) return null;
		const g = await this.beginLoop('arena');
		if (g !== this.gen || !this.engine) return null;
		const run = this.doCompare(tokens, stages);
		this.inflightTrain = run.then(
			() => undefined,
			() => undefined
		);
		try {
			return await run;
		} finally {
			this.inflightTrain = null;
			this.endLoop(g);
		}
	}

	private async doCompare(
		tokens: number[],
		stages: ArenaStage[]
	): Promise<Partial<Record<ArenaStage, Float32Array>> | null> {
		const engine = this.engine;
		const manifest = this.manifest;
		if (!engine || !manifest) return null;
		const resident = await engine.exportCheckpoint();
		const out: Partial<Record<ArenaStage, Float32Array>> = {};
		try {
			for (const stage of stages) {
				let flat: ArrayBuffer | null = null;
				if (stage === 'pretrained') {
					const wp = manifest.waypoints[manifest.waypoints.length - 1];
					flat = (await loadRookWaypoint(wp.file, manifest)).slice().buffer as ArrayBuffer;
				} else {
					const snap = this.stageSnapshots[stage];
					flat = snap ? snap.slice(0) : null;
				}
				if (!flat) continue;
				await engine.loadWeights(flat, { preserveStep: true });
				out[stage] = await engine.nextDistribution(tokens);
			}
		} finally {
			// restore unconditionally — even a cancelled compare must not leave
			// the resident engine wearing a contestant's weights
			try {
				await engine.loadWeights(resident, { preserveStep: true });
			} catch {
				/* worker gone — dispose() owns the cleanup then */
			}
		}
		return out;
	}

	/** Rewind (or fast-forward) the time machine to a manifest waypoint. */
	async loadWaypoint(step: number): Promise<void> {
		if (!this.engine || !this.manifest) return;
		const wp = this.manifest.waypoints.find((w) => w.step === step);
		if (!wp) return;
		await this.quiesce();
		const flat = await loadRookWaypoint(wp.file, this.manifest);
		// same copy-before-transfer dance as init — the cache must survive
		await this.engine.loadWeights(flat.slice().buffer as ArrayBuffer);
		this.waypointStep = wp.step;
		this.liveSteps = 0;
		this.stage = 'pretrained';
	}

	async ensureSftTokens(): Promise<Uint16Array> {
		if (this.sftTokens) return this.sftTokens;
		const res = await fetch(`${base}/data/rook-sft-tokens.bin`);
		if (!res.ok) throw new Error(`rook: sft corpus fetch failed (${res.status})`);
		const tokens = new Uint16Array(await res.arrayBuffer());
		this.sftTokens = tokens;
		const starts: number[] = [];
		for (let i = 0; i < tokens.length; i++) if (tokens[i] === 0) starts.push(i);
		this.sftProbeSet = buildProbeSet(tokens, starts, 32);
		return tokens;
	}

	/** Swap which corpus the worker trains on. Weights and optimizer survive —
	 * that is the whole point: fine-tuning is the same machine, better data. */
	async useCorpus(which: LabCorpus): Promise<void> {
		if (!this.engine || !this.data || this.corpus === which) return;
		const tokens = which === 'sft' ? await this.ensureSftTokens() : this.data.tokens;
		await this.engine.setTokens(tokens);
		this.corpus = which;
	}

	/** Stage learning rates: pretraining at the offline run's 1.2e-3,
	 * fine-tuning at 3e-4 (nudge, don't blast — or the new diet erases the old
	 * competence), RLVR at 1e-4 (REINFORCE with no KL leash collapses at
	 * anything hotter). Changing γ rebuilds Adam's moments, so this is a no-op
	 * unless a stage actually switches. */
	async useLr(lr: number): Promise<void> {
		if (!this.engine || this.lrNow === lr) return;
		await this.engine.setLr(lr);
		this.lrNow = lr;
	}
	private lrNow = LR;

	/** Held-out loss measured against a *specific* corpus, restoring whichever
	 * corpus was resident afterwards. The worker's val batches are seeded, so
	 * successive calls against the same corpus are comparable — a real curve. */
	async valLossOn(which: LabCorpus): Promise<number> {
		if (!this.engine) return NaN;
		const prev = this.corpus;
		try {
			await this.useCorpus(which);
			return await this.engine.valLoss();
		} finally {
			await this.useCorpus(prev);
		}
	}

	/** Sample n whole games from an empty board (the <game> token), for the
	 * sample displays and the capture-rate measurement. Returns null if
	 * interrupted. */
	async sampleGames(n: number, myGen: number): Promise<number[][] | null> {
		const out: number[][] = [];
		for (let i = 0; i < n; i++) {
			if (!this.engine || myGen !== this.gen) return null;
			const r = await this.engine.sample([0], { temperature: 0.7, maxTokens: 48 });
			out.push(r.tokens);
		}
		return out;
	}

	/** The legal-move probe: argmax next move from 32 fixed real positions,
	 * judged by chess.js — comparable across calls AND (for the random set)
	 * with the manifest's recorded waypoint rates. `from` picks which world
	 * the positions come from. Returns null if interrupted. */
	async probeLegal(myGen: number, from: LabCorpus = 'random'): Promise<number | null> {
		const engine = this.engine;
		const data = this.data;
		const set = from === 'sft' ? this.sftProbeSet : this.probeSet;
		if (!engine || !data || set.length === 0) return null;
		return probeLegalRate(
			(t) => engine.nextDistribution(t),
			set,
			data.decode,
			() => myGen === this.gen && this.engine !== null
		);
	}

	/** Back to powered-off; called by the page on leave. */
	dispose(): void {
		this.gen++;
		const engine = this.engine;
		this.engine = null;
		if (engine) void engine.dispose();
		this.phase = 'idle';
		this.stage = 'pretrained';
		this.corpus = 'random';
		this.waypointStep = 2600;
		this.liveSteps = 0;
		this.busy = '';
		this.error = '';
		this.inflightTrain = null;
		this.lrNow = LR;
		// data/manifest/sft caches stay — the loaders memoize them anyway
	}
}

export const lab = new RookLab();
