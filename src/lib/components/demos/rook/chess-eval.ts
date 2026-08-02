// Rook's judge: everything here replays model-written moves through chess.js,
// the external ground truth this chapter keeps appealing to. Ported from
// LLMVibes' ChessLab/TrainingLab measurement code, with per-ply verdicts kept
// so the plates can show *which* move broke, not just how many did.

type ChessCtor = (typeof import('chess.js'))['Chess'];
export type ChessGame = InstanceType<ChessCtor>;

let ctorCache: Promise<ChessCtor> | null = null;

/** chess.js is ~70KB — load it lazily, once, on first use. */
export function loadChess(): Promise<ChessCtor> {
	ctorCache ??= import('chess.js').then((m) => m.Chess);
	return ctorCache;
}

/** Filled glyph per piece type — both colors use the filled set and are told
 * apart by CSS color, so boards read correctly on light AND dark themes. */
export const PIECE_GLYPH: Record<string, string> = {
	k: '♚',
	q: '♛',
	r: '♜',
	b: '♝',
	n: '♞',
	p: '♟'
};

/** One replayed ply of a sampled game. */
export interface PlyMark {
	uci: string;
	legal: boolean;
	capture: boolean;
}

/** A photograph of a game at one story moment, for MiniBoard strips. For
 * 'illegal' the fen is the position BEFORE the attempted move — the board the
 * judge was looking at when it said no. */
export interface BoardSnap {
	fen: string;
	/** 1-based ply of the highlighted (or attempted) move. */
	ply: number;
	uci: string;
	/** Filled glyph of the moved piece, '' when unknowable (illegal attempts). */
	glyph: string;
	kind: 'move' | 'capture' | 'illegal' | 'final';
}

/**
 * Replay one sampled game and photograph it at up to three story moments: an
 * early position, a midpoint (the last capture, if the game has one), and
 * where it broke or ended — the chess-book strip the plates render.
 */
export async function snapshotGame(
	tokens: number[],
	decode: (id: number) => string
): Promise<BoardSnap[]> {
	const Chess = await loadChess();
	const board = new Chess();
	const frames: BoardSnap[] = [];
	let illegalSnap: BoardSnap | null = null;
	let n = 0;
	for (const id of tokens) {
		if (id === 0) break; // the model ended the game — the story stops here
		const uci = decode(id);
		n++;
		try {
			const mv = board.move({
				from: uci.slice(0, 2),
				to: uci.slice(2, 4),
				promotion: uci.length > 4 ? uci[4] : undefined
			});
			frames.push({
				fen: board.fen(),
				ply: n,
				uci,
				glyph: PIECE_GLYPH[mv.piece] ?? '',
				kind: mv.captured !== undefined ? 'capture' : 'move'
			});
		} catch {
			illegalSnap = { fen: board.fen(), ply: n, uci, glyph: '', kind: 'illegal' };
			break;
		}
	}
	if (frames.length === 0) return illegalSnap ? [illegalSnap] : [];
	const early = frames[Math.min(3, frames.length - 1)];
	const captures = frames.filter((f) => f.kind === 'capture');
	const mid = captures.length
		? captures[captures.length - 1]
		: frames[Math.floor(frames.length / 2)];
	const closing: BoardSnap = illegalSnap ?? { ...frames[frames.length - 1], kind: 'final' };
	const picked = [early, mid, closing]
		.filter((s, i, arr) => arr.findIndex((x) => x.ply === s.ply && x.kind === s.kind) === i)
		.sort((a, b) => a.ply - b.ply);
	return picked;
}

export interface ReplayReport {
	games: PlyMark[][];
	/** legal / attempted over every ply of every game; null if nothing to score. */
	legalRate: number | null;
	/** captures per legal ply, measured on each game's INTACT segment (before
	 * its first illegal move) — after a reset the board no longer matches the
	 * model's story, so captures there would be measurement noise. Null if no
	 * intact plies exist. */
	captureRate: number | null;
}

/**
 * Replay sampled token streams through real chess rules: what fraction of the
 * model's moves are legal in the position where it played them? On an illegal
 * move, mark it and reset to a fresh board so the rest of the sample is still
 * scored (the measureLegalRate convention from LLMVibes).
 */
export async function replayGames(
	games: number[][],
	decode: (id: number) => string
): Promise<ReplayReport> {
	const Chess = await loadChess();
	let attempted = 0;
	let legal = 0;
	let legalIntact = 0;
	let capturesIntact = 0;
	const marked: PlyMark[][] = [];
	for (const tokens of games) {
		let board = new Chess();
		let intact = true;
		const plies: PlyMark[] = [];
		for (const id of tokens) {
			if (id === 0) {
				// a <game> marker mid-stream: the model started a new game
				board = new Chess();
				intact = true;
				continue;
			}
			const uci = decode(id);
			attempted++;
			try {
				const mv = board.move({
					from: uci.slice(0, 2),
					to: uci.slice(2, 4),
					promotion: uci.length > 4 ? uci[4] : undefined
				});
				legal++;
				const capture = mv.captured !== undefined;
				if (intact) {
					legalIntact++;
					if (capture) capturesIntact++;
				}
				plies.push({ uci, legal: true, capture });
			} catch {
				plies.push({ uci, legal: false, capture: false });
				board = new Chess();
				intact = false;
			}
		}
		marked.push(plies);
	}
	return {
		games: marked,
		legalRate: attempted > 0 ? legal / attempted : null,
		captureRate: legalIntact > 0 ? capturesIntact / legalIntact : null
	};
}

/** Chess-book caption for a snap: "ply 14 · ♞f6 ×" / "ply 9 ✕ e3f4". */
export function snapCaption(s: BoardSnap): string {
	if (s.kind === 'illegal') return `ply ${s.ply} ✕ ${s.uci}`;
	const mv = `${s.glyph}${s.uci.slice(2, 4)}`;
	if (s.kind === 'capture') return `ply ${s.ply} · ${mv} ×`;
	if (s.kind === 'final') return `end · ply ${s.ply} · ${mv}`;
	return `ply ${s.ply} · ${mv}`;
}

export function snapTone(s: BoardSnap): 'accent' | 'warm' | 'bad' {
	return s.kind === 'illegal' ? 'bad' : s.kind === 'capture' ? 'warm' : 'accent';
}

/** One ply of an RLVR rollout. After the first illegal move the board state is
 * lost, so later plies are 'unchecked' — unverifiable rather than wrong. */
export interface RolloutPly {
	uci: string;
	state: 'legal' | 'illegal' | 'unchecked';
}

export interface RolloutScore {
	plies: RolloutPly[];
	/** Continuation plies scored (stops at a <game> token). */
	attempted: number;
	/** Consecutive legal plies from the start of the continuation. */
	legalPlies: number;
	allLegal: boolean;
	/** legalPlies/attempted, +0.5 if the whole rollout held; 0 if nothing to score. */
	reward: number;
	/** The rollout's photograph: the position where it broke, or its end. */
	snap: BoardSnap;
}

/**
 * The verifier for one rollout: replay the (known-legal) prefix, then score
 * the continuation. Reward = fraction of consecutive legal plies before the
 * first illegal one, plus a 0.5 bonus when every attempted ply was legal.
 * An empty continuation earns 0 — ending the game immediately is no evidence.
 */
export async function scoreRollout(
	prefixUci: string[],
	contTokens: number[],
	decode: (id: number) => string
): Promise<RolloutScore> {
	const Chess = await loadChess();
	const board = new Chess();
	for (const uci of prefixUci) {
		board.move({
			from: uci.slice(0, 2),
			to: uci.slice(2, 4),
			promotion: uci.length > 4 ? uci[4] : undefined
		});
	}
	let attempted = 0;
	let legalPlies = 0;
	let broken = false;
	let snap: BoardSnap | null = null;
	let lastUci = prefixUci.length ? prefixUci[prefixUci.length - 1] : '';
	let lastGlyph = '';
	const plies: RolloutPly[] = [];
	for (const id of contTokens) {
		if (id === 0) break; // the model ended the game — stop scoring here
		const uci = decode(id);
		attempted++;
		if (broken) {
			plies.push({ uci, state: 'unchecked' });
			continue;
		}
		try {
			const mv = board.move({
				from: uci.slice(0, 2),
				to: uci.slice(2, 4),
				promotion: uci.length > 4 ? uci[4] : undefined
			});
			legalPlies++;
			lastUci = uci;
			lastGlyph = PIECE_GLYPH[mv.piece] ?? '';
			plies.push({ uci, state: 'legal' });
		} catch {
			broken = true;
			// photograph the board the judge was looking at when it said no
			snap = { fen: board.fen(), ply: attempted, uci, glyph: '', kind: 'illegal' };
			plies.push({ uci, state: 'illegal' });
		}
	}
	snap ??= { fen: board.fen(), ply: attempted, uci: lastUci, glyph: lastGlyph, kind: 'final' };
	const allLegal = attempted > 0 && legalPlies === attempted;
	const reward = attempted === 0 ? 0 : legalPlies / attempted + (allLegal ? 0.5 : 0);
	return { plies, attempted, legalPlies, allLegal, reward, snap };
}

/**
 * A random real opening from the pretraining corpus: the <game> token plus
 * minPlies–maxPlies plies. Corpus games are legal by construction, so the
 * prefix always replays cleanly and a verifier can score from a live board.
 */
export function pickPrefix(
	tokens: Uint16Array,
	gameStarts: number[],
	rand: () => number = Math.random,
	minPlies = 4,
	maxPlies = 10
): number[] {
	for (let tries = 0; tries < 20; tries++) {
		const gi = Math.floor(rand() * gameStarts.length);
		const start = gameStarts[gi];
		const end = gi + 1 < gameStarts.length ? gameStarts[gi + 1] : tokens.length;
		const gameLen = end - start - 1;
		if (gameLen < maxPlies) continue;
		const k = minPlies + Math.floor(rand() * (maxPlies - minPlies + 1));
		const prefix: number[] = [0];
		for (let i = 1; i <= k; i++) prefix.push(tokens[start + i]);
		return prefix;
	}
	return [0];
}

/** Deterministic small RNG (mulberry32) — probe positions must be the SAME
 * across measurements, or the legal-rate curve would be noise, not signal. */
export function seededRand(seed: number): () => number {
	let s = seed;
	return () => {
		s |= 0;
		s = (s + 0x6d2b79f5) | 0;
		let t = Math.imul(s ^ (s >>> 15), 1 | s);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** A fixed set of real positions (1–40 plies into corpus games) for the
 * legal-rate probe — the same measurement the time-machine manifest recorded
 * offline, reproduced live so the gauge and the chips speak the same number. */
export function buildProbeSet(tokens: Uint16Array, gameStarts: number[], n = 32): number[][] {
	const rand = seededRand(4242);
	const out: number[][] = [];
	for (let i = 0; i < n; i++) out.push(pickPrefix(tokens, gameStarts, rand, 1, 40));
	return out;
}

/** The arena's read of one model's next-move beliefs in one position. */
export interface RowVerdict {
	/** Masked argmax — the move this model plays when it must move legally.
	 * Null when the position offers no legal move in the vocabulary. */
	pick: string | null;
	/** The pick's share of the model's total belief (unmasked). */
	pickP: number;
	/** How much probability sat on legal moves before the mask. */
	legalMass: number;
	/** Raw top-N (mask-blind), each judged for legality here. */
	top: Array<{ uci: string; p: number; legal: boolean }>;
}

/**
 * Grade one next-token log-prob row against a position's legal moves: the
 * masked argmax (a deterministic, comparable "decision"), the legal mass,
 * and the raw top-N. Shared by the arena's stage-vs-stage columns.
 */
export function analyzeRow(
	row: Float32Array,
	legalUci: string[],
	idOf: Map<string, number>,
	moves: string[],
	topN = 3
): RowVerdict {
	const probs = new Float64Array(row.length);
	let total = 0;
	for (let i = 0; i < row.length; i++) {
		probs[i] = Math.exp(row[i]);
		total += probs[i];
	}
	const legalIds = new Set<number>();
	let legalMass = 0;
	let pick: string | null = null;
	let pickP = -1;
	for (const uci of legalUci) {
		const id = idOf.get(uci);
		const p = id !== undefined ? probs[id] / total : 0;
		legalMass += p;
		if (id !== undefined) legalIds.add(id);
		if (p > pickP && id !== undefined) {
			pickP = p;
			pick = uci;
		}
	}
	// a vocabulary hole leaves every legal move unrepresented — no pick then
	if (pick === null && legalUci.length > 0) {
		pick = legalUci[0] ?? null;
		pickP = 0;
	}
	const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
	const top = order.slice(0, topN).map((i) => ({
		uci: i === 0 ? '⟨game⟩' : moves[i - 1],
		p: probs[i] / total,
		legal: legalIds.has(i)
	}));
	return { pick, pickP: Math.max(pickP, 0), legalMass, top };
}

/**
 * The manifest's legal-move metric, live: from each probe position (a real
 * game prefix — teacher-forced, so the model's own mistakes never compound),
 * take the model's argmax next move and ask the judge if it is legal here.
 * Returns legal/n, or null if interrupted via `alive`.
 */
export async function probeLegalRate(
	nextDist: (tokens: number[]) => Promise<Float32Array>,
	probeSet: number[][],
	decode: (id: number) => string,
	alive: () => boolean = () => true
): Promise<number | null> {
	const Chess = await loadChess();
	let legal = 0;
	for (const prefix of probeSet) {
		if (!alive()) return null;
		const board = new Chess();
		for (let k = 1; k < prefix.length; k++) {
			const uci = decode(prefix[k]);
			board.move({
				from: uci.slice(0, 2),
				to: uci.slice(2, 4),
				promotion: uci.length > 4 ? uci[4] : undefined
			});
		}
		const row = await nextDist(prefix);
		if (!alive()) return null;
		let best = 0;
		let bv = -Infinity;
		for (let v = 0; v < row.length; v++) {
			if (row[v] > bv) {
				bv = row[v];
				best = v;
			}
		}
		if (best === 0) continue; // predicting <game> mid-game is not a legal move
		const uci = decode(best);
		try {
			board.move({
				from: uci.slice(0, 2),
				to: uci.slice(2, 4),
				promotion: uci.length > 4 ? uci[4] : undefined
			});
			legal++;
		} catch {
			/* illegal here — that is the measurement */
		}
	}
	return legal / probeSet.length;
}
