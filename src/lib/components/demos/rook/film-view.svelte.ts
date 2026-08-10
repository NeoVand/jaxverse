// One sampled game, prepared for display. Both training plates show the model's
// own writing the same way — a board you scrub by pointing at a move, and the
// sentence that produced it — so the derivations live here rather than twice.
//
// Call it once during component init and read the getters in the template.

import type { FilmPly, GameFilm } from './chess-eval';
import type { BoardTone } from './board';

/** An early break leaves a long unverifiable tail — at step 0 that is nearly the
 * whole sample. Show this many, then count the rest out loud. */
const TAIL = 10;

export interface FilmView {
	readonly shown: FilmPly | null;
	readonly playable: FilmPly[];
	readonly sentence: FilmPly[];
	readonly hiddenTail: number;
	readonly tone: BoardTone;
	readonly color: string;
	readonly caption: string;
	readonly verdict: string;
	/** The model wrote the ⟨game⟩ marker instead of another move. */
	readonly ended: boolean;
	point(ply: number | null): void;
}

export function filmView(film: () => GameFilm | null): FilmView {
	/** The ply the reader is pointing at; null falls back to the game's own end. */
	let hovered = $state<number | null>(null);

	// Plies with a board behind them: everything up to and including the move the
	// judge refused. Past that the board is lost, so nothing is pointable.
	const playable = $derived((film()?.plies ?? []).filter((p) => p.state !== 'unchecked'));
	/** Where the game's own story ends — the refused move, or the last legal one. */
	const storyPly = $derived(playable.length ? playable[playable.length - 1].ply : null);
	const shown = $derived(playable.find((p) => p.ply === (hovered ?? storyPly)) ?? null);

	const tone = $derived<BoardTone>(
		shown?.state === 'illegal' ? 'bad' : shown?.capture ? 'warm' : 'accent'
	);
	const color = $derived(
		shown?.state === 'illegal' ? 'var(--bad)' : shown?.capture ? 'var(--warm)' : 'var(--ink-2)'
	);

	const caption = $derived.by(() => {
		if (!shown) return '';
		if (shown.state === 'illegal') return `ply ${shown.ply} ✕ ${shown.uci} — the judge said no`;
		const piece = shown.glyph ? `${shown.glyph} ` : '';
		return `ply ${shown.ply} · ${piece}${shown.uci}${shown.capture ? ' ×' : ''}`;
	});

	const sentence = $derived.by(() => {
		const f = film();
		const all = f?.plies ?? [];
		return f?.brokeAt == null ? all : all.slice(0, f.brokeAt + TAIL);
	});
	const hiddenTail = $derived((film()?.plies.length ?? 0) - sentence.length);

	const verdict = $derived.by(() => {
		const f = film();
		if (!f) return '';
		const n = f.plies.length;
		if (n === 0) return 'The model closed the game before playing a move.';
		if (f.brokeAt === null) {
			return f.ended
				? `${n} plies, every one legal — and the model closed the game itself.`
				: `${n} plies, every one legal; the sample ran out of room before the model ran out of chess.`;
		}
		const kept = f.brokeAt - 1;
		const rest = n - f.brokeAt;
		const broke = f.plies[f.brokeAt - 1].uci;
		const head =
			kept === 0
				? `The very first move, ✕ ${broke}, is already illegal in the opening position.`
				: `${kept} legal ${kept === 1 ? 'ply' : 'plies'}, then ✕ ${broke} — illegal in the position where it was played.`;
		return rest > 0
			? `${head} It kept writing; those ${rest} tokens are greyed, because once a move is refused the board no longer matches the model's story and nothing later can be checked.`
			: head;
	});

	return {
		get shown() {
			return shown;
		},
		get playable() {
			return playable;
		},
		get sentence() {
			return sentence;
		},
		get hiddenTail() {
			return hiddenTail;
		},
		get tone() {
			return tone;
		},
		get color() {
			return color;
		},
		get caption() {
			return caption;
		},
		get verdict() {
			return verdict;
		},
		get ended() {
			return film()?.ended ?? false;
		},
		point(ply: number | null) {
			hovered = ply;
		}
	};
}
