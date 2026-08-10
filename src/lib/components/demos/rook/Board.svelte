<script module lang="ts">
	import type { BoardTone } from './board';

	export interface BoardArrow {
		from: string;
		to: string;
		tone?: BoardTone;
	}
</script>

<script lang="ts">
	// The chapter's playable board: the shared ChessBoard with a rules engine
	// behind it. chess.js isn't reactive — every derivation keys off the `fen`
	// mirror the owner maintains. Picking a piece up (click or drag) asks
	// chess.js where it may go; committing a move is the owner's decision, so
	// this component never mutates the game itself.
	import type { Square } from 'chess.js';
	import type { ChessGame } from './chess-eval';
	import ChessBoard, { type LegalMove, type SquareMark } from './ChessBoard.svelte';

	interface Props {
		chess: ChessGame | null;
		fen: string;
		/** The side the reader may move; null while it is not their turn. */
		input?: 'w' | 'b' | null;
		/** Play the move. Return false to refuse it. */
		onMove?: (from: string, to: string) => boolean;
		lastMove?: { from: string; to: string } | null;
		hoverSq?: { from: string; to: string } | null;
		arrows?: BoardArrow[];
		ariaLabel?: string;
	}

	const {
		chess,
		fen,
		input = null,
		onMove,
		lastMove = null,
		hoverSq = null,
		arrows = [],
		ariaLabel = 'Chess board — you play White, at the bottom'
	}: Props = $props();

	const checkSq = $derived.by(() => {
		void fen;
		if (!chess || !chess.inCheck()) return null;
		const turn = chess.turn();
		for (const row of chess.board()) {
			for (const p of row) if (p && p.type === 'k' && p.color === turn) return p.square as string;
		}
		return null;
	});

	const marks = $derived.by(() => {
		const out: SquareMark[] = [];
		if (lastMove) {
			out.push({ square: lastMove.from, kind: 'wash', tone: 'accent' });
			out.push({ square: lastMove.to, kind: 'wash', tone: 'accent' });
		}
		// a hovered move from the move list reads louder than the last one played
		if (hoverSq) {
			out.push({ square: hoverSq.from, kind: 'wash', tone: 'accent' });
			out.push({ square: hoverSq.from, kind: 'selected' });
			out.push({ square: hoverSq.to, kind: 'wash', tone: 'accent' });
			out.push({ square: hoverSq.to, kind: 'selected' });
		}
		if (checkSq) out.push({ square: checkSq, kind: 'check' });
		return out;
	});

	function legalFrom(sq: string): LegalMove[] {
		void fen;
		if (!chess) return [];
		return chess.moves({ square: sq as Square, verbose: true });
	}
</script>

<ChessBoard
	{fen}
	{marks}
	{arrows}
	{input}
	{legalFrom}
	onMove={(from, to) => onMove?.(from, to) ?? false}
	showCoordinates
	animate
	label={ariaLabel}
/>
