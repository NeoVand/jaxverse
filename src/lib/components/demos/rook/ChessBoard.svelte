<script module lang="ts">
	import type { BoardTone } from './board';

	/** One painted square: a tint, the piece the reader picked up, or a king
	 * in check. Legal destinations are the board's own business. */
	export interface SquareMark {
		square: string;
		kind: 'wash' | 'selected' | 'check';
		tone?: BoardTone;
	}

	export interface MoveArrow {
		from: string;
		to: string;
		tone?: BoardTone;
	}

	/** A chess.js verbose move, narrowed to what the board needs. */
	export interface LegalMove {
		to: string;
		promotion?: string;
	}
</script>

<script lang="ts">
	// The book's chessboard: a thin, reactive skin over cm-chessboard. Every
	// board in the chapter — the one you play on, the arena, and the small
	// diagrams — is this component, so pieces, squares, markers and arrows look
	// the same everywhere and follow the theme.
	//
	// cm-chessboard owns imperative DOM: it is built once in an effect, then
	// told about position, markers and arrows as props change. The handle is a
	// plain field, never $state and never templated.
	import { untrack } from 'svelte';
	import type { Chessboard, MoveInputEvent } from 'cm-chessboard/src/Chessboard.js';
	import { ALL_MARKS, ARROW, BOARD_ASSETS, MARK, loadBoardLib } from './board';

	interface Props {
		/** Full FEN, or just its placement field. */
		fen: string;
		orientation?: 'w' | 'b';
		marks?: SquareMark[];
		arrows?: MoveArrow[];
		/** The side the reader may move; null (the default) is a diagram. */
		input?: 'w' | 'b' | null;
		/** Where a picked-up piece may go — chess.js verbose moves. */
		legalFrom?: (square: string) => LegalMove[];
		/** Commit a move. Return false to refuse it. */
		onMove?: (from: string, to: string) => boolean;
		showCoordinates?: boolean;
		/** Slide the pieces between positions. Off for diagrams that jump. */
		animate?: boolean;
		label?: string;
	}

	let {
		fen,
		orientation = 'w',
		marks = [],
		arrows = [],
		input = null,
		legalFrom,
		onMove,
		showCoordinates = false,
		animate = false,
		label = 'chess position'
	}: Props = $props();

	let host = $state<HTMLDivElement | null>(null);
	let board: Chessboard | null = null;
	let inputTypes: Record<string, string> = {};
	/** Bumped once the board exists, so the sync effects below rerun for it. */
	let built = $state(0);

	function markerFor(m: SquareMark) {
		if (m.kind === 'selected') return MARK.selected;
		if (m.kind === 'check') return MARK.check;
		return MARK.wash[m.tone ?? 'accent'];
	}

	function lift(square: string): boolean {
		const moves = legalFrom?.(square) ?? [];
		if (moves.length === 0) return false;
		board?.addMarker?.(MARK.selected, square);
		board?.addLegalMovesMarkers?.(moves);
		return true;
	}

	function drop(): void {
		board?.removeMarkers?.(MARK.selected);
		board?.removeLegalMovesMarkers?.();
	}

	/** cm-chessboard drives click AND drag through this one callback. */
	function handleInput(event: MoveInputEvent): boolean {
		if (!board) return false;
		if (event.type === inputTypes.moveInputStarted) {
			return event.squareFrom ? lift(event.squareFrom) : false;
		}
		if (event.type === inputTypes.validateMoveInput) {
			// clear here rather than on `moveInputFinished`: that one arrives from
			// an animation promise, and an owner who disables input on the same
			// tick (because it is now the model's turn) never lets it land
			drop();
			const { squareFrom, squareTo } = event;
			const played = Boolean(squareFrom && squareTo && onMove?.(squareFrom, squareTo));
			// refused: the piece stays picked up, so its options must come back
			if (!played && squareFrom) lift(squareFrom);
			return played;
		}
		if (
			event.type === inputTypes.moveInputCanceled ||
			event.type === inputTypes.moveInputFinished
		) {
			drop();
		}
		return true;
	}

	// ── build once, tear down on unmount. Props are read untracked here: a new
	//    FEN must repaint this board, never rebuild it. ──
	$effect(() => {
		const el = host;
		if (!el) return;
		let alive = true;
		void (async () => {
			const lib = await loadBoardLib();
			if (!alive) return;
			const start = untrack(() => ({ fen, orientation, input, showCoordinates, animate }));
			const reduce =
				typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
			const extensions: Array<{ class: unknown; props?: Record<string, unknown> }> = [
				// autoMarkers off: this component paints every marker itself
				{ class: lib.Markers, props: { autoMarkers: null } },
				{ class: lib.Arrows, props: {} }
			];
			if (start.input) {
				// keyboard players and screen readers get a board, not a picture
				extensions.push({
					class: lib.Accessibility,
					props: {
						brailleNotationInAlt: true,
						movePieceForm: true,
						boardAsTable: false,
						piecesAsList: false,
						keyboardMoveInput: true,
						visuallyHidden: true
					}
				});
			}
			inputTypes = lib.INPUT_EVENT_TYPE;
			board = new lib.Chessboard(el, {
				position: start.fen,
				orientation: start.orientation,
				assetsUrl: BOARD_ASSETS,
				style: {
					cssClass: 'jaxverse',
					showCoordinates: start.showCoordinates,
					borderType: lib.BORDER_TYPE.none,
					pieces: {
						type: lib.PIECES_FILE_TYPE.svgSprite,
						file: 'pieces/staunty.svg',
						tileSize: 40
					},
					animationDuration: start.animate && !reduce ? 220 : 0
				},
				extensions
			});
			// A diagram is one image with one label: hide cm-chessboard's own
			// role="img" SVG and let the host carry the caption. A playable board
			// keeps it — the Accessibility extension names and drives it there.
			if (!start.input) el.querySelector('svg.cm-chessboard')?.setAttribute('aria-hidden', 'true');
			built++;
		})();
		return () => {
			alive = false;
			board?.destroy();
			board = null;
		};
	});

	$effect(() => {
		const next = fen;
		const slide = animate;
		void built;
		if (board) void board.setPosition(next, slide);
	});

	$effect(() => {
		const side = orientation;
		void built;
		if (board) void board.setOrientation(side, false);
	});

	$effect(() => {
		const painted = marks;
		const drawn = arrows;
		void built;
		if (!board) return;
		for (const t of ALL_MARKS) board.removeMarkers?.(t);
		board.removeArrows?.();
		for (const m of painted) board.addMarker?.(markerFor(m), m.square);
		for (const a of drawn) board.addArrow?.(ARROW[a.tone ?? 'accent'], a.from, a.to);
	});

	$effect(() => {
		const side = input;
		void built;
		if (!board) return;
		if (side) {
			board.enableMoveInput(handleInput, side);
		} else {
			board.disableMoveInput();
			drop(); // a piece may have been lifted when the turn changed hands
		}
	});
</script>

<!-- a diagram is announced once, here; a playable board is announced by the
     Accessibility extension on the SVG itself, so this host stays silent -->
<div
	bind:this={host}
	class="board-host"
	role={input ? undefined : 'img'}
	aria-label={input ? undefined : label}
></div>

<style>
	.board-host {
		position: relative;
		width: 100%;
		aspect-ratio: 1;
		border-radius: var(--r-1);
		overflow: hidden;
	}
	/* one hairline over the SVG, so the board reads as an object on both papers */
	.board-host::after {
		content: '';
		position: absolute;
		inset: 0;
		border-radius: inherit;
		box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--ink) 14%, transparent);
		pointer-events: none;
	}
</style>
