// Types for cm-chessboard (shipped as plain ES modules, no declarations).
// Only the surface this book uses is described — enough to keep the board
// wrapper under `strict` without an `any` in sight.
//
// Note on the extension methods: `addMarker`, `addArrow` and friends are
// attached to the instance at runtime by the Markers and Arrows extensions.
// They are declared here as optional members, so a board built without those
// extensions still type-checks at the call site.

declare module 'cm-chessboard/src/Chessboard.js' {
	export const COLOR: { readonly white: 'w'; readonly black: 'b' };
	export const BORDER_TYPE: {
		readonly none: 'none';
		readonly thin: 'thin';
		readonly frame: 'frame';
	};
	export const PIECES_FILE_TYPE: { readonly svgSprite: 'svgSprite' };
	export const INPUT_EVENT_TYPE: {
		readonly moveInputStarted: 'moveInputStarted';
		readonly movingOverSquare: 'movingOverSquare';
		readonly validateMoveInput: 'validateMoveInput';
		readonly moveInputCanceled: 'moveInputCanceled';
		readonly moveInputFinished: 'moveInputFinished';
	};
	export const POINTER_EVENTS: {
		readonly pointerdown: 'pointerdown';
		readonly pointerup: 'pointerup';
		readonly pointermove: 'pointermove';
	};
	export const FEN: { readonly start: string; readonly empty: string };

	export interface MarkerType {
		class: string;
		slice: string;
		position?: 'above';
	}
	export interface ArrowType {
		class: string;
	}

	export interface MoveInputEvent {
		type: string;
		squareFrom?: string;
		squareTo?: string;
		piece?: string;
	}

	export interface SquareSelectEvent {
		square: string;
	}

	export interface ExtensionEntry {
		class: unknown;
		props?: Record<string, unknown>;
	}

	export interface ChessboardProps {
		position?: string;
		orientation?: 'w' | 'b';
		responsive?: boolean;
		assetsUrl?: string;
		assetsCache?: boolean;
		style?: {
			cssClass?: string;
			showCoordinates?: boolean;
			borderType?: 'none' | 'thin' | 'frame';
			aspectRatio?: number;
			pieces?: { type?: string; file?: string; tileSize?: number };
			animationDuration?: number;
		};
		extensions?: ExtensionEntry[];
	}

	export class Chessboard {
		constructor(context: HTMLElement, props?: ChessboardProps);
		setPosition(fen: string, animated?: boolean): Promise<void>;
		getPosition(): string;
		setOrientation(color: 'w' | 'b', animated?: boolean): Promise<void>;
		getOrientation(): 'w' | 'b';
		getPiece(square: string): string | null;
		enableMoveInput(handler: (event: MoveInputEvent) => boolean | void, color?: 'w' | 'b'): void;
		disableMoveInput(): void;
		enableSquareSelect(eventType: string, handler: (event: SquareSelectEvent) => void): void;
		disableSquareSelect(eventType: string): void;
		destroy(): void;
		/** Markers extension. */
		addMarker?: (type: MarkerType, square: string) => void;
		removeMarkers?: (type?: MarkerType, square?: string) => void;
		/** Markers extension — batched, and picks dot vs bevel by occupancy. */
		addLegalMovesMarkers?: (moves: Array<{ to: string; promotion?: string }>) => void;
		removeLegalMovesMarkers?: () => void;
		/** Arrows extension. */
		addArrow?: (type: ArrowType, from: string, to: string) => void;
		removeArrows?: (type?: ArrowType, from?: string, to?: string) => void;
	}
}

declare module 'cm-chessboard/src/extensions/markers/Markers.js' {
	export class Markers {}
	export const MARKER_TYPE: Record<string, { class: string; slice: string; position?: 'above' }>;
}

declare module 'cm-chessboard/src/extensions/arrows/Arrows.js' {
	export class Arrows {}
	export const ARROW_TYPE: Record<string, { class: string }>;
}

declare module 'cm-chessboard/src/extensions/accessibility/Accessibility.js' {
	export class Accessibility {}
}
