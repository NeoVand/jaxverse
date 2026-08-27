// The under-the-hood blocks: a second book hiding inside the first — a
// progressive course in jax-js. Each chapter contributes two or three blocks,
// and each block sits directly beneath the plate it explains, collapsed to a
// single quiet line until the reader wants the code. Blocks carry the real
// code quoted from this repository, with a separate tab for the stagecraft so
// the ML stays readable for the reader who came for the ML.

export interface HoodCode {
	/** The code itself, verbatim or lightly condensed from the repo. */
	code: string;
	/** Where the reader can find it — a path in this repository. */
	file?: string;
	lang?: 'ts' | 'sh' | 'json';
}

export interface HoodSection {
	title: string;
	/** One or two short paragraphs of HTML (inline tags only). */
	body: string;
	code?: HoodCode;
}

export interface HoodBlock {
	/** Anchor id, referenced from the chapter page: <UnderTheHood block="…" /> */
	id: string;
	/**
	 * What this block teaches, with no numeral in it: "params as a tree,
	 * valueAndGrad, jit". The "Lesson 4b — " in front is composed at render
	 * time from the chapter's own number and the block's place in it, so
	 * inserting a chapter or a block renumbers the whole course at once — the
	 * contract <ChapterRef> and <PlateRef> already keep for the visible book.
	 */
	lesson: string;
	/** Optional invitation paragraph shown as soon as the hood opens (HTML). */
	lede?: string;
	/** The machine-learning tab. */
	ml: HoodSection[];
	/** The interface tab — the stagecraft. Empty hides the tab. */
	ui: HoodSection[];
	/** Downloadable standalone lab; rides on the chapter's final block. */
	lab?: {
		/** Zip filename under /labs/. */
		file: string;
		/** One sentence: what the lab does when it runs. */
		note: string;
	};
}

export interface HoodChapter {
	slug: string;
	blocks: HoodBlock[];
}
