// The under-the-hood sections: one per chapter, collapsed by default, and a
// second book hiding inside the first — a progressive course in jax-js. Each
// chapter's entry teaches the JAX ideas its plates actually run on, with the
// real code quoted from this repository, and a separate tab for the stagecraft
// so the ML stays readable for the reader who came for the ML.

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

export interface HoodChapter {
	slug: string;
	/** The lesson this chapter contributes to the jax-js course. */
	lesson: string;
	/** The invitation paragraph shown as soon as the hood opens (HTML). */
	lede: string;
	/** The machine-learning tab. */
	ml: HoodSection[];
	/** The interface tab — the stagecraft. */
	ui: HoodSection[];
	/** Downloadable standalone lab, if the chapter ships one. */
	lab?: {
		/** Zip filename under /labs/. */
		file: string;
		/** One sentence: what the lab does when it runs. */
		note: string;
	};
}
