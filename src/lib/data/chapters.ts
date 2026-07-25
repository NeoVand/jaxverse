// The spine of the book: every chapter, in reading order.
// Slugs are routes (src/routes/<slug>); numbering is part of the design.

export type ChapterSlug =
	'descent' | 'neuron' | 'space' | 'digits' | 'latent' | 'language' | 'reward' | 'rook';

export interface Chapter {
	slug: ChapterSlug;
	/** Chapter numeral as displayed — the prologue is 0. */
	n: number;
	title: string;
	/** The paradigm or idea this chapter carries, shown as the eyebrow. */
	kicker: string;
	/** One-sentence promise, shown on the landing page and in prev/next cards. */
	deck: string;
	/** Rough reading + playing time, minutes. */
	minutes: number;
	/** True once the chapter page exists — unbuilt chapters render as “soon”. */
	live: boolean;
}

export const chapters: Chapter[] = [
	{
		slug: 'descent',
		n: 0,
		title: 'The Descent',
		kicker: 'Optimization',
		deck: 'Learning is falling downhill on a landscape of error. Meet the loss surface, the gradient, and the little step rule the rest of this book repeats.',
		minutes: 8,
		live: true
	},
	{
		slug: 'neuron',
		n: 1,
		title: 'The Approximator',
		kicker: 'Neural networks',
		deck: 'A neuron is a bump of influence; a layer is a sum of bumps. Watch a tiny network sculpt itself into any curve you draw.',
		minutes: 10,
		live: true
	},
	{
		slug: 'space',
		n: 2,
		title: 'Bending Space',
		kicker: 'Representation',
		deck: 'Why does deep learning work at all? Because a network is a smooth deformation of space — watch it untangle spirals until a straight line can tell them apart.',
		minutes: 12,
		live: true
	},
	{
		slug: 'digits',
		n: 3,
		title: 'Telling Things Apart',
		kicker: 'Supervised learning',
		deck: 'Ten thousand handwritten digits, a stack of layers, and a rule for being less wrong. Train a classifier and then look inside it.',
		minutes: 12,
		live: true
	},
	{
		slug: 'latent',
		n: 4,
		title: 'The Hidden Map',
		kicker: 'Representation learning',
		deck: 'Squeeze every digit through two numbers and back. No labels, no answers — yet a map of meaning appears on its own.',
		minutes: 10,
		live: true
	},
	{
		slug: 'language',
		n: 5,
		title: 'The Next Token',
		kicker: 'Self-supervised learning',
		deck: 'Predict what comes next: the simplest game in the world, and the one behind every large language model. Train one, live, on this book’s own words.',
		minutes: 12,
		live: true
	},
	{
		slug: 'reward',
		n: 6,
		title: 'Learning from Reward',
		kicker: 'Reinforcement learning',
		deck: 'No examples, no answers — only consequences. Watch a policy discover a path through a world that only ever says “more” or “less”.',
		minutes: 10,
		live: true
	},
	{
		slug: 'rook',
		n: 7,
		title: 'Rook',
		kicker: 'Everything at once',
		deck: 'One small language model learns chess three ways: pretrained on games, fine-tuned on better ones, then sharpened by verifiable reward. The modern pipeline, in miniature.',
		minutes: 15,
		live: true
	}
];

export const chapterBySlug = new Map<string, Chapter>(chapters.map((c) => [c.slug, c]));

export function neighbors(slug: string): { prev?: Chapter; next?: Chapter } {
	const i = chapters.findIndex((c) => c.slug === slug);
	if (i === -1) return {};
	return { prev: chapters[i - 1], next: chapters[i + 1] };
}
