// The bibliography: every paper the book leans on, and which chapter leans on it.
//
// A citation is a promise that the reader can go and check, so every entry
// here carries a url that lands on something actually readable — the arXiv
// abstract page where the work is on arXiv, an open PDF or a publisher's free
// full text where it is not. Nothing in this file points at a paywall.
//
// Prose cites by id through <Cite>, and the numeral comes from the paper's
// position in its chapter's list below rather than from a number typed into a
// sentence — the same contract plates.ts keeps for figures. Add a citation in
// the middle of a chapter and every later numeral moves with it.

export interface Paper {
	/** Authors as the reference line should read them — "Kingma & Ba". */
	authors: string;
	year: number;
	title: string;
	/** Where it appeared, short: "ICLR 2015", "Nature 323". */
	where: string;
	/** Somewhere the reader can read it without paying. */
	url: string;
	/** One line on why this paper is worth the click. Optional, but earn it. */
	note?: string;
}

export const papers = {
	'robbins-monro-1951': {
		authors: 'Robbins & Monro',
		year: 1951,
		title: 'A Stochastic Approximation Method',
		where: 'Annals of Mathematical Statistics 22(3)',
		url: 'https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full',
		note: 'The proof that you may take your steps from noisy estimates of the slope and still arrive, if the steps shrink at the right rate. Every minibatch since is a corollary.'
	},
	'polyak-1964': {
		authors: 'Polyak',
		year: 1964,
		title: 'Some methods of speeding up the convergence of iteration methods',
		where: 'USSR Computational Mathematics and Mathematical Physics 4(5)',
		url: 'https://www.mathnet.ru/eng/zvmmf7713',
		note: 'Momentum, as a heavy ball rolling on the surface rather than a walker stepping on it.'
	},
	'duchi-2011': {
		authors: 'Duchi, Hazan & Singer',
		year: 2011,
		title: 'Adaptive Subgradient Methods for Online Learning and Stochastic Optimization',
		where: 'JMLR 12',
		url: 'https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf',
		note: 'AdaGrad — the first widely used optimizer to give every parameter its own step size, scaled by the gradients it has seen so far.'
	},
	'tieleman-hinton-2012': {
		authors: 'Tieleman & Hinton',
		year: 2012,
		title: 'Lecture 6.5 — RMSProp',
		where: 'COURSERA: Neural Networks for Machine Learning',
		url: 'https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf',
		note: 'RMSProp was never written up as a paper. It was slide 29 of a lecture, and half the field adopted it anyway.'
	},
	'kingma-ba-2015': {
		authors: 'Kingma & Ba',
		year: 2015,
		title: 'Adam: A Method for Stochastic Optimization',
		where: 'ICLR 2015',
		url: 'https://arxiv.org/abs/1412.6980',
		note: 'Momentum and RMSProp in one update, with a bias correction for the first few steps.'
	},
	'loshchilov-hutter-2019': {
		authors: 'Loshchilov & Hutter',
		year: 2019,
		title: 'Decoupled Weight Decay Regularization',
		where: 'ICLR 2019',
		url: 'https://arxiv.org/abs/1711.05101',
		note: 'AdamW. The finding is subtler than its fame: weight decay folded into the gradient is not the same thing as weight decay applied to the weights, and for Adam the difference is large.'
	},
	'chen-2023': {
		authors: 'Chen et al.',
		year: 2023,
		title: 'Symbolic Discovery of Optimization Algorithms',
		where: 'NeurIPS 2023',
		url: 'https://arxiv.org/abs/2302.06675',
		note: 'Lion, found by a program search over update rules rather than derived by hand.'
	},
	'dauphin-2014': {
		authors: 'Dauphin et al.',
		year: 2014,
		title:
			'Identifying and attacking the saddle point problem in high-dimensional non-convex optimization',
		where: 'NeurIPS 2014',
		url: 'https://arxiv.org/abs/1406.2572',
		note: 'The argument that the folk fear of local minima is misplaced: in many dimensions, the flat places a walker gets stuck at are overwhelmingly saddles, not basins.'
	},
	'goodfellow-2015': {
		authors: 'Goodfellow, Vinyals & Saxe',
		year: 2015,
		title: 'Qualitatively characterizing neural network optimization problems',
		where: 'ICLR 2015',
		url: 'https://arxiv.org/abs/1412.6544',
		note: 'Walk the straight line from a network’s starting weights to its trained ones and the loss falls the whole way. The path training takes is not straight, but it never has to cross a wall.'
	},
	'li-2018': {
		authors: 'Li et al.',
		year: 2018,
		title: 'Visualizing the Loss Landscape of Neural Nets',
		where: 'NeurIPS 2018',
		url: 'https://arxiv.org/abs/1712.09913',
		note: 'What a real loss surface looks like once you slice it honestly — and how much of its apparent shape is an artifact of how you scaled the slice.'
	},
	'cohen-2021': {
		authors: 'Cohen et al.',
		year: 2021,
		title: 'Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability',
		where: 'ICLR 2021',
		url: 'https://arxiv.org/abs/2103.00065',
		note: 'Full-batch gradient descent does not settle into a curvature it can handle. It climbs until the curvature is exactly as sharp as its step size can survive, and then stays there, half-unstable, for the rest of training.'
	}
} as const satisfies Record<string, Paper>;

export type PaperId = keyof typeof papers;

/**
 * Which papers each chapter cites, in the order the prose first reaches for
 * them. Position here is the numeral the reader sees.
 */
export const citationOrder: Partial<Record<string, readonly PaperId[]>> = {
	descent: [
		'polyak-1964',
		'kingma-ba-2015',
		'loshchilov-hutter-2019',
		'chen-2023',
		'duchi-2011',
		'tieleman-hinton-2012',
		'cohen-2021',
		'robbins-monro-1951',
		'dauphin-2014',
		'goodfellow-2015',
		'li-2018'
	]
};

/** 1-based position of a paper in its chapter's list, or undefined. */
export function citationNumber(chapter: string, id: PaperId): number | undefined {
	const i = citationOrder[chapter]?.indexOf(id) ?? -1;
	return i < 0 ? undefined : i + 1;
}

/** The in-page anchor a reference publishes, so a citation can jump to it. */
export function refAnchor(id: string): string {
	return `ref-${id}`;
}

/**
 * What the outbound link is about to open, said plainly: "arXiv:1412.6980" if
 * that is where it goes, otherwise the host. Readers deserve to know whether
 * a click lands on a PDF, a preprint, or a lecture handout.
 */
export function sourceLabel(url: string): string {
	const arxiv = /arxiv\.org\/abs\/([^?#]+)/.exec(url);
	if (arxiv) return `arXiv:${arxiv[1]}`;
	const doi = /doi\.org\/(.+)$/.exec(url);
	if (doi) return `doi:${doi[1]}`;
	return new URL(url).hostname.replace(/^www\./, '');
}
