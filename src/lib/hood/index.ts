// The hidden course: one under-the-hood entry per chapter, keyed by slug.
// Content lives in one module per chapter so a chapter's code excerpts and
// its prose travel together.

import type { HoodChapter } from './types';
import { descent } from './descent';
import { neuron } from './neuron';
import { space } from './space';
import { digits } from './digits';
import { latent } from './latent';
import { language } from './language';
import { reward } from './reward';
import { rook } from './rook';

export const hood: Record<string, HoodChapter> = {
	descent,
	neuron,
	space,
	digits,
	latent,
	language,
	reward,
	rook
};

export type { HoodChapter, HoodSection, HoodCode } from './types';
