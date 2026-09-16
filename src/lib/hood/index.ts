// The hidden course: two or three under-the-hood blocks per chapter, keyed by
// slug, each anchored beneath the plate it explains. Content lives in one
// module per chapter so a chapter's code excerpts and its prose travel
// together.

import type { HoodChapter } from './types';
import { home } from './home';
import { descent } from './descent';
import { neuron } from './neuron';
import { space } from './space';
import { digits } from './digits';
import { latent } from './latent';
import { language } from './language';
import { reward } from './reward';
import { taste } from './taste';
import { rook } from './rook';
import { noise } from './noise';
import { flow } from './flow';
import { world } from './world';

export const hood: Record<string, HoodChapter> = {
	home,
	descent,
	neuron,
	space,
	digits,
	latent,
	language,
	reward,
	taste,
	rook,
	noise,
	flow,
	world
};

export type { HoodChapter, HoodBlock, HoodSection, HoodCode } from './types';
