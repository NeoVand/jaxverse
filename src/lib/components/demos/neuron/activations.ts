/**
 * The activation catalog — one source of truth for every σ this chapter
 * draws: the one-neuron plate's switcher, the activation atlas's field guide, and the neuron circuit
 * all read from here, so a curve, its formula, and its story never drift.
 *
 * Derivatives are closed-form where they are one line, numeric (central
 * difference) where the closed form would just restate the function.
 */

export type ActivationId =
	'sigmoid' | 'tanh' | 'relu' | 'lrelu' | 'elu' | 'softplus' | 'gelu' | 'silu' | 'mish';

export interface ActivationSpec {
	id: ActivationId;
	/** Tile / chip heading. */
	label: string;
	/** Short name used inside formulas, e.g. `a = v·gelu(wx + b)`. */
	code: string;
	/** The year beside the name in the field guide. For sigmoid that is the
	 *  logistic curve's own — Verhulst's population mathematics, a century
	 *  before anyone built a network. For the rest it is the year the function
	 *  arrived in one. */
	year: string;
	/** Compact KaTeX formula for the field-guide tile. */
	tex: string;
	/** One-line provenance — where this activation lives today. */
	note: string;
	fn(z: number): number;
	dfn(z: number): number;
	/** Horizontal asymptotes of σ itself (the one-neuron plate draws them at v·level). */
	levels?: number[];
}

const sig = (z: number) => 1 / (1 + Math.exp(-z));
const softplusFn = (z: number) => (z > 30 ? z : Math.log(1 + Math.exp(z)));
const numericD = (f: (z: number) => number) => (z: number) => (f(z + 1e-3) - f(z - 1e-3)) / 2e-3;

const geluFn = (z: number) => 0.5 * z * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z * z * z)));
const mishFn = (z: number) => z * Math.tanh(softplusFn(z));

/** Field-guide order: the two classics, the rectifier family, the modern
 *  self-gated three that run today's language models. */
export const ACTIVATIONS: ActivationSpec[] = [
	{
		id: 'sigmoid',
		label: 'Sigmoid',
		code: 'sigmoid',
		year: '1845',
		tex: '1/(1+e^{-z})',
		note: 'squashes to (0, 1) — the original; survives as the gate in LSTMs and GLUs',
		fn: sig,
		dfn: (z) => {
			const s = sig(z);
			return s * (1 - s);
		},
		levels: [0, 1]
	},
	{
		id: 'tanh',
		label: 'Tanh',
		code: 'tanh',
		year: '1989',
		tex: '\\tanh z',
		note: 'the zero-centered squash — the gentle default this chapter starts with',
		fn: Math.tanh,
		dfn: (z) => {
			const t = Math.tanh(z);
			return 1 - t * t;
		},
		levels: [-1, 1]
	},
	{
		id: 'relu',
		label: 'ReLU',
		code: 'relu',
		year: '2010',
		tex: '\\max(0,\\,z)',
		note: 'the crease that made deep nets train — cheap, sharp, occasionally dead',
		fn: (z) => Math.max(0, z),
		dfn: (z) => (z > 0 ? 1 : 0)
	},
	{
		id: 'lrelu',
		label: 'Leaky ReLU',
		code: 'lrelu',
		year: '2013',
		tex: '\\max(0.1z,\\,z)',
		note: 'a trickle of slope below zero keeps dead units trainable',
		fn: (z) => (z > 0 ? z : 0.1 * z),
		dfn: (z) => (z > 0 ? 1 : 0.1)
	},
	{
		id: 'elu',
		label: 'ELU',
		code: 'elu',
		year: '2015',
		tex: '\\max(z,0)+\\min(e^{z}{-}1,\\,0)',
		note: 'exponential below zero: smooth, and its negative outputs center the layer',
		fn: (z) => (z > 0 ? z : Math.exp(z) - 1),
		dfn: (z) => (z > 0 ? 1 : Math.exp(z)),
		levels: [-1]
	},
	{
		id: 'softplus',
		label: 'Softplus',
		code: 'softplus',
		year: '2001',
		tex: '\\ln(1+e^{z})',
		note: 'relu with the corner sanded off — everywhere in theory, rare in practice',
		fn: softplusFn,
		dfn: sig
	},
	{
		id: 'gelu',
		label: 'GELU',
		code: 'gelu',
		year: '2016',
		tex: 'z\\,\\Phi(z)',
		note: 'z weighted by the Gaussian CDF — BERT and the GPT line run on it',
		fn: geluFn,
		dfn: numericD(geluFn)
	},
	{
		id: 'silu',
		label: 'SiLU · Swish',
		code: 'silu',
		year: '2017',
		tex: 'z\\,\\sigma(z)',
		note: 'z gated by its own sigmoid — inside the SwiGLU blocks of the Llama family',
		fn: (z) => z * sig(z),
		dfn: (z) => {
			const s = sig(z);
			return s * (1 + z * (1 - s));
		}
	},
	{
		id: 'mish',
		label: 'Mish',
		code: 'mish',
		year: '2019',
		tex: 'z\\tanh(\\ln(1{+}e^{z}))',
		note: 'self-gated like silu, a touch softer — computer vision’s pick',
		fn: mishFn,
		dfn: numericD(mishFn)
	}
];

/** the one-neuron plate's switcher: the usual suspects plus the two the LMs use, with the
 *  chapter's own starting point first. */
export const PLATE_CHIPS: ActivationId[] = ['tanh', 'sigmoid', 'relu', 'lrelu', 'gelu', 'silu'];

export function activationById(id: ActivationId): ActivationSpec {
	return ACTIVATIONS.find((a) => a.id === id) ?? ACTIVATIONS[1];
}
