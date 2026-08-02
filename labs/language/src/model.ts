// A GPT in one file, condensed from jaxverse's src/lib/llm/model.ts.
// Idioms worth knowing: every projection is a 2-D matmul on [B·S, …] rows;
// tokens enter as one-hot matrices multiplied against the embedding table
// (a matmul differentiates cleanly everywhere); output projections start
// small-random, not zero, so gradient reaches the interiors at step 0.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { numpy as np, nn, random } from '@jax-js/jax';

export interface ModelConfig {
	nLayer: number;
	nEmbd: number;
	nHead: number;
	blockSize: number;
	vocab: number;
}

export function initParams(cfg: ModelConfig, seed: number): any {
	const s = Math.sqrt(3 / cfg.nEmbd);
	const n = 3 + cfg.nLayer * 8;
	const keys = random.split(random.key(seed), n);
	let ki = 0;
	const nk = () => {
		ki++;
		return ki < n ? keys.ref.slice(ki - 1) : keys.slice(ki - 1);
	};
	const params: any = {
		wte: random.normal(nk(), [cfg.vocab, cfg.nEmbd]).mul(0.02),
		wpe: random.normal(nk(), [cfg.blockSize, cfg.nEmbd]).mul(0.02),
		lmHead: random.normal(nk(), [cfg.nEmbd, cfg.vocab]).mul(0.001),
		layers: []
	};
	for (let i = 0; i < cfg.nLayer; i++) {
		params.layers.push({
			wq: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -s, maxval: s }),
			wk: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -s, maxval: s }),
			wv: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -s, maxval: s }),
			wo: random.uniform(nk(), [cfg.nEmbd, cfg.nEmbd], { minval: -0.2 * s, maxval: 0.2 * s }),
			mlpFc1: random.uniform(nk(), [cfg.nEmbd, 4 * cfg.nEmbd], {
				minval: -0.4 * s,
				maxval: 0.4 * s
			}),
			mlpFc2: random.uniform(nk(), [4 * cfg.nEmbd, cfg.nEmbd], {
				minval: -0.2 * s,
				maxval: 0.2 * s
			})
		});
	}
	return params;
}

function rmsnorm(x: any) {
	const ms = np.mean(np.square(x.ref), -1, { keepdims: true });
	return x.div(np.sqrt(ms.add(1e-5)));
}

/** Forward to log-probs over the whole sequence: [B·S, vocab]. */
export function forwardLogprobs(params: any, cfg: ModelConfig, tokenOH: any, posOH: any) {
	const S = cfg.blockSize;
	const headDim = cfg.nEmbd / cfg.nHead;
	let x = np.dot(tokenOH.reshape([-1, cfg.vocab]), params.wte);
	const posEmb = np.dot(posOH.reshape([-1, cfg.blockSize]), params.wpe);
	x = rmsnorm(x.add(posEmb));
	for (let li = 0; li < cfg.nLayer; li++) {
		const layer = params.layers[li];
		const xRes = x.ref;
		x = rmsnorm(x);
		const q = np.dot(x.ref, layer.wq);
		const k = np.dot(x.ref, layer.wk);
		const v = np.dot(x, layer.wv);
		const qH = q.reshape([-1, S, cfg.nHead, headDim]);
		const kH = k.reshape([-1, S, cfg.nHead, headDim]);
		const vH = v.reshape([-1, S, cfg.nHead, headDim]);
		const attnOut = nn.dotProductAttention(qH, kH, vH, { isCausal: true });
		x = np.dot(attnOut.reshape([-1, cfg.nEmbd]), layer.wo).add(xRes);
		const mlpRes = x.ref;
		x = rmsnorm(x);
		x = nn.relu(np.dot(x, layer.mlpFc1));
		x = np.dot(x, layer.mlpFc2).add(mlpRes);
	}
	const logits = np.dot(x, params.lmHead);
	return nn.logSoftmax(logits, -1);
}

/** Training loss: mean NLL of the true next token. */
export function lossFn(params: any, cfg: ModelConfig, tokenOH: any, posOH: any, targetOH: any) {
	const logprobs = forwardLogprobs(params, cfg, tokenOH, posOH);
	return np.mean(np.sum(logprobs.mul(targetOH.reshape([-1, cfg.vocab])), -1).neg());
}

/** Sample BATCH windows of next-token pairs from a token stream, as one-hots. */
export function makeBatchOH(
	cfg: ModelConfig,
	data: Uint16Array,
	rand: () => number,
	batch: number
) {
	const S = cfg.blockSize;
	const inputBuf = new Int32Array(batch * S);
	const targetBuf = new Int32Array(batch * S);
	const span = data.length - S - 1;
	for (let b = 0; b < batch; b++) {
		const start = Math.floor(rand() * span);
		for (let i = 0; i < S; i++) {
			inputBuf[b * S + i] = data[start + i];
			targetBuf[b * S + i] = data[start + i + 1];
		}
	}
	const inputIds = np.array(inputBuf, { dtype: np.int32 }).reshape([batch, S]);
	const posIds = np.tile(np.arange(S).astype(np.int32), [batch, 1]);
	const targetIds = np.array(targetBuf, { dtype: np.int32 }).reshape([batch, S]);
	return {
		tokenOH: nn.oneHot(inputIds, cfg.vocab),
		posOH: nn.oneHot(posIds, cfg.blockSize),
		targetOH: nn.oneHot(targetIds, cfg.vocab)
	};
}

/** Right-pad a prompt to blockSize and run one forward pass (causal attention
 * makes the padding irrelevant); returns log-probs as a flat Float32Array. */
export function forwardSeq(jitForward: any, cfg: ModelConfig, tokens: number[]): Float32Array {
	const S = cfg.blockSize;
	const buf = new Int32Array(S);
	for (let i = 0; i < Math.min(tokens.length, S); i++) buf[i] = tokens[i];
	const inputIds = np.array(buf, { dtype: np.int32 }).reshape([1, S]);
	const posIds = np.arange(S).astype(np.int32).reshape([1, S]);
	return jitForward(nn.oneHot(inputIds, cfg.vocab), nn.oneHot(posIds, cfg.blockSize)).dataSync();
}

/** logprobs row → temperature-scaled top-k categorical draw. */
export function sampleFromRow(
	row: Float32Array,
	temperature: number,
	topK: number,
	rand: () => number
): number {
	const V = row.length;
	const idx = Array.from({ length: V }, (_, i) => i);
	idx.sort((a, b) => row[b] - row[a]);
	const keep = topK > 0 ? idx.slice(0, topK) : idx;
	const scaled = keep.map((i) => row[i] / temperature);
	const mx = Math.max(...scaled);
	const ps = scaled.map((v) => Math.exp(v - mx));
	const total = ps.reduce((a, b) => a + b, 0);
	let r = rand() * total;
	for (let i = 0; i < keep.length; i++) {
		r -= ps[i];
		if (r <= 0) return keep[i];
	}
	return keep[keep.length - 1];
}
