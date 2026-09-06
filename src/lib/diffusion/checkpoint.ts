// Checkpoint packing.
//
// Two formats, one layout. Float32 is what the offline trainer resumes from;
// int8 is what ships, because the weights are a download the reader waits on
// and 2.5 million of them are 10 MB in full and 2.5 MB quantized. Each tensor
// keeps its own scale, so a small tensor is not flattened by a large one's
// range.

import { numpy as np, tree } from '@jax-js/jax';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

const MAGIC = 0x4a584446; // 'JXDF'
const VERSION = 1;

interface Header {
	dtype: 'f32' | 'i8';
	shapes: number[][];
	/** One per tensor, int8 only: value = code · scale. */
	scales?: number[];
	steps: number;
	objective: string;
}

function encode(header: Header, payload: Uint8Array): ArrayBuffer {
	const json = new TextEncoder().encode(JSON.stringify(header));
	const pad = (4 - (json.length % 4)) % 4;
	const out = new ArrayBuffer(12 + json.length + pad + payload.byteLength);
	const view = new DataView(out);
	view.setUint32(0, MAGIC);
	view.setUint32(4, VERSION);
	view.setUint32(8, json.length + pad);
	new Uint8Array(out, 12, json.length).set(json);
	new Uint8Array(out, 12 + json.length + pad).set(payload);
	return out;
}

function decode(buf: ArrayBuffer): { header: Header; payload: ArrayBuffer } {
	const view = new DataView(buf);
	if (view.getUint32(0) !== MAGIC) throw new Error('not a jaxverse diffusion checkpoint');
	if (view.getUint32(4) !== VERSION) throw new Error('checkpoint version mismatch');
	const len = view.getUint32(8);
	const json = new TextDecoder().decode(new Uint8Array(buf, 12, len));
	return { header: JSON.parse(json.replace(/\0+$/, '')), payload: buf.slice(12 + len) };
}

// dataSync() consumes the array it is called on, so the refs taken here are
// spent by the read itself — disposing them afterwards would free the weights.

export function packFloat32(params: Arr, steps: number, objective: string): ArrayBuffer {
	const leaves = tree.leaves(tree.ref(params)) as Arr[];
	const shapes = leaves.map((l) => [...l.shape]);
	const total = leaves.reduce((s, l) => s + l.size, 0);
	const payload = new Float32Array(total);
	let off = 0;
	for (const l of leaves) {
		const size = l.size;
		payload.set(l.dataSync() as Float32Array, off);
		off += size;
	}
	return encode({ dtype: 'f32', shapes, steps, objective }, new Uint8Array(payload.buffer));
}

export function packInt8(params: Arr, steps: number, objective: string): ArrayBuffer {
	const leaves = tree.leaves(tree.ref(params)) as Arr[];
	const shapes = leaves.map((l) => [...l.shape]);
	const total = leaves.reduce((s, l) => s + l.size, 0);
	const payload = new Int8Array(total);
	const scales: number[] = [];
	let off = 0;
	for (const l of leaves) {
		const data = l.dataSync() as Float32Array;
		let peak = 0;
		for (const v of data) peak = Math.max(peak, Math.abs(v));
		const scale = peak / 127 || 1;
		scales.push(scale);
		for (let i = 0; i < data.length; i++) {
			payload[off + i] = Math.max(-127, Math.min(127, Math.round(data[i] / scale)));
		}
		off += data.length;
	}
	return encode({ dtype: 'i8', shapes, scales, steps, objective }, new Uint8Array(payload.buffer));
}

export interface LoadedCheckpoint {
	params: Arr;
	steps: number;
	objective: string;
}

/**
 * Rebuild the tree. `template` supplies the shape of the object — a freshly
 * initialized set of parameters for the same config — and is consumed.
 */
export function unpack(buf: ArrayBuffer, template: Arr): LoadedCheckpoint {
	const { header, payload } = decode(buf);
	const [templateLeaves, def] = tree.flatten(template) as [Arr[], Arr];
	if (templateLeaves.length !== header.shapes.length) {
		tree.dispose(template);
		throw new Error(
			`checkpoint has ${header.shapes.length} tensors, this model has ${templateLeaves.length}`
		);
	}
	const leaves: Arr[] = [];
	let off = 0;
	for (let i = 0; i < header.shapes.length; i++) {
		const shape = header.shapes[i];
		const size = shape.reduce((a, b) => a * b, 1);
		const want = [...templateLeaves[i].shape];
		if (want.join('x') !== shape.join('x')) {
			tree.dispose(template);
			throw new Error(
				`tensor ${i}: checkpoint is ${shape.join('x')}, model wants ${want.join('x')}`
			);
		}
		templateLeaves[i].dispose();
		let values: Float32Array<ArrayBuffer>;
		if (header.dtype === 'f32') {
			// a copy, because np.array must not alias the transferred buffer
			values = new Float32Array(size);
			values.set(new Float32Array(payload, off * 4, size));
		} else {
			const codes = new Int8Array(payload, off, size);
			const scale = header.scales![i];
			values = new Float32Array(size);
			for (let k = 0; k < size; k++) values[k] = codes[k] * scale;
		}
		off += size;
		leaves.push(np.array(values).reshape(shape));
	}
	return {
		params: tree.unflatten(def, leaves),
		steps: header.steps,
		objective: header.objective
	};
}

/** Read the header without materializing any weights. */
export function inspect(buf: ArrayBuffer): Header {
	return decode(buf).header;
}
