// Packages each lab in labs/ into a self-contained zip under static/labs/.
// Copies the data files a lab needs into its public/data folder first, so
// every zip runs offline with nothing but `npm install && npm run dev`.
// Run before deploying: node scripts/build-labs.mjs

import { execFileSync } from 'node:child_process';
import { cpSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const dataDir = join(root, 'static', 'data');
const outDir = join(root, 'static', 'labs');

const MNIST = ['mnist-train.png', 'mnist-test.png', 'mnist-labels.bin', 'mnist-meta.json'];
const DATA = {
	descent: [],
	neuron: [],
	space: [],
	digits: MNIST,
	latent: MNIST,
	language: ['text-tokens.bin', 'text-vocab.json'],
	reward: [],
	taste: [],
	rook: ['rook-tokens.bin', 'rook-vocab.json']
};

mkdirSync(outDir, { recursive: true });

for (const slug of readdirSync(join(root, 'labs'))) {
	if (!(slug in DATA)) continue;
	const labDir = join(root, 'labs', slug);
	const files = DATA[slug];
	rmSync(join(labDir, 'public'), { recursive: true, force: true });
	if (files.length > 0) {
		mkdirSync(join(labDir, 'public', 'data'), { recursive: true });
		for (const f of files) cpSync(join(dataDir, f), join(labDir, 'public', 'data', f));
	}
	const zipPath = join(outDir, `${slug}.zip`);
	rmSync(zipPath, { force: true });
	execFileSync('zip', ['-r', '-q', zipPath, slug, '-x', `${slug}/node_modules/*`], {
		cwd: join(root, 'labs')
	});
	console.log('packed', `${slug}.zip`);
}
