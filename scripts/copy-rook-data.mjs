// Ship Rook's assets from the llmvibes repo: corpus, vocab and probe results
// verbatim, plus a trimmed slice of the training time machine. Of the twelve
// archived weight snapshots we keep four — step 0 (noise), ≈300 (grammar
// emerging), ≈1300 (mostly legal), and the final step — enough for the
// chapter's scrubber at a third of the bytes. Each kept waypoint gains a
// `file` field so loaders address snapshots by name, not array position.
//
// Written assets (static/data/):
//   rook-tokens.bin, rook-vocab.json, rook-probe.json      (verbatim copies)
//   timemachine/rook-manifest.json                         (trimmed)
//   timemachine/rook-w{0,8,10,11}.i8                       (kept snapshots)
//
// Usage: node scripts/copy-rook-data.mjs [llmvibesDataDir]

import { readFileSync, writeFileSync, copyFileSync, mkdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = process.argv[2] ?? join(ROOT, '..', 'llmvibes', 'static', 'data');
const TARGET_STEPS = [0, 300, 1300];

const outDir = join(ROOT, 'static', 'data');
const tmDir = join(outDir, 'timemachine');
mkdirSync(tmDir, { recursive: true });

for (const f of ['rook-tokens.bin', 'rook-vocab.json', 'rook-probe.json'])
	copyFileSync(join(SRC, f), join(outDir, f));

const manifest = JSON.parse(readFileSync(join(SRC, 'timemachine', 'rook-manifest.json'), 'utf8'));
const steps = manifest.waypoints.map((w) => w.step);

const keep = new Set();
for (const target of TARGET_STEPS) {
	let best = 0;
	for (let i = 1; i < steps.length; i++)
		if (Math.abs(steps[i] - target) < Math.abs(steps[best] - target)) best = i;
	keep.add(best);
}
keep.add(manifest.waypoints.length - 1); // the final snapshot, always

const indices = [...keep].sort((a, b) => a - b);
const waypoints = indices.map((i) => ({ ...manifest.waypoints[i], file: `rook-w${i}.i8` }));
for (const wp of waypoints) copyFileSync(join(SRC, 'timemachine', wp.file), join(tmDir, wp.file));

writeFileSync(join(tmDir, 'rook-manifest.json'), JSON.stringify({ ...manifest, waypoints }));

const bytes = waypoints.reduce((s, wp) => s + statSync(join(tmDir, wp.file)).size, 0);
console.log(
	`kept ${waypoints.length}/${manifest.waypoints.length} waypoints: ` +
		waypoints.map((wp) => `step ${wp.step} (${wp.file})`).join(', ')
);
console.log(`snapshot bytes: ${(bytes / 1e6).toFixed(1)} MB`);
