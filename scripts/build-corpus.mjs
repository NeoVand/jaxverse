// Char-level corpus for the language chapter: the first ~1.5M characters of
// the TinyStories-style quill corpus, cut at a story boundary (stories are
// separated by blank lines) and scrubbed down to a small ASCII vocabulary.
// The source carries a few hundred double-encoded UTF-8 quotes (â€œ …); those
// are repaired before whitelisting so no words get mangled.
//
// Written assets (static/data/):
//   text-tokens.bin  Uint8, one token per character (id = index into chars)
//   text-vocab.json  {chars: string[], vocabSize}
//
// Usage: node scripts/build-corpus.mjs [source.txt]

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = process.argv[2] ?? join(ROOT, '..', 'llmvibes', 'static', 'data', 'quill-corpus.txt');
const TARGET = 1_500_000;

// Mojibake first (longest sequences before their prefixes), then real
// typographic characters, all down to straight ASCII. Escapes, not literals:
// the third character of each broken sequence renders as a lookalike quote.
const FIXES = [
	['â€œ', '"'], // â€œ = “
	['â€™', "'"], // â€™ = ’
	['â€˜', "'"], // â€˜ = ‘
	['â€¦', '...'], // â€¦ = …
	['â€“', '-'], // â€“ = –
	['â€”', '-'], // â€” = —
	['â€', '"'], // â€ with the third byte lost = ”
	['“', '"'], // “
	['”', '"'], // ”
	['‘', "'"], // ‘
	['’', "'"], // ’
	['–', '-'], // –
	['—', '-'], // —
	['…', '...'] // …
];

const raw = readFileSync(SRC, 'utf8').replaceAll('\r\n', '\n').replaceAll('\r', '\n');
const cut = raw.lastIndexOf('\n\n', TARGET);
if (cut < 0) throw new Error('no story boundary found');
let text = raw.slice(0, cut);

for (const [from, to] of FIXES) text = text.replaceAll(from, to);
text = text
	.replace(/[^\n a-zA-Z0-9.,!?;:'"()-]/g, ' ') // whitelist; strays become spaces…
	.replace(/[ \t]+/g, ' ') // …then collapse runs
	.replace(/ +\n/g, '\n')
	.replace(/\n +/g, '\n')
	.replace(/\n{3,}/g, '\n\n')
	.trim();
text += '\n';

const chars = [...new Set(text)].sort();
if (chars.length > 256) throw new Error(`vocab ${chars.length} does not fit u8`);
const idOf = new Map(chars.map((c, i) => [c, i]));
const tokens = new Uint8Array(text.length);
for (let i = 0; i < text.length; i++) tokens[i] = idOf.get(text[i]);

const outDir = join(ROOT, 'static', 'data');
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'text-tokens.bin'), tokens);
writeFileSync(join(outDir, 'text-vocab.json'), JSON.stringify({ chars, vocabSize: chars.length }));

// ── roundtrip check ──
const decoded = [...tokens.subarray(0, 200)].map((id) => chars[id]).join('');
if (decoded !== text.slice(0, 200)) throw new Error('roundtrip mismatch');
console.log(`tokens: ${tokens.length.toLocaleString()}, vocab: ${chars.length}`);
console.log(`chars: ${JSON.stringify(chars.join(''))}`);
console.log(`first 200 decoded:\n${decoded}`);
