// Print a chapter's prose as plain text, so it can be read aloud rather than
// read through markup. Captions come out too, marked, because a caption is
// prose the reader will absolutely read. Usage: node scripts/read-chapter.mjs noise

import { readFileSync } from 'node:fs';

const slug = process.argv[2] ?? 'noise';
let s = readFileSync(`src/routes/${slug}/+page.svelte`, 'utf8');
s = s.replace(/<script[\s\S]*?<\/script>/, '');

const captions = [...s.matchAll(/caption="([^"]*)"/g)].map((m) => m[1]);
const titles = [...s.matchAll(/\btitle="([^"]*)"/g)].map((m) => m[1]);

s = s.replace(/caption="[^"]*"/g, '');
s = s.replace(/<Math[^>]*tex=\{'([\s\S]*?)'\}[^>]*\/>/g, (_m, t) => {
	const clean = String(t)
		.replace(/\\htmlClass\{[a-z0-9-]+\}/g, '')
		.replace(/\\[a-zA-Z]+/g, (k) => k.slice(1))
		.replace(/[{}]/g, '');
	return ` ⟨${clean.trim()}⟩ `;
});
s = s.replace(/<Cite id="([^"]+)"\s*\/>/g, '[$1]');
s = s.replace(/<ChapterRef slug="([^"]+)"\s*\/>/g, '«ch:$1»');
s = s.replace(/<PlateRef id="([^"]+)"[^>]*\/>/g, '«plate:$1»');
s = s.replace(/<\/p>/g, '\n\n');
s = s.replace(/<h2[^>]*>/g, '\n## ');
s = s.replace(/<\/h2>/g, '\n');
s = s.replace(/<[^>]+>/g, '');
s = s
	.replace(/&nbsp;/g, ' ')
	.replace(/&amp;/g, '&')
	.replace(/[ \t]+/g, ' ')
	.replace(/\n /g, '\n')
	.replace(/\n{3,}/g, '\n\n');

const words = s.split(/\s+/).filter(Boolean).length;
console.log(s.trim());
console.log('\n\n════════ PLATES ════════');
titles.forEach((t, i) => {
	console.log(`\n▸ ${t}\n  ${captions[i] ?? ''}`);
});
console.log(`\n════════ ${words} words of prose, ${titles.length} plates ════════`);
