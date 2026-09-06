// Build the emoji corpus for the diffusion chapters.
//
// Eight drawing styles of the same few thousand pictures, which is the whole
// point: a model that sees one concept rendered eight ways can be asked to
// separate what a thing IS from how it is drawn. Every set here is
// redistributable with attribution, and the credits in src/lib/data/emoji.ts
// name each one.
//
//   noto          Google Noto Color Emoji        Apache-2.0
//   twemoji       Twitter/jdecked Twemoji        CC BY 4.0
//   openmoji      OpenMoji, colour               CC BY-SA 4.0
//   openmoji-b    OpenMoji, black line art       CC BY-SA 4.0
//   fluent-3d     Microsoft Fluent, 3D render    MIT
//   fluent-color  Microsoft Fluent, outlined     MIT
//   fluent-flat   Microsoft Fluent, flat vector  MIT
//   blob          Google's retired blobs         Apache-2.0
//
// Output (static/data/):
//   emoji-<set>.png   RGBA spritesheet, 32x32 tiles, premultiplied
//   emoji-meta.json   sets, codepoints, names, tag vocabulary, tag indices
//
// Everything downloaded is cached under .cache/emoji so a re-run is cheap.
// Usage: node scripts/build-emoji.mjs [--limit N] [--refresh]

import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import sharp from 'sharp';

const ROOT = new URL('..', import.meta.url).pathname;
const CACHE = path.join(ROOT, '.cache/emoji');
const OUT = path.join(ROOT, 'static/data');

const TILE = 32; // the resolution the chapters train at
const RENDER = 160; // rasterize large, then box down — SVG edges stay clean
const args = process.argv.slice(2);
const LIMIT = args.includes('--limit') ? Number(args[args.indexOf('--limit') + 1]) : Infinity;

// ------------------------------------------------------------- fetching ---

const UA = 'jaxverse-emoji-build (+https://github.com/jaxverse)';

async function cached(key, url, { json = false, optional = false } = {}) {
	const file = path.join(CACHE, `${createHash('sha1').update(key).digest('hex').slice(0, 20)}.bin`);
	try {
		await access(file);
		const buf = await readFile(file);
		if (buf.length === 0) return optional ? null : buf;
		return json ? JSON.parse(buf.toString('utf8')) : buf;
	} catch {
		/* not cached yet */
	}
	for (let attempt = 0; attempt < 3; attempt++) {
		try {
			const res = await fetch(url, { headers: { 'user-agent': UA } });
			if (res.status === 404) {
				if (optional) {
					await writeFile(file, Buffer.alloc(0)); // remember the miss
					return null;
				}
				throw new Error(`404 ${url}`);
			}
			if (!res.ok) throw new Error(`${res.status} ${url}`);
			const buf = Buffer.from(await res.arrayBuffer());
			await writeFile(file, buf);
			return json ? JSON.parse(buf.toString('utf8')) : buf;
		} catch (e) {
			if (attempt === 2) {
				if (optional) return null;
				throw e;
			}
			await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
		}
	}
	return null;
}

/** Run `fn` over `items` with a fixed number of workers in flight. */
async function pool(items, width, fn, onProgress) {
	const results = new Array(items.length);
	let next = 0;
	let done = 0;
	await Promise.all(
		Array.from({ length: Math.min(width, items.length) }, async () => {
			for (;;) {
				const i = next++;
				if (i >= items.length) return;
				results[i] = await fn(items[i], i);
				done++;
				if (onProgress && done % 50 === 0) onProgress(done, items.length);
			}
		})
	);
	return results;
}

// -------------------------------------------------------------- the sets ---

const hexes = (cp) => [...cp].map((c) => c.codePointAt(0).toString(16));
/** Most sets drop the FE0F "render me in colour" selector from filenames. */
const variants = (cp) => {
	const full = hexes(cp);
	const bare = full.filter((h) => h !== 'fe0f');
	return bare.length && bare.join('-') !== full.join('-') ? [full, bare] : [full];
};

const jd = 'https://cdn.jsdelivr.net';

const SETS = [
	{
		id: 'noto',
		label: 'Noto',
		credit: 'Google Noto Color Emoji',
		license: 'Apache-2.0',
		url: 'https://github.com/googlefonts/noto-emoji',
		urls: (cp) =>
			variants(cp).map((v) => `${jd}/gh/googlefonts/noto-emoji@main/svg/emoji_u${v.join('_')}.svg`)
	},
	{
		id: 'twemoji',
		label: 'Twemoji',
		credit: 'Twemoji, maintained by jdecked',
		license: 'CC BY 4.0',
		url: 'https://github.com/jdecked/twemoji',
		urls: (cp) =>
			variants(cp).map((v) => `${jd}/gh/jdecked/twemoji@main/assets/svg/${v.join('-')}.svg`)
	},
	{
		id: 'openmoji',
		label: 'OpenMoji',
		credit: 'OpenMoji, colour',
		license: 'CC BY-SA 4.0',
		url: 'https://openmoji.org',
		urls: (cp) =>
			variants(cp).map(
				(v) => `${jd}/npm/openmoji@15.0.0/color/svg/${v.join('-').toUpperCase()}.svg`
			)
	},
	{
		id: 'openmoji-b',
		label: 'OpenMoji line',
		credit: 'OpenMoji, black line art',
		license: 'CC BY-SA 4.0',
		url: 'https://openmoji.org',
		urls: (cp) =>
			variants(cp).map(
				(v) => `${jd}/npm/openmoji@15.0.0/black/svg/${v.join('-').toUpperCase()}.svg`
			)
	},
	{
		id: 'blob',
		label: 'Blobs',
		credit: "Blobmoji, Google's retired blob set",
		license: 'Apache-2.0',
		url: 'https://github.com/C1710/blobmoji',
		urls: (cp) =>
			variants(cp).map(
				(v) => `${jd}/gh/C1710/blobmoji@2021-07-12-Emoji13.1-pre/svg/emoji_u${v.join('_')}.svg`
			)
	},
	{
		id: 'fluent-3d',
		label: 'Fluent 3D',
		credit: 'Microsoft Fluent Emoji, 3D',
		license: 'MIT',
		url: 'https://github.com/microsoft/fluentui-emoji',
		fluent: '3D'
	},
	{
		id: 'fluent-color',
		label: 'Fluent',
		credit: 'Microsoft Fluent Emoji, colour',
		license: 'MIT',
		url: 'https://github.com/microsoft/fluentui-emoji',
		fluent: 'Color'
	},
	{
		id: 'fluent-flat',
		label: 'Fluent flat',
		credit: 'Microsoft Fluent Emoji, flat',
		license: 'MIT',
		url: 'https://github.com/microsoft/fluentui-emoji',
		fluent: 'Flat'
	}
];

// ------------------------------------------------------------- selection ---

const STOP = new Set(
	`a an and or of the with in on at to for from is are be it its his her their they
	 them this that these those very more most other another some any all both each
	 person people man woman men women boy girl adult child sign symbol mark type
	 medium light dark skin tone plus minus one two three four five six seven eight
	 nine ten no not up down left right side front back over under out off into onto`
		.split(/\s+/)
		.filter(Boolean)
);

/** Fold plurals so "eyes" and "eye" are one tag. Deliberately conservative:
 *  a wrong stem invents a word the reader can type and the model never saw. */
function singular(w) {
	if (/(ss|us|is|as)$/.test(w) || w.length < 5) return w;
	if (/ies$/.test(w)) return `${w.slice(0, -3)}y`;
	if (/(ch|sh|s|x|z)es$/.test(w)) return w.slice(0, -2);
	if (/ves$/.test(w)) return `${w.slice(0, -3)}f`;
	if (/s$/.test(w)) return w.slice(0, -1);
	return w;
}

function tokens(...strings) {
	const seen = new Set();
	for (const s of strings) {
		if (!s) continue;
		for (const raw of String(s)
			.toLowerCase()
			.split(/[^a-z]+/)) {
			if (raw.length < 3 || STOP.has(raw)) continue;
			seen.add(singular(raw));
		}
	}
	return [...seen];
}

async function main() {
	await mkdir(CACHE, { recursive: true });
	await mkdir(OUT, { recursive: true });

	console.log('· metadata');
	const byGroup = await cached(
		'unicode-emoji-json-by-group',
		`${jd}/npm/unicode-emoji-json@0.8.0/data-by-group.json`,
		{ json: true }
	);
	const cldr = await cached(
		'cldr-annotations-en',
		'https://raw.githubusercontent.com/unicode-org/cldr-json/main/cldr-json/cldr-annotations-full/annotations/en/annotations.json',
		{ json: true }
	);
	const ann = cldr.annotations.annotations;

	// Candidates: one codepoint (optionally plus FE0F), no flags, no skin tones,
	// no keycaps. Those either fail to draw at 32px or repeat one silhouette
	// hundreds of times, and both would teach the model the wrong lesson.
	const candidates = [];
	for (const { name: group, emojis } of byGroup) {
		if (/flag/i.test(group)) continue;
		for (const e of emojis) {
			const cp = e.emoji;
			const hs = hexes(cp);
			const core = hs.filter((h) => h !== 'fe0f');
			if (core.length !== 1) continue;
			if (/1f3f[b-f]/.test(hs.join())) continue;
			if (hs.includes('20e3')) continue;
			const bare = core.map((h) => String.fromCodePoint(parseInt(h, 16))).join('');
			candidates.push({
				cp,
				name: e.name,
				group,
				keywords: ann[cp]?.default ?? ann[bare]?.default ?? []
			});
		}
	}
	console.log(`  ${candidates.length} candidate emoji`);

	// Fluent indexes by display-name folder, so build codepoint -> folder.
	console.log('· fluent index');
	const tree = await cached(
		'fluent-tree-main',
		'https://api.github.com/repos/microsoft/fluentui-emoji/git/trees/main?recursive=1',
		{ json: true }
	);
	const folders = [
		...new Set(
			tree.tree.map((t) => /^assets\/([^/]+)\/metadata\.json$/.exec(t.path)?.[1]).filter(Boolean)
		)
	];
	const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
	const byNorm = new Map(folders.map((f) => [norm(f), f]));
	const fluentFolder = new Map();
	const unmatched = [];
	for (const c of candidates) {
		const hit = byNorm.get(norm(c.name));
		if (hit) fluentFolder.set(c.cp, hit);
		else unmatched.push(c);
	}
	// Fall back to each folder's own metadata for the names that did not line up.
	const metaFolders = folders.filter((f) => ![...fluentFolder.values()].includes(f));
	console.log(
		`  ${fluentFolder.size} matched by name, resolving ${metaFolders.length} by metadata`
	);
	const metas = await pool(metaFolders, 24, async (f) => {
		const m = await cached(
			`fluent-meta-${f}`,
			`${jd}/gh/microsoft/fluentui-emoji@main/assets/${encodeURIComponent(f)}/metadata.json`,
			{ json: true, optional: true }
		);
		return m ? { folder: f, unicode: m.unicode, glyph: m.glyph } : null;
	});
	const byUnicode = new Map();
	for (const m of metas) {
		if (!m) continue;
		if (m.glyph) byUnicode.set(m.glyph, m.folder);
		if (m.unicode) byUnicode.set(m.unicode.replace(/\s+/g, '-').toLowerCase(), m.folder);
	}
	for (const c of unmatched) {
		const key = hexes(c.cp).join('-');
		const bare = hexes(c.cp)
			.filter((h) => h !== 'fe0f')
			.join('-');
		const hit = byUnicode.get(c.cp) ?? byUnicode.get(key) ?? byUnicode.get(bare);
		if (hit) fluentFolder.set(c.cp, hit);
	}
	console.log(`  ${fluentFolder.size} fluent folders resolved`);

	const slug = (folder) =>
		folder
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, '_')
			.replace(/^_|_$/g, '');
	for (const s of SETS) {
		if (!s.fluent) continue;
		const dir = s.fluent;
		const ext = dir === '3D' ? 'png' : 'svg';
		const suffix = dir.toLowerCase().replace(/\s+/g, '_');
		s.urls = (cp) => {
			const f = fluentFolder.get(cp);
			if (!f) return [];
			return [
				`${jd}/gh/microsoft/fluentui-emoji@main/assets/${encodeURIComponent(f)}/${dir}/${slug(f)}_${suffix}.${ext}`
			];
		};
	}

	// ------------------------------------------------------------ download ---

	console.log(`· downloading ${candidates.length} x ${SETS.length} tiles`);
	const have = new Map(); // cp -> Map(setId -> Buffer)
	for (const s of SETS) {
		let hits = 0;
		const got = await pool(
			candidates,
			24,
			async (c) => {
				for (const url of s.urls(c.cp)) {
					const buf = await cached(`${s.id}:${url}`, url, { optional: true });
					if (buf && buf.length > 64) return buf;
				}
				return null;
			},
			(d, n) => process.stdout.write(`\r  ${s.id.padEnd(13)} ${d}/${n}`)
		);
		candidates.forEach((c, i) => {
			if (!got[i]) return;
			hits++;
			if (!have.has(c.cp)) have.set(c.cp, new Map());
			have.get(c.cp).set(s.id, got[i]);
		});
		process.stdout.write(`\r  ${s.id.padEnd(13)} ${hits}/${candidates.length}\n`);
	}

	// Keep only emoji every set can draw — a hole in one style would otherwise
	// teach the model that this concept and that style never co-occur.
	let keep = candidates.filter((c) => have.get(c.cp)?.size === SETS.length);
	console.log(`· ${keep.length} emoji present in all ${SETS.length} sets`);
	if (keep.length > LIMIT) keep = keep.slice(0, LIMIT);

	// ----------------------------------------------------------- rasterize ---

	const cols = Math.ceil(Math.sqrt(keep.length));
	const rows = Math.ceil(keep.length / cols);
	console.log(
		`· rasterizing ${keep.length} x ${SETS.length} to ${TILE}px (${cols}x${rows} sheets)`
	);

	const sheets = new Map();
	for (const s of SETS) {
		const sheet = Buffer.alloc(cols * rows * TILE * TILE * 4);
		await pool(keep, 8, async (c, i) => {
			const src = have.get(c.cp).get(s.id);
			const raw = await sharp(src, { density: 384 })
				.resize(RENDER, RENDER, {
					fit: 'contain',
					background: { r: 0, g: 0, b: 0, alpha: 0 },
					kernel: 'lanczos3'
				})
				.resize(TILE, TILE, { kernel: 'lanczos3' })
				.ensureAlpha()
				.raw()
				.toBuffer();
			// premultiply: transparent pixels become an exact zero the model can
			// hit, instead of whatever colour happened to sit under the alpha
			const tx = (i % cols) * TILE;
			const ty = Math.floor(i / cols) * TILE;
			for (let y = 0; y < TILE; y++) {
				for (let x = 0; x < TILE; x++) {
					const o = (y * TILE + x) * 4;
					const a = raw[o + 3] / 255;
					const d = ((ty + y) * cols * TILE + tx + x) * 4;
					sheet[d] = Math.round(raw[o] * a);
					sheet[d + 1] = Math.round(raw[o + 1] * a);
					sheet[d + 2] = Math.round(raw[o + 2] * a);
					sheet[d + 3] = raw[o + 3];
				}
			}
		});
		const file = path.join(OUT, `emoji-${s.id}.png`);
		const { size } = await sharp(sheet, {
			raw: { width: cols * TILE, height: rows * TILE, channels: 4 }
		})
			.png({ compressionLevel: 9, effort: 10 })
			.toFile(file);
		console.log(`  emoji-${s.id}.png ${(size / 1024).toFixed(0)} KB`);
		sheets.set(s.id, sheet);
	}

	// A contact sheet to check by eye: the same twelve emoji down every style,
	// drawn at 4x on a mid grey so both dark and light artwork is visible.
	{
		const N = 12;
		const Z = 4;
		const W = N * TILE * Z;
		const H = SETS.length * TILE * Z;
		const proof = Buffer.alloc(W * H * 4);
		for (let i = 0; i < W * H; i++) {
			proof[i * 4] = proof[i * 4 + 1] = proof[i * 4 + 2] = 128;
			proof[i * 4 + 3] = 255;
		}
		SETS.forEach((s, si) => {
			const sheet = sheets.get(s.id);
			for (let e = 0; e < N; e++) {
				const idx = Math.floor((e * keep.length) / N);
				const sx = (idx % cols) * TILE;
				const sy = Math.floor(idx / cols) * TILE;
				for (let y = 0; y < TILE * Z; y++) {
					for (let x = 0; x < TILE * Z; x++) {
						const o = (((sy + y / Z) | 0) * cols * TILE + sx + ((x / Z) | 0)) * 4;
						const a = sheet[o + 3] / 255;
						const d = ((si * TILE * Z + y) * W + e * TILE * Z + x) * 4;
						for (let ch = 0; ch < 3; ch++)
							proof[d + ch] = Math.round(sheet[o + ch] + 128 * (1 - a));
					}
				}
			}
		});
		await sharp(proof, { raw: { width: W, height: H, channels: 4 } })
			.png()
			.toFile(path.join(ROOT, '.cache/emoji-proof.png'));
		console.log('  .cache/emoji-proof.png written');
	}

	// --------------------------------------------------------------- tags ---

	// Tags come from the emoji's CLDR name and keywords only. The Unicode group
	// ("Travel & Places") was tried and dropped: it labels two hundred pictures
	// at a time, which is a tag the model can satisfy without drawing anything.
	const counts = new Map();
	const perEmoji = keep.map((c) => {
		const t = tokens(c.name, ...(c.keywords ?? []));
		for (const w of t) counts.set(w, (counts.get(w) ?? 0) + 1);
		return t;
	});
	// A tag has to describe enough pictures to be learnable and few enough to
	// mean something: below the floor it is noise, above the ceiling it is "the".
	const vocab = [...counts.entries()]
		.filter(([, n]) => n >= 4 && n <= keep.length * 0.2)
		.sort((a, b) => b[1] - a[1])
		.slice(0, 384)
		.map(([w]) => w)
		.sort();
	const tagIndex = new Map(vocab.map((w, i) => [w, i]));
	const common = [...counts.entries()]
		.filter(([w]) => tagIndex.has(w))
		.sort((a, b) => b[1] - a[1])
		.slice(0, 12)
		.map(([w, n]) => `${w}(${n})`)
		.join(' ');
	console.log(`· ${vocab.length} tags, most common: ${common}`);

	const meta = {
		tile: TILE,
		cols,
		rows,
		count: keep.length,
		sets: SETS.map(({ id, label, credit, license, url }) => ({ id, label, credit, license, url })),
		tags: vocab,
		emoji: keep.map((c, i) => ({
			cp: c.cp,
			name: c.name,
			group: c.group,
			tags: perEmoji[i]
				.map((w) => tagIndex.get(w))
				.filter((n) => n !== undefined)
				.sort((a, b) => a - b)
		}))
	};
	const untagged = meta.emoji.filter((e) => e.tags.length === 0).length;
	console.log(`· ${untagged} emoji ended up with no tags`);
	await writeFile(path.join(OUT, 'emoji-meta.json'), JSON.stringify(meta));
	const bytes = (await readFile(path.join(OUT, 'emoji-meta.json'))).length;
	console.log(`  emoji-meta.json ${(bytes / 1024).toFixed(0)} KB`);
	console.log('done');
}

await main();
