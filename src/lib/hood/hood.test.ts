import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import { hood } from './index';
import { chapters } from '$lib/data/chapters';

// The hidden course is numbered by the registry, not by hand. These guard the
// two ways that can quietly stop being true.
describe('the under-the-hood course', () => {
	it('leaves the numeral to the renderer', () => {
		for (const [slug, chapter] of Object.entries(hood))
			for (const block of chapter.blocks)
				expect(block.lesson, `${slug}/${block.id}`).not.toMatch(/lesson\s*\d/i);
	});

	it('gives every chapter a block, and every block a home', () => {
		for (const c of chapters) expect(hood[c.slug]?.blocks.length, c.slug).toBeGreaterThan(0);
		for (const slug of Object.keys(hood))
			expect(slug === 'home' || chapters.some((c) => c.slug === slug), slug).toBe(true);
	});

	// Only where a lab names itself — its README heading, its page title and
	// heading, its source's first line. A lab pointing at another lab in prose
	// ("the same transformer as Lab 5") is a cross-reference, not a claim about
	// which lab this is.
	// Every sample carries the path a reader can go and find it at, and the
	// course calls them "the real jax-js code from this repository". Names
	// drift; this notices. Comments are prose and are stripped first, and the
	// short words below are language, not repository.
	const LANGUAGE = new Set(
		(
			'const let var function return if else for of in new await async import from export type ' +
			'interface class this true false null undefined number string boolean any void as typeof ' +
			'instanceof do while switch case break continue try catch finally throw yield delete ' +
			'extends implements public private readonly static get set Math JSON Object Array Number ' +
			'String Boolean Promise Set Map Float32Array Float64Array Int32Array Int8Array Uint8Array ' +
			'Uint16Array length push slice map filter reduce forEach console log'
		).split(' ')
	);

	it('quotes code that is actually in the file it points at', () => {
		for (const [slug, chapter] of Object.entries(hood))
			for (const block of chapter.blocks)
				for (const section of [...block.ml, ...block.ui]) {
					if (!section.code?.file) continue;
					const src = readFileSync(section.code.file, 'utf8');
					const body = section.code.code
						.replace(/\/\*[\s\S]*?\*\//g, '')
						.replace(/\/\/[^\n]*/g, '');
					for (const [id] of body.matchAll(/[A-Za-z_$][A-Za-z0-9_$]{2,}/g))
						if (!LANGUAGE.has(id))
							expect(src, `${slug}/${block.id}: ${id} is not in ${section.code.file}`).toContain(
								id
							);
				}
	});

	it('numbers each lab with its own chapter', () => {
		const TITLES = [
			/^# Lab (\d+) ·/m,
			/<title>Lab (\d+) ·/,
			/<h1>Lab (\d+) ·/,
			/^\/\/ Lab (\d+) —/
		];
		for (const c of chapters) {
			const src = ['README.md', 'index.html', 'src/main.ts'].map((f) =>
				readFileSync(`labs/${c.slug}/${f}`, 'utf8')
			);
			const found = TITLES.map((re) => src.map((s) => re.exec(s)).find(Boolean)).filter(Boolean);
			expect(found.length, `${c.slug} names itself in four places`).toBe(4);
			for (const m of found) expect(Number(m![1]), c.slug).toBe(c.n);
		}
	});

	it("hands the lab to each chapter's last block, once", () => {
		for (const [slug, chapter] of Object.entries(hood)) {
			const withLab = chapter.blocks.filter((b) => b.lab);
			if (slug === 'home') continue;
			expect(withLab.length, `${slug} labs`).toBe(1);
			expect(withLab[0], `${slug} lab rides the last block`).toBe(
				chapter.blocks[chapter.blocks.length - 1]
			);
		}
	});
});
