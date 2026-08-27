import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { citationNumber, citationOrder, papers, sourceLabel, type PaperId } from './papers';
import { chapters } from './chapters';

// The bibliography is data, and data drifts. These are the three ways it can
// go wrong without anyone noticing: a chapter cites a paper that is not in the
// registry, the same paper is listed twice so the numerals lie, or a chapter
// key is a typo that quietly renders no sources at all.

describe('the bibliography', () => {
	const slugs = new Set(chapters.map((c) => c.slug));

	it('cites only papers that exist', () => {
		for (const [chapter, list] of Object.entries(citationOrder))
			for (const id of list ?? []) expect(papers[id], `${chapter} cites ${id}`).toBeDefined();
	});

	it('lists each paper at most once per chapter', () => {
		for (const [chapter, list] of Object.entries(citationOrder))
			expect(new Set(list).size, `${chapter} has a duplicate`).toBe((list ?? []).length);
	});

	it('is keyed by real chapters', () => {
		for (const chapter of Object.keys(citationOrder))
			expect(slugs.has(chapter as never)).toBe(true);
	});

	it('numbers by position, from one', () => {
		const first = citationOrder.descent?.[0] as PaperId;
		expect(citationNumber('descent', first)).toBe(1);
		expect(citationNumber('nowhere', first)).toBeUndefined();
	});

	it('points every source somewhere readable', () => {
		for (const [id, paper] of Object.entries(papers)) {
			expect(paper.url, id).toMatch(/^https:\/\//);
			expect(() => new URL(paper.url), id).not.toThrow();
		}
	});

	// The registry decides the numerals, so if it drifts out of the order the
	// prose actually reaches for its sources, a chapter counts 1, 2, 4, 3 down
	// the page. Read the marks straight out of the chapter files and compare.
	it('numbers each chapter in the order its prose cites', () => {
		for (const [chapter, list] of Object.entries(citationOrder)) {
			const page = readFileSync(`src/routes/${chapter}/+page.svelte`, 'utf8');
			const cited = [...page.matchAll(/<Cite\s+id="([^"]+)"/g)].map((m) => m[1]);
			const firstUse = [...new Set(cited)];
			expect(firstUse, `${chapter} cites in a different order than it lists`).toEqual([
				...(list ?? [])
			]);
		}
	});

	it('says where a link goes', () => {
		expect(sourceLabel('https://arxiv.org/abs/1412.6980')).toBe('arXiv:1412.6980');
		expect(sourceLabel('https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf')).toBe(
			'jmlr.org'
		);
	});
});
