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

	it('says where a link goes', () => {
		expect(sourceLabel('https://arxiv.org/abs/1412.6980')).toBe('arXiv:1412.6980');
		expect(sourceLabel('https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf')).toBe(
			'jmlr.org'
		);
	});
});
