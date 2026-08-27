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
