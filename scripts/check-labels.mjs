// Asks every label in every plate whether it fits.
//
// A figure is drawn in viewBox units, but the text on it is measured in
// glyphs, and the two only agree until someone edits a caption. A word added
// to a line under a panel is enough to push its last letter past the frame,
// where the SVG quietly clips it — or onto the rule between two panels, which
// reads as a mistake even when the reader cannot say why. Neither shows up in
// a type check, a unit test or a diff.
//
// So measure it. Walk each chapter, take the rendered bounding box of every
// <text> node, and compare it with the viewBox it lives in. Usage:
//
//     node scripts/check-labels.mjs [baseURL]
//
// Like check-papers, this is not part of `npm test`: it needs a running site
// and a real browser to have laid the type out. Run it after touching a
// plate.

import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://localhost:5173';
const PAGES = [
	'',
	'descent',
	'neuron',
	'space',
	'digits',
	'latent',
	'language',
	'reward',
	'rook',
	'taste',
	'epilogue'
];
/** Sub-unit overhangs are antialiasing, not a mistake. */
const SLACK = 0.5;

/** Runs in the page: every text node that leaves its own viewBox. */
function measure(slack) {
	const out = [];
	for (const svg of document.querySelectorAll('svg[viewBox]')) {
		const vb = svg.viewBox.baseVal;
		if (!vb.width) continue;
		const plate = svg.closest('[id^="plate-"]');
		for (const node of svg.querySelectorAll('text')) {
			let box;
			try {
				box = node.getBBox();
			} catch {
				continue; // not laid out — a hidden or not-yet-mounted plate
			}
			if (!box.width) continue;
			const over = [];
			const push = (side, by) => by > slack && over.push(`${side} by ${by.toFixed(1)}`);
			push('left', vb.x - box.x);
			push('top', vb.y - box.y);
			push('right', box.x + box.width - (vb.x + vb.width));
			push('bottom', box.y + box.height - (vb.y + vb.height));
			if (over.length)
				out.push({
					plate: plate?.id ?? '(outside a plate)',
					text: node.textContent.trim().slice(0, 48),
					over: over.join(', ')
				});
		}
	}
	return out;
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
let bad = 0;
let seen = 0;

for (const slug of PAGES) {
	await page.goto(`${BASE}/${slug}`, { waitUntil: 'domcontentloaded' });
	// the plates load lazily, and an unmounted one has nothing to measure
	await page.keyboard.press('End');
	await page.waitForTimeout(2500);
	const svgs = await page.locator('svg[viewBox]').count();
	const hits = await page.evaluate(measure, SLACK);
	seen += svgs;
	bad += hits.length;
	console.log(`${hits.length ? '✗' : '✓'} /${slug} — ${svgs} figures`);
	for (const h of hits) console.log(`    ${h.plate}: “${h.text}” overflows ${h.over}`);
}

await browser.close();
console.log(`\n${seen} figures measured · ${bad} labels outside their frame`);
process.exit(bad ? 1 : 0);
