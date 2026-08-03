// Verifies the Under-the-Hood blocks on every chapter page: each block
// renders and expands with content, exactly one block per chapter carries the
// lab download, and the lab zip answers 200. Usage:
// node scripts/hood-e2e.mjs [baseURL]

import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://localhost:5173';
// slug → expected number of hood blocks on the page
const CHAPTERS = {
	descent: 2,
	neuron: 2,
	space: 2,
	digits: 2,
	latent: 2,
	language: 2,
	reward: 2,
	rook: 3
};

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
let failed = 0;

for (const [slug, expected] of Object.entries(CHAPTERS)) {
	try {
		await page.goto(`${BASE}/${slug}`, { waitUntil: 'domcontentloaded' });
		const toggles = page.locator('.hood-toggle');
		// scrolling wakes the chapter's lazy-loaded demos; walk to the bottom
		// so every hood block is mounted before counting
		await page.keyboard.press('End');
		await page.waitForTimeout(1500);
		const count = await toggles.count();

		// open every block and count its content sections
		let sections = 0;
		for (let i = 0; i < count; i++) {
			const toggle = toggles.nth(i);
			await toggle.scrollIntoViewIfNeeded();
			await toggle.click();
			const hood = page.locator('.hood').nth(i);
			try {
				await hood.locator('h3').first().waitFor({ timeout: 15_000 });
			} catch {
				await toggle.click();
				await hood.locator('h3').first().waitFor({ timeout: 15_000 });
			}
			sections += await hood.locator('h3').count();
		}

		// exactly one lab download per chapter, and the zip must answer 200
		const downloads = page.locator('.hood a[download]');
		const nDownloads = await downloads.count();
		const href = await downloads.first().getAttribute('href');
		const res = await page.request.get(`${BASE}${href?.startsWith('/') ? '' : '/'}${href}`);
		const size = (await res.body()).length;

		const ok =
			count === expected &&
			sections >= expected &&
			nDownloads === 1 &&
			res.status() === 200 &&
			size > 1000;
		if (!ok) failed++;
		console.log(
			`${ok ? '✓' : '✗'} ${slug} — blocks: ${count}/${expected}, sections: ${sections}, ` +
				`downloads: ${nDownloads}, zip: ${res.status()} (${(size / 1024).toFixed(0)} KB)`
		);
	} catch (e) {
		failed++;
		console.log(`✗ ${slug} — ${String(e).split('\n')[0]}`);
	}
}
await browser.close();
process.exit(failed ? 1 : 0);
