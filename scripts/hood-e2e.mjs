// Verifies the Under-the-Hood section on every chapter page: it renders,
// expands, both tabs show sections with code, and the lab zip download
// answers 200. Usage: node scripts/hood-e2e.mjs [baseURL]

import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://localhost:5173';
const CHAPTERS = ['descent', 'neuron', 'space', 'digits', 'latent', 'language', 'reward', 'rook'];

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
let failed = 0;

for (const slug of CHAPTERS) {
	try {
		await page.goto(`${BASE}/${slug}`, { waitUntil: 'domcontentloaded' });
		// scrolling down wakes the chapter's lazy-loaded demos; give the main
		// thread room, and retry the click once if the panel didn't open
		const toggle = page.locator('.hood-toggle');
		await toggle.scrollIntoViewIfNeeded();
		await page.waitForTimeout(1500);
		await toggle.click();
		try {
			await page.locator('.hood [role="tablist"]').waitFor({ timeout: 15_000 });
		} catch {
			await toggle.click();
			await page.locator('.hood [role="tablist"]').waitFor({ timeout: 15_000 });
		}
		const mlCode = await page.locator('.hood .code').count();
		await page.locator('.hood [role="tab"]', { hasText: 'stagecraft' }).click();
		await page.waitForTimeout(150);
		const uiSections = await page.locator('.hood h3').count();
		const href = await page.locator('.hood a[download]').getAttribute('href');
		const res = await page.request.get(`${BASE}${href?.startsWith('/') ? '' : '/'}${href}`);
		const size = (await res.body()).length;
		const ok = mlCode > 0 && uiSections > 0 && res.status() === 200 && size > 1000;
		if (!ok) failed++;
		console.log(
			`${ok ? '✓' : '✗'} ${slug} — ml code blocks: ${mlCode}, ui sections: ${uiSections}, ` +
				`zip: ${res.status()} (${(size / 1024).toFixed(0)} KB)`
		);
	} catch (e) {
		failed++;
		console.log(`✗ ${slug} — ${String(e).split('\n')[0]}`);
	}
}
await browser.close();
process.exit(failed ? 1 : 0);
