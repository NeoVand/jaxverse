// Train several diffusion recipes for equal wall time and write one sheet
// comparing their samples, so the choice of recipe is made by looking.
// Usage: node scripts/ablate-recipe.mjs [minutesPerRecipe]

import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const ROOT = new URL('..', import.meta.url).pathname;
const OUT = path.join(ROOT, '.cache/proof');
const minutes = Number(process.argv[2] ?? 10);
const port = 5700 + Math.floor(Math.random() * 200);

await mkdir(OUT, { recursive: true });
const server = spawn(
	'npx',
	['vite', 'tools/recipe-ablation', '--port', String(port), '--strictPort'],
	{ cwd: ROOT, stdio: 'ignore' }
);
const browser = await chromium.launch({
	headless: true,
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU']
});

try {
	await new Promise((r) => setTimeout(r, 3500));
	const page = await browser.newPage();
	page.on('pageerror', (e) => console.log('PAGEERROR', String(e)));
	page.on('console', (m) => console.log(`CONSOLE[${m.type()}]`, m.text()));
	page.on('requestfailed', (r) => console.log('REQFAIL', r.url(), r.failure()?.errorText));
	await page.goto(`http://localhost:${port}/?minutes=${minutes}`, { waitUntil: 'load' });

	// four recipes, plus compile and sampling overhead
	const deadline = Date.now() + (minutes * 4 + 12) * 60_000;
	let seen = '';
	while (Date.now() < deadline) {
		const text = await page
			.locator('#out')
			.innerText()
			.catch(() => seen);
		if (text.length > seen.length) {
			process.stdout.write(text.slice(seen.length));
			seen = text;
		}
		if (/error:/i.test(seen) || (await page.evaluate(() => window.__done === true))) break;
		await new Promise((r) => setTimeout(r, 4000));
		// keep the intermediate sheet on disk in case the run is cut short
		const partial = await page.evaluate(() => window.__sheet);
		if (partial) {
			await writeFile(
				path.join(OUT, 'ablation.png'),
				Buffer.from(String(partial).split(',')[1], 'base64')
			);
		}
	}
	console.log(`\nwrote ${path.relative(ROOT, path.join(OUT, 'ablation.png'))}`);
} finally {
	await browser.close();
	server.kill('SIGTERM');
}
