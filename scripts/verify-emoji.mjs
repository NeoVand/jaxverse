// Draw proof sheets from the two shipped checkpoints and write them to
// .cache/proof/, so the captions that make empirical claims about them can be
// checked against pictures rather than against hope.
//
// Usage: node scripts/verify-emoji.mjs

import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const ROOT = new URL('..', import.meta.url).pathname;
const OUT = path.join(ROOT, '.cache/proof');
const port = 5600 + Math.floor(Math.random() * 200);

await mkdir(OUT, { recursive: true });
const server = spawn(
	'npx',
	['vite', 'tools/checkpoint-proof', '--port', String(port), '--strictPort'],
	{
		cwd: ROOT,
		stdio: 'ignore'
	}
);
const browser = await chromium.launch({
	headless: true,
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU']
});

try {
	await new Promise((r) => setTimeout(r, 3500));
	const page = await browser.newPage();
	page.on('pageerror', (e) => console.log('PAGEERROR', String(e)));
	await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });

	const deadline = Date.now() + 600_000;
	let seen = '';
	while (Date.now() < deadline) {
		const text = await page
			.locator('#out')
			.innerText()
			.catch(() => seen);
		if (text !== seen) {
			process.stdout.write(text.slice(seen.length));
			seen = text;
		}
		if (/error:/i.test(seen) || (await page.evaluate(() => window.__done === true))) break;
		await new Promise((r) => setTimeout(r, 1500));
	}

	const sheets = await page.evaluate(() => window.__sheets ?? {});
	for (const [name, dataUrl] of Object.entries(sheets)) {
		const buf = Buffer.from(String(dataUrl).split(',')[1], 'base64');
		const file = path.join(OUT, `${name}.png`);
		await writeFile(file, buf);
		console.log(`\n  ${path.relative(ROOT, file)} (${(buf.length / 1024).toFixed(0)} KB)`);
	}
} finally {
	await browser.close();
	server.kill('SIGTERM');
}
