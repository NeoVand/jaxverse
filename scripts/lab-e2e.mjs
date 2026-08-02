// Smoke test for the standalone labs: serves each one with vite (deps resolve
// up to the root node_modules), loads it in headless Chromium with WebGPU on,
// waits for training output, and fails on any 'error:' in the page.
// Usage: node scripts/lab-e2e.mjs [slug ...]

import { spawn } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { chromium } from 'playwright';

const root = new URL('..', import.meta.url).pathname;
const slugs = process.argv.slice(2).length
	? process.argv.slice(2)
	: readdirSync(`${root}/labs`).filter((d) => !d.startsWith('.'));

// run scripts/build-labs.mjs first so each lab's public/data is populated
const browser = await chromium.launch({
	headless: true,
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU']
});

let failed = 0;
for (const slug of slugs) {
	const port = 5190 + Math.floor(Math.random() * 200);
	const server = spawn('npx', ['vite', `labs/${slug}`, '--port', String(port), '--strictPort'], {
		cwd: root,
		stdio: 'ignore'
	});
	try {
		await new Promise((r) => setTimeout(r, 2500));
		const page = await browser.newPage();
		const errors = [];
		page.on('pageerror', (e) => errors.push(String(e)));
		await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });
		// wait until the lab reports progress (a step/episode line) or errors out
		// progress = any step/episode counter beyond 1 (some labs update one
		// status line in place rather than appending)
		const progressed = (t) =>
			[...t.matchAll(/(?:step|episode|t=)\s*(\d+)/g)].some((m) => Number(m[1]) > 1);
		const deadline = Date.now() + 150_000;
		let text = '';
		while (Date.now() < deadline) {
			try {
				text = await page.locator('#out').innerText({ timeout: 60_000 });
			} catch {
				continue; // main thread busy with synchronous dispatches — retry
			}
			if (/error:/i.test(text) || errors.length) break;
			if (progressed(text)) break;
			await new Promise((r) => setTimeout(r, 1500));
		}
		const bad = /error:/i.test(text) || errors.length > 0;
		const gotProgress = progressed(text);
		if (bad || !gotProgress) {
			failed++;
			console.log(`✗ ${slug}`);
			console.log('  page errors:', errors.slice(0, 2));
			console.log('  out:', text.split('\n').slice(0, 8).join(' ⏎ '));
		} else {
			console.log(
				`✓ ${slug} — ${text
					.split('\n')
					.filter((l) => l.trim())
					.slice(-1)[0]
					.slice(0, 90)}`
			);
		}
		await page.close();
	} finally {
		server.kill('SIGTERM');
	}
}
await browser.close();
process.exit(failed ? 1 : 0);
