// Run a tools/ benchmark page in headless Chromium with WebGPU and print its log.
// Usage: node scripts/bench-denoiser.mjs [dir-under-tools] [timeoutSeconds]

import { spawn } from 'node:child_process';
import { chromium } from 'playwright';

const root = new URL('..', import.meta.url).pathname;
const slug = process.argv[2] ?? 'denoiser-bench';
const timeout = Number(process.argv[3] ?? 300) * 1000;
const port = 5300 + Math.floor(Math.random() * 300);

const server = spawn('npx', ['vite', `tools/${slug}`, '--port', String(port), '--strictPort'], {
	cwd: root,
	stdio: 'ignore'
});

const browser = await chromium.launch({
	headless: true,
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU']
});

try {
	await new Promise((r) => setTimeout(r, 3000));
	const page = await browser.newPage();
	page.on('pageerror', (e) => console.log('PAGEERROR', String(e)));
	page.on('console', (m) => {
		if (m.type() === 'error') console.log('CONSOLE', m.text());
	});
	await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });

	const deadline = Date.now() + timeout;
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
		if (/error:/i.test(seen) || /step 999 done/.test(seen)) break;
		await new Promise((r) => setTimeout(r, 1200));
	}
	console.log('\n--- end ---');
} finally {
	await browser.close();
	server.kill('SIGTERM');
}
