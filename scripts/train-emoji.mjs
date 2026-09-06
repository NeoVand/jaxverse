// Drive the offline trainer: open the page in headless Chromium with WebGPU,
// let it run, and pull a checkpoint out every few minutes.
//
// The book ships the int8 file; the float32 one only exists so a run can be
// resumed after a crash or a laptop lid.
//
// Usage:
//   node scripts/train-emoji.mjs --objective flow --minutes 150
//   node scripts/train-emoji.mjs --objective eps  --minutes 90 --resume

import { spawn } from 'node:child_process';
import { mkdir, writeFile, readFile, access } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const ROOT = new URL('..', import.meta.url).pathname;
const CKPT = path.join(ROOT, '.cache/ckpt');
const OUT = path.join(ROOT, 'static/data');

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
	const i = argv.indexOf(`--${name}`);
	return i === -1 ? fallback : argv[i + 1];
};
const objective = flag('objective', 'flow');
const minutes = Number(flag('minutes', 120));
const batch = Number(flag('batch', 32));
const lr = Number(flag('lr', 3e-4));
const horizon = Number(flag('horizon', 55000));
const resume = argv.includes('--resume');

const port = 5400 + Math.floor(Math.random() * 200);
const stamp = () => new Date().toISOString().slice(11, 19);

await mkdir(CKPT, { recursive: true });

const server = spawn(
	'npx',
	['vite', 'tools/emoji-trainer', '--port', String(port), '--strictPort'],
	{
		cwd: ROOT,
		stdio: 'ignore'
	}
);
const browser = await chromium.launch({
	headless: true,
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU']
});

let exitCode = 0;
try {
	await new Promise((r) => setTimeout(r, 3500));
	const page = await browser.newPage();
	page.on('pageerror', (e) => console.log(`${stamp()} PAGEERROR ${e}`));
	await page.goto(
		`http://localhost:${port}/?objective=${objective}&batch=${batch}&lr=${lr}&horizon=${horizon}`,
		{ waitUntil: 'load' }
	);
	await page.waitForFunction(() => window.__state?.ready === true, null, { timeout: 180_000 });
	console.log(`${stamp()} trainer up (${objective}, batch ${batch}, lr ${lr})`);

	const f32 = path.join(CKPT, `emoji-${objective}.f32`);
	if (resume) {
		try {
			await access(f32);
			const b64 = (await readFile(f32)).toString('base64');
			await page.evaluate((s) => window.__resume(s), b64);
			console.log(`${stamp()} resumed from ${path.relative(ROOT, f32)}`);
		} catch {
			console.log(`${stamp()} nothing to resume from, starting fresh`);
		}
	}

	const save = async () => {
		const b64 = await page.evaluate(() => window.__snapshot('f32'));
		await writeFile(f32, Buffer.from(b64, 'base64'));
		const q = await page.evaluate(() => window.__snapshot('i8'));
		const buf = Buffer.from(q, 'base64');
		await writeFile(path.join(OUT, `emoji-${objective}.bin`), buf);
		return buf.length;
	};

	const deadline = Date.now() + minutes * 60_000;
	let lastSave = Date.now();
	let lastStep = 0;
	let stalls = 0;
	while (Date.now() < deadline) {
		await new Promise((r) => setTimeout(r, 20_000));
		const st = await page.evaluate(() => window.__state);
		const left = Math.round((deadline - Date.now()) / 60_000);
		process.stdout.write(
			`\r${stamp()} step ${st.step} · loss ${st.loss.toFixed(4)} · ${left} min left   `
		);
		if (st.step === lastStep && ++stalls >= 6) {
			console.log(`\n${stamp()} no progress in two minutes — stopping`);
			exitCode = 1;
			break;
		}
		if (st.step !== lastStep) stalls = 0;
		lastStep = st.step;
		if (Date.now() - lastSave > 3 * 60_000) {
			const bytes = await save();
			lastSave = Date.now();
			console.log(
				`\n${stamp()} saved at step ${st.step} (${(bytes / 1024 / 1024).toFixed(2)} MB int8)`
			);
		}
	}
	await page.evaluate(() => window.__stop());
	const bytes = await save();
	const st = await page.evaluate(() => window.__state);
	console.log(
		`\n${stamp()} done: ${st.step} steps, loss ${st.loss.toFixed(4)}, ` +
			`static/data/emoji-${objective}.bin ${(bytes / 1024 / 1024).toFixed(2)} MB`
	);
} catch (e) {
	console.error(`\n${stamp()} failed:`, e);
	exitCode = 1;
} finally {
	await browser.close();
	server.kill('SIGTERM');
}
process.exit(exitCode);
