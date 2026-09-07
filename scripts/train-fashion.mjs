// Drive the offline trainer: open the page in headless Chromium with WebGPU,
// let it run, and pull a checkpoint out every few minutes.
//
// The book ships the int8 file; the float32 one only exists so a run can be
// resumed after a crash or a laptop lid.
//
// Usage:
//   node scripts/train-fashion.mjs --objective flow --minutes 150
//   node scripts/train-fashion.mjs --objective eps  --minutes 90 --resume

import { spawn } from 'node:child_process';
import { mkdir, writeFile, readFile, access, rename, copyFile } from 'node:fs/promises';
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
// Measure throughput without letting a two-minute run near the real files.
const probe = argv.includes('--probe');
// A new architecture cannot resume the old checkpoint and must not overwrite
// it either: the book keeps serving the proven weights until the replacement
// has earned the slot. --tag gives a run its own pair of files.
const tag = flag('tag', '');
const name = `fashion-${objective}${tag ? `-${tag}` : ''}`;

const port = 5400 + Math.floor(Math.random() * 200);
const stamp = () => new Date().toISOString().slice(11, 19);

await mkdir(CKPT, { recursive: true });

const server = spawn(
	'npx',
	['vite', 'tools/fashion-trainer', '--port', String(port), '--strictPort'],
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
	// anything not passed falls back to the trainer page's own defaults
	const recipe = ['patch', 'dim', 'layers', 'heads', 'jitter', 'tsample', 'clip', 'ema']
		.map((k) => (flag(k, null) === null ? null : `&${k}=${flag(k, null)}`))
		.filter(Boolean)
		.join('');
	await page.goto(
		`http://localhost:${port}/?objective=${objective}&batch=${batch}&lr=${lr}&horizon=${horizon}${recipe}`,
		{ waitUntil: 'load' }
	);
	await page.waitForFunction(() => window.__state?.ready === true, null, { timeout: 180_000 });
	console.log(`${stamp()} trainer up (${objective}, batch ${batch}, lr ${lr})`);

	const f32 = path.join(CKPT, `${name}.f32`);
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

	// A checkpoint is only replaced by one that is demonstrably further along.
	// A run that wedges can still answer a snapshot call with a half-built or
	// stale buffer, and writing that over the resume file turns a bad hour into
	// a lost week. Every write lands on a temp path first and is read back
	// before it is allowed to become the real file.
	const stepOf = (buf) => {
		const len = buf.readUInt32BE(8);
		return JSON.parse(
			buf
				.subarray(12, 12 + len)
				.toString('utf8')
				.replace(/\0+$/, '')
		).steps;
	};
	const commit = async (dest, buf, floor) => {
		const at = stepOf(buf);
		if (!Number.isFinite(at) || at <= floor) {
			throw new Error(`refusing to write ${path.basename(dest)}: step ${at} is not past ${floor}`);
		}
		const tmp = `${dest}.tmp`;
		await writeFile(tmp, buf);
		if (stepOf(await readFile(tmp))) await rename(tmp, dest);
		return at;
	};

	let savedAt = -1;
	const save = async () => {
		if (probe) return 0;
		const f = Buffer.from(await page.evaluate(() => window.__snapshot('f32')), 'base64');
		const q = Buffer.from(await page.evaluate(() => window.__snapshot('i8')), 'base64');
		// keep one generation back, so a bad write is survivable by hand
		try {
			await copyFile(f32, `${f32}.bak`);
		} catch {
			/* first save of a fresh run */
		}
		savedAt = await commit(f32, f, savedAt);
		await commit(path.join(OUT, `${name}.bin`), q, -1);
		return q.length;
	};

	const deadline = Date.now() + minutes * 60_000;
	let lastSave = Date.now();
	let lastStep = 0;
	let stalls = 0;
	let stalled = false;
	while (Date.now() < deadline) {
		await new Promise((r) => setTimeout(r, 20_000));
		const st = await page.evaluate(() => window.__state);
		const left = Math.round((deadline - Date.now()) / 60_000);
		process.stdout.write(
			`\r${stamp()} step ${st.step} · loss ${st.loss.toFixed(4)} · ${left} min left   `
		);
		if (st.step === lastStep && ++stalls >= 6) {
			// Leave without saving. Whatever is on disk came from a loop that
			// was still moving, which is worth more than anything this one can
			// hand over now.
			console.log(`\n${stamp()} no progress in two minutes — stopping, keeping last good save`);
			exitCode = 1;
			stalled = true;
			break;
		}
		if (st.step !== lastStep) stalls = 0;
		lastStep = st.step;
		if (!probe && Date.now() - lastSave > 3 * 60_000) {
			const bytes = await save();
			lastSave = Date.now();
			console.log(
				`\n${stamp()} saved at step ${st.step} (${(bytes / 1024 / 1024).toFixed(2)} MB int8)`
			);
		}
	}
	await page.evaluate(() => window.__stop());
	const st = await page.evaluate(() => window.__state);
	if (stalled || probe) {
		console.log(`${stamp()} stopped at step ${st.step} without saving`);
	} else {
		const bytes = await save();
		console.log(
			`\n${stamp()} done: ${st.step} steps, loss ${st.loss.toFixed(4)}, ` +
				`static/data/${name}.bin ${(bytes / 1024 / 1024).toFixed(2)} MB`
		);
	}
} catch (e) {
	console.error(`\n${stamp()} failed:`, e);
	exitCode = 1;
} finally {
	await browser.close();
	server.kill('SIGTERM');
}
process.exit(exitCode);
