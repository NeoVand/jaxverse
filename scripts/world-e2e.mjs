// Actual chapter flow: browser training, readout, latent planning, comparison,
// reset, and worker disposal. Run against a stable dev or production server:
// node scripts/world-e2e.mjs http://localhost:5173
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://localhost:5173';
const browser = await chromium.launch({
	headless: true,
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU']
});
const page = await browser.newPage({
	viewport: { width: 1280, height: 900 },
	reducedMotion: 'reduce'
});
const errors = [];
let stage = 'loading';
const progress = setInterval(async () => {
	try {
		const status = await page.locator('#plate-train .plate-head').innerText({ timeout: 2000 });
		console.log(`${stage}: ${status.replace(/\s+/g, ' ')}`);
	} catch {
		/* Navigation may replace the chapter. */
	}
}, 15_000);
page.on('pageerror', (error) => errors.push(String(error)));
page.on('framenavigated', (frame) => {
	if (frame === page.mainFrame()) console.log(`Navigation: ${frame.url()}`);
});
page.on('response', (response) => {
	if (response.status() >= 400 && response.url().includes('/_app/'))
		errors.push(`Application asset failed: ${response.status()} ${response.url()}`);
});
const plate = (name) => page.locator(`#plate-${name}`);
const button = (name, text) => plate(name).getByRole('button', { name: text, exact: true });
const report = async (name) =>
	console.log(`${name}: ${(await plate(name).innerText()).replace(/\s+/g, ' ')}`);
try {
	await page.goto(`${base}/world`, { waitUntil: 'networkidle' });
	assert.deepEqual(errors, [], 'The built application must load before testing interactions');
	assert.equal(await page.locator('main select').count(), 0);
	assert.equal(await page.locator('.architecture').count(), 1);
	assert.ok((await page.locator('main .math-display .eq-model').count()) > 0);
	await plate('train').scrollIntoViewIfNeeded();
	await button('train', 'Train').waitFor();
	stage = 'training';
	await button('train', 'Train').click({ timeout: 120_000 });
	await page.waitForFunction(
		() => {
			const p = document.querySelector('#plate-train');
			return (
				p?.textContent.includes('step 5000') &&
				[...p.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Train more')
			);
		},
		null,
		{ timeout: 180_000 }
	);
	await report('train');
	stage = 'forecasting';
	assert.match(await plate('train').innerText(), /152,464 parameters/);
	assert.match(await plate('train').innerText(), /16,384 collected transitions/);

	await button('forecast', 'Forecast & replay').click();
	await page.waitForFunction(
		() => document.querySelector('#plate-forecast')?.textContent.includes('Rollout ghost error'),
		null,
		{ timeout: 120_000 }
	);
	await report('forecast');
	stage = 'planning';
	await button('plan', 'Rehearse').click();
	await page.waitForFunction(
		() => document.querySelector('#plate-plan')?.textContent.includes('plan cost'),
		null,
		{ timeout: 60_000 }
	);
	await report('plan');
	await button('plan', 'Step').click();
	await page.waitForFunction(() =>
		document.querySelector('#plate-plan .plan-readout')?.textContent.includes('0.2 s')
	);
	assert.doesNotMatch(await plate('plan').locator('.plan-readout').innerText(), /plan cost/);
	await button('plan', 'Run').click();
	await page.waitForFunction(
		() => {
			const p = document.querySelector('#plate-plan');
			return (
				p?.textContent.includes('Within 0.15 rad for five') ||
				p?.querySelector('.plan-readout')?.textContent.includes('14.6 s')
			);
		},
		null,
		{ timeout: 180_000 }
	);
	await report('plan');

	stage = 'changing the goal';
	await plate('transfer')
		.getByRole('group', { name: 'Desired pose', exact: true })
		.getByRole('button', { name: 'Curl', exact: true })
		.click();
	await button('transfer', 'Run').click();
	await page.waitForFunction(
		() => {
			const p = document.querySelector('#plate-transfer');
			return (
				p?.textContent.includes('Within 0.15 rad for five') ||
				p?.querySelector('.plan-readout')?.textContent.includes('14.4 s')
			);
		},
		null,
		{ timeout: 180_000 }
	);
	await report('transfer');
	assert.match(await plate('transfer').innerText(), /Within 0.15 rad for five/);
	assert.match(await plate('transfer').innerText(), /weights frozen · step 5000/);

	stage = 'comparing objectives';
	await button('collapse', 'Compare').click();
	await page.waitForFunction(
		() => {
			const p = document.querySelector('#plate-collapse');
			const progress = p?.querySelector('progress');
			const note = p?.querySelectorAll('.checkpoint-label')[1]?.textContent ?? '';
			return (
				progress &&
				progress.value > 0 &&
				progress.value < progress.max &&
				/Measured so far/.test(note)
			);
		},
		null,
		{ timeout: 30_000 }
	);
	assert.match(
		await plate('collapse').locator('.checkpoint-label').first().innerText(),
		/Frozen · 5,000 updates/
	);
	await button('collapse', 'Pause').click();
	await button('collapse', 'Restart comparison').waitFor({ timeout: 30_000 });
	assert.equal(await plate('collapse').locator('svg rect').count(), 48);
	assert.doesNotMatch(await plate('collapse').innerText(), /First measurement on its way/);
	if (process.env.WORLD_EVIDENCE_DIR)
		await plate('collapse').screenshot({
			path: `${process.env.WORLD_EVIDENCE_DIR}/comparison-progress.png`
		});
	await button('collapse', 'Restart comparison').click();
	await page.waitForFunction(
		() =>
			document.querySelector('#plate-collapse')?.textContent.includes('Both runs: 5,000 updates'),
		null,
		{ timeout: 180_000 }
	);
	await report('collapse');
	if (process.env.WORLD_EVIDENCE_DIR)
		await plate('collapse').screenshot({
			path: `${process.env.WORLD_EVIDENCE_DIR}/comparison-complete.png`
		});
	stage = 'reset and disposal';
	await button('train', 'Reset').click();
	await page.waitForFunction(() =>
		document.querySelector('#plate-train')?.textContent.includes('from random weights')
	);
	assert.doesNotMatch(await plate('forecast').innerText(), /Rollout ghost error/);
	assert.doesNotMatch(await plate('plan').innerText(), /plan cost/);
	await button('train', 'Train').click();
	await page.waitForFunction(() =>
		/step [1-9]\d/.test(document.querySelector('#plate-train .plate-head')?.textContent ?? '')
	);
	await button('train', 'Pause').click();
	await button('train', 'Train more').waitFor({ timeout: 30_000 });
	await report('train');
	await button('train', 'Train more').click();
	// Client-side navigation must dispose an active worker without rejected promises.
	await page.locator('a[href="/epilogue"]').last().click();
	await page.waitForURL('**/epilogue');
	await page.waitForLoadState('networkidle');
	assert.deepEqual(errors, []);
	console.log(
		'PASS: train → forecast → rehearse → act → new goal → compare → reset → pause → dispose'
	);
} catch (error) {
	console.error(
		await page
			.locator('[id^="plate-"] .plate-head, [role="alert"]')
			.allTextContents()
			.catch(() => 'Page unavailable')
	);
	throw error;
} finally {
	clearInterval(progress);
	await browser.close();
}
