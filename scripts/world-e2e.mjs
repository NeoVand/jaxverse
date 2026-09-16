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
const capture = async (name, label) => {
	if (process.env.WORLD_EVIDENCE_DIR)
		await plate(name).screenshot({ path: `${process.env.WORLD_EVIDENCE_DIR}/${label}.png` });
};
try {
	await page.goto(`${base}/world`, { waitUntil: 'networkidle' });
	assert.deepEqual(errors, [], 'The built application must load before testing interactions');
	assert.equal(await page.locator('main select').count(), 0);
	assert.equal(await page.locator('.architecture').count(), 1);
	assert.ok((await page.locator('main .math-display .eq-model').count()) > 0);
	await plate('train').scrollIntoViewIfNeeded();
	await button('train', 'Train').waitFor();
	await plate('train').locator('.candidate').first().waitFor({ timeout: 120_000 });
	assert.equal(await plate('train').locator('button.score-view').count(), 0);
	const lossCurve = plate('train').getByRole('img', { name: /Training objective terms/ });
	assert.ok(
		await lossCurve.isVisible(),
		'Training loss is visible before training without opening details'
	);
	assert.equal(await lossCurve.locator('xpath=ancestor::details').count(), 0);
	const sensor = plate('train').locator('.observation canvas').first();
	const rawPixels = await sensor.evaluate((canvas) => canvas.toDataURL());
	for (const [preference, system, inverted] of [
		['dark', 'light', true],
		['light', 'dark', false],
		['system', 'dark', true],
		['system', 'light', false]
	]) {
		await page.emulateMedia({ colorScheme: system, reducedMotion: 'reduce' });
		await page.evaluate((preference) => {
			document.documentElement.classList.remove('light', 'dark');
			if (preference !== 'system') document.documentElement.classList.add(preference);
		}, preference);
		// Emulated system-media changes are applied on the browser's rendering tick.
		await page.waitForFunction(
			(expected) =>
				getComputedStyle(document.querySelector('#plate-train .observation canvas')).filter ===
				expected,
			inverted ? 'invert(1)' : 'none'
		);
		assert.equal(
			await sensor.evaluate((canvas) => getComputedStyle(canvas).filter),
			inverted ? 'invert(1)' : 'none',
			`${preference} preference on a ${system} system`
		);
		assert.equal(
			await sensor.evaluate((canvas) => canvas.toDataURL()),
			rawPixels,
			'Theme changes preserve the model-input pixels'
		);
	}
	const initialScore = Number.parseInt(
		await plate('train').locator('.score-value').first().innerText()
	);
	assert.equal(await plate('train').locator('.candidate').count(), 6);
	await capture('train', 'learning-before');
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
	assert.ok(await lossCurve.isVisible(), 'Training loss stays visible after learning');
	assert.ok((await lossCurve.locator('.prediction-path').getAttribute('d')).length > 0);
	assert.match(await plate('train').locator('.plate-head').innerText(), /loss \d+\.\d+/);
	const learnedScore = Number.parseInt(
		await plate('train').locator('.score-value').nth(1).innerText()
	);
	assert.ok(learnedScore > initialScore, 'The same held-out matching test improves with learning');
	assert.equal(
		Number.parseInt(await plate('train').locator('.score-value').first().innerText()),
		initialScore,
		'The before-learning checkpoint stays fixed'
	);
	await plate('train').getByRole('button', { name: 'Before learning', exact: false }).click();
	assert.ok(await plate('train').locator('.candidate.before.chosen').count());
	await plate('train').getByRole('button', { name: /^Now/ }).click();
	await plate('train').getByRole('button', { name: 'Copy the present', exact: true }).click();
	assert.match(
		await plate('train').locator('.method-note').innerText(),
		/without using the predictor or actions/
	);
	await plate('train').getByRole('button', { name: 'Example 2', exact: true }).click();
	assert.equal(
		await plate('train')
			.getByRole('button', { name: 'Example 2', exact: true })
			.getAttribute('aria-pressed'),
		'true'
	);
	await plate('train').getByRole('button', { name: 'Example 1', exact: true }).click();
	await plate('train').getByRole('button', { name: 'Now', exact: true }).click();
	for (const width of [1280, 640, 390]) {
		await page.setViewportSize({ width, height: 900 });
		for (const theme of ['light', 'dark']) {
			await page.evaluate((theme) => {
				document.documentElement.classList.remove('light', 'dark');
				document.documentElement.classList.add(theme);
			}, theme);
			await page.waitForFunction(
				(expected) =>
					getComputedStyle(document.querySelector('#plate-train .observation canvas')).filter ===
					expected,
				theme === 'dark' ? 'invert(1)' : 'none'
			);
			assert.equal(
				await sensor.evaluate((canvas) => getComputedStyle(canvas).filter),
				theme === 'dark' ? 'invert(1)' : 'none'
			);
			assert.ok(
				await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
				`No overflow at ${width}px`
			);
			await capture('train', `learning-${width}-${theme}`);
		}
	}
	await page.setViewportSize({ width: 1280, height: 900 });
	await page.evaluate(() => {
		document.documentElement.classList.remove('dark');
		document.documentElement.classList.add('light');
	});
	stage = 'forecasting';
	assert.match(await plate('train').innerText(), /152,464 parameters/);
	assert.match(await plate('train').innerText(), /16,384 collected transitions/);

	const forecastStart = await plate('forecast')
		.locator('.instrument polygon.arm')
		.evaluateAll((arms) => arms.map((arm) => arm.getAttribute('points')));
	await button('forecast', 'Forecast & replay').click();
	await page.waitForFunction(
		() => document.querySelector('#plate-forecast')?.textContent.includes('Rollout ghost error'),
		null,
		{ timeout: 120_000 }
	);
	await report('forecast');
	await capture('forecast', 'forecast-actions');
	const pushGhost = await plate('forecast').locator('.ghost-line').getAttribute('d');
	await plate('forecast').getByRole('button', { name: 'Reverse torques', exact: true }).click();
	assert.deepEqual(
		await plate('forecast')
			.locator('.instrument polygon.arm')
			.evaluateAll((arms) => arms.map((arm) => arm.getAttribute('points'))),
		forecastStart,
		'Every action choice has the same moving start'
	);
	await button('forecast', 'Forecast & replay').click();
	await plate('forecast').getByText('Rollout ghost error', { exact: false }).waitFor();
	assert.notEqual(
		await plate('forecast').locator('.ghost-line').getAttribute('d'),
		pushGhost,
		'Changing actions changes the prediction'
	);
	for (const horizon of [3, 20]) {
		await plate('forecast').getByRole('button', { name: 'Release', exact: true }).click();
		await plate('forecast')
			.getByRole('group', { name: 'Forecast length', exact: true })
			.getByRole('button', { name: String(horizon), exact: true })
			.click();
		const start = await plate('forecast')
			.locator('.instrument polygon.arm')
			.evaluateAll((arms) => arms.map((arm) => arm.getAttribute('points')));
		assert.deepEqual(start, forecastStart, 'Changing the horizon preserves the starting history');
		await page.emulateMedia({ reducedMotion: horizon === 20 ? 'no-preference' : 'reduce' });
		await button('forecast', 'Forecast & replay').click();
		await page.waitForFunction(
			(initial) => {
				const arms = [...document.querySelectorAll('#plate-forecast .instrument polygon.arm')];
				return (
					arms.length === 2 && arms.some((arm, i) => arm.getAttribute('points') !== initial[i])
				);
			},
			start,
			{ timeout: 30_000 }
		);
		await page.waitForFunction(
			() =>
				[...document.querySelectorAll('#plate-forecast button')].some(
					(button) => button.textContent.trim() === 'Forecast & replay' && !button.disabled
				),
			null,
			{ timeout: 30_000 }
		);
		await capture('forecast', `release-${horizon}`);
	}
	await page.emulateMedia({ reducedMotion: 'reduce' });
	stage = 'planning';
	await button('plan', 'Rehearse').click();
	await page.waitForFunction(
		() => document.querySelector('#plate-plan')?.textContent.includes('plan cost'),
		null,
		{ timeout: 60_000 }
	);
	await report('plan');
	assert.equal(await plate('plan').locator('.future').count(), 3);
	await capture('plan', 'planning-futures');
	await button('plan', 'Step').click();
	await page.waitForFunction(() =>
		document.querySelector('#plate-plan .plan-readout')?.textContent.includes('0.2 s')
	);
	assert.doesNotMatch(await plate('plan').locator('.plan-readout').innerText(), /plan cost/);
	assert.match(await plate('plan').locator('.action-evidence').innerText(), /After one action/);
	assert.equal(await plate('plan').locator('.instrument .ghost-line').count(), 1);
	await capture('plan', 'prediction-and-observation');
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
	await button('transfer', 'Rehearse').click();
	await plate('transfer').locator('.future').first().waitFor({ timeout: 60_000 });
	const cachedPaths = await plate('transfer')
		.locator('.candidate-futures .prediction, .candidate-futures .tip-trail')
		.evaluateAll((paths) => paths.map((path) => path.getAttribute('d')));
	const cachedCosts = await plate('transfer').locator('.score-label .num').allTextContents();
	await plate('transfer')
		.getByRole('group', { name: 'Desired pose', exact: true })
		.getByRole('button', { name: 'Curl', exact: true })
		.click();
	assert.deepEqual(
		await plate('transfer')
			.locator('.candidate-futures .prediction, .candidate-futures .tip-trail')
			.evaluateAll((paths) => paths.map((path) => path.getAttribute('d'))),
		cachedPaths,
		'Changing the goal preserves the predicted futures'
	);
	assert.notDeepEqual(
		await plate('transfer').locator('.score-label .num').allTextContents(),
		cachedCosts,
		'Changing the goal changes their costs'
	);
	assert.match(await plate('transfer').innerText(), /cached futures rescored/);
	await capture('transfer', 'same-futures-new-goal');
	await plate('transfer').getByRole('button', { name: 'Match the pose', exact: true }).click();
	assert.deepEqual(
		await plate('transfer')
			.locator('.candidate-futures .prediction, .candidate-futures .tip-trail')
			.evaluateAll((paths) => paths.map((path) => path.getAttribute('d'))),
		cachedPaths,
		'Changing intention also preserves predictions'
	);
	await plate('transfer').getByRole('button', { name: 'Arrive and remain', exact: true }).click();
	await button('transfer', 'Step').click();
	await plate('transfer').locator('.action-evidence').waitFor();
	await button('transfer', 'Reset scene').click();
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
	assert.equal(await plate('plan').locator('.action-evidence').count(), 0);
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
