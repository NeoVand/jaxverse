// One-off: screenshot the plates touched by the consistency pass.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const OUT = 'shots';
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({
	headless: true,
	args: [
		'--enable-unsafe-webgpu',
		'--enable-features=Vulkan',
		'--use-angle=metal',
		'--disable-blink-features=AutomationControlled'
	]
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

async function shoot(url, figures) {
	await page.goto(`http://localhost:5173${url}`, { waitUntil: 'networkidle' });
	for (const { match, name, settle = 4500 } of figures) {
		const fig = page.locator('figure', { hasText: match }).first();
		await fig.scrollIntoViewIfNeeded();
		await page.waitForTimeout(settle);
		await fig.screenshot({ path: `${OUT}/${name}.png` });
		console.log('shot', name);
	}
}

await shoot('/rook', [
	{ match: 'Pretraining', name: 'rook-pretrain', settle: 12000 },
	{ match: 'Play it', name: 'rook-play', settle: 6000 },
	{ match: 'Fine-tuning', name: 'rook-sft', settle: 3000 },
	{ match: 'RLVR', name: 'rook-rlvr', settle: 3000 }
]);
await shoot('/reward', [{ match: 'The gridworld', name: 'gridworld', settle: 5000 }]);
await shoot('/space', [{ match: 'The experiment', name: 'spacelab', settle: 9000 }]);
await shoot('/digits', [
	{ match: 'softmax, by hand', name: 'softmaxplay', settle: 2500 },
	{ match: 'The classifier', name: 'classifier', settle: 8000 }
]);
await shoot('/language', [{ match: 'Words become vectors', name: 'wordvectors', settle: 12000 }]);
await shoot('/latent', [{ match: 'Search by drawing', name: 'neighbors', settle: 10000 }]);

await browser.close();
console.log('done');
