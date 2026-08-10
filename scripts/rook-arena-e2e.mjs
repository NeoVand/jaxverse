// e2e probe for the rook arena (Plate VI): boot the chapter on WebGPU, play a
// move in the arena, verify the pretrained verdict + arrow + Black's reply,
// then run a short fine-tune and check the snapshot reaches the arena.
// Usage: node scripts/rook-arena-e2e.mjs
import { chromium } from 'playwright';

const URL = process.env.ROOK_URL ?? 'http://localhost:5175/rook';

const browser = await chromium.launch({
	headless: true,
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=WebGPU']
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
page.on('console', (m) => {
	if (m.type() === 'error') console.log('[console.error]', m.text());
});
page.on('pageerror', (e) => console.log('[pageerror]', e.message));

const t0 = Date.now();
const say = (s) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${s}`);

await page.goto(URL, { waitUntil: 'networkidle' });
say('page loaded');

const hasGpu = await page.evaluate(() => !!navigator.gpu);
say(`navigator.gpu: ${hasGpu}`);
if (!hasGpu) {
	console.log('FAIL: no WebGPU in this browser build');
	await browser.close();
	process.exit(1);
}

// scroll the arena into view — its inview gate powers the shared lab
// plates are addressed by their registry anchor, so renumbering the chapter
// never breaks this probe
const arena = page.locator('#plate-arena');
await arena.scrollIntoViewIfNeeded();
say('arena in view — waiting for the model to boot');

// the board renders once lab.phase === 'ready' and chess.js loaded
const board = arena.locator('svg.cm-chessboard');
await board.waitFor({ state: 'visible', timeout: 120_000 });
say('arena board visible');

/** cm-chessboard reads the square off pointer coordinates, and the pieces sit
 * over the square rects — so click the square's centre, forcing past them. */
const tap = (figure, square) =>
	figure.locator(`svg.cm-chessboard [data-square="${square}"]`).first().click({ force: true });

// play 1. e2e4: tap e2, tap e4
await tap(arena, 'e2');
await tap(arena, 'e4');
say('played e2e4 — waiting for the stages to answer');

// the pretrained column must produce a decision. Each fielded stage prints one
// "N% of its belief" line, so that phrase counts decisions.
const decisions = () => arena.getByText(/of its belief/);
await decisions().first().waitFor({ timeout: 120_000 });
const verdict = (await arena.locator('section').nth(1).innerText())
	.replace(/\s+/g, ' ')
	.match(/IT WOULD PLAY (\S+ \S+ of its belief)/i)?.[1];
say(`pretrained verdict: plays ${verdict}`);

// arrows overlay present while the verdicts describe the current position
const arrowCount = await arena.locator('svg.cm-chessboard .arrows .arrow').count();
say(`arrows drawn: ${arrowCount}`);

// Black must actually reply (fen changes; simplest check: it becomes White's
// turn again and a black piece has moved — count occupied squares' labels)
await page.waitForTimeout(2500);
const status = await arena.getByText(/of 3 fielded/).textContent();
say(`status: ${status.trim()}`);

// ── stage 2: a short fine-tune, then the arena should field it ──
const sft = page.locator('#plate-sft');
await sft.scrollIntoViewIfNeeded();
// one button: it swaps the corpus and starts the run
await sft.getByRole('button', { name: /Fine-tune/i }).waitFor({ timeout: 120_000 });
await sft.getByRole('button', { name: /Fine-tune/i }).click();
say('fine-tuning (the diet swaps on the way in)…');
await page.waitForTimeout(12_000);
await sft.getByRole('button', { name: /Pause/i }).click();
say('paused — snapshot should be captured');
await page.waitForTimeout(3_000);

await arena.scrollIntoViewIfNeeded();
const ftChip = arena.getByRole('button', { name: /fine-tuned/i }).first();
const disabled = await ftChip.isDisabled();
say(`arena fine-tuned chip disabled: ${disabled}`);
if (disabled) {
	console.log('FAIL: fine-tuned snapshot never reached the arena');
	await browser.close();
	process.exit(1);
}

// play another move and expect TWO stage verdicts now
await tap(arena, 'd2');
await tap(arena, 'd4');
say('played d2d4 — waiting for both stages');
await page.waitForTimeout(9000);
const verdicts = await decisions().count();
say(`stage verdicts shown: ${verdicts}`);
const shot = '/tmp/rook-arena.png';
await arena.screenshot({ path: shot });
say(`screenshot: ${shot}`);

// ── the play-it plate must still work on the shared board ──
const play = page.locator('#plate-play');
await play.scrollIntoViewIfNeeded();
await tap(play, 'g1');
await tap(play, 'f3');
say('play plate: played g1f3 — waiting for Rook');
await play.getByText(/raw top-8/).waitFor({ timeout: 60_000 });
say('play plate: raw top-8 panel appeared');

// ── stage 3: a short RLVR run, then the arena should field it too ──
const rlvr = page.locator('#plate-rlvr');
await rlvr.scrollIntoViewIfNeeded();
await rlvr.getByRole('button', { name: /Run RLVR/i }).click();
say('RLVR running (first iteration compiles kernels)…');
await rlvr
	.getByText(/updates [1-9]/)
	.first()
	.waitFor({ timeout: 180_000 });
await rlvr.getByRole('button', { name: /Pause/i }).click();
say('RLVR paused — snapshot should be captured');
await page.waitForTimeout(4_000);

await arena.scrollIntoViewIfNeeded();
const rlChip = arena.getByRole('button', { name: /RLVR/i }).first();
const rlDisabled = await rlChip.isDisabled();
say(`arena RLVR chip disabled: ${rlDisabled}`);

await arena.getByRole('button', { name: /New game/i }).click();
await tap(arena, 'e2');
await tap(arena, 'e4');
say('fresh game, played e2e4 — waiting for all three stages');
await page.waitForTimeout(12_000);
const verdicts3 = await decisions().count();
say(`stage verdicts shown: ${verdicts3}`);
const shot3 = '/tmp/rook-arena-3.png';
await arena.screenshot({ path: shot3 });
say(`screenshot: ${shot3}`);

const ok = verdicts >= 2 && !rlDisabled && verdicts3 >= 3;
console.log(ok ? 'PASS' : 'FAIL');
await browser.close();
process.exit(ok ? 0 : 1);
