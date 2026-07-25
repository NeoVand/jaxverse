// Renders the social-card image (static/og.png, 1200×630): the descent mark
// on warm near-black, wordmark in the system serif. Run: node scripts/make-og.mjs
import sharp from 'sharp';
import { writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const outDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'static');

// the same double-well the landing page animates
const raw = (x) => {
	const t = x * 3.9 - 1.95;
	return t * t * t * t - 2.6 * t * t + 0.9 * t;
};
let lo = Infinity;
let hi = -Infinity;
for (let i = 0; i <= 200; i++) {
	const v = raw(i / 200);
	lo = Math.min(lo, v);
	hi = Math.max(hi, v);
}
const f = (x) => 0.08 + (0.8 * (raw(x) - lo)) / (hi - lo);

const W = 1200;
const H = 630;
const pad = 90;
const curveTop = 300;
const curveH = 240;
const px = (u) => pad + u * (W - 2 * pad);
const py = (v) => curveTop + curveH - v * curveH;

let d = '';
for (let i = 0; i <= 160; i++) {
	const u = i / 160;
	d += `${i === 0 ? 'M' : 'L'}${px(u).toFixed(1)} ${py(f(u)).toFixed(1)} `;
}
// the ball rests in the deeper well (find argmin)
let bx = 0;
let bv = Infinity;
for (let i = 0; i <= 400; i++) {
	const v = f(i / 400);
	if (v < bv) {
		bv = v;
		bx = i / 400;
	}
}

const svg = `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">
  <rect width="${W}" height="${H}" fill="#141310"/>
  <text x="${pad}" y="150" font-family="Georgia, 'Times New Roman', serif" font-style="italic" font-size="96" fill="#e9e6dc">jaxverse</text>
  <text x="${pad}" y="218" font-family="Georgia, 'Times New Roman', serif" font-style="italic" font-size="34" fill="#a29e91">a little universe of learning machines — trained live in your browser</text>
  <path d="${d}" fill="none" stroke="#93a3ff" stroke-width="5" stroke-linecap="round"/>
  <circle cx="${px(bx)}" cy="${py(f(bx))}" r="16" fill="#ff8e57"/>
  <text x="${pad}" y="${H - 48}" font-family="Georgia, serif" font-style="italic" font-size="24" fill="#6c695e">gradient descent, running now — the only trick this whole book needs</text>
</svg>`;

writeFileSync(join(outDir, 'og.svg.tmp'), svg);
await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toFile(join(outDir, 'og.png'));
const { unlinkSync } = await import('node:fs');
unlinkSync(join(outDir, 'og.svg.tmp'));
console.log('wrote static/og.png');
