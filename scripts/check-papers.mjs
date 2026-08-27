// Ask every source in the bibliography whether it is still there.
//
// A citation is a promise that the reader can go and check, and the web keeps
// breaking that promise quietly — a course page is reorganised, an author
// moves university, a preprint server changes its URL scheme, and a chapter
// ends up pointing at a 404 that nobody notices because nobody clicks.
//
// This is deliberately NOT part of `npm test`: it depends on the whole
// internet being up and on hosts that rate-limit, so a red run here would
// mean "someone's server is slow" as often as it means "fix this link". Run
// it by hand when you touch the bibliography, and before a release:
//
//     node scripts/check-papers.mjs
//
// Some hosts refuse a bare script — JSTOR, Springer and Semantic Scholar all
// serve a challenge page to anything that does not look like a browser — so
// this sends a browser's User-Agent and treats 2xx as the only success. A
// host that answers 403 to everything is reported as UNSURE rather than
// FAILED: check it by eye instead of chasing it.

import { readFileSync } from 'node:fs';

const UA =
	'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36';
const TIMEOUT_MS = 30_000;
/** Polite: these are other people's servers, and several are university boxes. */
const CONCURRENCY = 4;
/** Long enough that a host which just dropped us has moved on. */
const RETRY_PAUSE_MS = 2_000;

/** Pull the id/url pairs straight out of the registry's source text.
 *  Importing it would need a TypeScript loader for no benefit — the shape is
 *  a flat object literal, and a regex over it cannot go stale silently
 *  because a miscount shows up as a wrong total on the first line. */
function readPapers() {
	const src = readFileSync(new URL('../src/lib/data/papers.ts', import.meta.url), 'utf8');
	const body = src.slice(src.indexOf('export const papers'), src.indexOf('export type PaperId'));
	const out = [];
	const entry = /'([a-z0-9-]+)':\s*\{/g;
	for (let m; (m = entry.exec(body));) {
		const rest = body.slice(m.index);
		const url = /url:\s*'([^']+)'/.exec(rest.slice(0, rest.indexOf('\n\t},')));
		if (url) out.push({ id: m[1], url: url[1] });
	}
	return out;
}

/** One request. `answered` distinguishes "the server said something" from
 *  "the connection went nowhere", which is the only distinction worth
 *  retrying on. */
async function ask(url) {
	const ctl = new AbortController();
	const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
	try {
		// HEAD is cheaper but a surprising number of these hosts answer 405 to
		// it, so ask for the document and drop the body.
		const res = await fetch(url, {
			headers: { 'user-agent': UA, accept: '*/*' },
			redirect: 'follow',
			signal: ctl.signal
		});
		res.body?.cancel();
		if (res.ok) return { answered: true, state: 'ok', detail: String(res.status) };
		if (res.status === 403 || res.status === 429 || res.status === 202)
			return {
				answered: true,
				state: 'unsure',
				detail: `${res.status} — bot check, verify by eye`
			};
		return { answered: true, state: 'failed', detail: String(res.status) };
	} catch (err) {
		const why = err.name === 'AbortError' ? `no answer in ${TIMEOUT_MS / 1000}s` : err.message;
		return { answered: false, state: 'failed', detail: why };
	} finally {
		clearTimeout(timer);
	}
}

/** Archives and university boxes drop perfectly good connections under any
 *  load at all — the Internet Archive does it to roughly two requests in five
 *  here. A dropped connection is a fact about the minute, not about the link,
 *  so it gets one more try; an HTTP status is an answer and never does. */
async function check({ id, url }) {
	let r = await ask(url);
	if (!r.answered) {
		await new Promise((done) => setTimeout(done, RETRY_PAUSE_MS));
		r = await ask(url);
		if (r.state === 'ok') r.detail += ' (second try)';
	}
	return { id, url, state: r.state, detail: r.detail };
}

const papers = readPapers();
console.log(`checking ${papers.length} sources…\n`);

const queue = papers.slice();
const results = [];
await Promise.all(
	Array.from({ length: CONCURRENCY }, async () => {
		for (let next; (next = queue.shift());) {
			const r = await check(next);
			results.push(r);
			const mark = r.state === 'ok' ? '  ok  ' : r.state === 'unsure' ? ' ?    ' : ' FAIL ';
			console.log(`${mark} ${r.id.padEnd(26)} ${r.detail}`);
		}
	})
);

const failed = results.filter((r) => r.state === 'failed');
const unsure = results.filter((r) => r.state === 'unsure');
console.log(
	`\n${results.length - failed.length - unsure.length} reachable · ${unsure.length} unsure · ${failed.length} broken`
);
for (const r of failed) console.log(`  broken: ${r.id} → ${r.url}`);
process.exitCode = failed.length ? 1 : 0;
