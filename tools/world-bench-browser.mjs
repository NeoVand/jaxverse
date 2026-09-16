/** Run the shipped browser core through Vite; no surrogate model implementation. */
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import ts from 'typescript';
import { createHash } from 'node:crypto';

const options = JSON.parse(process.argv[2] ?? '{}');
const origin = process.env.WORLD_BENCH_ORIGIN ?? 'http://127.0.0.1:5176';
const output = process.argv[3] ?? '/tmp/world-bench-result.json';
const root = path.resolve(import.meta.dirname, '..');
const args = ['--enable-unsafe-webgpu', '--enable-features=WebGPU'];
if (process.platform === 'darwin') args.push('--use-angle=metal');
const browser = await chromium.launch({ headless: true, args });
try {
	const page = await browser.newPage();
	// Research-only module substitution; the chapter and its default model stay unchanged.
	const modelPath = process.env.WORLD_BENCH_MODEL;
	const modelSource = await fs.readFile(
		modelPath ?? path.join(root, 'src/lib/world/model.ts'),
		'utf8'
	);
	const modelHash = createHash('sha256').update(modelSource).digest('hex');
	if (modelPath) {
		const modelCode = ts
			.transpileModule(modelSource, {
				compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext }
			})
			.outputText.replace(
				/from ['"]@jax-js\/jax['"]/g,
				`from '${origin}/node_modules/.vite/deps/@jax-js_jax.js'`
			);
		await page.route('**/src/lib/world/model.ts*', (route) =>
			route.fulfill({
				contentType: 'application/javascript',
				body: modelCode
			})
		);
	}
	page.on('console', (message) => console.log(message.text()));
	page.on('pageerror', (error) => console.error(error));
	// A blank same-origin document avoids chapter rendering and Vite HMR reloads.
	await page.route('**/__world-bench', (route) =>
		route.fulfill({
			contentType: 'text/html',
			body: '<!doctype html><title>World benchmark</title>'
		})
	);
	await page.goto(`${origin}/__world-bench`);
	let source = ts.transpileModule(
		await fs.readFile(path.join(root, 'tools/world-bench.ts'), 'utf8'),
		{
			compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext }
		}
	).outputText;
	source = source.replaceAll("'@jax-js/jax'", `'${origin}/node_modules/.vite/deps/@jax-js_jax.js'`);
	for (const file of ['engine', 'sensor', 'simulator'])
		source = source.replaceAll(
			`'../src/lib/world/${file}'`,
			`'${origin}/src/lib/world/${file}.ts'`
		);
	const result = await page.evaluate(
		async ({ url, options }) => (await import(url)).runWorldBench(options),
		{ url: `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`, options }
	);
	await fs.writeFile(
		output,
		JSON.stringify(
			{
				options,
				model: { file: modelPath ? path.basename(modelPath) : 'shipped', sha256: modelHash },
				...result
			},
			null,
			2
		)
	);
	console.log(`Saved ${output}`);
} finally {
	await browser.close();
}
