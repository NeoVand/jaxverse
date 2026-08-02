// One-time scaffolder for the downloadable labs: writes the boilerplate
// (package.json, tsconfig, index.html, README) for each lab. The interesting
// files — src/main.ts and friends — are written by hand and never touched
// here. Safe to re-run; it only overwrites boilerplate.

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const LABS = {
	descent: {
		title: 'Lab 0 · The Descent',
		blurb:
			'Five optimizer update rules descending the Rosenbrock valley — by hand, then with jax-js grad.',
		deps: { '@jax-js/jax': '^0.1.18' }
	},
	neuron: {
		title: 'Lab 1 · The Approximator',
		blurb:
			'A small MLP fits a sine wave live on a canvas: pytree params, valueAndGrad, optax Adam.',
		deps: { '@jax-js/jax': '^0.1.18', '@jax-js/optax': '^0.1.2' }
	},
	space: {
		title: 'Lab 2 · Bending Space',
		blurb: 'A two-moons classifier that prints its own decision boundary as ASCII while it trains.',
		deps: { '@jax-js/jax': '^0.1.18', '@jax-js/optax': '^0.1.2' }
	},
	digits: {
		title: 'Lab 3 · Telling Things Apart',
		blurb:
			'A real MNIST classifier: spritesheet decoding, minibatch cross-entropy, honest test accuracy.',
		deps: { '@jax-js/jax': '^0.1.18', '@jax-js/optax': '^0.1.2' }
	},
	latent: {
		title: 'Lab 4 · The Hidden Map',
		blurb:
			'A variational autoencoder on MNIST, printing its 2-D latent map as ASCII digit clusters.',
		deps: { '@jax-js/jax': '^0.1.18', '@jax-js/optax': '^0.1.2' }
	},
	language: {
		title: 'Lab 5 · The Next Token',
		blurb:
			'A tiny GPT trained on the story corpus, sampling text as the loss falls. WebGPU recommended.',
		deps: { '@jax-js/jax': '^0.1.18', '@jax-js/optax': '^0.1.2' }
	},
	reward: {
		title: 'Lab 6 · Learning from Reward',
		blurb:
			'REINFORCE on a gridworld in plain TypeScript — watch the policy arrows sharpen. No GPU, no tensors.',
		deps: {}
	},
	rook: {
		title: 'Lab 7 · Rook',
		blurb:
			'Pretrain a small chess language model on real games and watch legality emerge, judged by chess.js.',
		deps: { '@jax-js/jax': '^0.1.18', '@jax-js/optax': '^0.1.2', 'chess.js': '^1.4.0' }
	}
};

const root = new URL('..', import.meta.url).pathname;

for (const [slug, meta] of Object.entries(LABS)) {
	const dir = join(root, 'labs', slug);
	mkdirSync(join(dir, 'src'), { recursive: true });

	writeFileSync(
		join(dir, 'package.json'),
		JSON.stringify(
			{
				name: `jaxverse-lab-${slug}`,
				private: true,
				version: '1.0.0',
				type: 'module',
				scripts: { dev: 'vite', build: 'vite build', preview: 'vite preview' },
				dependencies: meta.deps,
				devDependencies: { typescript: '^5.6.0', vite: '^6.0.0' }
			},
			null,
			'\t'
		) + '\n'
	);

	writeFileSync(
		join(dir, 'tsconfig.json'),
		JSON.stringify(
			{
				compilerOptions: {
					target: 'ES2022',
					module: 'ESNext',
					moduleResolution: 'bundler',
					strict: true,
					noEmit: true,
					skipLibCheck: true,
					lib: ['ES2022', 'DOM', 'DOM.Iterable']
				},
				include: ['src']
			},
			null,
			'\t'
		) + '\n'
	);

	writeFileSync(
		join(dir, 'index.html'),
		`<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />
		<title>${meta.title}</title>
		<style>
			:root {
				color-scheme: light dark;
			}
			body {
				margin: 0 auto;
				max-width: 72ch;
				padding: 2rem 1.25rem 4rem;
				background: #faf9f5;
				color: #1d1c18;
				font-family: ui-monospace, 'SF Mono', Menlo, monospace;
				font-size: 13px;
				line-height: 1.55;
			}
			@media (prefers-color-scheme: dark) {
				body {
					background: #141310;
					color: #e9e6dc;
				}
			}
			h1 {
				font-size: 15px;
				font-weight: 600;
				letter-spacing: 0.02em;
			}
			p.blurb {
				opacity: 0.65;
			}
			pre {
				white-space: pre-wrap;
				word-break: break-word;
			}
			canvas {
				width: 100%;
				image-rendering: pixelated;
				border: 1px solid rgba(128, 128, 128, 0.3);
				border-radius: 6px;
			}
			a {
				color: inherit;
			}
		</style>
	</head>
	<body>
		<h1>${meta.title}</h1>
		<p class="blurb">
			${meta.blurb} From
			<a href="https://github.com/NeoVand/jaxverse">jaxverse</a> — the little universe of learning machines.
		</p>
		<canvas id="stage" hidden></canvas>
		<pre id="out">booting…</pre>
		<script type="module" src="/src/main.ts"></script>
	</body>
</html>
`
	);

	writeFileSync(
		join(dir, 'README.md'),
		`# ${meta.title}

${meta.blurb}

This is a standalone lab from [jaxverse](https://github.com/NeoVand/jaxverse). It has no build
requirements beyond Node:

\`\`\`sh
npm install
npm run dev
\`\`\`

then open the printed localhost URL. Everything of interest is in \`src/main.ts\` — small on
purpose, so you can change a number and see what happens.
`
	);
	console.log('scaffolded', slug);
}
