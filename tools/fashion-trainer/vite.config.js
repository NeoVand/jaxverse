import { fileURLToPath } from 'node:url';

// HMR is off on purpose. This config serves a page that trains for hours, and
// it imports the same src/lib/diffusion modules that are still being edited —
// with the watcher on, saving any of them reloads the tab and throws the run
// away. Editing during a run is now inert; restart to pick changes up.
export default {
	publicDir: fileURLToPath(new URL('../../static', import.meta.url)),
	resolve: {
		alias: {
			$lib: fileURLToPath(new URL('../../src/lib', import.meta.url))
		}
	},
	server: {
		hmr: false,
		watch: { ignored: ['**/*'] },
		fs: { allow: [fileURLToPath(new URL('../..', import.meta.url))] }
	}
};
