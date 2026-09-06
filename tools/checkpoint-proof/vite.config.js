import { fileURLToPath } from 'node:url';

export default {
	publicDir: fileURLToPath(new URL('../../static', import.meta.url)),
	resolve: {
		alias: { $lib: fileURLToPath(new URL('../../src/lib', import.meta.url)) }
	},
	server: {
		hmr: false,
		watch: { ignored: ['**/*'] },
		fs: { allow: [fileURLToPath(new URL('../..', import.meta.url))] }
	}
};
