// A deliberately small TypeScript highlighter — five token classes, colored
// with the site's own CSS variables so code reads like part of the page
// rather than an embedded editor. No grammar dependency: comments, strings,
// keywords, numbers, and call-sites cover the excerpts we quote.

const KEYWORDS = new Set([
	'const',
	'let',
	'var',
	'function',
	'return',
	'if',
	'else',
	'for',
	'while',
	'of',
	'in',
	'new',
	'class',
	'extends',
	'import',
	'export',
	'from',
	'default',
	'async',
	'await',
	'type',
	'interface',
	'typeof',
	'null',
	'undefined',
	'true',
	'false',
	'this',
	'break',
	'continue',
	'switch',
	'case',
	'throw',
	'try',
	'catch',
	'finally',
	'yield',
	'void',
	'delete',
	'do',
	'as'
]);

function esc(s: string): string {
	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const TOKEN =
	/(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|(`(?:\\.|[^`\\])*`|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*")|(\b\d(?:[\d_]*\.?[\d_]*)(?:e[+-]?\d+)?\b)|([A-Za-z_$][\w$]*)/g;

/** Escape + tokenize `code` into HTML with hl-* span classes. */
export function highlight(code: string): string {
	let out = '';
	let last = 0;
	for (const m of code.matchAll(TOKEN)) {
		out += esc(code.slice(last, m.index));
		last = m.index + m[0].length;
		const [, comment, str, num, word] = m;
		if (comment) out += `<span class="hl-com">${esc(comment)}</span>`;
		else if (str) out += `<span class="hl-str">${esc(str)}</span>`;
		else if (num) out += `<span class="hl-num">${esc(num)}</span>`;
		else if (word) {
			if (KEYWORDS.has(word)) out += `<span class="hl-kw">${esc(word)}</span>`;
			else if (code[last] === '(') out += `<span class="hl-fn">${esc(word)}</span>`;
			else out += esc(word);
		}
	}
	return out + esc(code.slice(last));
}
