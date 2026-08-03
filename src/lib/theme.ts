// Three-way theme: 'light' | 'dark' | 'system', with dark as the default when
// nothing has been chosen. 'system' is an explicit choice and is stored too.
// The pre-paint script in app.html mirrors this logic — keep them in lockstep.

export type ThemePreference = 'light' | 'dark' | 'system';

const KEY = 'jaxverse-theme';

export function storedTheme(): ThemePreference {
	if (typeof localStorage === 'undefined') return 'dark';
	try {
		const t = localStorage.getItem(KEY);
		return t === 'light' || t === 'system' ? t : 'dark';
	} catch {
		return 'dark';
	}
}

export function applyTheme(pref: ThemePreference): void {
	const root = document.documentElement;
	root.classList.remove('light', 'dark');
	if (pref !== 'system') root.classList.add(pref);
	try {
		localStorage.setItem(KEY, pref);
	} catch {
		/* private mode — theme stays for this page only */
	}
}

/** The theme currently in effect, resolving 'system' to what the OS says. */
export function effectiveTheme(): 'light' | 'dark' {
	const pref = storedTheme();
	if (pref !== 'system') return pref;
	return typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches
		? 'dark'
		: 'light';
}
