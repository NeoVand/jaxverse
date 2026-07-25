// Three-way theme: 'light' | 'dark' pinned by the user, or 'system' (default).
// The pre-paint script in app.html mirrors this logic — keep them in lockstep.

export type ThemePreference = 'light' | 'dark' | 'system';

const KEY = 'jaxverse-theme';

export function storedTheme(): ThemePreference {
	if (typeof localStorage === 'undefined') return 'system';
	try {
		const t = localStorage.getItem(KEY);
		return t === 'light' || t === 'dark' ? t : 'system';
	} catch {
		return 'system';
	}
}

export function applyTheme(pref: ThemePreference): void {
	const root = document.documentElement;
	root.classList.remove('light', 'dark');
	if (pref !== 'system') root.classList.add(pref);
	try {
		if (pref === 'system') localStorage.removeItem(KEY);
		else localStorage.setItem(KEY, pref);
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
