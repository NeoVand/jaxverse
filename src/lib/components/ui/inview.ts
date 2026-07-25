// Svelte action: fire once when the element scrolls into view. Demos use it
// to auto-load their engines/data — the reader never presses a Load button;
// the first button they see is the one that DOES something (Train, Play).
export function inview(node: HTMLElement, onVisible: () => void) {
	if (typeof IntersectionObserver === 'undefined') {
		onVisible();
		return;
	}
	const obs = new IntersectionObserver(
		(entries) => {
			for (const e of entries) {
				if (e.isIntersecting) {
					obs.disconnect();
					onVisible();
					return;
				}
			}
		},
		{ rootMargin: '160px 0px', threshold: 0.05 }
	);
	obs.observe(node);
	return {
		destroy() {
			obs.disconnect();
		}
	};
}
