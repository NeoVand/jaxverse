// Reading progress, kept humbly in localStorage. Visited chapters and small
// per-chapter milestones (“trained the classifier past 90%”) — nothing more.

import { browser } from '$app/environment';

const KEY = 'jaxverse-progress-v1';

interface Stored {
	visited: string[];
	milestones: string[];
}

function load(): Stored {
	if (!browser) return { visited: [], milestones: [] };
	try {
		const raw = localStorage.getItem(KEY);
		if (!raw) return { visited: [], milestones: [] };
		const p = JSON.parse(raw) as Partial<Stored>;
		return {
			visited: Array.isArray(p.visited) ? p.visited : [],
			milestones: Array.isArray(p.milestones) ? p.milestones : []
		};
	} catch {
		return { visited: [], milestones: [] };
	}
}

class Progress {
	visited = $state<Set<string>>(new Set(load().visited));
	milestones = $state<Set<string>>(new Set(load().milestones));

	private save() {
		if (!browser) return;
		try {
			localStorage.setItem(
				KEY,
				JSON.stringify({ visited: [...this.visited], milestones: [...this.milestones] })
			);
		} catch {
			/* storage full or blocked — progress is a courtesy, not a requirement */
		}
	}

	visit(slug: string) {
		if (this.visited.has(slug)) return;
		this.visited = new Set([...this.visited, slug]);
		this.save();
	}

	reach(milestone: string) {
		if (this.milestones.has(milestone)) return;
		this.milestones = new Set([...this.milestones, milestone]);
		this.save();
	}

	reset() {
		this.visited = new Set();
		this.milestones = new Set();
		this.save();
	}
}

export const progress = new Progress();
