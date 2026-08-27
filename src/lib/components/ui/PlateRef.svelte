<script lang="ts">
	import { getChapter, plateAnchor, plateLabel } from '$lib/data/plates';

	// "Plate VII", written by the registry rather than by hand, and clickable.
	// Prose never spells a numeral, so renumbering a chapter can't leave a
	// sentence pointing at the wrong figure.
	//
	// `lower` lowercases the word and only the word: the numeral is roman, and
	// toLowerCase() on the whole label renders "Plate III" as "plate iii",
	// which reads as a typo rather than as a cross-reference.
	interface Props {
		id: string;
		/** Mid-sentence use wants "plate VII", not "Plate VII". */
		lower?: boolean;
	}

	let { id, lower = false }: Props = $props();

	const chapter = getChapter();
	const label = $derived(plateLabel(chapter(), id));
</script>

<a class="plate-ref" href="#{plateAnchor(id)}">{lower ? label.replace(/^Plate/, 'plate') : label}</a
>
