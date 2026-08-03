<script lang="ts">
	import { onDestroy } from 'svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Pretrain from '$lib/components/demos/rook/Pretrain.svelte';
	import PlayRook from '$lib/components/demos/rook/PlayRook.svelte';
	import Sft from '$lib/components/demos/rook/Sft.svelte';
	import Rlvr from '$lib/components/demos/rook/Rlvr.svelte';
	import Arena from '$lib/components/demos/rook/Arena.svelte';
	import { lab } from '$lib/components/demos/rook/rook-context.svelte';
	import { resolve } from '$app/paths';

	// One engine serves all four plates; the page owns its lifetime.
	onDestroy(() => lab.dispose());
</script>

<ChapterShell slug="rook">
	<Prose>
		<p>
			Every large language model you have talked to was made in three acts.
			<em>Pretraining</em>: predict the next token on an enormous, indiscriminate corpus, until the
			model speaks the language of its data. <em>Supervised fine-tuning</em>: keep training, but on
			a small corpus somebody curated, until it speaks the way you want. And a last act of
			<em>reinforcement learning</em>: let the model generate, let a judge score what comes back,
			and move the weights toward whatever scored well. The whole modern pipeline fits in one
			sentence — pretrain on everything, fine-tune on the good stuff, reinforce on verified
			outcomes.
		</p>
		<p>
			This chapter runs all three acts on one small transformer, live, in this page. Not a diagram
			of the pipeline — the pipeline, with every gradient computed on your GPU. The model is called
			Rook, and it learns chess.
		</p>
		<p>
			Chess, first, because <strong>moves are tokens</strong>. A move written in <em>UCI</em>
			notation — <code>e2e4</code>, <code>g8f6</code> — is a short string, and only 1,930 distinct
			ones ever occur in Rook's world. Add one marker token, ⟨game⟩, for “a new game begins”, and
			you have a vocabulary of 1,931. A game is a sentence written in it. The game of
			<a href={resolve('/language')}>Chapter 5</a> — predict the next token, score it with cross-entropy,
			descend — carries over without changing a comma.
		</p>
		<p>
			Chess, second, because <strong>every claim is checkable</strong>. A move is legal in a
			position or it is not, and a small library of rules (this page carries chess.js, a complete
			referee) settles the question instantly, every time. That is what the V means in
			<em>RLVR — reinforcement learning from verifiable rewards</em>: the judge is not anyone's
			taste but a verifier that cannot be flattered, bribed, or fooled by confident nonsense. When
			frontier models are trained to reason about mathematics and code, this is the same move:
			reward what can be checked.
		</p>
		<p>
			One more thing, and it shapes everything that follows: Rook never sees a board. No 8×8 grid,
			no piece list, no rules — only text, one move-token after another, 128 plies of context at
			most. Everything it appears to know about squares and pieces has to be squeezed out of the
			statistics of move strings. Keep that in mind as you watch it play: the little diagrams on
			this page are for you; the model gets the sentence.
		</p>
		<p>
			Meet the student, then: four transformer blocks, 128 numbers wide, four attention heads — 1.3
			million parameters. The same weights thread through this entire chapter; what pretraining
			builds is exactly what fine-tuning bends and reinforcement sharpens. The plates need WebGPU (a
			current Chrome or Edge) and wake on their own as you reach them — the whole chapter shares one
			model, running on your GPU.
		</p>
	</Prose>

	<Wide>
		<Pretrain />
	</Wide>

	<UnderTheHood slug="rook" block="engine" />

	<Prose>
		<p>
			What did pretraining buy? At step 0 the time machine shows you noise: roughly one sampled move
			in a hundred is legal, and that one by accident. By step 2,600, about 87 in a hundred are
			legal — played in positions the model has never seen, under rules nobody stated. The only
			pressure ever applied was next-token prediction over some six thousand games. Legality
			condensed out of statistics, the way grammar condenses out of text: not because the loss asked
			for it, but because predicting the corpus well is impossible without it.
		</p>
		<p>
			Now look at what pretraining did <em>not</em> buy. Read the sampled games: pieces drift,
			nothing is hunted, material hangs unpunished. Of course it does — the corpus is
			<em>random</em> legal play, and imitation is faithful to a fault. Rook plays like its diet, because
			its diet is the only definition of “good move” it has ever been shown. The fix is not more data.
			It is better data. Before we administer it, sit across the board from your student and feel what
			87% legal actually plays like.
		</p>
	</Prose>

	<Wide>
		<PlayRook />
	</Wide>

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			Better data, same machine
		</h2>
		<p>
			To see why the second act exists, look at what the first one actually produces. A pretrained
			model is a <em>completion machine</em>: ask a raw base model a question and it may answer — or
			continue with three more questions, or a survey form, because that is what question-shaped
			text does in its corpus. The assistant you have talked to was made from such a base model by
			<em>supervised fine-tuning</em>: continued training on a curated set of (instruction → good
			response) pairs, written and vetted by people. It adds little knowledge. What it teaches is
			the <em>format</em> of being helpful — that in this house, a question is followed by an answer.
			One pair from such a corpus looks like this:
		</p>
		<div class="my-7 grid gap-3 sm:grid-cols-2">
			<div class="rounded-lg border border-line-soft bg-surface px-4 py-3.5">
				<span class="eyebrow">instruction</span>
				<div class="mt-1.5 font-serif text-[15.5px] leading-normal text-ink">
					Explain gravity to a child.
				</div>
			</div>
			<div class="rounded-lg border border-line-soft bg-surface px-4 py-3.5">
				<span class="eyebrow">curated response</span>
				<div class="mt-1.5 font-serif text-[15.5px] leading-normal text-ink">
					The Earth is always gently pulling everything toward itself — you, your ball, the rain.
					That pull is called gravity. It is why things fall down instead of up, and why you come
					back when you jump.
				</div>
			</div>
		</div>
		<p>
			Mechanically, nothing changes: the same weights, the same cross-entropy, the same update
			<Math tex={'\\theta \\leftarrow \\theta - \\gamma \\nabla \\mathcal{L}'} /> — only the corpus is
			now chosen on purpose. For Rook the mapping is direct. The “instruction” is the game so far; the
			“good response” is what a competent player did next. Where a frontier lab curates tens of thousands
			of demonstrations, we curate 2,381 games played by a greedy little bot that grabs material whenever
			it can. Its style is loud: 38% of its moves are captures, against about 8% in random play, and one
			game in twenty ends in checkmate.
		</p>
		<p>
			One paragraph of honesty before the button. Fine-tuning shifts <em>style</em>, and it charges
			a price: as the weights lean toward greedy play, the old random-play corpus becomes more
			surprising to the model, and its validation loss there quietly rises. That drift is not a
			malfunction — it is what specialization looks like from the old distribution's point of view.
			It is also why fine-tuning is run at a smaller step size than pretraining (we ease
			<Math tex="\gamma" /> from 1.2·10⁻³ down to 3·10⁻⁴): nudge the weights and the style shifts; blast
			them and the old competence goes too. The plate below measures both sides of the trade, because
			a curve that only shows the win is an advertisement, not an experiment.
		</p>
	</Prose>

	<Wide>
		<Sft />
	</Wide>

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			A judge instead of examples
		</h2>
		<p>
			Before moving on, scroll back and play your fine-tuned student — Plate II always drives the
			current weights, and you will feel the difference: it comes for your pieces now.
		</p>
		<p>
			The last act needs no examples at all. It needs a judge. Here is one full iteration of RLVR,
			as this page runs it: take a real opening from the corpus — four to ten plies, different every
			time — and have Rook finish it <Math tex="G = 8" /> different ways, up to fourteen plies each, sampled
			hot enough to disagree with itself. Then chess.js replays every continuation from that exact position
			and issues a verdict no one can argue with. Rollout
			<Math tex="i" /> earns
		</p>
		<Math
			display
			tex={'r_i \\;=\\; \\frac{\\ell_i}{n_i} \\;+\\; \\tfrac{1}{2}\\,\\big[\\,\\ell_i = n_i\\,\\big]'}
		/>
		<p>
			where <Math tex="\ell_i" /> counts the consecutive legal plies before the first illegal one, <Math
				tex="n_i"
			/> counts the plies attempted, and the bracket pays a half-point bonus only when the whole rollout
			survives the judge. Play twelve clean plies out of fourteen and you score well; break on ply two
			and you score badly; stay perfect and you are paid extra.
		</p>
		<p>
			Raw rewards are not used directly. Within the group of eight, each reward is compared to the
			others:
		</p>
		<Math
			display
			tex={'\\hat{A}_i \\;=\\; \\frac{r_i - \\operatorname{mean}(r_1,\\ldots,r_G)}{\\operatorname{std}(r_1,\\ldots,r_G)}'}
		/>
		<p>
			This <Math tex={'\\hat{A}_i'} /> is the <em>advantage</em>: how much better or worse rollout
			<Math tex="i" /> did than its own siblings. <a href={resolve('/reward')}>Chapter 6</a>
			stabilized learning by measuring outcomes against a baseline instead of trusting raw reward; this
			is the same idea with the scaffolding removed — <strong>the group is the baseline</strong>.
			That trick (the heart of the GRPO family of methods) needs no second network to estimate
			values: eight siblings, sampled from the same position, are estimate enough.
		</p>
		<p>The update is REINFORCE, exactly as the policy-gradient chapter promised it would be:</p>
		<Math
			display
			tex={'\\nabla_\\theta J \\;\\approx\\; \\sum_{i=1}^{G} \\hat{A}_i \\;\\nabla_\\theta \\log \\pi_\\theta(\\text{rollout}_i)'}
		/>
		<p>
			— raise the log-probability of every token in the above-average rollouts, lower it for the
			below-average ones, and touch only the <em>generated</em> tokens: the prefix was given, so it
			gets no credit. And one honest edge case, which you will see happen: if all eight rollouts
			earn the same reward, the standard deviation is zero, the advantages are undefined, and the
			step is skipped. When every answer is equally good, there is no gradient — reinforcement
			learning learns from <em>differences</em>, and a group of equals teaches nothing.
		</p>
	</Prose>

	<Wide>
		<Rlvr />
	</Wide>

	<UnderTheHood slug="rook" block="rlvr" />

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			Line them up
		</h2>
		<p>
			You have now made three different chess players out of one set of weights, and the honest way
			to compare them is not a curve — it is a decision. The page has been quietly photographing
			your work: every time the fine-tuning or RLVR loop pauses, it snapshots the weights, so all
			three stages survive even though the resident model keeps moving on. The arena below puts them
			at the same board. Play a move, and every fielded Rook is asked the same question —
			<em>what would you do here?</em> — each answer drawn as an arrow in its stage's color, with the
			numbers that explain it: how much of its belief was even legal, and what it wanted most before the
			mask. Watch the pretrained arrow wander while the fine-tuned one aims at your pieces; watch the
			RLVR column hold its legal mass in positions that make the earlier stages guess.
		</p>
	</Prose>

	<Wide>
		<Arena />
	</Wide>

	<UnderTheHood slug="rook" block="arena" />

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			Where the judge works today
		</h2>
		<p>
			The loop you just ran is not a toy version of something else — it is the something else, at
			reduced scale. When today's models are trained to write code, the verifier is a bank of unit
			tests: the program compiles and passes, or it does not, and no amount of confident prose
			changes the verdict. That is a move being legal or not, wearing different clothes. In
			mathematics, the judge checks the final answer, or a proof checker walks the argument line by
			line. Same iteration, exactly: sample a group of attempts from one problem, let the verifier
			score each, standardize within the group, reinforce what held up. The graders got bigger; the
			gradient did not change.
		</p>
		<p>
			This is also why the verifiable domains are the ones moving fastest. A judge made of rules is
			cheap, instant, tireless, and incorruptible — you can ask it a million times a day and it
			never lowers its standards — while a judge made of human preference is expensive, slow, noisy,
			and can be charmed. So coding and mathematics improve at a pace that essay-writing and advice
			do not: the pipeline is the same, but only some subjects come with an answer key. One honest
			caveat belongs here: a model rewarded for passing the tests will sometimes learn to game the
			tests rather than solve the problem — <em>reward hacking</em> — and building judges that cannot
			be fooled is a live research problem, not a solved one.
		</p>
		<p>
			And notice how the three stages fit: reinforcement could not have started from noise — a group
			of rollouts that are all hopeless earns identical rewards, and as Plate IV showed, a group of
			equals has no gradient. Pretraining built enough competence for the judge to have something to
			grade; fine-tuning aimed it; only then could verified reward pull it the rest of the way. Each
			act feeds the next. That ordering is not a convention — it is the reason the pipeline has this
			shape.
		</p>
		<p>
			Step back and look at what you ran. One set of weights was pretrained on everything available,
			fine-tuned on a curated slice, then reinforced against a verifier. Swap chess for language.
			Swap legality for human preference and verified answers. Swap 1.3 million parameters for a
			trillion, and a browser tab for a datacenter. The recipe you just ran is the recipe. The ideas
			do not change — only the appetite.
		</p>
		<p>
			And that is the book. A loss surface and a step downhill; a neuron, a bump of influence; space
			bent until classes come apart; a map that draws itself; a game of guess-the-next-word; a
			policy learning from consequences — and here, all of it at once, teaching a pocket of numbers
			to play chess it was never taught. None of it stayed mysterious once you could watch the
			gradients move. That was the point.
		</p>
	</Prose>
</ChapterShell>
