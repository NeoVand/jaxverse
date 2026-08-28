<script lang="ts">
	import { onDestroy } from 'svelte';
	import { resolve } from '$app/paths';
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import AttentionDiagram from '$lib/components/demos/language/AttentionDiagram.svelte';
	import NextTokenGame from '$lib/components/demos/language/NextTokenGame.svelte';
	import SkipGramDiagram from '$lib/components/demos/language/SkipGramDiagram.svelte';
	import TokenTree from '$lib/components/demos/language/TokenTree.svelte';
	import TransformerMap from '$lib/components/demos/language/TransformerMap.svelte';
	import WordVectors from '$lib/components/demos/language/WordVectors.svelte';
	import Tokenizer from '$lib/components/demos/language/Tokenizer.svelte';
	import Scribe from '$lib/components/demos/language/Scribe.svelte';
	import Inspector from '$lib/components/demos/language/Inspector.svelte';
	import Walkthrough from '$lib/components/demos/language/Walkthrough.svelte';
	import { scribe } from '$lib/components/demos/language/lab.svelte';

	// one engine serves the scribe, the meter and the walkthrough (the first two
	// plates are pure CPU and own nothing); the page is the engine's lifetime
	onDestroy(() => scribe.disposeAll());
</script>

<ChapterShell slug="language">
	<Prose>
		<p>
			Cover the next word of a sentence with your thumb and guess it. "The little dog wagged his
			____" is barely a puzzle; "my favorite word is ____" is a coin toss over a dictionary. Guess,
			uncover, compare, score yourself; slide one word to the right and play again. That game is
			this chapter's entire subject — and the claim it has to make good on is that nothing else is
			needed. A large language model is this game, won at scale.
		</p>
	</Prose>

	<NextTokenGame />

	<Prose>
		<p>
			Before a machine can play it, though, there is a problem to solve. Gradient descent works on
			numbers; it needs a slope to walk down. Words are not numbers, and the obvious fix — number
			the dictionary, so "cat" is 3,412 — is worse than useless, because that numbering claims cat
			and 3,411 are neighbors when nothing about the language agrees. What the model needs is for
			<em>similar words to sit near each other</em>, so that a gradient learned about one word does
			some good for its neighbors.
		</p>
		<p>
			So each word gets a <em>vector</em> instead: a short list of numbers, a position in a space of
			many dimensions. Nobody assigns those numbers. They are learned, and they are learned by the
			very game this chapter is about. Push a word's vector toward the words that tend to sit beside
			it, pull it away from words drawn at random, and after enough sentences the space arranges
			itself — animals collecting in one region, names in another, "he" beside "she". The idea that
			a word should be a learned vector is older than it is famous<Cite id="bengio-2003" />; this
			particular recipe is <em>skip-gram</em>, the heart of word2vec<Cite id="mikolov-2013a" />, and
			it is small enough to run in the page you are reading.
		</p>
	</Prose>

	<SkipGramDiagram />

	<Prose>
		<p>
			Note what the word trades away in that picture. Its honest representation is the
			<em>one-hot</em> column on the left — six hundred slots, a single 1, every word exactly as far
			from every other. The <em>dense</em> row it picks up instead is short, crowded with real numbers,
			and initially random garbage; it becomes meaningful only because the rule on the right hammers on
			it once per lesson. Written out, one lesson costs
		</p>
		<Math
			display
			tex={'\\mathcal{L} \\;=\\; -\\log \\htmlClass{eq-op}{\\sigma}\\big(\\htmlClass{eq-model}{v}^{\\top}\\htmlClass{eq-world}{u_{+}}\\big) \\;-\\; \\sum_{k=1}^{5} \\log\\Big(1 - \\htmlClass{eq-op}{\\sigma}\\big(\\htmlClass{eq-model}{v}^{\\top}\\htmlClass{eq-mute}{u^{-}_{k}}\\big)\\Big)'}
		/>
		<p>
			Take it in two halves. <Math tex={'\\htmlClass{eq-model}{v}'} /> is the centre word's row of the
			table — the dense vector everything here is about — and
			<Math tex={'\\htmlClass{eq-world}{u_{+}}'} /> belongs to the neighbor actually seen beside it. The
			sigmoid
			<Math tex={'\\htmlClass{eq-op}{\\sigma}'} /> turns their dot product into a verdict between 0 and
			1 — <em>did these two really co-occur?</em> — and
			<Math tex={'-\\log \\htmlClass{eq-op}{\\sigma}'} /> is the surprise at the truth, large exactly
			when the verdict on a real pair was low. The sum plays prosecutor: five words
			<Math tex={'\\htmlClass{eq-mute}{u^{-}_{k}}'} /> drawn at random, each punished for scoring. Nothing
			else appears in <Math tex={'\\mathcal{L}'} />, so the only way down is to move the vectors —
			pull <Math tex={'\\htmlClass{eq-model}{v}'} /> toward
			<Math tex={'\\htmlClass{eq-world}{u_{+}}'} />, push it off the five impostors.<Cite
				id="mikolov-2013b"
			/> Geometry is not a by-product of this objective; it is the objective. An untrained sigmoid shrugs
			<Math tex="\ln 2" /> at each of the six verdicts, so the plate's loss meter starts near 4.2 and
			falls as the space takes shape.
		</p>
		<p>
			What that loss is doing underneath is less mysterious than it looks. Run the algebra out and
			skip-gram with negative sampling turns out to be quietly factorising a plain table of counts —
			how often each word appears beside each other word, compared with how often chance alone would
			put them there.<Cite id="levy-goldberg-2014" /> The network is a way of doing that arithmetic one
			pair at a time, on a corpus far too large to hold the table for. The meaning was in the co-occurrence
			statistics all along; what the vectors add is that you can carry them around.
		</p>
	</Prose>

	<WordVectors />

	<Prose>
		<p>
			What you are looking at is a shadow of a sixteen-dimensional space — its three principal
			directions, the rest projected away, spinnable because even a shadow keeps more of its shape
			in three dimensions than in two — so read distances loosely. What survives the projection is
			the grouping, and the neighbor list beside it is computed in the full space, by
			<em>cosine similarity</em>: the angle between two vectors, ignoring their length. Nothing
			labeled any of this. The only pressure applied was "predict your neighbors", and geometry was
			the answer.
		</p>
		<p>
			The vector arithmetic is the famous part. If the step from "boy" to "girl" is roughly the same
			displacement as from "he" to "she", then subtracting one and adding the other should land near
			the fourth word — and on this corpus it does. With 600 words and sixteen dimensions the hits
			are approximate and easy to break; at web scale, on billions of words, this stops being a
			party trick and becomes the substrate every language model computes on. You will not find the
			famous king − man + woman ≈ queen here, and the reason is instructive: these are children's
			stories, and "king" appears nine times in three hundred thousand words. A vector is learned
			from the company its word keeps — a word barely mentioned barely exists.
		</p>
		<p>
			That famous example has always been sold harder than it deserves. The usual way of scoring it
			forbids the answer from being any of the three words you put in, which quietly removes the
			most likely wrong answers before the arithmetic is judged — and
			<em>king</em> is a very likely wrong answer to <em>king − man + woman</em>. Take the rule away
			and a good deal of the accuracy goes with it.<Cite id="linzen-2016" /> The vectors do carry direction
			and relation. They carry rather less of it than the party trick implies.
		</p>
		<p>
			One question remains before the real model: vectors for <em>what</em>, exactly? Words are a
			convenient story, but a vocabulary of English words is both enormous and never enough —
			someone will always write "unbelievability". Real models predict <em>tokens</em>: word-pieces,
			fragments like "wag" and "ged", and nobody designs those either. They are grown out of the
			data by <em>byte-pair encoding</em><Cite id="sennrich-2016" />, an algorithm of almost
			embarrassing plainness — it was a file-compression trick before anyone pointed it at language:
			start from the raw alphabet, count every adjacent pair in the corpus, fuse the most frequent
			pair into a new token, repeat. Common words end up as single tokens, rare words shatter into a
			few, and every entry in the vocabulary is a vote cast by frequency.
		</p>
	</Prose>

	<TokenTree />

	<Prose>
		<p>
			The plate below runs that exact loop — not a replay of a stored merge list, the algorithm
			itself — on the same 1.5 million characters the model further down is trained on. It needs no
			GPU and trains no network; it only counts. One rule keeps the result readable, and every real
			tokenizer has it: a merge may never cross from one word into the next. A word carries its
			leading space, drawn ␣ here, which is why <em>“the”</em> and <em>“ the”</em> are two different tokens
			— and why nothing in the vocabulary is ever half of one word glued to half of another. Watch which
			fusions this corpus elects first.
		</p>
	</Prose>

	<Tokenizer />

	<Prose>
		<h2 class="h2">The game itself</h2>
		<p>
			Now the machine. A model that plays left to right, predicting each token from everything
			before it, is called <em>autoregressive</em>. The design gives up nothing, because
			probability's chain rule factors any sequence exactly:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-out}{P}(x_1, x_2, \\ldots, x_T) \\;=\\; \\prod_{t=1}^{T} \\htmlClass{eq-out}{P}(x_t \\mid x_{<t})'}
		/>
		<p>
			The left side is the thing we actually want — a probability for whole sentences, paragraphs,
			books. The right side is one small question, <em>what comes next?</em>, asked once per
			position. Answer the small question well and the identity hands you everything else.
		</p>
		<p>
			And the answers cost nothing to grade. In <ChapterRef slug="digits" /> every training example needed
			a person to write down the answer — ten thousand digits, ten thousand labels. Here the label is
			the text itself: the covered token was there all along, put down for free by whoever wrote the sentence.
			Every position in every sentence is a graded exercise that nobody had to grade. This is
			<em>self-supervised learning</em>, and the zero price of its labels is precisely why it scales
			to the whole internet while hand-labeled datasets stall at millions.
		</p>
		<p>
			Training scores each answer by the probability the model gave to the token that actually came;
			the <em>cross-entropy</em> is its average surprise,
		</p>
		<Math
			display
			tex={'\\mathcal{L} \\;=\\; -\\,\\frac{1}{T} \\sum_{t=1}^{T} \\log \\htmlClass{eq-out}{P}(\\htmlClass{eq-world}{x_t} \\mid \\htmlClass{eq-world}{x_{<t}})'}
		/>
		<p>
			measured in <em>nats</em>, the natural-log unit of surprise. Learn to read that number like a
			gauge: at loss 1.2 the model is, on average, as uncertain as someone choosing among
			<Math tex={'e^{1.2} \\approx 3.3'} /> plausible next tokens. A model that knows nothing sits at
			<Math tex="\ln V" /> for a vocabulary of <Math tex="V" />, every door held equally open; the
			ultramarine curve starts exactly there, and every hundredth of a nat it sheds is a regularity
			of English found and kept.
		</p>
		<p>
			And the vocabulary is the one you just grew. The scribe below reads the 369 word-pieces that
			three hundred merges elect on this corpus — the same three hundred the plate above runs by
			default — so its attention rows, two plates from now, are words reading words rather than
			letters reading letters. If you kept merging up there, hand your longer vocabulary over with
			<em>send to the scribe</em>; a different vocabulary means a different embedding table, so the
			model restarts, which is the honest cost of the decision. You can also switch it back to
			single characters, and should at least once: it is the slow, legible version, where you can
			watch spelling itself get invented.
		</p>
	</Prose>

	<Scribe />

	<UnderTheHood slug="language" block="transformer" />

	<Prose>
		<h2 class="h2">From noise to grammar</h2>
		<p>
			If you let the scribe run, you watched an order of acquisition that nobody programmed. Word
			shapes first, then real words in plausible company, then clauses whose subject and verb mostly
			agree, and somewhere past a thousand steps a sentence you could believe a child wrote. Grammar
			arrived in that order for a plain reason: gradient descent spends its budget where the loss
			falls fastest, and short, local regularities pay off first. The curriculum fell out of
			prediction pressure alone.
		</p>
		<p>
			Switch the vocabulary to characters and you can watch the layer underneath. A character model
			starts by learning English's letter frequencies — too many e's and spaces to be random — then
			pairs: q finds u, h learns to trail t and s. Only then do word shapes appear, and the whole
			schedule runs slower. The word-piece scribe skipped all of it, because that layer of
			regularity was already cashed into the vocabulary by three hundred merges: <em>the</em>,
			<em>said</em> and <em>little</em> arrive whole, leading space and all, as one token each, and spelling
			them is not a problem the model ever has. Same corpus, coarser atoms, longer reach — the same ninety-six-slot
			window now holds about two hundred and thirty characters instead of ninety-six.
		</p>
		<p>
			There is a colder way to say what happened: the model compressed the corpus. Cross-entropy is
			literally a size — a nat is <Math tex="1/\ln 2 \approx 1.44" /> bits — and the plate reports
			<em>bits per character</em> beside its loss so that the two vocabularies can be compared at
			all. Guessing uniformly among 69 characters costs 6.11 bits per character. The tokenizer alone
			drops that to 3.53, before a single gradient step, purely by making the guesses coarser.
			Training then takes it under 1.6, which is better than <code>gzip -9</code> manages on the
			same text (2.42) and close to <code>brotli</code> (1.87). The honest asterisk: those two ship a
			self-contained file, while the scribe's bits assume you already have its weights — and a complete
			description would have to count those too. Prediction and compression are one skill in two vocabularies,
			and the loss chart doubles as a receipt for how much structure the model has taken in.
		</p>
		<p>
			None of which is new. Shannon measured English this way in 1951 by sitting people down with
			covered text and asking them to guess the next letter, then turning their guesses into a
			number — somewhere between 0.6 and 1.3 bits per character, which is roughly where a good model
			sits today.<Cite id="shannon-1951" /> The identity runs the other way too: hand a large language
			model to a compression routine and it beats the specialist formats outright, on text and — stranger
			— on images and audio it was never trained on.<Cite id="deletang-2023" /> Something that predicts
			well enough is a compressor, whatever it was built to be.
		</p>
		<p>
			You may also have noticed the scribe's register — sunny, simple, faintly like a bedtime story,
			whatever you prompt it with. Its whole world is 1.5 million characters of children's stories,
			so that is the only English in existence for it. A model is its diet. The large models
			everyone talks to differ from this one less in kind than in menu: they have read a substantial
			fraction of everything, and every register — helpful, legalistic, purple — is in there,
			waiting on the prompt.
		</p>
		<p>
			The loss you watched was an average over thousands of positions, and averages hide texture.
			Some tokens are nearly free; some cost dearly. The meter below un-averages the number: it
			bills a sentence one token at a time and, for each one, shows the <em>entropy</em> of the model's
			guess — how widely it was hedging — along with the five candidates it liked best. Watch where the
			heat lands. A common word arriving where it is expected costs almost nothing; the expensive tokens
			are the ones carrying the actual news of the sentence, which is a fair working definition of information.
		</p>
	</Prose>

	<Inspector />

	<UnderTheHood slug="language" block="attention" />

	<Prose>
		<h2 class="h2">Inside one guess</h2>
		<p>
			So far the model has been a box that eats context and emits a distribution. Open it. The
			scribe is a <em>transformer</em>, the architecture behind essentially every model in this
			book's orbit, and a single guess travels five stages — each one visible, with real numbers, in
			the last plate of this chapter.
		</p>
	</Prose>

	<TransformerMap />

	<Prose>
		<p>
			<strong>One:</strong> every token becomes a vector, exactly as words did in the first plate,
			plus a second vector encoding <em>where</em> it sits — attention sees a bag of tokens and has
			no inherent sense of order, so position must be supplied. This model supplies it the plain
			way, one learned vector per slot; current models mostly rotate each query and key by an angle
			set by its position instead, which lets attention see how far apart two tokens are rather than
			which slots they sit in.<Cite id="su-2021" />
			<strong>Two:</strong>
			each vector is projected three ways, into a <em>query</em> (what am I looking for?), a
			<em>key</em>
			(what do I offer?) and a
			<em>value</em> (what I would pass along).
		</p>
		<p>
			<strong>Three:</strong> attention itself — and it has earned the chapter's slowest minute,
			because this single line is most of what the word <em>transformer</em> means<Cite
				id="vaswani-2017"
			/>:
		</p>
		<Math
			display
			tex={'\\mathrm{attention}(\\htmlClass{eq-model}{Q}, \\htmlClass{eq-model-2}{K}, \\htmlClass{eq-model-3}{V}) \\;=\\; \\htmlClass{eq-op}{\\operatorname{softmax}}\\!\\left(\\frac{\\htmlClass{eq-model}{Q}\\,\\htmlClass{eq-model-2}{K}^{\\top}}{\\sqrt{d_k}} + \\htmlClass{eq-mute}{M}\\right)\\htmlClass{eq-model-3}{V}'}
		/>
	</Prose>

	<AttentionDiagram />

	<Prose>
		<p>
			Read it inside out. <Math
				tex={'\\htmlClass{eq-model}{Q}\\,\\htmlClass{eq-model-2}{K}^{\\top}'}
			/> dots every query against every key —
			<em>how well does what I'm looking for match what you offer?</em>
			— one number per pair of positions, a whole table of raw affinities at once. The
			<Math tex={'\\sqrt{d_k}'} /> underneath is quiet but load-bearing: dot products of longer vectors
			are larger by accident of dimension, and dividing by
			<Math tex={'\\sqrt{24}'} /> here keeps the softmax from saturating into all-or-nothing before training
			has said anything. <Math tex={'\\htmlClass{eq-mute}{M}'} /> is the mask — zero at and below the
			diagonal,
			<Math tex="-\infty" /> above it — the game's one rule, <em>no reading the future</em>,
			enforced as arithmetic: <Math tex={'e^{-\\infty} = 0'} />, so a future token gets exactly
			nothing, not merely little. Then
			<Math tex={'\\htmlClass{eq-op}{\\operatorname{softmax}}'} /> — the same machine that turned scores
			into beliefs in <ChapterRef slug="digits" /> — runs along each row and turns it into a budget of
			exactly 1.0. And multiplying by
			<Math tex={'\\htmlClass{eq-model-3}{V}'} /> spends the budget: each token's output is a weighted
			blend of what earlier tokens offered to pass along.
		</p>
		<p>
			Two things about this line repay staring. It is the only place in the whole architecture where
			tokens touch — everywhere else each position is processed alone. And it contains no
			parameters: everything learnable lives in the three lenses
			<Math
				tex={'W_{\\htmlClass{eq-model}{Q}}, W_{\\htmlClass{eq-model-2}{K}}, W_{\\htmlClass{eq-model-3}{V}}'}
			/>
			that manufacture the queries, keys and values, so what training changes is not <em>how</em>
			attention works but <em>what each token asks for and offers</em>. The scribe runs this line
			four times per block — four <em>heads</em>, each in its own 24 dimensions, each free to learn
			a different sense of relevance — and you can watch their real rows disagree in the last plate.
		</p>
		<p>
			<strong>Four:</strong> each token thinks alone. Its vector is widened fourfold, rectified —
			negatives clipped to zero — and squeezed back to size; most of the parameters live here.
			Stages two through four form one <em>block</em>, and blocks stack: this model has two,
			frontier models roughly a hundred. <strong>Five:</strong> the final vector meets one last matrix
			that turns 96 numbers into one score per token in the vocabulary, and a softmax turns scores into
			probabilities. Draw one, append it, run the pass again. That loop is all "generating text" has ever
			meant.
		</p>
		<p>
			That list of five leaves out what happens between the blocks, and it is the whole shape of the
			thing. A block does not hand its output onward. It <em>adds</em> to what it was given:
			attention computes something and adds it in, the little network computes something and adds it
			in, and the vector for each token runs the entire length of the model, picking up
			contributions along the way. What travels through a transformer is not a signal being
			transformed stage by stage but a shared channel — a <em>residual stream</em> — that every
			block reads from and writes into.<Cite id="elhage-2021" /> It is why depth degrades gracefully rather
			than catastrophically: a block with nothing useful to say can write approximately nothing and cost
			the model almost nothing. And it is the reason interpretability work talks about what a particular
			head <em>writes</em> to the stream, as though the model were a workshop of specialists sharing one
			notebook. Which, read this way, it is.
		</p>
		<p>
			One thing rides along with that picture. Because every block adds, the stream grows: by the
			last block a token's vector is the sum of everything written into it, at whatever size that
			came to. So a block does not read the stream raw — it reads a rescaled copy, normalized to a
			predictable size, while the stream itself passes through untouched. Two lines of arithmetic,
			and where they sit is not a detail: put the rescaling <em>between</em> the blocks instead of
			inside them and the gradients near the output blow up at the start of training, which is why
			the early transformers needed a careful warm-up to survive their own first steps.<Cite
				id="xiong-2020"
			/> Moving it inside is why a hundred blocks stack without one.
		</p>
	</Prose>

	<Walkthrough />

	<Prose>
		<h2 class="h2">The same game, a billionfold</h2>
		<p>
			Everything in this chapter scales without changing shape. Take the scribe's loss, its update
			rule <Math
				tex={'\\htmlClass{eq-model}{\\theta} \\leftarrow \\htmlClass{eq-model}{\\theta} - \\htmlClass{eq-knob}{\\eta} \\htmlClass{eq-world}{\\nabla_\\theta \\mathcal{L}}'}
			/>, the five stages you just walked; multiply the parameters by a few million, the corpus by a
			billion, and the training run by months on thousands of GPUs — and you have the large language
			models everyone talks to. They descend the same cross-entropy on the same next-token game, and
			their loss charts look like yours with more zeros on the axis. What you trained here is not a
			metaphor for them. It is one of them, small.
		</p>
		<p>
			That the multiplying works is not a hope; it is a measured curve. Loss falls as a power law in
			parameters, in data and in compute, and it stays a power law across seven orders of magnitude
			— straight lines on a log plot, straight enough that a run's final loss can be predicted
			before it starts.<Cite id="kaplan-2020" /> Straight lines are what made the spending defensible.
			They are also easy to read wrong: for two years the field built models far larger than the text
			it was feeding them, until the measurement was redone more carefully and the recommended trade between
			size and data moved sharply toward data.<Cite id="hoffmann-2022" /> A law with the wrong constants
			still looks like a law.
		</p>
		<p>
			Scale buys a continuation of exactly what you watched: fluency first, then knowledge — because
			past a certain point the cheapest way to keep shaving nats off "the capital of France is ____"
			is to know the capital of France. It does not buy truth, since the objective rewards the
			plausible continuation and confident nonsense is often the most plausible continuation of
			confident nonsense. And nowhere in <Math tex={'\\mathcal{L}'} /> is there a goal: nothing in the
			game prefers being helpful, or honest, or anything at all beyond sounding like the diet.
		</p>
		<p>
			That is not a flaw in the training so much as a boundary of it. Prediction has no preferences.
			To give a model preferences — to make it want to be useful rather than merely likely — you
			need a signal that judges outcomes instead of continuations: a reward. Consequences are a
			different teacher, and they get
			<a href={resolve('/reward')}>the next chapter</a>.
		</p>
	</Prose>
</ChapterShell>
