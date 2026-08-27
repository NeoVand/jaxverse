// The bibliography: every paper the book leans on, and which chapter leans on it.
//
// A citation is a promise that the reader can go and check, so every entry
// here carries a url that lands on something actually readable — the arXiv
// abstract page where the work is on arXiv, an open PDF or a publisher's free
// full text where it is not. Nothing in this file points at a paywall.
//
// Prose cites by id through <Cite>, and the numeral comes from the paper's
// position in its chapter's list below rather than from a number typed into a
// sentence — the same contract plates.ts keeps for figures. Add a citation in
// the middle of a chapter and every later numeral moves with it.

export interface Paper {
	/** Authors as the reference line should read them — "Kingma & Ba". */
	authors: string;
	year: number;
	title: string;
	/** Where it appeared, short: "ICLR 2015", "Nature 323". */
	where: string;
	/** Somewhere the reader can read it without paying. */
	url: string;
	/** One line on why this paper is worth the click. Optional, but earn it. */
	note?: string;
}

export const papers = {
	'robbins-monro-1951': {
		authors: 'Robbins & Monro',
		year: 1951,
		title: 'A Stochastic Approximation Method',
		where: 'Annals of Mathematical Statistics 22(3)',
		url: 'https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full',
		note: 'The proof that you may take your steps from noisy estimates of the slope and still arrive, if the steps shrink at the right rate. Every minibatch since is a corollary.'
	},
	'polyak-1964': {
		authors: 'Polyak',
		year: 1964,
		title: 'Some methods of speeding up the convergence of iteration methods',
		where: 'USSR Computational Mathematics and Mathematical Physics 4(5)',
		url: 'https://www.mathnet.ru/eng/zvmmf7713',
		note: 'Momentum, as a heavy ball rolling on the surface rather than a walker stepping on it.'
	},
	'duchi-2011': {
		authors: 'Duchi, Hazan & Singer',
		year: 2011,
		title: 'Adaptive Subgradient Methods for Online Learning and Stochastic Optimization',
		where: 'JMLR 12',
		url: 'https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf',
		note: 'AdaGrad — the first widely used optimizer to give every parameter its own step size, scaled by the gradients it has seen so far.'
	},
	'tieleman-hinton-2012': {
		authors: 'Tieleman & Hinton',
		year: 2012,
		title: 'Lecture 6.5 — RMSProp',
		where: 'COURSERA: Neural Networks for Machine Learning',
		url: 'https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf',
		note: 'RMSProp was never written up as a paper. It was slide 29 of a lecture, and half the field adopted it anyway.'
	},
	'kingma-ba-2015': {
		authors: 'Kingma & Ba',
		year: 2015,
		title: 'Adam: A Method for Stochastic Optimization',
		where: 'ICLR 2015',
		url: 'https://arxiv.org/abs/1412.6980',
		note: 'Momentum and RMSProp in one update, with a bias correction for the first few steps.'
	},
	'loshchilov-hutter-2019': {
		authors: 'Loshchilov & Hutter',
		year: 2019,
		title: 'Decoupled Weight Decay Regularization',
		where: 'ICLR 2019',
		url: 'https://arxiv.org/abs/1711.05101',
		note: 'AdamW. The finding is subtler than its fame: weight decay folded into the gradient is not the same thing as weight decay applied to the weights, and for Adam the difference is large.'
	},
	'chen-2023': {
		authors: 'Chen et al.',
		year: 2023,
		title: 'Symbolic Discovery of Optimization Algorithms',
		where: 'NeurIPS 2023',
		url: 'https://arxiv.org/abs/2302.06675',
		note: 'Lion, found by a program search over update rules rather than derived by hand.'
	},
	'dauphin-2014': {
		authors: 'Dauphin et al.',
		year: 2014,
		title:
			'Identifying and attacking the saddle point problem in high-dimensional non-convex optimization',
		where: 'NeurIPS 2014',
		url: 'https://arxiv.org/abs/1406.2572',
		note: 'The argument that the folk fear of local minima is misplaced: in many dimensions, the flat places a walker gets stuck at are overwhelmingly saddles, not basins.'
	},
	'goodfellow-2015': {
		authors: 'Goodfellow, Vinyals & Saxe',
		year: 2015,
		title: 'Qualitatively characterizing neural network optimization problems',
		where: 'ICLR 2015',
		url: 'https://arxiv.org/abs/1412.6544',
		note: 'Walk the straight line from a network’s starting weights to its trained ones and the loss falls the whole way. The path training takes is not straight, but it never has to cross a wall.'
	},
	'li-2018': {
		authors: 'Li et al.',
		year: 2018,
		title: 'Visualizing the Loss Landscape of Neural Nets',
		where: 'NeurIPS 2018',
		url: 'https://arxiv.org/abs/1712.09913',
		note: 'What a real loss surface looks like once you slice it honestly — and how much of its apparent shape is an artifact of how you scaled the slice.'
	},
	'cohen-2021': {
		authors: 'Cohen et al.',
		year: 2021,
		title: 'Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability',
		where: 'ICLR 2021',
		url: 'https://arxiv.org/abs/2103.00065',
		note: 'Full-batch gradient descent does not settle into a curvature it can handle. It climbs until the curvature is exactly as sharp as its step size can survive, and then stays there, half-unstable, for the rest of training.'
	},
	'cybenko-1989': {
		authors: 'Cybenko',
		year: 1989,
		title: 'Approximation by superpositions of a sigmoidal function',
		where: 'Mathematics of Control, Signals and Systems 2',
		url: 'https://web.njit.edu/~usman/courses/cs675_fall18/10.1.1.441.7873.pdf',
		note: 'The universal approximation theorem, for one hidden layer of sigmoids. Read the proof and notice what it never mentions: how many neurons, or how to find their weights.'
	},
	'lu-2017': {
		authors: 'Lu et al.',
		year: 2017,
		title: 'The Expressive Power of Neural Networks: A View from the Width',
		where: 'NeurIPS 2017',
		url: 'https://arxiv.org/abs/1709.02540',
		note: 'The mirror image of Cybenko: hold the width just above the input dimension and stack deep, and you are universal again. Width and depth are two ways to buy the same guarantee.'
	},
	'glorot-bengio-2010': {
		authors: 'Glorot & Bengio',
		year: 2010,
		title: 'Understanding the difficulty of training deep feedforward neural networks',
		where: 'AISTATS 2010',
		url: 'https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf',
		note: 'Measures the saturation directly, layer by layer, and shows how a badly scaled start drives sigmoid units flat and holds them there.'
	},
	'nair-hinton-2010': {
		authors: 'Nair & Hinton',
		year: 2010,
		title: 'Rectified Linear Units Improve Restricted Boltzmann Machines',
		where: 'ICML 2010',
		url: 'https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf',
		note: 'Where the crease enters modern practice.'
	},
	'glorot-2011': {
		authors: 'Glorot, Bordes & Bengio',
		year: 2011,
		title: 'Deep Sparse Rectifier Neural Networks',
		where: 'AISTATS 2011',
		url: 'https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf',
		note: 'The case that relu wins not despite its dead half but partly because of it: at any moment most units are silent, and the network computes on a sparse subset of itself.'
	},
	'maas-2013': {
		authors: 'Maas, Hannun & Ng',
		year: 2013,
		title: 'Rectifier Nonlinearities Improve Neural Network Acoustic Models',
		where: 'ICML 2013 WDLASL',
		url: 'https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf',
		note: 'Leaky relu — a trickle of slope below zero, so a unit that falls silent can still be argued back.'
	},
	'hendrycks-gimpel-2016': {
		authors: 'Hendrycks & Gimpel',
		year: 2016,
		title: 'Gaussian Error Linear Units (GELUs)',
		where: 'arXiv preprint',
		url: 'https://arxiv.org/abs/1606.08415',
		note: 'The bend the GPT and BERT lines are built on: weight the input by the chance a standard normal falls below it.'
	},
	'elfwing-2017': {
		authors: 'Elfwing, Uchibe & Doya',
		year: 2017,
		title:
			'Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning',
		where: 'Neural Networks 107',
		url: 'https://arxiv.org/abs/1702.03118',
		note: 'SiLU, found while building reinforcement learners.'
	},
	'ramachandran-2017': {
		authors: 'Ramachandran, Zoph & Le',
		year: 2017,
		title: 'Searching for Activation Functions',
		where: 'arXiv preprint',
		url: 'https://arxiv.org/abs/1710.05941',
		note: 'The same function found again from the other end — by automated search over candidate formulas, which reported it as Swish before noticing it already had a name.'
	},
	'misra-2019': {
		authors: 'Misra',
		year: 2019,
		title: 'Mish: A Self Regularized Non-Monotonic Activation Function',
		where: 'BMVC 2020',
		url: 'https://arxiv.org/abs/1908.08681',
		note: 'Self-gated like silu, a shade softer, and the one computer vision kept.'
	},
	'telgarsky-2016': {
		authors: 'Telgarsky',
		year: 2016,
		title: 'Benefits of depth in neural networks',
		where: 'COLT 2016',
		url: 'https://arxiv.org/abs/1602.04485',
		note: 'A function a deep network draws with a handful of units that any shallow network would need an exponential number to match. Depth is not a convenience.'
	},
	'montufar-2014': {
		authors: 'Montúfar et al.',
		year: 2014,
		title: 'On the Number of Linear Regions of Deep Neural Networks',
		where: 'NeurIPS 2014',
		url: 'https://arxiv.org/abs/1402.1869',
		note: 'Counts the creases. A relu network folds its input space, and each new layer folds the folds — so the pieces multiply with depth and only add with width.'
	},
	'zhang-2017': {
		authors: 'Zhang et al.',
		year: 2017,
		title: 'Understanding deep learning requires rethinking generalization',
		where: 'ICLR 2017',
		url: 'https://arxiv.org/abs/1611.03530',
		note: 'Replace every label in a photograph dataset with a random one. The same networks fit all of it, perfectly. Whatever stops them from memorising real data, it is not a shortage of capacity.'
	},
	'belkin-2019': {
		authors: 'Belkin et al.',
		year: 2019,
		title: 'Reconciling modern machine learning practice and the bias-variance trade-off',
		where: 'PNAS 116(32)',
		url: 'https://arxiv.org/abs/1812.11118',
		note: 'The double-descent curve. Test error peaks exactly where a model can just barely fit its training set, and then falls again as the model keeps growing.'
	},
	'nakkiran-2019': {
		authors: 'Nakkiran et al.',
		year: 2019,
		title: 'Deep Double Descent: Where Bigger Models and More Data Hurt',
		where: 'ICLR 2020',
		url: 'https://arxiv.org/abs/1912.02292',
		note: 'The same second descent in real deep networks — in model size, in training time, and, uncomfortably, in dataset size.'
	},
	'olah-2014': {
		authors: 'Olah',
		year: 2014,
		title: 'Neural Networks, Manifolds, and Topology',
		where: 'colah.github.io',
		url: 'https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/',
		note: 'The essay that made the rubber-sheet picture stick, and the source of the tangled-rings example this chapter runs live.'
	},
	'fefferman-2016': {
		authors: 'Fefferman, Mitter & Narayanan',
		year: 2016,
		title: 'Testing the Manifold Hypothesis',
		where: 'Journal of the AMS 29(4)',
		url: 'https://arxiv.org/abs/1310.0425',
		note: 'The supposition that real data of very high dimension lies close to a surface of far lower dimension, treated for once as a claim that can be tested rather than assumed.'
	},
	'bengio-2013': {
		authors: 'Bengio, Courville & Vincent',
		year: 2013,
		title: 'Representation Learning: A Review and New Perspectives',
		where: 'IEEE TPAMI 35(8)',
		url: 'https://arxiv.org/abs/1206.5538',
		note: 'The case, made before it was obvious, that the useful output of a deep network is the coordinates it invents rather than the answer it prints.'
	},
	'naitzat-2020': {
		authors: 'Naitzat, Zhitnikov & Lim',
		year: 2020,
		title: 'Topology of deep neural networks',
		where: 'JMLR 21',
		url: 'https://arxiv.org/abs/2004.06093',
		note: 'Measures the holes in the data — literally, by counting them — at every layer of a trained network, and watches them close. Networks with a folding bend simplify the topology far faster than smooth ones.'
	},
	'alain-bengio-2016': {
		authors: 'Alain & Bengio',
		year: 2016,
		title: 'Understanding intermediate layers using linear classifier probes',
		where: 'ICLR 2017 workshop',
		url: 'https://arxiv.org/abs/1610.01644',
		note: 'Hang a bare linear classifier off each hidden layer and see how well it does. Separability rises layer by layer, which is the untangling of this chapter measured in a number.'
	},
	'lecun-1998': {
		authors: 'LeCun, Bottou, Bengio & Haffner',
		year: 1998,
		title: 'Gradient-Based Learning Applied to Document Recognition',
		where: 'Proceedings of the IEEE 86(11)',
		url: 'https://leon.bottou.org/publications/pdf/ieee-1998.pdf',
		note: 'Where MNIST gets its name, and the account of the cheque-reading system this recipe was actually deployed as. Forty-six pages, and worth the afternoon.'
	},
	'recht-2019': {
		authors: 'Recht et al.',
		year: 2019,
		title: 'Do ImageNet Classifiers Generalize to ImageNet?',
		where: 'ICML 2019',
		url: 'https://arxiv.org/abs/1902.10811',
		note: 'Builds a fresh test set the same way the original was built, and finds every model does worse on it. A held-out set stops being held out once a field has spent a decade choosing what to publish against it.'
	},
	'hestness-2017': {
		authors: 'Hestness et al.',
		year: 2017,
		title: 'Deep Learning Scaling is Predictable, Empirically',
		where: 'arXiv preprint',
		url: 'https://arxiv.org/abs/1712.00409',
		note: 'Error falls as a power law in the amount of training data, with the same shape across vision, speech and language. Adding data is not just usually better; how much better is forecastable.'
	},
	'simonyan-2014': {
		authors: 'Simonyan, Vedaldi & Zisserman',
		year: 2014,
		title:
			'Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps',
		where: 'ICLR 2014 workshop',
		url: 'https://arxiv.org/abs/1312.6034',
		note: 'The gradient of a class score with respect to the pixels, read as a picture — the evidence square in this chapter, in its original form.'
	},
	'adebayo-2018': {
		authors: 'Adebayo et al.',
		year: 2018,
		title: 'Sanity Checks for Saliency Maps',
		where: 'NeurIPS 2018',
		url: 'https://arxiv.org/abs/1810.03292',
		note: "Randomise a trained network's weights and several popular saliency methods keep producing the same convincing picture. A map that survives the destruction of the model was never explaining the model."
	},
	'szegedy-2014': {
		authors: 'Szegedy et al.',
		year: 2014,
		title: 'Intriguing properties of neural networks',
		where: 'ICLR 2014',
		url: 'https://arxiv.org/abs/1312.6199',
		note: 'The discovery of adversarial examples: a change too small for a person to see, chosen deliberately, flips the verdict with confidence.'
	},
	'goodfellow-adv-2015': {
		authors: 'Goodfellow, Shlens & Szegedy',
		year: 2015,
		title: 'Explaining and Harnessing Adversarial Examples',
		where: 'ICLR 2015',
		url: 'https://arxiv.org/abs/1412.6572',
		note: 'Argues the cause is not exotic: in high dimensions a great many tiny nudges, each harmless, add up along the direction the model is most sensitive to.'
	},
	'lecun-1990': {
		authors: 'LeCun et al.',
		year: 1990,
		title: 'Handwritten Digit Recognition with a Back-Propagation Network',
		where: 'NeurIPS 1989',
		url: 'https://proceedings.neurips.cc/paper_files/paper/1989/hash/53c3bce66e43be4f209556518c2fcb54-Abstract.html',
		note: 'The postal-code reader, trained by backpropagation on zip codes off real US mail — the first convincing demonstration that this recipe works on pixels.'
	},
	'hinton-salakhutdinov-2006': {
		authors: 'Hinton & Salakhutdinov',
		year: 2006,
		title: 'Reducing the Dimensionality of Data with Neural Networks',
		where: 'Science 313(5786)',
		url: 'https://www.cs.toronto.edu/~hinton/absps/science.pdf',
		note: 'The paper that made deep autoencoders work, and the one that shows what the waist buys over its linear ancestor: strip every bend out of an autoencoder and the best it can do is principal component analysis.'
	},
	'kingma-welling-2014': {
		authors: 'Kingma & Welling',
		year: 2014,
		title: 'Auto-Encoding Variational Bayes',
		where: 'ICLR 2014',
		url: 'https://arxiv.org/abs/1312.6114',
		note: 'The variational autoencoder, and the reparameterisation that makes it trainable — sample the noise outside the network so the gradient has a path through the sampling step rather than into it.'
	},
	'rezende-2014': {
		authors: 'Rezende, Mohamed & Wierstra',
		year: 2014,
		title: 'Stochastic Backpropagation and Approximate Inference in Deep Generative Models',
		where: 'ICML 2014',
		url: 'https://arxiv.org/abs/1401.4082',
		note: 'The same idea, arrived at independently and published within months. It happens more often than the textbooks admit.'
	},
	'higgins-2017': {
		authors: 'Higgins et al.',
		year: 2017,
		title: 'β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework',
		where: 'ICLR 2017',
		url: 'https://openreview.net/forum?id=Sy2fzU9gl',
		note: "Where the β in this chapter's loss comes from. Turn the rent up and the map gets tidier and the rebuilds get worse; the paper is an argument about what the tidiness is worth."
	},
	'goodfellow-gan-2014': {
		authors: 'Goodfellow et al.',
		year: 2014,
		title: 'Generative Adversarial Networks',
		where: 'NeurIPS 2014',
		url: 'https://arxiv.org/abs/1406.2661',
		note: 'The first widely used answer to the blur: stop scoring a rebuild against its original pixel by pixel, and train a second network to say whether it looks real.'
	},
	'ho-2020': {
		authors: 'Ho, Jain & Abbeel',
		year: 2020,
		title: 'Denoising Diffusion Probabilistic Models',
		where: 'NeurIPS 2020',
		url: 'https://arxiv.org/abs/2006.11239',
		note: 'The answer that won. Learn to remove a little noise at a time and the model never has to average over everything an address could mean — it commits, gradually.'
	},
	'oord-2018': {
		authors: 'van den Oord, Li & Vinyals',
		year: 2018,
		title: 'Representation Learning with Contrastive Predictive Coding',
		where: 'arXiv preprint',
		url: 'https://arxiv.org/abs/1807.03748',
		note: 'The contrastive loss in the form nearly everything since uses it: score the true partner against a crowd of impostors.'
	},
	'chen-simclr-2020': {
		authors: 'Chen et al.',
		year: 2020,
		title: 'A Simple Framework for Contrastive Learning of Visual Representations',
		where: 'ICML 2020',
		url: 'https://arxiv.org/abs/2002.05709',
		note: 'Two crops of one photograph should land in the same place; two crops of different photographs should not. That, done carefully enough, is most of what a modern image encoder learns.'
	},
	'radford-2021': {
		authors: 'Radford et al.',
		year: 2021,
		title: 'Learning Transferable Visual Models From Natural Language Supervision',
		where: 'ICML 2021',
		url: 'https://arxiv.org/abs/2103.00020',
		note: 'CLIP — the same contrastive trick with the two views being a picture and its caption, which puts images and text into one shared space and is why you can search a photo library by typing.'
	},
	'bengio-2003': {
		authors: 'Bengio et al.',
		year: 2003,
		title: 'A Neural Probabilistic Language Model',
		where: 'JMLR 3',
		url: 'https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf',
		note: 'Twenty years early: learn a vector per word and predict the next one with a neural network, so that a sentence never seen can borrow from a sentence that was.'
	},
	'mikolov-2013a': {
		authors: 'Mikolov et al.',
		year: 2013,
		title: 'Efficient Estimation of Word Representations in Vector Space',
		where: 'ICLR 2013 workshop',
		url: 'https://arxiv.org/abs/1301.3781',
		note: 'word2vec. The models are deliberately shallow — the point of the paper is that throwing away the hidden layer let them train on a hundred billion words.'
	},
	'mikolov-2013b': {
		authors: 'Mikolov et al.',
		year: 2013,
		title: 'Distributed Representations of Words and Phrases and their Compositionality',
		where: 'NeurIPS 2013',
		url: 'https://arxiv.org/abs/1310.4546',
		note: 'Negative sampling — the loss written out in this chapter — and the paper that put the analogy arithmetic in front of everyone.'
	},
	'levy-goldberg-2014': {
		authors: 'Levy & Goldberg',
		year: 2014,
		title: 'Neural Word Embedding as Implicit Matrix Factorization',
		where: 'NeurIPS 2014',
		url: 'https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html',
		note: 'Shows that skip-gram with negative sampling is quietly factorising a word-by-context table of co-occurrence statistics. The neural part is a way of doing the arithmetic, not the source of the magic.'
	},
	'linzen-2016': {
		authors: 'Linzen',
		year: 2016,
		title: 'Issues in evaluating semantic spaces using word analogies',
		where: 'RepEval 2016',
		url: 'https://arxiv.org/abs/1606.07736',
		note: 'The famous analogy result is doing less work than it appears to: the standard scoring rule forbids the answer from being any of the three input words, and much of the accuracy comes from that exclusion rather than from the arithmetic.'
	},
	'sennrich-2016': {
		authors: 'Sennrich, Haddow & Birch',
		year: 2016,
		title: 'Neural Machine Translation of Rare Words with Subword Units',
		where: 'ACL 2016',
		url: 'https://arxiv.org/abs/1508.07909',
		note: 'Byte-pair encoding was a data-compression trick from 1994. This is the paper that pointed it at vocabulary, and every tokenizer since is a variation on it.'
	},
	'shannon-1951': {
		authors: 'Shannon',
		year: 1951,
		title: 'Prediction and Entropy of Printed English',
		where: 'Bell System Technical Journal 30(1)',
		url: 'https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf',
		note: 'The next-token game, played by people, in 1951 — Shannon sat readers down with covered text and had them guess the next letter, and turned their guesses into a measurement of English itself.'
	},
	'deletang-2023': {
		authors: 'Delétang et al.',
		year: 2023,
		title: 'Language Modeling Is Compression',
		where: 'ICLR 2024',
		url: 'https://arxiv.org/abs/2309.10668',
		note: 'Takes the identity literally and uses a language model as a general-purpose compressor. It beats the specialist formats — on text, and, oddly, on images and audio it was never trained on.'
	},
	'vaswani-2017': {
		authors: 'Vaswani et al.',
		year: 2017,
		title: 'Attention Is All You Need',
		where: 'NeurIPS 2017',
		url: 'https://arxiv.org/abs/1706.03762',
		note: 'The transformer. The title is an argument: everything before it wrapped attention around a recurrent network, and the finding was that you could throw the recurrence away.'
	},
	'elhage-2021': {
		authors: 'Elhage et al.',
		year: 2021,
		title: 'A Mathematical Framework for Transformer Circuits',
		where: 'Anthropic',
		url: 'https://transformer-circuits.pub/2021/framework/index.html',
		note: 'Where the residual-stream reading of a transformer is set out carefully: not a pipeline that transforms a vector but a shared channel that every block reads from and adds to.'
	},
	'su-2021': {
		authors: 'Su et al.',
		year: 2021,
		title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding',
		where: 'Neurocomputing 568',
		url: 'https://arxiv.org/abs/2104.09864',
		note: 'How position is actually supplied in current models: rotate each query and key by an angle proportional to its position, so what attention sees is the distance between two tokens rather than their absolute slots.'
	},
	'kaplan-2020': {
		authors: 'Kaplan et al.',
		year: 2020,
		title: 'Scaling Laws for Neural Language Models',
		where: 'arXiv preprint',
		url: 'https://arxiv.org/abs/2001.08361',
		note: 'Loss falls as a power law in parameters, data and compute, over seven orders of magnitude. The straightness of those lines is the reason anyone was willing to spend the money.'
	},
	'hoffmann-2022': {
		authors: 'Hoffmann et al.',
		year: 2022,
		title: 'Training Compute-Optimal Large Language Models',
		where: 'NeurIPS 2022',
		url: 'https://arxiv.org/abs/2203.15556',
		note: 'Chinchilla. Redid the measurement and found the field had been building models far too large for the amount of text it was feeding them — a correction worth more than most architectural ideas of the same period.'
	},
	'minsky-1961': {
		authors: 'Minsky',
		year: 1961,
		title: 'Steps Toward Artificial Intelligence',
		where: 'Proceedings of the IRE 49(1)',
		url: 'https://courses.csail.mit.edu/6.803/pdf/steps.pdf',
		note: 'Where the credit-assignment problem is named and stated, decades before anyone could do much about it. Still one of the best-written papers in the field.'
	},
	'williams-1992': {
		authors: 'Williams',
		year: 1992,
		title:
			'Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning',
		where: 'Machine Learning 8',
		url: 'https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf',
		note: 'REINFORCE, in its original form, with the baseline already in it — including the proof that subtracting one cannot bias the estimate.'
	},
	'sutton-2000': {
		authors: 'Sutton et al.',
		year: 2000,
		title: 'Policy Gradient Methods for Reinforcement Learning with Function Approximation',
		where: 'NeurIPS 1999',
		url: 'https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html',
		note: 'The policy gradient theorem stated in general, and the result that lets a learned value function stand in for the return without breaking the gradient.'
	},
	'sutton-barto-2018': {
		authors: 'Sutton & Barto',
		year: 2018,
		title: 'Reinforcement Learning: An Introduction (2nd edition)',
		where: 'MIT Press',
		url: 'https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf',
		note: 'The book, given away by its authors. If any single chapter of this one leaves you wanting the whole subject properly, it is this one.'
	},
	'kakade-langford-2002': {
		authors: 'Kakade & Langford',
		year: 2002,
		title: 'Approximately Optimal Approximate Reinforcement Learning',
		where: 'ICML 2002',
		url: 'https://www.cs.cmu.edu/~./jcl/papers/aoarl/Final.pdf',
		note: 'Makes the case that where you start an episode is part of the algorithm, not part of the problem: scattering the start states is what keeps the gradient informative when the goal is somewhere a wanderer never reaches.'
	},
	'florensa-2017': {
		authors: 'Florensa et al.',
		year: 2017,
		title: 'Reverse Curriculum Generation for Reinforcement Learning',
		where: 'CoRL 2017',
		url: 'https://arxiv.org/abs/1707.05300',
		note: 'Start the agent next to the goal, where success is easy and the reward is reachable, and walk the start states backwards as it improves. The catch in this chapter is trained exactly this way.'
	},
	'krakovna-2020': {
		authors: 'Krakovna et al.',
		year: 2020,
		title: 'Specification gaming: the flip side of AI ingenuity',
		where: 'DeepMind',
		url: 'https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/',
		note: 'A long, funny, slightly alarming collection of agents that got exactly what was asked for. The boat that farms turbo pickups in a circle instead of finishing the race is the canonical one.'
	}
} as const satisfies Record<string, Paper>;

export type PaperId = keyof typeof papers;

/**
 * Which papers each chapter cites, in the order the prose first reaches for
 * them. Position here is the numeral the reader sees.
 */
export const citationOrder: Partial<Record<string, readonly PaperId[]>> = {
	neuron: [
		'cybenko-1989',
		'lu-2017',
		'glorot-bengio-2010',
		'nair-hinton-2010',
		'maas-2013',
		'glorot-2011',
		'hendrycks-gimpel-2016',
		'elfwing-2017',
		'misra-2019',
		'ramachandran-2017',
		'montufar-2014',
		'telgarsky-2016',
		'zhang-2017',
		'belkin-2019',
		'nakkiran-2019'
	],
	space: ['olah-2014', 'fefferman-2016', 'bengio-2013', 'alain-bengio-2016', 'naitzat-2020'],
	digits: [
		'lecun-1998',
		'recht-2019',
		'hestness-2017',
		'simonyan-2014',
		'adebayo-2018',
		'szegedy-2014',
		'goodfellow-adv-2015',
		'lecun-1990'
	],
	latent: [
		'hinton-salakhutdinov-2006',
		'kingma-welling-2014',
		'rezende-2014',
		'higgins-2017',
		'goodfellow-gan-2014',
		'ho-2020',
		'oord-2018',
		'chen-simclr-2020',
		'radford-2021'
	],
	language: [
		'bengio-2003',
		'mikolov-2013a',
		'mikolov-2013b',
		'levy-goldberg-2014',
		'linzen-2016',
		'sennrich-2016',
		'shannon-1951',
		'deletang-2023',
		'su-2021',
		'vaswani-2017',
		'elhage-2021',
		'kaplan-2020',
		'hoffmann-2022'
	],
	reward: [
		'minsky-1961',
		'sutton-2000',
		'williams-1992',
		'sutton-barto-2018',
		'kakade-langford-2002',
		'florensa-2017',
		'krakovna-2020'
	],
	descent: [
		'polyak-1964',
		'kingma-ba-2015',
		'loshchilov-hutter-2019',
		'chen-2023',
		'duchi-2011',
		'tieleman-hinton-2012',
		'cohen-2021',
		'robbins-monro-1951',
		'dauphin-2014',
		'goodfellow-2015',
		'li-2018'
	]
};

/** 1-based position of a paper in its chapter's list, or undefined. */
export function citationNumber(chapter: string, id: PaperId): number | undefined {
	const i = citationOrder[chapter]?.indexOf(id) ?? -1;
	return i < 0 ? undefined : i + 1;
}

/** The in-page anchor a reference publishes, so a citation can jump to it. */
export function refAnchor(id: string): string {
	return `ref-${id}`;
}

/**
 * What the outbound link is about to open, said plainly: "arXiv:1412.6980" if
 * that is where it goes, otherwise the host. Readers deserve to know whether
 * a click lands on a PDF, a preprint, or a lecture handout.
 */
export function sourceLabel(url: string): string {
	const arxiv = /arxiv\.org\/abs\/([^?#]+)/.exec(url);
	if (arxiv) return `arXiv:${arxiv[1]}`;
	const doi = /doi\.org\/(.+)$/.exec(url);
	if (doi) return `doi:${doi[1]}`;
	return new URL(url).hostname.replace(/^www\./, '');
}
