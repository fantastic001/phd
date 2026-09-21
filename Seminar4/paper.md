---
title: "Effect of Graph Partitioning on Link Prediction with Bilinear, Hadamard, and Random Forest Decoders over Static Node2Vec Embeddings"
author: "Stefan Nožinić"
abstract: |
  This paper studies how graph partitioning affects link prediction when node embeddings are computed independently on each partition with static node2vec and merged afterwards. Unlike graph reconstruction, which asks how well embeddings recover the graph they were trained on, link prediction asks how well they generalize to held-out edges, and the two questions turn out to have opposite answers under partitioning. Using four real-world graphs (CITESEER, AstroPh, Cit-HepPh, AS-Oregon), partition counts $P \in \{1, 2, 4, 8\}$, and 10 independent runs per configuration (480 runs), we compare three link-prediction decoders on the merged embeddings: a Hadamard-product logistic decoder, a bilinear decoder $u^\top W v$, and a random forest on concatenated embeddings. Decoders are evaluated with a ranking protocol (Hits@k and mean reciprocal rank, MRR) in which each held-out edge is ranked against 500 negative candidates, and configurations are compared with Mann--Whitney tests and Cohen's $d$. Partitioning strongly degrades the neural decoders: from $P=1$ to $P=8$ the MRR of the bilinear decoder falls by 39--60% (significantly at 11 of 12 partition-count steps) and that of the Hadamard decoder by 8--40%, mostly in the first step, whereas the MRR of the random forest changes only between +3% and -25%. Consequently the decoders are ordered random forest > bilinear > Hadamard in most configurations, with the bilinear decoder best only on unpartitioned dense graphs. In a preliminary single-dataset comparison, label propagation (LPA) partitioning gave better link prediction than the Lancichinetti-Fortunato-Kertesz method (LFM) at every partition count tested. Finally, the reconstruction F1 score of the same embeddings increases with the partition count on three of the four datasets while neural-decoder link prediction deteriorates, so reconstruction quality is not a reliable proxy for downstream link-prediction quality in a partitioned embedding pipeline.
bibliography: ./refs.bib
---

# Introduction

A graph is a mathematical structure consisting of vertices (or nodes) connected by edges. Graphs are widely used to model relationships and interactions in various domains [@van_der_hofstad_random_2024], such as social networks [@leskovec_signed_2010] [@backstrom_group_2006] [@rozemberczki_twitch_2021], collaboration networks [@savic_analysis_2017], terrorist networks [@krebs_mapping_2002] and blog citation networks [@adamic_political_2005]. Graph vertex embeddings represent vertices as low-dimensional vectors, enabling machine learning tasks such as vertex classification, community detection, and link prediction [@leskovec_predicting_2010]. Among these downstream tasks, link prediction is of particular practical importance: it underlies friend and content recommendation, knowledge-graph completion [@bordes_translating_2013], and the inference of missing or future edges in evolving networks.

State-of-the-art embedding methods such as node2vec [@grover_node2vec_2016] compute embeddings from a single, monolithic training pass over the whole graph. When the graph is too large to embed on one machine, a common strategy is to partition the graph, embed each partition independently, and combine the resulting per-partition embeddings for downstream use [@fang_distributed_2023] [@lombardo_scalable_2019]. This paper's companion study (Seminar 3) examined how such partitioning affects the *reconstruction* quality of the embeddings — how well the embeddings recover the edges of the graph they were trained on. Link prediction poses a different and arguably more practically relevant question: how well do the embeddings generalize to edges that were *not* observed during training? Because independently embedded partitions have no shared frame of reference — each partition's embedding space is the arbitrary output of a separate skip-gram training run [@church_word2vec_2017] — a decoder or classifier that combines embeddings from different partitions must cope with per-partition embedding spaces that are not directly comparable to one another. This raises questions that do not arise in the single-partition setting: which partitioning strategy best preserves the community structure that link prediction relies on, which decoder architecture best tolerates misaligned per-partition latent spaces, and whether a downstream classifier can be made robust to this misalignment at all.

# Problem Formulation

Given a static graph partitioned into $P$ partitions, each embedded independently with static node2vec, we want to know how

* the number of partitions $P$, and the partitioning strategy (LFM vs.\ LPA), and
* the link-prediction decoder (bilinear, Hadamard, or random forest)

affect link-prediction quality, and how that quality relates to the graph-reconstruction quality of the same embeddings.

# Related Work

Graph vertex embedding is a well-studied area, with various methods proposed to generate low-dimensional representations of vertices in a graph. Node2Vec [@grover_node2vec_2016] is the state-of-the-art method used throughout this paper; it generates embeddings via biased random walks and has been shown to be effective at capturing community structure. As its improvement, DistGER [@fang_distributed_2023] is a distributed graph embedding method that extends Node2Vec by leveraging distributed computing to handle large graphs, and [@lombardo_scalable_2019] proposes an actor-model-based distributed framework for the same purpose. The common ground for these methods is that they generate walks which are later used to train a Word2Vec model [@church_word2vec_2017] commonly used for generating embeddings in natural language processing tasks.

Link prediction from vertex embeddings is commonly formulated as a binary classification problem over pairs of vertices, using a decoder that maps a pair of embeddings to a link probability. The original node2vec paper [@grover_node2vec_2016] popularized the use of simple binary operators — including the Hadamard product, which is one of the decoders compared in this paper — to combine two node embeddings into a single edge feature vector before classification. Leskovec et al. [@leskovec_predicting_2010] study link sign and formation prediction more broadly, motivating the classification-based framing of link prediction adopted here. Beyond the Hadamard product, a learned bilinear form over the two embeddings is a common, more expressive scoring function, and non-neural classifiers such as random forests [@breiman_random_2001] can be applied directly to the concatenated embeddings.

Evaluation protocol matters as much as the decoder. Li et al. [@li_evaluating_2023] point out that link-prediction results are often not comparable because of inconsistent evaluation settings, and propose the HeaRT benchmark, which ranks every positive test edge against a set of 500 negative candidates and reports the mean reciprocal rank (MRR) and Hits@k. The ranking protocol used in this paper follows this per-positive design, but draws the candidates uniformly at random instead of selecting hard negatives with heuristics.

Partitioning a graph while preserving the community structure it relies on is itself a well-studied problem. Community-detection-based partitioners are the natural choice when partitioning is intended to preserve locality for downstream embedding: the label propagation algorithm (LPA) [@raghavan_near_2007] assigns each vertex the most frequent label among its neighbors, iterating until labels stabilize into communities, while the Lancichinetti-Fortunato-Kertesz (LFM) method [@lancichinetti_detecting_2009] detects (possibly overlapping) communities by greedily optimizing a local fitness function around a seed vertex. Both methods have been used as static graph partitioners in this line of work, and this paper compares their impact on downstream link-prediction quality, complementing Seminar 3's neighbor-based buffered partitioner for online node arrivals, and the classical static partitioning literature surveyed there [@benlic_effective_2010] [@sanders_distributed_2012] [@sanders_engineering_2011] [@romero_ruiz_memetic_2018] [@catalyurek_more_2023].

Distributed embedding systems that partition a graph before embedding face a fundamental representation problem once embeddings must be combined across partitions: because each partition is embedded independently, there is no guarantee that geometrically similar vectors in two different partitions' embedding spaces represent structurally similar vertices. This paper studies the practical consequences of that misalignment for link prediction, complementing Seminar 3's study of its consequences for graph reconstruction.

# Contributions

This paper makes the following contributions:

* We measure, on four real-world graphs with 10 independent runs per configuration, how the partition count affects link prediction over independently computed static node2vec embeddings for three decoders (bilinear, Hadamard, random forest), using a per-positive ranking protocol (Hits@k, MRR) and statistical tests (Mann--Whitney, Cohen's $d$).
* We show that the neural decoders lose a large part of their ranking quality when the graph is partitioned, whereas a random forest on concatenated embeddings is nearly insensitive to the partition count, and we derive a decoder ordering (random forest > bilinear > Hadamard) together with the conditions in which it changes.
* We show that graph-reconstruction F1 and neural-decoder link-prediction quality move in opposite directions as the partition count grows, so reconstruction quality is not a reliable proxy for downstream link-prediction quality, and we compare LFM and LPA partitioning in a preliminary experiment.

# Paper Organization

The rest of the paper is organized as follows: first, the system overview is presented, then graph partitioning, the embedding model, and link-prediction decoders and classifiers are described. Next, benchmarks and evaluation metrics are explained in detail. Finally, results and discussion are presented, followed by the conclusion and references.

# Methods

## System overview

This paper considers a fully known static graph $G = (V, E)$. The pipeline has five stages: (1) a fraction of the edges is held out as positive test pairs, and an equal number of non-adjacent vertex pairs is drawn as negative test pairs; (2) the remaining training graph $G_{\text{train}}$ is partitioned once into $P$ partitions; (3) each partition's induced subgraph is embedded independently with static node2vec; (4) the per-partition embeddings are merged into one embedding per vertex, on which a link-prediction decoder is trained; and (5) the decoder is evaluated on the held-out pairs. Because each partition is embedded in isolation from the others, the resulting per-partition embedding spaces are not trained to be mutually consistent: a large dot product between two vectors in partition $p_i$'s embedding space carries no guaranteed relationship to a large dot product in partition $p_j$'s embedding space. This is the central complication that distinguishes distributed link prediction from the single-partition case, and every design choice examined in this paper --- partitioning strategy and decoder --- is, in one way or another, a response to it.

A candidate pair $(u, v)$ may span two different partitions. For a pair whose vertices lie in the same partition, both embeddings come from the same, internally consistent embedding space. For a pair whose vertices lie in different partitions, the decoder must combine embeddings from two independently trained spaces, and it is this cross-partition case that is expected to drive most of the quality loss observed in the Results section.

## Graph partitioning

Both partitioners are community-detection-based and follow the same two-step scheme: detect communities in $G_{\text{train}}$, then pack the communities into exactly $P$ partitions with a greedy, size-balancing bin-packing heuristic (communities sorted by size, each placed into the currently smallest partition). Vertices of $G_{\text{train}}$ therefore stay together with their community, and the partition sizes are kept approximately equal.

**LPA (Label Propagation Algorithm)** [@raghavan_near_2007]: every vertex is initialized with a unique label $\ell(v) = v$. In each sweep, vertices are visited in random order and each vertex adopts the label held by the plurality of its neighbors, with ties broken uniformly at random,

$$ \ell(v) \leftarrow \arg\max_{\ell} \; \big|\{ w \in N(v) : \ell(w) = \ell \}\big| , $$

until no label changes. Vertices sharing a final label form a community. The asynchronous variant is used, in which updated labels are visible immediately within a sweep. LPA has no tunable parameters and produces communities of very different sizes, which the bin-packing step then groups into $P$ partitions. Communities are disjoint, so every vertex belongs to exactly one partition.

**LFM (Lancichinetti-Fortunato-Kertesz method)** [@lancichinetti_detecting_2009]: a community $C$ is grown greedily from a randomly chosen seed vertex that is not yet covered. Community quality is measured by the fitness

$$ f(C) = \frac{k_{\text{in}}(C)}{\big(k_{\text{in}}(C) + k_{\text{out}}(C)\big)^{\alpha}} $$

where $k_{\text{in}}(C)$ and $k_{\text{out}}(C)$ are the total internal and external degrees of $C$ and $\alpha$ is a resolution parameter. At each step the neighboring vertex with the largest positive fitness gain is added, vertices whose removal increases fitness are dropped, and growth stops when no neighbor improves fitness. Communities produced this way may overlap. A modified variant is used here, which (i) repeats seeding until the fraction of uncovered vertices falls to a threshold $\tau$, (ii) caps the number of communities at $10 P$ to guarantee termination when growth does not converge, (iii) pads with empty communities if fewer than $P$ are found, and (iv) assigns any vertex still uncovered to a uniformly random community. Parameters are $\alpha = 1$ and $\tau = 0$ (every vertex must be covered). Because LFM communities can overlap, a vertex may end up in more than one partition after bin packing; this is handled in the embedding-merging step below.

Both methods partition the graph based on its community structure rather than on vertex arrival order, in contrast to the online, neighbor-count-based partitioner used in Seminar 3. Because the full training graph is known in advance, both can consider the entire neighborhood of every vertex.

## Embedding model

Each partition's induced subgraph $G_{\text{train}}[p_i]$ is embedded independently using static node2vec [@grover_node2vec_2016]. Node2vec generates $r$ second-order random walks of length $l$ from every vertex. For a walk that has just moved from $t$ to $x$, the unnormalized probability of stepping next to a neighbor $y$ of $x$ is

$$ \pi(x, y \mid t) = \begin{cases} 1/p & \text{if } y = t, \\ 1 & \text{if } y \in N(t), \\ 1/q & \text{otherwise,} \end{cases} $$

so that the return parameter $p$ and in-out parameter $q$ control how strongly the walk backtracks or moves outward. The walks are treated as sentences and used to train a skip-gram model [@church_word2vec_2017] with negative sampling, which maximizes, for every vertex $v$ and every context vertex $c$ within a window of $w$ positions,

$$ \log \sigma(z_v^\top z_c') + \sum_{k=1}^{K} \mathbb{E}_{n_k \sim P_n}\, \log \sigma(-z_v^\top z_{n_k}') $$

where $z$ and $z'$ are the input and context vectors and $K$ negative samples are drawn per positive pair. The input vectors $z_v \in \mathbb{R}^d$ are the embeddings. The following parameters are used for all datasets: $r = 10$ walks per vertex, walk length $l = 80$, window size $w = 10$, $K = 5$ negative samples, learning rate $0.01$, and $10$ training epochs. The return parameter $p$, in-out parameter $q$, and dimension $d$ are dataset-specific and were chosen by a preliminary search over $p, q \in \{0.25, 0.5, 1, 2, 4\}$ and several embedding dimensions, scored by reconstruction F1 (Table 1). No information is shared between partitions during training: the skip-gram model of partition $p_i$ is initialized and optimized independently of every other partition's model.

| Dataset   | p   | q    | d   |
|-----------|-----|------|-----|
| CITESEER  | 0.5 | 0.25 | 50  |
| AstroPh   | 2   | 0.25 | 50  |
| Cit-HepPh | 4   | 0.25 | 100 |
| AS-Oregon | 0.5 | 2    | 128 |
Table: Node2vec return parameter $p$, in-out parameter $q$, and embedding dimension $d$ for each dataset.

Since a vertex can belong to several partitions (only under LFM), the final embedding of vertex $u$ is the mean of its per-partition embeddings, as in Seminar 3:

$$ z_u = \frac{1}{|\mathcal{R}_u|} \sum_{p \in \mathcal{R}_u} z_u^{(p)} , $$

where $\mathcal{R}_u$ is the set of partitions containing $u$. Under LPA, $|\mathcal{R}_u| = 1$ and $z_u$ is simply the embedding from the single partition containing $u$. All decoders below operate on these merged embeddings, so the decoder itself is trained once, on vertices from all partitions, rather than once per partition.

## Link-prediction data and decoders

**Held-out edges.** A uniformly random $10\%$ of the edges of $G$ is removed and kept as the positive test set $E^{+}_{\text{test}}$; the rest form $G_{\text{train}}$, on which partitioning and embedding are performed. The negative test set $E^{-}_{\text{test}}$ contains $|E^{+}_{\text{test}}|$ vertex pairs $(u, v)$, $u \neq v$, drawn uniformly at random from $V \times V$ subject to $(u, v) \notin E$, so the test set is balanced.

**Decoder training data.** Positive training pairs are edges of $G_{\text{train}}$ and negative training pairs are uniformly random vertex pairs that are not edges of $G_{\text{train}}$, with as many negatives as positives. The training edges are randomly split $80\%/20\%$ into training and validation sets. This procedure is repeated $10$ times with independent random splits and negatives, and the decoder with the lowest final validation loss is kept.

Given the merged embeddings $u = z(a)$ and $v = z(b)$ of two vertices $a, b$, three decoders producing a link score are compared.

**Hadamard decoder.** The embeddings are combined with the elementwise (Hadamard) product [@grover_node2vec_2016] and passed through a learned linear layer without bias:

$$ s(u, v) = w^\top (u \odot v) = \sum_{i=1}^{d} w_i u_i v_i , $$

with learned weights $w \in \mathbb{R}^d$. This is the simplest decoder considered and is equivalent to logistic regression on the Hadamard feature vector.

**Bilinear decoder.** A learned matrix $W \in \mathbb{R}^{d \times d}$ captures interactions between all pairs of coordinates of the two embeddings:

$$ s(u, v) = u^\top W v , $$

without bias and without any symmetry constraint on $W$; the Hadamard decoder is the special case in which $W$ is restricted to be diagonal.

For both neural decoders the link probability is $\hat{y} = \sigma(s(u, v))$ and the parameters are trained by minimizing the binary cross-entropy

$$ \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \Big] $$

with minibatch stochastic gradient descent (no momentum), batch size $32$, learning rate $0.1$, and $10$ epochs.

**Random forest (RDF).** Instead of a hand-designed interaction between the two embeddings, a random forest [@breiman_random_2001] is trained on the concatenated feature vector $[u; v] \in \mathbb{R}^{2d}$. The forest consists of $20$ decision trees, each grown on a bootstrap sample of the training pairs and choosing every split from a random subset of the features, with a fixed random seed; the predicted link probability is the mean of the trees' class-probability estimates. Because it builds axis-aligned splits directly on the raw coordinates, the random forest does not assume that the two embeddings live in a common, smoothly varying latent geometry.

For all three decoders a pair is predicted to be an edge if $\hat{y} > 0.5$.

## Benchmarks and evaluation metrics

The decoders are evaluated on the held-out pairs with two complementary protocols, alongside the reconstruction score used in Seminar 3.

**Classification metrics**: on the balanced test set $E^{+}_{\text{test}} \cup E^{-}_{\text{test}}$, precision, recall, F1 score, and accuracy of the edge/non-edge decision at threshold $0.5$ are computed. Accuracy is used in the preliminary comparison of partitioning strategies.

**Ranking metrics**: following the per-positive ranking design of the HeaRT benchmark [@li_evaluating_2023], but with uniformly random rather than heuristically selected negatives, for each of up to $1000$ randomly sampled positive test edges $(a, b)$, $500$ negative candidates $(a, c)$ are generated by drawing $c$ uniformly at random from $V$ such that $(a, c)$ is not an edge of $G_{\text{train}}$. The true edge and its candidates are scored by the decoder, and the rank of the true edge is

$$ \text{rank}(a, b) = 1 + \big|\{ c : \hat{y}(a, c) > \hat{y}(a, b) \}\big| , $$

so that ties are resolved in favor of the true edge. Over the sampled set $S$ of positive edges, the following are reported: **Hits@k**, the fraction of positive edges ranked within the top $k$, for $k \in \{1, 3, 10\}$; and the **Mean Reciprocal Rank**,

$$ \text{MRR} = \frac{1}{|S|} \sum_{(a,b) \in S} \frac{1}{\text{rank}(a, b)} ; $$

Unlike the balanced classification metrics, the ranking metrics ask whether the true neighbor scores above hundreds of alternatives for the same source vertex, which is a considerably harder and more discriminative task.

**RESTORE F1 (graph reconstruction)**: as in Seminar 3 [@yip_restore_2023], the graph is reconstructed from the merged embeddings by connecting the $m = |E_{\text{train}}|$ closest vertex pairs in the embedding space, and the F1 score of the neighborhood-level precision and recall with respect to $G_{\text{train}}$ is computed, using the definitions given in Seminar 3. It is evaluated on the training graph, not on the held-out edges, and is reported alongside the link-prediction metrics specifically to contrast reconstruction quality against generalization quality on the same set of embeddings.

# Results and discussion

## Experimental setup

The main experiment covers four datasets (CITESEER, AstroPh, Cit-HepPh, AS-Oregon), four partition counts $P \in \{1, 2, 4, 8\}$, and the three decoders (bilinear, Hadamard, random forest), with $10$ independent runs per configuration, i.e.\ $4 \times 4 \times 3 \times 10 = 480$ runs. All runs use LPA partitioning and the dataset-specific node2vec parameters of Table 1. A run repeats the whole pipeline from scratch: a new random hold-out split, a new partitioning, new random walks and skip-gram training, and new decoder training, so the $10$ runs of a configuration are independent samples. Tables report the mean $\pm$ standard deviation over these $10$ runs. Configurations are compared with the two-sided Mann--Whitney $U$ test together with Cohen's $d$ (computed with the pooled standard deviation), at $\alpha = 0.05$ and without correction for multiple comparisons; with $n = 10$ per group, a large $|d|$ accompanies most significant differences, and the effect size rather than the $p$-value should carry the practical interpretation.

The RESTORE F1 score of a run is computed from the merged embeddings before any decoder is trained, so it does not depend on the decoder. This serves as a sanity check of the pipeline: when the F1 scores of two decoder sweeps are compared, only $2$ of $16$ (dataset, $P$) groups differ significantly for the bilinear-vs-Hadamard comparison, and $1$--$2$ of $16$ for the comparisons with the random forest, with no consistent sign, which is the run-to-run noise expected from independent partitionings.

## Datasets

CITESEER, AstroPh, and AS-Oregon are the same datasets studied in Seminar 3, whose structural statistics are reproduced below; Cit-HepPh is an additional SNAP citation network of high-energy physics phenomenology papers whose structural statistics are not reproduced here.

| Dataset | Nodes | Edges | Average degree | Average clustering coefficient | Density | Modularity |
| --- | --- | --- |--- | --- | --- | --- |
| CITESEER | 3264 | 4536 | 2.78 | 0.145 | 0.0009 | 0.72 |
| AstroPh | 18772 | 198110 | 21.11 | 0.631 | 0.00112 | 0.33 |
| AS-Oregon | 11806 | 38781 | 6.57 | 0.399 | 0.00056 | 0.57 |
Table: Structural statistics for CITESEER, AstroPh, and AS-Oregon, reproduced from Seminar 3.

CITESEER is a citation network of scientific publications. AstroPh is a collaboration network of co-authors of astrophysics papers, and AS-Oregon is a network of autonomous systems and their peering connections; both are derived from SNAP. All four datasets are treated as static, undirected graphs, in contrast to Seminar 3's treatment of the same graphs as event streams.

## Impact of partitioning strategy

Table 3 compares LFM and LPA on CITESEER across partition counts, together with the single-partition baseline. This is a preliminary, single-dataset experiment run with an earlier version of the evaluation protocol, so its absolute numbers are not comparable to the main experiment below; it is reported because it motivates the use of LPA throughout. "LP accuracy" is the accuracy of the edge/non-edge decision at threshold $0.5$ on the held-out pairs.

| Strategy | Partitions | LP Accuracy | RESTORE F1 |
|----------|------------|-------------|------------------|
| Baseline | 1 | 0.9209 ± 0.0105 | 0.3782 ± 0.0110 |
| LFM | 2 | 0.6884 ± 0.0109 | 0.3154 ± 0.0122 |
| LFM | 4 | 0.5576 ± 0.0159 | 0.2693 ± 0.0087 |
| LFM | 8 | 0.5401 ± 0.0173 | 0.2099 ± 0.0065 |
| LPA | 2 | 0.8026 ± 0.0193 | 0.4516 ± 0.0360 |
| LPA | 4 | 0.7485 ± 0.0061 | 0.4816 ± 0.0128 |
| LPA | 8 | 0.7128 ± 0.0200 | 0.5401 ± 0.0107 |
| LPA | 16 | 0.6874 ± 0.0148 | 0.6264 ± 0.0133 |
| LPA | 32 | 0.6765 ± 0.0169 | 0.7017 ± 0.0049 |
| LPA | 64 | 0.6681 ± 0.0114 | 0.6922 ± 0.0047 |
| LPA | 128 | 0.6888 ± 0.0180 | 0.6569 ± 0.0141 |
| LPA | 256 | 0.6791 ± 0.0157 | 0.6293 ± 0.0199 |
Table: CITESEER link-prediction accuracy and RESTORE F1 for LFM and LPA partitioning across partition counts (preliminary experiment).

At every partition count where both strategies were measured (2, 4, 8), LPA gives substantially higher link-prediction accuracy than LFM (0.8026 vs.\ 0.6884 at $P=2$, 0.7128 vs.\ 0.5401 at $P=8$) and also higher RESTORE F1. A plausible reason is that LPA yields disjoint communities that are only grouped, never split, by the bin-packing step, whereas LFM's overlapping communities and randomly assigned uncovered vertices give partitions that cut more of the local neighborhood structure. Since this evidence comes from one dataset, the preference for LPA is a working choice rather than an established general result, and all remaining experiments use LPA.

Table 3 also shows, for LPA, that accuracy falls from 0.9209 at $P=1$ to 0.6681 at $P=64$ while RESTORE F1 rises from 0.3782 to a peak of 0.7017 at $P=32$; the same divergence is examined systematically in the main experiment below.

## Effect of the partition count on link prediction

Tables 4--7 report the ranking metrics of the three decoders for every dataset and partition count. Hits@$k$ is the fraction of held-out positive edges ranked within the top $k$ among $500$ negative candidates, and MRR is the mean reciprocal rank (Methods).

| Dataset | Decoder | P=1 | P=2 | P=4 | P=8 |
|------------------|-------------|----------------|----------------|----------------|----------------|
| CITESEER | Bilinear | 27.61±1.82 | 17.90±1.80 | 13.80±0.71 | 10.94±1.17 |
| CITESEER | Hadamard | 22.92±2.80 | 13.88±1.88 | 12.49±1.37 | 13.87±1.66 |
| CITESEER | RDF | 31.06±2.12 | 28.00±4.07 | 24.24±4.47 | 23.37±3.73 |
| AstroPh | Bilinear | 68.74±1.32 | 51.03±1.55 | 45.19±2.19 | 39.64±2.28 |
| AstroPh | Hadamard | 32.34±4.13 | 26.84±3.22 | 28.24±1.63 | 29.63±2.50 |
| AstroPh | RDF | 65.93±1.48 | 60.33±5.81 | 64.27±1.86 | 64.69±5.55 |
| Cit-HepPh | Bilinear | 62.95±1.13 | 44.85±3.05 | 39.60±2.81 | 36.91±2.02 |
| Cit-HepPh | Hadamard | 37.05±2.79 | 28.38±1.22 | 27.15±1.43 | 32.15±1.98 |
| Cit-HepPh | RDF | 52.49±5.50 | 48.00±3.34 | 50.12±2.97 | 46.33±5.41 |
| AS-Oregon | Bilinear | 25.08±0.95 | 18.69±1.05 | 15.96±1.51 | 15.34±1.63 |
| AS-Oregon | Hadamard | 14.49±1.68 | 10.48±0.78 | 9.12±1.08 | 9.97±0.86 |
| AS-Oregon | RDF | 44.88±5.13 | 43.97±6.14 | 44.92±4.17 | 46.07±3.09 |
Table: Mean reciprocal rank (MRR), in percent, mean ± standard deviation over $10$ runs, LPA partitioning.

| Dataset | Decoder | P=1 | P=2 | P=4 | P=8 |
|------------------|-------------|----------------|----------------|----------------|----------------|
| CITESEER | Bilinear | 17.70±1.14 | 11.08±1.79 | 8.08±0.99 | 5.76±1.10 |
| CITESEER | Hadamard | 14.37±2.65 | 8.21±1.77 | 7.64±1.59 | 9.38±1.50 |
| CITESEER | RDF | 21.88±1.91 | 21.40±3.72 | 19.00±4.50 | 17.90±3.28 |
| AstroPh | Bilinear | 56.79±1.58 | 38.54±1.70 | 32.57±2.29 | 27.68±2.34 |
| AstroPh | Hadamard | 20.73±3.46 | 16.93±2.98 | 18.53±1.79 | 19.56±2.49 |
| AstroPh | RDF | 56.82±2.09 | 52.40±6.92 | 56.90±3.45 | 56.60±6.06 |
| Cit-HepPh | Bilinear | 48.38±1.47 | 31.05±2.81 | 27.10±2.54 | 24.90±1.91 |
| Cit-HepPh | Hadamard | 22.45±2.49 | 17.08±1.24 | 16.08±1.42 | 19.93±1.69 |
| Cit-HepPh | RDF | 38.80±7.07 | 36.10±4.23 | 37.90±3.63 | 34.40±6.82 |
| AS-Oregon | Bilinear | 14.99±1.07 | 9.70±1.23 | 7.83±1.13 | 7.57±1.40 |
| AS-Oregon | Hadamard | 6.99±1.44 | 4.92±0.73 | 4.08±0.88 | 4.49±0.91 |
| AS-Oregon | RDF | 39.70±4.85 | 39.10±6.26 | 40.40±4.43 | 42.00±3.02 |
Table: Hits@1, in percent, mean ± standard deviation over $10$ runs, LPA partitioning.

| Dataset | Decoder | P=1 | P=2 | P=4 | P=8 |
|------------------|-------------|----------------|----------------|----------------|----------------|
| CITESEER | Bilinear | 32.25±2.93 | 19.69±1.80 | 15.54±1.02 | 12.01±1.69 |
| CITESEER | Hadamard | 25.94±3.57 | 15.56±2.43 | 13.75±1.36 | 15.67±1.49 |
| CITESEER | RDF | 35.85±2.98 | 30.60±4.45 | 25.80±5.09 | 24.40±4.48 |
| AstroPh | Bilinear | 77.48±1.50 | 58.30±2.02 | 51.70±2.66 | 44.71±3.24 |
| AstroPh | Hadamard | 36.65±4.97 | 29.82±3.68 | 30.90±1.76 | 32.72±2.94 |
| AstroPh | RDF | 71.10±1.44 | 63.50±5.78 | 68.30±2.31 | 68.60±6.47 |
| Cit-HepPh | Bilinear | 73.21±1.82 | 51.77±4.09 | 45.33±4.16 | 42.48±2.77 |
| Cit-HepPh | Hadamard | 43.07±3.61 | 32.01±1.33 | 30.70±1.90 | 37.74±2.51 |
| Cit-HepPh | RDF | 60.30±4.97 | 52.10±3.96 | 56.70±3.89 | 51.60±4.50 |
| AS-Oregon | Bilinear | 29.22±1.24 | 21.21±1.34 | 17.64±1.63 | 16.60±2.03 |
| AS-Oregon | Hadamard | 15.45±2.20 | 10.43±1.09 | 9.12±1.22 | 10.19±1.17 |
| AS-Oregon | RDF | 45.90±5.47 | 44.20±6.18 | 46.20±4.29 | 46.50±3.37 |
Table: Hits@3, in percent, mean ± standard deviation over $10$ runs, LPA partitioning.

| Dataset | Decoder | P=1 | P=2 | P=4 | P=8 |
|------------------|-------------|----------------|----------------|----------------|----------------|
| CITESEER | Bilinear | 47.37±3.50 | 30.71±2.55 | 24.50±1.34 | 20.73±1.73 |
| CITESEER | Hadamard | 38.96±3.50 | 24.46±2.45 | 21.52±1.72 | 21.39±2.55 |
| CITESEER | RDF | 47.31±2.42 | 39.70±5.91 | 33.80±4.57 | 33.90±6.59 |
| AstroPh | Bilinear | 89.91±0.78 | 74.46±1.37 | 69.70±1.91 | 63.97±2.55 |
| AstroPh | Hadamard | 55.58±5.68 | 46.86±4.04 | 47.57±2.19 | 49.85±2.50 |
| AstroPh | RDF | 82.48±1.81 | 76.00±4.90 | 77.50±1.90 | 80.70±4.55 |
| Cit-HepPh | Bilinear | 89.85±0.56 | 72.40±3.12 | 64.52±2.83 | 60.36±2.20 |
| Cit-HepPh | Hadamard | 67.77±3.84 | 52.46±1.58 | 50.80±1.67 | 57.23±2.52 |
| Cit-HepPh | RDF | 80.00±5.16 | 73.50±2.99 | 74.70±4.79 | 71.40±3.66 |
| AS-Oregon | Bilinear | 45.39±1.44 | 36.97±1.19 | 32.74±3.00 | 31.59±2.28 |
| AS-Oregon | Hadamard | 29.62±2.72 | 21.46±1.93 | 18.50±2.07 | 20.43±1.33 |
| AS-Oregon | RDF | 54.90±6.45 | 54.00±6.88 | 53.10±4.82 | 54.30±5.01 |
Table: Hits@10, in percent, mean ± standard deviation over $10$ runs, LPA partitioning.

The three decoders respond to partitioning very differently, and the pattern is the same for all four ranking metrics.

**Bilinear.** Every metric decreases monotonically with $P$ on every dataset, and the largest single drop is always between $P=1$ and $P=2$: for example, AstroPh MRR falls from 68.74% to 51.03% and Hits@1 from 56.79% to 38.54%, and Cit-HepPh MRR from 62.95% to 44.85%. The decline continues, more slowly, up to $P=8$.

**Hadamard.** The drop from $P=1$ to $P=2$ is also clear on all datasets, but for larger $P$ the curve flattens and on several datasets partially recovers: for example, Cit-HepPh MRR is 28.38% at $P=2$, 27.15% at $P=4$, and 32.15% at $P=8$, and CITESEER MRR is 12.49% at $P=4$ and 13.87% at $P=8$. The Hadamard decoder is also much weaker than the bilinear one even without partitioning (AstroPh MRR at $P=1$: 32.34% vs.\ 68.74%), so part of its apparent robustness to partitioning is simply that it has less quality to lose.

**Random forest.** The metrics are nearly flat in $P$. AS-Oregon MRR is 44.88% at $P=1$ and 46.07% at $P=8$, AstroPh drops only from 65.93% to 64.69%, and the largest decline, on CITESEER, is from 31.06% to 23.37%. The standard deviations of the random forest are larger than those of the neural decoders (see the limitations below).

| Dataset | Bilinear | Hadamard | RDF |
|---|---|---|---|
| CITESEER | -60% | -40% | -25% |
| AstroPh | -42% | -8% | -2% |
| Cit-HepPh | -41% | -13% | -12% |
| AS-Oregon | -39% | -31% | +3% |
Table: Relative change of the mean MRR from $P=1$ to $P=8$, per decoder.

Table 8 summarizes the effect as the relative change of the mean MRR between $P=1$ and $P=8$: the bilinear decoder loses $39$--$60\%$ of its MRR, the Hadamard decoder $8$--$40\%$, and the random forest between $+3\%$ and $-25\%$.

### Significance of the partition-count effect

Table 9 tests the effect of each doubling of $P$ on the MRR, for every decoder, using Cohen's $d$ (negative means MRR decreases with more partitions) and marking with an asterisk the steps that are significant under the Mann--Whitney test.

| Dataset | Step | Bilinear | Hadamard | RDF |
|---|---|---|---|---|
| CITESEER | $P$: 1 $\to$ 2 | -5.35\* | -3.79\* | -0.94\* |
| CITESEER | $P$: 2 $\to$ 4 | -2.98\* | -0.84 | -0.88 |
| CITESEER | $P$: 4 $\to$ 8 | -2.96\* | +0.91\* | -0.21 |
| AstroPh | $P$: 1 $\to$ 2 | -12.27\* | -1.49\* | -1.32\* |
| AstroPh | $P$: 2 $\to$ 4 | -3.07\* | +0.55 | +0.91\* |
| AstroPh | $P$: 4 $\to$ 8 | -2.48\* | +0.66 | +0.10 |
| Cit-HepPh | $P$: 1 $\to$ 2 | -7.86\* | -4.03\* | -0.99 |
| Cit-HepPh | $P$: 2 $\to$ 4 | -1.79\* | -0.93 | +0.67 |
| Cit-HepPh | $P$: 4 $\to$ 8 | -1.10\* | +2.90\* | -0.87 |
| AS-Oregon | $P$: 1 $\to$ 2 | -6.37\* | -3.06\* | -0.16 |
| AS-Oregon | $P$: 2 $\to$ 4 | -2.10\* | -1.43\* | +0.18 |
| AS-Oregon | $P$: 4 $\to$ 8 | -0.39 | +0.86 | +0.31 |
Table: Cohen's $d$ of the MRR between adjacent partition counts ($n = 10$ runs per group); \* marks Mann--Whitney $p < 0.05$.

For the bilinear decoder, $11$ of the $12$ steps are significant and all effects are negative, with very large effect sizes for the first step ($|d|$ between $5.4$ and $12.3$); the only non-significant step is AS-Oregon $P=4 \to 8$ ($p = 0.43$), where the MRR has essentially plateaued. For the Hadamard decoder, $7$ of $12$ steps are significant; the $P=1 \to 2$ step is significantly negative on all datasets, but the later steps are mostly not significant, and two of them are significantly *positive* (CITESEER and Cit-HepPh, $P=4 \to 8$), i.e.\ the MRR recovers. For the random forest, only $3$ of $12$ steps are significant, the effects are mostly below $1$ in magnitude, and their sign changes between steps; on AS-Oregon no step is significant. Partitioning therefore has a large, monotone, and highly significant effect on the bilinear decoder, a mostly one-time effect on the Hadamard decoder, and at most a small, non-monotone effect on the random forest.

## Comparison of decoders

Table 10 compares the decoders directly on the MRR in every (dataset, $P$) group, again with Cohen's $d$ and Mann--Whitney significance (asterisk); a positive value means the first-named decoder scores higher.

| Dataset | $P$ | Bilinear $-$ Hadamard | RDF $-$ Bilinear | RDF $-$ Hadamard |
|---|---|---|---|---|
| CITESEER | 1 | +1.99\* | +1.74\* | +3.28\* |
| CITESEER | 2 | +2.18\* | +3.21\* | +4.45\* |
| CITESEER | 4 | +1.20 | +3.26\* | +3.55\* |
| CITESEER | 8 | -2.05\* | +4.50\* | +3.29\* |
| AstroPh | 1 | +11.87\* | -2.00\* | +10.83\* |
| AstroPh | 2 | +9.57\* | +2.19\* | +7.13\* |
| AstroPh | 4 | +8.76\* | +9.37\* | +20.57\* |
| AstroPh | 8 | +4.18\* | +5.90\* | +8.14\* |
| Cit-HepPh | 1 | +12.17\* | -2.63\* | +3.54\* |
| Cit-HepPh | 2 | +7.08\* | +0.99 | +7.81\* |
| Cit-HepPh | 4 | +5.59\* | +3.64\* | +9.86\* |
| Cit-HepPh | 8 | +2.38\* | +2.31\* | +3.49\* |
| AS-Oregon | 1 | +7.77\* | +5.37\* | +7.97\* |
| AS-Oregon | 2 | +8.86\* | +5.74\* | +7.66\* |
| AS-Oregon | 4 | +5.21\* | +9.23\* | +11.75\* |
| AS-Oregon | 8 | +4.12\* | +12.44\* | +15.93\* |
Table: Cohen's $d$ of the MRR between decoders in each (dataset, $P$) group ($n = 10$); \* marks Mann--Whitney $p < 0.05$; positive means the first-named decoder is higher.

The decoders are ordered **random forest $>$ bilinear $>$ Hadamard** in most configurations. The bilinear decoder is significantly better than the Hadamard decoder in $14$ of $16$ groups, usually with very large effect sizes, the exceptions being CITESEER $P=4$ (a tie) and CITESEER $P=8$, where the Hadamard decoder is significantly better (the same holds for Hits@1 and Hits@3). The random forest is significantly better than the Hadamard decoder in all $16$ groups on all four ranking metrics ($d$ from $+2.5$ to $+20.6$) and significantly better than the bilinear decoder on MRR in $13$ of $16$ groups. Its advantage over the bilinear decoder grows with the partition count, because the bilinear decoder degrades while the random forest stays roughly flat. The exceptions are the unpartitioned dense graphs: at $P=1$ on AstroPh and Cit-HepPh the bilinear decoder is significantly better than the random forest (MRR 68.74% vs.\ 65.93% and 62.95% vs.\ 52.49%), and Cit-HepPh at $P=2$ is a tie.

## Reconstruction quality versus link prediction

| Dataset | P=1 | P=2 | P=4 | P=8 |
|------------------|----------------|----------------|----------------|----------------|
| CITESEER | 35.88±1.46 | 43.92±3.25 | 49.72±2.34 | 55.68±6.12 |
| AstroPh | 70.62±0.08 | 67.76±0.09 | 67.84±0.12 | 67.75±0.11 |
| Cit-HepPh | 54.87±0.06 | 57.41±0.79 | 60.35±2.12 | 61.59±0.87 |
| AS-Oregon | 33.88±0.28 | 38.88±1.91 | 45.97±3.79 | 43.20±1.78 |
Table: RESTORE F1 of the reconstruction from the merged embeddings, in percent, mean ± standard deviation over $10$ runs (values of the bilinear sweep; the other two sweeps agree within run-to-run noise since the score does not depend on the decoder).

Seminar 3 evaluated partitioned embeddings by graph reconstruction. Table 11 shows the RESTORE F1 of the same embeddings used in this experiment: it *increases* with $P$ on CITESEER (35.88% $\to$ 55.68%), Cit-HepPh (54.87% $\to$ 61.59%), and AS-Oregon (33.88% $\to$ 43.20%), and is essentially flat on AstroPh apart from a small drop of about three points between $P=1$ and $P=2$. Compared with Tables 4--7, the reconstruction score and the neural-decoder link-prediction quality therefore move in *opposite directions* as the partition count grows: on CITESEER, for instance, F1 rises by more than $19$ points while the bilinear MRR falls by $60\%$. Partitioning makes each partition's local reconstruction task easier and more self-contained, which the reconstruction score rewards, whereas link prediction on held-out edges additionally requires comparing embeddings of vertices that may lie in different partitions and were trained independently.

The pooled Spearman correlation between the per-run F1 and MRR is positive and significant for all three decoders (bilinear $\rho = 0.62$, Hadamard $\rho = 0.65$, RDF $\rho = 0.67$, $n = 160$ each, $p < 10^{-17}$), but this does not contradict the opposite trends above: it is dominated by differences between datasets (graphs that are easy to embed score high on both), while, within a dataset, the partition count pushes the two quantities apart. Reconstruction F1 is therefore a reasonable indicator of how learnable a graph is, but not a reliable proxy for how well partitioned embeddings support link prediction.

## Discussion

The main results are consistent with a single explanation. Because every partition is embedded independently, the embedding spaces of different partitions are unrelated, so a pair of vertices from different partitions is scored from vectors that are not comparable. The neural decoders learn one smooth scoring function ($w^\top(u \odot v)$ or $u^\top W v$) that presupposes a shared latent geometry, and the first partition split, which introduces the first cross-partition pairs, is where they lose most (Tables 4--7, Table 9). The random forest builds axis-aligned splits directly on the raw coordinates of $[u; v]$ and does not assume such a geometry, and it is the decoder least affected by partitioning. We stress that this explanation is consistent with the evidence but was not tested directly: the experiments do not separate within-partition test pairs from cross-partition ones.

The practical recommendations are as follows. When the graph must be partitioned, use LPA and a random forest on the concatenated embeddings, which gives the best ranking quality in nearly all configurations with $P \ge 2$ and hardly degrades with the number of partitions. When the graph fits in a single partition and is dense (AstroPh, Cit-HepPh), the bilinear decoder is the best choice. The Hadamard decoder is dominated by the other two in almost every configuration and is not recommended.

**Limitations.** (i) The random forest ranking metrics take values on a grid of $0.01$, whereas those of the neural decoders lie on a grid of $0.001$, which suggests that the random-forest runs were evaluated on a smaller sample of positive edges (about $100$ instead of $1000$); this explains part of their larger standard deviations, and the gaps between the random forest and the neural decoders may change when the sweeps are repeated with an equal sample size. (ii) With $20$ trees the forest's probabilities take only $21$ distinct values, and ties in the ranking are resolved in favor of the true edge, which can favor the random forest relative to the neural decoders. (iii) The neural decoders were not tuned separately for each partition count. (iv) The comparison of LFM and LPA is limited to one dataset and an earlier evaluation protocol, and all main experiments use LPA and four datasets. (v) The significance tests are not corrected for multiple comparisons.

# Conclusion

This paper studied how graph partitioning and the choice of link-prediction decoder affect link prediction over node embeddings that are computed independently on each partition with static node2vec. On four datasets, four partition counts, and $10$ runs per configuration, we compared a bilinear decoder, a Hadamard decoder, and a random forest on concatenated embeddings using Hits@$k$ and MRR, with Mann--Whitney tests and Cohen's $d$.

Partitioning strongly degrades the neural decoders: from $P=1$ to $P=8$ the MRR of the bilinear decoder decreases by $39$--$60\%$ (significantly at $11$ of $12$ steps) and that of the Hadamard decoder by $8$--$40\%$, mostly in the first step. The random forest is nearly insensitive: its MRR changes between $+3\%$ and $-25\%$, with only $3$ of $12$ steps significant. Consequently, the decoders are ordered random forest $>$ bilinear $>$ Hadamard in most configurations, with the bilinear decoder best only on unpartitioned dense graphs. In a preliminary single-dataset comparison, LPA gave clearly better link prediction than LFM at every partition count tested. Finally, the reconstruction F1 of the same embeddings *increases* with the partition count on three of the four datasets while neural-decoder link-prediction quality decreases, so reconstruction quality is not a reliable proxy for downstream link-prediction quality in a partitioned embedding pipeline, a distinction that Seminar 3's reconstruction-only study could not surface.

Future work should (a) repeat the random-forest sweep with the same number of ranked positives as the neural decoders and with random tie-breaking, (b) split the test pairs into within-partition and cross-partition pairs to test directly whether cross-partition pairs cause the degradation, (c) tune the neural decoders per partition count and consider aligning the per-partition embedding spaces (for instance by a learned linear map) before decoding, (d) extend the LFM/LPA comparison and the decoder study to the remaining Seminar 3 datasets (DBLP, Enron), and (e) study the replication factor, which lets a vertex be embedded in several partitions, as a means to recover link-prediction quality.

# References
