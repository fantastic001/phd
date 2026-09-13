---
title: "Partitioning Strategy, Decoder Architecture, and Classifier Choice for Link Prediction over Distributed Static Node2Vec Embeddings"
author: "Stefan Nožinić"
abstract: |
  This paper studies how the choice of graph partitioning strategy, link-prediction decoder architecture, and classifier family affects downstream link-prediction quality when node embeddings are computed independently, per partition, with static node2vec. Unlike graph reconstruction, which asks how well embeddings recover the training graph itself, link prediction asks how well embeddings generalize to edges withheld from training, and this paper shows that the two questions can have different, sometimes opposite, answers under partitioning. Using four real-world graphs (CITESEER, AstroPh, Cit-HepPh, AS-Oregon), we compare label propagation (LPA) against the Lancichinetti-Fortunato-Kertesz method (LFM) as partitioning strategies, three link-prediction decoder architectures (a Hadamard-product logistic model, a bilinear $x^\top W y$ model, and a Mahalanobis-style $(x-y)^\top W (x-y)$ model), and two ways of turning per-partition embeddings into link-prediction features (concatenation and averaging, consumed either by a neural decoder or by a random forest classifier). We find that LPA consistently outperforms LFM for downstream link-prediction accuracy at every partition count tested, even though this paper's companion study of graph reconstruction (Seminar 3) found the opposite ranking is not what determines reconstruction quality; further, link-prediction accuracy under LPA *decreases* monotonically as the partition count grows on CITESEER even as the graph-reconstruction F1 score of the same embeddings *increases* with partition count, showing that reconstruction F1 is not a reliable proxy for downstream link-prediction performance. Neural decoders lose accuracy as the partition count increases because independently trained per-partition embeddings occupy unaligned latent spaces; a random forest classifier trained on the concatenation of per-partition embeddings is markedly more robust to this misalignment and is a strong, simple baseline across all four datasets and partition counts tested. Among the neural decoders, the bilinear $x^\top W y$ architecture gives the best mean reciprocal rank.
bibliography: ./refs.bib
---

# Introduction

A graph is a mathematical structure consisting of vertices (or nodes) connected by edges. Graphs are widely used to model relationships and interactions in various domains [@van_der_hofstad_random_2024], such as social networks [@leskovec_signed_2010] [@backstrom_group_2006] [@rozemberczki_twitch_2021], collaboration networks [@savic_analysis_2017], terrorist networks [@krebs_mapping_2002] and blog citation networks [@adamic_political_2005]. Graph vertex embeddings represent vertices as low-dimensional vectors, enabling machine learning tasks such as vertex classification, community detection, and link prediction [@leskovec_predicting_2010]. Among these downstream tasks, link prediction is of particular practical importance: it underlies friend and content recommendation, knowledge-graph completion [@bordes_translating_2013], and the inference of missing or future edges in evolving networks.

State-of-the-art embedding methods such as node2vec [@grover_node2vec_2016] compute embeddings from a single, monolithic training pass over the whole graph. When the graph is too large to embed on one machine, a common strategy is to partition the graph, embed each partition independently, and combine the resulting per-partition embeddings for downstream use [@fang_distributed_2023] [@lombardo_scalable_2019]. This paper's companion study (Seminar 3) examined how such partitioning affects the *reconstruction* quality of the embeddings — how well the embeddings recover the edges of the graph they were trained on. Link prediction poses a different and arguably more practically relevant question: how well do the embeddings generalize to edges that were *not* observed during training? Because independently embedded partitions have no shared frame of reference — each partition's embedding space is the arbitrary output of a separate skip-gram training run [@church_word2vec_2017] — a decoder or classifier that combines embeddings from different partitions must cope with per-partition embedding spaces that are not directly comparable to one another. This raises questions that do not arise in the single-partition setting: which partitioning strategy best preserves the community structure that link prediction relies on, which decoder architecture best tolerates misaligned per-partition latent spaces, and whether a downstream classifier can be made robust to this misalignment at all.

# Problem Formulation

Given a static graph partitioned into $P$ partitions, each embedded independently with static node2vec, we want to know how the choice of

* partitioning strategy (LFM vs.\ LPA),
* link-prediction decoder architecture, and
* classifier family and embedding-combination scheme

affects link-prediction quality, and how that quality relates to the graph-reconstruction quality of the same embeddings.

# Related Work

Graph vertex embedding is a well-studied area, with various methods proposed to generate low-dimensional representations of vertices in a graph. Node2Vec [@grover_node2vec_2016] is the state-of-the-art method used throughout this paper; it generates embeddings via biased random walks and has been shown to be effective at capturing community structure. As its improvement, DistGER [@fang_distributed_2023] is a distributed graph embedding method that extends Node2Vec by leveraging distributed computing to handle large graphs, and [@lombardo_scalable_2019] proposes an actor-model-based distributed framework for the same purpose. The common ground for these methods is that they generate walks which are later used to train a Word2Vec model [@church_word2vec_2017] commonly used for generating embeddings in natural language processing tasks.

Link prediction from vertex embeddings is commonly formulated as a binary classification problem over pairs of vertices, using a decoder that maps a pair of embeddings to a link probability. The original node2vec paper [@grover_node2vec_2016] popularized the use of simple binary operators — including the Hadamard product used as the primary decoder in this paper — to combine two node embeddings into a single edge feature vector before classification. Leskovec et al. [@leskovec_predicting_2010] study link sign and formation prediction more broadly, motivating the classification-based framing of link prediction adopted here.

Partitioning a graph while preserving the community structure it relies on is itself a well-studied problem. Community-detection-based partitioners are the natural choice when partitioning is intended to preserve locality for downstream embedding: the label propagation algorithm (LPA) [@raghavan_near_2007] assigns each vertex the most frequent label among its neighbors, iterating until labels stabilize into communities, while the Lancichinetti-Fortunato-Kertesz (LFM) method [@lancichinetti_detecting_2009] detects (possibly overlapping) communities by greedily optimizing a local fitness function around a seed vertex. Both methods have been used as static graph partitioners in this line of work, and this paper compares their impact on downstream link-prediction quality, complementing Seminar 3's neighbor-based buffered partitioner for online node arrivals, and the classical static partitioning literature surveyed there [@benlic_effective_2010] [@sanders_distributed_2012] [@sanders_engineering_2011] [@romero_ruiz_memetic_2018] [@catalyurek_more_2023].

Distributed embedding systems that partition a graph before embedding face a fundamental representation problem once embeddings must be combined across partitions: because each partition is embedded independently, there is no guarantee that geometrically similar vectors in two different partitions' embedding spaces represent structurally similar vertices. This paper studies the practical consequences of that misalignment for link prediction, complementing Seminar 3's study of its consequences for graph reconstruction.

# Contributions

This paper makes the following contributions:

* We compare LFM and LPA as static partitioning strategies for distributed link prediction, and show that LPA is consistently the better choice across the partition counts tested.
* We compare three link-prediction decoder architectures and two classifier families (neural decoders vs.\ random forest) on embeddings produced by independently embedding each partition with static node2vec, and identify random forest on concatenated embeddings as a partition-count-robust baseline.
* We show, using the same underlying embeddings and datasets studied for graph reconstruction, that reconstruction F1 and downstream link-prediction accuracy can move in opposite directions as the partition count grows, indicating that reconstruction quality is not a reliable proxy for link-prediction quality.

# Paper Organization

The rest of the paper is organized as follows: first, the system overview is presented, then graph partitioning, the embedding model, and link-prediction decoders and classifiers are described. Next, benchmarks and evaluation metrics are explained in detail. Finally, results and discussion are presented, followed by the conclusion and references.

# Methods

## System overview

Unlike Seminar 3's buffered, online setting, this paper considers a fully known static graph $G = (V, E)$. The graph is partitioned once, ahead of time, into $P$ partitions $\mathcal{P} = \{p_1, \dots, p_P\}$ using a community-detection-based partitioner (LFM or LPA, described below). Each partition's induced subgraph is then embedded independently using static node2vec, producing one embedding matrix per partition. Because each partition is embedded in complete isolation from the others, the resulting per-partition embedding spaces are not trained to be mutually consistent: a large dot product between two vectors in partition $p_i$'s embedding space carries no guaranteed relationship to a large dot product in partition $p_j$'s embedding space. This is the central complication that distinguishes distributed link prediction from the single-partition case, and every design choice examined in this paper — partitioning strategy, decoder architecture, and combination scheme — is, in one way or another, a response to it.

Link-prediction candidate pairs $(u, v)$ are formed after partitioning, and may span two different partitions. For a pair whose vertices lie in the same partition, both embeddings come from the same, internally consistent embedding space. For a pair whose vertices lie in different partitions, the decoder or classifier must combine embeddings from two independently trained spaces, and it is this cross-partition case that drives most of the quality loss observed in the Results section.

## Graph partitioning

Two static, community-detection-based partitioning strategies are compared:

**LPA (Label Propagation Algorithm)** [@raghavan_near_2007]: each vertex is initialized with a unique label, and, at every iteration, updates its label to the one most frequent among its neighbors (ties broken at random). The process is repeated until labels stabilize; vertices sharing a final label form one partition. LPA is near-linear in the number of edges and requires no parameter tuning beyond a maximum iteration count.

**LFM (Lancichinetti-Fortunato-Kertesz method)** [@lancichinetti_detecting_2009]: starting from a randomly chosen seed vertex, a community is grown greedily by repeatedly adding the neighboring vertex that most increases a local fitness function, until no further addition improves fitness. The process is repeated with new seeds until every vertex is assigned to a community. Unlike LPA, LFM can produce overlapping communities; for the partitioning use considered here, each vertex is assigned to a single partition (its highest-fitness community) to produce a disjoint partitioning comparable to LPA's output.

Both methods partition the graph based on its community structure rather than on vertex arrival order, in contrast to the online, neighbor-count-based partitioner used in Seminar 3 for streaming vertex arrivals. Because the full graph is known in advance, both methods can consider the entire vertex neighborhood when forming partitions, which is not possible in the online setting.

## Embedding model

Given a partitioned graph, each partition's induced subgraph is embedded independently using static node2vec [@grover_node2vec_2016]. Node2vec performs biased second-order random walks controlled by return parameter $p$ and in-out parameter $q$, and trains a skip-gram model [@church_word2vec_2017] over the resulting corpus of walks to produce an embedding matrix for that partition's vertices. No information is shared between partitions during training: the skip-gram model for partition $p_i$ is initialized and optimized independently of every other partition's model, so the resulting embedding spaces are not aligned with one another beyond whatever structural similarity the underlying communities happen to share.

## Link-prediction decoders and classifiers

Given the embeddings $u = z(a)$ and $v = z(b)$ of two vertices $a, b$ — possibly drawn from different partitions — a decoder or classifier produces a probability, or ranking score, that $(a, b)$ is an edge of $G$. Three neural decoder architectures are compared:

**Hadamard-product logistic decoder.** Following the edge-feature operators introduced alongside node2vec [@grover_node2vec_2016], the two embeddings are combined via elementwise (Hadamard) product and passed through a linear layer and sigmoid:

$$ p(\text{connected} \mid u, v) = \sigma\big(w^\top (u \odot v) + b\big) $$

where $\odot$ denotes the Hadamard product, $w$ is a learned weight vector, $b$ a learned bias, and $\sigma$ the sigmoid function. This is the simplest decoder considered and serves as the baseline architecture.

**Bilinear decoder ($x^\top W y$).** A learned matrix $W$ captures pairwise interactions between the coordinates of the two embeddings directly:

$$ p(\text{connected} \mid u, v) = \sigma\big(u^\top W v\big) $$

A symmetric-$W$ variant is also considered, in which $W$ is constrained to be symmetric by parameterizing only its upper triangle and mirroring it; this halves the number of free parameters relative to an unconstrained $W$.

**Mahalanobis-style decoder ($(x-y)^\top W (x-y)$).** Rather than the product of the two embeddings, the decoder operates on their difference:

$$ p(\text{connected} \mid u, v) = \sigma\big((u - v)^\top W (u - v)\big) $$

All three decoders are trained with binary cross-entropy loss against labeled positive (true edge) and negative (non-edge) pairs.

In addition to these neural decoders, a random forest classifier is trained directly on hand-combined embedding features, using one of five preprocessing schemes to turn a pair $(u, v)$ into a single feature vector: **Concatenate** ($[u; v]$), **Average** ($(u+v)/2$), **Hadamard** ($u \odot v$), **L1** ($|u - v|$), and **L2** ($(u-v)^2$). Logistic regression and a support vector classifier (SVC) are evaluated over the same five preprocessing schemes as additional, simpler baselines.

## Benchmarks and evaluation metrics

Link prediction is evaluated using the standard held-out-edge protocol: 20% of the graph's edges are removed uniformly at random and held out as positive test pairs; the embeddings are computed on the remaining 80% of the graph, partitioned and embedded as described above; and an equal number of non-edges are sampled as negative test pairs. The decoder or classifier is trained on the training-graph's edges (and an equal number of sampled non-edges) and evaluated on the held-out test pairs using the following metrics:

**Accuracy and F1**: standard binary classification accuracy and F1 score of the edge/non-edge prediction, computed at a fixed decision threshold (0.5 for the sigmoid-based decoders).

**Hits@k**: for each held-out positive pair $(a, b)$, the true endpoint $b$ is ranked against a set of candidate negative endpoints by decoder score; Hits@k is the fraction of held-out positive pairs for which $b$ ranks among the top $k$ candidates. This paper reports Hits@5 and Hits@10.

**Mean Reciprocal Rank (MRR)**: the mean, over held-out positive pairs, of the reciprocal of the rank of the true endpoint $b$ among the same candidate set used for Hits@k:

$$ \text{MRR} = \frac{1}{|E_{\text{test}}|} \sum_{(a,b) \in E_{\text{test}}} \frac{1}{\text{rank}(b \mid a)} $$

**RESTORE F1 (graph reconstruction)**: as in Seminar 3 [@yip_restore_2023], the reconstruction F1 score measures how well the same embeddings, used to reconstruct the *training* graph rather than to predict held-out edges, recover the training graph's edge set. This paper reports RESTORE F1 alongside the link-prediction metrics above specifically to contrast reconstruction quality against generalization quality on the same set of embeddings.

# Results and discussion

## Datasets

Experiments are performed on four real-world graph datasets. CITESEER, AstroPh, and AS-Oregon are the same datasets studied in Seminar 3, whose dataset statistics are reproduced below for reference; Cit-HepPh is an additional SNAP citation network of high-energy physics phenomenology papers, for which the vertex_voyage project notes report link-prediction results but not the full set of structural statistics computed for the other three datasets, so it is described qualitatively only.

| Dataset | Nodes | Edges | Average degree | Average clustering coefficient | Density | Modularity |
| --- | --- | --- |--- | --- | --- | --- |
| CITESEER | 3264 | 4536 | 2.78 | 0.145 | 0.0009 | 0.72 |
| AstroPh | 18772 | 198110 | 21.11 | 0.631 | 0.00112 | 0.33 |
| AS-Oregon | 11806 | 38781 | 6.57 | 0.399 | 0.00056 | 0.57 |
Table: Structural statistics for CITESEER, AstroPh, and AS-Oregon, reproduced from Seminar 3.

CITESEER is a citation network of scientific publications. AstroPh is a collaboration network of co-authors of astrophysics papers, and AS-Oregon is a network of autonomous systems and their peering connections; both are derived from SNAP. Cit-HepPh is a SNAP citation network among high-energy physics phenomenology papers. All four datasets are treated as static, undirected graphs for the experiments in this paper, in contrast to Seminar 3's treatment of the same graphs as event streams.

<!-- SUGGESTION: DBLP and Enron are conspicuously absent here despite being two of Seminar3's five datasets. If link-prediction runs can be produced for them (even a single decoder/partition-count sweep), adding them would let the paper's central claim (F1-vs-accuracy divergence) be checked against datasets already characterized structurally in Seminar3, rather than resting on CITESEER alone. Cit-HepTh also appears later (Table 6) but is missing from this dataset list and from Table 3/4 — worth deciding whether it's in scope or should be dropped from Table 6 for consistency. -->

## Impact of partitioning strategy on link-prediction performance

Table 2 compares the LFM and LPA partitioning strategies on CITESEER, using the Hadamard-product logistic decoder, across a wide range of partition counts, alongside the RESTORE F1 score of the same embeddings and the single-partition baseline.

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
Table: CITESEER link-prediction accuracy and RESTORE F1 for LFM and LPA partitioning strategies across partition counts.

At every partition count where both strategies are measured (2, 4, 8), LPA gives substantially higher link-prediction accuracy than LFM — for example, 0.8026 vs.\ 0.6884 at $P=2$, and 0.7128 vs.\ 0.5401 at $P=8$ — and higher RESTORE F1 as well. This is consistent with LPA producing partitions that better preserve the local neighborhood structure that both reconstruction and link prediction rely on. On this basis, all remaining experiments in this paper use LPA as the partitioning strategy.

The same table also reveals the central empirical finding of this paper: as the partition count increases under LPA, **link-prediction accuracy decreases monotonically** from 0.9209 at $P=1$ to 0.6681 at $P=64$ (with a slight, likely noise-level, uptick at $P=128$), while **RESTORE F1 increases** over most of the same range, from 0.3782 at $P=1$ to a peak of 0.7017 at $P=32$, before declining slightly at the largest partition counts. The two metrics are computed from the same embeddings and move in opposite directions over most of the partition-count range: the embeddings become progressively better at reconstructing the training graph's own edges as $P$ grows, while becoming progressively worse at predicting held-out edges. This indicates that RESTORE F1, while a reasonable measure of how much of the training graph's structure is captured within each partition's local embedding space, is not a reliable proxy for the embeddings' ability to generalize to unseen edges, particularly ones that may cross partition boundaries.

<!-- SUGGESTION: This is the paper's headline claim, and it currently rests on one dataset (CITESEER) with no significance testing on the accuracy side (the ± values shown are per-config std, but no pairwise test like Seminar3's Welch/Mann-Whitney/Cohen's d section has been run to confirm the accuracy decline and F1 rise are each individually significant, let alone that the *divergence* between them is). Given Seminar3 devoted a full section to exactly this kind of test, reviewers will likely expect the same treatment here before accepting "opposite directions" as more than an eyeballed trend. LFM vs LPA is also only compared on CITESEER — repeating even one other dataset would show whether the divergence is CITESEER-specific or general. -->

## Effect of partition count on link-prediction quality across datasets

Table 3 reports link-prediction accuracy, Hits@5, Hits@10, MRR, and RESTORE F1 for CITESEER, AstroPh, Cit-HepPh, and AS-Oregon under LPA partitioning, using the Hadamard-product logistic decoder, for $P \in \{1, 2, 4, 8\}$.

| Dataset | P | Accuracy | Hits@5 | Hits@10 | MRR | RESTORE F1 |
|---|---|---|---|---|---|---|
| CITESEER | 1 | 0.7649 | 0.4040 | 0.4768 | 0.2899 | 0.3772 |
| CITESEER | 2 | 0.7020 | 0.2737 | 0.3135 | 0.1848 | 0.4516 |
| CITESEER | 4 | 0.6468 | 0.1876 | 0.2318 | 0.1487 | 0.4816 |
| CITESEER | 8 | 0.6137 | 0.1589 | 0.1921 | 0.1043 | 0.5401 |
| AstroPh | 1 | 0.9597 | 0.8460 | 0.9130 | 0.6913 | 0.7041 |
| AstroPh | 2 | 0.9188 | 0.7080 | 0.7850 | 0.5390 | 0.6693 |
| AstroPh | 4 | 0.8919 | 0.5950 | 0.7010 | 0.4396 | 0.5157 |
| AstroPh | 8 | 0.8601 | 0.5280 | 0.6310 | 0.3782 | 0.6624 |
| Cit-HepPh | 1 | 0.9550 | 0.8180 | 0.9020 | 0.6272 | -- |
| Cit-HepPh | 2 | 0.9119 | 0.5820 | 0.7070 | 0.4363 | -- |
| Cit-HepPh | 4 | 0.8753 | 0.5040 | 0.6360 | 0.3754 | -- |
| Cit-HepPh | 8 | 0.8519 | 0.5350 | 0.6320 | 0.3909 | -- |
| AS-Oregon | 1 | 0.7726 | 0.3580 | 0.4620 | 0.2568 | 0.3437 |
| AS-Oregon | 2 | 0.7642 | 0.2820 | 0.3860 | 0.1808 | 0.3782 |
| AS-Oregon | 4 | 0.7468 | 0.2210 | 0.3130 | 0.1500 | 0.3916 |
| AS-Oregon | 8 | 0.7348 | 0.2280 | 0.3020 | 0.1550 | 0.2935 |
Table: Link-prediction accuracy, Hits@5, Hits@10, MRR, and RESTORE F1 across partition counts, four datasets, LPA partitioning, Hadamard-product logistic decoder.

Accuracy, Hits@5, Hits@10, and MRR all decrease monotonically with $P$ on every one of the four datasets, including AstroPh and Cit-HepPh, which are considerably denser than CITESEER and AS-Oregon. This is a materially different picture from Seminar 3's reconstruction-F1 results, where partitioning was found to be stable or even beneficial on several datasets (CITESEER, DBLP, Enron); here, under the link-prediction metrics, every dataset degrades with partition count, and the RESTORE F1 column shows no consistent relationship to that degradation — it rises with $P$ on CITESEER, is non-monotonic on AstroPh and AS-Oregon, and is not measured on Cit-HepPh in these notes. The drop is steepest between $P=1$ and $P=2$ for Hits@5 and Hits@10 on every dataset (e.g.\ AstroPh Hits@10 falls from 0.9130 to 0.7850, and Cit-HepPh from 0.9020 to 0.7070), consistent with the introduction of *any* cross-partition candidate pairs being the dominant source of quality loss, rather than the degree of fragmentation increasing gradually with further splits.

<!-- SUGGESTION: Table 3 reports no error bars / run counts, unlike Table 2, so it's unclear whether these are single runs or averages — worth clarifying and, ideally, repeating a few configs to get variance estimates. Also, the Methods section's held-out-edge protocol never specifies how the negative candidate set for Hits@k/MRR is constructed (how many candidates, sampled how — uniformly at random, degree-matched, restricted to non-neighbors?). This materially affects the absolute Hits@k/MRR numbers and should be pinned down for reproducibility, even if the qualitative "monotonic decline with P" conclusion is robust to it. -->

## Decoder architecture comparison

Table 4 compares the bilinear $x^\top W y$ decoder, its symmetric-$W$ variant, and the $(x-y)^\top W(x-y)$ decoder on CITESEER (and, where available, AstroPh) across partition counts.

| Architecture | Dataset | P | LP Accuracy | Train Loss | Val Loss |
|--------------|---------|---|-------------|------------|----------|
| $x^\top W y$ | CITESEER | 1 | 0.9382 ± 0.0079 | 0.10 | 0.19 |
| $x^\top W y$ | CITESEER | 2 | 0.8582 ± 0.0162 | 0.13 | 0.42 |
| $x^\top W y$ | CITESEER | 4 | 0.8183 ± 0.0140 | 0.158 | 0.606 |
| $x^\top W y$ | CITESEER | 8 | 0.8223 ± 0.0087 | 0.15 | 0.66 |
| $x^\top W y$ | CITESEER | 16 | 0.7822 ± 0.0181 | 0.19 | 0.79 |
| $x^\top W y$ | AstroPh | 1 | 0.9889 | -- | -- |
| $x^\top W y$ | AstroPh | 2 | 0.9460 | -- | -- |
| $x^\top W y$ | AstroPh | 4 | 0.8988 | -- | -- |
| $x^\top W y$ | AstroPh | 8 | 0.8898 | -- | -- |
| $x^\top W y$ | AstroPh | 16 | 0.8583 | -- | -- |
| $x^\top W y$ (symmetric $W$) | CITESEER | 1 | 0.9196 ± 0.0148 | 0.185 | 0.358 |
| $x^\top W y$ (symmetric $W$) | CITESEER | 2 | 0.8493 ± 0.0133 | 0.253 | 0.581 |
| $(x-y)^\top W(x-y)$ | CITESEER | 1 | 0.8244 ± 0.0145 | 0.29 | 0.45 |
| $(x-y)^\top W(x-y)$ | CITESEER | 2 | 0.6635 ± 0.0194 | 0.37 | 0.68 |
Table: Link-prediction accuracy and training/validation loss for the bilinear decoder (unconstrained and symmetric-$W$) and the Mahalanobis-style decoder, LPA partitioning.

All architectures degrade with $P$, and validation loss climbs faster than training loss in every case (e.g.\ $x^\top W y$ on CITESEER: training loss rises from 0.10 to 0.19 between $P=1$ and $P=16$, while validation loss rises from 0.19 to 0.79), which is the signature of the model overfitting to whichever partition-specific quirks are present in the training pairs rather than learning a decoder that generalizes across the misaligned per-partition embedding spaces. Comparing architectures directly, the $(x-y)^\top W(x-y)$ decoder is clearly weaker than the bilinear decoder at both $P=1$ (0.8244 vs.\ 0.9382) and $P=2$ (0.6635 vs.\ 0.8582) — using the *difference* between two embeddings as the decoding signal discards the magnitude and direction information that the bilinear form retains, and this cost is larger, not smaller, once cross-partition pairs are introduced. Constraining $W$ to be symmetric costs a small but consistent amount of accuracy relative to the unconstrained bilinear decoder, at both $P=1$ (0.9196 vs.\ 0.9382) and $P=2$ (0.8493 vs.\ 0.8582), suggesting that the extra degrees of freedom of an unconstrained $W$ are put to some, if modest, use on this task.

A separate, smaller-scale run of the simplest decoder — the Hadamard-product logistic model used as the default architecture in Tables 2 and 3 above — on CITESEER without LPA partitioning gives a consistent picture using raw confusion-matrix counts: F1 falls from 0.9355 (accuracy 0.9338) at $P=1$, to 0.6104 (accuracy 0.5364) at $P=2$, to 0.5493 (accuracy 0.5000) at $P=4$. This run uses a different partitioner and a single trial per configuration, so it is not directly comparable in absolute terms to Table 4, but it shows the same qualitative pattern: decoder quality collapses fastest between one and two partitions, and continues to erode as $P$ grows further. Among all decoders considered in this paper, the bilinear $x^\top W y$ decoder gives the best mean reciprocal rank in the ranking-based evaluation of the previous section, indicating that it preserves a more useful relative ordering among ranked candidates even where its top-1 accuracy is comparable to the simpler Hadamard decoder.

<!-- SUGGESTION: The paper is explicit (appropriately so) that Table 4's rows and the Hadamard confusion-matrix numbers come from different runs/protocols and aren't directly comparable in absolute terms. That honesty is good, but it also means the paper never actually runs all three decoders under one matched protocol — which would be needed to make a clean "$x^\top W y$ is best" claim rather than a qualitative one. Also missing: AstroPh results for the symmetric-$W$ and Mahalanobis variants (only CITESEER is covered for those two), and no significance testing on any decoder comparison. -->

## Random forest and simple classifier baselines

Rather than training a neural decoder, a classifier can be trained directly on hand-combined embedding features. Table 5 reports accuracy and F1 for CITESEER at $P=1$, comparing three classifier families (random forest, SVC, logistic regression) across five embedding-combination preprocessors (Concatenate, Average, Hadamard, L1, L2).

| Model | Preprocessor | Accuracy | F1 |
|---|---|---|---|
| Random Forest | Concatenate | 0.7870 | 0.7409 |
| Random Forest | Average | 0.7826 | 0.7432 |
| SVC | Concatenate | 0.7781 | 0.7287 |
| SVC | Average | 0.7748 | 0.7265 |
| Logistic Regression | Hadamard | 0.7693 | 0.7149 |
| SVC | Hadamard | 0.7671 | 0.7114 |
| SVC | L1 | 0.7660 | 0.7014 |
| SVC | L2 | 0.7638 | 0.6994 |
| Logistic Regression | L2 | 0.7539 | 0.6810 |
| Logistic Regression | L1 | 0.7494 | 0.6724 |
| Random Forest | Hadamard | 0.7373 | 0.6510 |
| Random Forest | L1 | 0.7208 | 0.6184 |
| Random Forest | L2 | 0.7208 | 0.6184 |
| Logistic Regression | Average | 0.5673 | 0.5525 |
| Logistic Regression | Concatenate | 0.5651 | 0.5583 |
Table: CITESEER, $P=1$: accuracy and F1 for classifier/preprocessor combinations, best-first by accuracy.

At $P=1$, the best combinations (random forest or SVC, with Concatenate or Average preprocessing) are competitive with, though not clearly better than, the Hadamard logistic neural decoder from the previous section (0.7870 vs.\ 0.9382 in Table 4 at $P=1$ — the two tables use different train/test splits and are not directly comparable in absolute terms, but the *ranking* of preprocessors within this table is informative in its own right). Notably, logistic regression with Concatenate or Average preprocessing performs far worse (0.5651–0.5673 accuracy, barely above chance) than the same model with Hadamard, L1, or L2 preprocessing (0.7494–0.7693): a linear decision boundary over a raw concatenation or average of two embeddings is a poor fit for the link-prediction task, whereas an explicit pairwise interaction term (Hadamard) or distance term (L1/L2) gives a linear classifier something more directly informative to work with. Random forest and SVC do not show this sensitivity, since both are able to construct nonlinear decision boundaries over the raw concatenated or averaged features without an explicit interaction term.

The real value of the random forest with concatenated (or averaged) embeddings, however, is its robustness to partitioning, shown in Table 6 across all four datasets used in this paper plus Cit-HepTh, for $P \in \{1, 2, 4, 8\}$.

| Dataset | P | Accuracy | F1 |
|---|---|---|---|
| CITESEER | 1 | 0.7848 | 0.7354 |
| CITESEER | 2 | 0.7494 | 0.6825 |
| CITESEER | 4 | 0.7351 | 0.6581 |
| CITESEER | 8 | 0.7285 | 0.6424 |
| AstroPh | 1 | 0.9534 | 0.9524 |
| AstroPh | 2 | 0.9501 | 0.9491 |
| AstroPh | 4 | 0.9484 | 0.9472 |
| AstroPh | 8 | 0.9462 | 0.9449 |
| Cit-HepPh | 1 | 0.9539 | 0.9533 |
| Cit-HepPh | 2 | 0.9415 | 0.9401 |
| Cit-HepPh | 4 | 0.9356 | 0.9339 |
| Cit-HepPh | 8 | 0.9342 | 0.9322 |
| Cit-HepTh | 1 | 0.9467 | 0.9459 |
| Cit-HepTh | 2 | 0.9424 | 0.9415 |
| Cit-HepTh | 4 | 0.9413 | 0.9401 |
| Cit-HepTh | 8 | 0.9416 | 0.9404 |
| AS-Oregon | 1 | 0.9157 | 0.9097 |
| AS-Oregon | 2 | 0.9127 | 0.9067 |
| AS-Oregon | 4 | 0.9088 | 0.9023 |
| AS-Oregon | 8 | 0.9145 | 0.9092 |
Table: Random forest with concatenated embeddings: accuracy and F1 across partition counts and datasets.

Where the neural decoders lost 10–30 percentage points of accuracy between $P=1$ and $P=8$ (Table 3), the random forest classifier loses at most 5.6 percentage points over the same range, and on three of the five datasets (AstroPh, Cit-HepTh, AS-Oregon) the loss is under 1.5 percentage points, with Cit-HepTh and AS-Oregon even recovering slightly at $P=8$ relative to $P=4$. This supports the discussion in the vertex_voyage project notes: since each partition is embedded independently, the resulting latent spaces are not mutually aligned, which is expected to hurt a neural decoder that implicitly assumes a smooth, shared latent geometry across the pairs it is trained and evaluated on. A random forest, built from axis-aligned splits over the raw (concatenated or averaged) coordinates rather than from a smooth learned interaction function, is far less sensitive to this misalignment, and can still exploit whatever structure is present within each partition's own coordinates even when cross-partition comparisons are less meaningful.

## Discussion

Two threads run through the results above. First, partitioning strategy matters for downstream quality just as it does for reconstruction quality: LPA's community-preserving assignment consistently outperforms LFM for link prediction at every partition count tested. Second, and more importantly, the relationship between reconstruction quality and downstream link-prediction quality is not a simple positive correlation, and can go in the opposite direction: CITESEER's RESTORE F1 nearly doubles between $P=1$ and $P=32$ while its link-prediction accuracy falls by roughly a quarter over a comparable range. A plausible explanation is that RESTORE F1 measures how well the training graph's own edges are recovered from the embeddings' internal geometry, which increasing partition granularity can improve simply by making each partition's local reconstruction task easier and more self-contained; link-prediction accuracy, in contrast, is measured on edges that were never seen during training, including edges that cross partition boundaries where the decoder or classifier must reconcile two independently trained embedding spaces. A metric that rewards easier per-partition self-consistency is not the same as a metric that rewards generalization across those partitions, and the results in Table 2 show these two properties trading off against each other rather than moving together.

Regarding decoder and classifier choice, the practical recommendation supported by the results above mirrors the take-home messages already reached in the vertex_voyage project notes: prefer LPA over LFM for partitioning; prefer a random forest over a neural decoder when partition counts are non-trivial, since its robustness to unaligned per-partition latent spaces far outweighs the modest accuracy gap observed at $P=1$; and, if a neural decoder is required (for instance, to obtain a ranking rather than a binary decision), prefer the bilinear $x^\top W y$ architecture, which achieves the best mean reciprocal rank among the architectures tested even though its top-1 accuracy is comparable to the simpler Hadamard-product decoder.

# Conclusion

This paper studied how partitioning strategy, decoder architecture, and classifier choice affect link-prediction quality when embeddings are computed independently, per partition, with static node2vec. Label propagation (LPA) consistently outperformed the Lancichinetti-Fortunato-Kertesz method (LFM) as a partitioning strategy for downstream link prediction, at every partition count compared. More significantly, the same embeddings' graph-reconstruction F1 score and their link-prediction accuracy were found to move in opposite directions as the partition count grew on CITESEER — reconstruction F1 rose from 0.38 to a peak of 0.70 while link-prediction accuracy fell from 0.92 to 0.67 over a comparable range — demonstrating that reconstruction quality is not a reliable stand-in for downstream generalization quality in a partitioned embedding pipeline, a distinction that Seminar 3's reconstruction-only study could not surface on its own. Across four datasets, link-prediction accuracy, Hits@k, and MRR all degraded monotonically with partition count under every neural decoder tested, with the steepest drop occurring between one and two partitions, consistent with the introduction of cross-partition candidate pairs being the dominant source of quality loss. Among the neural decoders, the Hadamard-product logistic model and the bilinear $x^\top W y$ model achieved comparable top-1 accuracy, with $x^\top W y$ giving the best mean reciprocal rank, while a decoder based on the difference of the two embeddings performed markedly worse at every partition count. A random forest classifier trained on concatenated (or averaged) per-partition embeddings was substantially more robust to increasing partition count than any neural decoder, losing at most a few percentage points of accuracy between one and eight partitions across five datasets, because its axis-aligned decision boundaries do not assume the smooth, mutually aligned latent geometry that a neural decoder implicitly requires across independently trained partitions.

As future work, the dedicated static node2vec sweep referenced in the vertex_voyage project notes (varying epoch count and replication factor systematically across datasets and partition counts) remains to be completed and would sharpen the epoch-budget and replication-factor recommendations only informally available today; extending the LFM/LPA and decoder comparisons to the full five-dataset set used in Seminar 3, once link-prediction runs exist for DBLP and Enron, would also strengthen the generality of the conclusions drawn here. Finally, a more direct investigation of *why* reconstruction F1 and link-prediction accuracy diverge under partitioning — for instance, by decomposing link-prediction accuracy into within-partition and cross-partition test pairs — would help establish whether the divergence is driven entirely by cross-partition pairs, as this paper's evidence suggests, or partly by a more general effect of partition granularity on the two metrics.

# References
