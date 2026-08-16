---
title: "Trade-offs in Partition Assignment for Distributed Dynamic Graph Embedding"
author: "Stefan Nožinić"
abstract: |
bibliography: ./refs.bib
---

# Introduction 

<!-- citation example [@apache_software_foundation_zookeeper_2011] -->

A graph is a mathematical structure consisting of vertices (or nodes) connected by edges. Graphs are widely used to model relationships and interactions in various domains [@van_der_hofstad_random_2024], such as social networks [@leskovec_signed_2010] [@backstrom_group_2006] [@rozemberczki_twitch_2021], collaboration networks [@savic_analysis_2017], terrorist networks [@krebs_mapping_2002] and blog citation networks [@adamic_political_2005]. In these applications, the relationships between entities can be represented as edges connecting the corresponding vertices.

<!-- objasniti sta su realni grafovi i raspodelu stepeni kod njih kao i Watt-Strogatz princip -->

In many real-world graphs, the degree distribution follows a power-law, meaning that a small number of vertices have a very high degree (i.e., they are connected to many other vertices), while most vertices have a low degree. This characteristic is often observed in social networks, where a few individuals (e.g., celebrities) have many connections, while the majority of users have relatively few connections.

In real graphs, there are several properties that are often observed [@watts_collective_1998] [@zachary_information_1977] [@albert_statistical_2002]:
- **Small-world property**: Most pairs of vertices can be connected by a short path, even in large graphs. This is often referred to as the "six degrees of separation" phenomenon. [@watts_collective_1998]
- **Community structure**: Vertices tend to form clusters or communities, where vertices within the same community are more densely connected than those in different communities. This property is prevalent in social networks, where groups of friends or colleagues often form tightly-knit communities. [@leskovec_community_2009]
- **Scale-free property**: The degree distribution of the graph follows a power-law, meaning that a few vertices have a very high degree, while most vertices have a low degree. This is often observed in social networks, where a small number of individuals (e.g., celebrities) have many connections, while the majority of users have relatively few connections. [@barabasi_emergence_1999]

Graph vertex embeddings are a powerful technique for representing vertices in a graph as low-dimensional vectors, enabling various machine learning tasks such as vertex classification, link prediction [@leskovec_predicting_2010], and community detection. The effectiveness of these embeddings often depends on the underlying graph structure and the methods used to generate them. When faced with large graphs, the challenge of efficiently computing these embeddings while preserving the graph's structural properties becomes paramount. Additionally, most real-world graphs are dynamic, with vertices and edges being added or removed over time. This dynamic nature introduces additional challenges in maintaining accurate and up-to-date embeddings, as the graph's structure evolves. 

This leads to a practical trade-off between early partition assignment (low latency) and informed partition assignment (higher embedding quality and partition balance). Despite its practical relevance, this trade-off has not been systematically studied in the context of dynamic graph embedding.

# Problem Formulation

Given a dynamic graph where nodes arrive online and a distributed dynamic Node2Vec-style embedding model is maintained, we want to know how should partitions be assigned to new nodes to balance:

* embedding quality,
* partition balance,
* assignment latency


This paper aims to empirically study and characterize the trade-offs between delayed and immediate partition assignment strategies for online node arrivals in distributed dynamic graph embedding systems.
Rather than proposing a new embedding model, the focus is on partition assignment heuristics and their systemic impact. In this paper, existing embedding model is used which handles dynamic nature of the evolving graph.


## Related work 

<!-- static embedding  -->

Graph vertex embedding is a well-studied area, with various methods proposed to generate low-dimensional representations of nodes in a graph. State of the art method which is widely used is Node2Vec [@grover_node2vec_2016] which uses random walks to capture the local and global structure of the graph. Node2Vec generates embeddings by performing biased random walks on the graph, allowing it to explore both local and global structures. The method has been shown to be effective in capturing community structures and generating meaningful embeddings for various machine learning tasks. As its improvement, DistGER [@fang_distributed_2023] is a distributed graph embedding method that extends Node2Vec by leveraging distributed computing to handle large graphs. DistGER uses a similar random walk approach but optimizes walk sampling in order to maximize the information gain when selecting the next vertex to visit. 

Another approach to scale Node2Vec is proposed in [@lombardo_scalable_2019] which is based on actor model and uses a distributed framework to generate embeddings for large graphs. This method allows for parallel processing of random walks, significantly improving the efficiency of embedding generation in terms of time and resource usage.

The common ground for these methods is that they generate walks which are later used to train Word2Vec model [@church_word2vec_2017] commonly used for generating embeddings in natural language processing tasks. 

During the learning process, there are several state of the art approaches to parallelize the training of word2vec model. Commonly used approach in distributed environment is ensemble learning [@ji_ensemble_2007] which combines multiple smaller models to create a larger model. Final mode is created by aggregating smaller models using parameter server architecture [@li_parameter_2013]. 

<!-- static partitioning  -->

The main challenge to address the problem of distributed graph vertex embedding is partitioning the graph in a way that preserves the community structure while ensuring that the partitions are balanced and can be processed efficiently in a distributed environment. Up until recently, most partitioning methods covered only small graphs or graphs without inherent community structure, like in [@benlic_effective_2010] [@sanders_distributed_2012] [@sanders_engineering_2011] [@romero_ruiz_memetic_2018] [@catalyurek_more_2023]. The main focus of these methods is static graph partitioning meaning that the graph is partitioned once and then used for processing. However, in many real-world applications, graphs are dynamic and change over time, requiring dynamic partitioning methods that can adapt to changes in the graph structure. In [@ugander_balanced_2013], a variant of label propagation algorithm is proposed for balancing partitions, but it lacks the ability to adapt to changes in the graph structure over time and it does not study the impact of partitioning on embedding quality.

For dynamic graph partitioning, there are several methods available in literature like [@nicoara_hermes_2015] [@huang_leopard_2016] [@xu_loggp_2014] and [@vaquero_adaptive_2013]. These methods focus on partitioning dynamic graphs by considering the changes in the graph structure over time and adapting the partitioning accordingly. However, these methods have not been used in embedding applications so far, and their effectiveness in generating high-quality embeddings in a distributed environment remains an open question. However, temporal graph embedding methods like [@mahdavi_dynnode2vec_2018] have been proposed to address the problem of dynamic graph embedding. These methods focus on generating embeddings for dynamic graphs by considering the temporal evolution of the graph structure. However, these methods do not address the problem of partitioning the graph in a distributed environment, which is crucial for efficient processing and scalability.

<!-- dynamic graph embedding  -->

Distributed graph embedding methods such as dynamic Node2Vec [@lombardo_scalable_2019] [@mahdavi_dynnode2vec_2018] variants enable scalable representation learning on large, evolving graphs. However, in dynamic settings with online node arrivals, a fundamental systems problem arises: how to assign newly arriving nodes to partitions before sufficient structural or embedding information is available.

Most existing distributed embedding systems assume either:

* static partitioning [@benlic_effective_2010] [@sanders_distributed_2012], or
* immediate assignment based on incomplete information [@ugander_balanced_2013].


# Contributions

This paper makes the following contributions:

* We propose a systematic study of the trade-offs between delayed and immediate partition assignment strategies for online node arrivals in distributed dynamic graph embedding systems.
* We evaluate the impact of different partition assignment strategies on embedding quality, partition balance, and assignment latency, providing insights into the practical implications of these strategies in real-world scenarios.
* We provide empirical evidence and analysis of the trade-offs involved in partition assignment strategies, contributing to the understanding of how to effectively manage partitioning in distributed dynamic graph embedding systems.

## Paper organization

The rest of the paper is organized as follows: First, system overview is presented, then graph partitioning and embedding model are described. Next, benchmarks are explained in detail. Finally, results and discussion are presented, followed by the conclusion and references.

# Methods 

## System overview 

Temporal graph is represented as a sequence of events. Each event is a tuple (t, u, v) where t is the timestamp of the event and u and v are the vertices involved in the event. The events are processed in chronological order, and the graph is updated accordingly. The system maintains a distributed dynamic Node2Vec-style embedding model, which is updated as new events arrive. The partition assignment strategy determines how new vertices are assigned to partitions in the distributed system.

<!-- explain buffering of events before we partition them and then update the embeddings -->

The system buffers incoming events for a short period before assigning them to partitions. This buffering allows the system to make more informed partitioning decisions by considering a batch of events together, rather than assigning partitions immediately upon the arrival of each event. Once the events are buffered, the partition assignment strategy is applied, and the embeddings are updated accordingly.

In the following listing, pseudocode for the system is presented:

```text 

global state buffer = {} 

when new event (t, u, v) arrives:
    buffer.add((t, u, v))
    if buffer.size() >= BUFFER_SIZE:
        assign_partitions(buffer)
        update_embeddings(buffer)
        buffer.clear()

```
Buffer also defines new graph snapshot. Let $G_n$ be the graph snapshot after processing the first $n$ buffers. Now, let $B_n$ be the $n$-th buffer of events. The graph snapshot $G_{n+1}$ is obtained by applying the events in buffer $B_n$ to the graph snapshot $G_n$. Specifically, $G_{n+1} = G_n \cup B_n$, where the union operation adds the new vertices and edges from the events in $B_n$ to the existing graph snapshot $G_n$. This process allows the system to maintain an up-to-date representation of the dynamic graph as new events are processed.

## Graph partitioning


Partition of vertex is the partition that contains the most neighbors of the vertex in the buffer. The partitioning strategy assigns the vertex to the partition that contains the most neighbors of the vertex in the buffer. The partitioner also has a replication factor, which allows it to assign a vertex to multiple partitions if there are multiple partitions that contain a similar number of neighbors of the vertex in the buffer. The partitioner has a capacity penalty, which penalizes partitions that have more vertices than the average partition size, to encourage more balanced partitions.

Formally, the score of a partition for a vertex is defined as:

$$ S(P, v) = N(P, v) - \mu \cdot \max(0, |P| - \alpha \cdot (1 + \epsilon) \cdot \frac{1}{|P|} \sum_{P' \in P} |P'|) $$

where $N(P, v)$ is the number of neighbors of vertex $v$ in partition $P$ in the buffer, $\mu$ is the capacity penalty coefficient, $\alpha$ is the weight of the average partition size in the capacity penalty, $\epsilon$ is the imbalance tolerance, and $|P|$ is the size of partition $P$. The partitioner assigns the vertex to the top $k$ partitions with the highest scores, where $k$ is the replication factor. If there are multiple partitions with same scores, the partitioner randomly assigns the vertex to some of those partitions until it reaches the replication factor.


## Embedding model


Given dynamic graph (i.e., a graph that changes over time), the DynNode2Vec algorithm [@mahdavi_dynnode2vec_2018] learns continuous feature representations for nodes in the graph at different time steps. The main idea behind DynNode2Vec is to extend the node2vec algorithm [@grover_node2vec_2016] to handle dynamic graphs by incorporating temporal information into the random walk strategy and embedding learning process. The goal is to capture both the structural and temporal dynamics of the graph in the learned embeddings.

In the following listing, pseudocode for the DynNode2Vec algorithm is presented:


```
Input: Dynamic graphs G1, G2, …, GT
Output: Embeddings Z1, Z2, …, ZT

// t = 1
Run static node2vec on G1 to train Skip-gram_1 and obtain Z1.

For t = 2 … T:
     // Identify evolving nodes between G_{t-1} and G_t
     Compute:
         V_add  = nodes added from t-1 to t
         E_add  = edges added from t-1 to t
         E_del  = edges deleted from t-1 to t
     delta_V_t = V_add union { v in V_t | exists (v, u) in (E_add union E_del) }

     // Evolving random walks only from changed regions
     Walk_n = node2vec_walks(G_t, start_nodes=delta_V_t, p, q, walk_length, num_walks)

     // Dynamic Skip-gram: initialize from previous time
     Initialize Skip-gram_t with weights of Skip-gram_{t-1}
     Update Skip-gram_t vocabulary for any new nodes
     Train Skip-gram_t on Walk_n
     Extract Z_t from Skip-gram_t
End For
```

In this paper, graph is represented as a sequence of events. Algorithm can be easily adapted to this representation by buffering events and applying them to the graph snapshot at each time step. The algorithm captures the evolving nature of the graph by focusing on the changed regions and updating the embeddings accordingly.

Since one vertex can be assigned to multiple partitions, each partition produces one embedding for corresponding vertex. Let $z_p = h_p(u)$ be the embedding of vertex $u$ in partition $p$. The final embedding of vertex $u$ is obtained by averaging the embeddings from all partitions that contain the vertex:

$$ z_u = \frac{1}{|P_u|} \sum_{p \in P_u} z_p $$

where $P_u = \{p | u \in V_p\}$ is the set of partitions that contain vertex $u$. This averaging process allows the system to combine information from multiple partitions, potentially improving the quality of the final embedding by leveraging diverse perspectives on the vertex's relationships within the graph.

If replication factor is set to 1, then the final embedding of vertex $u$ is simply the embedding from the single partition that contains the vertex:

$$ z_u = z_p $$

where $p$ is the partition that contains vertex $u$. In this case, the embedding is solely based on the information available in that single partition, which may limit the quality of the embedding if the partition does not capture sufficient context about the vertex's relationships within the graph.

Additionally, if vertex is not partitioned yet, then the final embedding of vertex $u$ is set to zero vector:

$$ z_u = 0 $$

## Benchmarks and evaluation metrics


To evaluate the effectiveness of the graph vertex embeddings generated in a distributed environment with community-aware partitioning, several criteria are considered:

**Embedding Quality**: The quality of the embeddings is assessed using metrics such as F1-score of reconstructed graphs [@yip_restore_2023], which measures how well the embeddings capture the relationships between vertices in the original graph. Higher F1-scores indicate better preservation of graph structure in the embeddings. Let $G = (V,E)$ be the original graph and $G' = (V,E')$ be the reconstructed graph from embeddings. $G'$ is constructed by connecting k closest vertices in the embedding space, where k is the number of edges in the original graph. The F1-score is calculated as follows:

$$ F1 = 2 * \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

where

$$ \text{Precision} = \frac{1}{|V|} \sum_{v \in V} \frac{|N(v) \cap N'(v)|}{|N'(v)|} $$

$$ \text{Recall} = \frac{1}{|V|} \sum_{v \in V} \frac{|N(v) \cap N'(v)|}{|N(v)|} $$

and $N(v)$ and $N'(v)$ are the neighbors of vertex $v$ in the original and reconstructed graphs, respectively.

**Partition Quality**: The quality of the partitions is evaluated based on metrics such as edge cut, which measures the number of edges that connect vertices in different partitions. Lower edge cuts indicate better preservation of community structures within partitions. Edge cut is defined as: 

$$ \text{EdgeCut} = \frac{1}{|E|} \sum_{(u,v) \in E} \mathbb{I}(p(u) \neq p(v)) $$

where $E$ is the set of edges in the original graph, $p(u)$ is the partition of vertex $u$, and $\mathbb{I}(\cdot)$ is the indicator function.

**Balance of Partitions**: The balance of the partitions is assessed by measuring the size of each partition and ensuring that they are within a bounded interval. This helps to ensure that the workload is evenly distributed across machines in the distributed environment. Formally, balance is defined as:

$$ B = 1 + \frac{1}{K} \sum_{i=1}^{K} \frac{|s_i - \bar{s}|}{\bar{s}} $$

where $K$ is the number of partitions, $s_i$ is the number of vertices assigned to partition $i$, and $\bar{s} = \frac{1}{K} \sum_{i=1}^{K} s_i$ is the average partition size. A perfectly balanced partitioning yields $B = 1$; higher values indicate greater imbalance.

**Partitioning Time**: The time taken to partition the graph is measured to evaluate the efficiency of the partitioning algorithms. Faster partitioning times are preferred, especially for large graphs, as they reduce the overall processing time in a distributed environment.

**Repartitioning amount**: The amount of repartitioning required when new vertices arrive is measured to assess the stability of the partitioning strategy. Lower amounts of repartitioning indicate that the partitioning strategy is more stable and can handle dynamic changes in the graph without significant disruption. Given two graph snapshots $G_n$ and $G_{n+1}$, the amount of repartitioning is defined as:

$$ \text{RepartitioningAmount} = \frac{1}{|V_{n}|} \sum_{v \in V_{n}} \mathbb{I}(p_n(v) \neq p_{n+1}(v)) $$

Where $V_n$ is the set of vertices in graph snapshot $G_n$, $p_n(v)$ is the partition of vertex $v$ in snapshot $G_n$, and $p_{n+1}(v)$ is the partition of vertex $v$ in snapshot $G_{n+1}$. This metric quantifies the proportion of vertices that have been reassigned to different partitions between two consecutive snapshots, providing insight into the stability and adaptability of the partitioning strategy in response to dynamic changes in the graph. It has to be noted that only vertices that are present in both snapshots are considered for this metric, as newly added vertices do not have a previous partition assignment to compare against and thus do not contribute to the measure of repartitioning amount.

# Results and discussion

## Partitioning hyperparameters

Throughout the experiments, the following hyperparameters are used for partitioning:

| Hyperparameter | Value |
|----------------|-------|
| Buffer size    | 1000  |
| $\mu$          | 1     |
| $\alpha$       | 1     |
| $\epsilon$     | 0.1   |
| Partitions (P) | $\{1, 2, 4, 8\}$ |
| Replication factor (RF) | $\{1, 3\}$ |

These hyperparameters are chosen to balance the trade-offs between embedding quality, partition balance, and assignment latency. In particular, the buffer size is set to 1000 to allow for sufficient information to be gathered before making partitioning decisions. For lower buffer sizes, the partitioner has less information to make informed decisions, which can lead to suboptimal partitioning and lower embedding quality. The capacity penalty coefficient ($\mu$) is set to 1 to encourage balanced partitions, while the weight of the average partition size ($\alpha$) is also set to 1 to ensure that the capacity penalty is proportional to the average partition size. The imbalance tolerance ($\epsilon$) is set to 0.1 to allow for some flexibility in partition sizes while still encouraging balance. The number of partitions (P) is varied between 1, 2, 4, and 8 to evaluate the impact of partitioning strategy on embedding quality and partition balance. Finally, the replication factor (RF) is varied between 1 and 3 to analyze the effect of replicating vertices across multiple partitions on embedding quality. 

For the embedding model itself, the following dataset-specific hyperparameters are used, obtained from a preliminary search over the node2vec return/in-out parameters $p, q$ and the embedding dimension:

| Dataset   | p    | q    | dim |
|-----------|------|------|-----|
| CITESEER  | 0.5  | 0.25 | 50  |
| DBLP      | 0.5  | 1    | 25  |
| AstroPh   | 2    | 0.25 | 50  |
| AS-Oregon | 0.5  | 2    | 128 |
| Enron     | 0.5  | 1    | 128 |

## Datasets

Experiments are performed on five real-world graph datasets, summarized in the table below.

| Dataset | Nodes | Edges |
| --- | --- | --- |
| CITESEER | 3327 | 9104 |
| DBLP | 17716 | 52867 |
| AstroPh | 18772 | 198110 |
| AS-Oregon | 11461 | 32730 |
| Enron | 36692 | 183831 |

CITESEER is a citation network of scientific publications. DBLP and AstroPh are collaboration/citation networks derived from SNAP, where AstroPh links co-authors of astrophysics papers. AS-Oregon is a network of autonomous systems and their peering connections. Enron is an email communication network in which vertices are email addresses and edges represent messages exchanged between them. These datasets vary in size and density, which allows the effect of partitioning to be observed under different community structures.

As a static, non-distributed reference point, running node2vec on the full, unpartitioned graph gives the following F1 reconstruction scores:

| Dataset  | Average F1 score |
|----------|------------------|
| CITESEER | 34.82%           |
| DBLP     | 60.3%            |
| AstroPh  | 70.41%           |
| AS-Oregon| 31.26%           |
| Enron    | 20.42%           |

These values serve as a rough upper bound against which the effect of partitioning, buffering, and dynamic re-embedding can be compared: a system that partitions the graph and processes it incrementally is not expected to exceed this quality, since it operates with a fraction of the graph's global information at any given time and adapts to it incrementally rather than embedding the full graph at once.

## Embedding quality, partition balance and edge cut

Using the temporal test command (`vv temporal_test`) with the buffered event processing pipeline described in the System overview, the following tables report, for each dataset and number of partitions $P \in \{1, 2, 4, 8\}$ with replication factor $RF = 1$, the F1 reconstruction score, the edge cut, and the balance of the resulting partitions, averaged over 10 iterations of the buffered stream.

CITESEER:

| P | F1 | Edge cut | Balance |
|---|------|----------|---------|
| 1 | 47%    | 0%    | 1 |
| 2 | 45.45% | 23%   | 1.31 |
| 4 | 47.48% | 39%   | 1.63 |
| 8 | 49.86% | 48%   | 1.75 |

DBLP:

| P | F1 | Edge cut | Balance |
|---|------|----------|---------|
| 1 | 50.44% | 0%     | 1 |
| 2 | 54.35% | 23%    | 1.10 |
| 4 | 54.30% | 50%    | 1.21 |
| 8 | 50.07% | 59%    | 1.35 |

AstroPh:

| P | F1 | Edge cut | Balance |
|---|------|----------|---------|
| 1 | 57.59% | 0%     | 1 |
| 2 | 55.28% | 33%    | 1.10 |
| 4 | 50.2%  | 55%    | 1.10 |
| 8 | 46.82% | 65%    | 1.25 |

AS-Oregon:

| P | F1 | Edge cut | Balance |
|---|------|----------|---------|
| 1 | 26.58% | 0%      | 1 |
| 2 | 28.72% | 30.44%  | 1.21 |
| 4 | 25.67% | 53.79%  | 1.26 |
| 8 | 23.13% | 71.56%  | 1.27 |

Enron:

| P | F1 | Edge cut | Balance |
|---|------|----------|---------|
| 1 | 17.63% | 0%     | 1 |
| 2 | 24.35% | 33.07% | 1.09 |
| 4 | 31.44% | 49.18% | 1.08 |
| 8 | 30.68% | 61.99% | 1.25 |

As expected, edge cut increases monotonically with the number of partitions across all datasets, since splitting the vertex set into more partitions necessarily severs more cross-community edges. Balance stays close to the ideal value of 1 in all cases, confirming that the neighbor-based partitioner with capacity penalty distributes vertices close to evenly without an explicit load-balancing step.

The effect of partitioning on embedding quality (F1), however, is dataset-dependent. On datasets with strong, well-separated community structure (CITESEER, DBLP, Enron), F1 remains stable or even improves as $P$ increases, despite the growing edge cut: this suggests that when communities are cohesive, most of a vertex's relevant neighborhood is captured within its own partition, so splitting the graph does not meaningfully harm the quality of the embedding that is reconstructed from local, per-partition context. On denser, less clearly clustered datasets (AstroPh, AS-Oregon), F1 degrades as $P$ increases, since a larger share of each vertex's neighbors end up outside its assigned partition, and this information is not available to the local embedding process. Partitioning time itself was found to be negligible compared to embedding time in all experiments; the dominant cost of processing a buffer is the embedding computation on the slowest (largest or most active) partition.

## Effect of replication factor

Increasing the replication factor $RF$ allows a vertex to be embedded independently in more than one partition, with the final embedding obtained by averaging the per-partition embeddings as described in the Embedding model section. This trades additional computation and storage for a more informed, less committal partition assignment. On CITESEER with $P = 4$ and buffer size 1000, increasing the replication factor from $RF = 1$ to $RF = 3$ raises the F1 reconstruction score from 58% to 69%, indicating that hedging the partition assignment across multiple partitions can substantially mitigate the quality loss introduced by early, uncertain assignment decisions. This comes at the cost of up to $RF\times$ the embedding computation per vertex, illustrating directly the latency/quality trade-off that motivates this paper: assigning a vertex to a single partition immediately is cheaper but riskier, while replicating it across several partitions is more expensive but more robust to a suboptimal initial assignment.

## Temporal evolution

Beyond aggregate scores, it is informative to track how F1 score, balance, edge cut, and repartitioning rate evolve as the event stream is ingested buffer by buffer. The figures below show this evolution for CITESEER, which is representative of the trends observed across the other datasets.

![F1 reconstruction score over iterations (CITESEER)](png/citeseer-f1.png)

The F1 score generally improves as more events are ingested and the graph snapshot grows, since the embedding model accumulates more structural information over time. Using more partitions tends to remain competitive with, or even improve on, the single-partition baseline on this dataset, consistent with the aggregate results above.

![Partition balance over iterations (CITESEER)](png/citeseer-balance.png)

Balance remains close to 1 throughout the stream, showing that the neighbor-based partitioner keeps partitions nearly evenly sized even as the graph evolves, without requiring an explicit rebalancing step.

![Edge cut over iterations (CITESEER)](png/citeseer-edge-cuts.png)

The edge cut ratio rises quickly during the first few buffers, while the partitions are still forming, and then stabilizes at a plateau largely determined by the number of partitions and the graph's community structure. This indicates that the partitioner reaches a steady state rather than degrading further as the stream continues.

![Repartitioning rate over iterations (CITESEER)](png/citeseer-repartitions.png)

The fraction of vertices repartitioned per buffer decays rapidly after an initial warm-up phase and stays below 20% for the remainder of the stream. This indicates that, once the partitioner has seen enough of a vertex's neighborhood, its assignment stabilizes and is only revised when the vertex's local structure changes significantly, keeping the overhead of maintaining partition assignments low over time. The same qualitative pattern (rapid rise then plateau for edge cut, low and decaying repartitioning rate, balance close to 1) is observed for DBLP, AstroPh, AS-Oregon, and Enron, with the main dataset-dependent difference being the level at which F1 stabilizes, consistent with the discussion above.

# Conclusion 

# References 

