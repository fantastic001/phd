---
title: "Graph Embedding Techniques with Applications in Software Engineering: A Systematic Literature Review"
author: "Stefan Nožinić"
date: "`r Sys.Date()`"
email: "stefan@lugons.org"
bibliography: ./refs.bib
---

# Introduction 

<!-- graph embedding with applications in software engineering -->

<!-- 
Structure:

inclusion/exclusion criteria

- Include papers that specifically address graph embedding techniques.
- Include applications of graph embeddings in software engineering.
- Exclude papers that do not provide empirical results or case studies.
- Exclude papers that focus solely on theoretical aspects without practical applications.

Quality assessment criteria:
- Evaluate the relevance of the graph embedding techniques to software engineering tasks.
- Assess the clarity and rigor of the methodology used in the studies.
- static vs dynamic graphs
- types of graphs (e.g., control flow graphs, call graphs, dependency graphs)
- distance metrics on embeddings
- scalability of graph embedding techniques
- types of machine learning models used with graph embeddings
- applications in software engineering (e.g., bug prediction, code recommendation, vulnerability detection)

Data extraction criteria:
- Extract information on the graph embedding techniques used.
- Extract details on the datasets and benchmarks employed in the studies.
- Extract the main findings and contributions of each paper.
- Extract any limitations or future work suggested by the authors.
 -->

## Motivation and related work

Today, software systems are becoming increasingly complex, making it challenging to analyze and understand their structure and behavior. Graphs provide a natural way to represent the relationships and interactions within software systems, such as control flow [@cheng_static_2019], function calls, and dependencies [@lyu_embedding_2021]. When represented as graphs, software systems can be analyzed using various graph-based techniques, including graph embeddings [@gedeon_embedding_2019]. Graph embeddings are a powerful tool for transforming graph-structured data into a continuous vector space, enabling the application of machine learning algorithms to graph data. 

Despite their potential, the application of graph embeddings in software engineering is still an emerging area of research. There is a need for a comprehensive review of the existing literature to identify the state-of-the-art techniques, their applications, and the challenges that remain to be addressed.

In [@wang_application_2023], the authors provide a survey of applications of knowledge graphs in software engineering. 

Also, [@wang_survey_2022] provides a comprehensive survey of graph embedding methods used when graph nodes and links have their own attributes.



## Problem statement 

A graph is a mathematical structure defined as tuple $G = (V, E)$, where $V$ is a set of vertices (or nodes) and $E$ is a set of edges (or links) connecting pairs of vertices. Graphs can be directed or undirected, weighted or unweighted, and can represent various types of relationships between entities.

Also, nodes in a graph can have attributes or features associated with them, which can provide additional information about the entities they represent. Graphs can be static, where the structure remains unchanged over time, or dynamic, where the structure evolves as nodes and edges are added or removed.

For dynamic graphs, the graph at time $t$ can be represented as $G_t = (V_t, E_t)$, where $V_t$ and $E_t$ are the sets of vertices and edges at time $t$. Dynamic graphs can capture temporal changes in relationships and interactions, making them suitable for modeling evolving systems.

A real graph or complex network is a graph that represents real-world systems, such as social networks [@zachary_information_1977], biological networks [@li_graph_2025], or software systems [@chatzigeorgiou_application_2006]. Real graphs often exhibit properties such as scale-freeness [@barabasi_emergence_1999], small-worldness [@watts_collective_1998], and community structure [@leskovec_community_2009], which can influence the choice of graph embedding techniques.

Graph embeddings are techniques that aim to represent the nodes, edges, or entire graphs in a continuous vector space while preserving the structural and relational information of the original graph. The goal of graph embeddings is to learn low-dimensional representations that capture the essential characteristics of the graph, enabling efficient analysis and machine learning tasks.

Formally, a graph embedding can be defined as a mapping function $f: V \rightarrow \mathbb{R}^d$, where $d$ is the dimensionality of the embedding space. The embedding should preserve the proximity and relationships between nodes in the original graph, such that nodes that are close or connected in the graph are also close in the embedding space.

## Scope 

This paper aims to systematically review the existing literature on graph embedding techniques with applications in software engineering. The review will focus on identifying the state-of-the-art methods, their effectiveness in various software engineering tasks, and the challenges that need to be addressed for further advancements in this field.

# Overview of graph embedding methods

## Static graph embeddings

One of the first attempts to embed graph nodes into a continuous vector space was presented in [@belkin_laplacian_2003] [@cheng_spectral_2020] where the authors proposed a method based on spectral graph theory. The method involves computing the eigenvectors of the graph Laplacian matrix and using them as the embedding vectors for the nodes. The approach aims to preserve the local structure of the graph by minimizing the distance between connected nodes in the embedding space. Formally, the graph Laplacian matrix $L$ is defined as $L = D - A$, where $D$ is the degree matrix and $A$ is the adjacency matrix of the graph. The eigenvectors corresponding to the smallest non-zero eigenvalues of $L$ are used as the embedding vectors. To construct the embedding $f(v_i)$ for a node $v_i$, the $i$-th row of the matrix formed by these eigenvectors can be taken. Disadvantages of this method include its computational complexity, which can be prohibitive for large graphs, and its sensitivity to noise and outliers in the graph structure.

One of the most well-known static graph embedding technique is Node2Vec [@grover_node2vec_2016] which extends the Word2Vec model [@church_word2vec_2017] to graphs by using random walks to generate node sequences. Node2Vec introduces two parameters, $p$ and $q$, to control the breadth-first and depth-first search strategies during the random walks, allowing for flexible exploration of the graph structure. Node2Vec was also extended to be more scalable by [@lombardo_scalable_2019].

Another popular static graph embedding is DistGER [@fang_distributed_2023] which extends random walk sampling to account increased information gain by considering the neighbors of the sampled nodes. This approach aims to capture more comprehensive structural information from the graph, leading to improved embedding quality. This paper also introduces a distributed training framework to handle large-scale graphs efficiently.

Another static graph embedding technique is LINE [@tang_line_2015] which is designed to handle large-scale information networks. LINE optimizes an objective function that preserves both local and global network structures, making it suitable for various types of graphs, including undirected, directed, and weighted graphs. The method employs an edge-sampling algorithm to improve the efficiency of the training process, allowing it to scale to networks with millions of nodes and billions of edges.

SDNE [@wang_structural_2016] is another static graph embedding technique that uses deep autoencoders to learn node representations. SDNE uses deep neural networks to capture the non-linear relationships in the graph, allowing for more expressive embeddings. The method incorporates both first-order and second-order proximity to preserve the local and global structures of the graph.

## Dynamic graph embeddings

For dynamic graphs, there is dynamic version of Node2Vec [@mahdavi_dynnode2vec_2018] which extends the original Node2Vec algorithm to handle dynamic graphs by updating the embeddings incrementally as the graph evolves. The method uses a combination of random walks and temporal information to capture the changes in the graph structure over time, allowing for efficient updates to the node embeddings without retraining from scratch.

Most recent papers on dynamic graph embedding focus more on building self-attention layers inspired by transformer models like [@sankar_dysat_2020] and [@wang_apan_2021] which use attention mechanisms to capture the temporal dependencies and structural information in dynamic graphs. These methods typically involve learning node embeddings through a series of attention layers that aggregate information from neighboring nodes and previous time steps, allowing for the modeling of complex temporal patterns in the graph data.

Other models make use of variational recurrent neural networks like [@hajiramezanali_variational_2019] which combines variational autoencoders with recurrent neural networks to learn dynamic graph embeddings. The model captures the temporal evolution of the graph by modeling the latent representations of nodes and edges over time. 



# Types of graphs in software engineering

In this section, several types of graphs commonly used in software engineering are discussed, along with their characteristics and applications.

## Control flow graphs and call graphs

Control flow graphs (CFGs) are directed graphs that represent the flow of control within a program. In a CFG, nodes represent basic blocks of code, and edges represent the control flow between these blocks. CFGs are widely used in various software engineering tasks, such as program analysis, optimization, and testing. 

A call graph is a directed graph that represents the calling relationships between functions in a program. In a call graph, nodes represent functions, and edges represent calls from one function to another. Call graphs are useful for understanding the structure of a program, identifying performance bottlenecks, and optimizing function inlining.

Control-flow graphs and call graphs are successfully used as a representation method of a program for software watermarking [@chroni_embedding_2012]. 

## Dependency graphs

A dependency graph is a directed graph that represents the dependencies between various components of a software system, such as modules, classes, or functions. In a dependency graph, nodes represent the components, and edges represent the dependencies between them. Dependency graphs are useful for analyzing the structure of a software system, identifying potential issues, and managing changes in the codebase. 

In object-oriented design, dependency graphs can play a huge role in identifying design flaws [@chatzigeorgiou_application_2006]. 

## Other types of graphs

# Scalability of graph embedding techniques

In order to handle large-scale graphs, graph can be partitioned into smaller subgraphs $ G_1, G_2, \ldots, G_k$ such that $G = \bigcup_{i=1}^{k} G_i$. Each subgraph can be embedded independently, and the resulting embeddings can be combined to form the final embedding for the entire graph. This approach allows for parallel processing and reduces the computational complexity of embedding large graphs.

Graph partitioning is NP-hard problem which is optimization problem thet involves dividing a graph into smaller subgraphs while minimizing the number of edges between the subgraphs and ensuring that each subgraph is of roughly equal size.

Over the years, several partitioning algorithms have been presented in the literature. Some of them are based on evolutionary algorithms while others use spectral clustering or modularity optimization. The choice of partitioning algorithm can significantly impact the quality of the resulting embeddings, as it determines how well the subgraphs capture the structural properties of the original graph.



The most notable evolutionary algorithm for graph partitioning is the memetic algorithm presented in [@romero_ruiz_memetic_2018] which combines genetic algorithm with local search technique. It uses Hungarian algorithm to solve the assignment problem during the crossover operation, ensuring that the offspring inherit the best traits from their parents. The local search technique is applied to refine the partitions and improve the overall quality of the solution.

Another multilevel memetic algorithm is presented in [@benlic_effective_2010] which introduces a novel recombination operator that combines the best features of two parent partitions to create an offspring partition. The algorithm also incorporates a local search technique to refine the partitions and improve the overall quality of the solution. Although this approach has been shown to be computationally expensive, it produces high-quality partitions.

Apart from evolutionary algorithms, in real networks which often exhibit community structures, community detection with balancing algorithm can be used. For instance, detecting communities using LFM algorithm [@lancichinetti_detecting_2009] and then creating balanced partitions by using a bin packing algorithm [@gupta_new_1999]. This approach leverages the inherent community structure of real-world graphs to create partitions that are both balanced and have a low number of inter-partition edges.

In case of dynamic graphs, there are several algorithms that have been proposed. For instance, adaptive partitioning [@vaquero_adaptive_2013] where at each iteration of vertex creation or removal, the algorithm checks if the current partitioning is still optimal. If not, it re-partitions the graph to ensure that the partitions remain balanced and the number of edges between partitions is minimized. Repartitioning is done through vertex migration, where vertices are moved from one partition to another to improve the overall quality of the partitioning.



# Applications in software engineering

In the Table 1, an overview of various applications of graph embeddings in software engineering is provided. Each application is briefly described, along with the specific graph embedding techniques used and the relevant references.

| Application | Graph representation | Embedding technique | References |
|-------------|----------------------|---------------------|------------|
| Design pattern detection | Dependency graphs | Node2Vec, LINE, DistGER | [@chatzigeorgiou_application_2006] |
| Bug prediction | Dependency graphs | Node2Vec, LINE, SDNE | [@qu_node2defect_2018] [@qu_node2defect_2018] | 
| Malware detection | Call graphs | Spectral methods | [@hashemi_graph_2017] | 


## Design pattern detection

As pointed out in [@chatzigeorgiou_application_2006], design patterns are typical solutions to common problems in software design. They provide a way to reuse successful designs and architectures, making it easier to develop and maintain complex software systems. In mentioned paper, graph similarity was used to identify which design pattern was used. Similarly, embedding methods can map nodes to a vector space where node similarity can be measured using distance metrics like cosine similarity or Euclidean distance. This allows for efficient comparison of nodes and identification of similar patterns in the graph.

## Bug prediction

Bug prediction is a crucial task in software engineering that aims to identify potential defects in software systems before they manifest as actual bugs. By leveraging graph embeddings, it is possible to capture the structural and relational information of software components, enabling more accurate predictions of bug-prone areas in the codebase. In [@qu_node2defect_2018], the authors propose a method for predicting software defects using Node2Vec embeddings of dependency graphs. The approach involves constructing a dependency graph from the class dependencies in the codebase and then generating node embeddings using several state-of-the-art graph embedding techniques, including Node2Vec, LINE, and SDNE. The resulting embeddings are concatenated with traditional software metrics to form a comprehensive feature set for each class. These features are then used to train a machine learning model to predict the likelihood of defects in the classes. Models such as Random Forest, Logistic Regression, and Support Vector Machines are employed for the prediction task. 

## Code recommendation




## Other applications





# Challenges and future directions

# Conclusion

# References