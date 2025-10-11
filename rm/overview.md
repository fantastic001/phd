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

One of the most well-known static graph embedding technique is Node2Vec [@grover_node2vec_2016] which extends the Word2Vec model [@church_word2vec_2017] to graphs by using random walks to generate node sequences. Node2Vec introduces two parameters, $p$ and $q$, to control the breadth-first and depth-first search strategies during the random walks, allowing for flexible exploration of the graph structure. Node2Vec was also extended to be more scalable by [@lombardo_scalable_2019].

Another popular static graph embedding is DistGER [@fang_distributed_2023] which extends random walk sampling to account increased information gain by considering the neighbors of the sampled nodes. This approach aims to capture more comprehensive structural information from the graph, leading to improved embedding quality. This paper also introduces a distributed training framework to handle large-scale graphs efficiently.



## Dynamic graph embeddings

# Types of graphs in software engineering
## Control flow graphs
## Call graphs
## Dependency graphs
## Other types of graphs

# Distance metrics for graph embeddings

# Scalability of graph embedding techniques

# Applications in software engineering
## Bug prediction
## Code recommendation
## Vulnerability detection
## Other applications

# Comparison of graph embedding techniques and their effectiveness in software engineering tasks

# Challenges and future directions

# Conclusion

# References