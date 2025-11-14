---
title: "Graph Embedding Techniques with Applications in Software Engineering: A Systematic Literature Review"
author: "Stefan Nožinić"
date: "2025-10-12"
email: "stefan@lugons.org"
bibliography: ./refs.bib
abstract: |
    While graph embeddings have been widely studied in various domains, their application in software engineering is still an emerging area of research. This paper presents a systematic review of graph embedding techniques with applications in software engineering. The review focuses on identifying the state-of-the-art methods, their effectiveness in various software engineering tasks, and the challenges that need to be addressed for further advancements in this field. The findings suggest that graph embeddings have the potential to significantly improve the performance of machine learning models in software engineering tasks, such as bug prediction, code recommendation, and vulnerability detection. However, there are still several challenges that need to be addressed, including scalability, interpretability, and the need for more comprehensive datasets and benchmarks.
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

## Motivation 

Today, software systems are becoming increasingly complex, making it challenging to analyze and understand their structure and behavior. Graphs provide a natural way to represent the relationships and interactions within software systems, such as control flow [@cheng_static_2019], function calls, and dependencies [@lyu_embedding_2021]. When represented as graphs, software systems can be analyzed using various graph-based techniques, including graph embeddings [@gedeon_embedding_2019]. Graph embeddings are a powerful tool for transforming graph-structured data into a continuous vector space, enabling the application of machine learning algorithms to graph data. 

Despite their potential, the application of graph embeddings in software engineering is still an emerging area of research. There is a need for a comprehensive review of the existing literature to identify the state-of-the-art techniques, their applications, and the challenges that remain to be addressed.

## Related work 

There are several surveys on graph embedding techniques and their applications in various domains. For instance [@xu_understanding_2021] provides a listing of several graph embedding techniques and their applications in social networks, citation networks, biological networks and genome analysis.


In [@wang_survey_2022], the authors provide a survey of vast majority of graph embedding techniques and also mention some notable deployments in industry. They also mention several open source datasets which can be used for benchmarking graph embedding techniques. Similarly, in [@wang_application_2023], the authors provide a comprehensive survey of several applications of knowledge graphs in the field of software engineering, including software testing, bug prediction, and code recommendation. 

There are also several surveys which focus specifically on formal definition of graph embedding techniques like [@cai_comprehensive_2018] where the authors provide a comprehensive survey of graph embedding techniques along with some challenges and future directions in this field. 

When it comes to more specific areas like biomedical networks, [@chen_literature_2020] provides a survey of graph embedding techniques and their applications in biomedical data analysis. The authors discuss various graph embedding methods and their effectiveness in tasks such as drug discovery, disease prediction, and protein function prediction along with several public datasets.

Knowledge graph embeddings are surveyed in [@ge_knowledge_2024] where the authors provide a comprehensive survey of knowledge graph embedding techniques based on distance and semantic matching models. The authors discuss various embedding methods and focus on distance-based models like CoumpoundE and CoumpoundE3D which originate from affine transformations in geometry.





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

When graph represents multiple different types of relationships between nodes, then heterogeneous graph embedding techniques can be used like HERec [@shi_heterogeneous_2018] which is designed to handle heterogeneous graphs with multiple types of nodes and edges. HERec uses meta-path-based random walks to capture the complex relationships in heterogeneous graphs, allowing for more informative embeddings. The method employs a two-level embedding approach, where node embeddings are learned at both the node type level and the overall graph level.


When graph represents knowledge graph, where nodes represent entities and edges represent relationships between entities, then knowledge graph embedding techniques can be used like TransE [@bordes_translating_2013] which is designed to embed entities and relationships in a knowledge graph into a continuous vector space. TransE represents relationships as translations in the embedding space, allowing for efficient modeling of multi-relational data. The method optimizes an objective function that encourages the embeddings of related entities to be close together in the embedding space. Formally, for a triplet $(h, r, t)$ representing a head entity $h$, a relation $r$, and a tail entity $t$, TransE aims to satisfy the condition $f(h) + f(r) \approx f(t)$, where $f$ is the embedding function. It minimizes the margin-based ranking loss to ensure that valid triplets have lower energy than invalid ones.



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


## Knowledge graphs

Usually, in representing descriptive information about knowledge, where knowledge is represented as a set of entities and relationships between them, knowledge graphs are used. In a knowledge graph, nodes represent entities, and edges represent the relationships between them. Knowledge graphs are widely used in various applications, such as information retrieval, recommendation systems, and natural language processing. One of the most notable advantages of knowledge graphs is their ability to represent different types of relationships between entities.

One example of knowledge graph is Freebase [@bollacker_freebase_2008] which is a large-scale knowledge graph that contains millions of entities and relationships. Freebase was used in various applications, such as question answering and recommendation systems.

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
| Class name recommendation | Dependency graphs, Call graphs | HERec | [@kurimoto_class_2019] |
| Code weakness reasoning | Knowledge graphs | TransE | [@han_deepweak_2018] |
| Code search | Dependency graphs, call graphs | LINE | [@zou_graph_2018] |


## Design pattern detection

As pointed out in [@chatzigeorgiou_application_2006], design patterns are typical solutions to common problems in software design. They provide a way to reuse successful designs and architectures, making it easier to develop and maintain complex software systems. In mentioned paper, graph similarity was used to identify which design pattern was used. Similarly, embedding methods can map nodes to a vector space where node similarity can be measured using distance metrics like cosine similarity or Euclidean distance. This allows for efficient comparison of nodes and identification of similar patterns in the graph.

## Bug prediction

Bug prediction is a crucial task in software engineering that aims to identify potential defects in software systems before they manifest as actual bugs. By leveraging graph embeddings, it is possible to capture the structural and relational information of software components, enabling more accurate predictions of bug-prone areas in the codebase. In [@qu_node2defect_2018], the authors propose a method for predicting software defects using Node2Vec embeddings of dependency graphs. The approach involves constructing a dependency graph from the class dependencies in the codebase and then generating node embeddings using several state-of-the-art graph embedding techniques, including Node2Vec, LINE, and SDNE. The resulting embeddings are concatenated with traditional software metrics to form a comprehensive feature set for each class. These features are then used to train a machine learning model to predict the likelihood of defects in the classes. Models such as Random Forest, Logistic Regression, and Support Vector Machines are employed for the prediction task. 

## Class name recommendation

Class name recommendation is an important task in software engineering that aims to suggest meaningful and descriptive names for classes based on their functionality and relationships with other classes. In [@kurimoto_class_2019], the authors propose a method for recommending class names using HERec embeddings of heterogeneous graphs. The approach involves constructing a heterogeneous graph from the class dependencies and method calls in the codebase, where nodes represent classes and methods, and edges represent the relationships between them. The results show that the proposed method outperforms several baseline approaches like rule-based models. 



## Code weakness detection

CWE is database of software weaknesses that can lead to buggy behavior or security vulnerabilities. In [@han_deepweak_2018], the authors propose a method for embedding CWE entities in order to predict new relationships between entities. This means that if a certain weakness documented, it is possible to predict its consequences or related weaknesses even if they are not explicitly documented. The approach involves constructing a knowledge graph from the CWE database, where nodes represent weaknesses, consequences, and related weaknesses, and edges represent the relationships between them. The authors use TransE to generate embeddings for the entities in the knowledge graph. The resulting embeddings are then used to train a machine learning model to predict new relationships between entities. In their paper, authors combine the knowledge graph embedding and word embedding of textual descriptions of weaknesses to improve the quality of the embeddings. The results show that the proposed method outperforms several baseline approaches.

## Code search

Code search is an important task in software engineering that aims to retrieve relevant code snippets based on a given query. In [@zou_graph_2018], the authors propose a method for code search using LINE embeddings of dependency graphs and call graphs. The approach involves constructing a graph from the codebase, where nodes represent functions and classes, and edges represent the dependencies and calls between them. The authors use LINE to generate embeddings for the nodes in the graph. Evaluation was performed on Java code snippets and natural language queries. 

# Evolution of software projects

In [@bhattacharya_graph-based_2012], the authors analyzed software project evolution by analyzing graph topology over time for several open source projects. They analyzed code evolution as well as process-related artifacts like bug reports and version control commits. They found that software evolution exhibits certain patterns which can be used as predictors of future changes, defects and maintenance needs.

There are several datasets available for studying software project evolution like [@ahrabian_software_2020] which provides a collection of software project evolution datasets extracted from version control systems available as open source. The datasets include knowledge-graphs representing various relationships along with evaluation of several graph embedding techniques on them.


# Challenges and future directions


Currently, most of the embedding techniques are not scalable to large graphs which can limits their applicability to real-world enterprise software projects or whole organizations. 

Another challenge is the dynamic nature of software systems, which can change over time due to updates, bug fixes, and new feature additions. This requires embedding techniques to be able to adapt to these changes and incorporate new information into the embeddings. Reviewed papers rarely utilize dynamic graph embedding techniques even though software systems are inherently dynamic and snapshots of software at different points in time can be found in version control systems and represented as temporal graphs.


Also, there is a need for more comprehensive datasets and benchmarks to evaluate the performance of graph embedding techniques in software engineering tasks. This includes datasets that capture the complexity and diversity of real-world software systems, as well as benchmarks that allow for fair comparisons between different embedding techniques. This includes static and dynamic graphs representing software systems in different programming languages and paradigms.



# Conclusion

This paper presents a systematic review of graph embedding techniques with applications in software engineering. The review highlights the state-of-the-art methods, their effectiveness in various software engineering tasks, and the challenges that need to be addressed for further advancements in this field. The findings suggest that graph embeddings have the potential to significantly improve the software analysis and process improvements. However, there are still several challenges that need to be addressed, including scalability, adaptability to dynamic changes, and the need for more comprehensive datasets and benchmarks.

# References