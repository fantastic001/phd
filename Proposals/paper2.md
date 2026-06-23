---
title: Evaluation of downstream task of link prediction performance using HEART benchmarks on distributed node2vec embedding
---



# Background and Motivation

Given a graph, the node2vec algorithm learns continuous feature representations for nodes in the graph. The main idea behind node2vec is to use a biased random walk strategy to generate sequences of nodes, which are then used to train a Skip-gram model (similar to word2vec) to learn the node embeddings. The goal is to capture both the local and global structure of the graph in the learned embeddings, allowing for tasks such as node classification, link prediction, and clustering.

Motivation for this research stems from the need to evaluate the performance of node2vec embeddings in downstream tasks, specifically link prediction, using the HEART benchmarks. The HEART benchmarks provide a standardized way to assess the quality of embeddings in various graph-based tasks, and understanding how distributed node2vec embeddings perform in this context is crucial for improving graph representation learning techniques.

We distribute graph using simple partitioning based on label propagation, but we also investigate LFM partitioning strategy to see if it improves the performance of link prediction task. 



# Research objective

The objective of this research is to investigate the effect of different graph partitioning strategies on the performance of distributed node2vec embeddings in link prediction tasks, using the HEART benchmarks as the evaluation framework.

# Research Questions

1. How do different graph partitioning strategies affect the quality of node2vec embeddings in distributed settings on link prediction tasks?
2. Does the LFM partitioning strategy improve the performance of link prediction tasks compared to simple label propagation-based partitioning?
3. What are the trade-offs between partitioning strategies in terms of embedding quality and computational efficiency?
4. What models for link prediction can be used when embeddings are distributed across multiple partitions, and how do they perform in terms of accuracy and efficiency?

# Scope and Assumptions

- The research will focus on evaluating the performance of distributed node2vec embeddings in link prediction tasks using the HEART benchmarks. 
- The study will compare the performance of embeddings generated using simple label propagation-based partitioning and LFM partitioning strategies.
- The evaluation will be conducted on a variety of graph datasets to ensure generalizability of the findings.


# Proposed Methods

## Graph Partitioning Strategies

- Simple Label Propagation-based Partitioning: We will implement a straightforward label propagation algorithm to partition the graph into K subgraphs. K is the number of partitions, which corresponds to the number of available computing nodes in the distributed environment. Each partition will be processed independently to generate node embeddings.
- LFM Partitioning Strategy: We will evaluate the performance of the LFM partitioning strategy which is combined with bin packing to ensure exact number of needed partitions (computing nodes). 

## Node2vec Embedding Generation

- Distributed Node2vec: We will implement a distributed version of the node2vec algorithm that can operate on the partitioned subgraphs. Each partition will generate its own set of node embeddings, which will then be combined to form a global embedding representation for the entire graph.

## Link Prediction Models

We will explore various link prediction models that can utilize the distributed node2vec embeddings. These models may include:
- Logistic Regression: A simple yet effective model for binary classification tasks, which can be used to predict the existence of links between nodes based on their embeddings.
- Random Forest: An ensemble learning method that can capture complex relationships in the data and may improve link prediction performance.
- Neural network-based models: We will train neural network $f(u,v) = \sigma(h(u)^T W h(v))$ where $h(u)$ and $h(v)$ are the embeddings of nodes u and v, respectively, and W is a learnable weight matrix. The output will be passed through a sigmoid activation function to predict the probability of a link existing between the two nodes.

# Experimental Design

## Evaluation Metrics

We will use HEART benchmark to evaluate the performance of the link prediction models. 

For every positive edge in the test set, we will generate N negative edges by randomly sampling pairs of nodes that are not connected by an edge. The evaluation metrics will include:
- MRR: Mean Reciprocal Rank, given by $MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$, where $Q$ is the set of positive edges in the test set, and $rank_i$ is the rank of the positive edge among the N negative edges.
- Hits@K: The proportion of positive edges that are ranked in the top K among the N negative edges. This metric will be computed for various values of K (e.g., K=1, 3, 5, 10) to assess the model's ability to rank positive edges highly.
- Accuracy: The overall accuracy of the link prediction model, calculated as the ratio of correctly predicted edges (both positive and negative) to the total number of edges in the test set.

## Experimental Procedure

We will conduct experiments on multiple graph datasets:
- CITESEER
- DBLP
- AS-Oregon 
- AstroPh

as well as some synthetic datasets generated using the stochastic block model (SBM) to evaluate the performance of distributed node2vec embeddings in link prediction tasks.

## Expected Outcomes

- We expect to observe differences in link prediction performance based on the graph partitioning strategy used. Specifically, we hypothesize that performance will decrease as the number of partitions increases, due to the loss of global graph structure information. However, this decrease may be worth the trade-off in terms of computational efficiency and scalability.

## Limitations

- The study will be limited to the evaluation of distributed node2vec embeddings in link prediction tasks using the HEART benchmarks. Other downstream tasks, such as node classification or clustering, will not be considered in this research.
- The performance of the link prediction models may be influenced by the choice of hyperparameters, which will not be exhaustively explored in this study. Instead, we will use a set of default hyperparameters for the node2vec algorithm and link prediction models.
- The study will focus on a limited number of graph datasets, which may not fully capture the diversity of real-world graphs. The findings may not generalize to all types of graphs or applications.

# Deliverables

- A distributed implementation of the node2vec algorithm that can operate on partitioned subgraphs.
- A comprehensive evaluation of the performance of distributed node2vec embeddings in link prediction tasks using the HEART benchmarks, comparing different graph partitioning strategies.
- A set of recommendations for practitioners on the choice of graph partitioning strategies and link prediction models based on the findings of this research.