# Coreset
Work done in this repository is for my Master's thesis titled "Comparative Lightweight Corset Construction for K-Means Problem."
In this paper, we will create a coresets in the context of the k-means problem. In
particular, we will discuss techniques of coreset construction such as geometric
decomposition and random sampling from the works of [[1]](https://link.springer.com/article/10.1007/s13218-017-0519-3). The aim of the study
is to present theoretical results and to implement the techniques of building
coresets comparative to lightweight coresets which were made by the authors of
[[2]](https://arxiv.org/abs/1702.08248).

## Algorithms Implemented
### Lightweight Coreset Construction
Scalable k-Means Clustering via Lightweight Coresets [[2]](https://arxiv.org/abs/1702.08248)

### Geometric Decomposition
Farthest Point Algorithm by [[3]](https://www.sciencedirect.com/science/article/pii/0304397585902245) 

Fast Constant Factor Approximation by [[4]](https://arxiv.org/abs/1810.12826)

## References
[1] Munteanu, A., Schwiegelshohn, C. Coresets-Methods and History: A Theoreticians Design Pattern for Approximation and Streaming Algorithms. Künstl Intell 32, 37–53 (2018). https://doi.org/10.1007/s13218-017-0519-3

[2] Olivier Bachem, Mario Lucic, and Andreas Krause. 2018. Scalable k -Means Clustering via Lightweight Coresets. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). Association for Computing Machinery, New York, NY, USA, 1119–1127. DOI:https://doi.org/10.1145/3219819.3219973

[3] Gonzalez, Teofilo F.. “Clustering to Minimize the Maximum Intercluster Distance.” Theor. Comput. Sci. 38 (1985): 293-306.

[4] Sariel Har-Peled and Soham Mazumdar. 2004. On coresets for k-means and k-median clustering. In Proceedings of the thirty-sixth annual ACM symposium on Theory of computing (STOC '04). Association for Computing Machinery, New York, NY, USA, 291–300. DOI:https://doi.org/10.1145/1007352.1007400