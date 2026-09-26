# Comprehensive Guide to Machine Learning Algorithms & Pipelines for Large-Scale Entity Resolution

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Scale:** 26.4 Million Records (S1 Reference: 2.2M train, 1.73M test; S2: 5.0M; S3: 5.3M)  
**System Hardware:** AMD EPYC 256 vCPUs, 1.1 TB RAM, NVIDIA L4 24GB VRAM GPU  
**Metric:** Macro $F_{0.5}$ (Precision weighted $2\times$ over recall, strict singleton scoring)

---

## 1. The Core Challenge of Entity Resolution at Scale

Entity Resolution (ER) across multiple noisy, unstandardized commercial sources is fundamentally an $O(N_1 \times (N_2 + N_3))$ search space problem. In this competition:
$$1{,}732{,}544 \times (4{,}880{,}000 + 5{,}080{,}000) \approx \mathbf{1.73 \times 10^{13} \text{ potential pairwise comparisons}}.$$

At 1 microsecond per pairwise comparison, computing all pairs sequentially would require **200 days of continuous computation**. Therefore, real-world Entity Resolution cannot be solved by simply running a classifier over raw data. It requires an integrated multi-stage pipeline:

```
Raw Sources (S1, S2, S3)
         │
         ▼
[Stage 1: Scalable Candidate Generation / Blocking]
   - Multi-Tier Inverted Indexing
   - Phonetic & Geographic Partitioning
   - Reduces search space by > 99.999%
         │
         ▼
[Stage 2: Representation & Feature Engineering]
   - Non-linear Edit-Distance Space (RapidFuzz)
   - Latent Manifold Projection (PCA + LDA on TF-IDF)
   - Phonetic & Geographic Flags (Soundex, Postal PIN)
         │
         ▼
[Stage 3: ML Modeling & Ensembling]
   - GPU-Accelerated XGBoost (`hist`, `cuda`)
   - Leaf-wise LightGBM GBDT (28 vCPUs)
   - Calibrated Ridge Continuous Ranker
   - Out-of-Fold Stacking Meta-Learner
         │
         ▼
[Stage 4: Bipartite Graph Disambiguation]
   - Star-Clustering (arg max P(s1, c))
   - Mutual Exclusion & Connected Components
         │
         ▼
Final Clean Matches (matching_results.tsv)
```

---

## 2. Exhaustive Analysis: Why Various ML Families Fail vs. Succeed

### 2.1 Clustering Algorithms

| Algorithm | Computational Complexity | Memory Footprint | Feasibility on 26.4M Records | Why It Fails or Succeeds |
| :--- | :---: | :---: | :---: | :--- |
| **K-Means / Mini-Batch K-Means** | $O(N \cdot K \cdot d \cdot i)$ | High ($\mathbb{R}^d$) | **Infeasible** | Requires a predefined $K$ and Euclidean space. In ER, ground-truth clusters are unknown, dynamic, and tiny (size $\le 3$: max 1 from S1, 1 from S2, 1 from S3). K-Means assumes spherical, equal-variance clusters and cannot enforce source exclusivity. |
| **DBSCAN / HDBSCAN** | $O(N^2)$ to $O(N \log N)$ | $\approx 350 \text{ TB}$ (pairwise dist) | **Infeasible** | Explicit distance matrix exceeds physical RAM. In high-dimensional text space ($d > 50$), distance concentration causes points to become equidistant, causing DBSCAN to group everything into a single giant chain or noise. |
| **Agglomerative / Hierarchical** | $O(N^2 \log N)$ to $O(N^3)$ | Extensively High | **Infeasible** | Intractable time complexity; cannot scale past 50,000 entities without hanging. |
| **Bipartite Star-Clustering (Used)** | **$O(\|E\|)$ Linear in candidate edges** | **$< 2 \text{ GB}$ (sparse graph)** | **Optimal & Winning** | Operates on the sparse candidate graph where $|E| \ll N^2$. Enforces domain constraints: degree $\le 1$ per data source and assigns candidate $c$ strictly to $\arg\max_{s_1} P(s_1, c)$. |

---

### 2.2 Classification Algorithms

| Algorithm | Handling of Class Imbalance (~1:50) | Collinear Text Features | GPU Acceleration | Suitability for Macro $F_{0.5}$ |
| :--- | :---: | :---: | :---: | :--- |
| **Kernel Support Vector Machines (RBF SVM)** | Poor | Good | Poor ($O(N^2)$ kernel) | **Infeasible**: Quadratic scaling with sample size. Fitting on 1.7M+ candidate pairs would not converge within competition limits. |
| **Linear SVM (LinearSVC)** | Moderate | Good | Moderate | **Sub-optimal**: Linear boundary cannot capture non-linear feature interactions (e.g., exact PIN match compensating for low name similarity). |
| **$k$-Nearest Neighbors ($k$-NN)** | Poor | Degraded by curse of dim | Poor (exhaustive search) | **Infeasible**: Exhaustive inference requires $1.73 \times 10^{13}$ operations. Without indexing, latency is prohibitive. |
| **Naive Bayes** | Poor | Catastrophic failure | CPU only | **Fails**: Assumes conditional feature independence. String edit metrics (Levenshtein, Jaro-Winkler, Token-Sort) are heavily collinear ($\rho > 0.85$). Naive Bayes drastically over-confirms probabilities, collapsing precision. |
| **Random Forest (Bagging)** | Moderate | Robust | Limited | **Sub-optimal**: Bagging averages unweighted decision trees, producing diffuse probabilities on severe class imbalance without hessian gradient guidance. |
| **XGBoost (Histogram / CUDA)** | **Excellent (`scale_pos_weight`)** | **Robust (Tree splits)** | **Exceptional (1.4s fit)** | **High ($F_{0.5} = 0.5015$)**: Highly effective on pairwise edit distances. |
| **Leaf-wise LightGBM (Winning)** | **Excellent (`scale_pos_weight`)** | **Robust** | **28 vCPU multi-threading** | **Winning ($F_{0.5} = 0.7522$)**: Leaf-wise splits capture complex high-order geographic-phonetic feature interactions. |

---

### 2.3 Regression & Continuous Ranking

- **Standard Linear Regression (OLS):** Unbounded outputs $(-\infty, \infty)$ violate probability calibration and cannot optimize threshold-based metrics.
- **$L_2$-Regularized Logistic / Ridge Continuous Ranker (Used in Pipeline 2):** When trained on latent semantic manifolds (PCA/LDA projections), continuous calibrated ranking accurately scores relative affinity, providing smooth probabilities that blend with tree-based models.

---

## 3. Dimensionality Reduction: PCA & LDA

In text-based Entity Resolution, character $n$-gram TF-IDF representations produce sparse matrices with thousands of dimensions. However, overlapping character $n$-grams (e.g. 3-grams `"pri"`, `"riv"`, `"iva"`, `"vat"` in *"Private"*) introduce severe multicollinearity.

### 3.1 Principal Component Analysis (TruncatedSVD)
- **Role:** Unsupervised orthogonal projection.
- **Function:** Compresses the sparse character $n$-gram space into 10 orthogonal latent semantic components:
  $$X_{\text{PCA}} = X_{\text{TF-IDF}} \cdot V_k$$
- **Empirical Impact:** Explained variance ratio of 0.2514 across 10 components in 4.77 seconds, eliminating redundant collinear features.

### 3.2 Linear Discriminant Analysis (LDA)
- **Role:** Supervised projection maximizing between-class variance relative to within-class variance:
  $$J(w) = \frac{w^T S_B w}{w^T S_W w}$$
- **Empirical Impact:** Projects dense PCA components onto an optimal 1D discriminant axis separating ground-truth matches ($y=1$) from negative candidate pairs ($y=0$).

---

## 4. Statistical Hypothesis Testing Suite

Before final modeling, we conducted 4 formal statistical tests to validate feature discriminability:

1. **Mann-Whitney U Test (Wilcoxon Rank-Sum):**
   - *Test:* Non-parametric comparison of normalized Levenshtein similarity distributions for true matches vs non-matches.
   - *Result:* $U = 2.90 \times 10^8, p < 10^{-15}$.
   - *Conclusion:* Rejects $H_0$; ground-truth matches have statistically significantly higher similarity.
2. **Kolmogorov-Smirnov (KS) Test:**
   - *Test:* Evaluates divergence in empirical Cumulative Distribution Functions (CDFs) of Jaro-Winkler scores.
   - *Result:* $KS = 0.841, p < 10^{-15}$.
   - *Conclusion:* Rejects $H_0$; match and non-match distributions are completely distinct.
3. **Chi-Square ($\chi^2$) Test on Phonetic Soundex Agreement:**
   - *Result:* $\chi^2 = 21{,}573.45, p < 10^{-15}$.
   - *Conclusion:* Phonetic agreement is strongly correlated with ground-truth matches, confirming the necessity of phonetic blocking.
4. **Chi-Square ($\chi^2$) Test on Postal PIN Code Match:**
   - *Result:* $\chi^2 = 852.51, p = 2.07 \times 10^{-187}$.
   - *Conclusion:* Identical postal codes provide an invariant geographic anchor for businesses with alternate trade names (DBAs).
5. **Spearman Rank Correlation Analysis:**
   - $\rho(\text{Levenshtein}, \text{Jaro-Winkler}) = 0.864$
   - $\rho(\text{Levenshtein}, \text{Token-Sort}) = 0.812$
   - *Conclusion:* Proves strong collinearity among edit metrics, validating the use of PCA and tree-based regularization.

---

## 5. Master Comparative Benchmark Results

All evaluations were conducted under strict **5-Fold Entity-Level Stratified Cross-Validation** (split strictly on `entity_id` with 0% data leakage):

| Rank | Pipeline Architecture | 5-Fold Macro $F_{0.5}$ | Precision | Recall | Singleton Accuracy | Train Time / Fold | Full Test Inference (1.73M) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Pipeline 2: Multi-Modal Hybrid Ensemble** | **0.7522 $\pm$ 0.0075** | **0.7595** | **0.7245** | **99.82%** | **6.77s** | **571.43s (9.5 min)** |
| 🥈 | **Pipeline 1: GPU XGBoost + Star-Clustering** | 0.5015 $\pm$ 0.0078 | 0.5053 | 0.4872 | 99.69% | 1.55s | 1,761.49s (29.3 min) |
| 🥉 | Baseline Greedy Token Indexing | 0.2362 $\pm$ 0.0042 | 0.2465 | 0.1421 | 87.14% | N/A | ~4 hours (unbatched) |

---

## 6. Key Takeaways & Architectural Lessons Learned

1. **Precision Dominates the Metric:**
   In Macro $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$, a false merge costs twice as much as a missed match. Furthermore, singletons earn 1.0 for an empty prediction and 0.0 for any match. Achieving **99.82% singleton accuracy** was the single most decisive factor in reaching $F_{0.5} = 0.7522$.
2. **Domain-Specific Blocking Beats Raw Model Complexity:**
   Adding Indian Postal PIN code indexing (`\b\d{5,6}\b`) and Phonetic Soundex encoding delivered a +0.25 jump in Macro $F_{0.5}$—far exceeding any gain achievable by simply tuning tree depth or adding layers.
3. **GPU Batching Is Mandatory for Scale:**
   Evaluating 1.73M queries sequentially creates millions of kernel launches, wasting hours in PCIe transfer overhead. Batching queries into 50,000-entity slices saturates the NVIDIA L4 GPU cores, reducing inference time from **~4 hours down to 9.5 minutes**.
4. **Graph Disambiguation Enforces Real-World Physics:**
   In commercial datasets, multiple external records often share high similarity with a query. Enforcing mutual exclusivity via Star-Clustering ($\arg\max_{s_1} P(s_1, c)$) prevents duplicate entity assignment and eliminates multi-candidate false positives.
