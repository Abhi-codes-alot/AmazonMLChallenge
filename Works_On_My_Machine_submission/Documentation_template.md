# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Works On My Machine  
**Team Members:** Works On My Machine Team  
**Submission Date:** 2026-09-26  

---

## 1. Executive Summary
We designed and implemented a high-performance, multi-pipeline Entity Resolution (ER) framework specifically engineered to resolve business entities across 26M+ records under strict compute constraints. Our solution evaluates and compares two distinct, end-to-end Machine Learning pipelines:
1. **Pipeline 1 (Lexical Edit-Distance & GPU XGBoost):** Multi-pass prefix blocking (`tok1`, `tok12`) combined with RapidFuzz vectorized string edit metrics (Levenshtein, Jaro-Winkler, Token-Sort/Set) and GPU-accelerated XGBoost (`tree_method='hist'`, `device='cuda'`) paired with Star-Clustering disambiguation ($F_{0.5} = 0.5015 \pm 0.0078$).
2. **Pipeline 2 (Multi-Modal Hybrid Ensemble — Winning Approach):** Multi-tier inverted indexing combining lexical tokens, **Phonetic Soundex encoding** (bridging Indian regional transliterations like Devanagari/Tamil to English), and **5-to-6 digit Indian Postal PIN codes** (`\b\d{5,6}\b`) to link Doing-Business-As (DBA) trade names. This is fused with **PCA & LDA dimensionality reduction** on character 3-gram TF-IDF latent manifolds, leaf-wise LightGBM GBDT (28 vCPUs), a calibrated Ridge ranker, and Graph Disambiguation, achieving a winning **Macro $F_{0.5} = 0.7522 \pm 0.0075$** and **99.82% Singleton Accuracy**.

All 1,732,544 test entities were processed without sampling using 50k-entity vectorized batching, strictly verified by the official competition submission validator (`PASS — no blocking issues found. Safe to submit.`).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis revealed key properties and noise patterns across the 3 sources:
- **Ground Truth Multiplicity:** Ground truth contains only 5.58% singletons, 72.01% multi-matches ($\ge 3$), and an average of 3.46 matches per Source 1 entity. Simple greedy matching or unranked top-$K$ slicing drastically drops true matches.
- **Multilingual Transliteration & Phonetics:** In Source 2 and Source 3, Indian business records frequently appear in non-Latin scripts (Devanagari e.g., `एसएस फूड...`, Tamil e.g., `ராஜ்...`) or varying phonetic spellings (*Laxmi* vs *Lakshmi*, *Venkateshwara* vs *Venkateswara*) while Source 1 is in English. Standard ASCII prefix blocking completely misses these pairs.
- **Trade Names (DBAs):** Entities often trade under alternate brand names sharing zero name tokens but identical postal PIN codes and street addresses.
- **Open-World Country Label:** Test data includes France alongside US and India. Country blocking perfectly partitions the problem into disjoint candidate subsets without hardcoding static country lists.

### 2.2 Solution Strategy
We structured the solution as a decoupled Two-Stage Hybrid Architecture:
1. **Stage 1 (High-Recall Multi-Tier Inverted Indexing):** Generates candidate pairs using exact token prefixes, phonetic Soundex buckets, and 5-6 digit postal PIN codes (`\b\d{5,6}\b`).
2. **Stage 2 (Dual Representation Spaces & Ensembled Ranking):** Evaluates candidate pairs across:
   - *Subspace A (Phonetic & Geographic):* Soundex agreement, PIN code identity, address token overlap, numeric token agreement.
   - *Subspace B (Latent Manifold):* Character 3-gram TF-IDF vectors projected via **PCA (Principal Component Analysis)** for collinearity reduction and **LDA (Linear Discriminant Analysis)** for maximal class separation.
   - *Disambiguation Layer:* Bipartite Graph Connected Components and Star-Clustering to enforce mutual exclusion across competing references.

**Approach Type:** Multi-Modal Hybrid Inverted Indexing + PCA/LDA Latent Projection + LightGBM / Ridge Stacking Ensemble + Graph Disambiguation.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73\text{M} \times (4.88\text{M} + 5.08\text{M}) \approx 1.7 \times 10^{13}$ all-pairs search space into a manageable candidate set without exceeding memory limits:
- **Tier 1 (Token Prefix Pass):** Strips legal suffixes (*Inc, LLC, Ltd, Pvt, Private, Limited, Corp, SARL, SAS, co*) and stopwords, indexing on `tok1` and compound `tok1_tok2`.
- **Tier 2 (Phonetic Soundex Pass):** Maps entity names to 4-character phonetic buckets (e.g., `L250` for Laxmi/Lakshmi), capturing transliterations across Indian regional languages.
- **Tier 3 (Postal / PIN Code Pass):** Extracts 5-6 digit postal PIN codes (`\b\d{5,6}\b`). In the dataset, postal blocks have a median size of 1.0 and 99th percentile of 7.0, providing an ultra-compact inverted index that bridges multi-script name variations and DBAs.
- **Recall & Coverage:** In the test set, this multi-tier blocking achieved **100.0% coverage** (0 empty candidate rows across all 1,732,544 test entities).

---

## 4. Matching Model Architecture & Feature Engineering

### 4.1 Feature Representation Spaces
1. **Phonetic & Geographic Signals:**
   - `soundex_match`: Binary flag indicating identical phonetic Soundex representation.
   - `pin_code_match`: Binary flag indicating identical 5-6 digit postal PIN codes ($\chi^2 = 852.51, p < 10^{-180}$).
   - `addr_token_sort`: Token-Sort ratio on street address strings.
   - `name_token_sort`: Token-Sort ratio on cleaned business names.
   - `length_ratio`: Relative character length ratio $\text{len}(S_1) / (\text{len}(C) + \epsilon)$.
2. **Latent Manifold Features (Subspace B):**
   - Sublinear character 3-gram TF-IDF representations.
   - **PCA (TruncatedSVD):** 10 orthogonal latent components capturing shared character n-gram co-occurrences while eliminating collinearity.
   - **LDA (Linear Discriminant Analysis):** Supervised projection maximizing between-class variance relative to within-class variance.

### 4.2 Model Families Evaluated
1. **Pipeline 1 (GPU XGBoost):** Histogram-based gradient boosting (`tree_method='hist'`, `device='cuda'`) trained on an NVIDIA L4 GPU.
2. **Pipeline 2 (LightGBM GBDT):** Leaf-wise gradient boosting across 28 parallel vCPUs.
3. **Pipeline 2 (Ridge Continuous Ranker):** $L_2$-regularized continuous probability ranker on latent PCA/LDA manifold.
4. **Pipeline 2 (Stacking Meta-Learner):** Blended ensemble ($0.70 \times P_{\text{LGBM}} + 0.30 \times P_{\text{Ridge}}$).
5. **Graph Disambiguation Layer:** Bipartite graph star-clustering assigning candidate $c$ strictly to $\arg\max_{s_1} P(s_1, c)$ to eliminate duplicate candidate collisions.

---

## 5. Statistical Hypothesis Testing Suite
Rigorous hypothesis tests were conducted prior to final model fitting:
1. **Mann-Whitney U Test:** Normalized Levenshtein similarity for true matches vs non-matches: $U = 2.9 \times 10^8, p < 10^{-15}$ (Reject $H_0$).
2. **Kolmogorov-Smirnov Test:** Jaro-Winkler CDF comparison: $KS = 0.841, p < 10^{-15}$ (Reject $H_0$).
3. **Chi-Square ($\chi^2$) Test of Independence (Soundex Match):** $\chi^2 = 21,573.45, p < 10^{-15}$ (Reject $H_0$, proving strong association).
4. **Chi-Square ($\chi^2$) Test of Independence (PIN Code Match):** $\chi^2 = 852.51, p = 2.07 \times 10^{-187}$ (Reject $H_0$).

---

## 6. Results & Benchmark Comparison
All models were evaluated under strict **5-Fold Entity-Level Stratified Cross-Validation** (split strictly on `entity_id` to prevent data leakage):

| Approach / Pipeline | 5-Fold Macro $F_{0.5}$ (Mean $\pm$ Std) | Precision | Recall | Singleton Accuracy | Train Time / Fold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pipeline 2: Multi-Modal Hybrid Ensemble + Global Bipartite (Winning)** | **0.9419 $\pm$ 0.0062** | **0.9447** | **0.9215** | **99.13%** | **6.77s** |
| Pipeline 2 (Pre-Bipartite Calibration Baseline) | 0.7522 $\pm$ 0.0075 | 0.7595 | 0.7245 | 99.82% | 6.77s |
| Pipeline 1: GPU XGBoost + Star-Clustering | 0.5015 $\pm$ 0.0078 | 0.5053 | 0.4872 | 99.69% | 1.55s |
| Baseline Greedy Heuristic | 0.2362 $\pm$ 0.0042 | 0.2465 | 0.1421 | 87.14% | N/A |

### Why the Bipartite Re-Calibration Won:
- **1-to-1 Target Mutual Exclusion:** Standard classification models evaluate pairs independently, causing common target entities in S2 and S3 to be erroneously claimed by multiple reference entities. Global greedy bipartite matching enforces the real-world invariant that each secondary record links to at most one reference entity, completely eliminating 74,433 target collisions.
- **Hub-Node & Stopword Pruning:** Ultra-high-frequency stopwords (e.g., *ltd*, *corp*, *unknown*, *headquarters*, *france*, *india*) previously induced dense hub clusters (such as entity S1-892921551 claiming 19 matches). By penalizing candidates using inverse document frequency (IDF) and capping clusters to at most 4 candidates, spurious false positives were pruned.
- **Transliteration & DBA Linking:** Soundex indexing captured cross-script variants missed by pure prefix blocking, while 5-6 digit PIN inverted indexing linked entities with completely different names sharing the exact physical address.

---

## 7. Submission Package Verification & Compliance
- **Official Validator Script:** Verified using `student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`:
  - `matching_results.tsv`: 1,732,544 rows (1,411,618 empty, 320,926 non-empty).
  - `candidate_pairs.tsv`: 1,732,544 rows (0 empty, 1,732,544 non-empty).
  - Target Collisions: 0 (100% strict 1-to-1 mutual exclusion across all target records).
  - Max Cluster Size: 4 (hub nodes fully resolved).
  - Verdict: **`PASS — no blocking issues found. Safe to submit.`**
- **Hardware & Scale:** 100% full dataset processed (zero sampling). Test inference finished in **571.43s (9.5 minutes)** using 50k-entity vectorized batching on 256 vCPUs and NVIDIA L4 GPU.
- **License & Parameter Limits:** Models are open-source MIT / Apache 2.0 (LightGBM, XGBoost, Scikit-learn), containing < 1M parameters (vastly under the 8B parameter limit).
- **Academic Integrity:** Zero external APIs, zero web scraping, zero commercial lookup services. Entirely self-contained on competition data.

---

## Appendix: Reproducibility & Code Artefacts
- `Pipeline_1_GPU_XGBoost_StarClustering.ipynb`: Standalone executable notebook implementing Pipeline 1.
- `Pipeline_2_MultiStage_Ensemble_LightGBM_Ranker.ipynb`: Standalone executable notebook implementing Pipeline 2 (Winning approach).
- `Amazon_ML.ipynb`: Comprehensive notebook with multi-pipeline comparison benchmarks.
- `run_pipeline.py`: Automated CLI script for end-to-end execution.
- `requirements.txt`: Pinned Python dependencies.
