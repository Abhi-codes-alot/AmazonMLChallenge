# Deep Dive: The Winning Multi-Modal Hybrid Pipeline (Pipeline 2)

**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Winning Architecture:** Multi-Tier Inverted Indexing (Token + Phonetic + PIN) + PCA/LDA Latent Manifold + Leaf-wise LightGBM & Calibrated Ridge Ranker + Graph Disambiguation  
**Validated Score:** 5-Fold Cross-Validated Macro $F_{0.5} = \mathbf{0.7522 \pm 0.0075}$  
**Singleton Accuracy:** $\mathbf{99.82\%}$  
**Test Scale:** 100% Full Dataset Evaluated (**1,732,544 test entities** in 571.43s)  
**Submission Status:** Fully verified by `validate_submission.py` (`PASS — safe to submit`)

---

## 1. Executive Summary & Why Pipeline 2 Won

While standard Entity Resolution pipelines rely exclusively on lexical string distances (Levenshtein, Jaro-Winkler) over token prefixes, real-world commercial entity data exhibits two severe noise patterns:
1. **Cross-Script Transliterations:** Indian business entities in Source 2 and Source 3 frequently appear in regional scripts (Devanagari, Tamil, Marathi) or phonetic transliterations into English (*Laxmi* vs *Lakshmi*, *Venkateshwara* vs *Venkateswara*). Exact token matching fails to index them.
2. **Doing-Business-As (DBA) Trade Names:** Enterprises frequently operate under trade names sharing zero name tokens with their corporate reference, but sharing the exact physical establishment and Postal PIN code.

**Pipeline 2 overcomes both obstacles** by uniting:
- **Phonetic Soundex Encoding:** Mapping phonetic equivalents into invariant 4-character phonetic buckets ($\chi^2 = 21{,}573.45, p < 10^{-15}$).
- **5-to-6 Digit Indian Postal PIN Code Indexing (`\b\d{5,6}\b`):** Linking DBAs sharing identical geographic premises ($\chi^2 = 852.51, p = 2.07 \times 10^{-187}$).
- **PCA & LDA Latent Projection:** Compressing high-dimensional character 3-gram TF-IDF representations into an orthogonal 10-component manifold to eliminate collinearity.
- **Ensemble Fusion:** Blending leaf-wise LightGBM GBDT (0.70) with a continuous calibrated Ridge ranker (0.30) to achieve **99.82% Singleton Accuracy**.

---

## 2. End-to-End Architectural Dataflow

```
                             Source 1, Source 2, Source 3 Raw TSVs
                                               │
                                               ▼
               ┌──────────────────────────────────────────────────────────────┐
               │         Multi-Tier High-Recall Inverted Indexing             │
               │  Tier 1: Cleaned Token Prefix (tok1[:4])                    │
               │  Tier 2: Phonetic Soundex Buckets (soundex(name))            │
               │  Tier 3: 5-to-6 Digit Postal PIN Inverted Index (\b\d{5,6}\b)│
               └──────────────────────────────┬───────────────────────────────┘
                                              │
                    Candidate Pools (Mean: 29.97 cands / S1 entity)
                    [Search Space Reduction Ratio: 99.999699%]
                                              │
                                              ▼
               ┌──────────────────────────────────────────────────────────────┐
               │                 Dual Feature Representation                  │
               ├──────────────────────────────┬───────────────────────────────┤
               │   Subspace A: Phonetic/Geo   │  Subspace B: Latent Manifold  │
               │  - soundex_match (binary)    │  - Char 3-gram TF-IDF         │
               │  - pin_code_match (binary)   │  - PCA (TruncatedSVD, k=10)   │
               │  - name_token_sort (float)   │  - LDA (1D Discriminant Axis) │
               │  - addr_token_sort (float)   │                               │
               │  - length_ratio (float)      │                               │
               └──────────────┬───────────────┴──────────────┬────────────────┘
                              │                              │
                              ▼                              ▼
                 [Leaf-Wise LightGBM GBDT]      [Calibrated Ridge Ranker]
                    (28 parallel vCPUs)             (L2 Regularization)
                     P_LGBM Probability             P_Ridge Probability
                              │                              │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                             [Stacking Blended Meta-Learner]
                           P_Blend = 0.70*P_LGBM + 0.30*P_Ridge
                                             │
                                             ▼
                           [Optimal Macro F0.5 Thresholding]
                                       (tau = 0.55)
                                             │
                                             ▼
                          [Bipartite Graph Disambiguation]
                          Star-Clustering: arg max P(s1, c)
                                             │
                                             ▼
                                Final Verified Submission
                         matching_results.tsv (1,732,544 rows)
                         candidate_pairs.tsv  (1,732,544 rows)
```

---

## 3. Candidate Generation & Scaling Efficiency

A key mandate of the Amazon ML Challenge is candidate generation efficiency:
> *"The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard."*

Pipeline 2 achieves optimal candidate efficiency:

| Metric | Result | Analysis |
| :--- | :---: | :--- |
| **Total Test Entities** | **1,732,544** | 100% of required Source 1 entities |
| **Total Candidate Pairs Generated** | **51,917,630** | Reduced from $1.73 \times 10^{13}$ all-pairs |
| **Search Space Reduction Ratio** | **99.999699%** | **333,000× search space compression** |
| **Mean Candidates per Entity** | **29.97 candidates** | Ultra-compact, well below the 50 candidate bloat threshold |
| **Median Candidates per Entity** | **30.0 candidates** | Uniform candidate distribution across dataset |
| **Max Candidates per Entity** | **30 candidates** | Strict cap preventing memory spikes |
| **Candidate Coverage on Test Set** | **100.00% (0 empty rows)** | Zero dropped entities; every S1 has plausible candidates |

---

## 4. Dual Feature Spaces & Dimensionality Reduction

### 4.1 Subspace A: Phonetic & Geographic Structural Features
1. **Phonetic Soundex Matching (`soundex_match`):**
   Converts names into 4-character phonetic representations (e.g. `L250` for *Laxmi*, *Lakshmi*, *Laxmee*).
   - $\chi^2 = 21{,}573.45$ ($p < 10^{-15}$), proving extreme statistical dependency.
2. **Postal PIN Code Matching (`pin_code_match`):**
   Extracts 5-6 digit Indian PIN codes (`\b\d{5,6}\b`) and flags identical geographic premises.
   - $\chi^2 = 852.51$ ($p = 2.07 \times 10^{-187}$).
3. **Token-Sort Similarities (`name_token_sort`, `addr_token_sort`):**
   Order-invariant string similarities computed via RapidFuzz C++ core.
4. **Length Ratio (`length_ratio`):**
   $\text{len}(S_1) / (\text{len}(C) + 10^{-5})$ to penalize extreme length asymmetries.

### 4.2 Subspace B: Latent Text Manifold (PCA + LDA)
Character $n$-gram TF-IDF matrices suffer from high collinearity across overlapping substrings. To resolve this:
1. **TF-IDF Vectorization:** Sublinear character 3-gram frequencies over `[business_name] [business_address]`.
2. **Principal Component Analysis (TruncatedSVD):**
   Projects sparse TF-IDF vectors into 10 orthogonal latent semantic dimensions in 4.77s:
   $$X_{\text{PCA}} = X_{\text{TF-IDF}} \cdot V_{10}$$
   Retains the dominant variance while discarding high-frequency character noise.
3. **Linear Discriminant Analysis (LDA):**
   Supervised 1D projection maximizing the Fisher criterion:
   $$J(w) = \frac{w^T S_B w}{w^T S_W w}$$
   This produces a single, highly separated feature axis isolating true matches from false candidates.

---

## 5. Machine Learning Models & Cross-Validation Results

### 5.1 Estimator Configuration
1. **Leaf-Wise LightGBM GBDT (Model A):**
   - `objective`: `'binary'`, `metric`: `'binary_logloss'`
   - `num_leaves`: 63, `learning_rate`: 0.08, `feature_fraction`: 0.85
   - `scale_pos_weight`: 20.0 (compensating for ~1:25 candidate imbalance)
   - `n_jobs`: 28 parallel worker threads on AMD EPYC cores
2. **Regularized Ridge / Logistic Continuous Ranker (Model B):**
   - $L_2$ regularization penalty ($C = 1.0$) with balanced class weighting.
   - Continuous Platt scaling calibration for smooth probability outputs.
3. **Stacking Blended Meta-Learner (Model C):**
   $$P_{\text{Blend}}(s_1, c) = 0.70 \cdot P_{\text{LGBM}}(s_1, c) + 0.30 \cdot P_{\text{Ridge}}(s_1, c)$$
   Blending tree-based leaf-wise splits with continuous latent manifold probabilities reduces prediction variance and prevents edge-case false positives.

### 5.2 5-Fold Entity-Level Stratified Cross-Validation Summary
Evaluated strictly at the `entity_id` level (ensuring zero data leakage across folds):

```
================================================================================
5-FOLD HYBRID ENSEMBLE SUMMARY (PIPELINE 2):
================================================================================
 Fold  Macro F0.5  Precision   Recall  Singleton Accuracy  Train Time (s)
    1    0.761948   0.772312 0.723131            0.998304        7.28s
    2    0.742517   0.748409 0.719849            0.998138       11.45s
    3    0.747669   0.752529 0.728840            0.998160        8.65s
    4    0.752422   0.758621 0.728608            0.998197        3.28s
    5    0.756335   0.765424 0.722039            0.998248        3.20s

Mean Macro F0.5: 0.7522 +/- 0.0075
Mean Precision:  0.7595 +/- 0.0096
Mean Recall:     0.7245 +/- 0.0042
Singleton Acc:   0.9982 +/- 0.0001
================================================================================
```

---

## 6. Full 1.73M Test Inference & High-Performance Scaling

Inference on the complete test dataset (1,732,544 test S1 entities against 9,969,589 candidate records) was accelerated using **50,000-entity vectorized batching**:
- **Why Batching Matters:** Sequential entity-by-entity scoring produces 1.73 million model calls, taking ~4 hours due to CPU-GPU communication overhead. Batching bundles 50,000 queries together, scoring all candidate rows in parallel.
- **Inference Time:** **571.43 seconds (~9.5 minutes)** for the entire 1,732,544 test set.
- **Disk Streaming:** Streamed predictions directly to disk, keeping RAM consumption under 18 GB and disk usage under 1 GB.

---

## 7. Official Competition Validator Verification

The official competition validator script was executed over both output files:
```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir /content/dataset/test \
    --check-ids
```

**Validator Output:**
```
ML Challenge 2026 — submission validator
  test dir: /content/dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (1152172 empty, 580372 non-empty).
  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## 8. Exact Reproduction Steps

To reproduce the winning pipeline end-to-end:

1. **Activate Environment & Install Requirements:**
   ```bash
   pip install -r Works_On_My_Machine_submission/code/business_entity_resolution/requirements.txt
   ```
2. **Execute Winning Pipeline Notebook:**
   ```bash
   jupyter nbconvert --to notebook --execute \
       Works_On_My_Machine_submission/code/business_entity_resolution/src/Pipeline_2_MultiStage_Ensemble_LightGBM_Ranker.ipynb \
       --output src/Pipeline_2_MultiStage_Ensemble_LightGBM_Ranker.ipynb
   ```
3. **Verify Generated Submission Files:**
   ```bash
   python3 student_resource/utils/validate_submission.py \
       --matching Works_On_My_Machine_submission/output/matching_results.tsv \
       --candidate Works_On_My_Machine_submission/output/candidate_pairs.tsv \
       --test-dir /content/dataset/test \
       --check-ids
   ```
