# Amazon ML Challenge 2026: Business Entity Resolution Roadmap

## Project Overview
* **Objective:** Map Source 1 (`S1`) business entities to duplicate records in Source 2 (`S2`) and Source 3 (`S3`).
* **Scale:** ~2.2M train records, ~1.73M test reference entities evaluated against ~10M secondary records (~11.7M records total).
* **Metric:** Macro-averaged $F_{0.5}$ score (precision-weighted, singletons scored as 1.0 if empty, 0.0 if false positive).
* **Execution Strategy:** Progressive benchmarking (Local 50k $\to$ 200k) followed by JarvisLabs cloud execution on the full dataset.

---

```
                       END-TO-END PIPELINE ARCHITECTURE
 ┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
 │ Raw Input Data  │ ────> │ Multi-View Normalize │ ────> │ 5-Pass Sharded Block │
 │ (S1, S2, S3)    │       │ (raw, core, numbers) │       │ (Empirical Top-K)    │
 └─────────────────┘       └──────────────────────┘       └──────────┬───────────┘
                                                                     │
 ┌─────────────────┐       ┌──────────────────────┐                  │ Candidate Pool
 │ Final Outputs   │ <──── │ Per-S1 Decision &    │ <──── ┌──────────┴───────────┐
 │ & Submissions   │       │ Ambiguity Filter     │       │ LightGBM Classifier  │
 └─────────────────┘       └──────────────────────┘       │ + Rank & Cross Feats │
                                                          └──────────────────────┘
```

---

## Phase 0: Data Sanity, Country Validation & Baseline Pipeline
> **Goal:** Verify data integrity, test the empirical country matching assumption, and build a minimal working submission that passes the official validator.

- [ ] **0.1 Data Sanity & Cardinality Inspection**
  - Verify schema, line counts, and missingness across all train and test TSVs.
  - Measure ground-truth match cardinality distribution (P50, P90, P99).
  - Calculate true singleton percentage in training set.
- [ ] **0.2 Empirical Country Matching Check**
  - Verify whether any true matches cross country boundaries:
    ```python
    sum(country_S1 != country_S2_or_S3)
    ```
  - If 0, establish country as an absolute hard partition; if > 0, preserve cross-country candidates with a penalty feature.
- [ ] **0.3 Fail-Safe Baseline Pipeline**
  - Implement a fast rule-based blocker (exact normalized name) on a small slice.
  - Generate initial dummy `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
  - Run `student_resource/utils/validate_submission.py` and confirm `PASS (exit 0)`.

---

## Phase 1: Multi-View Normalization & Robust Validation Scheme
> **Goal:** Create rich multi-view text representations without destructive replacement, and establish a cross-country zero-shot validation harness.

- [ ] **1.1 Multi-View Data Normalization**
  - Extract parallel representations for each record:
    - Name views: `name_raw`, `name_normalized`, `name_core`, `name_alnum`, `name_tokens_sorted`.
    - Address views: `address_raw`, `address_normalized`, `address_numbers`, `postal_tokens`.
- [ ] **1.2 Group-Aware Validation Split (by `source1_entity_id`)**
  - Maintain a locked validation set split strictly by `source1_entity_id` (prevent candidate leakage).
- [ ] **1.3 Cross-Country Zero-Shot Holdout Experiment**
  - Train on `US` (+ partial `India`), validate zero-shot on held-out `India` to proxy the test set `France` gap.
- [ ] **1.4 Offline Macro $F_{0.5}$ Evaluation Harness**
  - Replicate official competition scoring with strict singleton mechanics (empty = 1.0; false positive = 0.0).

---

## Phase 2: 5-Pass Sharded Blocking & Candidate Pool Curve
> **Goal:** Build diverse, independent blocking passes using sharded inverted postings and empirically optimize the candidate cap.

- [ ] **2.1 5 Independent Sharded Blocking Passes**
  - **Pass 1 (Exact Structural):** `(country, postal_code, name_tokens_sorted[0..1])` and `(country, street_number, name_core)`.
  - **Pass 2 (Name Postings):** Sharded token inverted index + character 3-gram/4-gram postings.
  - **Pass 3 (Address Anchors):** `(country, street_number, first 3 chars of name)`.
  - **Pass 4 (Phonetic Fallback):** Double Metaphone / Soundex on primary name tokens.
  - **Pass 5 (Semantic / Dense Fallback):** Multilingual bi-encoder (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS) applied selectively where candidate density is low.
- [ ] **2.2 Empirical Candidate Pool Optimization**
  - Benchmark candidate pool sizes ($K \in [10, 20, 30, 50, 75, 100]$).
  - Measure: Blocking Recall, Average Candidates, P95, P99, RAM, and Indexing Time.
  - Select the optimal $K$ guaranteeing $>99\%$ recall.
- [ ] **2.3 Export Candidate File**
  - Save candidate set as `output/candidate_pairs.tsv` (`source1_entity_id\tcandidate_entity_ids`).

---

## Phase 3: Stratified Hard-Negative Mining
> **Goal:** Construct a robust, reproducible training pair dataset reflecting the true difficulty of entity resolution.

- [ ] **3.1 Hard-Negative Mining**
  - For each positive pair ($y=1$), sample:
    - 3–10 **Hard Negatives** ($y=0$): Candidates sharing same postal code or street number with high string similarity.
    - 1–3 **Easy Negatives** ($y=0$): Random candidates from the blocking pool.
- [ ] **3.2 Persist Training Pairs**
  - Save the constructed pair dataset to disk for repeatable, deterministic experiments across model iterations.

---

## Phase 4: Tri-Partite Feature Engineering Engine
> **Goal:** Extract fine-grained discriminative signals using C++ RapidFuzz, relative within-group ranks, and cross-field interaction rules.

- [ ] **4.1 Absolute String & Numerical Similarities (RapidFuzz C++)**
  - Name: `token_sort_ratio`, `token_set_ratio`, `WRatio`, `partial_ratio`, `levenshtein_distance`, `jaro_winkler`, 3-gram/4-gram overlap, exact first-token match.
  - Address: `token_jaccard`, `token_overlap`, `street_number_exact`, `street_number_sim`, `postal_exact`, `postal_prefix_match`, shared numeric sequence count.
- [ ] **4.2 Within-Group Relative Rank Features**
  - `rank_in_group`: Rank of this candidate's name similarity among all candidates for the same $S1$ entity.
  - `margin_to_best`: Difference between top candidate score and current candidate score.
  - `margin_to_second`: Gap between top-1 and second-best candidate.
  - `score_to_mean_ratio`: Ratio of candidate similarity to group average.
- [ ] **4.3 Non-Linear Cross-Field Interaction Features**
  - `name_strong AND address_strong`
  - `name_strong AND postal_exact`
  - `address_strong AND street_number_exact`
  - `name_exact BUT address_conflict` (guards against branch stores at different locations)
  - `postal_exact BUT name_conflict` (guards against different stores in the same shopping mall)

---

## Phase 5: LightGBM Model Training
> **Goal:** Train a fast gradient-boosted decision tree optimized for high-precision entity resolution.

- [ ] **5.1 LightGBM Binary Classifier**
  - Train with GroupKFold cross-validation, configuring `scale_pos_weight` to account for imbalance.
  - Feature importance analysis: prune low-gain features to maximize inference speed.
- [ ] **5.2 Comparison with LightGBM Ranker**
  - Train LambdaMART ranker; retain only if it measurably outperforms the binary classifier on Macro $F_{0.5}$.

---

## Phase 6: Per-$S1$ Decision Engine, Ambiguity Filter & 2D Calibration
> **Goal:** Translate pairwise probabilities into global macro-optimal predictions.

- [ ] **6.1 Per-Entity Structural Logic**
  - Compute $P_{(1)}$ (best score), $P_{(2)}$ (runner-up), and margin $\Delta$.
  - **Singleton Gate:** If $P_{(1)} < \tau_{\text{singleton}}$, output empty string (protects 1.0 score).
  - **Multi-Match Inclusion Gate:** Include candidates where $P(c) \ge \tau_{\text{match}}$ and $(P_{(1)} - P(c)) \le \delta_{\text{margin}}$.
- [ ] **6.2 Secondary Record Ambiguity Filter**
  - Detect when the same secondary record ($S2\text{-}X$) is strongly claimed by multiple $S1$ entities.
  - Test conflict-resolution rules on validation set to minimize false merges.
- [ ] **6.3 Joint 2D Calibration Search**
  - Run grid search over $(\tau_{\text{singleton}}, \tau_{\text{match}})$ directly maximizing Macro $F_{0.5}$.

---

## Phase 7: Progressive Scaling & Cloud Deployment
> **Goal:** Benchmark and profile before scaling to full test inference.

- [ ] **7.1 Progressive Local Verification**
  - Run pipeline on 50k slice $\to$ benchmark runtime, RAM, and $F_{0.5}$.
  - Scale to 200k slice $\to$ confirm linear memory scaling and index performance.
- [ ] **7.2 JarvisLabs Full-Scale Execution**
  - Spin up cloud instance (RTX 4090 / RTX 5000 Ada, 80–100 GB disk).
  - Run full test candidate generation across 11.7M records.
  - Run batched LightGBM inference and export final `matching_results.tsv` and `candidate_pairs.tsv`.

---

## Phase 8: Submission Packaging & Documentation
> **Goal:** Validate formatting compliance and build the official zip package.

- [ ] **8.1 Run Official Submission Validator**
  - Verify output with `student_resource/utils/validate_submission.py --check-ids`.
- [ ] **8.2 Fill Documentation Template**
  - Complete [`student_resource/Documentation_template.md`](file:///C:/Users/ritvi/AmazonMLChallenge/student_resource/Documentation_template.md).
- [ ] **8.3 Build Final Zip Package**
  - Package `output/`, `code/business_entity_resolution/`, and `Documentation_template.md`.
