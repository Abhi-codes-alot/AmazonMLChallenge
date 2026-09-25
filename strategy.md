# Amazon ML Challenge 2026: Winning Strategy & Architecture Specification

> **For Team Members & Autonomous Coding Agents (`agy`)**  
> **Challenge:** Business Entity Resolution (Scale: ~12M records across 3 sources)  
> **Primary Evaluation Metric:** Macro-averaged $F_{0.5}$ (Precision-Weighted)  

---

## 1. Executive Summary & Objective

Our goal is to build an end-to-end Entity Resolution pipeline that links reference business entities in **Source 1 (`S1`)** to their duplicate records in **Source 2 (`S2`)** and **Source 3 (`S3`)**.

```
                CANDIDATE GENERATION                    PAIRWISE SCORING                     DECISION ENGINE
            ┌───────────────────────────┐         ┌──────────────────────────┐         ┌─────────────────────────┐
            │ Multi-View Sharded Index  │         │  LightGBM Matcher        │         │ Per-S1 Decision Engine  │
Input Data ─┤ • Multi-View Normalization├────────>│ • Absolute String Sim    ├────────>│ • Best Score & Gap     │──> Final Outputs
(11.7M rows)│ • 5 Independent Passes    │         │ • Relative Rank Features │         │ • Singleton Detector    │
            │ • Empirical Candidate Cap │         │ • Cross-Field Interaction│         │ • Ambiguity Filter      │
            │   (Top 10..100 evaluation)│         │ • Hard-Negative Mining   │         │ • Joint 2D Calibration  │
            └───────────────────────────┘         └──────────────────────────┘         └─────────────────────────┘
                 (Recall Ceiling > 99%,               (Discriminative Ranking)              (Macro F0.5 Optimized)
                  P95/P99 Tracked)
```

---

## 2. Why Most Competitors Will Fail (The Core Traps)

1. **The $F_{0.5}$ Math Trap**:
   * The evaluation metric is **$F_{0.5}$ Macro-Averaged** across all Source 1 entities:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   * Precision is weighted **$2\times$ higher than Recall**. Every false positive is penalized twice as harshly as a missed link.
   * **The Singleton Penalty:** A Source 1 entity with zero true matches scores **$1.0$** if our predicted list is empty, but drops to **$0.0$** if even a single false match is predicted. Teams tuning for standard $F_1$ or pairwise thresholding will bleed singleton points.

2. **The Hard-Capping Fallacy**:
   * A Source 1 entity can have **zero, one, or many** true matches (e.g. 40+ genuine matches across S2 and S3).
   * Arbitrarily hard-capping candidates at $\le 20$ or $\le 30$ immediately makes high recall mathematically impossible. Candidate pool size must be an **experimentally measured curve** (evaluating Top 10, 20, 30, 50, 75, 100 against recall, P95, P99, and RAM).

3. **The 17.3 Trillion Pair Scaling Wall**:
   * $1.73\text{M}$ test $S1$ entities $\times$ $10\text{M}$ test secondary records ($S2 + S3$) $= \mathbf{17.3\text{ trillion}}$ possible comparisons.
   * Naive loops or monolithic sparse TF-IDF matrices across 10M rows will crash with OOM errors. Candidate generation must be **sharded and index-based**.

4. **The Zero-Shot "France" Generalization Curveball**:
   * Training set contains only `US` and `India`. Test set introduces `France`.
   * Hardcoding US state codes, Indian PIN-code logic, or country one-hot models will fail catastrophically on French test records.

5. **Single-String Aggressive Normalization**:
   * Replacing original strings with a single aggressively normalized version destroys subtle but decisive brand signals (e.g. `ABC Ltd` vs `ABC Logistics` vs `ABC Logistics Pvt Ltd`).

---

## 3. Our Key Differentiators & Upgraded Strategy

| Component | Standard Competitor Approach | Our Upgraded Strategy |
| :--- | :--- | :--- |
| **Data Representation** | Single aggressive canonical string | **Multi-View Representation**: Preserves raw text, normalized tokens, core alphanumeric keys, and extracted numbers simultaneously. |
| **Blocking Passes** | Single key or un-chunked TF-IDF | **5 Independent Sharded Passes**: Exact structural + Name token/n-gram index + Address number/anchor index + Phonetic + High-recall fallback. |
| **Candidate Cap** | Fixed arbitrary limit (e.g. 20) | **Empirical Curve Selection**: Measure recall vs P95/P99 candidate count across Top 10, 20, 30, 50, 75, 100. |
| **Training Samples** | Balanced random pairs (1:1) | **Stratified Hard-Negative Mining**: 1 positive : 3–10 hard negatives (same street/zip + similar name) : 1–3 random negatives. |
| **Feature Engineering** | Basic string distances only | **Tri-Partite Features**: Absolute string similarities + Group-relative ranks + Non-linear cross-field interactions. |
| **Decision Engine** | Single scalar probability cutoff | **Per-S1 Structural Gate + Secondary Ambiguity Filter**: Evaluates best score, second-best score, margin, and multi-S1 conflict resolution. |
| **Threshold Calibration** | 1D search on accuracy | **Joint 2D Calibration Harness** $(\tau_{\text{singleton}}, \tau_{\text{match}})$ optimizing Macro $F_{0.5}$ directly. |
| **Validation Scheme** | Random pair split (data leakage) | **GroupKFold by `source1_entity_id`** + **Cross-Country Zero-Shot Holdout** (train on US, validate on India to simulate France). |
| **Execution Scaling** | Immediate leap to full cloud dataset | **Progressive Verification**: Local (50k) $\to$ 200k $\to$ 500k $\to$ 1M $\to$ Full Cloud Scale. |

---

## 4. End-to-End Technical Architecture

### 4.1 Step 0: Initial Data Sanity & Country Verification
Before training or hard-blocking:
1. Verify schema, row counts, and null counts across all TSVs.
2. **Empirical Country Match Check:** Compute in `train_ground_truth.tsv`:
   $$\sum \mathbf{1}[\text{country}(S1) \ne \text{country}(S2/S3)]$$
   * If strictly 0: enforce `candidate_country == source1_country` as an absolute blocking constraint (cuts search space by ~70%).
   * If $> 0$: retain country compatibility as a high-weight soft feature rather than a hard drop.
3. Compute baseline singleton percentage and match cardinality distribution (P50, P90, P99).

### 4.2 Step 1: Multi-View Data Normalization
Rather than destructive string replacement, retain parallel views:
* **Name Views:**
  * `name_raw`: Original string as provided.
  * `name_normalized`: Lowercased, unicode NFKD accent-stripped, punctuation normalized.
  * `name_core`: Stopwords and legal suffixes (`inc`, `llc`, `pvt ltd`, `sarl`, `sa`, `corp`) stripped.
  * `name_alnum`: Alphanumeric-only representation.
  * `name_tokens_sorted`: Core tokens sorted alphabetically (eliminates word-order flips).
* **Address Views:**
  * `address_raw`: Original address string.
  * `address_normalized`: Standardized abbreviations (`rd` $\to$ `road`, `st` $\to$ `street`, etc.).
  * `address_numbers`: Extracted sequence of building/street/unit digits (language-agnostic).
  * `postal_tokens`: Extracted 5-digit (US/France) and 6-digit (India) postal tokens.

### 4.3 Step 2: 5 Independent Sharded Blocking Passes
Instead of a monolithic sparse matrix, build sharded inverted postings:
* **PASS 1 — Exact Structural Match:**
  * Keys: `(country, postal_code, name_tokens_sorted[0..1])` and `(country, street_number, name_core)`.
* **PASS 2 — Name Inverted Postings:**
  * Sharded inverted index on core name tokens and character 3-grams/4-grams.
  * Top-$K$ retrieval per query using token frequency scoring.
* **PASS 3 — Address Anchor Match:**
  * Key: `(country, street_number, first 3 chars of name)`.
  * Recovers businesses with alternate trade names (DBAs) sharing the exact physical location.
* **PASS 4 — Phonetic Recovery:**
  * Double Metaphone / Soundex on primary name tokens to recover typos and transliterations.
* **PASS 5 — High-Recall Semantic / Dense Fallback (Optional):**
  * Multilingual embedding search (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS) applied selectively for records with low candidate density.
* **Candidate Pool Curve Measurement:**
  * Evaluate candidate pool sizes ($K \in [10, 20, 30, 50, 75, 100]$).
  * Select the optimal $K$ that achieves $>99\%$ blocking recall while minimizing average and P99 candidate count.
  * Export intermediate candidates directly as `candidate_pairs.tsv`.

### 4.4 Step 3: Tri-Partite Feature Engineering Engine (RapidFuzz C++)
For every candidate pair $(S1, S2/S3)$, extract:
1. **Absolute Similarity Vector:**
   * Name: `token_sort_ratio`, `token_set_ratio`, `WRatio`, `partial_ratio`, `levenshtein_distance`, `jaro_winkler`, 3-gram/4-gram overlap, exact first-token match.
   * Address: `token_jaccard`, `token_overlap`, `street_number_exact`, `street_number_sim`, `postal_exact`, `postal_prefix_match`, shared numeric sequence count.
2. **Within-Group Relative & Rank Features (High Leverage):**
   * `rank_in_group`: Rank of this candidate's name similarity score among all candidates for the same $S1$ entity.
   * `margin_to_best`: Difference between top candidate score and current candidate score.
   * `margin_to_second`: Gap between top-1 and second-best candidate (large gap indicates unambiguous match).
   * `score_to_mean_ratio`: Ratio of candidate similarity to the group average.
3. **Cross-Field Interaction Features (Non-Linear Discriminators):**
   * `name_strong AND address_strong`
   * `name_strong AND postal_exact`
   * `address_strong AND street_number_exact`
   * `name_exact BUT address_conflict` (guards against branch stores at different locations)
   * `postal_exact BUT name_conflict` (guards against different stores in the same shopping mall)

### 4.5 Step 4: Stratified Hard-Negative Mining & LightGBM
* **Training Set Construction:**
  * Positive pairs ($y=1$): All ground-truth matches.
  * Hard negatives ($y=0$): Candidates sharing same postal code or street number with high string similarity but different entity ID ($3\text{–}10\times$ positives).
  * Easy negatives ($y=0$): Random candidate samples ($1\text{–}3\times$ positives).
  * Persist this dataset to disk for strict experimental reproducibility.
* **Model Training:**
  * LightGBM Binary Classifier with GroupKFold cross-validation (grouped by `source1_entity_id`).
  * Compare against LightGBM Ranker (`lambdarank`) and retain whichever demonstrates superior Macro $F_{0.5}$.

### 4.6 Step 5: Per-$S1$ Structural Decision Engine & Ambiguity Filter
Rather than applying a single naive probability cutoff:
1. **Candidate Group Profiling:**
   * For each $S1$, compute $P_{(1)}$ (best probability), $P_{(2)}$ (second best), and margin $\Delta = P_{(1)} - P_{(2)}$.
2. **Singleton Detection Gate:**
   * If $P_{(1)} < \tau_{\text{singleton}}$, predict an empty match (preserves $1.0$ singleton score).
3. **Multi-Match Inclusion Gate:**
   * If $P_{(1)} \ge \tau_{\text{singleton}}$, include all candidates $c$ where $P(c) \ge \tau_{\text{match}}$ and $(P_{(1)} - P(c)) \le \delta_{\text{margin}}$.
4. **Secondary Record Ambiguity Check:**
   * If a single secondary record (e.g. $S2\text{-}X$) is strongly claimed by two distinct $S1$ entities ($S1_A$ with $0.94$ and $S1_B$ with $0.91$), apply a conflict resolution check before final output.
5. **Joint 2D Calibration Harness:**
   * Grid search over $(\tau_{\text{singleton}}, \tau_{\text{match}})$ directly evaluating Macro $F_{0.5}$ on the validation set.

---

## 5. Experiment Evaluation Dashboard

Every experiment must report this standardized metric block before code changes are merged:

```text
================================================
EXPERIMENT: [experiment_id]
================================================
Blocking Recall:             XX.X %
Avg Candidates / S1:         XX.X
P95 Candidates:              XX
P99 Candidates:              XX

Pair Precision:              XX.X %
Pair Recall:                 XX.X %

Macro F0.5:                  0.XXXX
Singleton Accuracy:          XX.X %

False Positives (Total):     XXXX
False Negatives (Total):     XXXX
  - FN from Blocking:        XXXX  (Unrecoverable by model)
  - FN from Model/Cutoff:    XXXX  (Tuning target)

Runtime:                     XX min
Peak RAM:                    XX GB
================================================
```

---

## 6. Progressive Scaling Plan

```text
Local Laptop (50k sample) ────> Local Verification (200k) ────> JarvisLabs Cloud
           │                                 │                            │
   Fast logic, unit tests,          Benchmark RAM & time,       Full 12M records,
   features, baseline F0.5          verify sharded indexing     blocking, LightGBM,
                                                                final submission
```

* **Step 1:** Local 50k slice: test multi-view normalization, blocking recall, and baseline classifier.
* **Step 2:** Local 200k slice: profile memory, index build times, and hard-negative mining.
* **Step 3:** Cloud execution on JarvisLabs: run full candidate generation and inference across the complete 11.7M dataset.
