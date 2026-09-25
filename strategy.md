# Amazon ML Challenge 2026: Winning Strategy & Architecture Specification

> **For Team Members & Autonomous Coding Agents (`agy`)**  
> **Challenge:** Business Entity Resolution (Scale: ~12M records across 3 sources)  
> **Primary Evaluation Metric:** Macro-averaged $F_{0.5}$ (Precision-Weighted)  

---

## 1. Executive Summary & Objective

Our goal is to build an end-to-end Entity Resolution pipeline that links reference business entities in **Source 1 (`S1`)** to their duplicate records in **Source 2 (`S2`)** and **Source 3 (`S3`)**.

```
                CANDIDATE GENERATION                    PAIRWISE SCORING                     CALIBRATION
            ┌───────────────────────────┐         ┌──────────────────────────┐         ┌─────────────────────┐
            │  Multi-Pass Blocking      │         │  LightGBM Matcher        │         │  Dual-Gated Filter  │
Input Data ─┤  • Canonical Name Keys    ├────────>│  • RapidFuzz C++ Vector  ├────────>│  • Gate 1: Singleton│──> Final Outputs
(11.7M rows)│  • 3-Gram MinHash Index   │         │  • Universal Numbers     │         │  • Gate 2: Precision│
            │  • Address Anchor Index   │         │  • Asymmetric Loss       │         │    Threshold (F0.5) │
            └───────────────────────────┘         └──────────────────────────┘         └─────────────────────┘
                 (Recall > 97%,                      (Discriminative Ranking)              (Macro F0.5 Peak)
                  < 30 cands/entity)
```

---

## 2. Why Most Competitors Will Fail (The Core Traps)

1. **The $F_{0.5}$ Math Trap**:
   * The evaluation metric is **$F_{0.5}$ Macro-Averaged** across all Source 1 entities:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   * Precision is weighted **$2\times$ higher than Recall**. Every false positive is penalized twice as harshly as a missed link.
   * **The Singleton Penalty:** A Source 1 entity with zero true matches scores **$1.0$** if our predicted list is empty, but drops to **$0.0$** if even a single false match is predicted. Teams tuning for standard $F_1$ will drown their leaderboard score with singleton false positives.

2. **The 17.3 Trillion Pair Scaling Wall**:
   * $1.73\text{M}$ test $S1$ entities $\times$ $10\text{M}$ test secondary records ($S2 + S3$) $= \mathbf{17.3\text{ trillion}}$ possible comparisons.
   * Brute force, quadratic pairwise comparisons, or naive Python loops will crash memory (OOM) or take weeks to run.

3. **The Zero-Shot "France" Generalization Curveball**:
   * Training set contains only `US` and `India`.
   * Test set introduces `France`.
   * Any team hardcoding US 2-letter state abbreviations, Indian 6-digit PIN codes, or one-hot encoding the `country` column will fail catastrophically on the French test records.

4. **The Blocking Bottleneck (Recall Ceiling)**:
   * The final model can only score candidates that survive the blocking stage. If blocking recall is $85\%$, the final submission score can never exceed $0.85$.

---

## 3. Our 5 Core Winning Differentiators

| Component | Standard Approach | Our Winning Strategy |
| :--- | :--- | :--- |
| **Blocking / Candidate Gen** | Single key (e.g. first 3 chars or zip code) | **3-Pass Hybrid Ensemble** (Semantic + Lexical + Address Anchor). Guarantees $>97\%$ recall while compressing candidate set to $\le 30$ pairs/entity. |
| **Thresholding & Decision** | Single global threshold (e.g. 0.5) | **Dual-Gated Calibrator**: Dedicated Singleton Detection Gate ($\tau_{\text{singleton}}$) + High-Precision Match Gate ($\tau_{\text{match}}$). |
| **Cross-Country Generalization**| Hardcoded regexes & dictionaries | **Language-Agnostic Structural Parsing**: Universal street number extraction, token shapes, and character n-grams. |
| **Execution Performance** | Pandas / Python loops | **Polars + DuckDB + RapidFuzz (C++)**: Fully vectorized, multithreaded, and streaming-capable. |
| **Candidate Audit** | Ad-hoc candidate filtering | **Strict Alignment**: `candidate_pairs.tsv` represents the exact candidate set scored by LightGBM, satisfying competition auditing rules. |

---

## 4. End-to-End Technical Architecture

### 4.1 Step 1: Preprocessing & Open-Set Normalization
* **Country Partitioning**: Hard partition by `country`. An entity in `US` will never match an entity in `India` or `France`. This immediately cuts search space by $\sim 70\%$.
* **Text Normalization**:
  * Strip accents (Unicode normalization `NFKD`), lowercase, and normalize whitespace.
  * Standardize global business suffixes (`pvt ltd`, `inc`, `corp`, `llc`, `gmbh`, `sarl`, `sa` $\to$ canonical tokens).
  * Extract structural address features: building/street numbers, unit/apartment tokens, and postal codes.

### 4.2 Step 2: Multi-Pass Ensemble Candidate Generation (Blocking)
To ensure maximum recall ceiling with minimum candidate count ($\le 30$ per $S1$ record):
* **Pass 1: Canonical Core-Name & Location Key**
  * Key: `(First 2 core name tokens, Country)` + `(Postal code where present)`.
* **Pass 2: 3-Gram MinHash / Sparse Inverted Index**
  * Catches typos, spelling discrepancies, abbreviations, and token transpositions (e.g. `Wal-Mart` vs `Walmart Supercenter`).
* **Pass 3: Physical Address Anchor Index**
  * Key: `(Street Number, First 3 letters of Name, Country)`.
  * Catches entities with heavily shortened names or DBAs sharing the same physical location.
* **Candidate Set Export**:
  * Take the union of all passes, deduplicate, and rank by blocking confidence to keep top 20–30 candidates per $S1$ record.
  * This candidate pool is saved directly as `candidate_pairs.tsv`.

### 4.3 Step 3: Feature Engineering Engine (RapidFuzz C++)
For every candidate pair $(S1_i, S2_j)$ or $(S1_i, S3_k)$, compute:
* **Name Features:**
  * `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `levenshtein_distance`, `jaro_winkler`.
  * `first_token_exact_match` (binary flag for brand integrity).
  * `name_length_ratio` and character count delta.
* **Address Features:**
  * `street_number_match`: Exact match, prefix match, or missing.
  * `address_token_jaccard` and `address_token_overlap_ratio`.
  * `postal_code_match`: 1.0 (exact), 0.5 (prefix match), 0.0 (mismatch).
* **Metadata & Structural Features:**
  * Source indicator (`is_S2`, `is_S3`).
  * Candidate retrieval pass count (how many blocking passes flagged this candidate?).

### 4.4 Step 4: Pairwise Scoring Model (LightGBM)
* **Training Data Construction**:
  * Positive pairs ($y=1$): True matches from `train_ground_truth.tsv`.
  * Negative pairs ($y=0$): High-scoring candidate pairs from blocking that are not in ground truth (hard negative mining).
* **Model Choice**: LightGBM Ranker / Binary Classifier.
  * Extremely fast CPU/GPU inference.
  * Handles non-linear feature interactions seamlessly (e.g. street number match + high name similarity).

### 4.5 Step 5: Dual-Gated Calibration for Macro $F_{0.5}$
* **Gate 1 — Singleton Detector**:
  * For each $S1$ entity, let $P_{\max} = \max_{c \in \text{Candidates}} P(\text{match} \mid c)$.
  * If $P_{\max} < \tau_{\text{singleton}}$ (e.g., $0.65$), predict **no matches** (output empty string).
  * Protects our $1.0$ scores on singletons.
* **Gate 2 — Conservative Match Filter**:
  * For non-singletons, only predict candidates where $P(c) \ge \tau_{\text{match}}$ (e.g., $0.78$).
  * Enforces the heavy precision bias required by $F_{0.5}$.

---

## 5. Work Division & Guidelines for AI Agents (`agy`) & Teammates

### Role 1: Data Engineering & Blocking (`agy-data`)
* **Focus:** Build and maintain preprocessing, tokenization, and multi-pass candidate generation.
* **Key Metric to Maximize:** **Blocking Recall Ceiling** (must be $> 97\%$) while keeping average candidates per entity $\le 30$.
* **Deliverable:** `src/blocking.py` which outputs `candidate_pairs.tsv`.

### Role 2: Feature Engineering & Model Training (`agy-model`)
* **Focus:** Build RapidFuzz feature pipeline, train LightGBM classifier, and calibrate thresholds.
* **Key Metric to Maximize:** **Macro $F_{0.5}$** on validation split (with singletons included).
* **Deliverable:** `src/features.py`, `src/train.py`, and `src/matcher.py`.

### Role 3: Cloud Scaling & Deployment (`agy-infra`)
* **Focus:** Manage JarvisLabs cloud workflow, multi-threading, memory profiling, and packaging.
* **Key Metric:** Total inference time $< 2$ hours for the full 11.7M dataset without exceeding 32GB RAM.
* **Deliverable:** `run_pipeline.py`, submission zip builder, and `validate_submission.py` verification.

---

## 6. Strict Rules & Constraints Checklist
- [x] **No External Data / APIs:** Zero external geocoding, Google Places, or web searches (results in immediate disqualification).
- [x] **Model Size:** $\le 8$ Billion parameters, MIT / Apache 2.0 license.
- [x] **Output Format:** Both `matching_results.tsv` and `candidate_pairs.tsv` must pass `validate_submission.py` with 0 errors.
- [x] **Open-Set Ready:** Code must run seamlessly on `France` without modification.
