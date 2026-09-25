# Amazon ML Challenge 2026: Winning Strategy & Architecture Specification

> **For Team Members & Autonomous Coding Agents (`agy`)**  
> **Challenge:** Business Entity Resolution (Scale: ~12M records across 3 sources)  
> **Primary Evaluation Metric:** Macro-averaged $F_{0.5}$ (Precision-Weighted)  

---

## 1. Executive Summary & Objective

Our goal is to build an end-to-end Entity Resolution pipeline that links reference business entities in **Source 1 (`S1`)** to their duplicate records in **Source 2 (`S2`)** and **Source 3 (`S3`)**.

```
                CANDIDATE GENERATION                    PAIRWISE SCORING                     CALIBRATION
            ┌───────────────────────────┐         ┌──────────────────────────┐         ┌─────────────────────────┐
            │  Multi-Pass Hybrid Index  │         │  LightGBM Matcher        │         │  Joint 2D Calibrator    │
Input Data ─┤  • Pass 1: Sorted Core Key│────────>│  • RapidFuzz C++ Vector  ├────────>│  • Gate 1: Singleton    │──> Final Outputs
(11.7M rows)│  • Pass 2: 3-Gram MinHash │         │  • Relative Rank Features│         │    Detector (τ_sing)    │
            │  • Pass 3: Address Anchor │         │  • Class Imbalance Weight│         │  • Gate 2: Precision    │
            │  • Pass 4: Multilingual   │         │  • Asymmetric Loss       │         │    Filter (τ_match)     │
            │    Bi-Encoder + FAISS GPU │         └──────────────────────────┘         └─────────────────────────┘
            └───────────────────────────┘              (Within-Group Ranking)              (Macro F0.5 Optimized)
                 (Recall Ceiling > 98%,
                  < 30 cands/entity)
```

---

## 2. Why Most Competitors Will Fail (The Core Traps)

1. **The $F_{0.5}$ Math Trap**:
   * The evaluation metric is **$F_{0.5}$ Macro-Averaged** across all Source 1 entities:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   * Precision is weighted **$2\times$ higher than Recall**. Every false positive is penalized twice as harshly as a missed link.
   * **The Singleton Penalty:** A Source 1 entity with zero true matches scores **$1.0$** if our predicted list is empty, but drops to **$0.0$** if even a single false match is predicted. Teams tuning for standard $F_1$ or pairwise thresholding will drown their leaderboard score with singleton false positives.

2. **The 17.3 Trillion Pair Scaling Wall**:
   * $1.73\text{M}$ test $S1$ entities $\times$ $10\text{M}$ test secondary records ($S2 + S3$) $= \mathbf{17.3\text{ trillion}}$ possible comparisons.
   * Brute force, quadratic pairwise comparisons, or naive Python loops will crash memory (OOM) or take weeks to run. Even sparse cosine similarity without chunking will crash on 1M $\times$ 6M pairs.

3. **The Zero-Shot "France" Generalization Curveball**:
   * Training set contains only `US` and `India`. Test set introduces `France`.
   * Any team hardcoding US 2-letter state abbreviations, Indian 6-digit PIN codes, or one-hot encoding the `country` column will fail catastrophically on French test records.

4. **The Blocking Bottleneck (Recall Ceiling)**:
   * The final model can only score candidates that survive the blocking stage. If blocking recall is $85\%$, the final submission score can never exceed $0.85$, regardless of classifier quality.

---

## 3. Our Key Differentiators & Upgraded Strategy

| Component | Standard Competitor Approach | Our Upgraded Winning Strategy |
| :--- | :--- | :--- |
| **Blocking Passes** | Single key or pure lexical n-grams | **4-Pass Hybrid Ensemble**: Sorted core keys + 3-Gram MinHash + Address Anchor + Multilingual Bi-Encoder (`paraphrase-multilingual-MiniLM-L12-v2` + FAISS GPU) to capture transliterations. |
| **Threshold Calibration** | Independent 1D search on pairwise accuracy | **Joint 2D Grid Search** $(\tau_{\text{singleton}}, \tau_{\text{match}})$ running directly on the macro $F_{0.5}$ evaluation harness. |
| **Feature Engineering** | Only absolute string similarities | **Absolute + Within-Group Relative Rank Features** (`margin_to_best`, `margin_to_second`, `rank_in_group`, `score_ratio`). |
| **Class Imbalance** | Ignored (skewed probability calibration) | Handled via **Hard Negative Subsampling** + `scale_pos_weight` in LightGBM. |
| **Validation Scheme** | Random pair split (data leakage) | **GroupKFold by `source1_entity_id`** + **Cross-Country Zero-Shot Holdout** (train on US, validate on India to simulate France). |
| **Memory Management** | Full-matrix sparse dot products (OOM risk) | **Row-Chunked Sparse Inverted Index** (batches of 10,000 queries). |
| **Hardware Choice** | Random GPU instance | **Tailored Allocation**: High-VRAM GPU instance (e.g. RTX 4090 / RTX 5000 Ada) for Pass 4 vector indexing; high-RAM multithreaded CPU for LightGBM. |

---

## 4. End-to-End Technical Architecture

### 4.1 Step 0: De-Risking Baseline (Fail-Safe First)
* Build a minimal end-to-end script using single-pass blocking and simple thresholding.
* Generate initial `matching_results.tsv` and `candidate_pairs.tsv`.
* Run and pass `student_resource/utils/validate_submission.py` to ensure submission mechanics work before building complex modules.

### 4.2 Step 1: Preprocessing & Open-Set Normalization
* **Country Partitioning**: Hard partition by `country`. An entity in `US` will never match an entity in `India` or `France`. This immediately cuts search space by $\sim 70\%$.
* **Text Normalization**:
  * Strip accents (Unicode normalization `NFKD`), lowercase, and normalize whitespace.
  * Standardize global business suffixes (`pvt ltd`, `inc`, `corp`, `llc`, `gmbh`, `sarl`, `sa` $\to$ canonical tokens).
  * Extract structural address features: building/street numbers, unit/apartment tokens, and postal codes.

### 4.3 Step 2: Multi-Pass Ensemble Candidate Generation (Blocking)
To ensure maximum recall ceiling with minimum candidate count ($\le 30$ per $S1$ record):
* **Pass 1: Word-Order Invariant Canonical Name Key**
  * Strip legal suffixes and stopwords, **sort core name tokens alphabetically**, take the first 2 sorted tokens + Country + Postal code (where present). Solves word-order transpositions at the root.
* **Pass 2: 3-Gram MinHash / Sparse Inverted Index**
  * Catches typos, spelling discrepancies, abbreviations, and token transpositions.
  * **Memory Safeguard:** Process queries in chunks of 10,000 records to prevent memory spikes.
* **Pass 3: Physical Address Anchor Index**
  * Key: `(Street Number, First 3 letters of Name, Country)`.
  * Catches entities with heavily shortened names or DBAs sharing the same physical location.
* **Pass 4: Multilingual Bi-Encoder + FAISS GPU (Semantic & Transliteration Recovery)**
  * Encode concatenated name + address using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  * Retrieve top-5 nearest neighbors via FAISS GPU index. Catches non-English transliterations and semantic variants that lexical passes miss.
* **Candidate Set Export**:
  * Take the union of all passes, deduplicate, and keep top 20–30 candidates per $S1$ record.
  * This candidate pool is saved directly as `candidate_pairs.tsv`.

### 4.4 Step 3: Feature Engineering Engine (RapidFuzz C++)
For every candidate pair $(S1_i, S2_j)$ or $(S1_i, S3_k)$, compute:
* **Absolute Name Similarity:**
  * `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `levenshtein_distance`, `jaro_winkler`.
  * `first_token_exact_match` (binary flag for brand integrity).
  * `name_length_ratio` and character count delta.
* **Absolute Address Similarity:**
  * `street_number_match`: Exact match, prefix match, or missing.
  * `address_token_jaccard` and `address_token_overlap_ratio`.
  * `postal_code_match`: 1.0 (exact), 0.5 (prefix match), 0.0 (mismatch).
* **Within-Group Relative & Rank Features (Key High-Value Addition):**
  * `rank_in_group`: Rank of this candidate's name similarity score among all candidates for the same $S1$ entity.
  * `margin_to_best`: Difference between the top-1 candidate's score and this candidate's score.
  * `margin_to_second`: Difference between top-1 and second-best candidate (large margin indicates an unambiguous single match).
  * `score_to_mean_ratio`: Ratio of this candidate's score to the average score of all candidates for that entity.
* **Metadata & Pass Flags:**
  * Source indicator (`is_S2`, `is_S3`).
  * Hit count across blocking passes (1 to 4).

### 4.5 Step 4: Pairwise Scoring Model (LightGBM)
* **Training Data Construction**:
  * Positive pairs ($y=1$): True matches from `train_ground_truth.tsv`.
  * Negative pairs ($y=0$): Hard negative mining from blocking candidates not in ground truth.
  * Adjust `scale_pos_weight` (or calibrate negative sampling ratio at ~1:8) to prevent probability distortion.
* **Model Choice**: LightGBM Classifier / Ranker.
  * Fast CPU/GPU inference.
  * Handles non-linear feature interactions seamlessly (e.g. street number match + high name similarity).

### 4.6 Step 5: Joint 2D Calibration Harness for Macro $F_{0.5}$
* **Evaluation Harness**:
  * Write a fast evaluation function that takes a candidate dataframe with predicted probabilities, applies $(\tau_{\text{singleton}}, \tau_{\text{match}})$, builds prediction lists, and calculates the exact official Macro $F_{0.5}$ score.
* **Gate 1 — Singleton Detector ($\tau_{\text{singleton}}$)**:
  * For each $S1$ entity, let $P_{\max} = \max_{c \in \text{Candidates}} P(\text{match} \mid c)$.
  * If $P_{\max} < \tau_{\text{singleton}}$, predict **no matches** (output empty string) $\to$ guarantees $1.0$ on singletons.
* **Gate 2 — Conservative Match Filter ($\tau_{\text{match}}$)**:
  * For non-singletons, only predict candidates where $P(c) \ge \tau_{\text{match}}$.
* Run a **Joint 2D Grid Search** over $\tau_{\text{singleton}} \in [0.50, 0.85]$ and $\tau_{\text{match}} \in [0.65, 0.90]$ to find the global peak of Macro $F_{0.5}$.

---

## 5. Validation Strategy & Simulating the France Gap

1. **Strict GroupKFold by `source1_entity_id`**:
   * Never randomly split pairs. All candidates for an $S1$ entity must be strictly in Train or strictly in Validation.
2. **Zero-Shot Cross-Country Holdout Experiment**:
   * Train model on `US` records only.
   * Evaluate zero-shot on `India` records without tuning country-specific parameters.
   * If the model maintains strong $F_{0.5}$ across this boundary, it will generalize reliably to `France` in the test set.

---

## 6. Work Division & Guidelines for AI Agents (`agy`) & Teammates

### Role 1: Data Engineering & Blocking (`agy-data`)
* **Focus:** Build and maintain preprocessing, tokenization, multi-pass candidate generation (Passes 1–3 + chunked TF-IDF).
* **Key Metric:** **Blocking Recall Ceiling** (must be $> 98\%$) with $\le 30$ candidates/entity.
* **Deliverable:** `src/blocking.py` which outputs `candidate_pairs.tsv`.

### Role 2: Embedding & Transliteration Search (`agy-semantic`)
* **Focus:** Multilingual bi-encoder encoding (`sentence-transformers`) and GPU-accelerated FAISS ANN search (Pass 4).
* **Deliverable:** `src/embeddings.py`.

### Role 3: Feature Engineering & Model Training (`agy-model`)
* **Focus:** Absolute & relative rank feature extraction, LightGBM training with imbalance weighting, and joint 2D $(\tau_{\text{singleton}}, \tau_{\text{match}})$ grid search.
* **Key Metric:** **Macro $F_{0.5}$** on validation split (with singletons included).
* **Deliverable:** `src/features.py`, `src/train.py`, and `src/matcher.py`.

### Role 4: Cloud Scaling & Deployment (`agy-infra`)
* **Focus:** Manage JarvisLabs GPU instance, streaming execution, memory limits, and packaging.
* **Key Metric:** Total inference time $< 2$ hours on full test set without OOM.
* **Deliverable:** `run_pipeline.py`, submission zip builder, and `validate_submission.py` verification.

---

## 7. Strict Rules & Constraints Checklist
- [x] **No External Data / APIs:** Zero external geocoding, Google Places, or web searches (immediate disqualification).
- [x] **Model Size:** $\le 8$ Billion parameters, MIT / Apache 2.0 license.
- [x] **Output Format:** Both `matching_results.tsv` and `candidate_pairs.tsv` must pass `validate_submission.py` with 0 errors.
- [x] **Open-Set Ready:** Tested via cross-country zero-shot holdout to guarantee zero-shot performance on `France`.
