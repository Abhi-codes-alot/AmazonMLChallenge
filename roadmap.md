# Amazon ML Challenge 2026: Business Entity Resolution Roadmap

## Project Overview
* **Objective:** Map Source 1 (`S1`) business entities to duplicate records in Source 2 (`S2`) and Source 3 (`S3`).
* **Scale:** ~2.2M train records, ~1.73M test reference entities evaluated against ~10M secondary records (~11.7M records total).
* **Metric:** Macro-averaged $F_{0.5}$ score (precision-weighted, singletons scored as 1.0 if empty, 0.0 if false positive).
* **Execution Strategy:** Fast local prototyping & verification + JarvisLabs cloud GPU for full-scale vector indexing and inference.

---

```
                       END-TO-END PIPELINE ARCHITECTURE
 ┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
 │ Raw Input Data  │ ────> │  Data Normalization  │ ────> │ Candidate Generation │
 │ (S1, S2, S3)    │       │  & Text Cleaning     │       │ (4-Pass Hybrid Block)│
 └─────────────────┘       └──────────────────────┘       └──────────┬───────────┘
                                                                     │
 ┌─────────────────┐       ┌──────────────────────┐                  │ Candidate Set
 │ Final Outputs   │ <──── │ Joint 2D Calibrator  │ <──── ┌──────────┴───────────┐
 │ & Submissions   │       │ (τ_sing, τ_match)    │       │ Pairwise Scorer &    │
 └─────────────────┘       └──────────────────────┘       │ LightGBM (w/ Ranks)  │
                                                          └──────────────────────┘
```

---

## Phase 0: De-Risking Baseline (Fail-Safe First)
> **Goal:** Build and validate a minimal end-to-end pipeline before investing in model sophistication, guaranteeing that submission mechanics and format checks work 100%.

- [ ] **0.1 Minimal Rule-Based Pipeline**
  - Implement a fast single-pass blocker (exact name match or first 3 words) on a tiny slice.
  - Generate dummy/naive `matching_results.tsv` and `candidate_pairs.tsv`.
- [ ] **0.2 Run Official Submission Validator**
  - Execute `student_resource/utils/validate_submission.py`.
  - Confirm `PASS (exit 0)` locally with zero format errors or warnings.

---

## Phase 1: Local Setup, EDA & Robust Validation Scheme
> **Goal:** Set up local development, extract a 50k slice, and design a validation harness that simulates the zero-shot "France" test distribution and accurately scores Macro $F_{0.5}$.

- [ ] **1.1 Local Environment Setup**
  - Set up Python virtual environment (`.venv`).
  - Install dependencies: `polars`, `duckdb`, `rapidfuzz`, `scikit-learn`, `lightgbm`, `sentence-transformers`, `faiss-cpu`, `tqdm`.
- [ ] **1.2 Data Slicing for Local Iteration**
  - Extract a 50,000-record subset from `train_source1.tsv` and corresponding records from S2, S3, and ground truth.
- [ ] **1.3 Exploratory Data Analysis (EDA) & Noise Profiling**
  - Analyze noise patterns in names: legal suffixes (`Inc`, `Pvt Ltd`, `SARL`), punctuation, typos, transliterations, word-order flips.
  - Analyze address structures: abbreviations (`St`, `Rd`), landmark markers ("Near SBI ATM"), PIN/ZIP codes, missing state/city tokens.
- [ ] **1.4 Group-Aware Validation Split (by `source1_entity_id`)**
  - Ensure all candidate pairs for any given $S1$ entity remain strictly in train or strictly in validation (no data leakage).
- [ ] **1.5 Zero-Shot Cross-Country Holdout Experiment**
  - Train on `US` (+ partial `India`), validate zero-shot on held-out `India` to proxy the test set `France` gap.
- [ ] **1.6 Offline Macro $F_{0.5}$ Evaluation Harness**
  - Replicate the exact competition scoring logic including the singleton penalty (empty prediction = 1.0; false positive on singleton = 0.0).

---

## Phase 2: Candidate Generation (Blocking) Engine
> **Goal:** Reduce the search space from $17.3\text{ trillion}$ comparisons to $\le 30$ candidates per entity while maintaining $>98\%$ recall ceiling.

- [ ] **2.1 Text Normalization Pipeline**
  - Lowercasing, accent stripping (`NFKD`), whitespace cleaning.
  - Suffix harmonization (`pvt ltd` $\to$ `private limited`, `corp` $\to$ `corporation`, `rd` $\to$ `road`, etc.).
  - Language-agnostic structural address extraction (street/building numbers, unit numbers, postal codes).
- [ ] **2.2 4-Pass Hybrid Blocking Strategy**
  - **Pass 1 (Sorted Core-Token Key):**
    - Strip stopwords and legal suffixes; **sort core tokens alphabetically** + Country + Postal code (where present). Solves word-order flips directly at the source.
  - **Pass 2 (3-Gram MinHash / Sparse Inverted Index):**
    - Word & 3-gram character TF-IDF on cleaned business name + address.
    - **Chunked dot products:** Process queries in chunks of 10,000 records to prevent OOM memory spikes.
  - **Pass 3 (Physical Address Anchor Index):**
    - Key: `(Street Number, First 3 letters of Name, Country)`.
    - Catches entities with heavily shortened names or DBAs sharing the same physical location.
  - **Pass 4 (Multilingual Bi-Encoder + FAISS GPU):**
    - Encode name + address using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
    - Retrieve top-5 nearest neighbors via FAISS GPU index to capture phonetic transliterations and semantic variants.
- [ ] **2.3 Candidate Aggregation & Deduplication**
  - Merge candidates from all 4 passes.
  - Enforce top-$N$ cap (20–30 candidates per $S1$ entity) based on multi-pass consensus.
  - Measure **Recall Ceiling** and **Reduction Ratio** on validation set.
- [ ] **2.4 Export Candidate File**
  - Save as `output/candidate_pairs.tsv` formatted as `source1_entity_id\tcandidate_entity_ids`.

---

## Phase 3: Feature Engineering, Model Training & Joint Calibration
> **Goal:** Train a high-precision ML classifier with group-rank features, resolve class imbalance, and jointly calibrate decision gates to maximize Macro $F_{0.5}$.

- [ ] **3.1 Pairwise Feature Engineering**
  - **Absolute Similarity Features (RapidFuzz C++):**
    - Name: `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `levenshtein_distance`, `jaro_winkler`.
    - Brand integrity: `first_token_exact_match`, length ratio.
    - Address: Universal street number match, token Jaccard, postal code match score.
  - **Within-Group Relative & Rank Features (Key High-Value Addition):**
    - `rank_in_group`: Rank of this candidate's name similarity among all candidates for the same $S1$ entity.
    - `margin_to_best`: Difference between top-1 candidate score and current candidate score.
    - `margin_to_second`: Difference between top-1 and second-best candidate (large margin indicates an unambiguous match).
    - `score_to_mean_ratio`: Ratio of candidate's score to the average score in the candidate group.
  - **Structural Flags:**
    - Source indicator (`is_source_2`, `is_source_3`).
    - Blocking hit count (1 to 4).
- [ ] **3.2 Addressing Class Imbalance & Training LightGBM**
  - Downsample negative candidate pairs to ~1:8 ratio and set `scale_pos_weight` in LightGBM.
  - Train LightGBM Binary Classifier / Ranker with GroupKFold cross-validation.
- [ ] **3.3 Joint 2D Calibration Harness for Macro $F_{0.5}$**
  - Build a vectorized evaluation harness that accepts $(\tau_{\text{singleton}}, \tau_{\text{match}})$ and calculates Macro $F_{0.5}$ directly.
  - **Gate 1 (Singleton Gate):** If $\max(P) < \tau_{\text{singleton}}$, predict empty match $\to$ preserve 1.0 singleton credit.
  - **Gate 2 (Match Gate):** For non-singletons, include candidate $c$ only if $P(c) \ge \tau_{\text{match}}$.
  - Execute 2D grid search over $\tau_{\text{singleton}} \in [0.50, 0.85]$ and $\tau_{\text{match}} \in [0.65, 0.90]$ to find the global optimum.

---

## Phase 4: JarvisLabs Setup & Full-Scale Scaling
> **Goal:** Scale the verified pipeline to the full 12+ million records on JarvisLabs cloud GPU.

- [ ] **4.1 Spin Up JarvisLabs Instance**
  - Select GPU: **RTX 5000 Ada / RTX 6000 Ada / RTX 4090** (24–32 GB VRAM).
  - Framework: **PyTorch**, Storage: **80 GB – 100 GB**.
- [ ] **4.2 Data & Code Upload**
  - Package local code and datasets:
    ```powershell
    Compress-Archive -Path student_resource -DestinationPath student_resource.zip
    scp -P <PORT> student_resource.zip root@<JARVIS_IP>:/home/
    ```
- [ ] **4.3 Full Training & Candidate Generation on JarvisLabs**
  - Run full-scale 4-pass blocking across test records (using chunked TF-IDF and GPU-accelerated FAISS).
  - Train LightGBM model on full training candidate pairs.
- [ ] **4.4 Full Test Inference & Export**
  - Generate full `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
  - Download artifacts back to local machine and terminate instance.

---

## Phase 5: Verification, Submission Package & Documentation
> **Goal:** Validate formatting compliance and generate the final zip package.

- [ ] **5.1 Local Output Validation**
  - Run official validator:
    ```bash
    python student_resource/utils/validate_submission.py \
        --matching student_resource/output/matching_results.tsv \
        --candidate student_resource/output/candidate_pairs.tsv \
        --test-dir student_resource/dataset/test \
        --check-ids
    ```
- [ ] **5.2 Complete Documentation Template**
  - Fill out [`student_resource/Documentation_template.md`](file:///C:/Users/ritvi/AmazonMLChallenge/student_resource/Documentation_template.md).
- [ ] **5.3 Final Package Construction**
  - Build final submission archive:
    ```
    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md
    ```
