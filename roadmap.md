# Amazon ML Challenge 2026: Business Entity Resolution Roadmap

## Project Overview
* **Objective:** Map Source 1 (`S1`) business entities to their corresponding duplicate records in Source 2 (`S2`) and Source 3 (`S3`).
* **Scale:** ~2.2M train records, ~1.73M test reference entities evaluated against ~10M secondary records (~11.7M records total).
* **Metric:** Macro-averaged $F_{0.5}$ score (precision-weighted, singletons scored as 1.0 if empty, 0.0 if false positive).
* **Execution Strategy:** Hybrid development (Local for rapid prototyping and pipeline verification + JarvisLabs GPU cloud for full-scale indexing, training, and inference).

---

```
                       END-TO-END PIPELINE ARCHITECTURE
 ┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
 │ Raw Input Data  │ ────> │  Data Normalization  │ ────> │ Candidate Generation │
 │ (S1, S2, S3)    │       │  & Text Cleaning     │       │ (Multi-Pass Blocking)│
 └─────────────────┘       └──────────────────────┘       └──────────┬───────────┘
                                                                     │
 ┌─────────────────┐       ┌──────────────────────┐                  │ Candidate Set
 │ Final Outputs   │ <──── │ Precision Threshold  │ <──── ┌──────────┴───────────┐
 │ & Submissions   │       │ & Singleton Filter   │       │ Pairwise Scorer &    │
 └─────────────────┘       └──────────────────────┘       │ Classifier (LightGBM)│
                                                          └──────────────────────┘
```

---

## Phase 1: Local Setup, Data Profiling & Fast Baseline
> **Goal:** Set up local development environment, build a sample slice for fast iteration, and establish an end-to-end working baseline with local validation.

- [ ] **1.1 Local Environment Setup**
  - Create a Python virtual environment (`.venv`) locally.
  - Install core processing tools: `polars`, `duckdb`, `rapidfuzz`, `scikit-learn`, `lightgbm`, `tqdm`.
- [ ] **1.2 Data Slicing for Local Iteration**
  - Extract a manageable sub-sample (e.g., 20,000 to 50,000 records from `train_source1.tsv` and corresponding records from S2, S3, and ground truth).
  - Use this sub-slice locally for instantaneous debugging without waiting for multi-gigabyte file reads.
- [ ] **1.3 Exploratory Data Analysis (EDA) & Noise Profiling**
  - Analyze noise patterns in names: legal entity types (`Inc`, `Corp`, `LLC`, `Pvt Ltd`), punctuation, transliterations, word-order flips.
  - Analyze address structures: abbreviations (`St`, `Rd`, `Blvd`), landmark markers ("Near SBI ATM"), PIN/ZIP codes, missing state/city tokens.
  - Inspect `country` distribution: handle `US`, `India`, and design pipeline to generalize zero-shot to open-set countries like `France` (test set).
- [ ] **1.4 Local Validation Framework**
  - Implement a Python function strictly computing the official **Macro $F_{0.5}$** metric.
  - Ensure singleton handling is identical to the official validator (empty predicted matches on a true singleton = 1.0; predicting any false match = 0.0).

---

## Phase 2: Candidate Generation (Blocking) Engine
> **Goal:** Reduce the search space from $1.73\text{M} \times 10\text{M} \approx 17.3\text{ trillion}$ comparisons to $\le 30$ candidates per entity while maintaining $>95\%$ recall.

- [ ] **2.1 Text Normalization Pipeline**
  - Lowercasing, accent stripping, punctuation standardization.
  - Suffix harmonization (`pvt ltd` $\to$ `private limited`, `corp` $\to$ `corporation`, `rd` $\to$ `road`, etc.).
  - Country-aware address parsing (extracting numbers, postal codes, and city tokens).
- [ ] **2.2 Multi-Pass Hybrid Blocking Strategy**
  - **Pass 1 (Deterministic Key Index):**
    - First 2 tokens of business name + Country.
    - Postal code / ZIP code exact match (where available).
  - **Pass 2 (Sparse TF-IDF / Token Inverted Index):**
    - Word & 3-gram character TF-IDF on cleaned business name + address.
    - Top-$K$ retrieval using sparse matrix multiplication.
  - **Pass 3 (Phonetic / Fuzzy Fallback):**
    - Double Metaphone / Soundex on key name tokens for handling typos and transliterations.
- [ ] **2.3 Candidate Aggregation & Deduplication**
  - Merge candidates from all passes.
  - Enforce top-$N$ cap (e.g. 20–40 candidates per S1 entity) based on candidate generation confidence.
  - Measure **Recall Ceiling** and **Reduction Ratio** on validation set.
- [ ] **2.4 Export Candidate File**
  - Format output as required for `candidate_pairs.tsv` (`source1_entity_id\tcandidate_entity_ids`).

---

## Phase 3: Feature Engineering & Pairwise Matching Model
> **Goal:** Train a high-precision ML classifier to rank and filter candidates, heavily optimizing for precision ($F_{0.5}$).

- [ ] **3.1 Pairwise Feature Engineering**
  - **Name Similary Features:**
    - RapidFuzz token sort ratio, token set ratio, partial ratio, Levenshtein distance, Jaro-Winkler.
    - Longest common substring length / ratio.
    - First token match boolean (critical for brand names).
  - **Address Similarity Features:**
    - Token Jaccard overlap, numerical token overlap (building/street numbers).
    - Postal code exact match / partial match.
    - Street name similarity.
  - **Contextual / Metadata Features:**
    - Country compatibility boolean.
    - Source flag (`is_source_2`, `is_source_3`).
    - Rank / retrieval score from the candidate generation stage.
- [ ] **3.2 Model Training (LightGBM / CatBoost)**
  - Construct balanced training pairs: True matches from `train_ground_truth.tsv` as positive class ($y=1$), non-matching candidates from blocking as negative class ($y=0$).
  - Train a fast gradient-boosted decision tree (LightGBM / CatBoost).
- [ ] **3.3 Threshold Calibration for Macro $F_{0.5}$**
  - Grid-search probability threshold $\tau$ on validation set.
  - Since $F_{0.5}$ weights precision $2\times$ over recall, calibrate a conservative/high threshold $\tau$ to suppress false merges.
  - Implement a singleton threshold: if max predicted probability for an S1 entity is below $\tau_{singleton}$, predict an empty match.

---

## Phase 4: JarvisLabs Setup & Full-Scale Scaling
> **Goal:** Scale the tested pipeline to the full 12+ million records on JarvisLabs cloud GPU/high-memory instance.

- [ ] **4.1 Spin Up JarvisLabs Instance**
  - Select GPU: **RTX 5000 Ada / RTX 6000 Ada / A5000 / A6000** (or RTX 4090).
  - Framework: **PyTorch**.
  - Storage: **80 GB – 100 GB**.
- [ ] **4.2 Data & Code Upload**
  - Package local code and datasets:
    ```powershell
    Compress-Archive -Path student_resource -DestinationPath student_resource.zip
    scp -P <PORT> student_resource.zip root@<JARVIS_IP>:/home/
    ```
  - Unpack on instance and verify directory tree.
- [ ] **4.3 Instance Environment Configuration**
  - Install high-performance packages:
    ```bash
    pip install polars duckdb rapidfuzz scikit-learn lightgbm xgboost catboost sentence-transformers faiss-gpu tqdm
    ```
- [ ] **4.4 Full Training & Candidate Generation on JarvisLabs**
  - Run full-scale blocking on the 1.73M test S1 records against the 10M S2/S3 records using Polars/DuckDB multithreading.
  - Train LightGBM model on full candidate pairs generated from training set.
- [ ] **4.5 Full Test Inference**
  - Batch inference on test candidate pairs.
  - Generate final `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- [ ] **4.6 Download Artifacts to Local Machine**
  - Download `matching_results.tsv` and `candidate_pairs.tsv` to `student_resource/output/`.
  - Pause or terminate JarvisLabs instance to save credits.

---

## Phase 5: Verification, Submission Package & Documentation
> **Goal:** Validate compliance against challenge rules and generate the final zip package.

- [ ] **5.1 Local Output Validation**
  - Run official validator script:
    ```bash
    python student_resource/utils/validate_submission.py \
        --matching student_resource/output/matching_results.tsv \
        --candidate student_resource/output/candidate_pairs.tsv \
        --test-dir student_resource/dataset/test \
        --check-ids
    ```
  - Ensure zero formatting errors, correct tab-separations, no self-matches, and valid entity ID ranges.
- [ ] **5.2 Complete Documentation Template**
  - Fill in [`student_resource/Documentation_template.md`](file:///C:/Users/ritvi/AmazonMLChallenge/student_resource/Documentation_template.md):
    - Executive Summary
    - Problem Analysis (noise patterns, address quirks, France handling)
    - Candidate Generation / Blocking strategy (keys, reduction ratio, recall)
    - Matching Model & Features
    - Results & Error Analysis ($F_{0.5}$ score, failure cases)
- [ ] **5.3 Final Package Construction**
  - Structure the submission archive:
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
  - Upload `matching_results.tsv` to the challenge portal for public leaderboard scoring.
