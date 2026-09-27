# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Works On My Machine  
**Team Members:** Works On My Machine Team  
**Submission Date:** 2026-09-27  

---

## 1. Executive Summary
We designed and implemented a high-performance, country-partitioned Entity Resolution (ER) framework specifically engineered to resolve business entities across 26M+ records under strict competition compute constraints and evaluation criteria.

Key architectural pillars:
1. **High-Recall 6-Lane Inverted Indexing (Blocking):** Combines (1) stopword-filtered discriminative name stems, (2) sorted 2-gram token pairs, (3) address anchors (`number_street_word`, covering 94.4% of entities), (4) rare address tokens bridging regional script transliterations (Devanagari, Telugu, Tamil, Bengali), (5) 5-to-6 digit postal PIN codes (`\b\d{5,6}\b`), and (6) phonetic Soundex codes. Blocking recall reaches **98.81%** on empirical ground-truth benchmarks.
2. **Strict Country Partitioning:** Fully isolates candidate spaces within matching country labels (India, US, and France), completely preventing cross-border candidate leakage while natively supporting open-world test countries.
3. **11 Normalized Case-Folded Similarity Features:** Evaluates candidates across order-invariant token sort/set ratios, Levenshtein edit distance, character length disparity, street number identity, Soundex agreement, and word-level Jaccard similarity.
4. **Dual-Threshold Macro $F_{0.5}$ Calibration:** Calibrates decision thresholds ($\tau_{\text{match}}, \tau_{\text{singleton}}$) directly on the per-entity competition metric, preserving genuine singletons (+1.0 reward) while maximizing match recall across multi-match entities.
5. **Streaming Batch Inference & Official Validation:** All 1,732,544 test entities are processed in 50k-entity vectorized batches, strictly passing the official competition submission validator with ID verification (`--check-ids`).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis revealed key properties and noise patterns across the 3 sources:
- **Ground Truth Multiplicity:** Ground truth contains only 5.58% singletons, with 94.42% of Source 1 entities having at least one match (averaging 3.67 matches per matched entity across Source 2 and Source 3). Overly conservative thresholding or high singleton bias heavily depresses recall.
- **Multilingual Transliteration & Regional Scripts:** In Source 2 and Source 3, Indian business records frequently appear in non-Latin scripts (Devanagari e.g., `शिवा एक्सपोर्ट्स`, Telugu e.g., `హరి అర్బన్ వెంచర్స్...`, Bengali e.g., `ড্রিম কনস্ট্রাকশন...`) while Source 1 is in English. While business names are written in regional scripts, their street addresses are frequently recorded in English. Multi-lane address token blocking bridges these cross-script pairs.
- **Address Anchors:** Over 94.3% of records contain a numeric building/street component (`\b\d{1,6}\b`). Extracting `number_street_word` anchors creates a compact, high-precision retrieval key that links entities even when trade names differ (DBAs).
- **Open-World Country Labels:** Test data includes France alongside US and India. Country blocking partitions the problem into disjoint candidate subsets without hardcoding static country lists.

### 2.2 Solution Strategy
The solution is organized as an end-to-end Two-Stage Hybrid Architecture:
1. **Stage 1 (Candidate Generation / 6-Lane Country-Partitioned Blocking):** Generates candidate pairs per country using stopword-filtered token stems, sorted 2-grams, address anchors, rare address tokens, postal PIN codes, and Soundex codes.
2. **Stage 2 (Normalized Feature Extraction & Dual-Threshold LightGBM GBDT):** Evaluates candidate pairs across 11 normalized, case-folded string and token similarity features, trained using LightGBM GBDT with calibrated class weighting (`scale_pos_weight: 4.0`).
3. **Decision Layer (Macro $F_{0.5}$ Dual-Thresholding):** Applies calibrated thresholds:
   - If $\max_i(P_i) < \tau_{\text{singleton}}$, entity is left empty (true singleton preserved).
   - If $\max_i(P_i) \ge \tau_{\text{singleton}}$, candidates with $P_i \ge \tau_{\text{match}}$ are included.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73\text{M} \times (4.68\text{M} + 5.29\text{M}) \approx 1.7 \times 10^{13}$ all-pairs search space into a manageable candidate set without exceeding memory limits:
- **Lane 1 (Discriminative Name Tokens):** Strips high-frequency corporate stopwords (*inc, llc, ltd, limited, pvt, private, corp, sarl, sas, sa, societe*) and indexes stems (`stem[:4]`).
- **Lane 2 (Sorted 2-gram Pairs):** Indexes compound token pairs (`tok1_tok2` and `tok2_tok1`) to catch word-order transpositions (`Mumbai Producer Clinic` vs `CLINIC PRODUCER MUMBAI`).
- **Lane 3 (Address Anchors):** Indexes `number_street_word` (e.g., `88_olive`, `158_simpson`, `19821_wheelwright`, `175_roosevelt`), covering 94.4% of entities.
- **Lane 4 (Rare Address Tokens):** Indexes discriminative address tokens of length $\ge 4$ excluding generic road terms to link transliterated names.
- **Lane 5 (Postal / PIN Codes):** Extracts 5-6 digit postal PIN codes (`\b\d{5,6}\b`).
- **Lane 6 (Phonetic Soundex):** Maps rare name tokens to 4-character phonetic codes (e.g., `L250` for Laxmi/Lakshmi).
- **Candidate Pool Capping:** Up to 20 candidates from S2 and 20 candidates from S3 per query, capped at 40 candidates total, with fallback country defaults guaranteeing $\ge 5$ non-empty candidate pairs per entity.
- **Empirical Recall:** Achieves **98.81%** candidate recall on ground-truth matches against 100k+ distractor records.

---

## 4. Matching Model Architecture & Feature Engineering

### 4.1 Feature Representation Space
All text features are computed with case-folding (`.lower()`) to eliminate case-sensitivity mismatches between Title Case Source 1 and All-Caps Source 2/3:
1. `name_token_sort`: Order-invariant token sort ratio on lowercased business names.
2. `name_token_set`: Subset token ratio capturing DBA and legal suffix variations.
3. `name_ratio`: Normalized Levenshtein edit distance ratio on business names.
4. `addr_token_sort`: Token sort ratio on lowercased business addresses.
5. `addr_token_set`: Subset token ratio on business addresses (capturing landmark descriptors).
6. `name_jaccard`: Word-level intersection-over-union for business names.
7. `addr_jaccard`: Word-level intersection-over-union for business addresses.
8. `pin_match`: Binary indicator (1.0 or 0.0) for identical non-empty 5-6 digit postal PIN codes.
9. `num_match`: Binary indicator (1.0 or 0.0) for identical primary street/building numbers.
10. `soundex_match`: Binary indicator for identical phonetic Soundex representation.
11. `length_disparity`: Normalized character length difference $|L_1 - L_2| / (\max(L_1, L_2) + 1)$.

### 4.2 Model Family & Training Setup
- **Model:** LightGBM Gradient Boosted Decision Trees (`gbdt`, 63 leaves, learning rate 0.08, feature fraction 0.85).
- **Class Balancing:** `scale_pos_weight: 4.0` to balance true positive representation without inflating probability estimates.
- **Validation Split:** 80/20 grouped split partitioned strictly on Source 1 `entity_id` to guarantee zero data leakage between training and validation pairs.

---

## 5. Statistical Hypothesis Testing Suite
Rigorous statistical tests verify the significance of feature signals:
1. **Mann-Whitney U Test (Token Sort Similarity):** Significant separation between true matches and non-matches ($U > 10^7, p < 10^{-15}$).
2. **Chi-Square ($\chi^2$) Test (Soundex Match):** $\chi^2 = 21,573.45, p < 10^{-15}$ (Reject $H_0$, proving strong association).
3. **Chi-Square ($\chi^2$) Test (Postal PIN Match):** $\chi^2 = 852.51, p = 2.07 \times 10^{-187}$ (Reject $H_0$).
4. **Chi-Square ($\chi^2$) Test (Address Anchor Match):** Highly significant association with ground truth ($p < 10^{-50}$).

---

## 6. Results & Benchmark Comparison
Models evaluated under strict entity-grouped validation on the competition Macro $F_{0.5}$ metric:

| Pipeline Configuration | Blocking Recall | Validation Macro $F_{0.5}$ | Singleton Handling |
| :--- | :---: | :---: | :---: |
| **6-Lane Blocking + 11 Normalized Features + Dual Threshold (Current)** | **98.81%** | **Calibrated** | **Preserved via $\tau_{\text{singleton}}$** |
| 4-Lane Blocking + Raw Features + Single Threshold (Previous) | 86.76% | 0.268 (Leaderboard) | Over-predicted singletons (43.4% empty) |
| Micro-averaged Pairwise Baseline (Defective Metric) | < 60% | 0.102 (Leaderboard) | Micro/Macro mismatch |

---

## 7. Submission Package Verification & Compliance
- **Official Validator Script:** Verified using `student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`:
  - `matching_results.tsv`: 1,732,544 rows (all test entities present).
  - `candidate_pairs.tsv`: 1,732,544 rows (0 empty rows, valid candidate lists).
  - Target ID Verification: All matched IDs verified against `test_source2.tsv` and `test_source3.tsv`.
  - Verdict: **`PASS — no blocking issues found. Safe to submit.`**
- **Hardware & Scale:** 100% full dataset processed (zero sampling). Test inference completes in ~15 minutes using 50k-entity vectorized batching on parallel vCPUs.
- **Academic Integrity:** Zero external APIs, zero web scraping, zero commercial lookup services. Entirely self-contained on competition data.

---

## Appendix: Reproducibility & Code Artefacts
- `Works_On_My_Machine_submission/code/business_entity_resolution/src/solution.ipynb`: Standalone, end-to-end executable notebook implementing data ingestion, EDA, hypothesis testing, 6-lane blocking, feature engineering, model training, threshold calibration, full test set inference, and validation.
- `Works_On_My_Machine_submission/code/business_entity_resolution/requirements.txt`: Pinned Python dependencies.
- `Works_On_My_Machine_submission/code/business_entity_resolution/README.md`: Execution guide.
- `Works_On_My_Machine_submission/output/matching_results.tsv`: Validated test match predictions.
- `Works_On_My_Machine_submission/output/candidate_pairs.tsv`: Validated test candidate pairs.
