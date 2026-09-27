# Business Entity Resolution Pipeline

This repository contains the self-contained Machine Learning solution for the Amazon ML Challenge 2026: Business Entity Resolution Challenge.

## Repository Structure

```
Works_On_My_Machine_submission/
├── output/
│   ├── matching_results.tsv                      # Final entity matches (scored on leaderboard)
│   └── candidate_pairs.tsv                       # Candidate blocking sets
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── solution.ipynb                    # Master production end-to-end notebook
│       ├── README.md                             # Reproduction instructions
│       └── requirements.txt                      # Pinned Python dependencies
└── Documentation_template.md                     # Comprehensive technical methodology report
```

## Reproduction Instructions

### 1. Environment Setup
Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Production Pipeline
To execute the end-to-end production pipeline (Multi-Tier Inverted Indexing with Multi-Token Prefix + Soundex + Postal PIN + Leaf-wise LightGBM + Bipartite Star Disambiguation):
```bash
jupyter nbconvert --to notebook --execute src/solution.ipynb --output src/solution.ipynb
```
The notebook automatically checks for `dataset/` locally, and if missing, pulls it directly from Google Drive using `gdown`.

### 3. Submission Verification
Run the official competition validation script to verify formatting and ID constraints:
```bash
python3 ../../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../../dataset/test \
    --check-ids
```
This prints `PASS — no blocking issues found. Safe to submit.` (exit code 0).
