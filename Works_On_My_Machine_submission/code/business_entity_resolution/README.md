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
│       │   ├── Pipeline_1_GPU_XGBoost_StarClustering.ipynb    # Pipeline 1: Lexical Metric Space + GPU XGBoost
│       │   ├── Pipeline_2_MultiStage_Ensemble_LightGBM_Ranker.ipynb # Pipeline 2: Winning Multi-Modal Hybrid Ensemble
│       │   ├── Amazon_ML.ipynb                           # Master multi-pipeline benchmark notebook
│       │   └── run_pipeline.py                           # Automated end-to-end Python CLI runner
│       ├── README.md                                     # Reproduction instructions
│       └── requirements.txt                              # Pinned Python dependencies
└── Documentation_template.md                             # Comprehensive technical methodology report
```

## Reproduction Instructions

### 1. Environment Setup
Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Pipelines

#### Option A: Execute Winning Pipeline (Pipeline 2 - Multi-Modal Hybrid Ensemble)
To execute the winning hybrid pipeline (Phonetic Soundex + Postal PIN Inverted Index + PCA/LDA + LightGBM + Ridge Ranker):
```bash
jupyter nbconvert --to notebook --execute src/Pipeline_2_MultiStage_Ensemble_LightGBM_Ranker.ipynb --output src/Pipeline_2_MultiStage_Ensemble_LightGBM_Ranker.ipynb
```

#### Option B: Execute Pipeline 1 (GPU-Accelerated XGBoost)
To execute the GPU-accelerated XGBoost pipeline:
```bash
jupyter nbconvert --to notebook --execute src/Pipeline_1_GPU_XGBoost_StarClustering.ipynb --output src/Pipeline_1_GPU_XGBoost_StarClustering.ipynb
```

#### Option C: Automated CLI Execution
To execute via the CLI script:
```bash
python3 src/run_pipeline.py
```

### 3. Submission Verification
Run the official competition validation script to verify formatting and ID constraints:
```bash
python3 ../../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir /content/dataset/test \
    --check-ids
```
This prints `PASS — no blocking issues found. Safe to submit.` (exit code 0).
