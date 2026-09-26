# Business Entity Resolution Pipeline

This directory contains the self-contained pipeline for resolving business entities across Source 1, Source 2, and Source 3.

## Structure

- `src/`: Source code modules for preprocessing, blocking, candidate generation, and entity matching.
- `requirements.txt`: Pinned dependencies and environment configuration.
- `README.md`: Reproduction instructions.

## Reproduction Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the end-to-end pipeline:
   ```bash
   python3 src/run_pipeline.py
   ```
   Or open and execute the Jupyter / Colab notebook:
   `src/Amazon_ML.ipynb`

3. Validate the submission outputs:
   ```bash
   python3 ../../../student_resource/utils/validate_submission.py \
       --matching ../../output/matching_results.tsv \
       --candidate ../../output/candidate_pairs.tsv \
       --test-dir /content/dataset/test
   ```
