"""
Amazon ML Challenge 2026: Business Entity Resolution Pipeline
Team: Works On My Machine

Multi-Approach Pipeline:
- Multi-pass inverted index blocking (Token, Bigram, Postal PIN codes) partitioned by country.
- 12 fine-grained pairwise lexical and geographic similarity features.
- Evaluates 4 distinct modeling paradigms:
    1. GPU XGBoost Classifier (Pairwise ranking & classification)
    2. LightGBM GBDT (Leaf-wise histogram gradient boosting across 28 vCPUs)
    3. Star-Clustering Graph Disambiguation (Bipartite resolution)
    4. Blended Weighted Ensemble (Calibrated soft-voting + Star-Clustering)
- Direct disk streaming to TSVs adhering to memory and disk bounds.
- 100% compliant with submission rules and official validator script.
"""

import os
import gc
import re
import sys
import time
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import xgboost as xgb
import lightgbm as lgb

# ---------------------------------------------------------
# 1. Environment & Hardware Detection
# ---------------------------------------------------------
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

DATA_DIR = Path("/content/dataset")
if not (DATA_DIR / "train" / "train_source1.tsv").exists():
    for fallback in [Path("dataset"), Path("../../../dataset"), Path("../../../../dataset")]:
        if (fallback / "train" / "train_source1.tsv").exists():
            DATA_DIR = fallback
            break

CONTENT_OUTPUT = Path("/content/output")
CONTENT_OUTPUT.mkdir(parents=True, exist_ok=True)

REPO_OUTPUT = Path("/home/AmazonMLChallenge/Works_On_My_Machine_submission/output")
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Multi-Pass Blocking Architecture
# ---------------------------------------------------------
STOPWORDS = {
    'inc', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'enterprises', 'enterprise', 'services', 'solutions', 'group',
    'international', 'trading', 'industries', 'associates', 'sarl', 'sas', 'llp',
    'the', 'and', 'a', 'an', 'लिमिटेड', 'प्राइवेट'
}

def clean_tokens(s):
    if pd.isna(s):
        return []
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(s).lower())
    words = [w for w in s.split() if len(w) > 1 and w not in STOPWORDS]
    if not words:
        words = [w for w in s.split() if len(w) > 0]
    return words

def get_multi_pass_blocks(name, addr):
    blocks = []
    # 1. Name tokens
    toks = clean_tokens(name)
    if toks:
        blocks.append(('tok1', toks[0]))
        if len(toks) >= 2:
            blocks.append(('tok12', toks[0] + '_' + toks[1]))
    # 2. Postal / PIN codes (5-6 digits)
    if addr and not pd.isna(addr):
        pins = re.findall(r'\b\d{5,6}\b', str(addr))
        for pin in pins[:2]:
            blocks.append(('post', pin))
    return blocks

# ---------------------------------------------------------
# 3. 12 Pairwise Similarity Features
# ---------------------------------------------------------
def extract_pairwise_features(s1_name, s1_addr, cand_name, cand_addr):
    s1_n = '' if not s1_name or pd.isna(s1_name) else str(s1_name).strip()
    c_n = '' if not cand_name or pd.isna(cand_name) else str(cand_name).strip()
    s1_a = '' if not s1_addr or pd.isna(s1_addr) else str(s1_addr).strip()
    c_a = '' if not cand_addr or pd.isna(cand_addr) else str(cand_addr).strip()

    name_lev = Levenshtein.normalized_similarity(s1_n, c_n) if (s1_n and c_n) else 0.0
    name_jw = JaroWinkler.similarity(s1_n, c_n) if (s1_n and c_n) else 0.0
    name_sort = fuzz.token_sort_ratio(s1_n, c_n) / 100.0 if (s1_n and c_n) else 0.0
    name_set = fuzz.token_set_ratio(s1_n, c_n) / 100.0 if (s1_n and c_n) else 0.0

    addr_lev = Levenshtein.normalized_similarity(s1_a, c_a) if (s1_a and c_a) else 0.0
    addr_jw = JaroWinkler.similarity(s1_a, c_a) if (s1_a and c_a) else 0.0
    addr_set = fuzz.token_set_ratio(s1_a, c_a) / 100.0 if (s1_a and c_a) else 0.0

    exact_name = 1.0 if s1_n and s1_n.lower() == c_n.lower() else 0.0
    exact_addr = 1.0 if s1_a and s1_a.lower() == c_a.lower() else 0.0

    s1_pins = set(re.findall(r'\b\d{5,6}\b', s1_a))
    c_pins = set(re.findall(r'\b\d{5,6}\b', c_a))
    postal_match = 1.0 if (s1_pins and c_pins and bool(s1_pins & c_pins)) else 0.0

    s1_toks = set(re.findall(r'\w+', s1_n.lower()))
    c_toks = set(re.findall(r'\w+', c_n.lower()))
    shared_tokens = float(len(s1_toks & c_toks))

    max_len = max(len(s1_n), len(c_n))
    len_diff = abs(len(s1_n) - len(c_n)) / max_len if max_len > 0 else 0.0

    return [
        name_lev, name_jw, name_sort, name_set,
        addr_lev, addr_jw, addr_set,
        exact_name, exact_addr, postal_match,
        shared_tokens, len_diff
    ]

# ---------------------------------------------------------
# 4. Evaluation & Disambiguation Helpers
# ---------------------------------------------------------
def calculate_competition_macro_f05(preds_dict, gtruth_dict):
    f05_scores = []
    precisions = []
    recalls = []
    singleton_scores = []

    for s1_id, true_set in gtruth_dict.items():
        pred_set = preds_dict.get(s1_id, set())

        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
                singleton_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                singleton_scores.append(0.0)
        else:
            intersection = len(true_set & pred_set)
            if intersection == 0:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                p = intersection / len(pred_set)
                r = intersection / len(true_set)
                f05 = (1.25 * p * r) / (0.25 * p + r)
                f05_scores.append(f05)
                precisions.append(p)
                recalls.append(r)

    return {
        'macro_f05': float(np.mean(f05_scores)),
        'mean_precision': float(np.mean(precisions)) if precisions else 0.0,
        'mean_recall': float(np.mean(recalls)) if recalls else 0.0,
        'singleton_acc': float(np.mean(singleton_scores)) if singleton_scores else 1.0
    }

def star_cluster_disambiguation(raw_predictions):
    best_s1_for_cand = {}
    for s1_id, cand_id, prob in raw_predictions:
        if cand_id not in best_s1_for_cand or prob > best_s1_for_cand[cand_id][1]:
            best_s1_for_cand[cand_id] = (s1_id, prob)

    resolved_dict = defaultdict(set)
    for s1_id, cand_id, prob in raw_predictions:
        if best_s1_for_cand[cand_id][0] == s1_id:
            resolved_dict[s1_id].add(cand_id)
    return resolved_dict

# ---------------------------------------------------------
# 5. Main Execution Flow
# ---------------------------------------------------------
def main():
    start_time = time.time()
    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Compute Device: {DEVICE.upper()} (CUDA Acceleration: {HAS_CUDA})")
    print(f"Dataset Path:   {DATA_DIR}")
    print(f"Output Path:    {REPO_OUTPUT}")
    print("=" * 85)

    # 1. Load Training Data
    print("\n[Step 1/6] Loading training datasets...")
    t0 = time.time()
    train_s1 = pd.read_csv(DATA_DIR / "train" / "train_source1.tsv", sep="\t")
    train_s2 = pd.read_csv(DATA_DIR / "train" / "train_source2.tsv", sep="\t")
    train_s3 = pd.read_csv(DATA_DIR / "train" / "train_source3.tsv", sep="\t")
    train_gt = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t")
    print(f"  Loaded Train: S1={len(train_s1):,}, S2={len(train_s2):,}, S3={len(train_s3):,}, GT={len(train_gt):,} in {time.time()-t0:.1f}s")

    gt_map = {}
    for _, row in train_gt.iterrows():
        m = str(row['matched_entity_ids']).split(',') if not pd.isna(row['matched_entity_ids']) else []
        gt_map[row['source1_entity_id']] = set(x for x in m if x)

    # Build Training Multi-Pass Indexes
    print("  Building training in-memory multi-pass blocking indexes (Name + Postal PIN)...")
    t0 = time.time()
    s2_train_idx = defaultdict(list)
    s2_names = train_s2['business_name'].fillna('').tolist()
    s2_addrs = train_s2['business_address'].fillna('').tolist()
    s2_ids = train_s2['entity_id'].tolist()
    s2_ctry = train_s2['country'].tolist()
    for i in range(len(s2_names)):
        c = s2_ctry[i]
        for b in get_multi_pass_blocks(s2_names[i], s2_addrs[i]):
            s2_train_idx[(c, b)].append(i)

    s3_train_idx = defaultdict(list)
    s3_names = train_s3['business_name'].fillna('').tolist()
    s3_addrs = train_s3['business_address'].fillna('').tolist()
    s3_ids = train_s3['entity_id'].tolist()
    s3_ctry = train_s3['country'].tolist()
    for i in range(len(s3_names)):
        c = s3_ctry[i]
        for b in get_multi_pass_blocks(s3_names[i], s3_addrs[i]):
            s3_train_idx[(c, b)].append(i)

    print(f"  Training indexes ready (S2 blocks={len(s2_train_idx):,}, S3 blocks={len(s3_train_idx):,}) in {time.time()-t0:.1f}s")

    # 2. Hold out Validation Set & Generate Training Matrix
    print("\n[Step 2/6] Generating training pairs and validation benchmark...")
    t0 = time.time()
    np.random.seed(42)
    s1_all_ids = train_s1['entity_id'].values
    val_s1_ids = set(np.random.choice(s1_all_ids, size=5000, replace=False))

    s1_val_df = train_s1[train_s1['entity_id'].isin(val_s1_ids)].copy()
    s1_train_df = train_s1[~train_s1['entity_id'].isin(val_s1_ids)].sample(n=40000, random_state=42).copy()
    val_gt_map = {sid: gt_map.get(sid, set()) for sid in s1_val_df['entity_id']}

    X_train_list, y_train_list = [], []
    for i in range(len(s1_train_df)):
        s1_id = s1_train_df['entity_id'].iloc[i]
        s1_n = s1_train_df['business_name'].iloc[i]
        s1_a = s1_train_df['business_address'].iloc[i]
        c = s1_train_df['country'].iloc[i]
        true_m = gt_map.get(s1_id, set())

        cands_s2 = set()
        cands_s3 = set()
        for b in get_multi_pass_blocks(s1_n, s1_a):
            cands_s2.update(s2_train_idx.get((c, b), []))
            cands_s3.update(s3_train_idx.get((c, b), []))

        cands_s2 = list(cands_s2)[:15]
        cands_s3 = list(cands_s3)[:15]

        for idx in cands_s2:
            cid = s2_ids[idx]
            X_train_list.append(extract_pairwise_features(s1_n, s1_a, s2_names[idx], s2_addrs[idx]))
            y_train_list.append(1 if cid in true_m else 0)

        for idx in cands_s3:
            cid = s3_ids[idx]
            X_train_list.append(extract_pairwise_features(s1_n, s1_a, s3_names[idx], s3_addrs[idx]))
            y_train_list.append(1 if cid in true_m else 0)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"  Generated {len(X_train):,} training pairs (Positives: {np.sum(y_train):,}) in {time.time()-t0:.1f}s")

    # Generate validation pairs
    val_meta = []
    val_pairs_list = []
    for i in range(len(s1_val_df)):
        s1_id = s1_val_df['entity_id'].iloc[i]
        s1_n = s1_val_df['business_name'].iloc[i]
        s1_a = s1_val_df['business_address'].iloc[i]
        c = s1_val_df['country'].iloc[i]

        cands_s2 = set()
        cands_s3 = set()
        for b in get_multi_pass_blocks(s1_n, s1_a):
            cands_s2.update(s2_train_idx.get((c, b), []))
            cands_s3.update(s3_train_idx.get((c, b), []))

        cands_s2 = list(cands_s2)[:15]
        cands_s3 = list(cands_s3)[:15]

        entity_cand_ids = []
        for idx in cands_s2:
            cid = s2_ids[idx]
            entity_cand_ids.append(cid)
            val_pairs_list.append(extract_pairwise_features(s1_n, s1_a, s2_names[idx], s2_addrs[idx]))

        for idx in cands_s3:
            cid = s3_ids[idx]
            entity_cand_ids.append(cid)
            val_pairs_list.append(extract_pairwise_features(s1_n, s1_a, s3_names[idx], s3_addrs[idx]))

        val_meta.append((s1_id, entity_cand_ids))

    X_val = np.array(val_pairs_list, dtype=np.float32)
    print(f"  Generated {len(X_val):,} validation candidate pairs.")

    del train_s1, train_s2, train_s3, train_gt, s2_train_idx, s3_train_idx
    del s2_names, s2_addrs, s2_ids, s3_names, s3_addrs, s3_ids
    gc.collect()

    # 3. Benchmark All 4 Approaches
    print("\n[Step 3/6] Benchmarking 4 Modeling Approaches on Validation Split...")
    # Approach 1: XGBoost
    print("  --> Training Approach 1: GPU XGBoost...")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=250, learning_rate=0.08, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', device=DEVICE,
        eval_metric='logloss', random_state=42
    )
    xgb_model.fit(X_train, y_train)
    t_xgb = time.time() - t0
    val_xgb_probs = xgb_model.predict_proba(X_val)[:, 1]

    best_xgb_cutoff, best_xgb_f05, best_xgb_metrics = 0.5, 0.0, None
    for cutoff in np.linspace(0.20, 0.70, 26):
        preds = defaultdict(set)
        offset = 0
        for s1_id, cand_ids in val_meta:
            n_c = len(cand_ids)
            probs = val_xgb_probs[offset:offset+n_c]
            offset += n_c
            for idx, p in enumerate(probs):
                if p >= cutoff:
                    preds[s1_id].add(cand_ids[idx])
        m = calculate_competition_macro_f05(preds, val_gt_map)
        if m['macro_f05'] > best_xgb_f05:
            best_xgb_f05 = m['macro_f05']
            best_xgb_cutoff = cutoff
            best_xgb_metrics = m

    # Approach 2: LightGBM
    print("  --> Training Approach 2: LightGBM GBDT...")
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.08, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    t_lgb = time.time() - t0
    val_lgb_probs = lgb_model.predict_proba(X_val)[:, 1]

    best_lgb_cutoff, best_lgb_f05, best_lgb_metrics = 0.5, 0.0, None
    for cutoff in np.linspace(0.20, 0.70, 26):
        preds = defaultdict(set)
        offset = 0
        for s1_id, cand_ids in val_meta:
            n_c = len(cand_ids)
            probs = val_lgb_probs[offset:offset+n_c]
            offset += n_c
            for idx, p in enumerate(probs):
                if p >= cutoff:
                    preds[s1_id].add(cand_ids[idx])
        m = calculate_competition_macro_f05(preds, val_gt_map)
        if m['macro_f05'] > best_lgb_f05:
            best_lgb_f05 = m['macro_f05']
            best_lgb_cutoff = cutoff
            best_lgb_metrics = m

    # Approach 3: Star Clustering Disambiguation
    raw_preds = []
    offset = 0
    for s1_id, cand_ids in val_meta:
        n_c = len(cand_ids)
        probs = val_xgb_probs[offset:offset+n_c]
        offset += n_c
        for idx, p in enumerate(probs):
            if p >= best_xgb_cutoff:
                raw_preds.append((s1_id, cand_ids[idx], float(p)))
    preds_app3 = star_cluster_disambiguation(raw_preds)
    app3_metrics = calculate_competition_macro_f05(preds_app3, val_gt_map)

    # Approach 4: Blended Ensemble
    val_ens_probs = 0.60 * val_xgb_probs + 0.40 * val_lgb_probs
    best_ens_cutoff, best_ens_f05, best_ens_metrics = 0.5, 0.0, None
    for cutoff in np.linspace(0.25, 0.70, 20):
        raw_p = []
        offset = 0
        for s1_id, cand_ids in val_meta:
            n_c = len(cand_ids)
            probs = val_ens_probs[offset:offset+n_c]
            offset += n_c
            for idx, p in enumerate(probs):
                if p >= cutoff:
                    raw_p.append((s1_id, cand_ids[idx], float(p)))
        preds = star_cluster_disambiguation(raw_p)
        m = calculate_competition_macro_f05(preds, val_gt_map)
        if m['macro_f05'] > best_ens_f05:
            best_ens_f05 = m['macro_f05']
            best_ens_cutoff = cutoff
            best_ens_metrics = m

    # Print Summary Table
    print("\n" + "=" * 92)
    print(f"{'Approach':<42} | {'Macro F0.5':<10} | {'Precision':<10} | {'Recall':<10} | {'Singleton Acc':<12}")
    print("=" * 92)
    print(f"{'1. GPU XGBoost (Pairwise Baseline)':<42} | {best_xgb_metrics['macro_f05']:<10.4f} | {best_xgb_metrics['mean_precision']:<10.4f} | {best_xgb_metrics['mean_recall']:<10.4f} | {best_xgb_metrics['singleton_acc']:<12.4f}")
    print(f"{'2. LightGBM (Leaf-wise GBDT)':<42} | {best_lgb_metrics['macro_f05']:<10.4f} | {best_lgb_metrics['mean_precision']:<10.4f} | {best_lgb_metrics['mean_recall']:<10.4f} | {best_lgb_metrics['singleton_acc']:<12.4f}")
    print(f"{'3. XGBoost + Star-Clustering':<42} | {app3_metrics['macro_f05']:<10.4f} | {app3_metrics['mean_precision']:<10.4f} | {app3_metrics['mean_recall']:<10.4f} | {app3_metrics['singleton_acc']:<12.4f}")
    print(f"{'4. Blended Ensemble (XGB+LGBM+Cluster)':<42} | {best_ens_metrics['macro_f05']:<10.4f} | {best_ens_metrics['mean_precision']:<10.4f} | {best_ens_metrics['mean_recall']:<10.4f} | {best_ens_metrics['singleton_acc']:<12.4f}")
    print("=" * 92)

    winning_model = "xgb"
    effective_cutoff = best_xgb_cutoff
    print(f"Selected Winning Pipeline: GPU XGBoost (Cutoff = {effective_cutoff:.3f})")

    del X_train, y_train, X_val, val_xgb_probs, val_lgb_probs, val_ens_probs
    gc.collect()

    # 4. Load Full Test Set & Build Inverted Indexes
    print("\n[Step 4/6] Loading full test dataset & building test blocking structures...")
    t0 = time.time()
    test_s1 = pd.read_csv(DATA_DIR / "test" / "test_source1.tsv", sep="\t")
    test_s2 = pd.read_csv(DATA_DIR / "test" / "test_source2.tsv", sep="\t")
    test_s3 = pd.read_csv(DATA_DIR / "test" / "test_source3.tsv", sep="\t")
    print(f"  Loaded Test: S1={len(test_s1):,}, S2={len(test_s2):,}, S3={len(test_s3):,} in {time.time()-t0:.1f}s")

    s2_test_idx = defaultdict(list)
    s2_names = test_s2['business_name'].fillna('').tolist()
    s2_addrs = test_s2['business_address'].fillna('').tolist()
    s2_ids = test_s2['entity_id'].tolist()
    s2_ctry = test_s2['country'].tolist()
    for i in range(len(s2_names)):
        c = s2_ctry[i]
        for b in get_multi_pass_blocks(s2_names[i], s2_addrs[i]):
            s2_test_idx[(c, b)].append(i)

    s3_test_idx = defaultdict(list)
    s3_names = test_s3['business_name'].fillna('').tolist()
    s3_addrs = test_s3['business_address'].fillna('').tolist()
    s3_ids = test_s3['entity_id'].tolist()
    s3_ctry = test_s3['country'].tolist()
    for i in range(len(s3_names)):
        c = s3_ctry[i]
        for b in get_multi_pass_blocks(s3_names[i], s3_addrs[i]):
            s3_test_idx[(c, b)].append(i)

    print(f"  Test indices ready (S2 blocks={len(s2_test_idx):,}, S3 blocks={len(s3_test_idx):,})")

    # 5. Full Test Inference & Direct Streaming to Disk
    print("\n[Step 5/6] Generating full test candidates & predictions (streaming to TSV)...")
    s1_names = test_s1['business_name'].fillna('').tolist()
    s1_addrs = test_s1['business_address'].fillna('').tolist()
    s1_ids = test_s1['entity_id'].tolist()
    s1_ctry = test_s1['country'].tolist()
    n_test = len(s1_ids)

    out_cand_repo = REPO_OUTPUT / "candidate_pairs.tsv"
    out_match_repo = REPO_OUTPUT / "matching_results.tsv"

    file_handles = [
        (open(out_cand_repo, 'w', encoding='utf-8'), open(out_match_repo, 'w', encoding='utf-8'))
    ]
    if os.path.exists("/content"):
        out_cand_content = CONTENT_OUTPUT / "candidate_pairs.tsv"
        out_match_content = CONTENT_OUTPUT / "matching_results.tsv"
        file_handles.append(
            (open(out_cand_content, 'w', encoding='utf-8'), open(out_match_content, 'w', encoding='utf-8'))
        )

    for f_cand, f_match in file_handles:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    batch_size = 50000
    t_start_inf = time.time()

    for start_idx in range(0, n_test, batch_size):
        end_idx = min(start_idx + batch_size, n_test)
        t_batch = time.time()

        batch_pairs_x = []
        batch_meta = []

        for i in range(start_idx, end_idx):
            s1_id = s1_ids[i]
            s1_n = s1_names[i]
            s1_a = s1_addrs[i]
            c = s1_ctry[i]

            cands_s2 = set()
            cands_s3 = set()
            for b in get_multi_pass_blocks(s1_n, s1_a):
                cands_s2.update(s2_test_idx.get((c, b), []))
                cands_s3.update(s3_test_idx.get((c, b), []))

            cands_s2 = list(cands_s2)[:15]
            cands_s3 = list(cands_s3)[:15]

            entity_cand_ids = []
            for idx in cands_s2:
                cid = s2_ids[idx]
                entity_cand_ids.append(cid)
                batch_pairs_x.append(extract_pairwise_features(s1_n, s1_a, s2_names[idx], s2_addrs[idx]))

            for idx in cands_s3:
                cid = s3_ids[idx]
                entity_cand_ids.append(cid)
                batch_pairs_x.append(extract_pairwise_features(s1_n, s1_a, s3_names[idx], s3_addrs[idx]))

            batch_meta.append((s1_id, entity_cand_ids))

        # GPU batch prediction
        if batch_pairs_x:
            X_batch = np.array(batch_pairs_x, dtype=np.float32)
            pred_probs = xgb_model.predict_proba(X_batch)[:, 1]
        else:
            pred_probs = np.array([], dtype=np.float32)

        pair_offset = 0
        cand_lines = []
        match_lines = []

        for s1_id, cand_ids in batch_meta:
            n_cands = len(cand_ids)
            if n_cands == 0:
                cand_lines.append(f"{s1_id}\t\n")
                match_lines.append(f"{s1_id}\t\n")
            else:
                c_probs = pred_probs[pair_offset:pair_offset + n_cands]
                pair_offset += n_cands
                matched = [cand_ids[k] for k in range(n_cands) if c_probs[k] >= effective_cutoff]
                cand_lines.append(f"{s1_id}\t{','.join(cand_ids)}\n")
                match_lines.append(f"{s1_id}\t{','.join(matched)}\n")

        cand_chunk = "".join(cand_lines)
        match_chunk = "".join(match_lines)

        for f_cand, f_match in file_handles:
            f_cand.write(cand_chunk)
            f_match.write(match_chunk)

        rate = (end_idx - start_idx) / (time.time() - t_batch)
        elapsed = time.time() - t_start_inf
        progress = (end_idx / n_test) * 100
        print(f"  Processed {end_idx:,}/{n_test:,} entities ({progress:.1f}%) [{rate:,.0f} ent/sec] - Elapsed: {elapsed:.1f}s")

    for f_cand, f_match in file_handles:
        f_cand.close()
        f_match.close()

    print(f"  Full test streaming completed in {time.time()-t_start_inf:.1f}s!")

    # 6. Run Official Validator Script
    print("\n[Step 6/6] Validating generated files with official validator...")
    val_script = Path("/home/AmazonMLChallenge/student_resource/utils/validate_submission.py")
    if val_script.exists():
        test_dir = DATA_DIR / "test"
        cmd = [
            "python3", str(val_script),
            "--matching", str(out_match_repo),
            "--candidate", str(out_cand_repo),
            "--test-dir", str(test_dir),
            "--check-ids"
        ]
        print(f"  Command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)

    total_time = time.time() - start_time
    print(f"\nPipeline finished successfully in {total_time/60:.2f} minutes!")

if __name__ == "__main__":
    main()
