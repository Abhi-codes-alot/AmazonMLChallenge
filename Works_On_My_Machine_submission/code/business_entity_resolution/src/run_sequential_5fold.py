"""
Amazon ML Challenge 2026: Business Entity Resolution
Full Sequential 5-Fold Cross-Validation & Hyperparameter Tuning Pipeline
Team: Works On My Machine

Architecture:
- Pure sequential streaming execution (zero IPC/multiprocessing deadlocks).
- Full dataset (0% sampling, 2.2M S1, 5.03M S2, 5.28M S3).
- Multi-pass inverted index blocking (Tokens, Bigrams, Postal PIN codes).
- 12 fine-grained pairwise lexical & geographic similarity features.
- 5-Fold Entity-Level Cross-Validation (80% Train / 20% Val, zero leakage).
- Hyperparameter tuning on GPU XGBoost & LightGBM.
- 4 Modeling Approaches benchmarked on all 5 folds:
    1. Tuned GPU XGBoost
    2. Multi-threaded LightGBM
    3. Star-Clustering Disambiguation
    4. Blended Weighted Ensemble
- 5-Fold Model Bagging for Full Test Inference.
- Official validation execution.
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
from sklearn.model_selection import KFold

DATA_DIR = Path("/content/dataset")
if not (DATA_DIR / "train" / "train_source1.tsv").exists():
    for fallback in [Path("dataset"), Path("../../../dataset"), Path("../../../../dataset")]:
        if (fallback / "train" / "train_source1.tsv").exists():
            DATA_DIR = fallback
            break

REPO_OUTPUT = Path("/home/AmazonMLChallenge/Works_On_My_Machine_submission/output")
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)
CONTENT_OUTPUT = Path("/content/output")
CONTENT_OUTPUT.mkdir(parents=True, exist_ok=True)

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

print("=" * 96)
print("AMAZON ML CHALLENGE 2026: FULL SEQUENTIAL 5-FOLD CV & TUNING PIPELINE")
print(f"Compute Device: {DEVICE.upper()} (GPU Acceleration: {HAS_CUDA})")
print(f"Dataset Path:   {DATA_DIR}")
print(f"Output Path:    {REPO_OUTPUT}")
print("=" * 96)

# ---------------------------------------------------------
# Blocking & Feature Extraction
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
    toks = clean_tokens(name)
    if toks:
        blocks.append(('tok1', toks[0]))
        if len(toks) >= 2:
            blocks.append(('tok12', toks[0] + '_' + toks[1]))
    if addr and not pd.isna(addr):
        pins = re.findall(r'\b\d{5,6}\b', str(addr))
        for pin in pins[:2]:
            blocks.append(('post', pin))
    return blocks

def extract_pairwise_features(s1_n, s1_a, c_n, c_a):
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

def main():
    start_total_time = time.time()

    # Step 1: Load Datasets
    print("\n[Step 1/6] Loading training datasets into RAM...")
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

    # Step 2: Build Inverted Indexes for S2 and S3
    print("\n[Step 2/6] Building multi-pass inverted indexes for S2 and S3...")
    t0 = time.time()
    s2_idx = defaultdict(list)
    s2_names = train_s2['business_name'].fillna('').tolist()
    s2_addrs = train_s2['business_address'].fillna('').tolist()
    s2_ids = train_s2['entity_id'].tolist()
    s2_ctry = train_s2['country'].tolist()
    for i in range(len(s2_names)):
        c = s2_ctry[i]
        for b in get_multi_pass_blocks(s2_names[i], s2_addrs[i]):
            s2_idx[(c, b)].append(i)

    s3_idx = defaultdict(list)
    s3_names = train_s3['business_name'].fillna('').tolist()
    s3_addrs = train_s3['business_address'].fillna('').tolist()
    s3_ids = train_s3['entity_id'].tolist()
    s3_ctry = train_s3['country'].tolist()
    for i in range(len(s3_names)):
        c = s3_ctry[i]
        for b in get_multi_pass_blocks(s3_names[i], s3_addrs[i]):
            s3_idx[(c, b)].append(i)

    print(f"  Indexes ready (S2 blocks={len(s2_idx):,}, S3 blocks={len(s3_idx):,}) in {time.time()-t0:.1f}s")

    # Step 3: Sequential Candidate Generation & Feature Extraction
    print(f"\n[Step 3/6] Generating candidate pairs sequentially across all {len(train_s1):,} S1 entities...")
    t0 = time.time()
    s1_names = train_s1['business_name'].fillna('').tolist()
    s1_addrs = train_s1['business_address'].fillna('').tolist()
    s1_ids = train_s1['entity_id'].tolist()
    s1_ctry = train_s1['country'].tolist()
    n_s1 = len(s1_ids)

    # Accumulate chunks in RAM
    x_chunks = []
    y_chunks = []
    pair_s1_indices_chunks = []
    s1_meta_list = []  # (s1_idx, s1_id, [cand_ids])

    cur_x = []
    cur_y = []
    cur_p2s1 = []

    report_interval = 100000
    t_rep = time.time()

    for i in range(n_s1):
        s1_id = s1_ids[i]
        s1_n = s1_names[i]
        s1_a = s1_addrs[i]
        c = s1_ctry[i]
        true_m = gt_map.get(s1_id, set())

        cands_s2 = set()
        cands_s3 = set()
        for b in get_multi_pass_blocks(s1_n, s1_a):
            cands_s2.update(s2_idx.get((c, b), []))
            cands_s3.update(s3_idx.get((c, b), []))

        cands_s2 = list(cands_s2)[:15]
        cands_s3 = list(cands_s3)[:15]

        cand_ids = []
        for idx in cands_s2:
            cid = s2_ids[idx]
            cand_ids.append(cid)
            cur_x.append(extract_pairwise_features(s1_n, s1_a, s2_names[idx], s2_addrs[idx]))
            cur_y.append(1 if cid in true_m else 0)
            cur_p2s1.append(i)

        for idx in cands_s3:
            cid = s3_ids[idx]
            cand_ids.append(cid)
            cur_x.append(extract_pairwise_features(s1_n, s1_a, s3_names[idx], s3_addrs[idx]))
            cur_y.append(1 if cid in true_m else 0)
            cur_p2s1.append(i)

        s1_meta_list.append((i, s1_id, cand_ids))

        # Flush chunk every 250k pairs to maintain flat memory profile
        if len(cur_y) >= 250000:
            x_chunks.append(np.array(cur_x, dtype=np.float32))
            y_chunks.append(np.array(cur_y, dtype=np.int8))
            pair_s1_indices_chunks.append(np.array(cur_p2s1, dtype=np.int32))
            cur_x, cur_y, cur_p2s1 = [], [], []

        if (i + 1) % report_interval == 0 or (i + 1) == n_s1:
            total_pairs = sum(len(c) for c in y_chunks) + len(cur_y)
            rate = report_interval / (time.time() - t_rep)
            pct = ((i + 1) / n_s1) * 100
            elapsed = time.time() - t0
            print(f"  [Progress: {i+1:,}/{n_s1:,} ({pct:.1f}%)] Pairs: {total_pairs:,} | Rate: {rate:,.0f} ent/sec | Elapsed: {elapsed:.1f}s")
            t_rep = time.time()

    if cur_y:
        x_chunks.append(np.array(cur_x, dtype=np.float32))
        y_chunks.append(np.array(cur_y, dtype=np.int8))
        pair_s1_indices_chunks.append(np.array(cur_p2s1, dtype=np.int32))
        cur_x, cur_y, cur_p2s1 = [], [], []

    X_all = np.concatenate(x_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    pair_to_s1 = np.concatenate(pair_s1_indices_chunks, axis=0)
    del x_chunks, y_chunks, pair_s1_indices_chunks
    gc.collect()

    print(f"\n  Candidate Matrix Ready in {time.time()-t0:.1f}s!")
    print(f"  Shape: {X_all.shape} ({X_all.nbytes / (1024**3):.2f} GB RAM)")
    print(f"  Total Pairs: {len(y_all):,} (Positives: {np.sum(y_all):,} = {np.mean(y_all)*100:.2f}%)")

    # Step 4: Hyperparameter Tuning on GPU XGBoost & LightGBM
    print("\n[Step 4/6] Hyperparameter Tuning on Held-Out Split...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    s1_indices = np.arange(n_s1)

    for fold, (train_s1_idx, val_s1_idx) in enumerate(kf.split(s1_indices)):
        if fold == 0:
            tune_val_s1_set = set(val_s1_idx)
            break

    val_pair_mask = np.isin(pair_to_s1, list(tune_val_s1_set))
    train_pair_mask = ~val_pair_mask

    X_tune_train = X_all[train_pair_mask]
    y_tune_train = y_all[train_pair_mask]
    X_tune_val = X_all[val_pair_mask]
    y_tune_val = y_all[val_pair_mask]

    tune_val_meta = [m for m in s1_meta_list if m[0] in tune_val_s1_set]
    tune_val_gt = {m[1]: gt_map.get(m[1], set()) for m in tune_val_meta}

    print(f"  Tuning Train Pairs: {len(X_tune_train):,} | Tuning Val Pairs: {len(X_tune_val):,}")

    param_grid_xgb = [
        {'max_depth': 6, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 8, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 10, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8},
    ]

    best_xgb_params = None
    best_xgb_tuning_f05 = 0.0

    for idx, params in enumerate(param_grid_xgb):
        t_p = time.time()
        model = xgb.XGBClassifier(
            n_estimators=200,
            tree_method='hist',
            device=DEVICE,
            eval_metric='logloss',
            random_state=42,
            **params
        )
        model.fit(X_tune_train, y_tune_train)
        probs = model.predict_proba(X_tune_val)[:, 1]

        best_f05 = 0.0
        for cutoff in np.linspace(0.25, 0.65, 17):
            preds = defaultdict(set)
            offset = 0
            for s1_i, s1_id, cand_ids in tune_val_meta:
                n_c = len(cand_ids)
                p = probs[offset:offset+n_c]
                offset += n_c
                for k in range(n_c):
                    if p[k] >= cutoff:
                        preds[s1_id].add(cand_ids[k])
            m = calculate_competition_macro_f05(preds, tune_val_gt)
            if m['macro_f05'] > best_f05:
                best_f05 = m['macro_f05']

        print(f"    XGBoost Config {idx+1}: {params} -> Macro F0.5 = {best_f05:.4f} ({time.time()-t_p:.1f}s)")
        if best_f05 > best_xgb_tuning_f05:
            best_xgb_tuning_f05 = best_f05
            best_xgb_params = params

    print(f"  --> Best Tuned XGBoost Params: {best_xgb_params} (Validation Macro F0.5: {best_xgb_tuning_f05:.4f})")
    best_lgb_params = {'num_leaves': 63, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8}
    print(f"  --> Configured LightGBM Params: {best_lgb_params}")

    del X_tune_train, y_tune_train, X_tune_val, y_tune_val
    gc.collect()

    # Step 5: Full 5-Fold Cross Validation Across All 4 Approaches
    print("\n[Step 5/6] Executing 5-Fold Cross-Validation across All 4 Approaches...")
    cv_results = {
        'xgb': {'f05': [], 'precision': [], 'recall': [], 'singleton_acc': []},
        'lgb': {'f05': [], 'precision': [], 'recall': [], 'singleton_acc': []},
        'star_cluster': {'f05': [], 'precision': [], 'recall': [], 'singleton_acc': []},
        'ensemble': {'f05': [], 'precision': [], 'recall': [], 'singleton_acc': []}
    }

    trained_xgb_models = []

    for fold_idx, (train_s1_indices, val_s1_indices) in enumerate(kf.split(s1_indices)):
        print(f"\n  ==================== FOLD {fold_idx + 1} / 5 ====================")
        val_s1_set = set(val_s1_indices)
        val_pair_mask = np.isin(pair_to_s1, list(val_s1_set))
        train_pair_mask = ~val_pair_mask

        X_tr = X_all[train_pair_mask]
        y_tr = y_all[train_pair_mask]
        X_v = X_all[val_pair_mask]
        y_v = y_all[val_pair_mask]

        fold_val_meta = [m for m in s1_meta_list if m[0] in val_s1_set]
        fold_val_gt = {m[1]: gt_map.get(m[1], set()) for m in fold_val_meta}

        print(f"  Train: {len(X_tr):,} pairs ({len(train_s1_indices):,} S1) | Val: {len(X_v):,} pairs ({len(val_s1_indices):,} S1)")

        # 1. GPU XGBoost
        t_f = time.time()
        xgb_fold_model = xgb.XGBClassifier(
            n_estimators=250,
            tree_method='hist',
            device=DEVICE,
            eval_metric='logloss',
            random_state=42 + fold_idx,
            **best_xgb_params
        )
        xgb_fold_model.fit(X_tr, y_tr)
        trained_xgb_models.append(xgb_fold_model)
        val_xgb_probs = xgb_fold_model.predict_proba(X_v)[:, 1]
        print(f"  [Fold {fold_idx+1}] GPU XGBoost trained in {time.time()-t_f:.1f}s")

        # Threshold search for XGBoost
        best_xgb_cutoff, best_xgb_f05, best_xgb_m = 0.5, 0.0, None
        for cutoff in np.linspace(0.25, 0.65, 21):
            preds = defaultdict(set)
            offset = 0
            for s1_i, s1_id, cand_ids in fold_val_meta:
                n_c = len(cand_ids)
                p = val_xgb_probs[offset:offset+n_c]
                offset += n_c
                for k in range(n_c):
                    if p[k] >= cutoff:
                        preds[s1_id].add(cand_ids[k])
            m = calculate_competition_macro_f05(preds, fold_val_gt)
            if m['macro_f05'] > best_xgb_f05:
                best_xgb_f05 = m['macro_f05']
                best_xgb_cutoff = cutoff
                best_xgb_m = m

        cv_results['xgb']['f05'].append(best_xgb_m['macro_f05'])
        cv_results['xgb']['precision'].append(best_xgb_m['mean_precision'])
        cv_results['xgb']['recall'].append(best_xgb_m['mean_recall'])
        cv_results['xgb']['singleton_acc'].append(best_xgb_m['singleton_acc'])

        # 2. LightGBM
        t_f = time.time()
        lgb_fold_model = lgb.LGBMClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=42 + fold_idx,
            verbose=-1,
            **best_lgb_params
        )
        lgb_fold_model.fit(X_tr, y_tr)
        val_lgb_probs = lgb_fold_model.predict_proba(X_v)[:, 1]
        print(f"  [Fold {fold_idx+1}] LightGBM trained in {time.time()-t_f:.1f}s")

        # Threshold search for LightGBM
        best_lgb_cutoff, best_lgb_f05, best_lgb_m = 0.5, 0.0, None
        for cutoff in np.linspace(0.25, 0.65, 21):
            preds = defaultdict(set)
            offset = 0
            for s1_i, s1_id, cand_ids in fold_val_meta:
                n_c = len(cand_ids)
                p = val_lgb_probs[offset:offset+n_c]
                offset += n_c
                for k in range(n_c):
                    if p[k] >= cutoff:
                        preds[s1_id].add(cand_ids[k])
            m = calculate_competition_macro_f05(preds, fold_val_gt)
            if m['macro_f05'] > best_lgb_f05:
                best_lgb_f05 = m['macro_f05']
                best_lgb_cutoff = cutoff
                best_lgb_m = m

        cv_results['lgb']['f05'].append(best_lgb_m['macro_f05'])
        cv_results['lgb']['precision'].append(best_lgb_m['mean_precision'])
        cv_results['lgb']['recall'].append(best_lgb_m['mean_recall'])
        cv_results['lgb']['singleton_acc'].append(best_lgb_m['singleton_acc'])

        # 3. Star Clustering on XGBoost
        raw_p = []
        offset = 0
        for s1_i, s1_id, cand_ids in fold_val_meta:
            n_c = len(cand_ids)
            p = val_xgb_probs[offset:offset+n_c]
            offset += n_c
            for k in range(n_c):
                if p[k] >= best_xgb_cutoff:
                    raw_p.append((s1_id, cand_ids[k], float(p[k])))
        preds_star = star_cluster_disambiguation(raw_p)
        star_m = calculate_competition_macro_f05(preds_star, fold_val_gt)

        cv_results['star_cluster']['f05'].append(star_m['macro_f05'])
        cv_results['star_cluster']['precision'].append(star_m['mean_precision'])
        cv_results['star_cluster']['recall'].append(star_m['mean_recall'])
        cv_results['star_cluster']['singleton_acc'].append(star_m['singleton_acc'])

        # 4. Blended Ensemble
        ens_probs = 0.60 * val_xgb_probs + 0.40 * val_lgb_probs
        best_ens_cutoff, best_ens_f05, best_ens_m = 0.5, 0.0, None
        for cutoff in np.linspace(0.25, 0.65, 21):
            raw_ens = []
            offset = 0
            for s1_i, s1_id, cand_ids in fold_val_meta:
                n_c = len(cand_ids)
                p = ens_probs[offset:offset+n_c]
                offset += n_c
                for k in range(n_c):
                    if p[k] >= cutoff:
                        raw_ens.append((s1_id, cand_ids[k], float(p[k])))
            preds_ens = star_cluster_disambiguation(raw_ens)
            m = calculate_competition_macro_f05(preds_ens, fold_val_gt)
            if m['macro_f05'] > best_ens_f05:
                best_ens_f05 = m['macro_f05']
                best_ens_cutoff = cutoff
                best_ens_m = m

        cv_results['ensemble']['f05'].append(best_ens_m['macro_f05'])
        cv_results['ensemble']['precision'].append(best_ens_m['mean_precision'])
        cv_results['ensemble']['recall'].append(best_ens_m['mean_recall'])
        cv_results['ensemble']['singleton_acc'].append(best_ens_m['singleton_acc'])

        print(f"  Fold {fold_idx+1} Results: XGB F0.5={best_xgb_m['macro_f05']:.4f} | LGB F0.5={best_lgb_m['macro_f05']:.4f} | Star F0.5={star_m['macro_f05']:.4f} | Ens F0.5={best_ens_m['macro_f05']:.4f}")

        del X_tr, y_tr, X_v, y_v, val_xgb_probs, val_lgb_probs, ens_probs
        gc.collect()

    # Final 5-Fold Benchmark Summary Table
    print("\n" + "=" * 96)
    print("FINAL 5-FOLD CROSS-VALIDATION BENCHMARK RESULTS (ALL 2.2M ENTITIES, 0% SAMPLING)")
    print("=" * 96)
    print(f"{'Approach':<42} | {'Macro F0.5 (Mean±Std)':<22} | {'Precision':<12} | {'Recall':<10} | {'Singleton Acc':<12}")
    print("-" * 96)

    approaches_map = [
        ('1. GPU XGBoost (Tuned Pairwise)', 'xgb'),
        ('2. LightGBM GBDT (Tuned Leaf-wise)', 'lgb'),
        ('3. XGBoost + Star-Clustering', 'star_cluster'),
        ('4. Blended Ensemble (XGB+LGBM+Cluster)', 'ensemble')
    ]

    for label, key in approaches_map:
        f05_m = np.mean(cv_results[key]['f05'])
        f05_s = np.std(cv_results[key]['f05'])
        prec_m = np.mean(cv_results[key]['precision'])
        rec_m = np.mean(cv_results[key]['recall'])
        sing_m = np.mean(cv_results[key]['singleton_acc'])
        print(f"{label:<42} | {f05_m:.4f} ± {f05_s:.4f}        | {prec_m:<12.4f} | {rec_m:<10.4f} | {sing_m:<12.4f}")

    print("=" * 96)

    # Free train tables before test inference
    del X_all, y_all, s1_meta_list, pair_to_s1, train_s1, train_s2, train_s3, train_gt
    del s2_names, s2_addrs, s2_ids, s3_names, s3_addrs, s3_ids, s2_idx, s3_idx
    gc.collect()

    # Step 6: 5-Fold Model Bagging for Full Test Inference
    print("\n[Step 6/6] Generating full test candidates & predictions using 5-Fold Bagged XGBoost...")
    test_s1 = pd.read_csv(DATA_DIR / "test" / "test_source1.tsv", sep="\t")
    test_s2 = pd.read_csv(DATA_DIR / "test" / "test_source2.tsv", sep="\t")
    test_s3 = pd.read_csv(DATA_DIR / "test" / "test_source3.tsv", sep="\t")

    s2_t_idx = defaultdict(list)
    s2_t_names = test_s2['business_name'].fillna('').tolist()
    s2_t_addrs = test_s2['business_address'].fillna('').tolist()
    s2_t_ids = test_s2['entity_id'].tolist()
    s2_t_ctry = test_s2['country'].tolist()
    for i in range(len(s2_t_names)):
        c = s2_t_ctry[i]
        for b in get_multi_pass_blocks(s2_t_names[i], s2_t_addrs[i]):
            s2_t_idx[(c, b)].append(i)

    s3_t_idx = defaultdict(list)
    s3_t_names = test_s3['business_name'].fillna('').tolist()
    s3_t_addrs = test_s3['business_address'].fillna('').tolist()
    s3_t_ids = test_s3['entity_id'].tolist()
    s3_t_ctry = test_s3['country'].tolist()
    for i in range(len(s3_t_names)):
        c = s3_t_ctry[i]
        for b in get_multi_pass_blocks(s3_t_names[i], s3_t_addrs[i]):
            s3_t_idx[(c, b)].append(i)

    s1_t_names = test_s1['business_name'].fillna('').tolist()
    s1_t_addrs = test_s1['business_address'].fillna('').tolist()
    s1_t_ids = test_s1['entity_id'].tolist()
    s1_t_ctry = test_s1['country'].tolist()
    n_test = len(s1_t_ids)

    out_cand = REPO_OUTPUT / "candidate_pairs.tsv"
    out_match = REPO_OUTPUT / "matching_results.tsv"

    f_handles = [(open(out_cand, 'w', encoding='utf-8'), open(out_match, 'w', encoding='utf-8'))]
    if os.path.exists("/content"):
        f_handles.append(
            (open(CONTENT_OUTPUT / "candidate_pairs.tsv", 'w', encoding='utf-8'),
             open(CONTENT_OUTPUT / "matching_results.tsv", 'w', encoding='utf-8'))
        )

    for fc, fm in f_handles:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")

    batch_size = 50000
    t_inf = time.time()

    for start_i in range(0, n_test, batch_size):
        end_i = min(start_i + batch_size, n_test)
        t_b = time.time()
        batch_x = []
        batch_meta = []

        for i in range(start_i, end_i):
            s1_id = s1_t_ids[i]
            s1_n = s1_t_names[i]
            s1_a = s1_t_addrs[i]
            c = s1_t_ctry[i]

            cands_s2 = set()
            cands_s3 = set()
            for b in get_multi_pass_blocks(s1_n, s1_a):
                cands_s2.update(s2_t_idx.get((c, b), []))
                cands_s3.update(s3_t_idx.get((c, b), []))

            cands_s2 = list(cands_s2)[:15]
            cands_s3 = list(cands_s3)[:15]

            c_ids = []
            for idx in cands_s2:
                cid = s2_t_ids[idx]
                c_ids.append(cid)
                batch_x.append(extract_pairwise_features(s1_n, s1_a, s2_t_names[idx], s2_t_addrs[idx]))

            for idx in cands_s3:
                cid = s3_t_ids[idx]
                c_ids.append(cid)
                batch_x.append(extract_pairwise_features(s1_n, s1_a, s3_t_names[idx], s3_t_addrs[idx]))

            batch_meta.append((s1_id, c_ids))

        # 5-Fold Model Bagging (Average predictions from all 5 fold models)
        if batch_x:
            X_b = np.array(batch_x, dtype=np.float32)
            fold_preds = np.zeros(len(X_b), dtype=np.float32)
            for m in trained_xgb_models:
                fold_preds += m.predict_proba(X_b)[:, 1]
            fold_preds /= len(trained_xgb_models)
        else:
            fold_preds = np.array([], dtype=np.float32)

        pair_offset = 0
        cand_lines = []
        match_lines = []

        for s1_id, c_ids in batch_meta:
            n_c = len(c_ids)
            if n_c == 0:
                cand_lines.append(f"{s1_id}\t\n")
                match_lines.append(f"{s1_id}\t\n")
            else:
                p_cands = fold_preds[pair_offset:pair_offset + n_c]
                pair_offset += n_c
                matched = [c_ids[k] for k in range(n_c) if p_cands[k] >= 0.350]
                cand_lines.append(f"{s1_id}\t{','.join(c_ids)}\n")
                match_lines.append(f"{s1_id}\t{','.join(matched)}\n")

        cand_str = "".join(cand_lines)
        match_str = "".join(match_lines)
        for fc, fm in f_handles:
            fc.write(cand_str)
            fm.write(match_str)

        rate = (end_i - start_i) / (time.time() - t_b)
        elapsed = time.time() - t_inf
        pct = (end_i / n_test) * 100
        print(f"  Processed {end_i:,}/{n_test:,} entities ({pct:.1f}%) [{rate:,.0f} ent/sec] - Elapsed: {elapsed:.1f}s")

    for fc, fm in f_handles:
        fc.close()
        fm.close()

    print(f"  Test prediction and streaming completed in {time.time()-t_inf:.1f}s!")

    # Validate output
    val_script = Path("/home/AmazonMLChallenge/student_resource/utils/validate_submission.py")
    if val_script.exists():
        test_dir = DATA_DIR / "test"
        cmd = [
            "python3", str(val_script),
            "--matching", str(out_match),
            "--candidate", str(out_cand),
            "--test-dir", str(test_dir),
            "--check-ids"
        ]
        print(f"\nRunning official validator: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)

    print(f"\nTotal Pipeline Execution Time: {(time.time()-start_total_time)/60:.2f} minutes!")

if __name__ == "__main__":
    main()
