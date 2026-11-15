#!/usr/bin/env python3
"""
ML in Bioinformatics: complete pipeline (Days 1-7)
===================================================

Stages
------
  clinical    Day 1-3  Breast Cancer Wisconsin: EDA, preprocessing, 4 classifiers
  expression  Day 4    Synthetic TCGA-style RNA-seq: DEG analysis, volcano, heatmap
  tuning      Day 5    RandomizedSearchCV (RF, XGBoost) and Optuna (XGBoost)
  validation  Day 6    Repeated CV, nested CV, permutation sanity check, SHAP
  sequence    Day 7    CNN / BiLSTM / CNN+LSTM promoter classifier, mutagenesis

Usage
-----
  python bioml_pipeline.py                      # run everything
  python bioml_pipeline.py --quick              # fast smoke test
  python bioml_pipeline.py --stages clinical expression
  python bioml_pipeline.py --outdir results --seed 7

Leakage control
---------------
Every data-dependent step (scaling, feature selection, tuning) happens
inside a scikit-learn Pipeline, so it is fitted on training folds only.
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")
ALL_STAGES = ["clinical", "expression", "tuning", "validation", "sequence"]
RESULTS = {}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def banner(text):
    print("\n" + "=" * 64 + f"\n{text}\n" + "=" * 64)


def savefig(fig, outdir, name):
    path = Path(outdir) / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {path}")


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def make_expression_data(seed, n_tumor=80, n_normal=80, n_genes=1000):
    """Synthetic RNA-seq-like counts. 150 true DEGs: 75 up, 75 down in tumor."""
    rng = np.random.default_rng(seed)
    gene_base = rng.normal(2.0, 0.5, n_genes)           # per-gene baseline
    def sample(n):
        return rng.lognormal(mean=gene_base, sigma=1.2, size=(n, n_genes))
    tumor, normal = sample(n_tumor), sample(n_normal)
    tumor[:, :75] *= rng.uniform(1.3, 2.0, 75)           # oncogenes (modest effect)
    tumor[:, 75:150] *= rng.uniform(0.5, 0.8, 75)         # tumour suppressors
    X = np.log2(np.vstack([tumor, normal]) + 1)          # fixed transform, no leakage
    y = np.array([1] * n_tumor + [0] * n_normal)
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    return X, y, genes


def make_pipe(clf, k=50):
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("clf", clf)])


def xgb(**kw):
    return XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **kw)


# ----------------------------------------------------------------------
# Stage 1: clinical features (Days 1-3)
# ----------------------------------------------------------------------
def stage_clinical(args):
    banner("STAGE: clinical (Days 1-3), Breast Cancer Wisconsin")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)                           # 0 malignant, 1 benign
    print(f"Shape: {X.shape} | malignant={int((y==0).sum())} benign={int((y==1).sum())} "
          f"| missing={int(X.isna().sum().sum())}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    models = {
        "LogReg": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=args.seed),
        "SVM": SVC(probability=True, random_state=args.seed),
        "XGBoost": xgb(n_estimators=200),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # EDA panels
    ax[0].bar(["Malignant", "Benign"], y.value_counts().sort_index().values,
              color=["#e74c3c", "#2ecc71"])
    ax[0].set_title("Class distribution")
    pcs = PCA(2).fit_transform(StandardScaler().fit_transform(X))
    ax[1].scatter(pcs[:, 0], pcs[:, 1], c=np.where(y == 0, "#e74c3c", "#2ecc71"),
                  s=18, alpha=0.7)
    ax[1].set_title("PCA (exploratory)")

    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        auc_te = roc_auc_score(y_te, prob)
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax[2].plot(fpr, tpr, lw=2, label=f"{name} ({auc_te:.3f})")
        out[name] = {"cv_auc": float(cv_auc.mean()), "cv_std": float(cv_auc.std()),
                     "test_auc": float(auc_te),
                     "test_acc": float(accuracy_score(y_te, pipe.predict(X_te)))}
        print(f"  {name:13s} CV AUC {cv_auc.mean():.4f} +/- {cv_auc.std():.4f} | "
              f"test AUC {auc_te:.4f}")
    ax[2].plot([0, 1], [0, 1], "k--")
    ax[2].set_title("ROC (held-out test set)")
    ax[2].legend()
    savefig(fig, args.outdir, "01_clinical.png")
    RESULTS["clinical"] = out


# ----------------------------------------------------------------------
# Stage 2: expression analysis (Day 4)
# ----------------------------------------------------------------------
def stage_expression(args):
    banner("STAGE: expression (Day 4), DEG analysis")
    X, y, genes = make_expression_data(args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    # DEG analysis on the TRAINING set only (no peeking at test data)
    t, p = stats.ttest_ind(X_tr[y_tr == 1], X_tr[y_tr == 0], axis=0)
    fdr = bh_fdr(p)
    lfc = X_tr[y_tr == 1].mean(0) - X_tr[y_tr == 0].mean(0)   # data already log2
    deg = pd.DataFrame({"gene": genes, "log2FC": lfc, "p": p, "FDR": fdr})
    sig = (deg.FDR < 0.05) & (deg.log2FC.abs() > 0.5)
    deg["significant"] = sig
    true_deg = set(genes[:150])
    found = set(deg.loc[sig, "gene"])
    precision = len(found & true_deg) / max(len(found), 1)
    recall = len(found & true_deg) / 150
    print(f"  Significant DEGs: {sig.sum()} (up {(sig & (lfc>0)).sum()}, "
          f"down {(sig & (lfc<0)).sum()})")
    print(f"  Against simulated ground truth: precision {precision:.2f}, recall {recall:.2f}")
    deg.sort_values("FDR").to_csv(Path(args.outdir) / "deg_table.csv", index=False)

    # Volcano
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    nl = -np.log10(deg.p + 1e-300)
    ax[0].scatter(lfc[~sig], nl[~sig], s=8, c="grey", alpha=0.4)
    ax[0].scatter(lfc[sig & (lfc > 0)], nl[sig & (lfc > 0)], s=14, c="#e74c3c", label="Up")
    ax[0].scatter(lfc[sig & (lfc < 0)], nl[sig & (lfc < 0)], s=14, c="#3498db", label="Down")
    ax[0].set_xlabel("log2 fold change"); ax[0].set_ylabel("-log10 p")
    ax[0].set_title("Volcano plot (training set)"); ax[0].legend()

    # Heatmap of top 25 DEGs
    top = deg.sort_values("FDR").head(25).index.to_numpy()
    order = np.argsort(y_tr)
    Z = StandardScaler().fit_transform(X_tr)[:, top][order].T
    sns.heatmap(Z, cmap="RdBu_r", center=0, xticklabels=False,
                yticklabels=[genes[i] for i in top], ax=ax[1], cbar_kws={"shrink": 0.7})
    ax[1].set_title("Top 25 DEGs (left normal, right tumour)")

    # PCA on top DEGs
    pcs = PCA(2).fit_transform(Z.T)
    ax[2].scatter(pcs[:, 0], pcs[:, 1], c=np.where(y_tr[order] == 1, "#e74c3c", "#2ecc71"),
                  s=30, edgecolors="k", linewidths=0.3)
    ax[2].set_title("PCA of top DEGs")
    savefig(fig, args.outdir, "02_expression.png")

    # Leakage-free classification
    out = {}
    for name, clf in {"RandomForest": RandomForestClassifier(300, random_state=args.seed),
                      "SVM": SVC(probability=True, random_state=args.seed),
                      "XGBoost": xgb(n_estimators=200)}.items():
        pipe = make_pipe(clf).fit(X_tr, y_tr)
        out[name] = float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
        print(f"  {name:13s} test AUC {out[name]:.4f}")
    RESULTS["expression"] = {"n_sig_degs": int(sig.sum()), "precision": precision,
                             "recall": recall, "test_auc": out}


# ----------------------------------------------------------------------
# Stage 3: hyperparameter tuning (Day 5)
# ----------------------------------------------------------------------
def stage_tuning(args):
    banner("STAGE: tuning (Day 5)")
    X, y, _ = make_expression_data(args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    n_iter = 8 if args.quick else 40
    out = {}

    searches = {
        "RandomForest": (make_pipe(RandomForestClassifier(random_state=args.seed)), {
            "clf__n_estimators": randint(100, 400), "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_split": randint(2, 11), "clf__max_features": ["sqrt", "log2"],
            "select__k": [20, 50, 100]}),
        "XGBoost": (make_pipe(xgb()), {
            "clf__n_estimators": randint(50, 400), "clf__max_depth": randint(3, 10),
            "clf__learning_rate": uniform(0.01, 0.29), "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5), "select__k": [20, 50, 100]}),
    }
    for name, (pipe, dist) in searches.items():
        base = pipe.fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_te, base.predict_proba(X_te)[:, 1])
        t0 = time.time()
        rs = RandomizedSearchCV(pipe, dist, n_iter=n_iter, cv=cv, scoring="roc_auc",
                                n_jobs=-1, random_state=args.seed).fit(X_tr, y_tr)
        auc_te = roc_auc_score(y_te, rs.predict_proba(X_te)[:, 1])
        out[name] = {"baseline_test_auc": float(base_auc), "tuned_cv_auc": float(rs.best_score_),
                     "tuned_test_auc": float(auc_te), "best_params": {k: (v.item() if hasattr(v, "item") else v)
                                                                      for k, v in rs.best_params_.items()}}
        print(f"  {name:13s} baseline {base_auc:.4f} -> tuned {auc_te:.4f} "
              f"(CV {rs.best_score_:.4f}, {time.time()-t0:.0f}s)")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
            k = trial.suggest_categorical("k", [20, 50, 100])
            return cross_val_score(make_pipe(xgb(**params), k=k), X_tr, y_tr,
                                   cv=cv, scoring="roc_auc").mean()

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=10 if args.quick else 40)
        bp = dict(study.best_params); k = bp.pop("k")
        final = make_pipe(xgb(**bp), k=k).fit(X_tr, y_tr)
        auc_te = roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])
        out["XGBoost_Optuna"] = {"tuned_cv_auc": float(study.best_value),
                                 "tuned_test_auc": float(auc_te), "best_params": study.best_params}
        print(f"  XGB (Optuna)  CV {study.best_value:.4f} | test {auc_te:.4f}")
    except ImportError:
        print("  optuna not installed, skipping Bayesian search")
    RESULTS["tuning"] = out


# ----------------------------------------------------------------------
# Stage 4: validation + SHAP (Day 6)
# ----------------------------------------------------------------------
def stage_validation(args):
    banner("STAGE: validation (Day 6), CV, nested CV, permutation test, SHAP")
    X, y, genes = make_expression_data(args.seed)
    out = {}

    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2 if args.quick else 3,
                                  random_state=args.seed)
    s = cross_val_score(make_pipe(xgb(n_estimators=200, max_depth=4, learning_rate=0.05)),
                        X, y, cv=rcv, scoring="roc_auc", n_jobs=-1)
    out["repeated_cv_auc"] = [float(s.mean()), float(s.std())]
    print(f"  Repeated stratified CV AUC: {s.mean():.4f} +/- {s.std():.4f}")

    grid = {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1],
            "clf__n_estimators": [100, 200]}
    inner = StratifiedKFold(3, shuffle=True, random_state=args.seed)
    outer = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    gs = GridSearchCV(make_pipe(xgb()), grid, cv=inner, scoring="roc_auc", n_jobs=-1)
    ns = cross_val_score(gs, X, y, cv=outer, scoring="roc_auc")
    out["nested_cv_auc"] = [float(ns.mean()), float(ns.std())]
    print(f"  Nested CV AUC:              {ns.mean():.4f} +/- {ns.std():.4f}")

    rng = np.random.default_rng(args.seed)
    perm = cross_val_score(make_pipe(xgb(n_estimators=100)), X, rng.permutation(y),
                           cv=outer, scoring="roc_auc", n_jobs=-1)
    out["permuted_label_auc"] = float(perm.mean())
    print(f"  Permuted-label AUC:         {perm.mean():.4f}  (should be ~0.5; "
          f"if not, you have leakage)")

    # SHAP on a held-out split
    try:
        import shap
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=args.seed)
        pipe = make_pipe(xgb(n_estimators=200, max_depth=4, learning_rate=0.05)).fit(X_tr, y_tr)
        mask = pipe.named_steps["select"].get_support()
        names = np.array(genes)[mask]
        Xt = pipe[:-1].transform(X_te)
        sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
        imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
        imp.to_csv(Path(args.outdir) / "shap_importance.csv", header=["mean_abs_shap"])
        recovered = sum(g in set(genes[:150]) for g in imp.head(20).index)
        print(f"  Top-20 SHAP genes that are true simulated DEGs: {recovered}/20")
        out["shap_top20_true_degs"] = int(recovered)

        plt.figure()
        shap.summary_plot(sv, Xt, feature_names=list(names), max_display=15, show=False)
        plt.title("SHAP summary (held-out samples)")
        savefig(plt.gcf(), args.outdir, "04_shap_summary.png")
    except ImportError:
        print("  shap not installed, skipping interpretability")
    RESULTS["validation"] = out


# ----------------------------------------------------------------------
# Stage 5: deep learning on sequences (Day 7)
# ----------------------------------------------------------------------
def make_sequences(n_each, length, rng):
    nuc = np.array(list("ATGC"))
    seqs, labels = [], []
    for _ in range(n_each):                              # promoters
        s = rng.choice(nuc, length, p=[0.22, 0.22, 0.28, 0.28])
        for start, motif, keep in [(65, "TATAAA", 0.85), (20, "CCAAT", 0.8)]:
            for i, c in enumerate(motif):
                if rng.random() < keep:
                    s[start + i] = c
        seqs.append("".join(s)); labels.append(1)
    for _ in range(n_each):                              # background
        seqs.append("".join(rng.choice(nuc, length))); labels.append(0)
    return seqs, np.array(labels)


def one_hot(seqs):
    idx = {"A": 0, "T": 1, "G": 2, "C": 3}
    arr = np.zeros((len(seqs), len(seqs[0]), 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            arr[i, j, idx[c]] = 1
    return arr


def stage_sequence(args):
    banner("STAGE: sequence (Day 7), deep learning on DNA")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("  TensorFlow not installed, skipping. Run: pip install tensorflow")
        return
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    L = 100
    seqs, y = make_sequences(200 if args.quick else 500, L, rng)
    X = one_hot(seqs)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, stratify=y,
                                                random_state=args.seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.18,
                                              stratify=y_tmp, random_state=args.seed)
    print(f"  train {len(X_tr)} | val {len(X_va)} | test {len(X_te)} | shape {X.shape[1:]}")

    def cnn():
        i = keras.Input((L, 4))
        x = layers.Conv1D(64, 6, padding="same", activation="relu")(i)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(128, 12, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
        return keras.Model(i, layers.Dense(1, activation="sigmoid")(x), name="CNN")

    def bilstm():
        i = keras.Input((L, 4))
        x = layers.Bidirectional(layers.LSTM(48, return_sequences=True))(i)
        x = layers.Bidirectional(layers.LSTM(24))(x)
        x = layers.Dropout(0.3)(x); x = layers.Dense(32, activation="relu")(x)
        return keras.Model(i, layers.Dense(1, activation="sigmoid")(x), name="BiLSTM")

    def hybrid():
        i = keras.Input((L, 4))
        x = layers.Conv1D(96, 8, padding="same", activation="relu")(i)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
        x = layers.Bidirectional(layers.LSTM(48))(x)
        x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
        return keras.Model(i, layers.Dense(1, activation="sigmoid")(x), name="CNN_LSTM")

    epochs = 6 if args.quick else 40
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    out, best, best_auc = {}, None, -1
    for build in (cnn, bilstm, hybrid):
        m = build()
        m.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=["accuracy"])
        h = m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=32,
                  verbose=0, callbacks=[keras.callbacks.EarlyStopping(
                      patience=8, restore_best_weights=True, monitor="val_loss")])
        prob = m.predict(X_te, verbose=0).ravel()
        auc_te = roc_auc_score(y_te, prob)
        out[m.name] = {"test_auc": float(auc_te),
                       "test_acc": float(accuracy_score(y_te, prob > 0.5)),
                       "epochs_run": len(h.history["loss"]), "params": int(m.count_params())}
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax[0].plot(fpr, tpr, lw=2, label=f"{m.name} ({auc_te:.3f})")
        print(f"  {m.name:9s} test AUC {auc_te:.4f} | acc {out[m.name]['test_acc']:.3f} "
              f"| params {m.count_params():,}")
        if auc_te > best_auc:
            best, best_auc = m, auc_te
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].legend(); ax[0].set_title("ROC (test)")

    # In-silico mutagenesis: mask each position, measure drop in P(promoter)
    pos_sample = X_te[y_te == 1][:40]
    base = best.predict(pos_sample, verbose=0).ravel()
    imp = np.zeros(L)
    for p in range(L):
        mut = pos_sample.copy(); mut[:, p, :] = 0.25
        imp[p] = np.mean(base - best.predict(mut, verbose=0).ravel())
    ax[1].plot(imp, color="#6c3483")
    ax[1].axvspan(20, 25, alpha=0.2, color="orange", label="CAAT (planted)")
    ax[1].axvspan(65, 71, alpha=0.2, color="red", label="TATA (planted)")
    ax[1].set_title(f"In-silico mutagenesis ({best.name})")
    ax[1].set_xlabel("Position (bp)"); ax[1].set_ylabel("Mean drop in P(promoter)")
    ax[1].legend()
    savefig(fig, args.outdir, "05_sequence.png")
    top = np.argsort(imp)[::-1][:8]
    print(f"  Top mutagenesis positions: {sorted(int(t) for t in top)} "
          f"(planted motifs: 20-24 and 65-70)")
    RESULTS["sequence"] = out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small budgets for a fast test")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    funcs = {"clinical": stage_clinical, "expression": stage_expression,
             "tuning": stage_tuning, "validation": stage_validation,
             "sequence": stage_sequence}
    t0 = time.time()
    for s in ALL_STAGES:
        if s in args.stages:
            funcs[s](args)
    with open(Path(args.outdir) / "results_summary.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    banner(f"DONE in {time.time()-t0:.0f}s. Outputs in ./{args.outdir}/")


if __name__ == "__main__":
    main()
