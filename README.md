# ML in Bioinformatics: Complete Pipeline (Days 1-7)

One script, five stages, every figure and metric saved to `results/`.

## Setup
    python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## Run
    python bioml_pipeline.py --quick                    # about 1 minute smoke test
    python bioml_pipeline.py                            # full run, about 5-10 minutes on CPU
    python bioml_pipeline.py --stages clinical expression
    python bioml_pipeline.py --outdir my_run --seed 7

## Stages
| Stage      | Day | What it does                                                        | Outputs |
|------------|-----|---------------------------------------------------------------------|---------|
| clinical   | 1-3 | EDA, PCA, 4 classifiers (LogReg, RF, SVM, XGBoost), ROC             | 01_clinical.png |
| expression | 4   | Synthetic RNA-seq, t-test + BH-FDR DEGs, volcano, heatmap, PCA      | 02_expression.png, deg_table.csv |
| tuning     | 5   | RandomizedSearchCV (RF, XGB) and Optuna (XGB)                       | results_summary.json |
| validation | 6   | Repeated CV, nested CV, permuted-label check, SHAP                  | 04_shap_summary.png, shap_importance.csv |
| sequence   | 7   | CNN, BiLSTM, CNN+LSTM on DNA; in-silico mutagenesis                 | 05_sequence.png |

## Differences from the day-by-day code (deliberate fixes)
1. **No data leakage.** Earlier days scaled the full dataset and picked DEGs before
   splitting. Here scaling and feature selection live inside a scikit-learn `Pipeline`,
   so they are fitted on training folds only. DEG analysis uses the training split only.
2. **BH-FDR instead of Bonferroni** for multiple-testing correction (standard in RNA-seq).
3. **Weaker simulated signal.** The Day 4 data was so separable that every model scored
   AUC 1.0, which made tuning meaningless. The effect sizes are now modest.
4. **Permuted-label sanity check** in the validation stage. AUC should be about 0.5.
   If it is not, something is leaking.
5. Fixed a missing `accuracy_score` import in the Day 7 code.

## Honest caveats
- The expression and sequence data are **synthetic**. Results show the method works, not
  that it will work on real data. For your dissertation, swap in a real count matrix
  (GEO / TCGA) in `make_expression_data()`.
- With only 160 samples the 32-sample test set is noisy; trust the CV numbers more than
  the single test-set AUC.
- Deep learning stage is CPU-friendly but slowest. Use `--stages` to skip it.

## Using real data
Replace `make_expression_data()` so it returns:
`X` (samples x genes, log2-transformed), `y` (0/1 labels), `genes` (list of names).
Everything downstream works unchanged.
