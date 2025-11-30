## VPN Detector

End-to-end pipeline to preprocess MIT LL VNAT traffic captures, build flow-level features, and train/evaluate VPN vs non-VPN models (Random Forest, XGBoost, baseline Logistic Regression). All steps are reproducible via `config.yaml` and the CLI entrypoints.

### Project layout
```
vpn_detector/
  config.yaml
  requirements.txt
  src/
    preprocess.py   # parse PCAP/CSV/HDF5, aggregate to flows
    data_io.py      # load + clean processed parquet
    checks.py       # sanity checks, imbalance, correlations
    features.py     # feature engineering hooks
    splits.py       # group-aware stratified splits
    models.py       # model builders + search CV (RF/XGB/LogReg baseline)
    train.py        # orchestrate training, save models + thresholds
    evaluate.py     # test metrics, threshold tuning, plots
    plots.py        # plotting utilities
    cli.py          # CLI to run pipeline steps
  artifacts/
    datasets/flows.parquet
    models/
    reports/
    figures/
```

### Dataset
Set `data.raw_dir` in `config.yaml` to the VNAT root (default `/home/t/Downloads/Code/MLFINAL`). The preprocessor searches for `.pcap`, `.hdf5`, or `.csv` files recursively. Labels and capture IDs are derived from filenames (case-insensitive match on `vpn`). The processed path is absolute to reuse the cached Parquet.

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
Preprocess to flow-level Parquet:
```bash
python -m vpn_detector.src.cli preprocess --config config.yaml
```

Train models (grouped CV + hyperparameter search):
```bash
python -m vpn_detector.src.cli train --config config.yaml
```

Evaluate on held-out test set (plots + report):
```bash
python -m vpn_detector.src.cli eval --config config.yaml
```

Artifacts:
- `artifacts/datasets/flows.parquet` cached flow dataset
- `artifacts/models/best_rf.joblib`, `best_xgb.joblib`, `best_logreg.joblib`, `preprocess.joblib`, `thresholds.json`
- `artifacts/reports/report.md`, `env.txt`, per-capture CSVs per model
- `artifacts/figures/*` (ROC/PR/calibration/importance/confusion per model, feature importances, score histograms, per-capture recall for VPN captures, per-capture FPR for non-VPN captures)

### Notes
- Group-aware splits prevent capture leakage (`capture_id` stays unique per split).
- If class imbalance exceeds 4:1, training switches to `average_precision` scoring while still reporting F1/Recall.
- Thresholds are optimized on validation predictions for the chosen metric (default F1) and applied to test scores.
- Current snapshot (VNAT, grouped split): XGB/RF near-perfect ROC/PR and high recall; baseline LogReg lags (PR AUC ~0.46, F1 ~0.59), indicating nonlinear signal. For trust, consider per-capture metrics or holding out entire captures as a final test.
