## VPN Detector

End-to-end pipeline to preprocess the MIT LL VNAT traffic captures, build flow-level features, and train Random Forest and XGBoost models to distinguish VPN vs non-VPN traffic. All steps are reproducible via a YAML config and CLI entrypoints.

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
    models.py       # model builders + search CV
    train.py        # orchestrate training, save models
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
Set `data.raw_dir` in `config.yaml` to the VNAT root (default `/home/t/Downloads/Code/MLFINAL`). The preprocessor searches for `.pcap`, `.hdf5`, or `.csv` files recursively. Labels and capture IDs are derived from filenames (case-insensitive match on `vpn`).

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
- `artifacts/models/best_rf.joblib`, `best_xgb.joblib`
- `artifacts/reports/report.md`, `env.txt`
- `artifacts/figures/*` (ROC/PR/calibration/importance/correlation)

### Notes
- Group-aware splits prevent capture leakage (`capture_id` stays unique per split).
- If class imbalance exceeds 4:1, training switches to `average_precision` scoring while still reporting F1/Recall.
- Thresholds are optimized on validation predictions for the chosen metric (default F1) and applied to test scores.
