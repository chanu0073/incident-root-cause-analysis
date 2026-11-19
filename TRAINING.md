# Running the full pipeline on the GPU server

Everything below runs from the repository root. Total time is roughly 30–60 minutes on an A100. The download is about 3.5 GB; plan for ~6 GB of free disk.

## 1. Get the code and set up Python (Python 3.10+, tested with 3.12)
```bash
git clone https://github.com/chanu0073/incident-root-cause-analysis.git
cd incident-root-cause-analysis
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name rcaeval --display-name "RCAEval (.venv)"
```

Check that the GPU is visible (both lines must say the GPU is available):
```bash
nvidia-smi
.venv/bin/python -c "import torch, xgboost; print('torch cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0)); xgboost.XGBClassifier(device='cuda', n_estimators=1).fit([[0],[1]],[0,1]); print('xgboost cuda: ok')"
```
If `torch.cuda.is_available()` is `False`, install the torch build matching the server's CUDA from https://pytorch.org (e.g. `.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124`).

## 2. Download the data (RE2 suite: metrics, logs, traces)
```bash
.venv/bin/python scripts/download_data.py
```
Ends with `done: 270 RE2 cases in .../data/raw`. If Hugging Face rate-limits you, run `export HF_TOKEN=<your token>` (free account) and re-run; finished files are skipped.

## 3. Build the features (CPU, parallel)
```bash
.venv/bin/python scripts/build_features.py --jobs 16
```
Use about half of the server's CPU cores for `--jobs` (each Train Ticket case needs ~1 GB RAM). Ends with
`270 cases, 8548 service rows, ... -> data/processed/service_features.parquet`.

## 4. Run the notebooks (training on the GPU)
```bash
make notebooks
```
Runs `notebooks/01 … 06` in order and saves the outputs inside the notebooks. Notebook 05 (the CNN) is the longest part. To run one notebook only:
```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 04_ml_ranker.ipynb
```

## 5. Send the results back
```bash
git add notebooks reports
git commit -m "Add full training results"
git push
```
`data/` and `models/` are git-ignored (large files); the numbers are saved in `reports/results/*.json` and inside the executed notebooks.

## If something fails
Send me the last ~30 lines of the error output. A failed notebook does not affect the earlier ones.
