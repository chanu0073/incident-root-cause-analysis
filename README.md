# Incident root-cause analysis for microservices (RCAEval RE2)

When a microservice system fails, find **which service caused it** and **what kind of fault** it is, from metrics, logs and traces.

**Data:** the public [RCAEval](https://github.com/phamquiluan/RCAEval) benchmark (WWW 2025), suite **RE2**: 270 real failure cases from three systems (Online Boutique, Sock Shop, Train Ticket). Six fault types were injected on purpose (CPU, memory, disk, network delay, packet loss, socket), so every case has a **true root-cause label**.

**Status:** pipeline complete and tested; the full run (with trace features and the deep-learning model) is done on a GPU server, see [TRAINING.md](TRAINING.md). Results tables will be added here after that run.

## Approach
| Step | Notebook | What it does |
|---|---|---|
| Data understanding | `01_data_understanding` | case structure, systems, faults, what a failure looks like |
| Feature engineering | `02_feature_engineering` | per service: robust z-scores of every metric type (median/IQR of the case's own normal window), log volume/error/new-template changes, trace latency and error changes, within-case ranks |
| Baselines | `03_baselines` | rules with no training, including a BARO-style max robust z-score |
| ML ranker | `04_ml_ranker` | XGBoost learning-to-rank on the GPU; fault-type classifier; source ablation |
| Deep learning | `05_deep_learning` | 1D CNN on the raw per-service time series with a listwise ranking loss (PyTorch, GPU) |
| Summary | `06_conclusion` | all results in one table |

## Evaluation rules
- **Metrics:** AC@1 / AC@3 / AC@5 (true culprit ranked in the top 1 / 3 / 5), Avg@5, MRR. Ties count against the model.
- **Candidates:** every service with metrics (12, 15 or 68 per system), not only the 5 services the benchmark breaks.
- **No shortcuts:**
  - **Grouped cross-validation:** the 3 repetitions of each (service, fault) pair always stay together.
  - **Leave-one-system-out:** train on two systems, test on the third.
  - **Leave-one-culprit-service-out:** the model must find a culprit it never saw as one.
  - **No service names as features.**
- **Bootstrap confidence intervals** for every claimed gain.

## Run it
- **On a GPU server:** see [TRAINING.md](TRAINING.md). Results come back as an archive (`notebooks/` + `reports/`), not via git.
- **Laptop (no traces):** run `make install data-light features-light notebooks`.

## Layout
```
src/rcalens/   data.py (loading), features.py (per-service features), evaluate.py (metrics, splits),
               models.py (XGBoost ranker, fault classifier), dl.py (1D CNN)
scripts/       download_data.py, build_features.py, make_notebooks.py
notebooks/     01-06
reports/       results/*.json, figures/
```
