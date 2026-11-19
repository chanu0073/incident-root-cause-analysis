# Incident root-cause analysis for microservices (RCAEval RE2)

When a microservice system fails, find **which service caused it** and **what kind of fault** it is, from metrics, logs and traces.

**Data:** the public [RCAEval](https://github.com/phamquiluan/RCAEval) benchmark (WWW 2025), suite **RE2**: 270 real failure cases from three systems (Online Boutique, Sock Shop, Train Ticket). Six fault types were injected on purpose (CPU, memory, disk, network delay, packet loss, socket), so every case has a **true root-cause label**.

**Status:** complete. Full run (metrics + logs + traces, XGBoost and CNN) on an NVIDIA A100, see [TRAINING.md](TRAINING.md).

## Results (270 cases, true culprit among 12 / 15 / 68 candidate services)
| Method | AC@1 | AC@3 | MRR |
|---|---|---|---|
| Random guess (expected) | 0.055 | 0.165 | 0.184 |
| BARO-style max robust z-score (no training) | 0.511 | 0.819 | 0.683 |
| Best rule: average rank over all features (no training) | 0.707 | 0.926 | 0.820 |
| **XGBoost ranker**, grouped 5-fold CV (mean ± sd of 3 repeats) | **0.944 ± 0.010** | **0.990** | **0.968** |
| XGBoost ranker, culprit service never seen as a culprit | 0.944 | 0.993 | 0.970 |
| 1D CNN on raw time series, grouped CV (3 seeds) | 0.930 | 0.981 | 0.959 |
| XGBoost + CNN ensemble | 0.930 | 0.993 | 0.960 |
| 1D CNN, unseen system (train on 2 systems, test on the 3rd) | 0.859 | 0.930 | 0.906 |
| XGBoost (all sources), unseen system | 0.744 | 0.959 | 0.845 |

- **Ranker vs best rule:** +0.233 AC@1, 95% bootstrap CI [+0.178, +0.289].
- **Fault type** (6 classes, using the service the ranker picked): accuracy 0.826, macro-F1 0.825 (0.848 with the true service). Confusions: disk ↔ memory, delay ↔ packet loss.
- **Per fault, AC@1 of the ranker:** cpu / memory / socket 1.00, disk 0.98, delay 0.82, loss 0.84. The rule baseline got only 0.36 on delay.
- **CNN vs XGBoost:** CNN − XGBoost AC@1 is −0.011, 95% CI [−0.037, +0.015], so no significant difference within a system. The CNN transfers better to an unseen system (0.859 vs 0.744).
- **Sources:** in grouped CV, metrics alone (0.951) are as good as metrics + logs + traces (0.944). Latency-change ranks carry most of the ranker's gain.
- **Unseen-system drop:** comes from the trace features. Sock Shop has no traces, so when Online Boutique is held out, trace patterns are learned from Train Ticket alone (Online Boutique AC@1: 0.41). Without traces the ranker scores about 0.86 on unseen systems (0.81 on Online Boutique); notebook 04 measures this per source.

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
