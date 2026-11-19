"""Regenerate the notebooks from this file (overwrites them, unexecuted). Usage: python scripts/make_notebooks.py [names...]"""
import sys

import nbformat

from pathlib import Path
PROJ = str(Path(__file__).resolve().parents[1])
SETUP = """
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message=".*(mismatched devices|pandas version 4).*")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid"); pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
RESULTS = ROOT / "reports" / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
"""


def write(name, cells):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "rcaeval", "display_name": "RCAEval (.venv)", "language": "python"}
    nb.cells = [(nbformat.v4.new_markdown_cell if k == "md" else nbformat.v4.new_code_cell)(t.strip("\n")) for k, t in cells]
    nbformat.write(nb, f"{PROJ}/notebooks/{name}.ipynb")
    print("wrote", name)


NB = {}

# ======================================================================= 01
NB["01_data_understanding"] = [
("md", """
# 01 · Understanding the data: what does a microservice failure look like?

**Goal of this project:** when something breaks in a system made of many small services, find **which service caused it** (root-cause *localisation*) and **what kind of fault** it is. This is the "root-cause analysis" step of incident management, done on real telemetry.

### The dataset: RCAEval, suite RE2
RCAEval (WWW 2025 benchmark) ran three open-source demo shops/apps, deliberately broke one service at a time and recorded everything:

| System | Short name | Services with metrics | What it is |
|---|---|---|---|
| Online Boutique | `ob` | 12 | Google's e-commerce demo (Go, Python, Java, …) |
| Sock Shop | `ss` | 15 | Weaveworks' sock shop |
| Train Ticket | `tt` | 68 | a large ticket-booking system (Java) |

**Fault types (6):** CPU stress, memory stress, disk stress, network delay, packet loss, socket stress.

**One case** = 12 minutes of normal operation → fault injected at `inject_time` → 12 more minutes. We get:
- **metrics** (1 row per second; CPU, memory, disk I/O, sockets, workload, errors, latency per service),
- **logs** (every log line with its container),
- **traces** (every request "span": which service, which operation, how long it took). Not recorded for Sock Shop.

The **label** (`root_cause_service`) is known because the fault was injected on purpose, a real ground truth, unlike most incident datasets.

> **Why this matters:** in a bank's operations centre an alert says "checkout is slow", but the cause is often a *different* service. Automatically ranking the likely culprits saves the on-call engineer time.
"""),
("code", SETUP + """
from rcalens.data import case_index, load_case, split_metric
idx = case_index()
print(len(idx), "cases")
idx.groupby("system").agg(cases=("case", "size"), root_cause_services=("root_cause_service", "nunique"),
                          fault_types=("fault", "nunique"), metrics=("n_metrics", "median"),
                          log_lines=("n_logs", "median"), spans=("n_traces", "median"))
"""),
("md", """
**Reading the table:** 90 cases per system = 5 root-cause services × 6 fault types × 3 repetitions. Train Ticket is much bigger (360+ metric columns, ~1M spans per case). Sock Shop has no traces (0 spans).
"""),
("code", """
pd.crosstab([idx["system"], idx["root_cause_service"]], idx["fault"])
"""),
("md", """
Every (service, fault) pair appears exactly 3 times. **Important consequence for evaluation:** the 3 repetitions are near-copies of each other. If one repetition is in training and another in test, the model can "recognise" the case instead of reasoning about it. So our splits always keep all 3 repetitions together (notebook 04).

## What the model must choose from
For each case the candidate list is **every service that has metrics**, not just the 5 that the benchmark happens to break (using that list would be cheating: in real life you don't know which 5 services will fail).
"""),
("code", """
cand = {s: len(load_case(idx.loc[idx.system == s, "case"].iloc[0], traces=False).services) for s in ["ob", "ss", "tt"]}
rand = {s: {"candidates": n, "random top-1": 1 / n, "random top-3": min(3, n) / n} for s, n in cand.items()}
pd.DataFrame(rand).T.round(3)
"""),
("md", """
A random guess finds the culprit first time 8% (Online Boutique), 7% (Sock Shop) or **1.5%** (Train Ticket) of the time. That's the floor.

## Look at one failure: a CPU fault on `checkoutservice`
"""),
("code", """
c = load_case("re2ob_checkoutservice_cpu_1")
m = c.metrics.copy(); m["minute"] = (m["time"] - c.inject_time) / 60
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharex=True)
for ax, typ in zip(axes, ["cpu", "mem", "latency-90"]):
    for col in [k for k in m.columns if k.endswith("_" + typ)]:
        svc = split_metric(col)[0]
        ax.plot(m["minute"], m[col], lw=2.2 if svc == "checkoutservice" else 0.7,
                color="#c0392b" if svc == "checkoutservice" else "#9aa7b4", label="checkoutservice" if svc == "checkoutservice" else None)
    ax.axvline(0, color="k", ls="--", lw=1); ax.set_title(typ); ax.set_xlabel("minutes from fault injection")
axes[0].legend(); plt.tight_layout(); plt.savefig(FIG / "example_cpu_fault.png", dpi=120); plt.show()
"""),
("md", """
The red line (the root cause) jumps in CPU right at the injection (dashed line). Other services barely move. **But it is not always this clean:**
- a *delay* or *loss* fault on one service shows up as **higher latency in its callers** too (the slowness travels upstream),
- some metrics are noisy or flat (a service that normally has 0 errors makes any error look "infinitely" anomalous),
- Train Ticket has 68 services, so there are many chances for an innocent service to look odd.

That is why we compute several kinds of evidence per service and let a model learn how to weigh them.

## Logs and traces around the injection
"""),
("code", """
lg = c.logs.copy(); lg["minute"] = (lg["timestamp"].astype("int64") - c.inject_time) // 60
per_min = lg.groupby(["minute", "container_name"]).size().unstack(fill_value=0)
ax = per_min[per_min.sum().sort_values(ascending=False).index[:6]].plot(figsize=(10, 3.4), lw=1.2)
ax.axvline(0, color="k", ls="--"); ax.set_ylabel("log lines / minute"); ax.set_title("log volume per container"); plt.tight_layout(); plt.show()
if c.traces is not None:
    t = c.traces.copy(); t["minute"] = (t["startTimeMillis"].astype("int64") // 1000 - c.inject_time) // 60
    p95 = t.groupby(["minute", "serviceName"])["duration"].quantile(0.95).unstack()
    ax = (p95 / 1000).plot(figsize=(10, 3.4), lw=1.2, logy=True)
    ax.axvline(0, color="k", ls="--"); ax.set_ylabel("p95 span duration (ms)"); ax.set_title("trace latency per service"); plt.tight_layout(); plt.show()
"""),
("md", """
### Takeaways
1. Each case is a **ranking problem**: order 12–68 candidate services so the true culprit comes first.
2. Evidence comes from three sources (metrics, logs, traces). Traces are missing for Sock Shop, so models must cope with missing data.
3. The 3 repetitions per (service, fault) must stay together in any train/test split.

➡️ Next: **02_feature_engineering**: turn each service's behaviour before/after the fault into numbers.
"""),
]

# ======================================================================= 02
NB["02_feature_engineering"] = [
("md", """
# 02 · Feature engineering: "how differently did this service behave after the fault?"

Every row of our table = **one candidate service in one case**. Its features compare the 12 minutes *after* the injection with the 12 minutes *before*, using only that case's own data:

| Source | Features (per service) | Idea |
|---|---|---|
| Metrics | for each metric type (cpu, mem, diskio, socket, workload, error, latency-50, latency-90): **max robust z-score** after the fault, mean z-score, relative change of the mean | "how far outside its normal range did it go?" |
| Logs | change in log volume, change in error-like lines, share of log *templates* never seen before the fault | "did it start complaining, or go quiet?" |
| Traces | change in span count, p50/p95 latency change, worst single operation's p95 change, error-rate change | "did its requests get slower or fail?" |
| Ranks | every feature's **rank among the services of the same case** | makes systems with 12 and 68 services comparable |

### Robust z-score (the key statistic)
For a metric, take the normal window, compute its **median** and **IQR** (spread of the middle 50%). After the fault, `z = (value − median) / IQR`. Median/IQR instead of mean/std because one spike in the normal window would otherwise inflate the "normal" spread. This is the same idea as **BARO** (FSE 2024), a strong published method on this benchmark.

### Leakage rules
- No feature uses the label, other cases, or anything after the case's own time window.
- **Service names are not features.** Otherwise the model would just learn "the benchmark only breaks these 5 services", a shortcut that is useless in real life.
- Features are computed per case with fixed rules (nothing is *fitted*), so they cannot leak test information into training.

The features are built by `scripts/build_features.py` (parallel, ~1–3 s per case; Train Ticket is the slowest).
"""),
("code", SETUP + """
from rcalens.data import load_features
df, has_traces = load_features()
print("feature table:", df.shape, "| trace features included:", has_traces)
print("cases:", df["case"].nunique(), "| rows per case (candidate services):", df.groupby("system")["service"].nunique().to_dict())
df.head(3)
"""),
("code", """
src = {"metrics": [c for c in df.columns if c.startswith("m_") and not c.endswith("_rank")],
       "logs": [c for c in df.columns if c.startswith("l_") and not c.endswith("_rank")],
       "traces": [c for c in df.columns if c.startswith("t_") and not c.endswith("_rank")]}
cover = {s: {k: float(df.loc[df.system == s, v].notna().any(axis=1).mean()) if v else 0.0 for k, v in src.items()} for s in ["ob", "ss", "tt"]}
print({k: len(v) for k, v in src.items()}, "raw features (+ the same number of rank features)")
pd.DataFrame(cover).T.round(2).rename_axis("share of services with this source")
"""),
("md", """
Not every service has every source: some services write no logs, Sock Shop has no traces, databases have no latency metrics. **XGBoost handles missing values natively**, so we keep them as missing instead of inventing values.

## Do the features separate root causes from innocent services?
"""),
("code", """
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, col in zip(axes, ["m_all_maxz", "m_all_maxz_rank", "l_err_ratio"]):
    if col in df:
        sns.boxplot(data=df, x="system", y=col, hue=df["is_rc"].map({0: "other service", 1: "root cause"}), ax=ax, fliersize=1)
        ax.set_title(col); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
("md", """
The root cause usually has the most extreme metric change in its case (rank close to 1.0), but the boxes overlap: innocent services can look just as extreme (e.g. a caller suffering from the root cause's delay).

## Which metric reacts to which fault? (root-cause services only)
"""),
("code", """
rc = df[df["is_rc"] == 1]
types = [c for c in df.columns if c.startswith("m_") and c.endswith("_maxz") and c != "m_all_maxz"]
heat = rc.groupby("fault")[types].median().rename(columns=lambda c: c[2:-5])
plt.figure(figsize=(9, 3.8)); sns.heatmap(heat, annot=True, fmt=".1f", cmap="Reds")
plt.title("median log(1 + max |z|) of the root-cause service, by fault type"); plt.tight_layout()
plt.savefig(FIG / "fault_signatures.png", dpi=120); plt.show()
"""),
("md", """
**This is the "signature" of each fault:** CPU faults show up in `cpu`, memory faults in `mem`, disk faults in `diskio`, while **delay / loss** faults mostly show in latency and are the hard ones (the latency also rises in the services that call the culprit). These signatures are also what lets us classify the **fault type** (notebook 04).

### Recap
- One row per (case, candidate service), ~95 features from three sources, plus within-case ranks.
- Robust (median/IQR) statistics of each case's own normal window; no labels, no service names.

➡️ Next: **03_baselines**: how far do simple rules get?
"""),
]

# ======================================================================= 03
NB["03_baselines"] = [
("md", """
# 03 · Baselines: how far do simple rules get?

Before any machine learning, we measure **rules that need no training**. If a model can't beat them, it isn't worth deploying.

### Metrics (how the benchmark scores RCA)
For each case, sort services by score. Find the position (rank) of the true root cause.
- **AC@1** = share of cases where the true cause is ranked **first** (top-1 accuracy)
- **AC@3 / AC@5** = share where it is in the top 3 / top 5
- **Avg@5** = average of AC@1…AC@5 (the RCAEval summary number)
- **MRR** = average of 1 / rank

Ties are counted *pessimistically* (if the culprit shares the top score with 4 others, it counts as rank 5), so a rule cannot look good by luck.

### The rules
| Rule | Score of a service |
|---|---|
| random | random number (expected value shown) |
| **max metric z (BARO-style)** | its largest robust z-score over all its metrics |
| latency only | largest z of latency-50/90 |
| resource only | largest z of cpu / mem / diskio / socket |
| log errors | increase in error-like log lines |
| trace p95 | increase in p95 span latency (OB, TT only) |
| rank average | average of its within-case ranks over all features |

These rules use **all 270 cases** (no training, so no split needed).
"""),
("code", SETUP + """
from rcalens.data import load_features
from rcalens.evaluate import evaluate, true_ranks, rank_metrics
df, has_traces = load_features()
rng = np.random.default_rng(0)

def with_score(values):
    d = df[["case", "system", "service", "is_rc"]].copy(); d["score"] = np.nan_to_num(np.asarray(values, float), nan=-1e9); return d

z = lambda cols: df[[c for c in cols if c in df]].max(axis=1)
rules = {
    "max metric z (BARO-style)": df["m_all_maxz"],
    "latency only": z(["m_latency50_maxz", "m_latency90_maxz"]),
    "resource only (cpu/mem/disk/socket)": z(["m_cpu_maxz", "m_mem_maxz", "m_diskio_maxz", "m_socket_maxz"]),
    "log errors": df.get("l_err_ratio", pd.Series(np.nan, index=df.index)),
    "rank average (all features)": df[[c for c in df.columns if c.endswith("_rank")]].mean(axis=1),
}
if has_traces:
    rules["trace p95"] = df["t_p95_ratio"]

# random: expected metrics computed exactly from the number of candidates
n_cand = df.groupby("case").size()
exp_rand = {f"AC@{k}": float(np.mean(np.minimum(k, n_cand) / n_cand)) for k in (1, 3, 5)}
exp_rand["Avg@5"] = float(np.mean([np.mean(np.minimum(k, n_cand) / n_cand) for k in range(1, 6)]))
exp_rand["MRR"] = float(np.mean([(1 / np.arange(1, n + 1)).mean() for n in n_cand]))

tables = {"random (expected)": pd.Series(exp_rand)}
per_system = {}
for name, s in rules.items():
    ev = evaluate(with_score(s)); tables[name] = ev.loc["all"]; per_system[name] = ev
base = pd.DataFrame(tables).T
base.round(3)
"""),
("md", """
**Reading it:** compare each rule with *random*. The strongest single rule is the one to beat in notebook 04.

### Per system (AC@1, top-1 accuracy)
"""),
("code", """
pd.DataFrame({k: v["AC@1"] for k, v in per_system.items()}).T.round(3)
"""),
("md", """
### Per fault type (AC@1 of the best rule)
Which faults are easy and which are hard for the rule?
"""),
("code", """
best_rule = base.drop(index="random (expected)")["AC@1"].idxmax()
r = true_ranks(with_score(rules[best_rule]))
fault = df.drop_duplicates("case").set_index("case")["fault"]
print("best rule:", best_rule)
(r <= 1).groupby(fault.reindex(r.index).to_numpy()).mean().round(3).rename("AC@1").to_frame().T
"""),
("code", """
json.dump({"has_traces": has_traces, "best_rule": best_rule, "overall": base.round(4).to_dict(orient="index"),
           "ac1_by_system": pd.DataFrame({k: v["AC@1"] for k, v in per_system.items()}).round(4).to_dict()},
          open(RESULTS / "baselines.json", "w"), indent=1)
with_score(rules[best_rule]).to_parquet(ROOT / "data" / "processed" / "baseline_scores.parquet", index=False)
print("saved reports/results/baselines.json")
"""),
("md", """
### Takeaways
- A single robust statistic (the BARO idea) is already far above random: injected faults usually leave a clear mark on the culprit's own metrics.
- It struggles where the symptom *travels*: delay and loss faults make callers slow too, so the slowest service is often not the cause.
- That's the opening for a learned model that combines many weak signals (resource vs latency, logs, traces, ranks).

➡️ Next: **04_ml_ranker**: learning to rank with XGBoost.
"""),
]

# ======================================================================= 04
NB["04_ml_ranker"] = [
("md", """
# 04 · Learning to rank root causes with XGBoost (GPU)

**Learning to rank:** instead of predicting a class for each row independently, the model learns to *order* the services of a case so the culprit comes first. XGBoost's `rank:pairwise` objective looks at pairs (culprit, innocent service) inside the same case and learns to score the culprit higher.

### Honest evaluation (the most important part of this notebook)
We only have 270 cases, so we use **cross-validation** (train on part, test on the rest, rotate):
1. **Grouped 5-fold CV:** the 3 repetitions of the same (system, service, fault) always land in the same fold, so the model is tested on failures it has never seen a copy of. Repeated with 3 different shuffles (seeds) to average out luck.
2. **Leave-one-system-out:** train on two systems, test on the third. This is the toughest test: does the model generalise to a system it has *never seen*? (Closest to deploying it on a new bank application.)

Every case gets exactly one *out-of-fold* prediction per repetition of the CV, so all 270 cases are scored by a model that did not see them.

We compare with the best rule from notebook 03 and report **bootstrap confidence intervals** for the difference.
"""),
("code", SETUP + """
from rcalens.data import load_features, case_index
from rcalens.evaluate import evaluate, true_ranks, rank_metrics, bootstrap_diff, grouped_folds, leave_one_system_out
from rcalens.models import gpu_device, feature_columns, fit_ranker, fault_frame, fit_fault_classifier, FAULTS
df, has_traces = load_features()
cases = df.drop_duplicates("case")[["case", "system", "pair", "fault"]].reset_index(drop=True)
DEVICE = gpu_device()
print("training device:", DEVICE, "| trace features:", has_traces, "| cases:", len(cases))
base = json.load(open(RESULTS / "baselines.json"))
base_scores = pd.read_parquet(ROOT / "data" / "processed" / "baseline_scores.parquet")
print("best rule from notebook 03:", base["best_rule"], base["overall"][base["best_rule"]])
"""),
("code", """
def cv_scores(cols, seeds=(0, 1, 2), **params):
    \"\"\"Grouped 5-fold CV, repeated over seeds. Returns one out-of-fold score per (case, service) per seed.\"\"\"
    outs = []
    for seed in seeds:
        for tr_c, te_c in grouped_folds(cases, n_splits=5, seed=seed):
            m = fit_ranker(df[df["case"].isin(tr_c)], cols, DEVICE, seed=seed, **params)
            te = df[df["case"].isin(te_c)][["case", "system", "service", "is_rc"]].copy()
            te["score"] = m.predict(df.loc[te.index, cols]); te["seed"] = seed
            outs.append(te)
    return pd.concat(outs)

def summarise(oof):
    \"\"\"mean and spread of the metrics across the CV repetitions\"\"\"
    per_seed = pd.DataFrame({s: evaluate(g, by_system=False).loc["all"] for s, g in oof.groupby("seed")}).T
    return per_seed.mean(), per_seed.std()

ALL = feature_columns(df)
oof = cv_scores(ALL)
mean, sd = summarise(oof)
pd.DataFrame({"mean": mean, "± sd over CV repeats": sd}).round(3).T
"""),
("code", """
seed0 = oof[oof["seed"] == 0]
comp = pd.concat({"best rule": evaluate(base_scores), "XGBoost ranker (seed 0)": evaluate(seed0)}, axis=0)
comp.round(3)
"""),
("md", """
### Is the improvement real? Bootstrap over cases
We resample the 270 cases 2,000 times and look at the 95% interval of *(model − rule)*. If it stays above 0, the gain is not luck.
"""),
("code", """
r_model, r_rule = true_ranks(seed0), true_ranks(base_scores)
boot = {m: bootstrap_diff(r_model, r_rule, m) for m in ["AC@1", "AC@3", "Avg@5", "MRR"]}
pd.DataFrame(boot, index=["2.5%", "median", "97.5%"]).T
"""),
("md", """
## Which sources help? (ablation)
Same CV, same seeds, different feature sets.
"""),
("code", """
sets = {"metrics only": feature_columns(df, ("m_",)), "metrics + logs": feature_columns(df, ("m_", "l_"))}
if has_traces:
    sets["metrics + logs + traces (all)"] = ALL
sets["all, without within-case ranks"] = feature_columns(df, ranks=False)
abl = {}
for name, cols in sets.items():
    o = oof if cols == ALL else cv_scores(cols)
    abl[name] = summarise(o)[0]
pd.DataFrame(abl).T.round(3)
"""),
("md", """
## The hard test: leave one system out
Train on two systems, test on the third, never seen during training.
"""),
("code", """
loso = []
for sys_name, tr_c, te_c in leave_one_system_out(cases):
    m = fit_ranker(df[df["case"].isin(tr_c)], ALL, DEVICE, seed=0)
    te = df[df["case"].isin(te_c)][["case", "system", "service", "is_rc"]].copy(); te["score"] = m.predict(df.loc[te.index, ALL])
    loso.append(te)
loso = pd.concat(loso)
pd.concat({"rule (no training)": evaluate(base_scores), "XGBoost, unseen system": evaluate(loso)}, axis=0).round(3)
"""),
("md", """
**How to read it:** if the "unseen system" numbers stay close to the grouped-CV numbers, the model learned general failure behaviour. If they drop a lot, it partly memorised system-specific patterns, which is useful to know before deploying it anywhere new.

"""),
("md", """
### Which sources transfer to a new system?
Sock Shop has no traces, so when Online Boutique is held out the model learns trace patterns from Train Ticket alone. Repeat the unseen-system test with each feature set to see which sources carry over.
"""),
("code", """
def loso_scores(cols):
    out = []
    for _, tr_c, te_c in leave_one_system_out(cases):
        m = fit_ranker(df[df["case"].isin(tr_c)], cols, DEVICE, seed=0)
        te = df[df["case"].isin(te_c)][["case", "system", "service", "is_rc"]].copy(); te["score"] = m.predict(df.loc[te.index, cols])
        out.append(te)
    return pd.concat(out)

loso_src = {name: loso if cols == ALL else loso_scores(cols) for name, cols in sets.items()}
loso_ac1 = pd.DataFrame({name: evaluate(o)["AC@1"] for name, o in loso_src.items()}).T
loso_ac1.round(3).rename_axis("unseen-system AC@1")
"""),
("md", """
## Shortcut check: can it find a culprit it has never seen as a culprit?
Each system has only 5 services that the benchmark ever breaks. A model could cheat by learning to recognise *those 5 services* (e.g. by which metrics they expose) instead of learning failure behaviour. To rule this out we hold out **all cases of one root-cause service** (all its faults and repetitions), train on the rest, and test on the held-out ones, 15 times. In training, that service only ever appears as an *innocent* service.
"""),
("code", """
lorso = []
for (s, svc), g in df.drop_duplicates("case").groupby(["system", "root_cause_service"]):
    m = fit_ranker(df[~df["case"].isin(g["case"])], ALL, DEVICE, seed=0)
    te = df[df["case"].isin(g["case"])][["case", "system", "service", "is_rc"]].copy(); te["score"] = m.predict(df.loc[te.index, ALL])
    lorso.append(te)
lorso = pd.concat(lorso)
evaluate(lorso).round(3)
"""),
("md", """
If these numbers stay close to the grouped-CV ones, the model learned *how a culprit behaves*, not *which services are usually culprits*.

## Per fault type: where does the model help?
"""),
("code", """
fault = cases.set_index("case")["fault"]
tab = pd.DataFrame({"rule": (r_rule <= 1).groupby(fault.reindex(r_rule.index).to_numpy()).mean(),
                    "XGBoost": (r_model <= 1).groupby(fault.reindex(r_model.index).to_numpy()).mean()})
tab.round(3).T
"""),
("md", """
## Step 2: what kind of fault is it? (fault-type classifier)
Once we have a suspected service, its pattern of anomalies (CPU vs memory vs latency …) tells us the fault type. We train a 6-class XGBoost classifier on the **features of the service ranked first** (using the out-of-fold ranking, so nothing is leaked), with the same grouped folds. We also report an *oracle* version that is given the true culprit, to separate "wrong service" errors from "wrong fault" errors.
"""),
("code", """
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
def fault_cv(frame):
    pred = pd.Series(index=frame["case"], dtype=object)
    for tr_c, te_c in grouped_folds(cases, n_splits=5, seed=0):
        tr, te = frame[frame["case"].isin(tr_c)], frame[frame["case"].isin(te_c)]
        m, cols = fit_fault_classifier(tr, DEVICE)
        pred[te["case"].to_numpy()] = [FAULTS[i] for i in m.predict(te[cols])]
    truth = frame.set_index("case")["fault"].reindex(pred.index)
    return pred, truth

oracle = df.copy(); oracle["score"] = oracle["is_rc"]
ranked = df.merge(seed0[["case", "service", "score"]], on=["case", "service"])
fault_res = {}
for name, frame in {"service from the ranker (end-to-end)": fault_frame(ranked), "true service (oracle)": fault_frame(oracle)}.items():
    p, t = fault_cv(frame)
    fault_res[name] = {"accuracy": accuracy_score(t, p), "macro_F1": f1_score(t, p, average="macro")}
    if "end-to-end" in name: p_e2e, t_e2e = p, t
pd.DataFrame(fault_res).T.round(3)
"""),
("code", """
cm = confusion_matrix(t_e2e, p_e2e, labels=FAULTS, normalize="true")
plt.figure(figsize=(6.5, 5)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=FAULTS, yticklabels=FAULTS, vmin=0, vmax=1)
plt.xlabel("predicted fault"); plt.ylabel("true fault"); plt.title("fault type, end-to-end (rows sum to 1)"); plt.tight_layout()
plt.savefig(FIG / "fault_confusion.png", dpi=120); plt.show()
"""),
("md", """
## What does the ranker look at?
"""),
("code", """
final = fit_ranker(df, ALL, DEVICE, seed=0)
imp = pd.Series(final.get_booster().get_score(importance_type="total_gain")).reindex(ALL).fillna(0)
(imp / imp.sum() * 100).sort_values().tail(15).plot(kind="barh", figsize=(8, 5), color="#3b6fb6")
plt.xlabel("% of total gain"); plt.title("ranker: top 15 features"); plt.tight_layout(); plt.savefig(FIG / "ranker_importance.png", dpi=120); plt.show()
final.save_model(str(ROOT / "models" / "ranker.json"))
"""),
("code", """
results = {"has_traces": has_traces, "device": DEVICE,
           "rule": {k: round(v, 4) for k, v in rank_metrics(r_rule).items()},
           "xgb_grouped_cv_mean": mean.round(4).to_dict(), "xgb_grouped_cv_sd": sd.round(4).to_dict(),
           "xgb_by_system_seed0": evaluate(seed0).round(4).to_dict(orient="index"),
           "xgb_leave_one_system_out": evaluate(loso).round(4).to_dict(orient="index"),
           "xgb_leave_one_rc_service_out": evaluate(lorso).round(4).to_dict(orient="index"),
           "xgb_loso_by_source": {k: evaluate(o).round(4).to_dict(orient="index") for k, o in loso_src.items()},
           "bootstrap_xgb_minus_rule": {k: list(map(float, v)) for k, v in boot.items()},
           "ablation": pd.DataFrame(abl).T.round(4).to_dict(orient="index"),
           "fault_type": {k: {m: round(v, 4) for m, v in d.items()} for k, d in fault_res.items()}}
json.dump(results, open(RESULTS / "ml_ranker.json", "w"), indent=1)
seed0.to_parquet(ROOT / "data" / "processed" / "oof_xgb.parquet", index=False)
print(json.dumps(results["xgb_grouped_cv_mean"], indent=1))
"""),
("md", """
### Takeaways
- **The learned ranker clearly beats every rule:** AC@1 0.944 ± 0.010 vs 0.707 for the best rule (+0.233, 95% bootstrap CI [+0.178, +0.289]). AC@3 is 0.990, so the culprit is almost always in the top 3.
- **It learns behaviour, not names:** with each culprit service held out as a culprit, AC@1 stays at 0.944.
- **Hardest faults are network faults:** AC@1 is 0.82 for delay and 0.84 for loss (rule: 0.36 and 0.49). These symptoms spread to callers, so the slowest service is often not the cause. cpu, memory and socket faults reach 1.00.
- **Traces add nothing inside a system:** metrics alone give 0.951. The ranker mostly uses *latency-change ranks* plus CPU and memory changes.
- **Trace features hurt on a new system:** unseen-system AC@1 is 0.744 with all sources but 0.878 with metrics only. The loss is on Online Boutique (0.41 vs 0.81): Sock Shop has no traces, so the trace patterns come from Train Ticket alone and don't transfer. Removing the within-case rank features also recovers 0.867, so the *trace rank features* are what overfits. **For a new system, use the metrics-only model.**
- **Fault type:** 0.826 accuracy end-to-end (0.848 with the true service). The mix-ups are disk ↔ memory and delay ↔ loss, which look alike physically.

➡️ Next: **05_deep_learning**: can a neural network reading the raw time series do better?
"""),
]

# ======================================================================= 05
NB["05_deep_learning"] = [
("md", """
# 05 · Deep learning on the raw time series (GPU)

Notebook 04 used **hand-made features** (z-scores, ratios). Here a **1D convolutional neural network (CNN)** reads each service's raw metric series directly and learns its own features.

**Input** per (case, service): 8 metric types × 144 time steps (24 minutes at 10-second resolution, fault injected in the middle). Each series is standardised with the median/IQR of its own normal window (same rule as notebook 02), plus 8 channels saying which metric types exist for this service.

**Model:** the same small CNN scores every service; inside a case, the scores go through a **softmax** ("which one of these is the culprit?") and the loss is the cross-entropy with the true culprit. This is called a *listwise* ranking loss. Because the network is shared across services, it learns what a *culprit-like pattern* looks like, never the service names.

**Evaluation:** exactly the same grouped 5-fold splits as notebook 04 (seed 0), plus leave-one-system-out.

> With only 270 cases a neural network can easily over-fit; the comparison with XGBoost is the point of this notebook, not a guaranteed win.
"""),
("code", SETUP + """
import torch, time
from rcalens.data import case_index
from rcalens.dl import build_dataset, train_cnn, score_cases
from rcalens.evaluate import evaluate, true_ranks, bootstrap_diff, grouped_folds, leave_one_system_out
idx = case_index()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
t = time.time(); data = build_dataset(idx); print(f"tensors for {len(data)} cases built in {time.time() - t:.0f}s")
"""),
("code", """
import os
EPOCHS = int(os.environ.get("RCA_EPOCHS", 60))   # set RCA_EPOCHS=3 for a quick smoke test
SEEDS = (0, 1, 2)                                # 3 networks per fold, averaged (an "ensemble" reduces variance)
sys_of = idx.set_index("case")["system"]
def run(train_cases, test_cases):
    scores = None
    for s in SEEDS:
        m = train_cnn(data, list(train_cases), DEVICE, epochs=EPOCHS, seed=s)
        o = score_cases(m, data, list(test_cases), DEVICE)
        scores = o if scores is None else scores.assign(score=scores["score"] + o["score"])
    scores["system"] = scores["case"].map(sys_of)
    return scores

t = time.time()
cnn_cv = pd.concat([run(tr, te) for tr, te in grouped_folds(idx, n_splits=5, seed=0)])
print(f"grouped CV done in {time.time() - t:.0f}s")
evaluate(cnn_cv).round(3)
"""),
("code", """
xgb = pd.read_parquet(ROOT / "data" / "processed" / "oof_xgb.parquet")
rule = pd.read_parquet(ROOT / "data" / "processed" / "baseline_scores.parquet")
pd.concat({"rule": evaluate(rule).loc["all"], "XGBoost ranker": evaluate(xgb).loc["all"], "1D CNN": evaluate(cnn_cv).loc["all"]}, axis=1).T.round(3)
"""),
("code", """
r_cnn, r_xgb = true_ranks(cnn_cv), true_ranks(xgb)
pd.DataFrame({m: bootstrap_diff(r_cnn, r_xgb, m) for m in ["AC@1", "Avg@5", "MRR"]}, index=["2.5%", "median", "97.5%"]).T.rename_axis("CNN − XGBoost")
"""),
("md", """
### Ensemble: average the two model types
Scores are on different scales, so each case's scores are turned into within-case ranks first, then averaged.
"""),
("code", """
ens = xgb[["case", "service", "system", "is_rc", "score"]].rename(columns={"score": "score_xgb"})
ens = ens.merge(cnn_cv[["case", "service", "score"]].rename(columns={"score": "score_cnn"}), on=["case", "service"])
ens["score"] = ens.groupby("case")["score_xgb"].rank(pct=True) + ens.groupby("case")["score_cnn"].rank(pct=True)
evaluate(ens).round(3)
"""),
("md", """
## Unseen system (leave one system out)
"""),
("code", """
cnn_loso = pd.concat([run(tr, te) for _, tr, te in leave_one_system_out(idx)])
evaluate(cnn_loso).round(3)
"""),
("code", """
out = {"device": DEVICE, "epochs": EPOCHS, "seeds": list(SEEDS),
       "cnn_grouped_cv": evaluate(cnn_cv).round(4).to_dict(orient="index"),
       "ensemble_xgb_cnn": evaluate(ens).round(4).to_dict(orient="index"),
       "cnn_leave_one_system_out": evaluate(cnn_loso).round(4).to_dict(orient="index"),
       "bootstrap_cnn_minus_xgb": {m: list(map(float, bootstrap_diff(r_cnn, r_xgb, m))) for m in ["AC@1", "Avg@5", "MRR"]}}
json.dump(out, open(RESULTS / "deep_learning.json", "w"), indent=1)
cnn_cv.to_parquet(ROOT / "data" / "processed" / "oof_cnn.parquet", index=False)
print("saved reports/results/deep_learning.json")
"""),
("md", """
### Takeaways
- **The CNN is as good as XGBoost inside a system:** AC@1 0.930 vs 0.941 (same split). The difference is −0.011 with 95% CI [−0.037, +0.015], so no real difference. It needs no hand-made features.
- **The ensemble doesn't help** (AC@1 0.930, AC@3 0.993): averaging the two scores doesn't add accuracy at the top.
- **The CNN transfers better to a new system than XGBoost on all sources** (0.859 vs 0.744). It reads metrics only, so it has no trace problem. It is level with metrics-only XGBoost (0.878).
- **Cost:** ~17 minutes of A100 time for the grouped CV, vs seconds for XGBoost. For this data size, the gradient-boosted ranker is the practical choice; the CNN is the evidence that the signal is in the raw series.
"""),
]

# ======================================================================= 06
NB["06_conclusion"] = [
("md", """
# 06 · Results summary and comparison with IncidentLens

This notebook collects the saved results of notebooks 03–05 into one table (it re-trains nothing).
"""),
("code", SETUP + """
from rcalens.evaluate import evaluate
base = json.load(open(RESULTS / "baselines.json"))
ml = json.load(open(RESULTS / "ml_ranker.json"))
dl = json.load(open(RESULTS / "deep_learning.json")) if (RESULTS / "deep_learning.json").exists() else None
rows = {"random (expected)": base["overall"]["random (expected)"],
        f"best rule: {base['best_rule']}": base["overall"][base["best_rule"]],
        "XGBoost ranker (grouped CV, mean of 3)": ml["xgb_grouped_cv_mean"],
        "XGBoost ranker, unseen system": ml["xgb_leave_one_system_out"]["all"],
        "XGBoost ranker, unseen culprit service": ml["xgb_leave_one_rc_service_out"]["all"]}
for src, r in ml.get("xgb_loso_by_source", {}).items():
    rows[f"XGBoost, unseen system, {src}"] = r["all"]
if dl:
    rows["1D CNN (grouped CV)"] = dl["cnn_grouped_cv"]["all"]
    rows["XGBoost + CNN ensemble"] = dl["ensemble_xgb_cnn"]["all"]
    rows["1D CNN, unseen system"] = dl["cnn_leave_one_system_out"]["all"]
summary = pd.DataFrame(rows).T[["AC@1", "AC@3", "AC@5", "Avg@5", "MRR"]]
print("trace features:", ml["has_traces"], "| training device:", ml["device"])
summary.round(3)
"""),
("code", """
pd.DataFrame(ml["fault_type"]).T.round(3).rename_axis("fault-type classifier (6 classes)")
"""),
("md", """
## Conclusions
1. **Root-cause service:** a learned XGBoost ranker puts the true culprit first in **94.4%** of the 270 cases (top-3: 99.0%), vs 70.7% for the best rule, 51.1% for a BARO-style z-score and 5.5% for a random guess. The gain is significant (95% CI +17.8 to +28.9 points).
2. **Generalisation:**
   - A culprit service never seen as a culprit: 94.4%, no drop.
   - A completely new system: **87.8%** with metrics-only features, 85.9% with the CNN. Adding trace features drops this to 74.4%, because trace patterns learned from one system don't transfer.
3. **Fault type:** 82.6% accuracy over 6 fault types, end-to-end.
4. **Where it fails:** network delay and packet loss, where the symptom spreads to the callers.
5. **Deep learning:** a 1D CNN on raw time series matches XGBoost (93.0% vs 94.1%, not significant) without feature engineering, at much higher training cost.

**Comparison with IncidentLens** (ITSM ticket closure-category prediction, accuracy 0.718):
- RCAEval is the stronger project for a resume. Its labels are real (faults were injected, so the root cause is known), it has a published benchmark and baselines, and it was checked on unseen services and unseen systems.
- IncidentLens shows a different skill: messy business data and a served model with API and UI.
"""),
]

if __name__ == "__main__":
    for n in (sys.argv[1:] or NB):
        write(n, NB[n])
