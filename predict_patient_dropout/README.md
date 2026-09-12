# SDTM Discontinuation-Prediction Pipeline

A configurable pipeline that predicts whether a **randomized** clinical-trial
patient will discontinue before completing the study, from source data
organized to the **CDISC SDTM** standard. It works on **any** study in that
format -- not just one fixed dataset.

Given an SDTM study (a local folder, a URL, or the bundled public reference
study), the pipeline:

1. **Validates** the study against its own `define.xml` and produces a
   compliance report; it stops before any analysis if the data doesn't meet
   the standard.
2. **Converts** SDTM domains to clean CSVs.
3. **Engineers** leakage-safe, baseline-only features and makes a 70/30
   stratified development/holdout split.
4. **Trains and evaluates** two models (Logistic Regression, Gradient
   Boosting) on the primary binary outcome (discontinued vs. completed), plus
   Kaplan-Meier / Cox / competing-risks models on the time-to-event view of
   the same outcome.
5. **Explains** every model (odds ratios, permutation importance, SHAP if
   installed, hazard ratios, cause-specific incidence).
6. **Delivers** the results as files and/or a combined Word report that
   records where the data came from and whether it passed the compliance
   check.

> **Design priority: rigor over a headline metric.** Predicting dropout is
> trivially easy using information that exists only *because* a patient
> dropped out. Preventing that leakage is the core of the pipeline, so a
> modest, honest baseline-only accuracy is expected and is the point.

---

## Repository structure

```
run_pipeline.py                   # single entry point: resolve input → compliance gate → 5 stages → report
sdtm_reader.py                    # input resolution (local/URL) + define.xml compliance checker + SDTM → CSV
build_features_cdiscpilot01.py    # baseline features + 70/30 stratified split
model_cdiscpilot01.py             # binary models: train, tune, evaluate, explain
survival_cdiscpilot01.py          # Kaplan–Meier, log-rank, Cox + PH, competing risks
                                  # (the combined Word report is built inside run_pipeline.py)
requirements.txt
PROMPT.md                         # conversational "AI function" front-end for this pipeline
README.md                         # this file
<outdir>/                         # (generated) csv/ features/ model/ survival/, compliance
                                  # report, provenance record, and report.docx
```

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+ (3.11 recommended). `python-docx` is required only for the Word
report; `shap` and `lifelines` degrade gracefully if not installed (SHAP
attribution and the Cox model are simply skipped, with a note printed).

---

## Providing your data

You must supply exactly one of the following. In every case, the location you
give must contain (or resolve to) **`define.xml` plus every dataset it
declares**, as `.xpt` or `.csv` files.

| Flag | What it accepts |
|---|---|
| `--input-dir <path>` | A local folder |
| `--input-url <url>` | A GitHub folder URL (`https://github.com/<owner>/<repo>/tree/<branch>/<path>`), a `raw.githubusercontent.com` folder URL, or a direct link to a `.zip` archive of the study |
| `--fetch-reference` | Shorthand for the public CDISC Pilot 01 reference study (no other input needed) |

The pipeline never assumes a specific study, sponsor, or domain set beyond
what SDTM itself requires -- domains and variables are discovered from
`define.xml` at run time, not hardcoded.

---

## Quick start

```bash
# Your own study, from a local folder:
python run_pipeline.py --study "My Phase 2 Study" --input-dir ./my_sdtm --report both

# Your own study, from a URL (GitHub folder, raw.githubusercontent.com folder, or .zip):
python run_pipeline.py --study "Partner Study" \
  --input-url https://github.com/<org>/<repo>/tree/main/sdtm_data --report both

# The public reference study (downloads it automatically):
python run_pipeline.py --study "CDISC Pilot 01" --fetch-reference --report both
```

### Options

| Flag | Default | Purpose |
|---|---|---|
| `--study` | *(required)* | Label for the run; used in the Word report title/filename |
| `--input-dir` / `--input-url` / `--fetch-reference` | *(exactly one required)* | Where the SDTM study comes from |
| `--target-mode` | `clinical` | `clinical` drops administrative sponsor-terminations from the outcome; `any` keeps all non-completions |
| `--landmark-day` | `1` | Baseline window (study day ≤ N) -- the leakage guard |
| `--death` | `event` | Treat DEATH as an event, or `exclude` |
| `--qs-endpoints` | `ACTOT,CIBIC,NPTOT` | Comma-separated `QSTESTCD` codes used as baseline questionnaire/scale features; override for a study that uses different instruments |
| `--report` | `both` | `files`, `word`, or `both` |
| `--outdir` / `--seed` | `sdtm_analysis_out` / `42` | Output folder / reproducibility |

---

## The compliance gate

Before any analysis runs, the pipeline checks the resolved input against its
**own** `define.xml` and writes `<outdir>/compliance_report.json` and
`compliance_report.txt`. Checks performed:

- `define.xml` is present and well-formed XML.
- Every dataset `define.xml` declares (`ItemGroupDef`) is present as a
  `.xpt`/`.csv` file.
- Every **mandatory** variable declared for a dataset is present as a column
  (missing mandatory variables are **errors**; missing optional variables and
  undeclared extra columns are **warnings**).
- Where a `DOMAIN` column exists, its values match the domain code
  `define.xml` declares for that dataset.
- A light datatype spot-check: numeric-typed variables that mostly fail to
  parse as numbers, or date-typed variables that aren't ISO 8601(-partial),
  are flagged as warnings.

If there is **at least one error**, the pipeline prints:

```
User's input does not meet CDISC SDTM requirements that results in
termination of further analysis
```

and stops (exit code 2) with the full itemized list of deviations already
written to `compliance_report.txt` -- no partial or best-effort analysis is
attempted. Warnings alone do not stop the run.

**Scope note:** this is a `define.xml`-conformance check plus a few universal
SDTM invariants (key variables, `DOMAIN` values). It is *not* a substitute
for a full CDISC SDTM Implementation-Guide validator such as
[Pinnacle 21 / P21 Community](https://www.pinnacle21.com/), which additionally
checks controlled terminology, cross-domain consistency, and IG-version-
specific structural rules that this pipeline does not attempt.

On a compliant run, `<outdir>/provenance.json` records where the data came
from (path or URL), when it was retrieved, and the compliance result -- and
this same information is reproduced in the Word report's "Data source &
compliance" section, so every report is traceable to its input.

---

## Methodology (unchanged regardless of the study you point it at)

| Question | Answer |
|---|---|
| Baseline-only or time-varying **features**? | **Baseline-only** for every model -- enforced by a randomization landmark (day ≤ `--landmark-day`) |
| Cutoff relative to completion? | **None, and none needed** -- a completion-relative cutoff is only required with time-varying features; the landmark is *start*-relative |
| Binary or time-to-event **outcome**? | **Both.** Binary (Logistic Regression, Gradient Boosting) is primary; time-to-event (Kaplan–Meier, Cox, competing risks) adds *when*, *how fast*, and *why* |
| Explainable? | **Yes** -- LR odds ratios, permutation importance, Cox hazard ratios, cause-specific incidence (+ SHAP if installed) |

**Leakage policy.** Features use only information knowable at randomization:

- **Used:** DM demographics + *planned* arm; MH history; baseline VS/LB/QS
  (the questionnaire endpoints set by `--qs-endpoints`).
- **Excluded from features:** AE (a common dropout mechanism), EX/SV/CM/SE,
  SUPP\*, the *actual* arm, and end-of-participation dates -- all only
  knowable during/after the trial. DS is used **solely to define the
  outcome** (and event timing), never as a predictor.
- Imputation and scaling are fit on **development data only**, then applied
  to the holdout; the holdout is scored exactly once, after model selection.

Every decision is recorded to `features/feature_manifest.json`.

---

## Models at a glance

| Model | Dependent variable | Predictors | Objective |
|---|---|---|---|
| Logistic Regression | Binary (discontinued 1 / completed 0) | Baseline features | Interpretable risk score -- *who* |
| Gradient Boosting | Binary | Baseline features | Non-linear risk score -- *who* |
| Kaplan–Meier (+ log-rank) | Survival curve S(t) | Arm (grouping) | *When* dropout happens, by arm |
| Cox proportional hazards | Hazard (instantaneous risk) | Arm + baseline numeric/categorical covariates | *How much* each factor changes risk |
| Competing-risks (Aalen–Johansen) | Cause-specific cumulative incidence | Cause grouping (from `DISPOSITION`) | *Why* patients drop out |

---

## Outputs (`<outdir>/`)

- `compliance_report.json` / `.txt` -- the compliance findings for this run
- `provenance.json` -- where the data came from, when, and the compliance result
- `csv/` -- one clean CSV per SDTM domain; `subjects.csv`; `patient_long.csv`
- `features/` -- `model_matrix.csv`, `dev.csv`, `holdout.csv`, `feature_manifest.json`
- `model/` -- `metrics.json`, `evaluation_curves.png`, importances, predictions, saved models
- `survival/` -- `km_by_arm.png`, `cif_by_cause.png`, `cox_summary.csv`, `cox_ph_check.txt`, `survival_report.json`
- `<study>_report.docx` -- combined Word report (data source, cohort, methodology,
  binary + survival results), with `--report word`/`both`

---

## Two ways to run it

**a) Directly** -- run `run_pipeline.py` in any terminal or code editor with
the parameters above.

**b) As an AI function** -- paste `PROMPT.md` into any code-capable AI
assistant (e.g. GitHub Copilot Chat in Agent mode, a Copilot Studio agent, or
an OpenAI Custom GPT with Code Interpreter). It turns the assistant into a
user-friendly front-end: it collects the run parameters in plain language,
builds and executes the `run_pipeline.py` command, and reports the output
back in the chat using only real numbers from that run. `run_pipeline.py`
stays the fixed, validated backend; the assistant is only the interface.

---

## Environment

Python 3.9+ (3.11 recommended). **Required:** `pandas`, `numpy`,
`scikit-learn`, `scipy`, `matplotlib`, `joblib`, `requests`, `python-docx`,
`lifelines`. **Optional:** `shap` (per-patient attribution), `pyreadstat`
(faster XPORT reading).

## Data sources this pipeline is known to work with

The pipeline has been validated end-to-end against the CDISC SDTM Pilot 01
study published by PHUSE:
<https://github.com/phuse-org/phuse-scripts/tree/master/data/sdtm/cdiscpilot01>
(reachable via `--fetch-reference`, or directly via `--input-url` pointed at
that folder). Any study following the same SDTM structure -- your own trial,
or a partner's -- can be analyzed the same way via `--input-dir` or
`--input-url`.
