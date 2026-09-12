#!/usr/bin/env python3
"""
run_pipeline.py  (v3)  --  the single "AI function"
====================================================
One entry point that turns ANY user's CDISC-SDTM-formatted study into a
discontinuation-prediction analysis. It resolves the input, enforces a
compliance gate against the study's own define.xml, then runs every stage
end to end.

USER INPUTS  (exactly one of the three)
  --input-dir    local folder with the user's SDTM files; define.xml REQUIRED
  --input-url    a GitHub folder URL, a raw.githubusercontent.com folder URL,
                 or a direct .zip archive URL of an SDTM study; define.xml
                 REQUIRED in that location
  --fetch-reference   shorthand for the public CDISC Pilot 01 reference study

COMPLIANCE GATE (hard stop)
  The resolved input is checked against its OWN define.xml: every declared
  dataset must be present, every mandatory variable must be present, DOMAIN
  values must match, etc. (see sdtm_reader.py docstring for exact scope --
  this is a define.xml-conformance check, not a full IG validator). Every
  deviation found is written to <outdir>/compliance_report.{json,txt} and
  printed. If there is at least one ERROR, the pipeline prints:
     "User's input does not meet CDISC SDTM requirements that results in
      termination of further analysis"
  and stops (exit code 2) -- no further analysis is attempted. Warnings do
  not stop the run.

BACKEND (on compliant input)
  1. convert SDTM -> CSV (+ subject spine + patient-long)
  2. leakage-safe baseline feature engineering + 70/30 stratified split
  3. train / tune / evaluate models, with explainability
  4. time-to-event sensitivity analysis
  5. deliver results as FILES and/or a combined WORD report, including the
     data-source provenance (where the data came from, when it was
     retrieved, and the compliance result)

OUTPUT
  everything under --outdir (default: sdtm_analysis_out)

USAGE
  python run_pipeline.py --study "My Phase 2 Study" --input-dir ./my_sdtm --report both
  python run_pipeline.py --study "Partner Study" --input-url https://github.com/<org>/<repo>/tree/main/sdtm --report both
  python run_pipeline.py --study "CDISC Pilot 01" --fetch-reference --report word

Requires: pandas, numpy, scikit-learn, matplotlib, joblib, requests
Optional: python-docx (Word report), shap, lifelines, pyreadstat
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import sdtm_reader as reader   # same folder

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
def _run_script(script, args):
    cmd = [sys.executable, str(HERE / script)] + args
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"Stage failed: {script} (exit {r.returncode})")


def _safe_read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


# --------------------------------------------------------------------------
def build_word_report(outdir: Path, study: str):
    """Combine the key results into a single Word document."""
    try:
        from docx import Document
        from docx.shared import Pt, Inches, RGBColor
    except ImportError:
        print("[note] python-docx not installed -> Word report skipped "
              "(pip install python-docx). Files are still available.")
        return None

    feat = _safe_read_json(outdir / "features" / "feature_manifest.json")
    met = _safe_read_json(outdir / "model" / "metrics.json")
    prov = _safe_read_json(outdir / "provenance.json")
    comp = _safe_read_json(outdir / "compliance_report.json")

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    doc.add_heading(f"Discontinuation Prediction \u2014 {study}", level=0)
    doc.add_paragraph("Automated SDTM analysis report (baseline-only, "
                      "leakage-safe).").italic = True

    # ---- Data source & compliance (NEW) ------------------------------------
    doc.add_heading("Data source & compliance", level=1)
    if prov:
        src_line = (f"{prov.get('source_type', 'n/a')} "
                    f"({prov.get('source_location', 'n/a')})")
        doc.add_paragraph(f"Source: {src_line}")
        doc.add_paragraph(f"Retrieved: {prov.get('retrieved_at_utc', 'n/a')} (UTC)")
    if comp:
        result = "COMPLIANT" if comp.get("compliant") else "NON-COMPLIANT"
        doc.add_paragraph(
            f"CDISC SDTM compliance check: {result} "
            f"({comp.get('n_domains_declared', 'n/a')} domain(s) declared in "
            f"define.xml; {comp.get('n_errors', 0)} error(s), "
            f"{comp.get('n_warnings', 0)} warning(s)). Full findings are in "
            f"compliance_report.txt / compliance_report.json.")
        warnings_ = [f for f in comp.get("findings", []) if f["severity"] == "warning"]
        if warnings_:
            t = doc.add_table(rows=1, cols=3); t.style = "Light Grid Accent 1"
            for i, hd in enumerate(["Domain", "Check", "Detail"]):
                t.rows[0].cells[i].text = hd
            for f in warnings_[:15]:
                c = t.add_row().cells
                c[0].text = f["domain"]; c[1].text = f["check"]; c[2].text = f["detail"]
            if len(warnings_) > 15:
                doc.add_paragraph(f"...and {len(warnings_) - 15} more warning(s); "
                                  f"see compliance_report.txt.")
    else:
        doc.add_paragraph("No provenance/compliance record was found for this run.")

    # Executive summary
    doc.add_heading("Executive summary", level=1)
    best = met.get("best_model", "n/a")
    bkey = "gbm" if best == "GradientBoosting" else "logreg"
    bm = met.get(bkey, {})
    n_rand = feat.get("n_randomized", "n/a")
    n_coh = feat.get("n_modeling_cohort", "n/a")
    bal = feat.get("target_balance", {})
    doc.add_paragraph(
        f"From {n_rand} randomized subjects, a modeling cohort of {n_coh} was "
        f"analyzed (target balance {bal}). The best model ({best}) achieved "
        f"holdout ROC-AUC {bm.get('roc_auc', 'n/a')} and PR-AUC "
        f"{bm.get('pr_auc', 'n/a')}.")

    # Cohort & target
    doc.add_heading("Cohort & target", level=1)
    subj_p = outdir / "subjects.csv"
    if subj_p.exists():
        s = pd.read_csv(subj_p)
        t = doc.add_table(rows=1, cols=2); t.style = "Light Grid Accent 1"
        t.rows[0].cells[0].text = "Metric"; t.rows[0].cells[1].text = "Value"
        rows = [("Subjects (DM)", len(s)),
                ("Randomized", int(s["RANDOMIZED"].sum()) if "RANDOMIZED" in s else "n/a"),
                ("No-disposition excluded", feat.get("n_no_disposition_excluded", 0)),
                ("Modeling cohort", n_coh),
                ("Target balance (0/1)", bal)]
        for k, v in rows:
            c = t.add_row().cells; c[0].text = str(k); c[1].text = str(v)

    # Methodology (the four required answers)
    doc.add_heading("Methodology", level=1)
    meth = [
        ("Information used", feat.get("information_type", "baseline-only")
         + f" (randomization landmark, study day \u2264 {feat.get('landmark_day', 1)})"),
        ("Cutoff relative to completion", "not applicable \u2014 no post-baseline data is used"),
        ("Outcome", f"{feat.get('outcome_primary', 'binary')} primary; "
         f"{feat.get('outcome_also_provided', 'time-to-event')} provided for sensitivity"),
        ("Explainability", "logistic-regression odds ratios + permutation importance "
         "+ SHAP (if available)"),
        ("Leakage exclusions", ", ".join(feat.get("dm_excluded_leakage", [])) or "recorded in manifest"),
    ]
    for k, v in meth:
        p = doc.add_paragraph(); p.add_run(f"{k}: ").bold = True; p.add_run(str(v))

    # Holdout performance
    doc.add_heading("Holdout performance", level=1)
    t = doc.add_table(rows=1, cols=6); t.style = "Light Grid Accent 1"
    for i, hd in enumerate(["Model", "ROC-AUC", "PR-AUC", "F1", "Bal.Acc", "Brier"]):
        t.rows[0].cells[i].text = hd
    for key in ("logreg", "gbm"):
        m = met.get(key, {})
        if not m:
            continue
        c = t.add_row().cells
        c[0].text = m.get("model", key)
        c[1].text = str(m.get("roc_auc", "")); c[2].text = str(m.get("pr_auc", ""))
        c[3].text = str(m.get("f1", "")); c[4].text = str(m.get("balanced_accuracy", ""))
        c[5].text = str(m.get("brier", ""))

    # Key drivers
    doc.add_heading("Key drivers (attribution)", level=1)
    imp_p = outdir / "model" / "permutation_importance.csv"
    if imp_p.exists():
        imp = pd.read_csv(imp_p).head(8)
        t = doc.add_table(rows=1, cols=2); t.style = "Light Grid Accent 1"
        t.rows[0].cells[0].text = "Feature"; t.rows[0].cells[1].text = "Permutation importance"
        for _, r in imp.iterrows():
            c = t.add_row().cells
            c[0].text = str(r["feature"]); c[1].text = f"{r['importance']:.4f}"

    # Discontinuation by arm (key insight)
    mm_p = outdir / "features" / "model_matrix.csv"
    if mm_p.exists():
        mm = pd.read_csv(mm_p)
        arm = "ARMCD" if "ARMCD" in mm.columns else ("ARM" if "ARM" in mm.columns else None)
        if arm:
            doc.add_heading("Key insight \u2014 discontinuation by planned arm", level=1)
            rate = mm.groupby(arm)["TARGET"].agg(["mean", "size"]).sort_values("mean", ascending=False)
            t = doc.add_table(rows=1, cols=3); t.style = "Light Grid Accent 1"
            for i, hd in enumerate(["Arm", "Discontinue %", "N"]):
                t.rows[0].cells[i].text = hd
            for idx, r in rate.iterrows():
                c = t.add_row().cells
                c[0].text = str(idx); c[1].text = f"{r['mean']*100:.1f}"; c[2].text = str(int(r["size"]))

    # Figures
    for fig, cap in [("evaluation_curves.png", "ROC / PR / confusion (holdout)"),
                     ("shap_summary.png", "SHAP feature attribution")]:
        fp = outdir / "model" / fig
        if fp.exists():
            doc.add_heading(cap, level=1)
            doc.add_picture(str(fp), width=Inches(6.3))

    # Survival / time-to-event
    surv = _safe_read_json(outdir / "survival" / "survival_report.json")
    if surv:
        doc.add_heading("Survival analysis (time-to-event)", level=1)
        med = surv.get("median_time_overall")
        lr = surv.get("logrank", {})
        p = surv.get("cox", {})
        intro = (f"Complementing the binary models, a time-to-event view treats the "
                 f"outcome as days-until-dropout with completers censored. Across "
                 f"{surv.get('n', 'n/a')} patients ({surv.get('events', 'n/a')} events), "
                 f"the median time in study is "
                 f"{('%.0f days' % med) if isinstance(med, (int, float)) and med == med else 'not reached'}.")
        doc.add_paragraph(intro)
        if lr:
            doc.add_paragraph(
                f"Kaplan-Meier retention differs by treatment arm: log-rank "
                f"chi-square {lr.get('chi2')} (df {lr.get('dof')}), p = "
                f"{lr.get('p_value'):.2e}.")
        by_arm = surv.get("km_by_arm", {})
        if by_arm:
            t = doc.add_table(rows=1, cols=4); t.style = "Light Grid Accent 1"
            for i, hd in enumerate(["Arm", "N", "Events", "Median days (NR = not reached)"]):
                t.rows[0].cells[i].text = hd
            for a, d in by_arm.items():
                c = t.add_row().cells
                c[0].text = str(a); c[1].text = str(d.get("n"))
                c[2].text = str(d.get("events"))
                c[3].text = "NR" if d.get("median") is None else f"{d['median']:.0f}"
        if p and p.get("hazard_ratios"):
            doc.add_paragraph(
                f"Cox proportional-hazards model (baseline covariates; concordance "
                f"{p.get('concordance')}). A hazard ratio (HR) above 1 means higher "
                f"instantaneous risk of dropping out:")
            t = doc.add_table(rows=1, cols=3); t.style = "Light Grid Accent 1"
            for i, hd in enumerate(["Covariate", "Hazard ratio", "p"]):
                t.rows[0].cells[i].text = hd
            hrs = p["hazard_ratios"]; pv = p.get("p_values", {})
            for k in hrs:
                c = t.add_row().cells
                c[0].text = str(k); c[1].text = f"{hrs[k]:.2f}"
                c[2].text = f"{pv.get(k, float('nan')):.3f}"
            viol = p.get("ph_violations")
            doc.add_paragraph(
                "Proportional-hazards assumption: " +
                ("holds for all covariates tested." if viol == [] else
                 (f"possible violation for {viol} (interpret those HRs as period-averaged)."
                  if viol else "not tested.")))
        else:
            doc.add_paragraph(
                "Cox model: not available in this run (install lifelines to enable "
                "hazard ratios and the proportional-hazards check). Kaplan-Meier, "
                "the log-rank test, and competing-risks incidence are shown above.")
        cif = surv.get("cif_final", {})
        if cif:
            top = ", ".join(f"{k.title()} {v:.0%}" for k, v in
                            sorted(cif.items(), key=lambda kv: -kv[1])[:4])
            doc.add_paragraph(
                f"Competing risks: \u201Cany-cause\u201D dropout combines competing reasons. "
                f"By end of study the cause-specific cumulative incidence is {top}.")
        for fig, cap in [("km_by_arm.png", "Kaplan-Meier retention by arm"),
                         ("cif_by_cause.png", "Cause-specific cumulative incidence")]:
            fp = outdir / "survival" / fig
            if fp.exists():
                doc.add_picture(str(fp), width=Inches(5.6))

    doc.add_paragraph()
    note = doc.add_paragraph()
    note.add_run("Reproducibility: ").bold = True
    note.add_run("all methodology choices are recorded in "
                 "features/feature_manifest.json; provenance and the compliance "
                 "result are recorded in provenance.json / compliance_report.json; "
                 "the holdout was scored once after model selection on development.")

    safe_name = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in study).replace(" ", "_")
    out = outdir / f"{safe_name}_report.docx"
    doc.save(str(out))
    print(f"\nWord report written: {out}")
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, help="name/label for the study")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", help="local folder with the user's SDTM files (+ define.xml)")
    src.add_argument("--input-url", help="GitHub folder URL, raw.githubusercontent.com "
                                         "folder URL, or a direct .zip archive URL of an "
                                         "SDTM study (+ define.xml)")
    src.add_argument("--fetch-reference", action="store_true",
                     help="shorthand for the public CDISC Pilot 01 reference study")
    ap.add_argument("--outdir", default="sdtm_analysis_out")
    ap.add_argument("--target-mode", choices=["clinical", "any"], default="clinical")
    ap.add_argument("--landmark-day", type=int, default=1)
    ap.add_argument("--death", choices=["event", "exclude"], default="event")
    ap.add_argument("--qs-endpoints", default="ACTOT,CIBIC,NPTOT",
                    help="comma-separated QSTESTCD codes to use as baseline QS "
                         "features (default matches the CDISC Pilot 01 instruments)")
    ap.add_argument("--report", choices=["files", "word", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"STUDY: {args.study}")
    print("=" * 70)

    # ---- Stage 0a: resolve input (local dir, or download from URL) --------
    if args.fetch_reference:
        source_type, location = "url", reader.REFERENCE_STUDY_URL
    elif args.input_url:
        source_type, location = "url", args.input_url
    else:
        source_type, location = "local", args.input_dir

    print(f"\n[0a/5] Resolving input ({source_type}: {location}) ...")
    try:
        local_dir = reader.resolve_input(source_type, location, outdir)
    except Exception as e:
        sys.exit(f"Could not resolve input: {e}")

    # ---- Stage 0b: CDISC SDTM compliance gate (hard stop on error) --------
    print("\n[0b/5] Checking CDISC SDTM compliance against the study's define.xml ...")
    report = reader.check_compliance(local_dir)
    (outdir / "compliance_report.json").write_text(json.dumps(report, indent=2))
    report_text = reader.compliance_report_text(report)
    (outdir / "compliance_report.txt").write_text(report_text)
    print(report_text)

    provenance = {
        "study": args.study,
        "source_type": source_type,
        "source_location": location,
        "resolved_local_path": str(local_dir),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "compliant": report["compliant"],
        "n_domains_declared": report["n_domains_declared"],
        "n_errors": report["n_errors"],
        "n_warnings": report["n_warnings"],
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    if not report["compliant"]:
        print(f"\n{reader.VALIDATION_ERROR}", file=sys.stderr)
        print(f"See {outdir / 'compliance_report.txt'} for the full list of "
              f"deviations.", file=sys.stderr)
        sys.exit(2)

    # ---- Stage 1: SDTM -> CSV ----------------------------------------------
    print("\n[1/5] Converting SDTM to CSV ...")
    reader.convert(local_dir, outdir)

    # ---- Stage 2: features + split -----------------------------------------
    print("\n[2/5] Feature engineering + 70/30 split ...")
    _run_script("build_features_cdiscpilot01.py", [
        "--indir", str(outdir), "--outdir", str(outdir / "features"),
        "--study", args.study,
        "--target-mode", args.target_mode, "--landmark-day", str(args.landmark_day),
        "--death", args.death, "--qs-endpoints", args.qs_endpoints,
        "--seed", str(args.seed)])

    # ---- Stage 3: model + evaluate + explain -------------------------------
    print("\n[3/5] Modeling + evaluation + explainability ...")
    _run_script("model_cdiscpilot01.py", [
        "--indir", str(outdir / "features"), "--outdir", str(outdir / "model"),
        "--seed", str(args.seed)])

    # ---- Stage 4: survival / time-to-event ---------------------------------
    print("\n[4/5] Survival analysis (Kaplan-Meier, log-rank, Cox, competing risks) ...")
    _run_script("survival_cdiscpilot01.py", [
        "--indir", str(outdir / "features"),
        "--subjects", str(outdir / "subjects.csv"),
        "--outdir", str(outdir / "survival")])

    # ---- Stage 5: deliver results ------------------------------------------
    print("\n[5/5] Assembling results ...")
    if args.report in ("word", "both"):
        build_word_report(outdir, args.study)
    if args.report in ("files", "both"):
        print(f"File artifacts under {outdir}/ (csv/, features/, model/, survival/, "
              f"compliance_report.*, provenance.json).")

    print("\nDone.")


if __name__ == "__main__":
    main()
