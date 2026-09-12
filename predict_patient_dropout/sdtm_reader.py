#!/usr/bin/env python3
"""
sdtm_reader.py  (v3)
====================
Stage 1 of the discontinuation-prediction pipeline.

Accepts ANY study organized to the CDISC SDTM standard -- a local folder, a
GitHub folder URL, a raw.githubusercontent.com folder URL, or a direct .zip
archive URL -- validates it against its OWN define.xml (not a hardcoded
dataset list), and converts every declared domain to a clean CSV, plus a
patient-keyed long file and a subject-level spine with the discontinuation
target.

Public functions (imported by run_pipeline.py):
  resolve_input(source_type, location, workdir) -> Path   # local dir or downloaded copy
  parse_define_xml(define_path)                 -> {domain_name_lower: meta}
  check_compliance(input_dir)                   -> report dict (see below)
  compliance_report_text(report)                -> human-readable string
  convert(input_dir, outdir, domains=None)      -> dict summary

COMPLIANCE SCOPE (please read)
  This checks that a study's own define.xml is well-formed, that every
  dataset it declares is present, that each dataset carries its declared
  mandatory variables, that the DOMAIN column (where present) matches the
  declared domain code, and a light datatype spot-check. It is a
  define.xml-conformance check plus a few universal SDTM invariants -- it is
  NOT a substitute for a full CDISC SDTM Implementation-Guide validator such
  as Pinnacle 21 / P21 Community, which checks controlled terminology,
  cross-domain consistency, and IG-version-specific structural rules this
  tool does not attempt.

Requires: pandas  (requests only for URL/zip sources; pyreadstat optional)
"""
from __future__ import annotations
import argparse
import io
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

# Convenience default for --fetch-reference (the public CDISC Pilot 01 study).
# It is not treated specially anywhere else in this file -- it is just one
# valid raw.githubusercontent.com folder URL among any the user could supply.
REFERENCE_STUDY_URL = ("https://raw.githubusercontent.com/phuse-org/phuse-scripts/"
                       "master/data/sdtm/cdiscpilot01")

NON_SUBJECT_DOMAINS = {"ta", "te", "ti", "ts", "tv", "relrec"}

VALIDATION_ERROR = ("User's input does not meet CDISC SDTM requirements "
                    "that results in termination of further analysis")


# ==========================================================================
# 1. INPUT RESOLUTION -- local dir, GitHub folder, raw.githubusercontent.com
#    folder, or a direct .zip archive. All resolve to a local directory that
#    the rest of this module (and the compliance checker) treat identically.
# ==========================================================================
def _require_requests():
    if requests is None:
        raise RuntimeError("The 'requests' package is required to fetch a "
                           "study from a URL (pip install requests).")


def _parse_github_tree(url: str):
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)",
                url.rstrip("/"))
    return m.groups() if m else None


def _parse_raw_base(url: str):
    m = re.match(r"https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)",
                url.rstrip("/"))
    return m.groups() if m else None


def _github_list_dir(owner, repo, branch, path):
    _require_requests()
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    r = requests.get(api, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_remote_sdtm(url: str, dest: Path) -> Path:
    """Download a study from a URL into `dest`, returning the local folder
    that actually holds the files (handles a single top-level zip subfolder).
    Supports: a direct .zip URL, a GitHub folder URL (.../tree/<branch>/<path>),
    or a raw.githubusercontent.com folder URL (.../<branch>/<path>)."""
    _require_requests()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if url.lower().split("?")[0].endswith(".zip"):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(dest)
        entries = [p for p in dest.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
        return dest

    gh = _parse_github_tree(url)
    if gh:
        owner, repo, branch, path = gh
        items = _github_list_dir(owner, repo, branch, path)
        n = 0
        for it in items:
            if it.get("type") == "file" and it.get("download_url"):
                rr = requests.get(it["download_url"], timeout=60)
                if rr.status_code == 200:
                    (dest / it["name"]).write_bytes(rr.content)
                    n += 1
        if n == 0:
            raise RuntimeError(f"No files found at {url} via the GitHub API.")
        return dest

    rb = _parse_raw_base(url)
    if rb:
        owner, repo, branch, path = rb
        base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        r = requests.get(f"{base}/define.xml", timeout=30)
        if r.status_code != 200:
            raise RuntimeError(
                f"Could not find define.xml at {base}/define.xml "
                f"(HTTP {r.status_code}). A raw.githubusercontent.com folder "
                f"URL must point directly at the folder containing define.xml.")
        (dest / "define.xml").write_bytes(r.content)
        expected = list_expected_datasets(dest / "define.xml")
        got = 0
        for dom in expected:
            for ext in (".xpt", ".csv"):
                rr = requests.get(f"{base}/{dom}{ext}", timeout=60)
                if rr.status_code == 200:
                    (dest / f"{dom}{ext}").write_bytes(rr.content)
                    got += 1
                    break
        if got == 0:
            raise RuntimeError(
                f"define.xml downloaded, but none of its {len(expected)} "
                f"declared dataset(s) could be found at {base}/<name>.xpt|csv.")
        return dest

    raise RuntimeError(
        "Unsupported --input-url. Provide one of:\n"
        "  - a GitHub folder URL:  https://github.com/<owner>/<repo>/tree/<branch>/<path>\n"
        "  - a raw.githubusercontent.com folder URL: "
        "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>\n"
        "  - a direct link to a .zip archive of the study")


def resolve_input(source_type: str, location: str, workdir: Path) -> Path:
    """source_type: 'local' or 'url'. Returns a local directory path."""
    if source_type == "local":
        p = Path(location)
        if not p.exists():
            raise RuntimeError(f"Input folder not found: {location}")
        return p
    if source_type == "url":
        return fetch_remote_sdtm(location, Path(workdir) / "_source_download")
    raise ValueError(f"Unknown source_type: {source_type}")


# ==========================================================================
# 2. define.xml parsing -- namespace-agnostic, works across define.xml
#    v1.0 / 2.0 / 2.1 since we match on local (namespace-stripped) tag names.
# ==========================================================================
def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _find_define(input_dir: Path) -> Path | None:
    input_dir = Path(input_dir)
    for name in ("define.xml", "Define.xml", "DEFINE.XML"):
        if (input_dir / name).exists():
            return input_dir / name
    for h in list(input_dir.glob("*.xml")) + list(input_dir.glob("*.XML")):
        if h.name.lower() == "define.xml":
            return h
    return None


def list_expected_datasets(define_path: Path) -> list[str]:
    """Datasets declared in define.xml (ItemGroupDef Name), lower-cased,
    order preserved, de-duplicated. Regex fallback kept fast/dependency-free
    for the common case; parse_define_xml() is used for deep compliance."""
    xml = Path(define_path).read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r'<ItemGroupDef\b[^>]*\bName="([A-Za-z0-9_]+)"', xml)
    return [n.lower() for n in dict.fromkeys(names)]


def parse_define_xml(define_path: Path) -> dict:
    """Full parse: {dataset_name_lower: {name, domain, structure, variables:
    [{name, datatype, mandatory}, ...]}}. Raises xml.etree.ElementTree.ParseError
    on malformed XML."""
    tree = ET.parse(define_path)
    root = tree.getroot()

    itemdefs = {}
    for el in root.iter():
        if _local(el.tag) == "ItemDef":
            oid = el.get("OID")
            if oid:
                itemdefs[oid] = {"name": el.get("Name"), "datatype": el.get("DataType")}

    groups = {}
    for el in root.iter():
        if _local(el.tag) == "ItemGroupDef":
            name = el.get("Name")
            if not name:
                continue
            domain = el.get("Domain") or name
            structure = el.get("Structure") or ""
            variables = []
            for child in el:
                if _local(child.tag) == "ItemRef":
                    info = itemdefs.get(child.get("ItemOID"))
                    if info and info.get("name"):
                        variables.append({
                            "name": info["name"],
                            "datatype": info.get("datatype"),
                            "mandatory": (child.get("Mandatory") == "Yes"),
                        })
            groups[name.lower()] = {"name": name, "domain": domain,
                                    "structure": structure, "variables": variables}
    return groups


# ==========================================================================
# 3. Reading raw domain files (shared by compliance checking and convert())
# ==========================================================================
def _resolve_path(input_dir: Path, name: str) -> Path | None:
    for cand in (f"{name}.xpt", f"{name.upper()}.xpt", f"{name}.csv",
                f"{name.upper()}.csv", f"{name}.XPT", f"{name}.CSV"):
        if (Path(input_dir) / cand).exists():
            return Path(input_dir) / cand
    return None


def _dataset_file_present(input_dir: Path, name: str) -> bool:
    return _resolve_path(input_dir, name) is not None


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
    else:  # XPORT
        try:
            import pyreadstat
            df, _ = pyreadstat.read_xport(str(path))
        except Exception:
            df = pd.read_sas(path, format="xport", encoding=None)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.decode("utf-8", "ignore").strip()
                                  if isinstance(v, (bytes, bytearray)) else v)
                df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    df.columns = [c.upper() for c in df.columns]
    return df


# ==========================================================================
# 4. Compliance checker
# ==========================================================================
@dataclass
class Finding:
    severity: str   # "error" | "warning"
    domain: str
    check: str
    detail: str


_ISO_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}(:\d{2})?)?)?)?$")


def check_compliance(input_dir: Path) -> dict:
    """Run the compliance checks described in this module's docstring and
    return a structured report: {compliant, define_xml, n_domains_declared,
    n_errors, n_warnings, findings: [...]}."""
    input_dir = Path(input_dir)
    findings: list[Finding] = []

    def err(domain, check, detail):
        findings.append(Finding("error", domain, check, detail))

    def warn(domain, check, detail):
        findings.append(Finding("warning", domain, check, detail))

    define_path = _find_define(input_dir)
    if define_path is None:
        err("(study)", "define.xml presence", "No define.xml found in the input folder.")
        return _finalize(findings, None, {})

    try:
        groups = parse_define_xml(define_path)
    except ET.ParseError as e:
        err("(study)", "define.xml well-formedness", f"define.xml is not valid XML: {e}")
        return _finalize(findings, define_path, {})

    if not groups:
        err("(study)", "define.xml content", "define.xml declares no ItemGroupDef datasets.")
        return _finalize(findings, define_path, {})

    for dom, meta in groups.items():
        path = _resolve_path(input_dir, dom)
        if path is None:
            err(dom.upper(), "dataset presence",
                f"Declared in define.xml but no {dom}.xpt/.csv file was found.")
            continue

        try:
            df = _read_one(path)
        except Exception as e:
            err(dom.upper(), "dataset readable", f"Could not read {path.name}: {e}")
            continue

        cols = set(df.columns)
        declared = {v["name"] for v in meta["variables"] if v["name"]}
        mandatory = {v["name"] for v in meta["variables"] if v["name"] and v["mandatory"]}
        missing_mandatory = mandatory - cols
        missing_optional = (declared - mandatory) - cols
        extra = cols - declared if declared else set()

        if missing_mandatory:
            err(dom.upper(), "mandatory variables",
                f"Missing mandatory variable(s) declared in define.xml: "
                f"{sorted(missing_mandatory)}")
        if missing_optional:
            warn(dom.upper(), "optional variables",
                f"Missing optional variable(s) declared in define.xml: "
                f"{sorted(missing_optional)}")
        if extra:
            warn(dom.upper(), "undeclared variables",
                f"Column(s) present in the file but not declared in define.xml: "
                f"{sorted(extra)}")

        # universal SDTM invariants
        if dom not in NON_SUBJECT_DOMAINS:
            for key in ("STUDYID", "USUBJID"):
                if key in declared and key not in cols:
                    err(dom.upper(), "key variable",
                        f"{key} is declared for this domain but missing from the file.")
        if "DOMAIN" in cols:
            bad = set(df["DOMAIN"].dropna().astype(str).str.upper().unique()) - \
                {str(meta["domain"]).upper()}
            if bad:
                err(dom.upper(), "DOMAIN value",
                    f"DOMAIN column contains {sorted(bad)} but this dataset is "
                    f"declared as {meta['domain']}.")

        # light datatype spot-check (numeric-typed variables that don't parse)
        for v in meta["variables"]:
            name, dtype = v["name"], (v["datatype"] or "").lower()
            if not name or name not in cols:
                continue
            if dtype in ("integer", "float"):
                raw = df[name]
                non_missing = raw.notna() & (raw.astype(str).str.strip() != "")
                if non_missing.sum() == 0:
                    continue
                parsed = pd.to_numeric(raw, errors="coerce")
                fail_rate = (parsed.isna() & non_missing).sum() / non_missing.sum()
                if fail_rate > 0.10:
                    warn(dom.upper(), "datatype",
                        f"{name} is declared {dtype} but {fail_rate:.0%} of its "
                        f"non-missing values do not parse as numbers.")
            elif dtype in ("date", "datetime", "partialdate", "partialdatetime"):
                raw = df[name].dropna().astype(str)
                raw = raw[raw.str.strip() != ""]
                if len(raw) == 0:
                    continue
                fail_rate = (~raw.str.match(_ISO_DATE_RE)).mean()
                if fail_rate > 0.10:
                    warn(dom.upper(), "datatype",
                        f"{name} is declared {dtype} but {fail_rate:.0%} of its "
                        f"non-missing values are not ISO 8601 (-partial) dates.")

    return _finalize(findings, define_path, groups)


def _finalize(findings: list[Finding], define_path, groups) -> dict:
    errors = [f for f in findings if f.severity == "error"]
    return {
        "compliant": len(errors) == 0,
        "define_xml": str(define_path) if define_path else None,
        "n_domains_declared": len(groups),
        "n_errors": len(errors),
        "n_warnings": len(findings) - len(errors),
        "findings": [asdict(f) for f in findings],
    }


def compliance_report_text(report: dict) -> str:
    lines = [
        "CDISC SDTM Compliance Report",
        "=" * 40,
        f"define.xml           : {report.get('define_xml') or 'NOT FOUND'}",
        f"Domains declared     : {report.get('n_domains_declared', 0)}",
        f"Result               : "
        f"{'COMPLIANT' if report.get('compliant') else 'NON-COMPLIANT'} "
        f"({report.get('n_errors', 0)} error(s), {report.get('n_warnings', 0)} warning(s))",
        "",
    ]
    if not report.get("findings"):
        lines.append("No deviations found.")
    for f in report.get("findings", []):
        lines.append(f"[{f['severity'].upper():7}] {f['domain']:10} {f['check']}: {f['detail']}")
    return "\n".join(lines)


# ==========================================================================
# 5. Subject spine + patient-long + CSV conversion (unchanged logic from v2,
#    minus the hardcoded reference-download path -- fetching now happens via
#    resolve_input()/fetch_remote_sdtm() before convert() is ever called).
# ==========================================================================
def _build_subjects(frames):
    dm, ds = frames.get("dm"), frames.get("ds")
    if dm is None:
        raise RuntimeError("DM domain is required.")
    s = dm.copy()
    rnd_ids = set()
    if ds is not None and {"USUBJID", "DSDECOD"}.issubset(ds.columns):
        rnd_ids = set(ds[ds["DSDECOD"].astype(str).str.upper() == "RANDOMIZED"]["USUBJID"])

    def real_arm(r):
        armcd = str(r.get("ARMCD", "")).upper(); arm = str(r.get("ARM", "")).upper()
        bad = {"SCRNFAIL", "SCREEN FAILURE", "NOTASSGN", "NOT ASSIGNED", "", "NAN", "NONE"}
        return armcd not in bad and "SCREEN FAIL" not in arm

    s["RANDOMIZED_ARM"] = s.apply(real_arm, axis=1)
    s["RANDOMIZED"] = s["USUBJID"].isin(rnd_ids) | s["RANDOMIZED_ARM"]
    s["DISPOSITION"] = pd.NA
    s["DISCONTINUED"] = pd.NA
    if ds is not None and {"USUBJID", "DSDECOD"}.issubset(ds.columns):
        de = ds.copy()
        if "DSCAT" in de.columns:
            de = de[de["DSCAT"].astype(str).str.upper() == "DISPOSITION EVENT"]
        sc = [c for c in ["USUBJID", "DSSTDTC", "DSSEQ"] if c in de.columns]
        de = de.sort_values(sc).groupby("USUBJID", as_index=False).last()
        m = dict(zip(de["USUBJID"], de["DSDECOD"].astype(str)))
        s["DISPOSITION"] = s["USUBJID"].map(m)
        s["DISCONTINUED"] = s["DISPOSITION"].map(
            lambda v: pd.NA if pd.isna(v) else (0 if str(v).upper() == "COMPLETED" else 1))
    lead = [c for c in ["USUBJID", "SUBJID", "SITEID", "ARM", "ARMCD", "ACTARM",
                        "ACTARMCD", "AGE", "AGEU", "SEX", "RACE", "ETHNIC",
                        "COUNTRY", "RFSTDTC", "RFENDTC", "RANDOMIZED",
                        "DISPOSITION", "DISCONTINUED"] if c in s.columns]
    return s[lead + [c for c in s.columns if c not in lead]]


def _build_patient_long(frames):
    parts = []
    for dom, df in frames.items():
        if dom in NON_SUBJECT_DOMAINS or "USUBJID" not in df.columns:
            continue
        t = df.copy(); t.insert(0, "DOMAIN_SRC", dom.upper()); parts.append(t)
    if not parts:
        return pd.DataFrame()
    lg = pd.concat(parts, ignore_index=True, sort=False)
    front = [c for c in ["USUBJID", "DOMAIN_SRC"] if c in lg.columns]
    return lg[front + [c for c in lg.columns if c not in front]] \
        .sort_values(["USUBJID", "DOMAIN_SRC"])


def convert(input_dir, outdir, domains=None):
    """Read a resolved local SDTM folder and write CSV outputs. `input_dir`
    must already be a local directory (resolve_input()/fetch_remote_sdtm()
    handle any remote source before this is called)."""
    input_dir, outdir = Path(input_dir), Path(outdir)
    csv_dir = outdir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    if domains is None:
        define_path = _find_define(input_dir)
        if define_path is None:
            raise RuntimeError("No define.xml found; run check_compliance() first.")
        domains = list_expected_datasets(define_path)

    frames = {}
    for dom in domains:
        path = _resolve_path(input_dir, dom)
        if path is None:
            print(f"  [warn] {dom}: no file found", file=sys.stderr)
            continue
        df = _read_one(path)
        frames[dom] = df
        df.to_csv(csv_dir / f"{dom.upper()}.csv", index=False)
        print(f"  -> {dom.upper():8} {df.shape[0]:6d} rows x {df.shape[1]:2d} cols")

    summary = {"n_domains": len(frames)}
    if "dm" in frames:
        subjects = _build_subjects(frames)
        subjects.to_csv(outdir / "subjects.csv", index=False)
        long_df = _build_patient_long(frames)
        long_df.to_csv(outdir / "patient_long.csv", index=False)
        rnd = subjects[subjects["RANDOMIZED"] == True]
        summary.update({
            "n_subjects": int(len(subjects)),
            "n_randomized": int(subjects["RANDOMIZED"].sum()),
            "n_patient_long_rows": int(len(long_df)),
        })
        print(f"\nSubjects: {summary['n_subjects']}  |  randomized: "
              f"{summary['n_randomized']}")
        if "DISPOSITION" in rnd.columns:
            print("Disposition (randomized):")
            print(rnd["DISPOSITION"].value_counts(dropna=False).to_string())
    return summary


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", help="local folder with SDTM files (+ define.xml)")
    src.add_argument("--input-url", help="GitHub folder URL, raw.githubusercontent.com "
                                         "folder URL, or a direct .zip archive URL")
    src.add_argument("--fetch-reference", action="store_true",
                     help="shorthand for the public CDISC Pilot 01 reference study")
    ap.add_argument("--outdir", default="cdiscpilot01_out_v2")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.fetch_reference:
        source_type, location = "url", REFERENCE_STUDY_URL
    elif args.input_url:
        source_type, location = "url", args.input_url
    else:
        source_type, location = "local", args.input_dir

    local_dir = resolve_input(source_type, location, outdir)
    report = check_compliance(local_dir)
    print(compliance_report_text(report))
    if not report["compliant"]:
        print(f"\n{VALIDATION_ERROR}", file=sys.stderr)
        sys.exit(2)
    convert(local_dir, outdir)


if __name__ == "__main__":
    main()
