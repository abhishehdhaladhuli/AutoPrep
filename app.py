"""AutoPrep: leakage-safe data preparation for ML.   Run with:  streamlit run app.py

Upload a CSV/Excel file, optionally pick a target column, and get
  * with a target:     X_train, X_test, y_train, y_test
  * without a target:  X_train, X_test
Everything that is learned from the data (fill values, scaling, categories, outlier limits,
which columns to drop) is fitted on the training rows only and then applied to the test rows.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from etl import (
    ETLConfig, detect_issues, find_id_like_columns, load_data, missing_report, schema_summary,
)
from model_prep import (
    PrepConfig, build_split_report, bundle_to_bytes, describe_target, learned_parameters,
    run_split_pipeline,
)
from sample_data import make_sample_data

logger = logging.getLogger("etl.app")

# One source of truth: the limit set in .streamlit/config.toml (Streamlit enforces it).
MAX_MB = int(st.get_option("server.maxUploadSize") or 200)

# ---------------------------------------------------------------------------
# Presets: every option in one dict so the UI, the configs and the docs stay in sync
# ---------------------------------------------------------------------------
RECOMMENDED = dict(
    standardize_names=True, strip_text=True, placeholders=True, fix_numeric=True, lowercase=True,
    clean_target=False, keep_unseen=True,
    drop_dups=True, row_missing_pct=100,
    drop_dup_cols=True, drop_const=True, col_missing_pct=50, drop_id=True,
    drop_corr=False, corr_thr=0.95,
    num_impute="median", cat_impute="most_frequent", flags=False,
    clip=False, clip_q=(1, 99), fix_skew=False, scaler="standard",
    encoder="onehot", max_cats=20, encode_target=True,
    shuffle=True, stratify="auto", seed=42,
)
MAXIMUM = {**RECOMMENDED, "drop_corr": True, "flags": True, "clip": True, "fix_skew": True}
PRESET_HELP = {
    "Recommended": "Safe defaults: tidy text, remove junk columns, fill gaps with the train median / "
                   "most frequent value, scale numbers, one-hot encode categories.",
    "Maximum": "Everything in Recommended, plus outlier clipping, skew correction, "
               "near-duplicate numeric column removal and missing-value flags.",
    "Custom": "Choose every option yourself.",
}

REASONS = {
    "duplicate": "Same values as an earlier column",
    "high_missing": "Too many missing values in the training rows",
    "constant": "Only one value in the training rows",
    "id_like": "Identifier: unique per row, nothing to learn from it",
    "correlated": "Nearly identical to an earlier numeric column",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&display=swap');
:root {--ap-train:#0F766E; --ap-test:#A8327A;}
.block-container {padding-top: 2.4rem; max-width: 1180px;}
[data-testid="stMarkdownContainer"] h1.ap-title {
  font-family:'Bricolage Grotesque','Source Sans','Source Sans Pro',sans-serif; font-weight:700;
  font-size:2.7rem; letter-spacing:-.02em; line-height:1.05; margin:0 0 .55rem 0; padding:0;}
.ap-sub {font-size:1.1rem; line-height:1.5; max-width:44rem; margin:0 0 1.2rem 0; opacity:.85;}
.section-title {font-family:'Bricolage Grotesque','Source Sans','Source Sans Pro',sans-serif;
                font-weight:600; font-size:1.3rem;
                margin:1.9rem 0 .6rem 0;}
.ap-flow {display:grid; grid-template-columns:repeat(4,1fr); gap:24px; margin:1.5rem 0 .7rem 0;}
@media (max-width:1150px) {.ap-flow {grid-template-columns:repeat(2,1fr);}}
@media (max-width:600px) {.ap-flow {grid-template-columns:1fr;}}
[data-testid="stMetricValue"] {font-size:1.7rem;}
.ap-bar {display:flex; gap:2px; height:12px; margin-bottom:12px;}
.ap-bar i {display:block; border-radius:3px; min-width:4px;}
.ap-all {background:rgba(128,128,128,.55);}
.ap-train {background:var(--ap-train);}
.ap-test {background:var(--ap-test);}
.ap-off {background:repeating-linear-gradient(135deg, rgba(128,128,128,.6) 0 2px, transparent 2px 5px);
         outline:1px solid rgba(128,128,128,.45); outline-offset:-1px;}
.ap-stage h4 {font-family:'Bricolage Grotesque','Source Sans','Source Sans Pro',sans-serif;
              font-weight:600; font-size:1.1rem;
              margin:0 0 .3rem 0; padding:0;}
.ap-stage p {margin:0; font-size:.92rem; line-height:1.45; opacity:.8;}
.ap-stage p.ap-meta {opacity:1; font-weight:600; margin-top:.5rem; font-variant-numeric:tabular-nums;}
.ap-legend {display:flex; flex-wrap:wrap; gap:4px 20px; font-size:.85rem; opacity:.8; margin-bottom:.4rem;}
.ap-legend i {display:inline-block; width:16px; height:10px; border-radius:2px; margin-right:7px;
              vertical-align:-1px;}
</style>
"""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading file...", max_entries=3)
def cached_load(data: bytes, name: str) -> pd.DataFrame:
    return load_data(io.BytesIO(data), name)


@st.cache_data(show_spinner="Scanning for data issues...", max_entries=3)
def cached_issues(file_id: str, _df: pd.DataFrame) -> list[dict]:
    issues = detect_issues(_df)
    # detect_issues looks at the raw file, where duplicate rows repeat an ID and hide it. The
    # pipeline removes duplicates first, so look for ID-like columns the same way here.
    if not any(i["issue"].startswith("ID-like") for i in issues):
        ids = find_id_like_columns(_df.drop_duplicates())
        if ids:
            shown = ", ".join(map(str, ids[:6])) + (f" (+{len(ids) - 6} more)" if len(ids) > 6 else "")
            issues.append({"issue": "ID-like columns (unique per row)", "columns": shown,
                           "affected": f"{len(ids)} column" + ("" if len(ids) == 1 else "s"),
                           "fixed_by": "Dropped (decided on train)"})
    return issues


@st.cache_data
def sample_bytes() -> bytes:
    return make_sample_data().to_csv(index=False).encode("utf-8")


def csv_bytes(obj: pd.DataFrame | pd.Series) -> bytes:
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
    return frame.to_csv(index=False).encode("utf-8")


def json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, indent=2, default=str).encode("utf-8")


def usage_snippet(has_target: bool, classification: bool) -> str:
    lines = ["import pandas as pd", "",
             'X_train = pd.read_csv("X_train.csv")', 'X_test = pd.read_csv("X_test.csv")']
    if has_target:
        model = "RandomForestClassifier" if classification else "RandomForestRegressor"
        lines += ['y_train = pd.read_csv("y_train.csv").squeeze("columns")',
                  'y_test = pd.read_csv("y_test.csv").squeeze("columns")', "",
                  f"from sklearn.ensemble import {model}",
                  f"model = {model}(random_state=0).fit(X_train, y_train)",
                  'print("test score:", model.score(X_test, y_test))']
    lines += ["", "# Prepare brand-new raw data with the SAME fitted steps (needs model_prep.py)",
              "import joblib", "from model_prep import apply_to_new_data",
              'bundle = joblib.load("preprocessor.joblib")   # only load files you created yourself',
              'X_new = apply_to_new_data(pd.read_csv("new_raw_data.csv"), bundle)']
    return "\n".join(lines)


def build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def package_run(df: pd.DataFrame, name: str, file_id: str, etl_cfg: ETLConfig,
                prep_cfg: PrepConfig, cfg_key: str) -> dict:
    """Run the pipeline and prepare every download once (so button clicks never recompute)."""
    t0 = time.perf_counter()
    res = run_split_pipeline(df, etl_cfg, prep_cfg)
    report = json_bytes(build_split_report(name, file_id, df.shape, etl_cfg, prep_cfg, res))
    bundle = bundle_to_bytes(res.bundle)
    has_target = res.y_train is not None
    classification = res.bundle.target_classes is not None or (
        has_target and describe_target(res.y_train).startswith("classification"))
    files = {"X_train.csv": csv_bytes(res.X_train), "X_test.csv": csv_bytes(res.X_test)}
    if has_target:
        files["y_train.csv"] = csv_bytes(res.y_train)
        files["y_test.csv"] = csv_bytes(res.y_test)
    snippet = usage_snippet(has_target, classification)
    zip_files = {**files, "report.json": report, "preprocessor.joblib": bundle,
                 "how_to_use.py": snippet.encode("utf-8")}
    return {
        "file_id": file_id, "cfg_key": cfg_key, "stem": Path(name).stem, "result": res,
        "seconds": time.perf_counter() - t0, "files": files, "report": report, "bundle": bundle,
        "zip": build_zip(zip_files), "learned": learned_parameters(res.bundle),
        "snippet": snippet, "input_shape": df.shape,
    }


# ---------------------------------------------------------------------------
# Options -> configs
# ---------------------------------------------------------------------------
def render_custom_options() -> dict:
    o = dict(RECOMMENDED)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Clean every row**")
        o["standardize_names"] = st.checkbox("Standardise column names", True, key="o_std")
        o["strip_text"] = st.checkbox("Trim extra spaces", True, key="o_strip")
        o["placeholders"] = st.checkbox("NA, ?, -, null, blank -> missing", True, key="o_ph")
        o["fix_numeric"] = st.checkbox('Text numbers ("$1,234") -> numbers', True, key="o_num")
        o["lowercase"] = st.checkbox("Lowercase text (merges Male / male)", True, key="o_low")
        o["clean_target"] = st.checkbox("Also clean the target labels (trim / lowercase / placeholders)",
                                        False, key="o_cleantgt",
                                        help="Off: the target column is left exactly as uploaded.")
        o["drop_dups"] = st.checkbox("Remove duplicate rows", True, key="o_dups")
        o["row_missing_pct"] = st.slider("Drop rows missing more than (%)", 10, 100, 100,
                                         key="o_rowmiss", help="100 = off")
    with c2:
        st.markdown("**Drop columns (decided on train)**")
        o["drop_dup_cols"] = st.checkbox("Duplicate columns", True, key="o_dupcols")
        o["drop_const"] = st.checkbox("Constant columns", True, key="o_const")
        o["col_missing_pct"] = st.slider("Columns missing more than (%)", 10, 100, 50, key="o_colmiss")
        o["drop_id"] = st.checkbox("ID-like columns", True, key="o_id",
                                   help="Unique text columns, and unique integer columns named like an ID.")
        o["drop_corr"] = st.checkbox("Highly correlated numeric columns", False, key="o_corr")
        if o["drop_corr"]:
            o["corr_thr"] = st.slider("Correlation threshold", 0.5, 1.0, 0.95, 0.01, key="o_corrthr")
        st.markdown("**Split**")
        o["shuffle"] = st.checkbox("Shuffle rows", True, key="o_shuf",
                                   help="Turn off for time-ordered data: the last rows become the test set.")
        o["stratify"] = st.selectbox("Stratify by target", ["auto", "yes", "no"], key="o_strat")
        o["seed"] = st.number_input("Random seed", min_value=0, value=42, step=1, key="o_seed")
    with c3:
        st.markdown("**Transform (fitted on train)**")
        o["num_impute"] = st.selectbox("Numeric missing values", ["median", "mean", "constant", "none"],
                                       key="o_nimp", help="constant fills with 0; none leaves them.")
        o["cat_impute"] = st.selectbox("Categorical missing values",
                                       ["most_frequent", "constant", "none"], key="o_cimp",
                                       help="constant fills with 'Unknown'.")
        o["flags"] = st.checkbox("Add <column>_was_missing flags", False, key="o_flags")
        o["clip"] = st.checkbox("Clip outliers to train percentiles", False, key="o_clip")
        if o["clip"]:
            o["clip_q"] = st.slider("Percentiles", 0, 100, (1, 99), key="o_clipq")
        o["fix_skew"] = st.checkbox("Fix skewed numeric columns", False, key="o_skew",
                                    help="Yeo-Johnson on columns with |skew| > 1 (measured on train).")
        o["scaler"] = st.selectbox("Scaling", ["standard", "minmax", "robust", "none"], key="o_scale")
        o["encoder"] = st.selectbox("Categorical encoding", ["onehot", "ordinal", "none"], key="o_enc")
        if o["encoder"] == "onehot":
            o["max_cats"] = st.number_input("One-hot: max categories per column", 2, 500, 20, key="o_maxcat",
                                            help="Rarer categories share one 'infrequent' column.")
        o["encode_target"] = st.checkbox("Encode text target as 0..k-1", True, key="o_enct")
        o["keep_unseen"] = st.checkbox("Keep test rows with a class unseen in train (label -1)", True,
                                        key="o_unseen",
                                        help="Off: those rows are removed from the test set.")
    return o


def build_configs(o: dict, target: str | None, drop_cols: list[str], test_pct: int):
    etl_cfg = ETLConfig(
        drop_columns=tuple(drop_cols), target_column=target,
        standardize_names=o["standardize_names"], strip_text=o["strip_text"],
        placeholders_as_missing=o["placeholders"], fix_numeric_text=o["fix_numeric"],
        lowercase_text=o["lowercase"], clean_target_text=o["clean_target"],
        drop_duplicates=o["drop_dups"],
        drop_duplicate_columns=o["drop_dup_cols"], drop_constant=o["drop_const"],
        missing_col_threshold=o["col_missing_pct"] / 100,
        missing_row_threshold=o["row_missing_pct"] / 100,
    )
    prep_cfg = PrepConfig(
        test_size=test_pct / 100, random_state=int(o["seed"]), shuffle=o["shuffle"],
        stratify=o["stratify"], num_impute=o["num_impute"], cat_impute=o["cat_impute"],
        add_missing_flags=o["flags"], clip_outliers=o["clip"],
        clip_quantiles=(o["clip_q"][0] / 100, o["clip_q"][1] / 100), fix_skew=o["fix_skew"],
        scaler=o["scaler"], encoder=o["encoder"], max_categories=int(o["max_cats"]),
        drop_id_like=o["drop_id"], drop_correlated=o["drop_corr"], corr_threshold=o["corr_thr"],
        encode_target=o["encode_target"],
        unseen_target_classes="keep" if o["keep_unseen"] else "drop",
    )
    return etl_cfg, prep_cfg


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def hero() -> None:
    st.markdown(
        '<h1 class="ap-title">AutoPrep</h1>'
        '<p class="ap-sub">Upload a raw CSV or Excel file and get cleaned, model-ready X_train, '
        "X_test, y_train and y_test. Fill values, scaling and encodings are learned from the "
        "training rows only.</p>", unsafe_allow_html=True)


def _bar(*segments: tuple[str, float]) -> str:
    inner = "".join(f'<i class="{cls}" style="flex:{max(w, 0.02):.4f}"></i>' for cls, w in segments)
    return f'<div class="ap-bar" aria-hidden="true">{inner}</div>'


def flow_strip(counts: dict | None = None) -> None:
    """The four stages, with a row bar showing which rows each stage touches.
    Pass `counts` after a run to show the real numbers."""
    tr, te = (counts["n_train"], counts["n_test"]) if counts else (80, 20)
    total = max(tr + te, 1)
    wtr, wte = tr / total, te / total
    meta = {}
    if counts:
        meta = {
            1: f"{counts['rows_in']:,} rows in, {counts['rows_clean']:,} kept",
            2: f"{tr:,} train, {te:,} test",
            3: f"Fitted on {tr:,} rows, {counts['n_dropped']} columns dropped",
            4: f"{counts['n_features']} model features",
        }
    stages = [
        ("Clean every row",
         "Tidy text, turn NA and ? into missing values, convert $1,234 text to numbers, "
         "remove duplicates.", _bar(("ap-all", 1))),
        ("Split",
         "Hold out the test rows first. Seeded, and stratified when the target is a class.",
         _bar(("ap-train", wtr), ("ap-test", wte))),
        ("Learn from train",
         "Fill values, scaling, categories, outlier limits and dropped columns come from "
         "the training rows only.", _bar(("ap-train", wtr), ("ap-off", wte))),
        ("Apply to both",
         "The same fitted steps transform train and test, and are saved for new data.",
         _bar(("ap-train", wtr), ("ap-test", wte))),
    ]
    html = "".join(
        f'<div class="ap-stage">{bar}<h4>{i}. {title}</h4><p>{text}</p>'
        + (f'<p class="ap-meta">{meta[i]}</p>' if i in meta else "") + "</div>"
        for i, (title, text, bar) in enumerate(stages, start=1))
    st.markdown(
        f'<div class="ap-flow">{html}</div>'
        '<div class="ap-legend"><span><i class="ap-train"></i>Train rows</span>'
        '<span><i class="ap-test"></i>Test rows</span>'
        '<span><i class="ap-off"></i>Test rows, not used for learning</span></div>',
        unsafe_allow_html=True)


def empty_state() -> None:
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    if c1.button("Try the sample dataset", type="primary", icon=":material/science:"):
        st.session_state["use_sample"] = True
        st.rerun()
    c2.caption(f"or upload your own CSV or Excel file in the sidebar (up to {MAX_MB} MB). "
               "The sample is a messy customer-churn table.")

    flow_strip()

    st.markdown('<div class="section-title">What gets automated</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown(
        "**Cleaning, applied to every row**\n"
        "- trim spaces, merge `Male` / `male`\n"
        "- turn `NA`, `?`, `-` and blanks into real missing values\n"
        "- convert `\"$1,234\"` text into numbers (zip codes like `02134` are kept)\n"
        "- remove duplicate rows and rows without a target")
    b.markdown(
        "**Learning, fitted on the training rows only**\n"
        "- drop constant, duplicate, ID-like, mostly-empty and highly correlated columns\n"
        "- fill missing values, clip outliers, fix skew, scale numbers\n"
        "- encode categories and the target, flag imbalanced classes\n"
        "- save everything in a preprocessor for new data")
    st.caption("Left to you on purpose: feature engineering, which depends on your domain, "
               "and choosing and tuning the model.")


def render_overview(df: pd.DataFrame, issues: list[dict], expand_explore: bool) -> None:
    st.markdown('<div class="section-title">Your data</div>', unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Rows", f"{df.shape[0]:,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c[4].metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:,.1f} MB")

    st.markdown('<div class="section-title">Issues found and how they are handled</div>',
                unsafe_allow_html=True)
    if issues:
        st.dataframe(pd.DataFrame(issues).rename(columns={
            "issue": "Issue", "columns": "Where", "affected": "Affected", "fixed_by": "Handled by"}),
            hide_index=True)
    else:
        st.success("No common data problems detected.")

    with st.expander("Explore the raw data", expanded=expand_explore):
        t1, t2, t3, t4 = st.tabs(["Preview", "Schema", "Missing values", "Statistics"])
        t1.dataframe(df.head(100))
        t2.dataframe(schema_summary(df))
        t3.dataframe(missing_report(df), column_config={"missing_%": st.column_config.ProgressColumn(
            "missing_%", min_value=0, max_value=100, format="%.1f")})
        t4.dataframe(df.describe(include="all").astype(str))


def learned_table(learned: dict) -> pd.DataFrame:
    cols: dict[str, dict] = {}
    names = {"numeric_fill_values": "fill value", "scaler_mean": "scaler mean",
             "scaler_scale": "scaler scale", "scaler_min": "scaler min", "scaler_max": "scaler max",
             "scaler_median": "scaler median", "skew_lambdas": "skew lambda"}
    for key, label in names.items():
        if key in learned:
            cols[label] = learned[key]
    if "clip_bounds" in learned:
        cols["clip low"] = {c: v[0] for c, v in learned["clip_bounds"].items()}
        cols["clip high"] = {c: v[1] for c, v in learned["clip_bounds"].items()}
    return pd.DataFrame(cols)


def render_results(run: dict) -> None:
    res = run["result"]
    b = res.bundle
    has_target = res.y_train is not None
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    n_dropped = sum(len(v) for v in b.dropped.values())
    flow_strip({"rows_in": run["input_shape"][0], "rows_clean": res.split_info["rows_total"],
                "n_train": len(res.X_train), "n_test": len(res.X_test),
                "n_dropped": n_dropped, "n_features": res.X_train.shape[1]})
    st.caption(f"Finished in {run['seconds']:.1f}s. Fill values, scaling and categories were "
               f"learned from the {len(res.X_train):,} training rows only.")
    m = st.columns(5)
    m[0].metric("X_train", f"{len(res.X_train):,} × {res.X_train.shape[1]}")
    m[1].metric("X_test", f"{len(res.X_test):,} × {res.X_test.shape[1]}")
    if has_target:
        kind = describe_target(res.y_train)                     # "classification (3 classes)" / "regression"
        target_value = kind[kind.index("(") + 1:-1] if "(" in kind else kind
    else:
        target_value = "none"
    m[2].metric("Target", target_value)
    m[3].metric("Columns dropped", n_dropped)
    m[4].metric("Missing cells left", f"{int(res.X_train.isna().sum().sum() + res.X_test.isna().sum().sum()):,}")

    d1, d2 = st.columns([2, 1])
    d1.download_button("Download everything (ZIP)", run["zip"], f"autoprep_{run['stem']}.zip",
                       "application/zip", type="primary", icon=":material/download:")
    with d2.popover("Individual files", icon=":material/folder_open:"):
        for fname, content in run["files"].items():
            st.download_button(fname, content, fname, "text/csv", key=f"dl_{fname}")
        st.download_button("report.json", run["report"], f"report_{run['stem']}.json",
                           "application/json", key="dl_report")
        st.download_button("preprocessor.joblib", run["bundle"], f"preprocessor_{run['stem']}.joblib",
                           "application/octet-stream", key="dl_bundle",
                           help="Only load .joblib files you created yourself.")

    if res.warnings:
        with st.expander(f"Checks and notes ({len(res.warnings)})", expanded=True):
            for w in res.warnings:
                st.warning(w)

    labels = ["X_train", "X_test"] + (["y_train / y_test"] if has_target else []) + [
        "Learned from train", "Dropped columns", "Processing log", "Use in code"]
    tabs = st.tabs(labels)
    it = iter(tabs)
    t = next(it)
    t.dataframe(res.X_train.head(100))
    t.caption(f"{res.X_train.shape[0]:,} rows × {res.X_train.shape[1]} columns")
    t = next(it)
    t.dataframe(res.X_test.head(100))
    t.caption(f"{res.X_test.shape[0]:,} rows × {res.X_test.shape[1]} columns")
    if has_target:
        t = next(it)
        left, right = t.columns(2)
        left.caption("y_train")
        left.dataframe(res.y_train.head(100).to_frame())
        right.caption("y_test")
        right.dataframe(res.y_test.head(100).to_frame())
        info = res.split_info
        if "target_distribution_train" in info:
            t.caption("Share of each class (stratified: " + info["stratified"] + ")")
            t.bar_chart(pd.DataFrame({"train": info["target_distribution_train"],
                                      "test": info["target_distribution_test"]}))
            if b.target_classes:
                t.caption("Encoding: " + ", ".join(f"{c} = {i}" for i, c in enumerate(b.target_classes)))
        else:
            t.dataframe(pd.DataFrame({"y_train": res.y_train.describe(),
                                      "y_test": res.y_test.describe()}))
    t = next(it)
    t.caption("Values computed from the training rows, then applied unchanged to the test rows.")
    table = learned_table(run["learned"])
    if not table.empty:
        t.dataframe(table)
    if run["learned"].get("skew_corrected_columns"):
        t.caption("Skew corrected: " + ", ".join(run["learned"]["skew_corrected_columns"]))
    extra = {k: v for k, v in run["learned"].items()
             if k in ("categorical_fill_values", "categories_seen_in_train")}
    if extra:
        with t.expander("Categorical fill values and categories"):
            st.json(extra, expanded=False)
    t = next(it)
    rows = [{"column": c, "reason": REASONS[k]} for k, cols in b.dropped.items() for c in cols]
    if rows:
        t.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        t.write("No columns were dropped by the train-fitted rules.")
    t = next(it)
    t.dataframe(pd.DataFrame(res.steps), hide_index=True)
    t = next(it)
    t.code(run["snippet"], language="python")


def feature_list() -> None:
    with st.expander("What does this preset do?"):
        st.markdown(
            "1. Clean every row: trim spaces, lowercase text, placeholders to missing, text numbers to "
            "numbers, remove duplicate rows and rows without a target.\n"
            "2. Split into train and test.\n"
            "3. Learn from train only: drop constant, duplicate, ID-like and mostly-empty columns.\n"
            "4. Fit and apply: fill missing values, scale numbers, encode categories (and the target).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="AutoPrep: ML data preparation", page_icon=":material/auto_fix_high:",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # ---------------- sidebar: data ----------------
    with st.sidebar:
        st.markdown("## AutoPrep")
        st.caption("Data preparation that learns from the training rows only")
        st.markdown("**1 · Data**")
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"],
                                    label_visibility="collapsed")
        c1, c2 = st.columns(2)
        if c1.button("Sample", icon=":material/science:", help="Load a messy customer-churn dataset"):
            st.session_state["use_sample"] = True
        if st.session_state.get("use_sample") and c2.button("Clear", icon=":material/close:"):
            st.session_state.pop("use_sample", None)
            st.session_state.pop("run", None)
            st.rerun()

    if uploaded is not None:
        if uploaded.size > MAX_MB * 1024 * 1024:
            hero()
            st.error(f"File exceeds {MAX_MB} MB.")
            st.stop()
        data, name = uploaded.getvalue(), uploaded.name
    elif st.session_state.get("use_sample"):
        data, name = sample_bytes(), "sample_churn.csv"
    else:
        hero()
        empty_state()
        st.stop()

    file_id = hashlib.sha256(data).hexdigest()
    try:
        df = cached_load(data, name)
    except Exception as e:
        logger.exception("Load failed")
        hero()
        st.error(f"Could not read file: {e}")
        st.stop()

    # ---------------- sidebar: settings ----------------
    with st.sidebar:
        st.markdown("**2 · Target**")
        target = st.selectbox("Target column (optional)", [None, *df.columns],
                              format_func=lambda c: "No target (X only)" if c is None else c,
                              label_visibility="collapsed",
                              help="The column you want to predict. It is never cleaned like a feature.")
        if target:
            st.caption(f"Detected task: {describe_target(df[target].dropna())}")
        drop_cols = st.multiselect("Columns to ignore", [c for c in df.columns if c != target],
                                   help="Removed before anything else.")
        st.markdown("**3 · Test set size**")
        test_pct = st.slider("Test set size", 5, 50, 20, format="%d%%", label_visibility="collapsed")
        st.markdown("**4 · Cleaning level**")
        preset = st.radio("Cleaning level", ["Recommended", "Maximum", "Custom"],
                          label_visibility="collapsed")
        st.caption(PRESET_HELP[preset])
        run_slot = st.container()

    # ---------------- main ----------------
    hero()
    st.caption(f"File: **{name}**")

    custom = None
    if preset == "Custom":
        with st.expander("Custom options", expanded=True):
            custom = render_custom_options()
    opts = custom or (MAXIMUM if preset == "Maximum" else RECOMMENDED)

    cfg_error = None
    etl_cfg = prep_cfg = None
    try:
        etl_cfg, prep_cfg = build_configs(opts, target, drop_cols, test_pct)
    except ValueError as e:
        cfg_error = str(e)
    cfg_key = hashlib.sha256(
        json.dumps([file_id, asdict(etl_cfg), asdict(prep_cfg)], sort_keys=True, default=str).encode()
    ).hexdigest() if cfg_error is None else ""

    with run_slot:
        if cfg_error:
            st.error(cfg_error)
        clicked = st.button("Run preprocessing", type="primary", icon=":material/play_arrow:",
                            disabled=cfg_error is not None)

    if clicked and cfg_error is None:
        try:
            with st.spinner("Cleaning, splitting and fitting on the training rows..."):
                st.session_state["run"] = package_run(df, name, file_id, etl_cfg, prep_cfg, cfg_key)
        except Exception as e:
            logger.exception("Pipeline failed")
            st.error(f"Pipeline failed: {e}")

    run = st.session_state.get("run")
    has_run = run is not None and run["file_id"] == file_id
    if has_run:
        if run["cfg_key"] != cfg_key:
            st.info("Settings changed since the last run. Click **Run preprocessing** to refresh the results.")
        render_results(run)

    render_overview(df, cached_issues(file_id, df), expand_explore=not has_run)
    if preset != "Custom":
        feature_list()

    st.caption("AutoPrep never does feature engineering: that depends on your domain. "
               "Downloaded .joblib files should only be loaded by you.")


if __name__ == "__main__":
    main()
