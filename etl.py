"""ETL cleaning logic: profile, clean (ML-safe, stateless steps), report. No Streamlit here.

The steps in `run_pipeline` do not learn imputation values, scaling, encoding or outlier limits
from the data. The column-dropping rules (too many missing, constant, duplicate columns) are
structural but do look at the whole dataset, so when you need a strictly leakage-free train/test
workflow use `model_prep.run_split_pipeline`, which learns those rules on the training rows only.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("etl")

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
MAX_MB = 50   # keep modest for hosted demos (Streamlit Community Cloud has limited memory); raise locally
EXCEL_MAX_ROWS = 200_000  # building .xlsx is slow/memory-heavy beyond this; CSV has no limit

# Text values that almost always mean "missing" (case-insensitive, after trimming).
# "none" and "unknown" are deliberately NOT listed: they can be real categories.
PLACEHOLDERS = frozenset({"", "na", "n/a", "#n/a", "nan", "null", "nil", "-", "--", "?"})
_NUM_JUNK = re.compile(r"[,$€£₹\s]")
# Strings the file reader turns into missing values: pandas' defaults EXCEPT "None", which is a
# real category ("Internet service: None") and must not be silently turned into a missing value.
_FILE_NA_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
                   "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
_READ_KW = dict(keep_default_na=False, na_values=_FILE_NA_VALUES)


# =====================================================================
# ETL LOGIC (no Streamlit calls: unit-testable)
# =====================================================================
@dataclass(frozen=True)
class ETLConfig:
    drop_columns: tuple[str, ...] = ()        # names as they appear in the uploaded file
    target_column: str | None = None          # protected: never filled, converted or dropped
    standardize_names: bool = False           # "Customer Name" -> "customer_name"
    strip_text: bool = True                   # trim + collapse spaces, blank -> missing
    placeholders_as_missing: bool = False     # "NA", "N/A", "null", "?", "-" -> missing
    fix_numeric_text: bool = False            # "1,234" / "$50" stored as text -> numbers
    lowercase_text: bool = False              # "Male" / "male" -> "male"
    clean_target_text: bool = False           # also apply the text clean-up to the target labels
    drop_duplicates: bool = True
    drop_duplicate_columns: bool = False      # columns with identical content
    missing_col_threshold: float = 0.5        # drop columns with > this fraction missing
    missing_row_threshold: float = 1.0        # drop rows with > this fraction missing (1.0 = off)
    drop_constant: bool = True
    # Missing values: no statistic is learned from the data in any of these options.
    num_impute: Literal["none", "constant", "drop_rows"] = "none"   # constant = 0
    cat_impute: Literal["none", "constant", "drop_rows"] = "none"   # constant = "Unknown"

    def __post_init__(self):
        if not 0 < self.missing_col_threshold <= 1:
            raise ValueError("missing_col_threshold must be in (0, 1].")
        if not 0 < self.missing_row_threshold <= 1:
            raise ValueError("missing_row_threshold must be in (0, 1].")
        if self.num_impute not in ("none", "constant", "drop_rows"):
            raise ValueError("num_impute must be 'none', 'constant' or 'drop_rows'.")
        if self.cat_impute not in ("none", "constant", "drop_rows"):
            raise ValueError("cat_impute must be 'none', 'constant' or 'drop_rows'.")


@dataclass
class ETLResult:
    df: pd.DataFrame
    steps: list[dict]
    target: str | None = None                                   # target name in the output
    column_map: dict[str, str] = field(default_factory=dict)    # original -> output names


_LEADING_ZERO = re.compile(r"^\s*[+-]?0\d")


def _restore_leading_zeros(df: pd.DataFrame, data: bytes, encoding: str | None) -> pd.DataFrame:
    """read_csv and read_excel turn identifiers such as the zip code 02134 into the number 2134.

    Re-read only the numeric columns as text and keep the text version of any column where a
    value has a leading zero. This is the same rule the cleaning step uses for text numbers.
    """
    idx = [i for i, c in enumerate(df.columns)
           if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not idx:
        return df
    if encoding is None:    # Excel
        raw = pd.read_excel(io.BytesIO(data), usecols=idx, dtype=str, **_READ_KW)
    else:
        raw = pd.read_csv(io.BytesIO(data), usecols=idx, dtype=str, encoding=encoding, **_READ_KW)
    for i, raw_col in zip(idx, raw.columns):
        if raw[raw_col].dropna().str.match(_LEADING_ZERO).any():
            df[df.columns[i]] = raw[raw_col].to_numpy()
    return df


def load_data(file: BinaryIO, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Upload .csv or .xlsx.")
    if ext == ".csv":
        data = file.read()
        encoding = "utf-8"
        try:
            df = pd.read_csv(io.BytesIO(data), **_READ_KW)
        except UnicodeDecodeError:
            encoding = "latin-1"
            df = pd.read_csv(io.BytesIO(data), encoding=encoding, **_READ_KW)
        df = _restore_leading_zeros(df, data, encoding)
    else:
        data = file.read()
        df = pd.read_excel(io.BytesIO(data), **_READ_KW)  # first sheet only
        df = _restore_leading_zeros(df, data, None)
    if df.empty:
        raise ValueError("The file contains no data.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found; please rename them.")
    return df


# ---------- Profiling ----------
def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
        "example": df.apply(lambda s: str(s.dropna().iloc[0]) if s.notna().any() else ""),
    })


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percentage for every column, worst first."""
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    })
    rep.index.name = "column"
    return rep.sort_values(["missing_%", "missing"], ascending=False)


# ---------- Leakage checks ----------
def find_id_like_columns(df: pd.DataFrame, target: str | None = None,
                         min_rows: int = 20, ratio: float = 0.99) -> list[str]:
    """Integer/text columns that are unique for (almost) every row: IDs or free text."""
    n = len(df)
    if n < min_rows:
        return []
    found = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if (pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)):
            continue
        if s.nunique(dropna=True) / n >= ratio:
            found.append(c)
    return found


def find_target_leakage(df: pd.DataFrame, target: str | None, min_rows: int = 20,
                        corr_thr: float = 0.98, max_cols: int = 300) -> list[str]:
    """Columns that (almost) reproduce the target: likely leakage."""
    if not target or target not in df.columns:
        return []
    d = df.dropna(subset=[target])
    n = len(d)
    if n < min_rows:
        return []
    y = d[target]
    msgs = []
    for c in list(d.columns)[: max_cols + 1]:
        if c == target:
            continue
        s = d[c]
        if pd.api.types.is_numeric_dtype(y) and pd.api.types.is_numeric_dtype(s):
            try:
                with np.errstate(all="ignore"):     # constant columns give a 0/0 warning
                    corr = s.corr(y)
            except (TypeError, ValueError):
                corr = np.nan
            if pd.notna(corr) and abs(corr) >= corr_thr:
                msgs.append(f"'{c}' is almost identical to the target (correlation {corr:.3f})")
                continue
        k = s.nunique(dropna=True)
        if 2 <= k <= 0.5 * n:
            try:
                if d.groupby(c)[target].nunique().max() == 1:
                    msgs.append(f"'{c}' perfectly determines the target (possible leakage)")
            except TypeError:
                pass
    return msgs


def leakage_warnings(df: pd.DataFrame, target: str | None) -> list[str]:
    msgs = [
        f"'{c}' is unique for almost every row (ID or free text): consider dropping it"
        for c in find_id_like_columns(df, target)
    ]
    return msgs + find_target_leakage(df, target)


_DATE_LIKE = r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"


def detect_issues(df: pd.DataFrame) -> list[dict]:
    """Common data problems in the raw file and the step that deals with each one."""
    issues: list[dict] = []

    def add(issue: str, cols: list, count: int, unit: str, fix: str) -> None:
        shown = ", ".join(map(str, cols[:6])) + (f" (+{len(cols) - 6} more)" if len(cols) > 6 else "")
        affected = f"{int(count):,} {unit}" + ("" if count == 1 else "s")
        issues.append({"issue": issue, "columns": shown or "-", "affected": affected, "fixed_by": fix})

    n_dup = int(df.duplicated().sum())
    if n_dup:
        add("Duplicate rows", [], n_dup, "row", "Removed before the split")

    n_missing_cols = [c for c in df.columns if df[c].isna().any()]
    if n_missing_cols:
        add("Missing values", n_missing_cols, int(df.isna().sum().sum()), "cell",
            "Imputed with values learned on train")

    ph_cols, ph_total = [], 0
    ws_cols, ws_total = [], 0
    num_text_cols, mixed_case_cols, date_text_cols = [], [], []
    for c in _text_cols(df, None):
        s = df[c].astype("string")
        norm = s.str.strip().str.lower()
        ph = norm.isin(PLACEHOLDERS).fillna(False).to_numpy(dtype=bool)
        if ph.any():
            ph_cols.append(c)
            ph_total += int(ph.sum())
        ws = s.str.contains(r"^\s|\s$|\s{2,}", regex=True).fillna(False).to_numpy(dtype=bool)
        if ws.any():
            ws_cols.append(c)
            ws_total += int(ws.sum())
        clean_s = df[c].where(~ph)
        if clean_s.notna().any():
            if _numeric_from_text(clean_s) is not None:
                num_text_cols.append(c)
            elif (s.str.match(_DATE_LIKE).fillna(False).sum() >= 0.9 * max(int(s.notna().sum()), 1)
                  and s.notna().sum() > 0):
                date_text_cols.append(c)
            elif norm.nunique() < s.str.strip().nunique():
                mixed_case_cols.append(c)
    if ph_cols:
        add("Placeholder text meaning 'missing' (NA, ?, -, blank ...)", ph_cols, ph_total, "cell",
            "Converted to real missing values")
    if num_text_cols:
        add("Numbers stored as text ('$1,234')", num_text_cols, len(num_text_cols), "column",
            "Converted to numeric")
    if ws_cols:
        add("Extra spaces in text", ws_cols, ws_total, "cell", "Trimmed")
    if mixed_case_cols:
        add("Same category in different case ('Male' / 'male')", mixed_case_cols,
            len(mixed_case_cols), "column", "Lowercased")
    if date_text_cols:
        add("Date-like text (left as text)", date_text_cols, len(date_text_cols), "column",
            "Not touched: date features are domain-specific")

    inf_cols = [c for c in df.select_dtypes(include="number").columns
                if np.isinf(df[c].to_numpy(dtype=float, na_value=np.nan)).any()]
    if inf_cols:
        add("Infinite values", inf_cols, len(inf_cols), "column", "Converted to missing")

    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const:
        add("Constant columns (one value only)", const, len(const), "column", "Dropped (decided on train)")
    dup_cols = _duplicate_columns(df, None)
    if dup_cols:
        add("Duplicate columns", dup_cols, len(dup_cols), "column", "Dropped (decided on train)")
    frac = df.isna().mean()
    high = list(frac[frac > 0.5].index)
    if high:
        add("Columns with more than 50% missing", high, len(high), "column", "Dropped (decided on train)")
    ids = find_id_like_columns(df)
    if ids:
        add("ID-like columns (unique per row)", ids, len(ids), "column", "Dropped (decided on train)")

    skewed = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        if s.nunique() > 2 and len(s) > 2 and abs(s.skew()) > 1:
            skewed.append(c)
    if skewed:
        add("Strongly skewed numeric columns", skewed, len(skewed), "column", "Optional skew correction")
    return issues


# ---------- Cleaning helpers ----------
def standardize_column_names(cols: list) -> list[str]:
    """'Customer Name', 'customerID', '2nd Visit' -> customer_name, customer_id, col_2nd_visit."""
    out: list[str] = []
    used: set[str] = set()
    for c in cols:
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(c).strip())   # camelCase -> camel_Case
        s = re.sub(r"[\W_]+", "_", s).strip("_").lower()
        if not s:
            s = "column"
        if s[0].isdigit():
            s = f"col_{s}"
        base, k = s, 1
        while s in used:
            k += 1
            s = f"{base}_{k}"
        used.add(s)
        out.append(s)
    return out


def _clean_text_value(v, strip: bool, placeholders: bool, lower: bool):
    if not isinstance(v, str):
        return v
    if strip:
        v = " ".join(v.split())          # trim and collapse internal whitespace
    if placeholders and v.strip().lower() in PLACEHOLDERS:
        return np.nan
    if strip and v == "":
        return np.nan
    return v.lower() if lower else v


def _numeric_from_text(s: pd.Series) -> pd.Series | None:
    """Convert a text column to numbers only if EVERY non-null value parses cleanly.

    Handles thousands separators and currency symbols ("1,234", "$50"). Returns None (leave
    the column alone) when anything fails to parse, or when values have leading zeros such as
    "007" or "02134", which are identifiers whose zeros must not be lost.
    """
    if s.notna().sum() == 0:
        return None
    cleaned = s.astype(object).map(lambda v: _NUM_JUNK.sub("", v) if isinstance(v, str) else v)
    strs = cleaned[cleaned.map(lambda v: isinstance(v, str))]
    if len(strs) and strs.str.match(r"^[+-]?0\d").any():
        return None
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.notna().sum() != s.notna().sum() or np.isinf(parsed).any():
        return None
    return parsed


def _duplicate_columns(df: pd.DataFrame, protect: str | None) -> list[str]:
    """Columns whose content is identical to an earlier column (the first is kept)."""
    seen: dict = {}
    dups: list[str] = []
    for c in df.columns:
        if c == protect:
            continue
        s = df[c]
        try:
            key = (str(s.dtype), int(pd.util.hash_pandas_object(s, index=False).sum()))
        except TypeError:
            continue
        first = seen.get(key)
        if first is None:
            seen[key] = c
        elif df[first].equals(s):
            dups.append(c)
    return dups


def _text_cols(df: pd.DataFrame, target: str | None) -> list[str]:
    cols = df.select_dtypes(
        exclude=["number", "bool", "datetime", "datetimetz", "timedelta"]
    ).columns
    return [c for c in cols if c != target]


def _num_cols(df: pd.DataFrame, target: str | None) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if c != target]


# ---------- Pipeline ----------
def run_pipeline(df: pd.DataFrame, cfg: ETLConfig) -> ETLResult:
    out = df.copy()
    steps: list[dict] = []

    def log(step: str, detail: str) -> None:
        # Always call AFTER reassigning `out`, so the shape reported is current.
        steps.append({"step": step, "detail": detail, "rows": len(out), "cols": out.shape[1]})
        logger.info("%s | %s | shape=%s", step, detail, out.shape)

    log("Input", "loaded")

    # 0. Standard column names (user-chosen drop/target names are mapped across)
    col_map = {c: c for c in out.columns}
    tgt = cfg.target_column
    drop_req = list(cfg.drop_columns)
    if cfg.standardize_names:
        new = standardize_column_names(list(out.columns))
        col_map = dict(zip(out.columns, new))
        out.columns = new
        tgt = col_map.get(tgt, tgt) if tgt else None
        drop_req = [col_map.get(c, c) for c in drop_req]
        log("Standardise names", f"renamed {sum(a != b for a, b in col_map.items())} columns")

    # 1. Drop requested columns (target is protected)
    to_drop = [c for c in drop_req if c in out.columns and c != tgt]
    out = out.drop(columns=to_drop)
    log("Drop columns", f"{to_drop or 'none'}")

    # 2. Normalise: inf -> NaN (numeric only), category -> object
    num_all = out.select_dtypes("number").columns
    if len(num_all):
        out[num_all] = out[num_all].replace([np.inf, -np.inf], np.nan)
    for c in out.select_dtypes("category").columns:
        out[c] = out[c].astype(object)

    # 3. Text clean-up: trim/collapse spaces, blanks and placeholders -> missing, lowercase.
    #    The target is left untouched unless the user opts in with clean_target_text.
    if cfg.strip_text or cfg.placeholders_as_missing or cfg.lowercase_text:
        for c in _text_cols(out, None if cfg.clean_target_text else tgt):
            out[c] = out[c].map(
                lambda v: _clean_text_value(v, cfg.strip_text, cfg.placeholders_as_missing,
                                            cfg.lowercase_text)
            )
        opts = [n for n, on in (("trim/collapse spaces", cfg.strip_text),
                                ("placeholders -> missing", cfg.placeholders_as_missing),
                                ("lowercase", cfg.lowercase_text)) if on]
        log("Clean text", ", ".join(opts)
            + (" (target labels included)" if tgt and cfg.clean_target_text else ""))

    # 4. Numbers stored as text
    if cfg.fix_numeric_text:
        converted = []
        for c in _text_cols(out, tgt):
            new_col = _numeric_from_text(out[c])
            if new_col is not None:
                out[c] = new_col
                converted.append(c)
        log("Numbers stored as text", f"converted {converted or 'none'}")

    # 5. Duplicate rows
    if cfg.drop_duplicates:
        n = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        log("Duplicate rows", f"removed {n - len(out)} rows")

    # 6. Target must be present
    if tgt and tgt in out.columns:
        n = len(out)
        out = out.dropna(subset=[tgt]).reset_index(drop=True)
        log("Missing target", f"dropped {n - len(out)} rows with no target")

    # 7. Duplicate columns
    if cfg.drop_duplicate_columns:
        dup_cols = _duplicate_columns(out, tgt)
        out = out.drop(columns=dup_cols)
        log("Duplicate columns", f"dropped {dup_cols or 'none'}")

    # 8. High-missing columns, then high-missing rows
    frac = out.isna().mean()
    high = [c for c in frac[frac > cfg.missing_col_threshold].index if c != tgt]
    out = out.drop(columns=high)
    log("High-missing columns", f"dropped {high or 'none'}")

    if cfg.missing_row_threshold < 1.0 and out.shape[1]:
        n = len(out)
        out = out.loc[out.isna().mean(axis=1) <= cfg.missing_row_threshold].reset_index(drop=True)
        log("High-missing rows", f"dropped {n - len(out)} rows")

    # 9. Missing values (constant or drop only: nothing is learned from the data)
    num = _num_cols(out, tgt)
    if cfg.num_impute == "drop_rows" and num:
        n = len(out)
        out = out.dropna(subset=num).reset_index(drop=True)
        log("Missing numeric", f"dropped {n - len(out)} rows")
    elif cfg.num_impute == "constant" and num:
        filled = int(out[num].isna().sum().sum())
        out[num] = out[num].fillna(0)
        log("Missing numeric", f"constant 0: filled {filled} cells")

    txt = _text_cols(out, tgt)
    if cfg.cat_impute == "drop_rows" and txt:
        n = len(out)
        out = out.dropna(subset=txt).reset_index(drop=True)
        log("Missing categorical", f"dropped {n - len(out)} rows")
    elif cfg.cat_impute == "constant" and txt:
        filled = int(out[txt].isna().sum().sum())
        out[txt] = out[txt].fillna("Unknown")
        log("Missing categorical", f"constant 'Unknown': filled {filled} cells")

    # 10. Constant columns
    if cfg.drop_constant:
        const = [c for c in out.columns if c != tgt and out[c].nunique(dropna=False) <= 1]
        out = out.drop(columns=const)
        log("Constant columns", f"dropped {const or 'none'}")

    log("Output", "done")
    return ETLResult(df=out, steps=steps, target=tgt, column_map=col_map)


def build_report(source_name: str, source_sha256: str, input_shape: tuple,
                 cfg: ETLConfig, result: ETLResult, warnings: list[str]) -> dict:
    """JSON-serialisable audit record of one ETL run."""
    return {
        "report_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": source_name,
        "source_sha256": source_sha256,
        "input_shape": list(input_shape),
        "output_shape": list(result.df.shape),
        "config": asdict(cfg),
        "renamed_columns": {a: b for a, b in result.column_map.items() if a != b},
        "target_in_output": result.target,
        "steps": result.steps,
        "warnings": warnings,
        "output_missing_percent": missing_report(result.df)["missing_%"].to_dict(),
        "output_columns": {str(c): str(t) for c, t in result.df.dtypes.items()},
        "versions": {"pandas": pd.__version__, "numpy": np.__version__},
    }
