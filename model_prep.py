"""Split-first preprocessing for model training (no Streamlit calls: unit-testable).

Order of operations (this order is what makes the output leakage-safe):

  1. Stateless cleaning on ALL rows. Rules that look at one value or one row at a time
     (trim text, placeholders -> missing, "$1,234" -> number, duplicate rows, missing target).
     Duplicate rows are removed BEFORE the split so a row cannot land in both train and test.
  2. Train/test split (seeded, stratified for classification targets when possible).
  3. Everything that is LEARNED from data is fitted on the TRAIN rows only, then applied
     unchanged to the test rows:
       - which columns to drop (too many missing, constant, duplicate)
       - outlier clipping limits, imputation values, scaler parameters, category lists
  4. Feature engineering is deliberately out of scope: it depends on the domain.

The fitted pieces are returned in a PrepBundle so the same transformation can be applied to
new raw data later (`apply_to_new_data`) and saved with joblib.
"""
from __future__ import annotations

import io
import logging
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler,
)

from etl import (
    _DATE_LIKE, _NUM_JUNK, ETLConfig, _duplicate_columns, find_id_like_columns, leakage_warnings,
    missing_report, run_pipeline,
)

logger = logging.getLogger("etl.prep")

_NON_CAT = ["number", "bool", "datetime", "datetimetz", "timedelta"]
UNSEEN_LABEL = -1      # encoded label for test rows whose target class never appears in train


# =====================================================================
# Config
# =====================================================================
@dataclass(frozen=True)
class PrepConfig:
    # --- split ---
    test_size: float = 0.2                       # fraction of rows held out for testing
    random_state: int = 42
    shuffle: bool = True                         # False keeps row order (time-ordered data)
    stratify: Literal["auto", "yes", "no"] = "auto"
    # --- fitted on TRAIN only ---
    num_impute: Literal["none", "mean", "median", "constant"] = "median"   # constant = 0
    cat_impute: Literal["none", "most_frequent", "constant"] = "most_frequent"  # constant = "Unknown"
    add_missing_flags: bool = False              # <col>_was_missing for columns missing in train
    clip_outliers: bool = False                  # clip to train quantiles
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    scaler: Literal["none", "standard", "minmax", "robust"] = "standard"
    encoder: Literal["none", "onehot", "ordinal"] = "onehot"
    max_categories: int = 20                     # one-hot: rarer categories share one column
    fix_skew: bool = False                       # Yeo-Johnson on strongly skewed numeric columns
    skew_threshold: float = 1.0                  # |skewness| above this (measured on train)
    drop_id_like: bool = True                    # unique text columns / integer ID columns
    drop_correlated: bool = False                # numeric columns nearly identical to an earlier one
    corr_threshold: float = 0.95
    encode_target: bool = True                   # text/bool class labels -> 0..k-1 (fitted on train)
    unseen_target_classes: Literal["keep", "drop"] = "keep"   # test rows whose class is not in train:
    #   keep = stay in the test set with label -1 (and are reported); drop = removed from the test set

    def __post_init__(self):
        if not 0 < self.corr_threshold <= 1:
            raise ValueError("corr_threshold must be in (0, 1].")
        if self.skew_threshold <= 0:
            raise ValueError("skew_threshold must be positive.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1 (exclusive).")
        lo, hi = self.clip_quantiles
        if not 0 <= lo < hi <= 1:
            raise ValueError("clip_quantiles must satisfy 0 <= lower < upper <= 1.")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2.")
        for name, allowed in (
            ("stratify", ("auto", "yes", "no")),
            ("num_impute", ("none", "mean", "median", "constant")),
            ("cat_impute", ("none", "most_frequent", "constant")),
            ("scaler", ("none", "standard", "minmax", "robust")),
            ("encoder", ("none", "onehot", "ordinal")),
            ("unseen_target_classes", ("keep", "drop")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}.")


@dataclass
class PrepBundle:
    """Everything fitted on the training rows, so it can be re-applied later."""
    stateless_cfg: ETLConfig
    prep_cfg: PrepConfig
    target: str | None                  # target name after cleaning
    column_map: dict[str, str]          # original -> cleaned names
    input_columns: list[str]            # feature columns the preprocessor expects
    num_cols: list[str]
    cat_cols: list[str]
    bool_cols: list[str]
    dropped: dict[str, list[str]]       # reason -> columns dropped (learned on train)
    preprocessor: ColumnTransformer
    target_classes: list[str] | None = None   # class labels; the position is the encoded value


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series | None           # None when no target column was chosen
    y_test: pd.Series | None
    bundle: PrepBundle
    steps: list[dict]
    warnings: list[str]
    split_info: dict = field(default_factory=dict)

    @property
    def train(self) -> pd.DataFrame:
        """X_train with the target appended as the last column (handy for previews)."""
        return self.X_train if self.y_train is None else pd.concat([self.X_train, self.y_train], axis=1)

    @property
    def test(self) -> pd.DataFrame:
        return self.X_test if self.y_test is None else pd.concat([self.X_test, self.y_test], axis=1)


# =====================================================================
# Small fitted transformers
# =====================================================================
class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip numeric columns to quantiles learned in fit (NaN stays NaN)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column
            self.lower_ = np.nanquantile(arr, self.lower, axis=0)
            self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class SkewCorrector(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Yeo-Johnson transform for strongly skewed columns. Which columns, and the power
    parameters, are learned in fit (train rows); other columns pass through unchanged."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        frame = pd.DataFrame(arr)
        skew = np.nan_to_num(frame.skew().to_numpy())
        many_values = (frame.nunique() > 2).to_numpy()          # skip 0/1-style columns
        self.skewed_idx_ = np.where((np.abs(skew) > self.threshold) & many_values)[0]
        self.power_ = None
        if len(self.skewed_idx_):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.power_ = PowerTransformer(method="yeo-johnson", standardize=False).fit(
                    arr[:, self.skewed_idx_])
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        arr = np.array(X, dtype=float, copy=True)
        if self.power_ is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr[:, self.skewed_idx_] = self.power_.transform(arr[:, self.skewed_idx_])
        return arr


class MissingFlags(BaseEstimator, TransformerMixin):
    """0/1 flag columns for the columns that had missing values in the TRAINING data."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = [c for c in df.columns if df[c].isna().any()]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df[self.columns_].isna().astype(int).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_was_missing" for c in self.columns_], dtype=object)


# =====================================================================
# Helpers
# =====================================================================
def _is_classification_target(y: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(y):
        return True
    if pd.api.types.is_float_dtype(y):
        return False
    if pd.api.types.is_numeric_dtype(y):
        return y.nunique() <= 20
    return True


def _stratify_labels(y: pd.Series | None, test_size: float, mode: str, shuffle: bool):
    """Return (labels or None, note). Falls back to a plain split when stratifying can't work."""
    if y is None or mode == "no":
        return None, "off"
    if not shuffle:
        return None, "off (shuffle disabled)"
    if mode == "auto" and not _is_classification_target(y):
        return None, "off (target looks continuous)"
    counts = y.value_counts()
    n, k = len(y), len(counts)
    n_test = int(np.ceil(test_size * n))
    if counts.min() < 2 or n_test < k or n - n_test < k:
        return None, "off (a class is too rare to stratify)"
    return y, "on"


_ID_NAME = re.compile(r"(^|_)(id|idx|index|key|ref|no|num|number|code|uuid)($|_)", re.I)


def describe_target(y: pd.Series) -> str:
    """'classification (k classes)' or 'regression', for display."""
    if _is_classification_target(y):
        return f"classification ({y.nunique(dropna=True)} classes)"
    return "regression"


def _id_like_columns(df: pd.DataFrame, tgt: str | None, min_rows: int = 20,
                     ratio: float = 0.99) -> list[str]:
    """Text columns that are unique for almost every row (IDs, free text), and integer columns
    that are unique AND named like an identifier (customer_id, ref_no ...)."""
    n = len(df)
    if n < min_rows:
        return []
    found = []
    for c in df.columns:
        if c == tgt:
            continue
        s = df[c]
        if (pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)):
            continue
        unique_ratio = s.nunique(dropna=True) / n
        if pd.api.types.is_numeric_dtype(s):
            if unique_ratio >= ratio and _ID_NAME.search(str(c)):
                found.append(c)
        elif unique_ratio >= ratio:
            found.append(c)
    return found


def _date_like_text_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Text columns where (almost) every value looks like a date, e.g. 2024-03-15 or 15/03/2024."""
    found = []
    for c in cols:
        v = df[c].dropna().astype(str)
        if len(v) and v.str.match(_DATE_LIKE).mean() >= 0.9:
            found.append(c)
    return found


def _correlated_columns(df: pd.DataFrame, tgt: str | None, thr: float) -> list[str]:
    """Numeric columns whose |correlation| with an earlier kept column is >= thr."""
    num = [c for c in df.select_dtypes(include="number").columns if c != tgt]
    if len(num) < 2:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = df[num].corr().abs().fillna(0).to_numpy()
    kept: list[int] = []
    drop: list[str] = []
    for i, c in enumerate(num):
        if any(corr[i, j] >= thr for j in kept):
            drop.append(c)
        else:
            kept.append(i)
    return drop


def _fit_structural_drops(train: pd.DataFrame, tgt: str | None, cfg: ETLConfig,
                          prep: PrepConfig) -> dict[str, list[str]]:
    """Decide which columns to drop using the TRAINING rows only."""
    work = train
    dropped: dict[str, list[str]] = {"duplicate": [], "high_missing": [], "constant": [],
                                     "id_like": [], "correlated": []}
    if cfg.drop_duplicate_columns:
        dropped["duplicate"] = _duplicate_columns(work, tgt)
        work = work.drop(columns=dropped["duplicate"])
    frac = work.isna().mean()
    dropped["high_missing"] = [c for c in frac[frac > cfg.missing_col_threshold].index if c != tgt]
    work = work.drop(columns=dropped["high_missing"])
    if cfg.drop_constant:
        dropped["constant"] = [c for c in work.columns
                               if c != tgt and work[c].nunique(dropna=False) <= 1]
        work = work.drop(columns=dropped["constant"])
    if prep.drop_id_like:
        dropped["id_like"] = _id_like_columns(work, tgt)
        work = work.drop(columns=dropped["id_like"])
    if prep.drop_correlated:
        dropped["correlated"] = _correlated_columns(work, tgt, prep.corr_threshold)
    return dropped


def _coerce(X: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
            bool_cols: list[str]) -> pd.DataFrame:
    """Give every column the type it had in training (stateless, safe for any rows)."""
    X = X.copy()
    for c in num_cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            txt = X[c].astype(str).str.replace(_NUM_JUNK, "", regex=True)
            X[c] = pd.to_numeric(txt, errors="coerce")
    for c in bool_cols:
        X[c] = X[c].astype(float).astype(int) if X[c].notna().all() else X[c].astype(float)
    for c in cat_cols:   # one uniform type per column (mixed int/str breaks encoders)
        X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v))
    return X


def build_preprocessor(num_cols: list[str], cat_cols: list[str], cfg: PrepConfig) -> ColumnTransformer:
    transformers: list = []

    if num_cols:
        steps: list = []
        if cfg.clip_outliers:
            steps.append(("clip", QuantileClipper(*cfg.clip_quantiles)))
        if cfg.fix_skew:
            steps.append(("skew", SkewCorrector(cfg.skew_threshold)))
        if cfg.num_impute != "none":
            kw = {"strategy": cfg.num_impute, "keep_empty_features": True}
            if cfg.num_impute == "constant":
                kw["fill_value"] = 0
            steps.append(("impute", SimpleImputer(**kw)))
        scaler = {"standard": StandardScaler, "minmax": MinMaxScaler,
                  "robust": RobustScaler}.get(cfg.scaler)
        if scaler is not None:
            steps.append(("scale", scaler()))
        transformers.append(("num", Pipeline(steps) if steps else "passthrough", num_cols))

    if cat_cols:
        steps = []
        if cfg.cat_impute != "none":
            kw = {"strategy": cfg.cat_impute, "keep_empty_features": True}
            if cfg.cat_impute == "constant":
                kw["fill_value"] = "Unknown"
            steps.append(("impute", SimpleImputer(**kw)))
        if cfg.encoder == "onehot":
            steps.append(("encode", OneHotEncoder(
                handle_unknown="infrequent_if_exist", max_categories=cfg.max_categories,
                sparse_output=False)))
        elif cfg.encoder == "ordinal":
            steps.append(("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)))
        transformers.append(("cat", Pipeline(steps) if steps else "passthrough", cat_cols))

    if cfg.add_missing_flags and (num_cols or cat_cols):
        transformers.append(("flags", MissingFlags(), num_cols + cat_cols))

    if not transformers:
        raise ValueError("No usable feature columns are left after cleaning.")

    ct = ColumnTransformer(transformers, remainder="passthrough",
                           verbose_feature_names_out=False, sparse_threshold=0)
    ct.set_output(transform="pandas")
    return ct


# =====================================================================
# Main entry points
# =====================================================================
def run_split_pipeline(df: pd.DataFrame, etl_cfg: ETLConfig, prep_cfg: PrepConfig) -> SplitResult:
    steps: list[dict] = []
    notes: list[str] = []

    def log(stage: str, step: str, detail: str, rows: int, cols: int) -> None:
        steps.append({"stage": stage, "step": step, "detail": detail, "rows": rows, "cols": cols})
        logger.info("%s | %s | %s", stage, step, detail)

    # ---- 1. stateless cleaning on all rows ----
    stateless_cfg = replace(
        etl_cfg,
        drop_duplicate_columns=False, drop_constant=False,
        missing_col_threshold=1.0,                 # 1.0 = never drop a column here
        num_impute="none", cat_impute="none",      # filling is fitted after the split
    )
    base = run_pipeline(df, stateless_cfg)
    for s in base.steps:
        steps.append({"stage": "1. Clean (stateless, all rows)", **s})
    clean, tgt = base.df, base.target
    converted = [s for s in base.steps
                 if s["step"] == "Numbers stored as text" and "none" not in s["detail"]]
    if converted:
        notes.append("Text-to-number detection looked at every row before the split, so test rows "
                     "influenced which columns were converted (no statistic was fitted on them). "
                     f"{converted[0]['detail']}. Check the list if you need strict evaluation.")
    if len(clean) < 10:
        raise ValueError(f"Only {len(clean)} rows left after cleaning; too few to split.")

    # ---- 2. split ----
    y = clean[tgt] if tgt else None
    labels, strat_note = _stratify_labels(y, prep_cfg.test_size, prep_cfg.stratify, prep_cfg.shuffle)
    if prep_cfg.stratify == "yes" and labels is None and prep_cfg.shuffle:
        notes.append(f"Stratified split was requested but is not possible ({strat_note}); "
                     "a plain random split was used.")
    tr_idx, te_idx = train_test_split(
        np.arange(len(clean)), test_size=prep_cfg.test_size, random_state=prep_cfg.random_state,
        shuffle=prep_cfg.shuffle, stratify=labels,
    )
    train = clean.iloc[tr_idx].reset_index(drop=True)
    test = clean.iloc[te_idx].reset_index(drop=True)
    stage2 = "2. Split"
    log(stage2, "Train/test split",
        f"test_size={prep_cfg.test_size}, seed={prep_cfg.random_state}, "
        f"shuffle={prep_cfg.shuffle}, stratified={strat_note}", len(train), train.shape[1])
    log(stage2, "Test rows held out", "not used for any fitting", len(test), test.shape[1])

    # ---- 3. learn column drops on TRAIN, apply to both ----
    dropped = _fit_structural_drops(train, tgt, etl_cfg, prep_cfg)
    all_dropped = [c for cols in dropped.values() for c in cols]
    train = train.drop(columns=all_dropped)
    test = test.drop(columns=all_dropped)
    stage3 = "3. Fit on train, apply to both"
    log(stage3, "Drop columns (learned on train)",
        "; ".join(f"{k}: {v}" for k, v in dropped.items() if v) or "none",
        len(train), train.shape[1])

    # target: class labels -> integers, mapping learned on train
    target_classes: list[str] | None = None
    target_is_class = bool(tgt) and _is_classification_target(clean[tgt])
    if tgt and prep_cfg.encode_target and (
            pd.api.types.is_bool_dtype(train[tgt]) or not pd.api.types.is_numeric_dtype(train[tgt])):
        classes = sorted(train[tgt].dropna().unique(), key=str)
        mapping = {c: i for i, c in enumerate(classes)}
        known = test[tgt].isin(list(mapping))
        if not known.all():
            unseen = test.loc[~known, tgt].astype(str).value_counts()
            shown = ", ".join(f"'{k}' ({v})" for k, v in unseen.head(10).items())
            if prep_cfg.unseen_target_classes == "drop":
                notes.append(f"{int((~known).sum())} test rows had a target class that never appears in "
                             f"training and were removed from the test set: {shown}.")
                test = test[known].reset_index(drop=True)
            else:
                notes.append(f"{int((~known).sum())} test rows have a target class that never appears in "
                             f"training: {shown}. They were kept in the test set with label {UNSEEN_LABEL}; "
                             "a model trained on this data cannot predict them.")
        train[tgt] = train[tgt].map(mapping).astype(int)
        test[tgt] = test[tgt].map(mapping).fillna(UNSEEN_LABEL).astype(int)
        target_classes = [str(c) for c in classes]
        log(stage3, "Encode target", f"{len(classes)} classes -> 0..{len(classes) - 1}",
            len(train), train.shape[1])

    feats_tr = train.drop(columns=[tgt]) if tgt else train
    feats_te = test.drop(columns=[tgt]) if tgt else test
    num_cols = list(feats_tr.select_dtypes(include="number").columns)
    bool_cols = list(feats_tr.select_dtypes(include="bool").columns)
    cat_cols = list(feats_tr.select_dtypes(exclude=_NON_CAT).columns)
    other_cols = [c for c in feats_tr.columns if c not in num_cols + bool_cols + cat_cols]
    if other_cols:
        notes.append(f"Date/time columns {other_cols} are passed through unchanged: "
                     "extracting date parts is feature engineering and is left to you.")

    feats_tr = _coerce(feats_tr, num_cols, cat_cols, bool_cols)
    feats_te = _coerce(feats_te, num_cols, cat_cols, bool_cols)

    date_text = _date_like_text_columns(feats_tr, cat_cols)
    if date_text:
        notes.append(f"Date-like text columns {date_text} are encoded as ordinary categories, which "
                     "rarely helps a model. Add them to 'Columns to ignore', or turn them into "
                     "date features yourself (that depends on your domain).")

    # ---- 4. fit preprocessor on TRAIN, transform both ----
    pre = build_preprocessor(num_cols, cat_cols, prep_cfg)
    pre.fit(feats_tr)
    out_tr = pre.transform(feats_tr).reset_index(drop=True)
    out_te = pre.transform(feats_te).reset_index(drop=True)
    log(stage3, "Fit preprocessor",
        f"{len(num_cols)} numeric, {len(cat_cols)} categorical, {len(bool_cols)} boolean "
        f"-> {out_tr.shape[1]} model features", len(out_tr), out_tr.shape[1])

    # unseen categories in test (a useful signal, not an error)
    for c in cat_cols:
        seen = set(feats_tr[c].dropna().unique())
        unseen = int((feats_te[c].notna() & ~feats_te[c].isin(seen)).sum())
        if unseen:
            notes.append(f"'{c}': {unseen} test rows have categories never seen in training "
                         "(they get the 'infrequent'/unknown encoding).")

    y_tr = y_te = None
    if tgt:
        y_tr = train[tgt].reset_index(drop=True).rename(tgt)
        y_te = test[tgt].reset_index(drop=True).rename(tgt)
        if target_is_class:
            share = y_tr.value_counts(normalize=True)
            if len(share) >= 2 and share.min() < 0.10:
                label = target_classes[int(share.idxmin())] if target_classes else share.idxmin()
                notes.append(f"Imbalanced target: class '{label}' is only {share.min():.1%} of the "
                             "training rows. Consider class weights or resampling the TRAINING set only.")
    if len(out_te) < 30:
        notes.append(f"The test set has only {len(out_te)} rows; metrics on it will be noisy.")

    # warnings are computed on TRAIN only, so test rows never influence what the user decides
    notes = leakage_warnings(train, tgt) + notes

    bundle = PrepBundle(
        stateless_cfg=stateless_cfg, prep_cfg=prep_cfg, target=tgt, column_map=base.column_map,
        input_columns=list(feats_tr.columns), num_cols=num_cols, cat_cols=cat_cols,
        bool_cols=bool_cols, dropped=dropped, preprocessor=pre, target_classes=target_classes,
    )

    info = {
        "rows_total": len(clean), "rows_train": len(out_tr), "rows_test": len(out_te),
        "test_size": prep_cfg.test_size, "random_state": prep_cfg.random_state,
        "shuffle": prep_cfg.shuffle, "stratified": strat_note,
    }
    if tgt and target_is_class:
        def _dist(s: pd.Series) -> dict:
            d = s.value_counts(normalize=True).round(4)
            if target_classes:
                d.index = [target_classes[int(i)] if int(i) >= 0 else "(unseen in train)"
                           for i in d.index]
            return {str(k): float(v) for k, v in d.items()}
        info["target_distribution_train"] = _dist(train[tgt])
        info["target_distribution_test"] = _dist(test[tgt])

    log("Output", "Done", f"X_train {out_tr.shape}, X_test {out_te.shape}"
        + (", y_train/y_test ready" if tgt else ""), len(out_tr), out_tr.shape[1])
    return SplitResult(X_train=out_tr, X_test=out_te, y_train=y_tr, y_test=y_te, bundle=bundle,
                       steps=steps, warnings=notes, split_info=info)


def apply_to_new_data(raw: pd.DataFrame, bundle: PrepBundle) -> pd.DataFrame:
    """Transform new raw data with the already-fitted bundle. Never drops rows."""
    cfg = replace(bundle.stateless_cfg, drop_duplicates=False, missing_row_threshold=1.0,
                  target_column=None)
    cleaned = run_pipeline(raw, cfg).df
    X = cleaned.reindex(columns=bundle.input_columns)   # absent columns become missing
    X = _coerce(X, bundle.num_cols, bundle.cat_cols, bundle.bool_cols)
    return bundle.preprocessor.transform(X).reset_index(drop=True)


# =====================================================================
# Reporting / persistence
# =====================================================================
def _f(x) -> float | None:
    x = float(x)
    return None if np.isnan(x) else x


def learned_parameters(bundle: PrepBundle) -> dict:
    """What was learned from the training rows (JSON-safe)."""
    out: dict = {}
    ct = bundle.preprocessor
    num = ct.named_transformers_.get("num")
    if isinstance(num, Pipeline):
        cols = bundle.num_cols
        if "clip" in num.named_steps:
            clip = num.named_steps["clip"]
            out["clip_bounds"] = {c: [_f(lo), _f(hi)]
                                  for c, lo, hi in zip(cols, clip.lower_, clip.upper_)}
        if "skew" in num.named_steps:
            sk = num.named_steps["skew"]
            out["skew_corrected_columns"] = [cols[i] for i in sk.skewed_idx_]
            if sk.power_ is not None:
                out["skew_lambdas"] = {cols[i]: _f(v) for i, v in zip(sk.skewed_idx_, sk.power_.lambdas_)}
        if "impute" in num.named_steps:
            out["numeric_fill_values"] = {
                c: _f(v) for c, v in zip(cols, num.named_steps["impute"].statistics_)}
        if "scale" in num.named_steps:
            sc = num.named_steps["scale"]
            if hasattr(sc, "mean_"):
                out["scaler_mean"] = {c: _f(v) for c, v in zip(cols, sc.mean_)}
            if hasattr(sc, "data_min_"):
                out["scaler_min"] = {c: _f(v) for c, v in zip(cols, sc.data_min_)}
                out["scaler_max"] = {c: _f(v) for c, v in zip(cols, sc.data_max_)}
            if hasattr(sc, "center_"):
                out["scaler_median"] = {c: _f(v) for c, v in zip(cols, sc.center_)}
            if hasattr(sc, "scale_") and sc.scale_ is not None:
                out["scaler_scale"] = {c: _f(v) for c, v in zip(cols, sc.scale_)}
    cat = ct.named_transformers_.get("cat")
    if isinstance(cat, Pipeline):
        if "impute" in cat.named_steps:
            out["categorical_fill_values"] = {
                c: str(v) for c, v in zip(bundle.cat_cols, cat.named_steps["impute"].statistics_)}
        enc = cat.named_steps.get("encode")
        if enc is not None:
            out["categories_seen_in_train"] = {
                c: [str(v) for v in cats] for c, cats in zip(bundle.cat_cols, enc.categories_)}
    return out


def build_split_report(source_name: str, source_sha256: str, input_shape: tuple,
                       etl_cfg: ETLConfig, prep_cfg: PrepConfig, result: SplitResult) -> dict:
    import sklearn
    b = result.bundle
    return {
        "report_version": 4,
        "mode": "split_then_fit_on_train",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": source_name,
        "source_sha256": source_sha256,
        "input_shape": list(input_shape),
        "outputs": {
            "X_train": list(result.X_train.shape), "X_test": list(result.X_test.shape),
            "y_train": None if result.y_train is None else len(result.y_train),
            "y_test": None if result.y_test is None else len(result.y_test),
        },
        "target_classes": b.target_classes,
        "train_shape": list(result.X_train.shape),
        "test_shape": list(result.X_test.shape),
        "cleaning_config": asdict(etl_cfg),
        "prep_config": asdict(prep_cfg),
        "split": result.split_info,
        "renamed_columns": {a: c for a, c in b.column_map.items() if a != c},
        "target_in_output": b.target,
        "dropped_columns_learned_on_train": b.dropped,
        "feature_groups": {"numeric": b.num_cols, "categorical": b.cat_cols, "boolean": b.bool_cols},
        "learned_on_train": learned_parameters(b),
        "steps": result.steps,
        "warnings": result.warnings,
        "train_missing_percent": missing_report(result.X_train)["missing_%"].to_dict(),
        "test_missing_percent": missing_report(result.X_test)["missing_%"].to_dict(),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
    }


def bundle_to_bytes(bundle: PrepBundle) -> bytes:
    import joblib
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()
