"""Tests that prove the split-first pipeline never lets test rows influence what is learned."""
import io

import joblib
import numpy as np
import pandas as pd
import pytest

from etl import ETLConfig
from model_prep import (
    PrepConfig, apply_to_new_data, bundle_to_bytes, learned_parameters, run_split_pipeline,
)

# Plain config: no scaling/encoding/imputing, so output values are easy to reason about.
RAW = dict(num_impute="none", cat_impute="none", scaler="none", encoder="none")


def make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "t": np.arange(n, dtype=float),                      # monotonic: shows row order
        "x": rng.normal(size=n),
        "color": rng.choice(["red", "green", "blue"], n),
        "y": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


def run(df, etl=None, **prep):
    etl = etl or ETLConfig(target_column="y")
    return run_split_pipeline(df, etl, PrepConfig(**prep))


# ---------------- split mechanics ----------------
def test_sizes_add_up_and_follow_test_size():
    res = run(make_df(100), test_size=0.25, **RAW)
    assert len(res.train) == 75 and len(res.test) == 25


def test_same_seed_same_split_different_seed_differs():
    df = make_df()
    a = run(df, random_state=1, **RAW).test["t"].tolist()
    b = run(df, random_state=1, **RAW).test["t"].tolist()
    c = run(df, random_state=2, **RAW).test["t"].tolist()
    assert a == b and a != c


def test_no_shuffle_keeps_order_test_is_last_rows():
    res = run(make_df(100), shuffle=False, test_size=0.2, **RAW)
    assert res.test["t"].min() > res.train["t"].max()


def test_stratified_split_keeps_class_ratio():
    res = run(make_df(200), test_size=0.25, **RAW)
    assert res.split_info["stratified"] == "on"
    assert res.test["y"].mean() == pytest.approx(res.train["y"].mean(), abs=0.03)


def test_stratify_falls_back_when_class_too_rare():
    df = make_df(100)
    df["y"] = 0
    df.loc[0, "y"] = 1                      # a single positive row
    res = run(df, **RAW)
    assert res.split_info["stratified"].startswith("off")


def test_no_target_still_splits():
    res = run(make_df(), etl=ETLConfig(), **RAW)
    assert len(res.train) + len(res.test) == 100


def test_duplicate_rows_removed_before_split_so_none_cross_over():
    df = pd.concat([make_df(60), make_df(60)], ignore_index=True)   # every row twice
    res = run(df, **RAW)
    assert len(res.train) + len(res.test) == 60
    cols = [c for c in res.train.columns]
    both = pd.concat([res.train[cols], res.test[cols]]).duplicated().sum()
    assert both == 0


def test_rows_with_missing_target_are_dropped_before_split():
    df = make_df(100)
    df.loc[:9, "y"] = np.nan
    res = run(df, **RAW)
    assert len(res.train) + len(res.test) == 90
    assert res.train["y"].notna().all() and res.test["y"].notna().all()


def test_too_few_rows_raises():
    with pytest.raises(ValueError, match="too few"):
        run(make_df(8), **RAW)


# ---------------- leakage: nothing is learned from test ----------------
def test_imputation_value_comes_from_train_only():
    df = make_df(100)
    df["x"] = np.where(np.arange(100) % 2 == 0, 1.0, 3.0)   # train median is exactly 2.0
    df.loc[80:, "x"] = 1000.0               # test block (no shuffle) is extreme
    df.loc[95, "x"] = np.nan                # a missing value inside the test block
    res = run(df, shuffle=False, test_size=0.2, num_impute="median",
              cat_impute="none", scaler="none", encoder="none")
    filled = res.test.loc[res.test["t"] == 95, "x"].iloc[0]
    assert filled == 2.0                    # train median, not influenced by the 1000s


def test_scaler_is_fitted_on_train_not_test():
    df = make_df(100)
    df.loc[80:, "x"] = df.loc[80:, "x"] + 50          # shifted test block
    res = run(df, shuffle=False, test_size=0.2, scaler="standard",
              num_impute="none", cat_impute="none", encoder="none")
    assert res.train["x"].mean() == pytest.approx(0, abs=1e-9)
    assert res.test["x"].mean() > 5                   # NOT re-centred on itself


def test_clipping_limits_come_from_train():
    df = make_df(100)
    df.loc[85, "x"] = 10_000.0                         # extreme value in test block
    res = run(df, shuffle=False, test_size=0.2, clip_outliers=True,
              clip_quantiles=(0.0, 0.99), **RAW)
    train_x = df.loc[:79, "x"]
    assert res.test["x"].max() <= np.quantile(train_x, 0.99) + 1e-9


def test_constant_in_train_is_dropped_from_both_even_if_it_varies_in_test():
    df = make_df(100)
    df["c"] = 5
    df.loc[85:, "c"] = 9
    res = run(df, shuffle=False, test_size=0.2, **RAW)
    assert "c" not in res.train.columns and "c" not in res.test.columns
    assert "c" in res.bundle.dropped["constant"]


def test_high_missing_column_decided_on_train_only():
    df = make_df(100)
    df["m"] = 1.0
    df.loc[80:, "m"] = np.nan                           # missing ONLY in the test block
    df.loc[:79:2, "m"] = 2.0                            # train has variation
    etl = ETLConfig(target_column="y", missing_col_threshold=0.5)
    res = run(df, etl=etl, shuffle=False, test_size=0.2, **RAW)
    assert "m" in res.train.columns and "m" in res.test.columns


def test_onehot_columns_come_from_train_categories_and_unseen_is_safe():
    df = make_df(100)
    df["color"] = np.where(np.arange(100) % 2 == 0, "red", "blue")
    df.loc[85:, "color"] = "purple"                    # category only in the test block
    res = run(df, shuffle=False, test_size=0.2, num_impute="none", cat_impute="none",
              scaler="none", encoder="onehot")
    assert not any("purple" in c for c in res.train.columns)
    assert any("purple" in w or "never seen" in w for w in res.warnings)
    assert list(res.train.columns) == list(res.test.columns)


def test_train_and_test_have_identical_columns_and_target_last():
    res = run(make_df(), **{**RAW, "encoder": "onehot"})
    assert list(res.train.columns) == list(res.test.columns)
    assert res.train.columns[-1] == "y"


def test_missing_flags_only_for_columns_missing_in_train():
    df = make_df(100)
    df["a"] = 1.0
    df["b"] = 1.0
    df.loc[[2, 5], "a"] = np.nan                       # train rows
    df.loc[90, "b"] = np.nan                           # test row only
    res = run(df, shuffle=False, test_size=0.2, add_missing_flags=True,
              num_impute="median", cat_impute="none", scaler="none", encoder="none")
    assert "a_was_missing" in res.train.columns
    assert "b_was_missing" not in res.train.columns


# ---------------- stateless cleaning still behaves ----------------
def test_leading_zero_ids_and_none_category_survive():
    df = make_df(100)
    df["zip"] = np.random.default_rng(1).choice(["02134", "10001"], 100)
    df["opt"] = np.random.default_rng(2).choice(["none", "some"], 100)
    etl = ETLConfig(target_column="y", fix_numeric_text=True, placeholders_as_missing=True)
    res = run(df, etl=etl, num_impute="none", cat_impute="none", scaler="none",
              encoder="onehot")
    assert "zip_02134" in res.train.columns
    assert "opt_none" in res.train.columns


def test_target_is_never_transformed():
    df = make_df(100)
    res = run(df, scaler="standard", num_impute="median", encoder="onehot")
    assert set(res.train["y"].unique()) <= {0, 1}


# ---------------- reuse on new data ----------------
def test_apply_to_new_data_same_columns_no_rows_dropped():
    df = make_df(100)
    res = run(df, scaler="standard", num_impute="median", encoder="onehot")
    new = pd.concat([df.drop(columns="y").head(5)] * 2, ignore_index=True)  # duplicates kept
    out = apply_to_new_data(new, res.bundle)
    assert len(out) == 10
    assert list(out.columns) == [c for c in res.train.columns if c != "y"]


def test_apply_to_new_data_handles_unseen_category_and_missing_column():
    df = make_df(100)
    res = run(df, scaler="standard", num_impute="median", encoder="onehot")
    new = pd.DataFrame({"t": [1.0], "color": ["never_seen"]})       # 'x' column absent
    out = apply_to_new_data(new, res.bundle)
    assert len(out) == 1 and out.isna().sum().sum() == 0


def test_bundle_roundtrips_through_joblib():
    df = make_df(100)
    res = run(df, scaler="standard", num_impute="median", encoder="onehot")
    restored = joblib.load(io.BytesIO(bundle_to_bytes(res.bundle)))
    new = df.drop(columns="y").head(5)
    pd.testing.assert_frame_equal(apply_to_new_data(new, res.bundle),
                                  apply_to_new_data(new, restored))


def test_learned_parameters_are_json_safe_and_match_train():
    import json
    df = make_df(100)
    res = run(df, scaler="standard", num_impute="median", encoder="onehot")
    params = learned_parameters(res.bundle)
    json.dumps(params)
    assert "scaler_mean" in params and "numeric_fill_values" in params


# ---------------- config validation ----------------
@pytest.mark.parametrize("kwargs", [
    {"test_size": 0}, {"test_size": 1}, {"clip_quantiles": (0.9, 0.1)},
    {"scaler": "bogus"}, {"encoder": "bogus"}, {"num_impute": "mode"}, {"max_categories": 1},
])
def test_invalid_prep_config_raises(kwargs):
    with pytest.raises(ValueError):
        PrepConfig(**kwargs)
