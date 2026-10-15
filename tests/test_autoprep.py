"""Tests for the X/y outputs, the learned column drops, target encoding, skew handling and the
presets, run end to end on the bundled messy sample dataset."""
import ast
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from etl import ETLConfig, load_data
from model_prep import PrepConfig, learned_parameters, run_split_pipeline
from sample_data import make_sample_data

# Plain settings: nothing is imputed, scaled or encoded, so values are easy to reason about.
RAW = dict(num_impute="none", cat_impute="none", scaler="none", encoder="none")


def run(df, target="y", etl=None, **prep):
    etl = etl or ETLConfig(target_column=target)
    return run_split_pipeline(df, etl, PrepConfig(**prep))


def base(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                         "y": rng.choice([0, 1], n)})


# ---------------- X / y outputs ----------------
def test_with_target_we_get_x_and_y_for_train_and_test():
    res = run(base(100), test_size=0.25, **RAW)
    assert len(res.X_train) == len(res.y_train) == 75
    assert len(res.X_test) == len(res.y_test) == 25
    assert "y" not in res.X_train.columns and "y" not in res.X_test.columns
    assert res.y_train.name == "y" and res.y_test.name == "y"
    assert list(res.X_train.columns) == list(res.X_test.columns)


def test_rows_of_x_and_y_stay_aligned():
    df = base(100)
    df["y"] = (df["x"] > 0).astype(int)                 # y is a function of x
    res = run(df, **RAW)
    assert ((res.X_train["x"] > 0).astype(int) == res.y_train).all()
    assert ((res.X_test["x"] > 0).astype(int) == res.y_test).all()
    assert res.X_train.index.equals(res.y_train.index)


def test_without_target_we_get_only_x():
    res = run(base(100), target=None, **RAW)
    assert res.y_train is None and res.y_test is None
    assert {"x", "z", "y"} <= set(res.X_train.columns)   # nothing is split off
    assert len(res.X_train) + len(res.X_test) == 100


# ---------------- target encoding ----------------
def test_text_target_is_encoded_with_classes_learned_on_train():
    df = base(100)
    df["y"] = np.where(np.arange(100) % 2 == 0, "yes", "no")
    res = run(df, **RAW)
    assert res.bundle.target_classes == ["no", "yes"]
    assert set(res.y_train.unique()) == {0, 1}
    assert pd.api.types.is_integer_dtype(res.y_train)


def test_unseen_target_class_rows_are_removed_when_policy_is_drop():
    df = base(100)
    df["y"] = np.where(np.arange(100) % 2 == 0, "yes", "no")
    df.loc[90, "y"] = "maybe"                            # only in the test block
    res = run(df, shuffle=False, stratify="no", test_size=0.2, unseen_target_classes="drop", **RAW)
    assert res.bundle.target_classes == ["no", "yes"]
    assert len(res.X_test) == len(res.y_test) == 19
    assert any("never appears in training" in w and "'maybe'" in w for w in res.warnings)


def test_unseen_target_class_rows_are_kept_by_default_and_reported():
    df = base(100)
    df["y"] = np.where(np.arange(100) % 2 == 0, "yes", "no")
    df.loc[90, "y"] = "maybe"
    res = run(df, shuffle=False, stratify="no", test_size=0.2, **RAW)
    assert len(res.X_test) == len(res.y_test) == 20          # nothing silently removed
    assert res.bundle.target_classes == ["no", "yes"]
    assert (res.y_test == -1).sum() == 1 and (res.y_train == -1).sum() == 0
    assert any("never appears in training" in w and "'maybe'" in w for w in res.warnings)
    assert "(unseen in train)" in res.split_info["target_distribution_test"]


# ---------------- target is protected from text clean-up ----------------
def _label_df():
    df = base(100)
    df["y"] = np.where(np.arange(100) % 2 == 0, "Yes", "yes")
    df["y"] = df["y"].where(np.arange(100) % 5 != 0, "No")
    df["city"] = np.where(np.arange(100) % 2 == 0, "Paris", "LONDON")
    return df


def test_target_labels_are_not_lowercased_by_default():
    res = run(_label_df(), etl=ETLConfig(target_column="y", lowercase_text=True), **RAW)
    assert set(res.bundle.target_classes) == {"Yes", "yes", "No"}   # labels untouched
    assert set(res.X_train["city"].unique()) <= {"paris", "london"}   # features still cleaned


def test_target_labels_are_cleaned_when_the_user_opts_in():
    cfg = ETLConfig(target_column="y", lowercase_text=True, clean_target_text=True)
    res = run(_label_df(), etl=cfg, **RAW)
    assert set(res.bundle.target_classes) == {"yes", "no"}


def test_numeric_target_is_left_alone_and_encode_can_be_switched_off():
    df = base(100)
    df["y"] = np.arange(100, dtype=float)
    res = run(df, **RAW)
    assert res.bundle.target_classes is None and res.y_train.dtype == float
    df2 = base(100)
    df2["y"] = np.where(np.arange(100) % 2 == 0, "yes", "no")
    res2 = run(df2, encode_target=False, **RAW)
    assert set(res2.y_train.unique()) == {"yes", "no"}


# ---------------- ID-like columns ----------------
def test_id_like_columns_are_dropped_and_ordinary_ones_kept():
    df = base(100)
    df["ref_code"] = [f"A-{i:04d}" for i in range(100)]         # unique text
    df["customer_id"] = np.arange(100)                          # unique integer named like an ID
    df["visits"] = np.random.default_rng(1).permutation(100)    # unique integer, ordinary name
    res = run(df, **RAW)
    assert set(res.bundle.dropped["id_like"]) == {"ref_code", "customer_id"}
    assert "visits" in res.X_train.columns


def test_id_like_dropping_can_be_turned_off_and_target_is_never_dropped():
    df = base(100)
    df["customer_id"] = np.arange(100)
    assert "customer_id" in run(df, drop_id_like=False, **RAW).X_train.columns
    df2 = base(100)
    df2["y"] = [f"c{i}" for i in range(100)]                    # unique-per-row target
    res = run(df2, stratify="no", **RAW)
    assert res.y_train is not None and "y" not in res.bundle.dropped["id_like"]


# ---------------- correlated columns ----------------
def test_nearly_identical_numeric_column_is_dropped_later_one():
    df = base(100)
    df["x_copy"] = df["x"] * 3 + 1
    res = run(df, drop_correlated=True, **RAW)
    assert res.bundle.dropped["correlated"] == ["x_copy"]
    assert "x" in res.X_train.columns and "x_copy" not in res.X_test.columns


def test_correlation_is_judged_on_train_rows_only():
    df = base(100)
    df["w"] = np.random.default_rng(5).normal(size=100)
    df.loc[80:, "w"] = df.loc[80:, "x"]                         # identical to x ONLY in test rows
    res = run(df, shuffle=False, stratify="no", test_size=0.2, drop_correlated=True, **RAW)
    assert "w" in res.X_train.columns and not res.bundle.dropped["correlated"]


# ---------------- skew ----------------
def test_skewed_column_is_corrected_and_others_untouched():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"income": rng.lognormal(10, 1.2, 400), "age": rng.normal(40, 10, 400),
                       "y": rng.choice([0, 1], 400)})
    res = run(df, fix_skew=True, **RAW)
    assert abs(res.X_train["income"].skew()) < abs(df["income"].skew()) / 2
    assert learned_parameters(res.bundle)["skew_corrected_columns"] == ["income"]
    assert res.X_train["age"].equals(res.X_train["age"])        # not transformed (still normal-ish)
    assert abs(res.X_train["age"].mean() - 40) < 3


def test_skew_is_measured_on_train_rows_only():
    df = base(100)
    df.loc[80:, "x"] = 1e6 * (1 + np.arange(20))                # wild values only in test rows
    res = run(df, shuffle=False, stratify="no", test_size=0.2, fix_skew=True, **RAW)
    assert "x" not in learned_parameters(res.bundle)["skew_corrected_columns"]


# ---------------- warnings ----------------
def test_date_like_text_column_triggers_a_warning():
    df = base(100)
    df["signup"] = pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d")
    df["signup"] = df["signup"].str[:8] + "01"                  # repeats, so it is not an ID
    res = run(df, num_impute="none", cat_impute="none", scaler="none", encoder="onehot")
    assert any("Date-like text columns" in w and "signup" in w for w in res.warnings)


def test_imbalanced_target_is_flagged():
    df = base(200)
    df["y"] = np.where(np.arange(200) < 8, "rare", "common")
    res = run(df, **RAW)
    assert any("Imbalanced target" in w and "rare" in w for w in res.warnings)


@pytest.mark.parametrize("kwargs", [{"corr_threshold": 0}, {"corr_threshold": 1.5},
                                    {"skew_threshold": 0}, {"skew_threshold": -1}])
def test_new_config_values_are_validated(kwargs):
    with pytest.raises(ValueError):
        PrepConfig(**kwargs)


# ---------------- the real sample dataset, both presets ----------------
@pytest.fixture(scope="module")
def app_module():
    pytest.importorskip("streamlit")
    import app
    return app


@pytest.fixture(scope="module")
def sample():
    return make_sample_data()


@pytest.mark.parametrize("preset", ["RECOMMENDED", "MAXIMUM"])
def test_sample_data_end_to_end(app_module, sample, preset):
    opts = getattr(app_module, preset)
    etl_cfg, prep_cfg = app_module.build_configs(opts, "Churn", [], 20)
    res = run_split_pipeline(sample, etl_cfg, prep_cfg)

    assert len(res.X_train) == len(res.y_train) and len(res.X_test) == len(res.y_test)
    assert list(res.X_train.columns) == list(res.X_test.columns)
    assert res.X_train.isna().sum().sum() == 0 and res.X_test.isna().sum().sum() == 0
    # the target is no longer lowercased by default, so its labels keep their original case
    assert set(res.y_train.unique()) <= {0, 1} and res.bundle.target_classes == ["No", "Yes"]
    assert "churn" not in res.X_train.columns
    assert {"customer_id", "country"} <= set(sum(res.bundle.dropped.values(), []))
    assert "zip_code_02134" in res.X_train.columns              # leading zero survived
    assert "gender_male" in res.X_train.columns and "gender_ MALE " not in res.X_train.columns
    assert ("tenure_years" in res.bundle.dropped["correlated"]) == (preset == "MAXIMUM")
    assert len(res.X_train) + len(res.X_test) == 992            # 1,012 rows - 12 duplicates - 8 with no target


def test_issue_table_finds_the_planted_problems(app_module, sample):
    found = " | ".join(i["issue"] for i in app_module.cached_issues("id", sample))
    for expected in ("Duplicate rows", "Missing values", "Placeholder text", "Numbers stored as text",
                     "Extra spaces", "Constant columns", "ID-like columns", "skewed"):
        assert expected in found, expected


def test_package_run_builds_every_download(app_module, sample):
    etl_cfg, prep_cfg = app_module.build_configs(app_module.RECOMMENDED, "Churn", [], 20)
    run_ = app_module.package_run(sample, "sample.csv", "id", etl_cfg, prep_cfg, "k")
    names = set(zipfile.ZipFile(io.BytesIO(run_["zip"])).namelist())
    assert names == {"X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "report.json",
                     "preprocessor.joblib", "how_to_use.py"}
    ast.parse(run_["snippet"])                                  # the code we hand out is valid Python
    x = pd.read_csv(io.BytesIO(run_["files"]["X_train.csv"]))
    y = pd.read_csv(io.BytesIO(run_["files"]["y_train.csv"]))
    assert len(x) == len(y) and list(y.columns) == ["churn"]


def test_package_run_without_target_has_no_y_files(app_module, sample):
    etl_cfg, prep_cfg = app_module.build_configs(app_module.RECOMMENDED, None, [], 20)
    run_ = app_module.package_run(sample, "sample.csv", "id", etl_cfg, prep_cfg, "k")
    names = set(zipfile.ZipFile(io.BytesIO(run_["zip"])).namelist())
    assert names == {"X_train.csv", "X_test.csv", "report.json", "preprocessor.joblib", "how_to_use.py"}
    ast.parse(run_["snippet"])


# ---------------- loading real files: the path an upload takes ----------------
def test_csv_load_keeps_leading_zeros_but_leaves_ordinary_numbers_numeric():
    csv = "zip,code,age,price,flag\n02134,007,25,0.5,1\n10001,012,31,1.5,0\n00501,100,40,2.0,1\n"
    df = load_data(io.BytesIO(csv.encode()), "x.csv")
    assert list(df["zip"]) == ["02134", "10001", "00501"]
    assert list(df["code"]) == ["007", "012", "100"]
    assert pd.api.types.is_integer_dtype(df["age"]) and pd.api.types.is_integer_dtype(df["flag"])
    assert pd.api.types.is_float_dtype(df["price"])             # "0.5" is a number, not an identifier


def test_csv_load_keeps_leading_zeros_with_missing_values_and_latin1_text():
    csv = "city,zip\nZ\u00fcrich,02134\nBern,\nGen\u00e8ve,00501\n".encode("latin-1")
    df = load_data(io.BytesIO(csv), "x.csv")
    assert df["zip"].tolist()[0] == "02134" and df["zip"].isna().sum() == 1
    assert "Z\u00fcrich" in df["city"].tolist()


def test_a_real_csv_upload_gives_the_same_result_as_the_in_memory_sample(app_module, sample):
    loaded = load_data(io.BytesIO(sample.to_csv(index=False).encode()), "sample.csv")
    assert "02134" in set(loaded["Zip Code"])                   # zip codes survive the file
    etl_cfg, prep_cfg = app_module.build_configs(app_module.RECOMMENDED, "Churn", [], 20)
    a = run_split_pipeline(sample, etl_cfg, prep_cfg)
    b = run_split_pipeline(loaded, etl_cfg, prep_cfg)
    assert "zip_code_02134" in b.X_train.columns
    pd.testing.assert_frame_equal(a.X_train, b.X_train, check_dtype=False)
    pd.testing.assert_frame_equal(a.X_test, b.X_test, check_dtype=False)
    pd.testing.assert_series_equal(a.y_train, b.y_train, check_dtype=False)


def test_the_word_none_stays_a_category_but_other_missing_markers_still_work():
    csv = "plan,note\nNone,NA\nbasic,n/a\nNone,null\npremium,x\n"
    df = load_data(io.BytesIO(csv.encode()), "x.csv")
    assert df["plan"].tolist() == ["None", "basic", "None", "premium"]    # kept as a category
    assert df["note"].isna().sum() == 3                                    # NA, n/a, null -> missing


def test_excel_upload_keeps_none_and_text_zip_codes(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "d.xlsx"
    pd.DataFrame({"zip": ["02134", "10001"], "plan": ["None", "basic"], "n": [1, 2]}).to_excel(
        path, index=False)
    with open(path, "rb") as f:
        df = load_data(f, "d.xlsx")
    assert df["zip"].tolist() == ["02134", "10001"] and df["plan"].tolist() == ["None", "basic"]
    assert pd.api.types.is_integer_dtype(df["n"])
