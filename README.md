# AutoPrep

Upload a raw CSV or Excel file and get cleaned, model-ready **X_train, X_test, y_train and y_test**.
Everything that is *learned* from the data (fill values, scaling, categories, outlier limits, which
columns to drop) is fitted on the training rows only, then applied unchanged to the test rows, so
your test score stays honest.

Pick a target column and you get X and y for train and test. Leave it empty and you get X_train and
X_test.

## How it works

| Stage | Applies to | What happens |
|---|---|---|
| 1. Clean | every row | Trim spaces, lowercase text, turn `NA`, `?`, `-` and blanks into missing values, convert `"$1,234"` text to numbers (zip codes like `02134` are kept), remove duplicate rows and rows with no target. The **target column is left as uploaded** unless you tick "Also clean the target labels" |
| 2. Split | every row | Seeded train/test split, stratified when the target is a class label |
| 3. Learn | **train only** | Drop constant, duplicate, ID-like, mostly-empty and (optionally) highly correlated columns. Learn fill values, outlier limits, skew corrections, scaler parameters, categories, target classes |
| 4. Apply | train and test | The same fitted steps transform both sets and are saved so you can prepare new data later |

Duplicate rows are removed *before* the split so a row can never sit in both sets.

Feature engineering is left out on purpose: it depends on your domain.

## Presets

| Preset | Adds |
|---|---|
| Recommended | Tidy text, junk-column removal, median / most-frequent fill, standard scaling, one-hot encoding, target encoding |
| Maximum | Recommended plus outlier clipping, skew correction, near-duplicate numeric column removal, `<column>_was_missing` flags |
| Custom | Every option |

## Outputs

The ZIP contains `X_train.csv`, `X_test.csv`, `y_train.csv` and `y_test.csv` (y files only when a
target is chosen), `report.json` (config, what was learned on train, step log), `preprocessor.joblib`
and `how_to_use.py`.

Prepare brand-new raw data with the same fitted steps:

```python
import joblib, pandas as pd
from model_prep import apply_to_new_data

bundle = joblib.load("preprocessor.joblib")      # only load files you created yourself
X_new = apply_to_new_data(pd.read_csv("new_raw_data.csv"), bundle)
```

Load the `.joblib` with the same scikit-learn version that created it.

## Run locally

```bash
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository, with `app.py` and `requirements.txt` at the repository root
   and the `.streamlit/` folder alongside them.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Under **Advanced settings**, pick a recent Python version (this project was tested on 3.13).
5. Click **Deploy**. Pushing to the branch redeploys the app automatically.

Things to know:

- The upload limit is set in `.streamlit/config.toml` (`maxUploadSize`, in MB). The app reads it, so
  the message in the page always matches. It is 50 because Community Cloud apps have limited memory
  and AutoPrep keeps several copies of your data in memory while it works.
- An app with no visitors for 12 hours goes to sleep, and the next visitor can wake it.
- Streamlit's docs recommend pinning the Streamlit version, which `requirements.txt` does.
- A public app is reachable by anyone with the link. Uploaded files are processed in memory, but do
  not use real personal data on a public deployment.

## Project layout

```
app.py              Streamlit interface
etl.py              Stateless cleaning, profiling, issue detection (no Streamlit)
model_prep.py       Split, fit on train, apply to test, reusable preprocessor (no Streamlit)
sample_data.py      A deliberately messy customer-churn dataset
tests/              pytest suite
.streamlit/         Theme and upload limit
```

## Design decisions and limits

- **Split first, then fit.** Fill values, scaling, encoding and column drops come from the training
  rows only. The tests check this directly, for example that a missing test value is filled with the
  *train* median even when the test rows hold extreme values.
- **`"none"` and `"unknown"` are not treated as missing.** They can be real categories.
- **Leading zeros are kept.** `"02134"` stays text, so it is never turned into `2134`.
- **Cleaning decisions made on all rows.** Whether a text column holds only numbers is decided on the
  whole file before the split. This is a data-type decision, not a learned statistic, but it does see
  the test rows.
- **Date-like text is not converted.** The app warns you instead, because date features are
  domain-specific.
- **Excel files:** only the first sheet is read.
- **One split.** There is no validation set; carve one out of X_train and y_train if you tune models.
- **Target-leakage warnings are heuristics.** They can flag a harmless column and miss subtle leaks.

## Behaviour notes

- **Unseen target classes.** If a test row has a class that never appears in training, it is kept in the test set with label `-1` and listed in the warnings (a model cannot predict a class it never saw). Untick "Keep test rows with a class unseen in train" (or set `PrepConfig(unseen_target_classes="drop")`) to remove those rows instead.
- **Type detection uses all rows.** Deciding whether a text column holds numbers happens before the split, so test rows can influence which columns are converted. No statistic is fitted on test rows, and the app lists the converted columns in the warnings.
