"""Gold 04: Volkova's LightGBM with her cross-validation on gold_volkova.train_features.

This follows her notebooks/main.ipynb (DataProcessor2 data + one LightGBM)
at commit d5e228c. Her final submission is an ensemble of six data
processors with LightGBM and CatBoost; we only rebuild this one model.
https://github.com/evgeniavolkova/kagglehomecredit2024/tree/d5e228cdf7e9bf81d10805cff205b473a94c2063

Parts taken from her code:
- params_lgb, N_SPLITS = 5, features = all columns except case_id, date_decision, target
- utils.reduce_mem_usage (float16 when the range fits) and the sort by WEEK_NUM, case_id
- models/tree.py LGBM.fit: shuffled 1% hold-out (seed 42) for early stopping (200 rounds)
- pipeline.py PipelineCV.fit: KFold over the sorted unique weeks, no shuffle
- metrics.py gs_metric: gini stability, penalty=False

She also fits one model on all rows for the test set; that is skipped here
(FIT_FULL_MODEL) because it does not change the CV score.

Needs lightgbm 4.x (she uses ^4.3.0): add it in the Environment panel on
serverless, or as a cluster library. The full table needs about 20 GB of
driver memory. FOLDS_TO_RUN = [4] runs only the last fold; None runs all 5.
SAMPLE_FRACTION < 1 trains on a random share of the cases to save memory.

Output (gold files volume): reports/04_train_lgbm/ (fold_metrics.csv,
weekly_gini.csv, feature_importance.csv, oof_predictions.parquet,
run_config.json) and models/04_train_lgbm/fold{n}.txt
"""

import json
import os
import time

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    # Running the file from VS Code has no Environment panel, so install lightgbm
    # (Volkova uses ^4.3.0, i.e. 4.x) into the Python that runs this script.
    import importlib
    import subprocess
    import sys

    print("lightgbm not found, installing lightgbm~=4.3 ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "lightgbm~=4.3"])
    except subprocess.CalledProcessError as error:
        raise ModuleNotFoundError(
            "Could not pip install lightgbm. Install 'lightgbm~=4.3' on the compute instead: "
            "cluster -> Libraries -> Install new -> PyPI, or the notebook Environment panel on serverless."
        ) from error
    importlib.invalidate_caches()
    import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegralType, NumericType, StringType
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

spark = SparkSession.builder.getOrCreate()


def log(message):
    # flush right away, so progress shows up even when the output is buffered
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class FlushLogger:
    # LightGBM's training log goes through this logger
    def info(self, message):
        print(message, flush=True)

    def warning(self, message):
        print(message, flush=True)


lgb.register_logger(FlushLogger())

TABLE = "dbw_home_credit_audit.gold_volkova.train_features"
VOLUME_PATH = "/Volumes/dbw_home_credit_audit/gold_volkova/files"
REPORT_DIR = f"{VOLUME_PATH}/reports/04_train_lgbm"
MODEL_DIR = f"{VOLUME_PATH}/models/04_train_lgbm"

# From Volkova's notebooks/main.ipynb
params_lgb = {
    "boosting_type": "gbdt",
    "colsample_bynode": 0.8,
    "colsample_bytree": 0.8,
    "extra_trees": True,
    "learning_rate": 0.03,
    "max_depth": 10,
    "metric": "auc",
    "n_estimators": 4000,
    "num_leaves": 64,
    "objective": "binary",
    "random_state": 42,
    "reg_alpha": 10,
    "reg_lambda": 10,
    "verbose": -1,
    "max_bin": 150,
}
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 200
TEST_SIZE = 0.01
RANDOM_SEED = 42

FIT_FULL_MODEL = False
FOLDS_TO_RUN = [4]       # only the last fold (latest weeks as validation); None = all 5 folds
WEEKS_PER_CHUNK = 10     # load the table to pandas a few weeks at a time
# Our addition (not in Volkova's code): train on a random share of the cases when the
# full table does not fit in driver memory. 1.0 = all rows, like hers.
SAMPLE_FRACTION = 0.3

NOT_FEATURES = ["case_id", "date_decision", "target"]


# Volkova's metrics.py gs_metric (the plot is left out)
def gs_metric(true, pred, week, penalty=False):
    # Sort by week
    sorted_indices = np.argsort(week)
    week_sorted = week[sorted_indices]
    true_sorted = true[sorted_indices]
    pred_sorted = pred[sorted_indices]

    # Group by week
    week_unique, week_index = np.unique(week_sorted, return_index=True)
    grouped_true = np.split(true_sorted, week_index[1:])
    grouped_pred = np.split(pred_sorted, week_index[1:])

    # Calculate Gini for each week
    ginis = np.zeros(len(week_unique))
    for i, (t, p) in enumerate(zip(grouped_true, grouped_pred)):
        if len(np.unique(t)) == 1:
            gini = 0.0
        else:
            gini = roc_auc_score(t, p) * 2 - 1
        ginis[i] = gini

    # Calculate Gini Stability
    slope, intercept, _, _, _ = stats.linregress(week_unique, ginis)
    residuals = ginis - (slope * week_unique + intercept)
    out = np.mean(ginis) - 0.5 * np.std(residuals)
    if penalty:
        out += 88.0 * min(0, slope)

    out_dict = {
        "ginis": ginis,
        "weeks": week_unique,          # added by us, for the weekly report
        "slope": 88.0 * slope,
        "std": np.std(residuals),
    }
    return out, out_dict


# Volkova's models/tree.py LGBM.fit with early_stopping_rounds=200, test_size=0.01, shuffle=True
def fit_lgbm(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_SEED
    )
    model = lgb.LGBMClassifier(**params_lgb)
    model.fit(
        X_train, y_train,
        # Volkova also passes (X_train, y_train) here, but LightGBM's early stopping
        # skips the training set, so dropping it gives the same model, only faster
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.callback.log_evaluation(50),
            lgb.callback.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        ],
    )
    return model


def dtype_plan(gold, fields):
    # Volkova's utils.reduce_mem_usage picks the smallest dtype from each column's
    # min and max. We get min/max from Spark first, so every week chunk is cast
    # to the same dtype while loading (loading everything as float64 first does
    # not fit in memory).
    numeric = [f for f in fields if isinstance(f.dataType, NumericType) and f.name not in ("case_id", "target")]
    row = gold.agg(
        *[F.min(f.name).alias(f"lo__{i}") for i, f in enumerate(numeric)],
        *[F.max(f.name).alias(f"hi__{i}") for i, f in enumerate(numeric)],
        *[F.count(F.when(F.col(f.name).isNull(), 1)).alias(f"n__{i}") for i, f in enumerate(numeric)],
    ).first().asDict()

    plan = {}
    for i, f in enumerate(numeric):
        c_min, c_max, nulls = row[f"lo__{i}"], row[f"hi__{i}"], row[f"n__{i}"]
        # an int column with nulls becomes float in pandas, so her float rule applies
        if isinstance(f.dataType, IntegralType) and nulls == 0:
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                    plan[f.name] = dtype
                    break
        elif c_min is None:
            plan[f.name] = np.float64   # all null: her comparisons are False, stays float64
        else:
            # compare in float64 like her pandas code; comparing a python float with a
            # float16 limit would cast the value to float16 first (overflow + rounding)
            c_min, c_max = float(c_min), float(c_max)
            if c_min > float(np.finfo(np.float16).min) and c_max < float(np.finfo(np.float16).max):
                plan[f.name] = np.float16
            elif c_min > float(np.finfo(np.float32).min) and c_max < float(np.finfo(np.float32).max):
                plan[f.name] = np.float32
            else:
                plan[f.name] = np.float64
    return plan


# 1. Load the gold table into pandas

gold = spark.table(TABLE)
fields = gold.schema.fields
features = [f.name for f in fields if f.name not in NOT_FEATURES]   # WEEK_NUM stays a feature, like hers
categorical = [f.name for f in fields if isinstance(f.dataType, StringType) and f.name in features]
plan = dtype_plan(gold, fields)   # dtypes from the full table, so they do not depend on the sample

if SAMPLE_FRACTION < 1.0:
    # hash of case_id instead of rand(), so the same cases are picked every run
    bucket = F.abs(F.xxhash64("case_id", F.lit(RANDOM_SEED))) % 10000
    gold = gold.where(bucket < int(SAMPLE_FRACTION * 10000))
    log(f"Random sample of {SAMPLE_FRACTION:.0%} of the cases")

weeks_all = sorted(row.WEEK_NUM for row in gold.select("WEEK_NUM").distinct().collect())
chunks = []
for start in range(0, len(weeks_all), WEEKS_PER_CHUNK):
    block = weeks_all[start:start + WEEKS_PER_CHUNK]
    chunk = gold.where(F.col("WEEK_NUM").isin(block)).select("case_id", "target", *features).toPandas()
    chunks.append(chunk.astype({c: t for c, t in plan.items() if c in chunk.columns}))
    log(f"Loaded weeks {block[0]}-{block[-1]}: {len(chunk):,} rows")
log("Combining the chunks ...")
df = pd.concat(chunks, ignore_index=True)
del chunks

# Volkova: strings -> categorical (FinalDataProcessor) and df.sort_values([WEEK_NUM, case_id]).
# The row order matters because the 1% early-stopping split depends on it.
log("Strings -> category, sorting ...")
df[categorical] = df[categorical].astype("category")
df = df.sort_values(["WEEK_NUM", "case_id"]).reset_index(drop=True)
log(f"{len(df):,} rows, {len(features)} features ({len(categorical)} categorical), "
      f"{df.memory_usage(deep=True).sum() / 1024**3:.1f} GB")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if FIT_FULL_MODEL:
    full_model = fit_lgbm(df[features], df["target"])
    full_model.booster_.save_model(f"{MODEL_DIR}/full.txt")

# 2. Volkova's PipelineCV.fit: KFold over the unique weeks

preds_df = df[["case_id", "WEEK_NUM", "target"]].copy()
preds_df["preds"] = np.nan
preds_df["fold"] = -1
fold_rows, importances = [], {}

weeks_unique = df["WEEK_NUM"].sort_values().unique()
cv = KFold(n_splits=N_SPLITS, shuffle=False)
for fold, (train_idx, valid_idx) in enumerate(cv.split(weeks_unique)):
    if FOLDS_TO_RUN is not None and fold not in FOLDS_TO_RUN:
        continue
    log("-" * 20 + f"Fold {fold}" + "-" * 20)
    weeks_train = weeks_unique[train_idx]
    weeks_valid = weeks_unique[valid_idx]
    train_rows = df["WEEK_NUM"].isin(weeks_train)
    valid_rows = df["WEEK_NUM"].isin(weeks_valid)

    started = time.time()
    model = fit_lgbm(df.loc[train_rows, features], df.loc[train_rows, "target"])
    preds = model.predict_proba(df.loc[valid_rows, features])[:, 1]
    minutes = (time.time() - started) / 60

    preds_df.loc[valid_rows, "preds"] = preds
    preds_df.loc[valid_rows, "fold"] = fold
    valid = preds_df.loc[valid_rows]
    score, ginis_dict = gs_metric(valid["target"].values, preds, valid["WEEK_NUM"].values, penalty=False)
    score_penalty, _ = gs_metric(valid["target"].values, preds, valid["WEEK_NUM"].values, penalty=True)
    print(
        f"Stability gini: {score:.3f}"
        f", gini: {np.mean(ginis_dict['ginis']):.3f}"
        f", slope: {ginis_dict['slope']:.3f}"
        f", std: {ginis_dict['std']:.3f}"
        f", best iteration: {model.best_iteration_}, {minutes:.1f} min"
    )

    fold_rows.append({
        "fold": fold,
        "valid_weeks": f"{weeks_valid[0]}-{weeks_valid[-1]}",
        "train_rows": int(train_rows.sum()),
        "valid_rows": int(valid_rows.sum()),
        "valid_target_rate": float(valid["target"].mean()),
        "best_iteration": model.best_iteration_,
        "auc": roc_auc_score(valid["target"], preds),
        "gini_stability": score,
        "gini_stability_with_penalty": score_penalty,
        "mean_gini": float(np.mean(ginis_dict["ginis"])),
        "slope": ginis_dict["slope"],
        "residual_std": float(ginis_dict["std"]),
        "minutes": round(minutes, 1),
    })
    importances[fold] = pd.Series(model.feature_importances_, index=model.feature_name_)
    model.booster_.save_model(f"{MODEL_DIR}/fold{fold}.txt")
    del model

# 3. Score on all out-of-fold predictions (her evaluate_and_log), then reports

done = preds_df[preds_df["fold"] >= 0]
score, ginis_dict = gs_metric(done["target"].values, done["preds"].values, done["WEEK_NUM"].values, penalty=False)
score_penalty, _ = gs_metric(done["target"].values, done["preds"].values, done["WEEK_NUM"].values, penalty=True)
fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.loc[len(fold_metrics)] = {
    "fold": "all",
    "valid_rows": len(done),
    "valid_target_rate": float(done["target"].mean()),
    "auc": roc_auc_score(done["target"], done["preds"]),
    "gini_stability": score,
    "gini_stability_with_penalty": score_penalty,
    "mean_gini": float(np.mean(ginis_dict["ginis"])),
    "slope": ginis_dict["slope"],
    "residual_std": float(ginis_dict["std"]),
}
print(fold_metrics.to_string(index=False))

weekly = pd.DataFrame({"WEEK_NUM": ginis_dict["weeks"], "gini": ginis_dict["ginis"]})
weekly = weekly.merge(
    done.groupby("WEEK_NUM").agg(rows=("target", "size"), target_rate=("target", "mean"),
                                 fold=("fold", "first")).reset_index(),
    on="WEEK_NUM",
)

# feature importance like her PipelineCV.get_feature_importances
importance = pd.DataFrame(importances)
folds = list(importance.columns)
importance.columns = [f"fold{c}" for c in folds]
fold_columns = list(importance.columns)
importance["imp"] = importance[fold_columns].mean(axis=1)
importance["spread"] = importance[fold_columns].max(axis=1) - importance[fold_columns].min(axis=1)
importance = importance.sort_values("imp", ascending=False)
importance["imp_acc"] = importance["imp"].cumsum() / importance["imp"].sum()
importance = importance.rename_axis("feature").reset_index()
print(importance.head(20).to_string(index=False))

fold_metrics.to_csv(f"{REPORT_DIR}/fold_metrics.csv", index=False)
weekly.to_csv(f"{REPORT_DIR}/weekly_gini.csv", index=False)
importance.to_csv(f"{REPORT_DIR}/feature_importance.csv", index=False)
done.rename(columns={"preds": "pred"}).to_parquet(f"{REPORT_DIR}/oof_predictions.parquet", index=False)
with open(f"{REPORT_DIR}/run_config.json", "w") as handle:
    json.dump({
        "table": TABLE,
        "params": params_lgb,
        "n_splits": N_SPLITS,
        "folds_run": sorted(int(f) for f in folds),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "test_size": TEST_SIZE,
        "sample_fraction": SAMPLE_FRACTION,
        "features": len(features),
        "lightgbm": lgb.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
    }, handle, indent=2)
log(f"Saved reports to {REPORT_DIR} and models to {MODEL_DIR}")
