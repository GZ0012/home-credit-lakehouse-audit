# Home Credit Lakehouse Audit

technical audit of an automated decision
system (ADS), built on a Databricks lakehouse (Delta Lake + Unity Catalog).

## What this project does

We audit two independently developed algorithmic decision systems (ADSs) built
for the completed **Home Credit – Credit Risk Model Stability** Kaggle
competition (2024). Both systems estimate applicants' credit default risk.
We compare their predictive reliability, temporal stability, and group /
individual fairness on a common holdout set, rather than taking their
leaderboard scores at face value.

- **Solution A** — [yuuniee, 1st place solution](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/yuuniee-1st-place-solution-my-betting-strategy) (CatBoost + LightGBM + DenseLight ensemble)
- **Solution B** — [Evgeniia Volkova, independent submission](https://github.com/evgeniavolkova/kagglehomecredit2024) (LightGBM/CatBoost pipeline with time-window features)
- **Data** — [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) (Kaggle, 2024)

Full scope and audit plan: see the project proposal (submitted separately on
Gradescope per course requirements).

## Architecture

Data processing runs on a shared Databricks workspace (Delta Lake tables,
registered in Unity Catalog), layered as:

```
raw (Kaggle source files)
  -> bronze  (ingested as Delta tables, minimal typing)
  -> silver  (cleaned, deduplicated, joined across history tables)
  -> gold    (aggregated feature mart used for modeling / audit)
```

The final course deliverable is a **Google Colab notebook**, so the `gold`
layer is exported (Parquet/CSV) and pulled into Colab for modeling, fairness
analysis, and the report figures — Databricks is the pipeline, Colab is the
graded artifact.

## Repo structure

```
notebooks/
  bronze/              ingestion notebooks (raw -> bronze)
  silver/               cleaning / joins (bronze -> silver)
  gold/                  feature aggregation (silver -> gold)
  ml_reproduction/
    solution_a_yuuniee/    reproduction of Solution A
    solution_b_volkova/    reproduction of Solution B
  colab_export/          notebooks used to push gold-layer data to Colab
docs/                     architecture notes, audit plan, meeting notes
```


## Working on this repo

Notebooks are developed in Databricks and synced here via **Databricks
Repos** (Git integration) — each team member connects their own GitHub
credentials in Databricks User Settings, then commits/pushes from the Repos
Git panel. See `docs/databricks-repos-setup.md` (TBD) for the walkthrough.

## Citations

Code and data sources are cited inline in each notebook and in the project
proposal, per course requirement to "prominently cite your sources of code
and data."
