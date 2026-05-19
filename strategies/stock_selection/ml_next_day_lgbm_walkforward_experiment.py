import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_next_day_lgbm_baseline import (
    DEFAULT_OUTPUT_DIR,
    backtest_topn,
    build_dataset,
    lgb,
)


def train_predict_year(dataset, feature_cols, year):
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")

    labeled = dataset.dropna(subset=["next_ret_1", "benchmark_next_ret_1"]).copy()
    train_df = labeled[labeled["date"] <= train_end].copy()
    test_df = labeled[(labeled["date"] >= test_start) & (labeled["date"] <= test_end)].copy()
    if train_df.empty or test_df.empty:
        return None

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df["next_ret_1"])
    test_df["pred"] = model.predict(test_df[feature_cols])
    return test_df


def aggregate_metrics(rows):
    return {
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
    }


def run_experiment(top_ns=(1, 3, 5, 10), years=range(2020, 2026)):
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)

    scored_by_year = {}
    for year in years:
        scored = train_predict_year(dataset, feature_cols, year)
        if scored is not None:
            scored_by_year[year] = scored

    summary = {}
    for top_n in top_ns:
        rows = []
        for year, scored in scored_by_year.items():
            metrics = backtest_topn(scored.copy(), top_n=top_n)
            metrics["year"] = year
            rows.append(metrics)
        summary[str(top_n)] = {
            "years": rows,
            "aggregate": aggregate_metrics(rows),
        }
        print(f"top_n={top_n}", json.dumps(summary[str(top_n)]["aggregate"], ensure_ascii=False))

    out = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_walkforward_topn_experiment.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    run_experiment()
