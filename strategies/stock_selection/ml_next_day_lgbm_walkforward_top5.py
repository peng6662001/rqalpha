import json
from pathlib import Path

import pandas as pd

from ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, backtest_topn, build_dataset, lgb


TOP_N = 5
YEARS = range(2020, 2026)


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
        "top_n": TOP_N,
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
    }


def main():
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)

    rows = []
    latest_top10_by_year = {}
    for year in YEARS:
        scored = train_predict_year(dataset, feature_cols, year)
        if scored is None:
            continue
        metrics = backtest_topn(scored.copy(), top_n=TOP_N)
        metrics["year"] = year
        rows.append(metrics)

        year_last_date = scored["date"].max()
        latest_rows = (
            scored[scored["date"] == year_last_date]
            .sort_values("pred", ascending=False)
            .head(10)[["order_book_id", "symbol", "pred", "ret_20", "ret_60", "ma_gap_20", "breakout_60", "vol_20"]]
            .copy()
        )
        latest_top10_by_year[str(year)] = {
            "scored_date": year_last_date.date().isoformat(),
            "top10": latest_rows.to_dict(orient="records"),
        }
        print(year, json.dumps(metrics, ensure_ascii=False))

    summary = {
        "strategy": "ml_next_day_lgbm_walkforward_top5",
        "years": rows,
        "aggregate": aggregate_metrics(rows),
        "latest_top10_by_year": latest_top10_by_year,
    }

    out = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_walkforward_top5_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
