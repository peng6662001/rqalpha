import json

import pandas as pd

from ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, build_dataset, compute_metrics, lgb


TOP_N = 5
YEARS = range(2020, 2026)
PRED_THRESHOLD = 0.002


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


def backtest_topn_threshold(scored, top_n=TOP_N, pred_threshold=PRED_THRESHOLD):
    daily = []
    benchmark = []
    scored = scored.sort_values(["date", "pred"], ascending=[True, False])
    for _, group in scored.groupby("date"):
        picks = group[group["pred"] >= pred_threshold].head(top_n).copy()
        if picks.empty:
            daily.append(0.0)
            benchmark.append(float(group["benchmark_next_ret_1"].iloc[0]))
            continue
        executable = picks[picks["next_open_buyable"]].copy()
        if executable.empty:
            daily.append(0.0)
            benchmark.append(float(group["benchmark_next_ret_1"].iloc[0]))
            continue
        target_weight = 1.0 / float(top_n)
        realized = float(executable["next_open_to_close_ret"].sum() * target_weight)
        daily.append(realized)
        benchmark.append(float(group["benchmark_next_ret_1"].iloc[0]) * (len(executable) / float(top_n)))
    return compute_metrics(daily, benchmark)


def aggregate_metrics(rows):
    return {
        "top_n": TOP_N,
        "pred_threshold": PRED_THRESHOLD,
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
    for year in YEARS:
        scored = train_predict_year(dataset, feature_cols, year)
        if scored is None:
            continue
        metrics = backtest_topn_threshold(scored.copy(), top_n=TOP_N, pred_threshold=PRED_THRESHOLD)
        metrics["year"] = year
        rows.append(metrics)
        print(year, json.dumps(metrics, ensure_ascii=False))

    summary = {
        "strategy": "ml_next_day_lgbm_walkforward_top5_pred_threshold",
        "years": rows,
        "aggregate": aggregate_metrics(rows),
    }
    out = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_walkforward_top5_pred_threshold_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
