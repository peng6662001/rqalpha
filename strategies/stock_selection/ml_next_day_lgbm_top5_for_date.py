import argparse
import json
from pathlib import Path

import pandas as pd

from ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, build_dataset
from ml_next_day_lgbm_walkforward_top5 import TOP_N, train_predict_year


PICK_COLUMNS = [
    "order_book_id",
    "symbol",
    "pred",
    "ret_20",
    "ret_60",
    "ma_gap_20",
    "breakout_60",
    "vol_20",
]


def resolve_trade_date(scored, requested_date):
    requested = pd.Timestamp(requested_date)
    exact = scored[scored["date"] == requested].copy()
    if not exact.empty:
        return exact, requested
    eligible = scored[scored["date"] <= requested].copy()
    if eligible.empty:
        return None, None
    used_date = eligible["date"].max()
    return scored[scored["date"] == used_date].copy(), used_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="target trading date, e.g. 2025-04-30")
    parser.add_argument("--year", type=int, default=None, help="optional explicit walk-forward test year")
    parser.add_argument("--output", default=None, help="optional output json path")
    args = parser.parse_args()

    requested_date = pd.Timestamp(args.date)
    year = args.year or requested_date.year

    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)
    scored = train_predict_year(dataset, feature_cols, year)
    if scored is None:
        raise RuntimeError(f"no scored data for year={year}")

    day_rows, used_date = resolve_trade_date(scored, requested_date)
    if day_rows is None or used_date is None:
        raise RuntimeError(f"no available scored date on or before {requested_date.date().isoformat()} in year={year}")

    top = day_rows.sort_values("pred", ascending=False).head(TOP_N).copy()
    top["pred"] = top["pred"].astype(float)

    payload = {
        "strategy": "ml_next_day_lgbm_top5_for_date",
        "requested_date": requested_date.date().isoformat(),
        "used_scored_date": used_date.date().isoformat(),
        "walkforward_test_year": year,
        "top_n": TOP_N,
        "candidates": top[PICK_COLUMNS].to_dict(orient="records"),
    }

    if args.output:
        out = Path(args.output)
    else:
        out = DEFAULT_OUTPUT_DIR / f"ml_next_day_lgbm_top5_{used_date.date().isoformat()}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
