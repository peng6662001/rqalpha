import json
from pathlib import Path

import pandas as pd

from ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, build_dataset
from ml_next_day_lgbm_walkforward_top5 import TOP_N, YEARS, train_predict_year


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


def build_daily_payload(scored, year):
    daily = []
    for date, group in scored.sort_values(["date", "pred"], ascending=[True, False]).groupby("date"):
        top = group.head(TOP_N).copy()
        top["pred"] = top["pred"].astype(float)
        payload = {
            "selection_date": pd.Timestamp(date).date().isoformat(),
            "next_trading_date": pd.Timestamp(date).date().isoformat(),
            "top_n": TOP_N,
            "candidates": top[PICK_COLUMNS].to_dict(orient="records"),
        }
        daily.append(payload)
    return {
        "strategy": "ml_next_day_lgbm_walkforward_top5_daily_picks",
        "year": year,
        "days": daily,
    }


def main():
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)

    manifest = []
    for year in YEARS:
        scored = train_predict_year(dataset, feature_cols, year)
        if scored is None:
            continue
        payload = build_daily_payload(scored, year)
        out = DEFAULT_OUTPUT_DIR / f"ml_next_day_lgbm_walkforward_top5_daily_picks_{year}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(
            {
                "year": year,
                "file": out.name,
                "days": len(payload["days"]),
                "first_date": payload["days"][0]["selection_date"] if payload["days"] else None,
                "last_date": payload["days"][-1]["selection_date"] if payload["days"] else None,
            }
        )
        print(json.dumps(manifest[-1], ensure_ascii=False))

    manifest_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_walkforward_top5_daily_picks_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
