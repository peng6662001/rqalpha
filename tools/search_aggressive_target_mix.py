from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "tools").resolve()))

from tune_aggressive_extreme_leader_backup import aggregate, env_params, run_one_year


YEARS = range(2020, 2026)
OUTPUT = ROOT / ".temp" / "search_aggressive_target_mix_2020_2025.json"


BASE_PARAMS = {
    "RQALPHA_ENTRY_DRAWDOWN": 0.03,
    "RQALPHA_RESET_DRAWDOWN": 0.015,
    "RQALPHA_LOOKBACK": 20,
    "RQALPHA_BULL_GAP": 0.02,
    "RQALPHA_SUPER_BULL_GAP": 0.04,
    "RQALPHA_SWITCH_DELTA": 0.015,
    "RQALPHA_LEVEL1_DRAWDOWN": 0.02,
    "RQALPHA_LEVEL2_DRAWDOWN": 0.05,
    "RQALPHA_LEVEL3_DRAWDOWN": 0.08,
    "RQALPHA_LEVEL1_TARGET": 0.50,
    "RQALPHA_LEVEL2_TARGET": 0.80,
    "RQALPHA_LEVEL3_TARGET": 1.00,
    "RQALPHA_TREND_DOWN_CAP": 0.70,
    "RQALPHA_RECOVERY_GAP": 0.0,
    "RQALPHA_RECOVERY_CAP": 0.70,
    "RQALPHA_REBOUND_DRAWDOWN": 1.0,
    "RQALPHA_REBOUND_FAST_RETURN": 1.0,
    "RQALPHA_DEEP_DRAWDOWN": 1.0,
    "RQALPHA_DEEP_RECOVERY_GAP": -1.0,
    "RQALPHA_DEEP_DRAWDOWN_CAP": 0.70,
    "RQALPHA_PROFIT_TRIM_1": 0.75,
    "RQALPHA_PROFIT_TRIM_2": 1.10,
    "RQALPHA_PROFIT_EXIT": 1.80,
    "RQALPHA_TRAILING_ACTIVATION": 0.90,
    "RQALPHA_TRAILING_STOP": 0.26,
    "RQALPHA_TRAILING_EXIT": 0.38,
    "RQALPHA_LOCK1_PROFIT": 0.85,
    "RQALPHA_LOCK1_CAP": 0.92,
    "RQALPHA_LOCK2_PROFIT": 1.20,
    "RQALPHA_LOCK2_CAP": 0.78,
    "RQALPHA_LOCK3_PROFIT": 1.60,
    "RQALPHA_LOCK3_CAP": 0.50,
}


TARGET_COMBOS = [
    ("base_stock_cyb50", {"RQALPHA_PRIMARY": "300059.XSHE", "RQALPHA_BACKUP": "159949.XSHE"}),
    ("eastmoney_cyb", {"RQALPHA_PRIMARY": "300059.XSHE", "RQALPHA_BACKUP": "159915.XSHE"}),
    (
        "eastmoney_cyb_chip",
        {"RQALPHA_PRIMARY": "300059.XSHE", "RQALPHA_BACKUP": "159915.XSHE", "RQALPHA_TERTIARY": "159995.XSHE"},
    ),
    (
        "eastmoney_cyb_secu",
        {"RQALPHA_PRIMARY": "300059.XSHE", "RQALPHA_BACKUP": "159915.XSHE", "RQALPHA_TERTIARY": "512880.XSHG"},
    ),
    ("cyb_cyb50", {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "159949.XSHE"}),
    (
        "cyb_cyb50_chip",
        {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "159949.XSHE", "RQALPHA_TERTIARY": "159995.XSHE"},
    ),
    (
        "cyb_cyb50_secu",
        {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "159949.XSHE", "RQALPHA_TERTIARY": "512880.XSHG"},
    ),
    ("cyb_chip", {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "159995.XSHE"}),
    (
        "cyb_chip_half",
        {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "159995.XSHE", "RQALPHA_TERTIARY": "512480.XSHG"},
    ),
    (
        "cyb_chip_ai",
        {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "159995.XSHE", "RQALPHA_TERTIARY": "159819.XSHE"},
    ),
    ("cyb_secu", {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "512880.XSHG"}),
    (
        "cyb_secu_chip",
        {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "512880.XSHG", "RQALPHA_TERTIARY": "159995.XSHE"},
    ),
    (
        "cyb_secu_kc50",
        {"RQALPHA_PRIMARY": "159915.XSHE", "RQALPHA_BACKUP": "512880.XSHG", "RQALPHA_TERTIARY": "588000.XSHG"},
    ),
    ("secu_chip", {"RQALPHA_PRIMARY": "512880.XSHG", "RQALPHA_BACKUP": "159995.XSHE"}),
    (
        "secu_chip_half",
        {"RQALPHA_PRIMARY": "512880.XSHG", "RQALPHA_BACKUP": "159995.XSHE", "RQALPHA_TERTIARY": "512480.XSHG"},
    ),
    (
        "secu_cyb_chip",
        {"RQALPHA_PRIMARY": "512880.XSHG", "RQALPHA_BACKUP": "159915.XSHE", "RQALPHA_TERTIARY": "159995.XSHE"},
    ),
    ("chip_half", {"RQALPHA_PRIMARY": "159995.XSHE", "RQALPHA_BACKUP": "512480.XSHG"}),
    (
        "chip_half_cyb",
        {"RQALPHA_PRIMARY": "159995.XSHE", "RQALPHA_BACKUP": "512480.XSHG", "RQALPHA_TERTIARY": "159915.XSHE"},
    ),
    (
        "chip_ai_cyb",
        {"RQALPHA_PRIMARY": "159995.XSHE", "RQALPHA_BACKUP": "159819.XSHE", "RQALPHA_TERTIARY": "159915.XSHE"},
    ),
    (
        "kc50_chip_cyb",
        {"RQALPHA_PRIMARY": "588000.XSHG", "RQALPHA_BACKUP": "159995.XSHE", "RQALPHA_TERTIARY": "159915.XSHE"},
    ),
    (
        "kc50_ai_cyb",
        {"RQALPHA_PRIMARY": "588000.XSHG", "RQALPHA_BACKUP": "159819.XSHE", "RQALPHA_TERTIARY": "159915.XSHE"},
    ),
    (
        "ai_chip_cyb",
        {"RQALPHA_PRIMARY": "159819.XSHE", "RQALPHA_BACKUP": "159995.XSHE", "RQALPHA_TERTIARY": "159915.XSHE"},
    ),
]


def main() -> None:
    results = []
    for idx, (label, delta) in enumerate(TARGET_COMBOS, 1):
        params = dict(BASE_PARAMS)
        params.update(delta)
        rows = [run_one_year(year, env_params(params)) for year in YEARS]
        item = aggregate(label, params, rows)
        results.append(item)
        agg = item["aggregate"]
        print(
            idx,
            label,
            f"avg={agg['avg_total_return']:.2%}",
            f"best={agg['best_year_return']:.2%}",
            f"worst={agg['worst_year_return']:.2%}",
            f"mdd={agg['worst_max_drawdown']:.2%}",
            flush=True,
        )

    results.sort(key=lambda item: item["aggregate"]["avg_total_return"], reverse=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT}")
    print(json.dumps(results[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
