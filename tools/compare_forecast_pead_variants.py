from __future__ import annotations

import json
import os
from pathlib import Path

from rqalpha import run_file


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = "strategies/stock_selection/strategy_forecast_pead_luezeng_core.py"
OUTPUT = ROOT / ".temp" / "public_pead" / "forecast_pead_variant_compare_2020_2025.json"

VARIANTS = [
    {"name": "blend_luezeng_yuzeng_niukui", "types": ["略增", "预增", "扭亏"]},
    {"name": "single_luezeng", "types": ["略增"]},
]


def base_config() -> dict:
    return {
        "base": {
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
            "frequency": "1d",
            "matching_type": "current_bar",
            "benchmark": "000300.XSHG",
            "accounts": {"stock": 1000000},
            "data_bundle_path": str(Path.home() / ".rqalpha" / "bundle"),
        },
        "extra": {"log_level": "error"},
        "mod": {
            "sys_progress": {"enabled": False},
            "sys_analyser": {"enabled": True},
        },
    }


def run_variant(name: str, types: list[str]) -> dict:
    old_env = os.environ.get("RQALPHA_PUBLIC_PEAD_TYPES")
    os.environ["RQALPHA_PUBLIC_PEAD_TYPES"] = ",".join(types)
    try:
        result = run_file(STRATEGY, config=base_config())
    finally:
        if old_env is None:
            os.environ.pop("RQALPHA_PUBLIC_PEAD_TYPES", None)
        else:
            os.environ["RQALPHA_PUBLIC_PEAD_TYPES"] = old_env
    summary = result.get("sys_analyser", {}).get("summary", {})
    return {
        "variant": name,
        "types": types,
        "total_returns": summary.get("total_returns"),
        "annualized_returns": summary.get("annualized_returns"),
        "benchmark_annualized_returns": summary.get("benchmark_annualized_returns"),
        "alpha": summary.get("alpha"),
        "beta": summary.get("beta"),
        "sharpe": summary.get("sharpe"),
        "max_drawdown": summary.get("max_drawdown"),
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [run_variant(item["name"], item["types"]) for item in VARIANTS]
    payload = {"variants": rows}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
