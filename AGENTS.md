# AGENTS.md

RQAlpha is an algorithmic trading system for quantitative trading with backtesting and live trading capabilities.

**License**: Non-commercial use only (Apache 2.0). Commercial use requires authorization from Ricequant.

## Quick Commands

```bash
# Run backtest
rqalpha run -f strategy.py -s 2014-01-01 -e 2016-01-01 --account stock 100000

# With RQData connection
rqalpha run --rqdatac-uri tcp://user:password@host:port -f strategy.py -s 2014-01-01 -e 2016-01-01 --account stock 100000

# Download bundle data
rqalpha download-bundle

# Update bundle
rqalpha update-bundle --rqdatac-uri tcp://user:password@host:port

# Generate examples
rqalpha examples -d ./examples

# Run tests
pytest
pytest tests/unittest/
pytest tests/integration_tests/
```

## Project-Specific Rules

### When Writing Strategies

1. **Always consult documentation first**: Read `docs/source/intro/tutorial.rst` and `docs/source/api/base_api.rst` before writing strategies
2. **Use correct API signatures**: Check `docs/source/api/base_api.rst` for function parameters and return types
3. **Follow strategy lifecycle**: Implement `init()`, `before_trading()`, `handle_bar()`, `after_trading()` in correct order
4. **Stock code format**: Always use format like "000001.XSHE" (code + exchange)
5. **Date format**: Use 'YYYY-MM-DD' format for dates

### When Debugging/Testing

1. **Write minimal reproducible examples**: Create smallest possible strategy that reproduces the bug
2. **Use logger, not print**: Use `logger.info()` instead of `print()` in strategies
3. **Add assertions**: Verify expected behavior with assertions in test code
4. **Short date ranges**: Use 1-3 month ranges for faster iteration during testing

### Code Modifications

1. **Chinese comments OK**: Domain-specific logic can use Chinese comments
2. **Follow PEP 8**: Standard Python style guide
3. **Test before commit**: Run pytest to ensure tests pass

## Key Architecture Points

- **Environment singleton**: Access via `Environment.get_instance()` - central registry for all components
- **Event-driven**: Mods subscribe to events (BAR, TICK, BEFORE_TRADING, etc.)
- **Mod system**: Extensibility through `AbstractMod` interface
- **Data bundle**: HDF5 format stored in `~/.rqalpha/bundle/`
- **Config hierarchy**: CLI args > strategy `__config__` > config file > defaults

## Detailed Documentation

For detailed information, see:
- **Architecture**: `docs/Codex/architecture.md` - Core components and system design
- **Strategy Writing**: `docs/Codex/strategy-guide.md` - How to write strategies with API reference
- **Bug Reproduction**: `docs/Codex/bug-reproduction.md` - Writing backtests to reproduce bugs
- **Development**: `docs/Codex/development.md` - Development guidelines and debugging

Official documentation:
- Tutorial: `docs/source/intro/tutorial.rst`
- API Reference: `docs/source/api/base_api.rst`
- Examples: `docs/source/intro/examples.rst`

## Recent Research Summary

### Shared Backtest Assumptions

- Unless a task explicitly says otherwise, all stock backtests should include these explicit trading costs:
  - buy commission: `0.03%`
  - buy slippage: `0.10%`
  - sell commission: `0.03%`
  - sell tax: `0.10%`
  - sell slippage: `0.10%`
- Rolled-up totals:
  - buy total: `0.13%`
  - sell total: `0.23%`
  - round-trip total: `0.36%`

### GPU Preference for ML Stock Research

- For the current LightGBM stock-selection workflow, always try GPU first.
- Preferred backend:
  - `device_type = gpu`
  - `gpu_platform_id = 1`
  - `gpu_device_id = 0`
- This maps to `NVIDIA GeForce RTX 3080 Laptop GPU`.
- Only fall back to CPU if GPU training is unavailable or fails.

### Low-Turnover LightGBM Ranker Line

- Main scripts:
  - `tools/search_low_turnover_lgbm_targets.py`
  - `tools/run_low_turnover_lgbm_ranker_report.py`
  - `tools/compare_low_turnover_lgbm_profiles.py`
- Current baseline setup:
  - predict next `10` trading-day open-to-open return
  - rebalance every `5` trading days
  - hold `Top 5`
  - use `Top 20` hold buffer
  - no leverage
- This line passed reverse / shuffle / null-style validation checks well enough to rule out a trivial coding artifact.
- But the strategy is still regime-sensitive rather than fully stable across market cycles.

### Low-Turnover Profile Tests Completed

- `risk` profile:
  - highest return version among current profiles
  - on the current locked score snapshot for `2011-2025`:
    - average return: `54.02%`
    - weakest year: `0.00%`
    - worst max drawdown: `32.86%`
    - average exposure: `92.79%`
- `risk2` profile:
  - softer market-exposure-gated version of `risk`
  - on the current locked score snapshot for `2011-2025`:
    - average return: `42.66%`
    - weakest year: `-8.41%`
    - worst max drawdown: `30.97%`
    - average exposure: `76.77%`
- `steady` profile:
  - current best low-drawdown version
  - on the current locked score snapshot for `2011-2025`:
    - average return: `30.01%`
    - weakest year: `-3.66%`
    - worst max drawdown: `16.77%`
    - average exposure: `51.93%`
- `steady2` profile:
  - tested as a tighter exposure variant of `steady`
  - added with the same `max_vol20 = 0.05` filter, but tighter `ret20` window and `expo_mid = 0.2`
  - it did not improve `steady` and is not the preferred low-drawdown profile
- `risk3` profile:
  - recent-years balanced profile found by targeted exposure-rule search on the locked `2021-2026` snapshot
  - rule summary:
    - `ma60_lo = -0.08`
    - `ma60_hi = -0.02`
    - `ret20_lo = -0.10`
    - `ret20_hi = 0.00`
    - `ret60_lo = -0.04`
    - `ret60_hi = 0.09`
    - `ret5_gate_min = -0.06`
    - `expo_mid = 0.55`
    - `expo_lo = 0.0`
  - on the locked `2021-2026` snapshot:
    - average return: `23.55%`
    - weakest year: `-5.03%`
    - worst max drawdown: `16.85%`
    - average exposure: `61.41%`
  - on the locked full-cycle `2011-2025` snapshot:
    - average return: `32.03%`
    - weakest year: `-2.23%`
    - worst max drawdown: `18.93%`
    - average exposure: `62.80%`
- Current conclusion for this line:
  - `risk2` is a useful middle tradeoff, but not the new best profile
  - `risk3` is the strongest recent-years balanced profile found so far
  - `risk3` also held up as the strongest balanced profile on the current `2011-2025` full-cycle snapshot
  - `steady2` did not improve `steady`
  - `steady` remains a lower-exposure low-drawdown profile
  - `risk` remains the better high-return profile on the current locked snapshot
  - future profile work should compare on a locked score cache, not fresh retrains every time

### Reproducible Comparison Workflow

- GPU-first retraining can drift across separate runs, even with the same code and nominal seeds.
- `tools/run_low_turnover_lgbm_ranker_report.py` and `tools/compare_low_turnover_lgbm_profiles.py` now support `--score-cache`.
- Recommended workflow:
  - first run with GPU and `--score-cache` to generate yearly scored snapshots
  - then reuse the same cache file for all profile comparisons
- Use year-specific cache filenames so a later run for a different year set does not overwrite the old snapshot.
- Current recent-years cache example:
  - `.temp/low_turnover_lgbm_scores_2021_2026.pkl`

### High-Turnover Top1 Rotation Line

- Main scripts:
  - `tools/search_over_50_returns.py`
  - `tools/search_top1_trend_rotation.py`
  - `tools/search_top1_rank_ret_h5_leverage.py`
- This is a different research line from the low-turnover `Top 5` ranker.
- Current best no-leverage setup on `2020-2025`:
  - combo: `rank_ret_h5_none_none`
  - meaning:
    - LightGBM ranker
    - label next `5` trading-day return
    - rebalance every `5` trading days
    - hold `Top 1`
    - no market filter
    - no extra stock filter
    - no leverage
  - average return: `86.32%`
  - weakest year: `11.42%`
  - best year: `183.36%`
  - worst max drawdown: `70.57%`
  - positive years: `6 / 6`
- Full-year breakdown for the current no-leverage winner:
  - `2020`: `183.36%`
  - `2021`: `52.71%`
  - `2022`: `11.42%`
  - `2023`: `121.07%`
  - `2024`: `13.38%`
  - `2025`: `136.01%`
- Important interpretation:
  - this line is much more aggressive than the low-turnover ranker
  - it achieved higher average return without leverage, but with much larger drawdown
  - simple trend / breakout / low-vol filters usually did not beat the raw `rank_ret_h5_none_none` winner
  - shorter `3`-day holding variants produced bigger upside spikes in some years but were less stable overall

### Leverage Note for Top1 Rotation

- The `2020-2025` average return target above `100%` was only reached in leverage simulation, not in the current no-leverage baseline.
- Best leverage search results so far:
  - `fixed_1.25`
    - average return: `100.80%`
    - weakest year: `0.00%`
    - worst max drawdown: `79.68%`
  - `predgap_005_hi_1.35`
    - average return: `101.65%`
    - weakest year: `0.35%`
    - worst max drawdown: `81.90%`
  - `fixed_1.50`
    - average return: `110.13%`
    - weakest year: `-16.61%`
    - worst max drawdown: `86.43%`
- If the user says not to use leverage, do not present these `100%+` results as the active recommendation.
- Under a strict no-leverage constraint, the current best known result remains `rank_ret_h5_none_none` at `86.32%` average return.

### Earlier Stable Cross-Period Strategy Result

- Under the older `300-stock pool + daily rebalance` framework, the most stable non-ML strategy remained `strategies/stock_selection/strategy_low_volatility.py`.
- Best tuned setting from that line:
  - `top_n = 5`
  - `cash_buffer = 0.02`
  - `vol_window = 60`
  - `trend_filter = off`
  - `rebalance_frequency = daily`
  - `hold_buffer_rank = 18`
- Key outcome from that stable line:
  - profitable in all tested periods
  - best known average annualized return there was much lower than the ML ranker line
  - but it was materially more stable and easier to trust across regimes

### Practical Working Conclusions

- When the goal is maximum return, start from the low-turnover LightGBM `risk` profile.
- When the goal is maximum no-leverage return in the newer high-turnover line, start from `rank_ret_h5_none_none`.
- When the goal is a balanced ML profile with materially lower drawdown than `risk` but stronger upside than `steady`, start from `risk3`.
- When the goal is the lowest-exposure ML profile, start from `steady`; for simpler non-ML stability, start from the older low-volatility strategy.
- Do not present the current LightGBM ranker as a universally stable A-share money machine.
- Do not blur the line between no-leverage results and leverage simulations when summarizing research outcomes.
- For future work, record whether a change improves:
  - average return
  - weakest year
  - worst max drawdown
  - average exposure
  - and whether the run used GPU or CPU
