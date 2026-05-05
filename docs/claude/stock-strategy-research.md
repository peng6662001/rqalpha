# Stock Strategy Research

## 2026-04-30

### Universe and Rebalance Baseline

- Shared stock-selection framework is in [strategies/stock_selection/ss_common.py](/home/dom/Stock/rqalpha/strategies/stock_selection/ss_common.py).
- Current baseline uses a dynamic pool of 300 stocks from `all_instruments("CS")`.
- The pool is filtered to stocks that remain listed through the full backtest end date.
- Rebalance frequency is daily.

### Current Stability Winner

- Under `300-stock pool + daily rebalance`, the most stable strategy is `strategy_low_volatility`.
- Strategy file: [strategies/stock_selection/strategy_low_volatility.py](/home/dom/Stock/rqalpha/strategies/stock_selection/strategy_low_volatility.py).
- Cross-period windows:
  - `2014-01-01 -> 2016-12-31`
  - `2017-01-01 -> 2019-12-31`
  - `2020-01-01 -> 2022-12-31`
  - `2023-01-01 -> 2024-12-31`

### Parameter Tuning Result

- Tuning result file: [low_volatility_tuning.json](/home/dom/Stock/rqalpha/.temp/low_volatility_tuning.json).
- Round-1 best stability setting:
  - `top_n = 5`
  - `lookback = 80`
  - `cash_buffer = 0.05`
- Stability metrics:
  - `positive_period_ratio = 1.0`
  - `avg_calmar = 0.9775`
  - `min_calmar = 0.0983`
  - `avg_sharpe = 0.6895`
  - `worst_drawdown = 19.43%`

### Round-2 Tuning Result

- Round-2 result file: [low_volatility_tuning_round2.json](/home/dom/Stock/rqalpha/.temp/low_volatility_tuning_round2.json).
- Round-2 tested:
  - `top_n = 5`, `lookback = 40/60/80/100`, `cash_buffer = 0.05`
  - `top_n = 5`, `lookback = 80`, `cash_buffer = 0.02/0.08/0.10`
- Updated best setting:
  - `top_n = 5`
  - `lookback = 80`
  - `cash_buffer = 0.02`
- Updated stability metrics:
  - `positive_period_ratio = 1.0`
  - `avg_calmar = 1.1207`
  - `min_calmar = 0.1220`
  - `avg_sharpe = 0.7477`
  - `worst_drawdown = 19.82%`

### Round-2 Findings

- Lowering `cash_buffer` from `0.05` to `0.02` improved average Calmar, average Sharpe, and average annualized return.
- Raising `cash_buffer` to `0.08` or `0.10` reduced return and slightly improved worst drawdown, but did not improve overall stability ranking.
- `lookback = 40/60/80/100` produced nearly identical results under the current low-volatility ranking logic, so cash buffer is the more important tuning axis here.

### Trend Filter Experiment

- `strategy_low_volatility` now supports optional trend filters through env vars.
- Tested filters:
  - `off`
  - `positive_return`
  - `above_ma`
  - `both`
- Result:
  - `off` remained best.
  - All tested trend filters reduced `positive_period_ratio` from `1.0` to `0.5`.
  - Trend filtering did not improve stability under the current 300-stock daily-rebalance setup.

### Volatility Window Experiment

- Volatility window tuning script: [tools/tune_low_volatility_vol_window.py](/home/dom/Stock/rqalpha/tools/tune_low_volatility_vol_window.py).
- Result file: [low_volatility_vol_window.json](/home/dom/Stock/rqalpha/.temp/low_volatility_vol_window.json).
- Tested `vol_window = 20/40/60/80` with:
  - `top_n = 5`
  - `cash_buffer = 0.02`
  - `trend_filter = off`
- Best tested window remained `vol_window = 60`.
- Metrics for `vol_window = 60`:
  - `positive_period_ratio = 1.0`
  - `avg_calmar = 0.9698`
  - `min_calmar = 0.1097`
  - `avg_sharpe = 0.6784`
  - `worst_drawdown = 20.00%`
- `vol_window = 20` and `40` introduced losing periods.
- `vol_window = 80` stayed profitable across all periods but reduced average Calmar, average Sharpe, and average annualized return.

### Rebalance Frequency Experiment

- Rebalance frequency tuning script: [tools/tune_low_volatility_rebalance_frequency.py](/home/dom/Stock/rqalpha/tools/tune_low_volatility_rebalance_frequency.py).
- Result file: [low_volatility_rebalance_frequency.json](/home/dom/Stock/rqalpha/.temp/low_volatility_rebalance_frequency.json).
- Tested:
  - `daily`
  - `weekly`
  - `biweekly`
  - `monthly`
- Best tested frequency remained `daily`.
- Metrics for `daily`:
  - `positive_period_ratio = 1.0`
  - `avg_calmar = 1.0544`
  - `min_calmar = 0.1012`
  - `avg_sharpe = 0.6926`
  - `worst_drawdown = 19.55%`
- `weekly`, `biweekly`, and `monthly` all reduced `positive_period_ratio` to `0.75`.
- Lower rebalance frequency did not improve stability for the current low-volatility strategy.

### Hold Buffer Experiment

- Hold buffer tuning script: [tools/tune_low_volatility_hold_buffer.py](/home/dom/Stock/rqalpha/tools/tune_low_volatility_hold_buffer.py).
- Result file: [low_volatility_hold_buffer.json](/home/dom/Stock/rqalpha/.temp/low_volatility_hold_buffer.json).
- Tested:
  - disabled
  - `hold_buffer_rank = 10`
  - `hold_buffer_rank = 15`
  - `hold_buffer_rank = 20`
- Best tested setting: `hold_buffer_rank = 20`.
- Metrics for `hold_buffer_rank = 20`:
  - `positive_period_ratio = 1.0`
  - `avg_calmar = 1.2135`
  - `min_calmar = 0.1196`
  - `avg_sharpe = 0.8330`
  - `worst_drawdown = 18.24%`
  - `avg_annualized_returns = 14.74%`
- This is the strongest stability improvement so far.
- Updated best known setting:
  - `top_n = 5`
  - `cash_buffer = 0.02`
  - `vol_window = 60`
  - `trend_filter = off`
  - `rebalance_frequency = daily`
  - `hold_buffer_rank = 20`

### Hold Buffer Fine Tuning

- Fine tuning script: [tools/tune_low_volatility_hold_buffer_fine.py](/home/dom/Stock/rqalpha/tools/tune_low_volatility_hold_buffer_fine.py).
- Result file: [low_volatility_hold_buffer_fine.json](/home/dom/Stock/rqalpha/.temp/low_volatility_hold_buffer_fine.json).
- Tested:
  - `hold_buffer_rank = 18`
  - `hold_buffer_rank = 20`
  - `hold_buffer_rank = 25`
  - `hold_buffer_rank = 30`
- Best tested setting: `hold_buffer_rank = 18`.
- Metrics for `hold_buffer_rank = 18`:
  - `positive_period_ratio = 1.0`
  - `avg_calmar = 1.2298`
  - `min_calmar = 0.1768`
  - `avg_sharpe = 0.8136`
  - `worst_drawdown = 18.50%`
  - `avg_annualized_returns = 14.77%`
  - `avg_annualized_turnover = 12.54`
  - `avg_trade_count = 1096.25`
- Compared with `hold_buffer_rank = 20`, `18` improved average Calmar, minimum Calmar, and average annualized return.
- Tradeoff: `18` had slightly higher turnover than `20` (`12.54` vs `12.21`) and about `44` more trades per test window on average.
- Updated best known setting:
  - `top_n = 5`
  - `cash_buffer = 0.02`
  - `vol_window = 60`
  - `trend_filter = off`
  - `rebalance_frequency = daily`
  - `hold_buffer_rank = 18`

### Key Findings

- Increasing holdings to `top_n = 10` or `15` improves the strongest bull period but reduces stability.
- `top_n = 5` is the only tested setting that stayed profitable in all 4 periods.
- A hold buffer is useful: keeping current holdings while they remain in the top 18 improved stability and return.
- `strategy_mean_reversion_5` deteriorated sharply after switching to the 300-stock daily-rebalance framework and is no longer a stability candidate.
