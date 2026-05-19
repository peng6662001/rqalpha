# Production Strategies

## Overview

This file records the current production-oriented strategy copies under `strategies/stock_selection/`.

## 1. Balanced Deployment

- File: `strategy_market_drawdown_scale_in_core_satellite_512000_159949_prod.py`
- Style: balanced-plus / stable core with tactical upside sleeve
- Benchmark: `000300.XSHG`
- Targets: `512000.XSHG` core + `159949.XSHE` satellite

Core behavior:

- keep the `512000.XSHG` timed-trend engine as the main low-drawdown core
- add a capped `159949.XSHE` satellite sleeve only when benchmark trend and target trend are both supportive
- use a stricter satellite momentum filter plus deep-drawdown attenuation to keep the high-beta sleeve quieter in stressed tape
- limit the satellite to about `15%` max allocation
- preserve the core family drawdown profile while lifting upside in stronger years

Current profile from 2020-2025 research:

- average total return: about `7.77%`
- average excess annual return: about `6.10%`
- worst max drawdown: about `8.96%`
- positive years: `5/6`

Suggested run command:

```bash
rqalpha run -f strategies/stock_selection/strategy_market_drawdown_scale_in_core_satellite_512000_159949_prod.py -s 2020-01-01 -e 2025-12-31 --account stock 1000000 --benchmark 000300.XSHG
```

Suggested config:

- `strategies/stock_selection/config_core_satellite_512000_159949_prod.yml`

Run with config:

```bash
rqalpha run -c strategies/stock_selection/config_core_satellite_512000_159949_prod.yml
```

## 2. Low-Drawdown Stable Deployment

- File: `strategy_market_drawdown_scale_in_stable_plus_timed_trend_512000_prod.py`
- Style: balanced / low-drawdown deployment
- Benchmark: `000300.XSHG`
- Target: `512000.XSHG`

Core behavior:

- scale in as HS300 drawdown deepens
- use profit ladder for staged exits
- use timed-trend trimming for positions that stay weak for too long
- keep downside materially lower than the aggressive family

Current profile from 2020-2025 research:

- average total return: about `6.88%`
- average excess annual return: about `5.32%`
- worst max drawdown: about `8.96%`
- positive years: `5/6`

Suggested run command:

```bash
rqalpha run -f strategies/stock_selection/strategy_market_drawdown_scale_in_stable_plus_timed_trend_512000_prod.py -s 2020-01-01 -e 2025-12-31 --account stock 1000000 --benchmark 000300.XSHG
```

Suggested config:

- `strategies/stock_selection/config_stable_plus_timed_trend_512000_prod.yml`

Run with config:

```bash
rqalpha run -c strategies/stock_selection/config_stable_plus_timed_trend_512000_prod.yml
```

## 3. Maximum Return

- File: `strategy_market_drawdown_scale_in_aggressive_plus_prod.py`
- Style: high-beta / high-return deployment
- Benchmark: `000300.XSHG`
- Target: `159949.XSHE`

Core behavior:

- arm early on benchmark drawdown
- scale into a high-beta target aggressively
- use staged profit caps and wide trailing exits
- accept materially larger drawdown for much stronger upside

Current profile from 2020-2025 research:

- average total return: about `25.17%`
- average excess annual return: about `19.93%`
- worst max drawdown: about `25.07%`
- positive years: `4/6`

Suggested run command:

```bash
rqalpha run -f strategies/stock_selection/strategy_market_drawdown_scale_in_aggressive_plus_prod.py -s 2020-01-01 -e 2025-12-31 --account stock 1000000 --benchmark 000300.XSHG
```

Suggested config:

- `strategies/stock_selection/config_aggressive_plus_prod.yml`

Run with config:

```bash
rqalpha run -c strategies/stock_selection/config_aggressive_plus_prod.yml
```

## 4. Capital Parking Reference

- File: `strategy_market_drawdown_scale_in_cash_fortress_defensive.py`
- Style: capital-preservation / near-flat defensive reference
- Benchmark: `000300.XSHG`
- Targets: `512000.XSHG` attack sleeve + `510300.XSHG` defensive sleeve

Core behavior:

- use very strict benchmark and target trend gates before allowing the attack sleeve to deploy
- when the regime weakens, shrink the attack sleeve quickly and allow only a small `510300.XSHG` parking allocation
- use earlier timed-trend and trailing reductions than the balanced family
- prioritize avoiding bad years over capturing strong years

Current profile from 2020-2025 research:

- average total return: about `2.06%`
- average excess annual return: about `1.22%`
- worst year return: about `-1.08%`
- worst max drawdown: about `2.29%`
- positive years: `4/6`

Suggested run command:

```bash
rqalpha run -f strategies/stock_selection/strategy_market_drawdown_scale_in_cash_fortress_defensive.py -s 2020-01-01 -e 2025-12-31 --account stock 1000000 --benchmark 000300.XSHG
```

Suggested config:

- `strategies/stock_selection/config_cash_fortress_defensive.yml`

Run with config:

```bash
rqalpha run -c strategies/stock_selection/config_cash_fortress_defensive.yml
```

## 5. Capital Parking Upgrade

- File: `strategy_market_drawdown_scale_in_cash_fortress_parking_511260.py`
- Style: capital-preservation plus parking-yield upgrade
- Benchmark: `000300.XSHG`
- Targets: `512000.XSHG` attack sleeve + `510300.XSHG` recovery sleeve + `511260.XSHG` parking sleeve

Core behavior:

- keep the same strict cash-fortress regime filter for the attack sleeve
- use `510300.XSHG` only as a small recovery-stage bridge
- replace most idle cash with `511260.XSHG` as the default low-volatility parking asset
- let the parking sleeve carry the portfolio when the attack engine is mostly sidelined

Current profile from 2020-2025 research:

- average total return: about `4.93%`
- average excess annual return: about `4.29%`
- worst year return: about `0.08%`
- worst max drawdown: about `3.10%`
- positive years: `6/6`

Suggested run command:

```bash
rqalpha run -f strategies/stock_selection/strategy_market_drawdown_scale_in_cash_fortress_parking_511260.py -s 2020-01-01 -e 2025-12-31 --account stock 1000000 --benchmark 000300.XSHG
```

Suggested config:

- `strategies/stock_selection/config_cash_fortress_parking_511260.yml`

Run with config:

```bash
rqalpha run -f strategies/stock_selection/strategy_market_drawdown_scale_in_cash_fortress_parking_511260.py -s 2020-01-01 -e 2025-12-31 --account stock 1000000 --benchmark 000300.XSHG
```

## Selection Rule

- choose `strategy_market_drawdown_scale_in_core_satellite_512000_159949_prod.py` when you want the current best balanced deployment with a little extra upside
- choose `strategy_market_drawdown_scale_in_stable_plus_timed_trend_512000_prod.py` when you want the cleanest low-drawdown single-target deployment
- choose `strategy_market_drawdown_scale_in_aggressive_plus_prod.py` when maximizing upside matters more than path volatility
- choose `strategy_market_drawdown_scale_in_cash_fortress_defensive.py` only when your first priority is getting the yearly loss profile close to flat and you accept very muted upside
- choose `strategy_market_drawdown_scale_in_cash_fortress_parking_511260.py` when you want the current best “low loss plus low-risk carry” version and you prefer steady positive years over upside bursts
