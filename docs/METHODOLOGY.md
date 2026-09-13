# Statistical Methodology

## Research objective

The framework searches for two assets whose individual prices may be non-stationary but whose estimated relative-value spread is stationary. The hypothesis is that temporary deviations from this equilibrium may revert, creating a market-hedged signal.

This is a conditional mean-reversion model—not a claim that either asset price independently mean-reverts.

## Data and transformation

Let $P_{1,t}$ and $P_{2,t}$ be adjusted closing prices and $M_t$ a market factor. Only positive, overlapping observations are retained. The model uses log prices:

$$
x_t=\log(P_{1,t}), \quad y_t=\log(P_{2,t}), \quad m_t=\log(M_t).
$$

Log prices reduce scale differences and express the relationship multiplicatively. They do not guarantee stationarity or normal residuals.

## Market-adjusted spread

Ordinary least squares estimates this model inside each rolling window:

$$
x_t=\alpha+\beta_p y_t+\beta_m m_t+\varepsilon_t.
$$

The estimated residual is the spread:

$$
\hat\varepsilon_t=x_t-\hat\alpha-\hat\beta_p y_t-\hat\beta_m m_t.
$$

- $\hat\beta_p$ is the relative exposure to asset 2.
- $\hat\beta_m$ is the exposure to the market factor.
- $R^2$ is the in-sample fraction of variation in $x_t$ explained by the regressors. High $R^2$ does not prove stationarity, causality, or profitability.

The regression is directional, so reversing assets 1 and 2 can change the coefficients and signals.

## Augmented Dickey–Fuller (ADF) residual stationarity test

The Augmented Dickey–Fuller (ADF) test is applied to the market-adjusted residual:

$$
H_0:\text{ the residual has a unit root}, \qquad
H_1:\text{ the residual is stationary}.
$$

The default rule `adf_pvalue <= 0.05` rejects the unit-root null. A low p-value supports stationarity inside that window but does not show that the relationship will continue.

## Mean-reversion speed

The residual is approximated with

$$
\Delta\varepsilon_t=a+\lambda\varepsilon_{t-1}+u_t.
$$

When $\hat\lambda<0$, estimated half-life is

$$
h=-\frac{\ln 2}{\hat\lambda}.
$$

This estimates the trading periods needed for a deviation to decay by half. A non-negative or near-zero reversion speed produces an infinite or unstable half-life.

## Hedge-ratio stability

The diagnostic analyzer refits the regression over fixed-length rolling windows. For each hedge coefficient it calculates

$$
S_\beta=\frac{\mathrm{SD}(\hat\beta_t)}{|\mathrm{Mean}(\hat\beta_t)|}.
$$

Lower values mean greater relative stability. The code describes values below `0.25` as stable, `0.25` to below `0.50` as somewhat stable, and at least `0.50` as unstable. A near-zero mean produces an infinite ratio. Pair-beta and market-beta stability are evaluated separately.

## Standardized signal

Using the training residual mean $\mu_\varepsilon$ and sample standard deviation $s_\varepsilon$:

$$
z_t=\frac{\hat\varepsilon_t-\mu_\varepsilon}{s_\varepsilon}.
$$

| Signal | Statistical meaning | Position |
| --- | --- | --- |
| $z_t \ge z_{entry}$ | Spread is unusually high | Short spread |
| $z_t \le -z_{entry}$ | Spread is unusually low | Long spread |
| Near zero | Spread is near estimated equilibrium | Flat or exit |

A z-score of 2 means the spread is two training-window standard deviations above its mean. It does not imply an exact tail probability unless the residual distribution is stable and approximately normal.

## Portfolio construction

A unit long-spread portfolio has raw weights

$$
\mathbf{w}_{raw}=[1,-\hat\beta_p,-\hat\beta_m].
$$

A short spread multiplies them by $-1$. Weights are normalized to gross exposure $G$:

$$
\mathbf{w}=\mathbf{w}_{raw}\frac{G}{\sum_i|w_{raw,i}|}.
$$

This controls total absolute notional. It does not guarantee dollar or beta neutrality when estimates are unstable. Without daily rebalancing, weights drift with returns; with rebalancing, target weights are restored and turnover costs are charged.

## Walk-forward timing

For signal date $t$:

1. Fit using observations through $t-1$ only.
2. Calculate the signal using the close at $t$.
3. Execute at the next available close, $t+1$.

This prevents direct look-ahead bias. It does not model intraday execution or guarantee a fill at the recorded close. During a trade, the entry model can remain frozen for the mean-reversion exit while a fresh model monitors relationship deterioration.

The training set covers a trailing number of calendar years but must contain the configured minimum number of observations. The final available price date cannot be a signal date because no later close exists for execution.

## Analyzer workflows

`PairTradingAnalyzer` produces the latest model diagnostics, rolling-beta analysis, plots, and current signal. `run_daily_walk_forward_backtest_from_prices` performs the historical simulation. `analyze_pair_strategy_from_prices` runs both and joins their outputs into one screener summary.

The current diagnostic and backtest are therefore separate calculations. The latest analysis uses the previous available date as its training end, while every backtest day creates its own prior-data-only window.

## Entry and exit rules

Entry requires the z-score threshold and all enabled tests. Filters are checked in this order: ADF residual stationarity, minimum $R^2$, and finite positive half-life when the holding rule is enabled. The daily output records the first failed rule.

A position closes on the first applicable event:

- final available date;
- stop loss;
- maximum holding time, $\lceil kh\rceil$ trading periods;
- relationship failure from low live $R^2$ or excessive hedge-ratio change;
- mean reversion to the exit z-score.

No same-close reversal is allowed.

### Saved versus refitted model

With `use_saved_model_during_trade=True`, the entry model is frozen for the z-score used to decide mean reversion. A fresh daily model still monitors relationship failure. With the option disabled, the latest rolling model also controls the mean-reversion decision.

The maximum holding period is fixed when a trade opens:

$$
H=\max(1,\lceil k h_{entry}\rceil).
$$

### Relationship-break calculation

The strategy can exit when live $R^2$ falls below its floor or when either beta changes too far from its entry value. Relative beta change is

$$
D_\beta=\frac{|\hat\beta_{live}-\hat\beta_{entry}|}{\max(|\hat\beta_{entry}|,0.10)}.
$$

The denominator floor prevents a very small entry beta from producing a mechanically enormous ratio.

### Return and weight evolution

Daily portfolio return uses weights held before that day's price movement:

$$
r_{p,t}=\mathbf w_{t-1}^{\mathsf T}\mathbf r_t.
$$

Without daily rebalancing, weights drift according to

$$
w_{i,t}=w_{i,t-1}\frac{1+r_{i,t}}{1+r_{p,t}}.
$$

With rebalancing, target weights are restored and costs are charged. The simulation tracks gross equity before costs and net equity after costs separately.

### Transaction costs

For weight change $\Delta\mathbf w=\mathbf w_{new}-\mathbf w_{old}$,

$$
C_t=c_b\sum_i(\Delta w_i)^+ + c_s\sum_i(-\Delta w_i)^+.
$$

The monetary cost equals current equity multiplied by $C_t$. Costs apply at entry, exit, and optional rebalancing. This omits nonlinear impact, borrowing, funding, taxes, and failed execution.

### Exact exit priority

The first applicable rule wins: final date, stop loss, maximum holding period, low live $R^2$, excessive pair-beta change, excessive market-beta change, then mean reversion.

## Interpreting the outputs

| Output | Meaning | Main caution |
| --- | --- | --- |
| `good_pair` | Passed configured statistical and stability rules | Not a prediction of profit |
| Quality score | Heuristic ranking of diagnostics | Not a calibrated probability |
| Total return | Final net equity minus initial equity | Depends on exposure and test length |
| Annualized return | Geometrically annualized net return | Unstable for short samples |
| Annualized volatility | Daily volatility scaled by $\sqrt{252}$ | Misses tails and serial dependence |
| Sharpe ratio | Annualized mean daily return divided by volatility | Assumes zero risk-free rate; selection can inflate it |
| Maximum drawdown | Worst historical peak-to-trough loss | Describes only the observed path |
| Win rate | Fraction of completed trades with positive return | Ignores win and loss sizes |
| Profit factor | Gross wins divided by gross losses | Can be dominated by a few trades |
| Cost drag | Gross equity minus net equity | Does not include every real trading friction |
| Exposure days | Days with a non-zero position | Measures capital usage, not risk alone |

No result is meaningful without its sample dates, trade count, turnover, costs, and out-of-sample status. A high Sharpe ratio from five selected trades is weak evidence.

The Sharpe ratio uses mean daily net return divided by sample daily volatility, multiplied by $\sqrt{252}$, with a zero risk-free rate. Profit factor is total positive completed-trade return divided by the absolute total negative completed-trade return. It can be infinite or unstable when very few losses occur.

Daily records include live diagnostics, signal and decision z-scores, entry blocks, positions, drifting weights, returns, turnover, costs, and equity. Trade records include entry parameters, weights, costs, holding periods, returns, and exit reasons.

## `good_pair` and quality score

`good_pair=True` requires the enabled ADF residual-stationarity threshold, half-life range, pair-beta stability, market-beta stability, and minimum $R^2$. It does not require a current signal and does not use return, Sharpe ratio, win rate, or drawdown. It means the latest model passed structural filters—not that the strategy is profitable.

The quality score is a fixed heuristic from 0 to 100:

| Component | Maximum points | Code behavior |
| --- | ---: | --- |
| Augmented Dickey–Fuller (ADF) p-value | 25 | Linear improvement from threshold to zero |
| $R^2$ | 15 | `15 × R²`, clipped to 0–15 |
| Pair-beta stability | 10 | Linear improvement from limit to zero |
| Market-beta stability | 10 | Linear improvement from limit to zero |
| Half-life | 10 | All points when inside range |
| Current absolute z-score | 5 | All points when entry threshold is reached |

This score is not a probability, expected return, or statistically calibrated measure.

The six components total 75 raw points. The implementation multiplies the result by `100 / 75`, preserving their relative weights while reporting a 0–100 score.

## Current-signal meaning

The signal function returns `NO DATA` for a non-finite z-score and `NO TRADE` below the entry threshold. Otherwise it reports normalized long and short legs. A displayed trade direction means the z-score threshold was reached; it does not by itself guarantee that all `good_pair` filters passed.

Pairs are sorted using eligibility, Sharpe ratio, quality score, and absolute current signal. If the same period is used for selection and reported performance, the result contains selection bias.

## Limitations

- **Multiple testing:** screening thousands of pairs creates false discoveries. Use false-discovery-rate control and untouched test data.
- **Survivorship bias:** current S&P 500 members are not the historical member set.
- **Data quality:** Yahoo Finance is convenient research data, not execution-grade data.
- **Parameter uncertainty:** coefficients, half-life, p-values, and thresholds are estimates.
- **Structural breaks:** the estimated spread relationship may vanish after changes in firms or market regimes.
- **Trading frictions:** borrow availability, bid–ask spread, impact, funding, taxes, and short-sale constraints are incomplete.
- **Dependence:** overlapping windows and repeated trades weaken independent-observation assumptions.
- **Model risk:** persistent regressors, heteroskedastic residuals, and nonlinear relationships can violate ordinary least-squares assumptions.

## Stronger validation protocol

1. Define hypotheses and thresholds before viewing test results.
2. Split time into training, validation, and untouched test periods.
3. Select pairs using only training and validation data.
4. Correct for multiple testing across the screened universe.
5. Estimate confidence intervals with a time-series-aware bootstrap.
6. Stress-test costs, execution delays, and parameter choices.
7. Compare against simple benchmarks and report failed pairs.
8. Paper trade before considering capital deployment.

The framework provides evidence about historical statistical relationships. It does not establish causality or guarantee future returns.
