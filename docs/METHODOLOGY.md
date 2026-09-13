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
- $R^2$ is the in-sample fraction of variation in $x_t$ explained by the regressors. High $R^2$ does not prove stationarity, cointegration, causality, or profitability.

The regression is directional, so reversing assets 1 and 2 can change the coefficients and signals.

## Statistical eligibility tests

### Augmented Dickey–Fuller test

This test is applied to the market-adjusted residual:

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

## Entry and exit rules

Entry requires the z-score threshold and all enabled tests. It is blocked when cointegration, residual stationarity, fit, or half-life requirements fail.

A position closes on the first applicable event:

- final available date;
- stop loss;
- maximum holding time, $\lceil kh\rceil$ trading periods;
- relationship failure from low live $R^2$ or excessive hedge-ratio change;
- mean reversion to the exit z-score.

No same-close reversal is allowed.

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

Pairs are sorted using eligibility, Sharpe ratio, quality score, and absolute current signal. If the same period is used for selection and reported performance, the result contains selection bias.

## Limitations

- **Multiple testing:** screening thousands of pairs creates false discoveries. Use false-discovery-rate control and untouched test data.
- **Survivorship bias:** current S&P 500 members are not the historical member set.
- **Data quality:** Yahoo Finance is convenient research data, not execution-grade data.
- **Parameter uncertainty:** coefficients, half-life, p-values, and thresholds are estimates.
- **Structural breaks:** cointegration may vanish after changes in firms or market regimes.
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
