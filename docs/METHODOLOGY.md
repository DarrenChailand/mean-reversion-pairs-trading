# Methodology

The model estimates a market-adjusted log-price relationship:

$$
\log(P_{1,t}) = \alpha + \beta_p\log(P_{2,t}) + \beta_m\log(M_t) + \varepsilon_t.
$$

The residual $\varepsilon_t$ is the spread. A rolling z-score measures its distance from the training-window mean. Entries require the configured z-score, cointegration, augmented Dickey–Fuller, and fit-quality thresholds.

## Walk-forward discipline

- Every decision uses only information available before that decision.
- The model is refit on a trailing window.
- An open trade can freeze its entry model for exit decisions.
- Costs are charged on traded notional.
- Without daily rebalancing, weights drift with asset returns.
- The optional market leg hedges the estimated common-factor exposure.

## Exit rules

A trade closes on mean reversion, stop loss, maximum holding period, or relationship deterioration. The holding limit is the ceiling of entry half-life multiplied by the configured limit.

## Research limitations

- Screening many pairs creates a multiple-testing problem. Use false-discovery controls or an untouched holdout universe before making claims.
- Current S&P 500 constituents create survivorship bias in historical tests.
- Yahoo Finance data is convenient research data, not execution-grade market data.
- Cointegration can break structurally; historical stability does not prove future stability.
- Report turnover, capacity, slippage sensitivity, and out-of-sample performance—not only Sharpe ratio.
- This repository is research software, not investment advice.

