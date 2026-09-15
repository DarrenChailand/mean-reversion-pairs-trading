# Quant Pairs Research

A walk-forward research framework for screening and backtesting market-neutral pairs across United States equities, Indonesian banks, and cryptoassets.

> Research only. This is not investment advice and is not an execution system.

## Why this project is credible

- Market-adjusted three-leg spread model
- Rolling, prior-data-only model estimation
- Augmented Dickey–Fuller residual-stationarity entry filter
- Transaction costs, drifting weights, stop losses, and time stops
- Relationship-break exits based on fit and hedge-ratio changes
- Shared price downloads for efficient universe screening
- Structured comma-separated output for reproducible analysis

## Model

The residual from the following regression defines the spread:

$$
\log(P_{1,t}) = \alpha + \beta_p\log(P_{2,t}) + \beta_m\log(M_t) + \varepsilon_t.
$$

Signals are based on the rolling z-score of $\varepsilon_t$. See [the methodology](docs/METHODOLOGY.md) for assumptions and limitations.

## Project structure

```text
quant-pairs-research/
├── src/pairs_research/
│   ├── analyzer.py
│   └── screeners/
│       ├── crypto.py
│       ├── ihsg_banks.py
│       └── sp500.py
├── tests/
├── docs/METHODOLOGY.md
├── pyproject.toml
└── README.md
```

## Setup

```bash
git clone https://github.com/DarrenChailand/quant-pairs-research.git
cd quant-pairs-research
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Quick start

Use a small run first:

```bash
pairs-screen-crypto --max-pairs 10
pairs-screen-ihsg-banks --max-pairs 10
pairs-screen-sp500 --ticker-source fallback --max-pairs 10
```

Larger examples:

```bash
pairs-screen-crypto --ticker-source coingecko --top-n 50 --start 2024-01-01 --top 20
pairs-screen-ihsg-banks --start 2024-01-01 --rolling-window 126 --top 20
pairs-screen-sp500 --sector Financials --prefilter-top-corr 10 --start 2024-01-01
```

Each run writes timestamped result and error files under `results/`. Run any command with `--help` to see all controls.

## Reproducible research checklist

- Record the run command, date, data source, and package versions.
- Reserve an out-of-sample period before tuning thresholds.
- Correct for multiple testing when screening large universes.
- Test higher costs and slippage than expected.
- Do not claim live profitability from an in-sample backtest.

## Roadmap

- Point-in-time constituent data to remove survivorship bias
- False-discovery-rate control for large screens
- Purged time-series cross-validation
- Borrow availability, liquidity, and market-impact filters
- Continuous integration and benchmark datasets

## License

MIT. See [LICENSE](LICENSE).
