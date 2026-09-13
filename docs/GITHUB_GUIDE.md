# GitHub publishing guide

## Repository settings

- Name: `quant-pairs-research`
- Description: `Walk-forward research framework for market-neutral pairs trading across equities and crypto.`
- Visibility: Public
- Topics: `quantitative-finance`, `pairs-trading`, `statistical-arbitrage`, `cointegration`, `backtesting`, `python`
- Website: leave blank unless you later add a dashboard

Do not add another README, license, or `.gitignore` when creating the GitHub repository; this project already contains them.

## Publish

Run these commands inside the project folder after creating an empty GitHub repository:

```bash
git add .
git commit -m "Build walk-forward pairs trading research framework"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/quant-pairs-research.git
git push -u origin main
```

## Before sharing with recruiters

1. Replace `YOUR_USERNAME` in the README clone command.
2. Run a small reproducible experiment and commit one lightweight example result under `examples/`; do not commit the full `results/` folder.
3. Add one equity-curve chart and one spread z-score chart to the README only if they come from a clearly labeled out-of-sample run.
4. Create a GitHub release named `v0.1.0`.
5. Pin the repository on your GitHub profile.

Suggested resume line:

> Built a walk-forward statistical-arbitrage framework that screens equity and crypto pairs using market-adjusted cointegration, models transaction costs and hedge drift, and applies relationship-break risk controls.

Do not claim profitability unless you publish the exact test window, costs, benchmark, and untouched out-of-sample results.
