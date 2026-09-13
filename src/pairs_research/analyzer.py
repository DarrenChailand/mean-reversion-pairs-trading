# pip install yfinance pandas numpy matplotlib statsmodels

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

try:
    from statsmodels.regression.rolling import RollingOLS
except Exception:  # pragma: no cover - fallback for old statsmodels versions
    RollingOLS = None


@dataclass
class PairAnalysisConfig:
    stock1: str
    stock2: str
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    signal_date: pd.Timestamp | None = None
    market_ticker: str = "^JKSE"
    rolling_window: int = 252
    entry_zscore: float = 2.0
    trade_market_leg: bool = True
    show_plots: bool = True
    print_all_rolling_betas: bool = False
    verbose: bool = True


@dataclass
class PairAnalysisResult:
    alpha: float
    beta_pair: float
    beta_market: float
    r_squared: float
    spread_mean: float
    spread_std: float
    signal_spread: float
    signal_zscore: float
    half_life: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    signal_date: pd.Timestamp
    rolling_betas: pd.DataFrame
    all_zscore: pd.Series
    adf_statistic: float
    adf_pvalue: float
    beta_pair_stability_ratio: float
    beta_market_stability_ratio: float


def _to_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value)


def _clean_index(index: pd.Index) -> pd.Index:
    cleaned = pd.DatetimeIndex(index)
    if cleaned.tz is not None:
        cleaned = cleaned.tz_localize(None)
    return cleaned


def _extract_close_prices(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return a plain DataFrame of close prices from yfinance output or an existing price frame."""
    if isinstance(data, pd.Series):
        prices = data.to_frame()
    elif isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(-1):
            prices = data.xs("Close", axis=1, level=-1).copy()
        else:
            raise ValueError("Could not find Close prices in MultiIndex DataFrame.")
    else:
        prices = data.copy()

    prices.columns = [str(col).upper().strip() for col in prices.columns]
    prices.index = _clean_index(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def prepare_price_frame(
    prices: pd.DataFrame | pd.Series, required_tickers: list[str] | None = None
) -> pd.DataFrame:
    """Clean a price DataFrame and optionally select required tickers without downloading.

    This function is intentionally central. It must be used for both:
    1. one-pair runs that download prices inside this analyzer; and
    2. fast screener runs that pass in a pre-downloaded price table.

    Any zero/negative price is converted to NaN before logs are calculated.
    """
    price_frame = _extract_close_prices(prices)
    if required_tickers is not None:
        required = [ticker.upper().strip() for ticker in required_tickers]
        missing = [ticker for ticker in required if ticker not in price_frame.columns]
        if missing:
            raise ValueError(f"Ticker not found in supplied price data: {missing}")
        price_frame = price_frame[required]

    price_frame = price_frame.apply(pd.to_numeric, errors="coerce")
    price_frame = price_frame.replace([np.inf, -np.inf], np.nan)

    # Critical for log-price models: Yahoo can occasionally return 0 prices
    # for crypto/history glitches. log(0) = -inf. Do not leave any zero or
    # negative value in the frame before any log calculation.
    price_frame = price_frame.mask(price_frame <= 0)
    return price_frame


def safe_log_price_frame(
    prices: pd.DataFrame | pd.Series, required_tickers: list[str] | None = None
) -> pd.DataFrame:
    """Return log prices after dropping rows with missing/non-positive prices.

    This avoids pandas/numpy divide-by-zero warnings and prevents corrupted
    regressions. It raises clearly if no valid positive price rows remain.
    """
    clean = prepare_price_frame(prices, required_tickers=required_tickers).dropna(how="any")
    if clean.empty:
        raise ValueError("No valid positive price rows available for log calculation.")

    # Defensive check. prepare_price_frame should already mask these, but this
    # makes the analyzer safe even if this helper is used on unusual inputs.
    bad_mask = clean <= 0
    if bool(bad_mask.any().any()):
        bad_cols = clean.columns[bad_mask.any()].tolist()
        raise ValueError(f"Non-positive prices remain before log calculation: {bad_cols}")

    logged_values = np.log(clean.to_numpy(dtype=float))
    return pd.DataFrame(logged_values, index=clean.index, columns=clean.columns)


def safe_log_price_value(value: float, label: str) -> float:
    """Log one scalar price with a clear error instead of log(0) warning."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid non-positive price for {label}: {value}")
    return float(np.log(value))


def download_price_data(
    tickers: list[str],
    period: str = "max",
    interval: str = "1d",
    auto_adjust: bool = True,
    progress: bool = False,
    threads: bool = True,
) -> pd.DataFrame:
    """Download close prices once. Screeners reuse this instead of downloading per pair."""
    tickers = list(
        dict.fromkeys([ticker.upper().strip() for ticker in tickers if str(ticker).strip()])
    )
    if not tickers:
        raise ValueError("No tickers supplied.")

    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        threads=threads,
    )
    if data.empty:
        raise ValueError("No data downloaded. Check ticker symbols or internet connection.")

    return prepare_price_frame(data)


class PairTradingAnalyzer:
    def __init__(self, config: PairAnalysisConfig, prices_all: pd.DataFrame | None = None):
        self.config = config
        self._clean_config()

        self.tickers = list(
            dict.fromkeys(
                [
                    self.config.stock1,
                    self.config.stock2,
                    self.config.market_ticker,
                ]
            )
        )

        self.prices_all: pd.DataFrame | None = None
        if prices_all is not None:
            self.prices_all = prepare_price_frame(prices_all, self.tickers)

        self.prices_train: pd.DataFrame | None = None
        self.train_end_actual: pd.Timestamp | None = None
        self.signal_date_actual: pd.Timestamp | None = None

    def _clean_config(self) -> None:
        self.config.stock1 = self.config.stock1.upper().strip()
        self.config.stock2 = self.config.stock2.upper().strip()
        self.config.market_ticker = (self.config.market_ticker or "^JKSE").upper().strip()

        if self.config.stock1 == self.config.stock2:
            raise ValueError("stock1 and stock2 must be different tickers.")
        if self.config.market_ticker in {self.config.stock1, self.config.stock2}:
            raise ValueError("market_ticker must be different from stock1 and stock2.")
        if self.config.rolling_window <= 1:
            raise ValueError("rolling_window must be greater than 1.")
        if float(self.config.entry_zscore) <= 0:
            raise ValueError("entry_zscore must be greater than 0.")

    def _log(self, *args) -> None:
        if self.config.verbose:
            print(*args)

    def run(self) -> PairAnalysisResult:
        self._log(f"\nAnalyzing pair: {self.config.stock1} vs {self.config.stock2}")
        self._log(f"Market factor: {self.config.market_ticker}")

        if self.prices_all is None:
            self.prices_all = self._download_prices()
        else:
            self.prices_all = prepare_price_frame(self.prices_all, self.tickers)

        self.train_end_actual = self._resolve_available_date(self.config.train_end, "train end")
        self.signal_date_actual = self._resolve_available_date(self.config.signal_date, "signal")
        self._validate_dates()
        self.prices_train = self._make_training_prices()

        self.prices_train = prepare_price_frame(self.prices_train, self.tickers).dropna(how="any")
        if len(self.prices_train) < 100:
            raise ValueError(
                "Not enough positive-price training data after removing zero/negative prices."
            )
        if len(self.prices_train) < self.config.rolling_window:
            raise ValueError("Rolling window is longer than positive-price training data.")

        log_train = safe_log_price_frame(self.prices_train, self.tickers)
        x_train = log_train[self.config.stock1]
        y_train = log_train[self.config.stock2]
        m_train = log_train[self.config.market_ticker]

        model = self._fit_market_adjusted_model(x_train, y_train, m_train)
        alpha = float(model.params["const"])
        beta_pair = float(model.params[self.config.stock2])
        beta_market = float(model.params[self.config.market_ticker])
        train_spread = model.resid
        spread_mean = float(train_spread.mean())
        spread_std = float(train_spread.std())

        self._print_market_adjusted_model(model, alpha, beta_pair, beta_market)
        adf_statistic, adf_pvalue = self._adf_test(train_spread)
        half_life = self._calculate_half_life(train_spread)
        rolling_betas = self._calculate_rolling_betas()
        beta_pair_stability_ratio = self._stability_ratio(rolling_betas["beta_pair"])
        beta_market_stability_ratio = self._stability_ratio(rolling_betas["beta_market"])
        self._print_rolling_beta_stability(
            rolling_betas, beta_pair_stability_ratio, beta_market_stability_ratio
        )

        signal_spread, signal_zscore = self._calculate_signal(
            alpha=alpha,
            beta_pair=beta_pair,
            beta_market=beta_market,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

        all_zscore = self._calculate_all_zscore(
            alpha=alpha,
            beta_pair=beta_pair,
            beta_market=beta_market,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

        self._print_final_signal(
            spread_mean=spread_mean,
            spread_std=spread_std,
            signal_spread=signal_spread,
            signal_zscore=signal_zscore,
            beta_pair=beta_pair,
            beta_market=beta_market,
        )

        if self.config.show_plots:
            self._plot_all_zscore(all_zscore)
            self._plot_training_spread(train_spread, spread_mean)
            self._plot_rolling_betas(rolling_betas, beta_pair, beta_market)

        return PairAnalysisResult(
            alpha=alpha,
            beta_pair=beta_pair,
            beta_market=beta_market,
            r_squared=float(model.rsquared),
            spread_mean=spread_mean,
            spread_std=spread_std,
            signal_spread=float(signal_spread),
            signal_zscore=float(signal_zscore),
            half_life=float(half_life),
            train_start=self.prices_train.index[0],
            train_end=self.prices_train.index[-1],
            signal_date=self.signal_date_actual,
            rolling_betas=rolling_betas,
            all_zscore=all_zscore,
            adf_statistic=float(adf_statistic),
            adf_pvalue=float(adf_pvalue),
            beta_pair_stability_ratio=float(beta_pair_stability_ratio),
            beta_market_stability_ratio=float(beta_market_stability_ratio),
        )

    def _download_prices(self) -> pd.DataFrame:
        prices = download_price_data(
            self.tickers, period="max", interval="1d", progress=False, threads=True
        )
        prices = prepare_price_frame(prices, self.tickers).dropna()
        if prices.empty:
            raise ValueError("No overlapping price data after removing missing rows.")
        return prices

    def _resolve_available_date(
        self, requested_date: pd.Timestamp | None, label: str
    ) -> pd.Timestamp:
        assert self.prices_all is not None
        prices = prepare_price_frame(self.prices_all, self.tickers).dropna(how="any")
        if prices.empty:
            raise ValueError("No overlapping price data after removing missing rows.")

        if requested_date is None:
            return prices.index[-1]

        available = prices[prices.index <= requested_date]
        if available.empty:
            raise ValueError(f"No price data available before or on {label} date.")
        return available.index[-1]

    def _validate_dates(self) -> None:
        assert self.train_end_actual is not None
        assert self.signal_date_actual is not None

        if self.train_end_actual > self.signal_date_actual:
            raise ValueError("Train end date cannot be after signal date.")

        if self.train_end_actual == self.signal_date_actual:
            self._log("\nWARNING: Training includes the signal date.")
            self._log("This is okay only if you use today's close to trade tomorrow.")
            self._log("For a clean same-day signal, train until the previous trading day.")

    def _make_training_prices(self) -> pd.DataFrame:
        assert self.prices_all is not None
        assert self.train_end_actual is not None

        prices_train = self.prices_all[self.prices_all.index <= self.train_end_actual].copy()
        if self.config.train_start is not None:
            prices_train = prices_train[prices_train.index >= self.config.train_start]

        prices_train = prepare_price_frame(prices_train, self.tickers).dropna(how="any")
        if len(prices_train) < 100:
            raise ValueError("Not enough training data. Use a longer training period.")
        if len(prices_train) < self.config.rolling_window:
            raise ValueError("Rolling window is longer than training data.")
        return prices_train

    def _fit_market_adjusted_model(
        self, x_train: pd.Series, y_train: pd.Series, m_train: pd.Series
    ):
        X_train = pd.DataFrame(
            {
                self.config.stock2: y_train,
                self.config.market_ticker: m_train,
            }
        )
        X_train = sm.add_constant(X_train, has_constant="add")
        return sm.OLS(x_train, X_train).fit()

    def _print_market_adjusted_model(
        self, model, alpha: float, beta_pair: float, beta_market: float
    ) -> None:
        self._log("\nMarket-Adjusted Regression")
        self._log("Equation:")
        self._log(
            f"log({self.config.stock1}) = {round(alpha, 6)} "
            f"+ {round(beta_pair, 6)} * log({self.config.stock2}) "
            f"+ {round(beta_market, 6)} * log({self.config.market_ticker})"
        )
        self._log("\nRegression values:")
        self._log("Alpha:", alpha)
        self._log("Beta pair:", beta_pair)
        self._log("Beta market:", beta_market)
        self._log("R-squared:", round(float(model.rsquared), 4))

    def _adf_test(self, train_spread: pd.Series) -> tuple[float, float]:
        self._log("\nADF Test on Market-Adjusted Training Spread")
        adf_result = adfuller(train_spread.dropna())
        adf_statistic = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
        self._log("ADF statistic:", round(adf_statistic, 4))
        self._log("p-value:", adf_pvalue)
        self._log("p-value scientific:", f"{adf_pvalue:.10e}")
        if adf_pvalue < 0.05:
            self._log(
                "Interpretation: Market-adjusted spread is likely stationary / mean-reverting."
            )
        else:
            self._log(
                "Interpretation: Weak/no evidence of mean reversion in market-adjusted spread."
            )
        return adf_statistic, adf_pvalue

    def _calculate_half_life(self, train_spread: pd.Series) -> float:
        self._log("\nHalf-life on Market-Adjusted Training Spread")
        half_life_df = pd.DataFrame(
            {
                "spread_lag": train_spread.shift(1),
                "spread_delta": train_spread - train_spread.shift(1),
            }
        ).dropna()

        X_half = sm.add_constant(half_life_df["spread_lag"], has_constant="add")
        y_half = half_life_df["spread_delta"]
        half_life_model = sm.OLS(y_half, X_half).fit()
        beta_half_life = float(half_life_model.params["spread_lag"])

        if beta_half_life < 0:
            half_life = float(-np.log(2) / beta_half_life)
            self._log("Half-life:", round(half_life, 2), "trading days")
        else:
            half_life = float("inf")
            self._log("Half-life: Not valid because spread is not mean-reverting.")
        return half_life

    def _calculate_rolling_betas(self) -> pd.DataFrame:
        assert self.prices_train is not None
        self._log("\nRolling Beta Stability Test")

        clean_train = prepare_price_frame(self.prices_train, self.tickers).dropna(how="any")
        log_train = safe_log_price_frame(clean_train, self.tickers)
        y = log_train[self.config.stock1]
        X = log_train[[self.config.stock2, self.config.market_ticker]]
        X = sm.add_constant(X, has_constant="add")

        if RollingOLS is not None:
            model = RollingOLS(y, X, window=self.config.rolling_window)
            fit = model.fit(params_only=True)
            params = fit.params.dropna().copy()
            if params.empty:
                raise ValueError("Rolling regression produced no valid beta rows.")
            return pd.DataFrame(
                {
                    "alpha": params["const"],
                    "beta_pair": params[self.config.stock2],
                    "beta_market": params[self.config.market_ticker],
                    "r_squared": np.nan,
                },
                index=params.index,
            )

        # Fallback for older statsmodels: slower, but avoids breaking the script.
        rolling_rows = []
        for i in range(self.config.rolling_window, len(self.prices_train) + 1):
            window_prices = prepare_price_frame(
                self.prices_train.iloc[i - self.config.rolling_window : i],
                self.tickers,
            ).dropna(how="any")
            if len(window_prices) < self.config.rolling_window:
                continue
            log_window = safe_log_price_frame(window_prices, self.tickers)
            x_w = log_window[self.config.stock1]
            y_w = log_window[self.config.stock2]
            m_w = log_window[self.config.market_ticker]
            X_w = pd.DataFrame(
                {
                    self.config.stock2: y_w,
                    self.config.market_ticker: m_w,
                }
            )
            X_w = sm.add_constant(X_w, has_constant="add")
            model_w = sm.OLS(x_w, X_w).fit()
            rolling_rows.append(
                {
                    "date": window_prices.index[-1],
                    "alpha": model_w.params["const"],
                    "beta_pair": model_w.params[self.config.stock2],
                    "beta_market": model_w.params[self.config.market_ticker],
                    "r_squared": model_w.rsquared,
                }
            )
        return pd.DataFrame(rolling_rows).set_index("date")

    @staticmethod
    def _stability_ratio(series: pd.Series) -> float:
        series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return float("nan")
        mean_val = float(series.mean())
        if abs(mean_val) <= 1e-8:
            return float("inf")
        return float(series.std() / abs(mean_val))

    def _print_rolling_beta_stability(
        self,
        rolling_betas: pd.DataFrame,
        beta_pair_ratio: float,
        beta_market_ratio: float,
    ) -> None:
        self._print_beta_stability(
            rolling_betas["beta_pair"], f"Rolling beta for {self.config.stock2}", beta_pair_ratio
        )
        self._print_beta_stability(
            rolling_betas["beta_market"],
            f"Rolling beta for {self.config.market_ticker}",
            beta_market_ratio,
        )

        if self.config.print_all_rolling_betas:
            with pd.option_context("display.max_rows", None):
                self._log("\nAll rolling betas")
                self._log(rolling_betas[["beta_pair", "beta_market", "r_squared"]])

    def _print_beta_stability(self, series: pd.Series, name: str, ratio: float) -> None:
        self._log(f"\n{name}")
        self._log("Mean:", round(float(series.mean()), 6))
        self._log("Std:", round(float(series.std()), 6))
        self._log("Min:", round(float(series.min()), 6))
        self._log("Max:", round(float(series.max()), 6))
        self._log("Stability ratio:", round(float(ratio), 4))

        if ratio < 0.25:
            self._log("Interpretation: Stable.")
        elif ratio < 0.50:
            self._log("Interpretation: Somewhat stable, but watch carefully.")
        else:
            self._log("Interpretation: Unstable. Relationship may be changing.")

    def _calculate_signal(
        self,
        alpha: float,
        beta_pair: float,
        beta_market: float,
        spread_mean: float,
        spread_std: float,
    ) -> tuple[float, float]:
        assert self.prices_all is not None
        assert self.signal_date_actual is not None
        signal_prices = (
            prepare_price_frame(self.prices_all, self.tickers)
            .dropna(how="any")
            .loc[self.signal_date_actual]
        )
        x_signal = safe_log_price_value(signal_prices[self.config.stock1], self.config.stock1)
        y_signal = safe_log_price_value(signal_prices[self.config.stock2], self.config.stock2)
        m_signal = safe_log_price_value(
            signal_prices[self.config.market_ticker], self.config.market_ticker
        )
        signal_predicted = alpha + beta_pair * y_signal + beta_market * m_signal
        signal_spread = x_signal - signal_predicted
        signal_zscore = (signal_spread - spread_mean) / spread_std
        return float(signal_spread), float(signal_zscore)

    def _calculate_all_zscore(
        self,
        alpha: float,
        beta_pair: float,
        beta_market: float,
        spread_mean: float,
        spread_std: float,
    ) -> pd.Series:
        assert self.prices_all is not None
        clean_prices = prepare_price_frame(self.prices_all, self.tickers).dropna(how="any")
        log_all = safe_log_price_frame(clean_prices, self.tickers)
        x_all = log_all[self.config.stock1]
        y_all = log_all[self.config.stock2]
        m_all = log_all[self.config.market_ticker]
        all_spread = x_all - (alpha + beta_pair * y_all + beta_market * m_all)
        return (all_spread - spread_mean) / spread_std

    def _print_final_signal(
        self,
        spread_mean: float,
        spread_std: float,
        signal_spread: float,
        signal_zscore: float,
        beta_pair: float,
        beta_market: float,
    ) -> None:
        assert self.prices_train is not None
        assert self.signal_date_actual is not None
        self._log("\nPair Result")
        self._log("Training start:", self.prices_train.index[0].date())
        self._log("Training end:", self.prices_train.index[-1].date())
        self._log("Signal date:", self.signal_date_actual.date())
        self._log("\nSpread values:")
        self._log("Training spread mean:", round(spread_mean, 6))
        self._log("Training spread std:", round(spread_std, 6))
        self._log("Signal spread:", round(signal_spread, 6))
        self._log("\nMarket-adjusted signal z-score:", round(signal_zscore, 3))

        threshold = float(self.config.entry_zscore)
        direction = -1 if signal_zscore >= threshold else (1 if signal_zscore <= -threshold else 0)
        if direction != 0:
            weights = _target_spread_weights(
                direction,
                beta_pair,
                beta_market,
                1.0,
                self.config.trade_market_leg,
            )
            legs = []
            for ticker, weight in zip(self.tickers, weights, strict=True):
                side = "LONG" if weight > 0 else "SHORT"
                legs.append(f"{side} {ticker} ({abs(weight):.3f})")
            relative_word = "expensive" if direction == -1 else "cheap"
            self._log(
                f"\nSignal: {self.config.stock1} looks {relative_word} relative to the fitted three-leg spread."
            )
            self._log("Possible three-leg trade:", ", ".join(legs))
        else:
            self._log("\nSignal: No strong mean-reversion signal.")

    def _plot_all_zscore(self, all_zscore: pd.Series) -> None:
        assert self.prices_train is not None
        assert self.signal_date_actual is not None
        plt.figure(figsize=(12, 5))
        plt.plot(all_zscore.index, all_zscore, label="Market-adjusted z-score")
        plt.axhline(2, linestyle="--", label="+2")
        plt.axhline(0, linestyle="-", label="Mean")
        plt.axhline(-2, linestyle="--", label="-2")
        plt.axvline(self.prices_train.index[0], linestyle="--", label="Train start")
        plt.axvline(self.prices_train.index[-1], linestyle="--", label="Train end")
        plt.axvline(self.signal_date_actual, linestyle=":", label="Signal date")
        plt.title(f"{self.config.stock1} vs {self.config.stock2} Market-Adjusted Z-Score")
        plt.xlabel("Date")
        plt.ylabel("Z-score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_training_spread(self, train_spread: pd.Series, spread_mean: float) -> None:
        plt.figure(figsize=(12, 5))
        plt.plot(train_spread.index, train_spread, label="Market-adjusted training spread")
        plt.axhline(spread_mean, linestyle="-", label="Training mean")
        plt.title(f"{self.config.stock1} vs {self.config.stock2} Market-Adjusted Training Spread")
        plt.xlabel("Date")
        plt.ylabel("Residual spread")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_rolling_betas(
        self, rolling_betas: pd.DataFrame, beta_pair: float, beta_market: float
    ) -> None:
        plt.figure(figsize=(12, 5))
        plt.plot(
            rolling_betas.index,
            rolling_betas["beta_pair"],
            label=f"Rolling beta: {self.config.stock2}",
        )
        plt.axhline(beta_pair, linestyle="--", label="Full-training beta")
        plt.title(f"Rolling Beta Stability: {self.config.stock1} vs {self.config.stock2}")
        plt.xlabel("Date")
        plt.ylabel("Beta")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.plot(
            rolling_betas.index,
            rolling_betas["beta_market"],
            label=f"Rolling beta: {self.config.market_ticker}",
        )
        plt.axhline(beta_market, linestyle="--", label="Full-training market beta")
        plt.title(
            f"Rolling Market Beta Stability: {self.config.stock1} vs {self.config.market_ticker}"
        )
        plt.xlabel("Date")
        plt.ylabel("Market beta")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


@dataclass
class PairBacktestConfig:
    """Configuration for a no-look-ahead daily walk-forward backtest.

    The default strategy freezes the entry model while a trade is open and
    trades the market factor as a third hedge leg. While flat, a fresh rolling
    model is still fitted every day to search for the next entry.
    """

    stock1: str
    stock2: str
    market_ticker: str = "^JKSE"
    backtest_start: pd.Timestamp | None = None
    backtest_end: pd.Timestamp | None = None
    rolling_years: float = 2.0
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    buy_transaction_cost: float = 0.002
    sell_transaction_cost: float = 0.002
    gross_exposure: float = 1.0
    min_training_observations: int = 252
    annualization_days: int = 252
    use_saved_model_during_trade: bool = True
    trade_market_leg: bool = True
    rebalance_daily: bool = False
    max_holding_half_lives: float | None = 2.0
    stop_loss_fraction: float | None = 0.10
    relationship_break_r_squared: float | None = 0.60
    relationship_break_beta_change: float | None = 0.50
    adf_threshold: float | None = 0.05
    verbose: bool = False


@dataclass
class PairBacktestResult:
    daily: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, object]


@dataclass
class PairStrategyResult:
    analysis: PairAnalysisResult
    backtest: PairBacktestResult
    summary: dict


def _validate_fraction(value: float, label: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number.")
    return value


def _optional_positive_float(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite or None.")
    if value <= 0:
        return None
    return value


def _optional_positive_int(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def _optional_probability(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite or None.")
    if value < 0 or value > 1:
        raise ValueError(f"{label} must be between 0 and 1, or None.")
    return value


def _validate_backtest_config(config: PairBacktestConfig) -> PairBacktestConfig:
    config.stock1 = config.stock1.upper().strip()
    config.stock2 = config.stock2.upper().strip()
    config.market_ticker = (config.market_ticker or "^JKSE").upper().strip()

    if config.stock1 == config.stock2:
        raise ValueError("stock1 and stock2 must be different tickers.")
    if config.market_ticker in {config.stock1, config.stock2}:
        raise ValueError("market_ticker must be different from stock1 and stock2.")
    if float(config.rolling_years) <= 0:
        raise ValueError("rolling_years must be greater than 0.")
    if float(config.entry_zscore) <= 0:
        raise ValueError("entry_zscore must be greater than 0.")
    if float(config.exit_zscore) < 0 or float(config.exit_zscore) >= float(config.entry_zscore):
        raise ValueError("exit_zscore must be non-negative and smaller than entry_zscore.")
    if int(config.min_training_observations) < 30:
        raise ValueError("min_training_observations must be at least 30.")
    if int(config.annualization_days) <= 0:
        raise ValueError("annualization_days must be greater than 0.")
    if float(config.gross_exposure) <= 0:
        raise ValueError("gross_exposure must be greater than 0.")

    config.buy_transaction_cost = _validate_fraction(
        config.buy_transaction_cost, "buy_transaction_cost"
    )
    config.sell_transaction_cost = _validate_fraction(
        config.sell_transaction_cost, "sell_transaction_cost"
    )
    config.backtest_start = _to_timestamp(config.backtest_start)
    config.backtest_end = _to_timestamp(config.backtest_end)
    config.max_holding_half_lives = _optional_positive_float(
        config.max_holding_half_lives,
        "max_holding_half_lives",
    )
    config.stop_loss_fraction = _optional_positive_float(
        config.stop_loss_fraction, "stop_loss_fraction"
    )

    if config.relationship_break_r_squared is not None:
        value = float(config.relationship_break_r_squared)
        config.relationship_break_r_squared = None if not np.isfinite(value) or value < 0 else value
    config.relationship_break_beta_change = _optional_positive_float(
        config.relationship_break_beta_change,
        "relationship_break_beta_change",
    )
    config.adf_threshold = _optional_probability(config.adf_threshold, "adf_threshold")
    return config


def _fit_walk_forward_model(
    train_prices: pd.DataFrame, stock1: str, stock2: str, market_ticker: str
) -> dict[str, float]:
    """Fit the market-adjusted model using training data only."""
    tickers = [stock1, stock2, market_ticker]
    logged = safe_log_price_frame(train_prices, tickers)
    stock1_log = logged[stock1]
    stock2_log = logged[stock2]
    market_log = logged[market_ticker]

    y = stock1_log.to_numpy(dtype=float)
    X = np.column_stack(
        [
            np.ones(len(logged), dtype=float),
            stock2_log.to_numpy(dtype=float),
            market_log.to_numpy(dtype=float),
        ]
    )
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ params
    residual = y - fitted
    spread_mean = float(np.mean(residual))
    spread_std = float(np.std(residual, ddof=1))
    if not np.isfinite(spread_std) or spread_std <= 1e-12:
        raise ValueError("Training spread standard deviation is zero or invalid.")

    ss_res = float(np.sum(np.square(residual)))
    centred = y - float(np.mean(y))
    ss_total = float(np.sum(np.square(centred)))
    r_squared = float(1.0 - ss_res / ss_total) if ss_total > 0 else float("nan")

    # Stationarity test on the actual market-adjusted spread used for signals.
    residual_series = pd.Series(residual, index=logged.index, dtype=float)
    try:
        adf_result = adfuller(residual_series.dropna())
        adf_statistic = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
    except Exception:
        adf_statistic = float("nan")
        adf_pvalue = float("nan")

    # Estimate the spread half-life from the same no-look-ahead training window.
    # This is measured in observed trading periods, not calendar days.
    half_life_frame = pd.DataFrame(
        {
            "spread_lag": residual_series.shift(1),
            "spread_delta": residual_series - residual_series.shift(1),
        }
    ).dropna()
    try:
        half_X = np.column_stack(
            [
                np.ones(len(half_life_frame), dtype=float),
                half_life_frame["spread_lag"].to_numpy(dtype=float),
            ]
        )
        half_y = half_life_frame["spread_delta"].to_numpy(dtype=float)
        half_params, _, _, _ = np.linalg.lstsq(half_X, half_y, rcond=None)
        mean_reversion_speed = float(half_params[1])
        half_life = (
            float(-np.log(2.0) / mean_reversion_speed)
            if np.isfinite(mean_reversion_speed) and mean_reversion_speed < 0
            else float("inf")
        )
        if not np.isfinite(half_life) or half_life <= 0:
            half_life = float("inf")
    except Exception:
        half_life = float("inf")

    return {
        "alpha": float(params[0]),
        "beta_pair": float(params[1]),
        "beta_market": float(params[2]),
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "r_squared": r_squared,
        "adf_statistic": adf_statistic,
        "adf_pvalue": adf_pvalue,
        "half_life": half_life,
    }


def _signal_from_model(
    signal_prices: pd.Series,
    model: dict[str, float],
    stock1: str,
    stock2: str,
    market_ticker: str,
) -> tuple[float, float]:
    x = safe_log_price_value(signal_prices[stock1], stock1)
    y = safe_log_price_value(signal_prices[stock2], stock2)
    m = safe_log_price_value(signal_prices[market_ticker], market_ticker)
    predicted = model["alpha"] + model["beta_pair"] * y + model["beta_market"] * m
    spread = x - predicted
    zscore = (spread - model["spread_mean"]) / model["spread_std"]
    return float(spread), float(zscore)


def _entry_position(zscore: float, entry_zscore: float) -> int:
    """Return -1 for short spread, +1 for long spread, and 0 for flat."""
    if not np.isfinite(zscore):
        return 0
    if zscore >= entry_zscore:
        return -1
    if zscore <= -entry_zscore:
        return 1
    return 0


def _mean_exit_reached(position: int, zscore: float, exit_zscore: float) -> bool:
    if not np.isfinite(zscore):
        return False
    if position == 1:
        return bool(zscore >= -float(exit_zscore))
    if position == -1:
        return bool(zscore <= float(exit_zscore))
    return False


def _target_spread_weights(
    direction: int,
    beta_pair: float,
    beta_market: float,
    gross_exposure: float,
    trade_market_leg: bool,
) -> np.ndarray:
    """Create three-leg residual-spread weights and normalise by gross notional.

    The residual is log(stock1) - beta_pair*log(stock2) - beta_market*log(market).
    A long-spread trade therefore uses [1, -beta_pair, -beta_market]. If the
    market leg is disabled, its weight is set to zero for backward comparison.
    """
    if direction == 0:
        return np.zeros(3, dtype=float)
    market_weight = -float(beta_market) if trade_market_leg else 0.0
    raw = np.array([1.0, -float(beta_pair), market_weight], dtype=float) * float(direction)
    gross = float(np.sum(np.abs(raw)))
    if not np.isfinite(gross) or gross <= 1e-12:
        raise ValueError("Invalid hedge ratios produced zero/invalid gross exposure.")
    return raw * (float(gross_exposure) / gross)


def _transaction_cost_details(
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    buy_cost: float,
    sell_cost: float,
) -> tuple[float, float, float]:
    delta = np.asarray(new_weights, dtype=float) - np.asarray(old_weights, dtype=float)
    buy_turnover = float(np.clip(delta, 0.0, None).sum())
    sell_turnover = float(np.clip(-delta, 0.0, None).sum())
    cost_rate = buy_turnover * float(buy_cost) + sell_turnover * float(sell_cost)
    return buy_turnover, sell_turnover, float(cost_rate)


def _model_from_signal_row(signal_row: pd.Series) -> dict[str, float]:
    return {
        "alpha": float(signal_row["alpha"]),
        "beta_pair": float(signal_row["beta_pair"]),
        "beta_market": float(signal_row["beta_market"]),
        "spread_mean": float(signal_row["spread_mean"]),
        "spread_std": float(signal_row["spread_std"]),
        "r_squared": float(signal_row["r_squared"]),
        "adf_statistic": float(signal_row["adf_statistic"]),
        "adf_pvalue": float(signal_row["adf_pvalue"]),
        "half_life": float(signal_row["half_life"]),
    }


def _relative_beta_change(current: float, entry: float) -> float:
    denominator = max(abs(float(entry)), 0.10)
    return float(abs(float(current) - float(entry)) / denominator)


def _entry_block_reason(
    live_model: dict[str, float],
    config: PairBacktestConfig,
) -> str | None:
    """Return why a fresh model may not open a trade, or None if eligible."""
    if config.adf_threshold is not None:
        pvalue = float(live_model.get("adf_pvalue", float("nan")))
        if not np.isfinite(pvalue) or pvalue > float(config.adf_threshold):
            return "ENTRY_ADF"

    if config.relationship_break_r_squared is not None:
        r_squared = float(live_model.get("r_squared", float("nan")))
        if not np.isfinite(r_squared) or r_squared < float(config.relationship_break_r_squared):
            return "ENTRY_R_SQUARED"

    if config.max_holding_half_lives is not None:
        half_life = float(live_model.get("half_life", float("nan")))
        if not np.isfinite(half_life) or half_life <= 0:
            return "ENTRY_HALF_LIFE"

    return None


def _relationship_break_reason(
    live_model: dict[str, float],
    entry_model: dict[str, float],
    config: PairBacktestConfig,
) -> str | None:
    r2_floor = config.relationship_break_r_squared
    if r2_floor is not None:
        live_r2 = float(live_model.get("r_squared", float("nan")))
        if np.isfinite(live_r2) and live_r2 < float(r2_floor):
            return "RELATIONSHIP_R_SQUARED"

    beta_limit = config.relationship_break_beta_change
    if beta_limit is not None:
        pair_change = _relative_beta_change(live_model["beta_pair"], entry_model["beta_pair"])
        if pair_change > float(beta_limit):
            return "RELATIONSHIP_PAIR_BETA"
        if config.trade_market_leg:
            market_change = _relative_beta_change(
                live_model["beta_market"], entry_model["beta_market"]
            )
            if market_change > float(beta_limit):
                return "RELATIONSHIP_MARKET_BETA"
    return None


def _build_walk_forward_signals(prices: pd.DataFrame, config: PairBacktestConfig) -> pd.DataFrame:
    """Generate one signal per day using only observations strictly before that day."""
    rows: list[dict] = []
    index = prices.index
    if len(index) < config.min_training_observations + 2:
        raise ValueError("Not enough overlapping data for the requested walk-forward backtest.")

    first_signal = (
        index[0] if config.backtest_start is None else pd.Timestamp(config.backtest_start)
    )
    last_signal = (
        index[-2]
        if config.backtest_end is None
        else min(pd.Timestamp(config.backtest_end), index[-2])
    )

    for signal_pos in range(1, len(index) - 1):
        signal_date = index[signal_pos]
        if signal_date < first_signal or signal_date > last_signal:
            continue

        train_end = index[signal_pos - 1]
        rolling_start = signal_date - pd.Timedelta(
            days=int(round(365.25 * float(config.rolling_years)))
        )
        train_prices = prices.loc[(prices.index >= rolling_start) & (prices.index <= train_end)]
        if len(train_prices) < int(config.min_training_observations):
            continue

        try:
            model = _fit_walk_forward_model(
                train_prices,
                config.stock1,
                config.stock2,
                config.market_ticker,
            )
            spread, zscore = _signal_from_model(
                prices.loc[signal_date],
                model,
                config.stock1,
                config.stock2,
                config.market_ticker,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": index[signal_pos + 1],
                "train_start": train_prices.index[0],
                "train_end": train_prices.index[-1],
                "training_observations": int(len(train_prices)),
                "alpha": model["alpha"],
                "beta_pair": model["beta_pair"],
                "beta_market": model["beta_market"],
                "r_squared": model["r_squared"],
                "adf_statistic": model["adf_statistic"],
                "adf_pvalue": model["adf_pvalue"],
                "half_life": model["half_life"],
                "spread_mean": model["spread_mean"],
                "spread_std": model["spread_std"],
                "signal_spread": spread,
                "signal_zscore": zscore,
            }
        )

    signals = pd.DataFrame(rows)
    if signals.empty:
        raise ValueError(
            "No valid walk-forward signals were produced. Use an earlier download period, "
            "a later backtest start, or fewer minimum training observations."
        )
    return signals.set_index("signal_date").sort_index()


def run_daily_walk_forward_backtest_from_prices(
    prices_all: pd.DataFrame,
    stock1: str,
    stock2: str,
    market_ticker: str = "^JKSE",
    backtest_start: str | pd.Timestamp | None = None,
    backtest_end: str | pd.Timestamp | None = None,
    rolling_years: float = 2.0,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.0,
    buy_transaction_cost: float = 0.002,
    sell_transaction_cost: float = 0.002,
    gross_exposure: float = 1.0,
    min_training_observations: int = 252,
    annualization_days: int = 252,
    use_saved_model_during_trade: bool = True,
    trade_market_leg: bool = True,
    rebalance_daily: bool = False,
    max_holding_half_lives: float | None = 2.0,
    stop_loss_fraction: float | None = 0.10,
    relationship_break_r_squared: float | None = 0.60,
    relationship_break_beta_change: float | None = 0.50,
    adf_threshold: float | None = 0.05,
    verbose: bool = False,
) -> PairBacktestResult:
    """Run a daily rolling-window backtest with next-close execution.

    On signal date t, the model uses only prices through t-1. The signal uses
    the close at t and is executed at the next available close. While flat, the
    rolling model is refitted daily. If use_saved_model_during_trade is True,
    the entry model is frozen for mean-reversion exits while a separate fresh
    model is used only to monitor whether the relationship has broken. The
    maximum holding period is frozen at entry as a chosen multiple of the
    entry model's estimated spread half-life.
    """
    config = _validate_backtest_config(
        PairBacktestConfig(
            stock1=stock1,
            stock2=stock2,
            market_ticker=market_ticker,
            backtest_start=_to_timestamp(backtest_start),
            backtest_end=_to_timestamp(backtest_end),
            rolling_years=rolling_years,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            buy_transaction_cost=buy_transaction_cost,
            sell_transaction_cost=sell_transaction_cost,
            gross_exposure=gross_exposure,
            min_training_observations=min_training_observations,
            annualization_days=annualization_days,
            use_saved_model_during_trade=use_saved_model_during_trade,
            trade_market_leg=trade_market_leg,
            rebalance_daily=rebalance_daily,
            max_holding_half_lives=max_holding_half_lives,
            stop_loss_fraction=stop_loss_fraction,
            relationship_break_r_squared=relationship_break_r_squared,
            relationship_break_beta_change=relationship_break_beta_change,
            adf_threshold=adf_threshold,
            verbose=verbose,
        )
    )

    tickers = [config.stock1, config.stock2, config.market_ticker]
    prices = prepare_price_frame(prices_all, tickers).dropna(how="any")
    if prices.empty:
        raise ValueError("No overlapping positive price data for the pair and market ticker.")

    signals = _build_walk_forward_signals(prices, config)
    signals_by_execution = signals.reset_index().set_index("execution_date")
    simulation_start = pd.Timestamp(signals_by_execution.index.min())
    simulation_end = pd.Timestamp(signals_by_execution.index.max())
    simulation_prices = prices.loc[simulation_start:simulation_end]

    equity = 1.0
    gross_equity = 1.0
    current_position = 0
    current_weights = np.zeros(3, dtype=float)
    open_trade: dict | None = None
    daily_rows: list[dict] = []
    trade_rows: list[dict] = []
    total_cost_amount = 0.0

    previous_prices: np.ndarray | None = None
    previous_equity = equity

    for simulation_step, (date, price_row) in enumerate(simulation_prices.iterrows()):
        leg_prices = price_row[tickers].to_numpy(dtype=float)
        gross_return = 0.0
        if previous_prices is not None:
            leg_returns = leg_prices / previous_prices - 1.0
            gross_return = float(np.dot(current_weights, leg_returns))
            gross_equity *= 1.0 + gross_return
            equity *= 1.0 + gross_return

            # Drift the actual portfolio weights when positions are held rather
            # than silently assuming free daily rebalancing.
            growth = 1.0 + gross_return
            if current_position != 0 and np.isfinite(growth) and growth > 1e-12:
                current_weights = current_weights * (1.0 + leg_returns) / growth

        signal_row = signals_by_execution.loc[date] if date in signals_by_execution.index else None
        if isinstance(signal_row, pd.DataFrame):
            signal_row = signal_row.iloc[-1]

        signal_date = pd.NaT
        live_model: dict[str, float] | None = None
        live_zscore = float("nan")
        decision_zscore = float("nan")
        exit_reason: str | None = None
        entry_block_reason: str | None = None
        exited_today = False
        buy_turnover = 0.0
        sell_turnover = 0.0
        transaction_cost_amount = 0.0
        is_final_date = bool(date == simulation_prices.index[-1])

        if signal_row is not None:
            signal_date = pd.Timestamp(signal_row["signal_date"])
            live_model = _model_from_signal_row(signal_row)
            live_zscore = float(signal_row["signal_zscore"])

        # Manage an existing trade first. No same-close reversal is allowed;
        # after an exit the next fresh daily model must generate a new entry.
        if current_position != 0 and open_trade is not None:
            if config.use_saved_model_during_trade and signal_row is not None:
                _, decision_zscore = _signal_from_model(
                    prices.loc[signal_date],
                    open_trade["entry_model"],
                    config.stock1,
                    config.stock2,
                    config.market_ticker,
                )
            elif signal_row is not None:
                decision_zscore = live_zscore

            current_trade_return = equity / open_trade["entry_equity_before_cost"] - 1.0
            holding_trading_days = int(simulation_step - open_trade["entry_simulation_step"])
            holding_calendar_days = int((date - open_trade["entry_date"]).days)

            if is_final_date:
                exit_reason = "FINAL_DATE"
            elif (
                config.stop_loss_fraction is not None
                and current_trade_return <= -config.stop_loss_fraction
            ):
                exit_reason = "STOP_LOSS"
            elif (
                open_trade["entry_max_holding_days"] is not None
                and holding_trading_days >= open_trade["entry_max_holding_days"]
            ):
                exit_reason = "MAX_HOLDING"
            elif live_model is not None:
                exit_reason = _relationship_break_reason(
                    live_model, open_trade["entry_model"], config
                )

            if exit_reason is None and _mean_exit_reached(
                current_position, decision_zscore, config.exit_zscore
            ):
                exit_reason = "MEAN_REVERSION"

            if exit_reason is not None:
                close_buy, close_sell, close_cost_rate = _transaction_cost_details(
                    current_weights,
                    np.zeros(3, dtype=float),
                    config.buy_transaction_cost,
                    config.sell_transaction_cost,
                )
                close_cost_amount = equity * close_cost_rate
                equity_before_close_cost = equity
                equity -= close_cost_amount
                buy_turnover += close_buy
                sell_turnover += close_sell
                transaction_cost_amount += close_cost_amount
                total_cost_amount += close_cost_amount
                open_trade["trade_transaction_cost"] += close_cost_amount

                trade_rows.append(
                    {
                        **open_trade,
                        "exit_date": date,
                        "exit_signal_date": signal_date,
                        "exit_reason": exit_reason,
                        "exit_live_zscore": live_zscore,
                        "exit_decision_zscore": decision_zscore,
                        "exit_equity_before_cost": equity_before_close_cost,
                        "exit_equity_after_cost": equity,
                        "gross_trade_return": gross_equity / open_trade["entry_gross_equity"] - 1.0,
                        "net_trade_return": equity / open_trade["entry_equity_before_cost"] - 1.0,
                        "exit_transaction_cost": close_cost_amount,
                        "total_trade_transaction_cost": open_trade["trade_transaction_cost"],
                        "holding_days": holding_trading_days,
                        "holding_trading_days": holding_trading_days,
                        "holding_calendar_days": holding_calendar_days,
                    }
                )
                open_trade = None
                current_position = 0
                current_weights = np.zeros(3, dtype=float)
                exited_today = True

            elif config.rebalance_daily and live_model is not None:
                weight_model = (
                    open_trade["entry_model"] if config.use_saved_model_during_trade else live_model
                )
                target_weights = _target_spread_weights(
                    current_position,
                    weight_model["beta_pair"],
                    weight_model["beta_market"],
                    config.gross_exposure,
                    config.trade_market_leg,
                )
                rebalance_buy, rebalance_sell, rebalance_cost_rate = _transaction_cost_details(
                    current_weights,
                    target_weights,
                    config.buy_transaction_cost,
                    config.sell_transaction_cost,
                )
                rebalance_cost_amount = equity * rebalance_cost_rate
                equity -= rebalance_cost_amount
                buy_turnover += rebalance_buy
                sell_turnover += rebalance_sell
                transaction_cost_amount += rebalance_cost_amount
                total_cost_amount += rebalance_cost_amount
                open_trade["trade_transaction_cost"] += rebalance_cost_amount
                current_weights = target_weights

        # Search for a new entry only while flat and only if this close is not
        # the final available execution date.
        if (
            current_position == 0
            and open_trade is None
            and signal_row is not None
            and not is_final_date
            and not exited_today
        ):
            desired_position = _entry_position(live_zscore, config.entry_zscore)
            if desired_position != 0 and live_model is not None:
                entry_block_reason = _entry_block_reason(live_model, config)
                if entry_block_reason is None:
                    target_weights = _target_spread_weights(
                        desired_position,
                        live_model["beta_pair"],
                        live_model["beta_market"],
                        config.gross_exposure,
                        config.trade_market_leg,
                    )
                    entry_buy, entry_sell, entry_cost_rate = _transaction_cost_details(
                        np.zeros(3, dtype=float),
                        target_weights,
                        config.buy_transaction_cost,
                        config.sell_transaction_cost,
                    )
                    entry_equity_before_cost = equity
                    entry_cost_amount = equity * entry_cost_rate
                    equity -= entry_cost_amount
                    buy_turnover += entry_buy
                    sell_turnover += entry_sell
                    transaction_cost_amount += entry_cost_amount
                    total_cost_amount += entry_cost_amount

                    current_position = int(desired_position)
                    current_weights = target_weights
                    entry_half_life = float(live_model["half_life"])
                    entry_max_holding_days = (
                        max(1, int(np.ceil(config.max_holding_half_lives * entry_half_life)))
                        if config.max_holding_half_lives is not None
                        else None
                    )
                    open_trade = {
                        "entry_date": date,
                        "entry_signal_date": signal_date,
                        "entry_simulation_step": simulation_step,
                        "direction": "LONG_SPREAD" if current_position == 1 else "SHORT_SPREAD",
                        "entry_zscore": live_zscore,
                        "entry_alpha": live_model["alpha"],
                        "entry_beta_pair": live_model["beta_pair"],
                        "entry_beta_market": live_model["beta_market"],
                        "entry_spread_mean": live_model["spread_mean"],
                        "entry_spread_std": live_model["spread_std"],
                        "entry_r_squared": live_model["r_squared"],
                        "entry_adf_pvalue": live_model["adf_pvalue"],
                        "entry_half_life": entry_half_life,
                        "max_holding_half_lives": config.max_holding_half_lives,
                        "entry_max_holding_days": entry_max_holding_days,
                        "entry_model": dict(live_model),
                        "stock1_weight": float(current_weights[0]),
                        "stock2_weight": float(current_weights[1]),
                        "market_weight": float(current_weights[2]),
                        "entry_equity_before_cost": entry_equity_before_cost,
                        "entry_equity_after_cost": equity,
                        "entry_gross_equity": gross_equity,
                        "entry_transaction_cost": entry_cost_amount,
                        "trade_transaction_cost": entry_cost_amount,
                        "model_mode": "SAVED_ENTRY_MODEL"
                        if config.use_saved_model_during_trade
                        else "DAILY_REFIT_MODEL",
                        "market_leg_traded": bool(config.trade_market_leg),
                    }

        net_return = equity / previous_equity - 1.0
        daily_rows.append(
            {
                "date": date,
                "signal_date": signal_date,
                "live_signal_zscore": live_zscore,
                "live_r_squared": float(live_model["r_squared"])
                if live_model is not None
                else float("nan"),
                "live_adf_pvalue": float(live_model["adf_pvalue"])
                if live_model is not None
                else float("nan"),
                "live_half_life": float(live_model["half_life"])
                if live_model is not None
                else float("nan"),
                "active_max_holding_days": (
                    open_trade["entry_max_holding_days"] if open_trade is not None else None
                ),
                "decision_zscore": decision_zscore,
                "entry_block_reason": entry_block_reason,
                "position": current_position,
                "stock1_weight": float(current_weights[0]),
                "stock2_weight": float(current_weights[1]),
                "market_weight": float(current_weights[2]),
                "gross_return": gross_return,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "transaction_cost": transaction_cost_amount,
                "net_return": net_return,
                "gross_equity": gross_equity,
                "equity": equity,
                "exit_reason": exit_reason,
            }
        )
        previous_prices = leg_prices
        previous_equity = equity

    daily = pd.DataFrame(daily_rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        internal_columns = [
            column
            for column in ["entry_model", "entry_simulation_step"]
            if column in trades.columns
        ]
        if internal_columns:
            trades = trades.drop(columns=internal_columns)

    net_returns = daily["net_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    number_of_periods = max(int(len(net_returns) - 1), 1)
    gross_total_return = float(gross_equity - 1.0)
    total_return = float(equity - 1.0)
    annualized_return = float(
        (max(equity, 1e-12) ** (config.annualization_days / number_of_periods)) - 1.0
    )
    daily_std = float(net_returns.std(ddof=1))
    annualized_volatility = (
        float(daily_std * np.sqrt(config.annualization_days))
        if np.isfinite(daily_std)
        else float("nan")
    )
    sharpe_ratio = (
        float(net_returns.mean() / daily_std * np.sqrt(config.annualization_days))
        if np.isfinite(daily_std) and daily_std > 1e-12
        else float("nan")
    )
    running_max = daily["equity"].cummax()
    drawdown = daily["equity"] / running_max - 1.0
    max_drawdown = float(drawdown.min())

    if trades.empty:
        win_rate = float("nan")
        average_trade_return = float("nan")
        median_trade_return = float("nan")
        profit_factor = float("nan")
        average_holding_days = float("nan")
        exit_counts: dict[str, int] = {}
    else:
        trade_returns = trades["net_trade_return"].astype(float)
        win_rate = float((trade_returns > 0).mean())
        average_trade_return = float(trade_returns.mean())
        median_trade_return = float(trade_returns.median())
        gains = float(trade_returns[trade_returns > 0].sum())
        losses = float(-trade_returns[trade_returns < 0].sum())
        profit_factor = (
            float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else float("nan"))
        )
        average_holding_days = float(trades["holding_days"].mean())
        exit_counts = {
            str(k): int(v) for k, v in trades["exit_reason"].value_counts().to_dict().items()
        }

    metrics: dict[str, object] = {
        "start_date": daily.index[0],
        "end_date": daily.index[-1],
        "rolling_years": float(config.rolling_years),
        "model_mode": "saved_entry_model"
        if config.use_saved_model_during_trade
        else "daily_refit_model",
        "trade_market_leg": bool(config.trade_market_leg),
        "rebalance_daily": bool(config.rebalance_daily),
        "max_holding_half_lives": config.max_holding_half_lives,
        "stop_loss_fraction": config.stop_loss_fraction,
        "relationship_break_r_squared": config.relationship_break_r_squared,
        "relationship_break_beta_change": config.relationship_break_beta_change,
        "entry_adf_threshold": config.adf_threshold,
        "buy_transaction_cost": float(config.buy_transaction_cost),
        "sell_transaction_cost": float(config.sell_transaction_cost),
        "gross_exposure": float(config.gross_exposure),
        "gross_total_return": gross_total_return,
        "total_return": total_return,
        "transaction_cost_drag": float(gross_equity - equity),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "number_of_trades": int(len(trades)),
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "median_trade_return": median_trade_return,
        "profit_factor": profit_factor,
        "average_holding_days": average_holding_days,
        "exposure_days": int((daily["position"] != 0).sum()),
        "total_buy_turnover": float(daily["buy_turnover"].sum()),
        "total_sell_turnover": float(daily["sell_turnover"].sum()),
        "total_transaction_cost": float(total_cost_amount),
        "gross_final_equity": float(gross_equity),
        "final_equity": float(equity),
        "mean_reversion_exits": int(exit_counts.get("MEAN_REVERSION", 0)),
        "stop_loss_exits": int(exit_counts.get("STOP_LOSS", 0)),
        "max_holding_exits": int(exit_counts.get("MAX_HOLDING", 0)),
        "relationship_break_exits": int(
            sum(v for k, v in exit_counts.items() if k.startswith("RELATIONSHIP_"))
        ),
        "adf_blocked_entry_days": int((daily["entry_block_reason"] == "ENTRY_ADF").sum()),
        "r_squared_blocked_entry_days": int(
            (daily["entry_block_reason"] == "ENTRY_R_SQUARED").sum()
        ),
        "half_life_blocked_entry_days": int(
            (daily["entry_block_reason"] == "ENTRY_HALF_LIFE").sum()
        ),
        "total_blocked_entry_days": int(daily["entry_block_reason"].notna().sum()),
        "final_date_exits": int(exit_counts.get("FINAL_DATE", 0)),
    }

    if verbose:
        print("\nDaily walk-forward backtest")
        print("Rolling window:", rolling_years, "years")
        print("Model while open:", metrics["model_mode"])
        print("Market hedge leg traded:", metrics["trade_market_leg"])
        print("Backtest period:", daily.index[0].date(), "to", daily.index[-1].date())
        print("Total return:", round(total_return, 4))
        print("Sharpe ratio:", round(sharpe_ratio, 4) if np.isfinite(sharpe_ratio) else "N/A")
        print("Maximum drawdown:", round(max_drawdown, 4))
        print("Completed trades:", len(trades))
        print("ADF-blocked entry days:", metrics["adf_blocked_entry_days"])
        print("R-squared-blocked entry days:", metrics["r_squared_blocked_entry_days"])
        print("Half-life-blocked entry days:", metrics["half_life_blocked_entry_days"])
        print("Exit reasons:", exit_counts)
        print("Transaction costs paid:", round(total_cost_amount, 6))

    return PairBacktestResult(daily=daily, signals=signals, trades=trades, metrics=metrics)


def current_trade_signal(
    stock1: str,
    stock2: str,
    zscore: float,
    entry_zscore: float = 2.0,
    market_ticker: str | None = None,
    beta_pair: float | None = None,
    beta_market: float | None = None,
    trade_market_leg: bool = True,
) -> str:
    direction = _entry_position(float(zscore), float(entry_zscore))
    if direction == 0:
        return "NO DATA" if not np.isfinite(float(zscore)) else "NO TRADE"

    if market_ticker is None or beta_pair is None or beta_market is None:
        return (
            f"LONG {stock1}, SHORT {stock2}" if direction == 1 else f"SHORT {stock1}, LONG {stock2}"
        )

    weights = _target_spread_weights(
        direction,
        float(beta_pair),
        float(beta_market),
        1.0,
        bool(trade_market_leg),
    )
    tickers = [stock1, stock2, market_ticker]
    legs = []
    for ticker, weight in zip(tickers, weights, strict=True):
        if abs(float(weight)) <= 1e-12:
            continue
        side = "LONG" if weight > 0 else "SHORT"
        legs.append(f"{side} {ticker} ({abs(weight):.3f})")
    return ", ".join(legs)


def pair_quality_score(
    adf_pvalue: float,
    r_squared: float,
    beta_stability_ratio: float,
    market_beta_stability_ratio: float,
    half_life: float,
    signal_zscore: float,
    adf_threshold: float | None = 0.05,
    max_beta_stability: float = 0.50,
    min_half_life: float = 2.0,
    max_half_life: float = 60.0,
    entry_zscore: float = 2.0,
) -> float:
    """Score residual quality and hedge stability on a 0-100 scale."""
    score = 0.0
    if adf_threshold is not None and np.isfinite(adf_pvalue) and adf_pvalue <= adf_threshold:
        if adf_threshold > 0:
            score += max(0.0, 25.0 * (1.0 - adf_pvalue / adf_threshold))
        elif adf_pvalue <= 0:
            score += 25.0
    if np.isfinite(r_squared):
        score += min(15.0, max(0.0, 15.0 * r_squared))
    if np.isfinite(beta_stability_ratio) and beta_stability_ratio <= max_beta_stability:
        score += max(0.0, 10.0 * (1.0 - beta_stability_ratio / max_beta_stability))
    if (
        np.isfinite(market_beta_stability_ratio)
        and market_beta_stability_ratio <= max_beta_stability
    ):
        score += max(0.0, 10.0 * (1.0 - market_beta_stability_ratio / max_beta_stability))
    if np.isfinite(half_life) and min_half_life <= half_life <= max_half_life:
        score += 10.0
    if np.isfinite(signal_zscore) and abs(signal_zscore) >= entry_zscore:
        score += 5.0
    # The remaining components total 75 raw points. Rescaling preserves their
    # relative weights while keeping the public score on a 0-100 scale.
    return round(float(score * (100.0 / 75.0)), 4)


def analyze_pair_strategy_from_prices(
    prices_all: pd.DataFrame,
    stock1: str,
    stock2: str,
    market_ticker: str = "^JKSE",
    backtest_start: str | pd.Timestamp | None = None,
    backtest_end: str | pd.Timestamp | None = None,
    signal_date: str | pd.Timestamp | None = None,
    rolling_years: float = 2.0,
    rolling_beta_window: int = 252,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.0,
    buy_transaction_cost: float = 0.002,
    sell_transaction_cost: float = 0.002,
    gross_exposure: float = 1.0,
    min_training_observations: int = 252,
    annualization_days: int = 252,
    use_saved_model_during_trade: bool = True,
    trade_market_leg: bool = True,
    rebalance_daily: bool = False,
    max_holding_half_lives: float | None = 2.0,
    stop_loss_fraction: float | None = 0.10,
    relationship_break_r_squared: float | None = 0.60,
    relationship_break_beta_change: float | None = 0.50,
    adf_threshold: float | None = 0.05,
    min_half_life: float = 2.0,
    max_half_life: float = 60.0,
    max_beta_stability: float = 0.50,
    min_r_squared: float = 0.60,
    verbose: bool = False,
) -> PairStrategyResult:
    """Run the latest signal analysis and matching three-leg walk-forward backtest."""
    stock1 = stock1.upper().strip()
    stock2 = stock2.upper().strip()
    market_ticker = market_ticker.upper().strip()
    tickers = [stock1, stock2, market_ticker]
    prices = prepare_price_frame(prices_all, tickers).dropna(how="any")
    if len(prices) < min_training_observations + 2:
        raise ValueError("Not enough overlapping data for analysis and backtesting.")

    requested_signal = prices.index[-1] if signal_date is None else pd.Timestamp(signal_date)
    available_signal_dates = prices.index[prices.index <= requested_signal]
    if len(available_signal_dates) < 2:
        raise ValueError("No usable signal date with a prior training day.")
    signal_date_actual = available_signal_dates[-1]
    train_end_actual = available_signal_dates[-2]
    train_start_target = signal_date_actual - pd.Timedelta(
        days=int(round(365.25 * float(rolling_years)))
    )
    train_prices = prices.loc[
        (prices.index >= train_start_target) & (prices.index <= train_end_actual)
    ]
    if len(train_prices) < min_training_observations:
        raise ValueError(
            "The final signal does not have enough observations in its rolling-year window."
        )

    effective_rolling_beta_window = min(int(rolling_beta_window), len(train_prices))
    effective_rolling_beta_window = max(2, effective_rolling_beta_window)
    analysis = analyze_pair_from_prices(
        prices_all=prices,
        stock1=stock1,
        stock2=stock2,
        train_start=train_prices.index[0],
        train_end=train_end_actual,
        signal_date=signal_date_actual,
        market_ticker=market_ticker,
        rolling_window=effective_rolling_beta_window,
        entry_zscore=entry_zscore,
        trade_market_leg=trade_market_leg,
        show_plots=False,
        verbose=verbose,
    )

    backtest = run_daily_walk_forward_backtest_from_prices(
        prices_all=prices,
        stock1=stock1,
        stock2=stock2,
        market_ticker=market_ticker,
        backtest_start=backtest_start,
        backtest_end=backtest_end if backtest_end is not None else signal_date_actual,
        rolling_years=rolling_years,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        buy_transaction_cost=buy_transaction_cost,
        sell_transaction_cost=sell_transaction_cost,
        gross_exposure=gross_exposure,
        min_training_observations=min_training_observations,
        annualization_days=annualization_days,
        use_saved_model_during_trade=use_saved_model_during_trade,
        trade_market_leg=trade_market_leg,
        rebalance_daily=rebalance_daily,
        max_holding_half_lives=max_holding_half_lives,
        stop_loss_fraction=stop_loss_fraction,
        relationship_break_r_squared=relationship_break_r_squared,
        relationship_break_beta_change=relationship_break_beta_change,
        adf_threshold=adf_threshold,
        verbose=verbose,
    )

    passes_adf = bool(
        adf_threshold is None
        or (np.isfinite(analysis.adf_pvalue) and analysis.adf_pvalue <= adf_threshold)
    )
    passes_half_life = bool(
        np.isfinite(analysis.half_life) and min_half_life <= analysis.half_life <= max_half_life
    )
    passes_beta_stability = bool(
        np.isfinite(analysis.beta_pair_stability_ratio)
        and analysis.beta_pair_stability_ratio <= max_beta_stability
    )
    passes_market_beta_stability = bool(
        np.isfinite(analysis.beta_market_stability_ratio)
        and analysis.beta_market_stability_ratio <= max_beta_stability
    )
    passes_r_squared = bool(np.isfinite(analysis.r_squared) and analysis.r_squared >= min_r_squared)
    good_pair = bool(
        passes_adf
        and passes_half_life
        and passes_beta_stability
        and passes_market_beta_stability
        and passes_r_squared
    )
    quality_score = pair_quality_score(
        analysis.adf_pvalue,
        analysis.r_squared,
        analysis.beta_pair_stability_ratio,
        analysis.beta_market_stability_ratio,
        analysis.half_life,
        analysis.signal_zscore,
        adf_threshold=adf_threshold,
        max_beta_stability=max_beta_stability,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        entry_zscore=entry_zscore,
    )

    signal_direction = _entry_position(analysis.signal_zscore, entry_zscore)
    signal_weights = (
        _target_spread_weights(
            signal_direction,
            analysis.beta_pair,
            analysis.beta_market,
            gross_exposure,
            trade_market_leg,
        )
        if signal_direction != 0
        else np.zeros(3, dtype=float)
    )

    summary = {
        "stock1": stock1,
        "stock2": stock2,
        "market_ticker": market_ticker,
        "train_start": str(pd.Timestamp(analysis.train_start).date()),
        "train_end": str(pd.Timestamp(analysis.train_end).date()),
        "signal_date": str(pd.Timestamp(analysis.signal_date).date()),
        "observations": int(
            len(analysis.all_zscore.loc[analysis.train_start : analysis.train_end].dropna())
        ),
        "alpha": float(analysis.alpha),
        "beta_pair": float(analysis.beta_pair),
        "beta_market": float(analysis.beta_market),
        "r_squared": float(analysis.r_squared),
        "half_life": float(analysis.half_life),
        "spread_mean": float(analysis.spread_mean),
        "spread_std": float(analysis.spread_std),
        "signal_spread": float(analysis.signal_spread),
        "signal_zscore": float(analysis.signal_zscore),
        "abs_signal_zscore": float(abs(analysis.signal_zscore)),
        "signal_stock1_weight": float(signal_weights[0]),
        "signal_stock2_weight": float(signal_weights[1]),
        "signal_market_weight": float(signal_weights[2]),
        "adf_statistic": float(analysis.adf_statistic),
        "adf_pvalue": float(analysis.adf_pvalue),
        "beta_pair_stability_ratio": float(analysis.beta_pair_stability_ratio),
        "beta_market_stability_ratio": float(analysis.beta_market_stability_ratio),
        "passes_adf": passes_adf,
        "passes_half_life": passes_half_life,
        "passes_beta_stability": passes_beta_stability,
        "passes_market_beta_stability": passes_market_beta_stability,
        "passes_r_squared": passes_r_squared,
        "has_entry_signal": bool(abs(analysis.signal_zscore) >= entry_zscore),
        "trade_signal": current_trade_signal(
            stock1,
            stock2,
            analysis.signal_zscore,
            entry_zscore,
            market_ticker=market_ticker,
            beta_pair=analysis.beta_pair,
            beta_market=analysis.beta_market,
            trade_market_leg=trade_market_leg,
        ),
        "good_pair": good_pair,
        "quality_score": quality_score,
    }
    for key, value in backtest.metrics.items():
        if isinstance(value, pd.Timestamp):
            summary[f"backtest_{key}"] = str(value.date())
        else:
            summary[f"backtest_{key}"] = value

    return PairStrategyResult(analysis=analysis, backtest=backtest, summary=summary)


def run_daily_walk_forward_backtest(
    stock1: str,
    stock2: str,
    market_ticker: str = "^JKSE",
    period: str = "max",
    interval: str = "1d",
    **backtest_kwargs,
) -> PairBacktestResult:
    """Download the three required price series and run the central backtest."""
    tickers = [stock1, stock2, market_ticker]
    prices = download_price_data(
        tickers, period=period, interval=interval, progress=False, threads=True
    )
    return run_daily_walk_forward_backtest_from_prices(
        prices_all=prices,
        stock1=stock1,
        stock2=stock2,
        market_ticker=market_ticker,
        **backtest_kwargs,
    )


def run_pair_strategy(
    stock1: str,
    stock2: str,
    market_ticker: str = "^JKSE",
    period: str = "max",
    interval: str = "1d",
    **strategy_kwargs,
) -> PairStrategyResult:
    """Download once, then run both the latest signal analysis and backtest."""
    tickers = [stock1, stock2, market_ticker]
    prices = download_price_data(
        tickers, period=period, interval=interval, progress=False, threads=True
    )
    return analyze_pair_strategy_from_prices(
        prices_all=prices,
        stock1=stock1,
        stock2=stock2,
        market_ticker=market_ticker,
        **strategy_kwargs,
    )


def parse_date(raw: str) -> pd.Timestamp | None:
    raw = raw.strip()
    return pd.Timestamp(raw) if raw else None


def collect_user_inputs() -> PairAnalysisConfig:
    stock1 = input("Input stock/coin 1: ").upper().strip()
    stock2 = input("Input stock/coin 2: ").upper().strip()

    market_ticker = (
        input("Market ticker (press Enter = ^JKSE for IHSG, or use BTC-USD for crypto): ")
        .upper()
        .strip()
    )
    if market_ticker == "":
        market_ticker = "^JKSE"

    train_start = parse_date(input("Train start date YYYY-MM-DD (press Enter = earliest data): "))
    train_end = parse_date(input("Train end date YYYY-MM-DD (press Enter = latest data): "))
    signal_date = parse_date(input("Signal date YYYY-MM-DD (press Enter = latest data): "))

    rolling_window_raw = input("Rolling beta window in trading days (press Enter = 252): ").strip()
    rolling_window = int(rolling_window_raw) if rolling_window_raw else 252

    return PairAnalysisConfig(
        stock1=stock1,
        stock2=stock2,
        market_ticker=market_ticker,
        train_start=train_start,
        train_end=train_end,
        signal_date=signal_date,
        rolling_window=rolling_window,
    )


def run_pair_analysis(
    stock1: str,
    stock2: str,
    train_start: str | pd.Timestamp | None = None,
    train_end: str | pd.Timestamp | None = None,
    signal_date: str | pd.Timestamp | None = None,
    market_ticker: str = "^JKSE",
    rolling_window: int = 252,
    entry_zscore: float = 2.0,
    trade_market_leg: bool = True,
    show_plots: bool = True,
    print_all_rolling_betas: bool = True,
    prices_all: pd.DataFrame | None = None,
    verbose: bool = True,
) -> PairAnalysisResult:
    """
    Single-call function version.

    For one custom pair, this still downloads data automatically:
        run_pair_analysis("BBCA.JK", "BMRI.JK", market_ticker="^JKSE")

    For fast screeners, pass pre-downloaded prices_all so no pair-level download happens.
    """
    config = PairAnalysisConfig(
        stock1=stock1,
        stock2=stock2,
        train_start=_to_timestamp(train_start),
        train_end=_to_timestamp(train_end),
        signal_date=_to_timestamp(signal_date),
        market_ticker=market_ticker,
        rolling_window=rolling_window,
        entry_zscore=entry_zscore,
        trade_market_leg=trade_market_leg,
        show_plots=show_plots,
        print_all_rolling_betas=print_all_rolling_betas,
        verbose=verbose,
    )
    return PairTradingAnalyzer(config, prices_all=prices_all).run()


def analyze_pair_from_prices(
    prices_all: pd.DataFrame,
    stock1: str,
    stock2: str,
    train_start: str | pd.Timestamp | None = None,
    train_end: str | pd.Timestamp | None = None,
    signal_date: str | pd.Timestamp | None = None,
    market_ticker: str = "^JKSE",
    rolling_window: int = 252,
    entry_zscore: float = 2.0,
    trade_market_leg: bool = True,
    show_plots: bool = False,
    verbose: bool = False,
) -> PairAnalysisResult:
    """Fast path used by screeners. It reuses an already-downloaded price table."""
    return run_pair_analysis(
        stock1=stock1,
        stock2=stock2,
        train_start=train_start,
        train_end=train_end,
        signal_date=signal_date,
        market_ticker=market_ticker,
        rolling_window=rolling_window,
        entry_zscore=entry_zscore,
        trade_market_leg=trade_market_leg,
        show_plots=show_plots,
        print_all_rolling_betas=False,
        prices_all=prices_all,
        verbose=verbose,
    )


def _parse_yes_no(raw: str, default: bool) -> bool:
    raw = raw.strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes", "true", "1"}:
        return True
    if raw in {"n", "no", "false", "0"}:
        return False
    raise ValueError("Please answer y or n.")


def _parse_optional_number(raw: str, default: float | None) -> float | None:
    raw = raw.strip().lower()
    if not raw:
        return default
    if raw in {"off", "of", "none", "disable", "disabled", "0"}:
        return None
    return float(raw)


def main() -> None:
    stock1 = input("Input stock/coin 1: ").upper().strip()
    stock2 = input("Input stock/coin 2: ").upper().strip()
    market_ticker = input("Market ticker (press Enter = ^JKSE): ").upper().strip() or "^JKSE"
    backtest_start = parse_date(
        input("Backtest start YYYY-MM-DD (press Enter = earliest valid date): ")
    )
    backtest_end = parse_date(input("Backtest end YYYY-MM-DD (press Enter = latest): "))
    signal_date = parse_date(input("Latest signal date YYYY-MM-DD (press Enter = latest): "))

    rolling_years_raw = input("Walk-forward rolling window in years (press Enter = 2): ").strip()
    rolling_years = float(rolling_years_raw) if rolling_years_raw else 2.0
    rolling_beta_raw = input("Rolling beta stability window in days (press Enter = 252): ").strip()
    rolling_beta_window = int(rolling_beta_raw) if rolling_beta_raw else 252
    entry_raw = input("Entry z-score (press Enter = 2): ").strip()
    entry_zscore = float(entry_raw) if entry_raw else 2.0
    exit_raw = input("Exit z-score toward mean (press Enter = 0): ").strip()
    exit_zscore = float(exit_raw) if exit_raw else 0.0
    buy_cost_raw = input("Buy transaction cost decimal (press Enter = 0.002): ").strip()
    buy_cost = float(buy_cost_raw) if buy_cost_raw else 0.002
    sell_cost_raw = input("Sell transaction cost decimal (press Enter = 0.002): ").strip()
    sell_cost = float(sell_cost_raw) if sell_cost_raw else 0.002

    use_saved_model = _parse_yes_no(
        input("Freeze/save entry model while a trade is open? (press Enter = yes): "),
        True,
    )
    trade_market_leg = _parse_yes_no(
        input("Trade the market factor as the third hedge leg? (press Enter = yes): "),
        True,
    )
    rebalance_daily = _parse_yes_no(
        input("Rebalance all active legs daily? (press Enter = no): "),
        False,
    )
    max_holding_half_lives = _parse_optional_number(
        input("Maximum holding as half-life multiple (press Enter = 2, type off to disable): "),
        2.0,
    )
    stop_loss = _parse_optional_number(
        input("Stop-loss fraction (press Enter = 0.10, type off to disable): "),
        0.10,
    )
    relationship_r2 = _parse_optional_number(
        input("Relationship-break minimum R-squared (press Enter = 0.60, type off to disable): "),
        0.60,
    )
    relationship_beta = _parse_optional_number(
        input("Relationship-break maximum beta change (press Enter = 0.50, type off to disable): "),
        0.50,
    )
    adf_threshold = _parse_optional_number(
        input("Maximum entry ADF p-value (press Enter = 0.05, type off to disable): "),
        0.05,
    )

    annualization_days = (
        365 if all(ticker.endswith("-USD") for ticker in [stock1, stock2, market_ticker]) else 252
    )

    result = run_pair_strategy(
        stock1=stock1,
        stock2=stock2,
        market_ticker=market_ticker,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        signal_date=signal_date,
        rolling_years=rolling_years,
        rolling_beta_window=rolling_beta_window,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        buy_transaction_cost=buy_cost,
        sell_transaction_cost=sell_cost,
        annualization_days=annualization_days,
        use_saved_model_during_trade=use_saved_model,
        trade_market_leg=trade_market_leg,
        rebalance_daily=rebalance_daily,
        max_holding_half_lives=max_holding_half_lives,
        stop_loss_fraction=stop_loss,
        relationship_break_r_squared=relationship_r2,
        relationship_break_beta_change=relationship_beta,
        adf_threshold=adf_threshold,
        verbose=True,
    )

    print("\nCurrent three-leg trade signal")
    print(result.summary["trade_signal"])

    print("\nBacktest summary")
    for key, value in result.backtest.metrics.items():
        print(f"{key}: {value}")

    if not result.backtest.trades.empty:
        print("\nCompleted trades")
        display_columns = [
            "entry_date",
            "exit_date",
            "direction",
            "entry_zscore",
            "entry_adf_pvalue",
            "entry_r_squared",
            "entry_half_life",
            "entry_max_holding_days",
            "exit_decision_zscore",
            "exit_reason",
            "net_trade_return",
            "holding_days",
            "holding_calendar_days",
            "stock1_weight",
            "stock2_weight",
            "market_weight",
        ]
        print(result.backtest.trades[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
