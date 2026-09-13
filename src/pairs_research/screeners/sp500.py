"""
Fast S&P 500 Pair Screener
==========================

This screener downloads S&P 500 price data once, then reuses the shared price
matrix for every pair by calling your existing pair_trading_signal_analyzer.py.

Required file in the same folder:
    pair_trading_signal_analyzer.py

Recommended quick test:
    python sp500_pair_screener_fast.py --max-pairs 25

Full S&P 500 run:
    python sp500_pair_screener_fast.py --start 2024-01-01 --rolling-window 126 --top 100

Optional sector-only run:
    python sp500_pair_screener_fast.py --sector "Financials" --start 2024-01-01

Outputs:
    sp500_pair_results.csv
    sp500_good_pairs.csv
    sp500_errors.csv
    sp500_tickers_used.csv
    sp500_metadata_used.csv

Notes:
    - Full S&P 500 means roughly 500 choose 2 = ~125,000 pairs.
    - That is much faster than downloading data per pair, but still computationally heavy.
    - For debugging, use --max-pairs first.
    - For faster research, use --sector or --prefilter-top-corr.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
import time
import urllib.request
from datetime import datetime
from io import StringIO
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# Small fallback only. The normal/default path fetches the current S&P 500 table from Wikipedia.
FALLBACK_SP500_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "BRK-B",
    "AVGO",
    "LLY",
    "JPM",
    "TSLA",
    "V",
    "UNH",
    "XOM",
    "MA",
    "COST",
    "WMT",
    "PG",
    "NFLX",
]


def create_run_output_dir(base_dir: str) -> Path:
    """Create results/<executed script name - date.time>/ for this run."""
    script_name = Path(sys.argv[0]).stem or Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
    run_dir = Path(base_dir) / f"{script_name} - {timestamp}"

    # Avoid overwriting a folder if the same script starts twice in one second.
    candidate = run_dir
    counter = 2
    while candidate.exists():
        candidate = Path(f"{run_dir} - {counter}")
        counter += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def default_analyzer_path() -> str:
    return str(Path(__file__).parents[1] / "analyzer.py")


def yahoo_normalize_us_ticker(raw: str) -> str:
    """Convert S&P ticker format to Yahoo Finance format, e.g. BRK.B -> BRK-B."""
    ticker = str(raw).strip().upper()
    if not ticker:
        return ""
    if ticker.startswith("^"):
        return ticker
    return ticker.replace(".", "-")


def normalize_tickers(raw_tickers: list[str]) -> list[str]:
    clean = [yahoo_normalize_us_ticker(ticker) for ticker in raw_tickers]
    clean = [ticker for ticker in clean if ticker]
    return list(dict.fromkeys(clean))


def fetch_sp500_metadata() -> pd.DataFrame:
    """Fetch current S&P 500 constituents and metadata from Wikipedia.

    pandas.read_html(url) can get HTTP 403 from Wikipedia because it uses a
    plain/default request. Fetch the HTML ourselves with a browser-like
    User-Agent, then let pandas parse the already-downloaded HTML.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        request = urllib.request.Request(SP500_WIKIPEDIA_URL, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
        tables = pd.read_html(StringIO(html))
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch the S&P 500 table from Wikipedia, even with a browser-like request. "
            "Use --ticker-source fallback, --tickers, or --tickers-file instead."
        ) from exc

    if not tables:
        raise RuntimeError("Wikipedia returned no tables for the S&P 500 page.")

    df = tables[0].copy()
    if "Symbol" not in df.columns:
        raise RuntimeError("Could not find a 'Symbol' column in the S&P 500 table.")

    rename_map = {
        "Symbol": "ticker_raw",
        "Security": "company",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Headquarters Location": "headquarters",
        "Date added": "date_added",
        "CIK": "cik",
        "Founded": "founded",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["ticker"] = df["ticker_raw"].astype(str).map(yahoo_normalize_us_ticker)
    df = df[df["ticker"].astype(bool)].copy()
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    for col in [
        "company",
        "sector",
        "sub_industry",
        "headquarters",
        "date_added",
        "cik",
        "founded",
    ]:
        if col not in df.columns:
            df[col] = ""

    keep_cols = [
        "ticker",
        "ticker_raw",
        "company",
        "sector",
        "sub_industry",
        "headquarters",
        "date_added",
        "cik",
        "founded",
    ]
    return df[keep_cols]


def load_tickers_and_metadata(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame]:
    if args.tickers:
        tickers = normalize_tickers(args.tickers.split(","))
        meta = pd.DataFrame({"ticker": tickers})
        return tickers, meta

    if args.tickers_file:
        path = Path(args.tickers_file)
        if not path.exists():
            raise FileNotFoundError(f"Ticker file not found: {path}")
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            raw = (
                df["ticker"].astype(str).tolist()
                if "ticker" in df.columns
                else df.iloc[:, 0].astype(str).tolist()
            )
            tickers = normalize_tickers(raw)
            if "ticker" in df.columns:
                df = df.copy()
                df["ticker"] = df["ticker"].astype(str).map(yahoo_normalize_us_ticker)
                meta = df.drop_duplicates(subset=["ticker"])
            else:
                meta = pd.DataFrame({"ticker": tickers})
            return tickers, meta

        raw = path.read_text(encoding="utf-8").replace("\n", ",").split(",")
        tickers = normalize_tickers(raw)
        meta = pd.DataFrame({"ticker": tickers})
        return tickers, meta

    if args.ticker_source == "fallback":
        tickers = FALLBACK_SP500_TICKERS.copy()
        meta = pd.DataFrame({"ticker": tickers})
        return tickers, meta

    meta = fetch_sp500_metadata()

    if args.sector:
        wanted = args.sector.strip().lower()
        meta = meta[meta["sector"].astype(str).str.lower() == wanted].copy()
        if meta.empty:
            raise ValueError(f"No S&P 500 tickers found for sector: {args.sector!r}")

    if args.sub_industry_contains:
        needle = args.sub_industry_contains.strip().lower()
        meta = meta[
            meta["sub_industry"].astype(str).str.lower().str.contains(needle, na=False)
        ].copy()
        if meta.empty:
            raise ValueError(
                f"No S&P 500 tickers found matching sub-industry text: {args.sub_industry_contains!r}"
            )

    tickers = meta["ticker"].astype(str).tolist()
    return tickers, meta


def load_analyzer(analyzer_path: str) -> ModuleType:
    path = Path(analyzer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Analyzer file not found: {path}\n"
            "Put pair_trading_signal_analyzer.py in the same folder, or pass --analyzer-path."
        )

    spec = importlib.util.spec_from_file_location("pair_trading_signal_analyzer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import analyzer from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["pair_trading_signal_analyzer"] = module
    spec.loader.exec_module(module)

    required = ["download_price_data", "analyze_pair_strategy_from_prices"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Your analyzer is missing fast functions: {missing}. "
            "Use the updated pair_trading_signal_analyzer.py."
        )
    return module


def clean_price_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Defensive screener-level cleaning before sending prices to the analyzer."""
    prices = prices.copy()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.mask(prices <= 0)
    # Do not drop all rows here; each pair will drop rows based on its own three needed tickers.
    return prices


def download_prices_in_chunks(
    module: ModuleType, tickers: list[str], args: argparse.Namespace
) -> pd.DataFrame:
    chunks = [
        tickers[i : i + args.download_chunk_size]
        for i in range(0, len(tickers), args.download_chunk_size)
    ]
    frames: list[pd.DataFrame] = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Downloading chunk {i}/{len(chunks)}: {len(chunk)} tickers")
        frame = module.download_price_data(
            chunk,
            period=args.period,
            interval="1d",
            progress=False,
            threads=True,
        )
        frames.append(frame)
        if args.download_sleep > 0 and i < len(chunks):
            time.sleep(args.download_sleep)

    if not frames:
        raise ValueError("No price data frames were downloaded.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = clean_price_table(prices)
    return prices


def format_elapsed(seconds: float) -> str:
    seconds = float(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.2f}s"
    return f"{secs:.2f}s"


def validate_downloaded_tickers(
    prices: pd.DataFrame, tickers: list[str], min_days: int
) -> list[str]:
    valid = []
    invalid = []
    for ticker in tickers:
        if ticker in prices.columns and prices[ticker].dropna().shape[0] >= min_days:
            valid.append(ticker)
        else:
            invalid.append(ticker)
    if invalid:
        print("Removed no-data/low-data tickers:", ", ".join(invalid))
    if len(valid) < 2:
        raise ValueError("Fewer than 2 valid tickers remain after data validation.")
    return valid


def build_pairs(
    tickers: list[str], prices: pd.DataFrame, args: argparse.Namespace
) -> list[tuple[str, str]]:
    all_pairs = list(itertools.combinations(tickers, 2))

    if args.prefilter_top_corr is None:
        pairs = all_pairs
    else:
        # Optional speed filter: only keep each ticker's top-N most correlated names by daily return.
        print(
            f"Prefiltering pairs by top {args.prefilter_top_corr} absolute return correlations per ticker..."
        )
        subset = prices[tickers].copy()
        subset = subset.replace([np.inf, -np.inf], np.nan).mask(subset <= 0)
        log_prices = np.log(subset)
        returns = log_prices.diff().dropna(how="all")
        corr = returns.corr().abs()

        pair_set: set[tuple[str, str]] = set()
        for ticker in tickers:
            if ticker not in corr.columns:
                continue
            ranked = (
                corr[ticker]
                .drop(labels=[ticker], errors="ignore")
                .dropna()
                .sort_values(ascending=False)
            )
            for other in ranked.head(args.prefilter_top_corr).index:
                pair_set.add(tuple(sorted((ticker, str(other)))))
        pairs = sorted(pair_set)

    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    return pairs


def metadata_lookup(meta: pd.DataFrame) -> dict[str, dict]:
    if meta.empty or "ticker" not in meta.columns:
        return {}
    meta = meta.copy()
    meta["ticker"] = meta["ticker"].astype(str).map(yahoo_normalize_us_ticker)
    meta = meta.drop_duplicates(subset=["ticker"])
    return meta.set_index("ticker").to_dict(orient="index")


def meta_value(meta_map: dict[str, dict], ticker: str, key: str) -> str:
    value = meta_map.get(ticker, {}).get(key, "")
    if pd.isna(value):
        return ""
    return str(value)


def analyze_one_pair(
    module: ModuleType,
    prices: pd.DataFrame,
    stock1: str,
    stock2: str,
    args: argparse.Namespace,
    meta_map: dict[str, dict],
) -> dict:
    strategy = module.analyze_pair_strategy_from_prices(
        prices_all=prices,
        stock1=stock1,
        stock2=stock2,
        market_ticker=args.market_ticker,
        backtest_start=args.start,
        backtest_end=args.train_end if args.train_end is not None else args.signal_date,
        signal_date=args.signal_date,
        rolling_years=args.rolling_years,
        rolling_beta_window=args.rolling_window,
        entry_zscore=args.entry_zscore,
        exit_zscore=args.exit_zscore,
        buy_transaction_cost=args.buy_transaction_cost,
        sell_transaction_cost=args.sell_transaction_cost,
        gross_exposure=args.gross_exposure,
        min_training_observations=args.min_training_observations,
        annualization_days=252,
        use_saved_model_during_trade=args.use_saved_model_during_trade,
        trade_market_leg=args.trade_market_leg,
        rebalance_daily=args.rebalance_daily,
        max_holding_half_lives=args.max_holding_half_lives,
        stop_loss_fraction=args.stop_loss_fraction,
        relationship_break_r_squared=args.relationship_break_r_squared,
        relationship_break_beta_change=args.relationship_break_beta_change,
        coint_threshold=args.coint_threshold,
        adf_threshold=args.adf_threshold,
        min_half_life=args.min_half_life,
        max_half_life=args.max_half_life,
        max_beta_stability=args.max_beta_stability,
        min_r_squared=args.min_r_squared,
        verbose=False,
    )
    row = dict(strategy.summary)
    row.update(
        {
            "company1": meta_value(meta_map, stock1, "company"),
            "company2": meta_value(meta_map, stock2, "company"),
            "sector1": meta_value(meta_map, stock1, "sector"),
            "sector2": meta_value(meta_map, stock2, "sector"),
            "sub_industry1": meta_value(meta_map, stock1, "sub_industry"),
            "sub_industry2": meta_value(meta_map, stock2, "sub_industry"),
        }
    )
    return row


def screen_pairs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_start = time.perf_counter()
    module = load_analyzer(args.analyzer_path)
    tickers, meta = load_tickers_and_metadata(args)
    tickers = normalize_tickers(tickers)
    args.market_ticker = yahoo_normalize_us_ticker(args.market_ticker)

    if args.skip_market_leg:
        tickers = [ticker for ticker in tickers if ticker != args.market_ticker]

    download_tickers = list(dict.fromkeys(tickers + [args.market_ticker]))
    print(f"Loaded {len(tickers)} candidate S&P 500 tickers")
    if args.sector:
        print(f"Sector filter: {args.sector}")
    if args.sub_industry_contains:
        print(f"Sub-industry contains: {args.sub_industry_contains}")

    print(f"Downloading {len(download_tickers)} stock/market tickers once...")
    prices = download_prices_in_chunks(module, download_tickers, args)
    tickers = validate_downloaded_tickers(prices, tickers, args.min_validation_days)

    pairs = build_pairs(tickers, prices, args)
    full_pair_count = len(list(itertools.combinations(tickers, 2)))

    print(f"Testing {len(tickers)} valid S&P 500 tickers = {len(pairs)} pairs")
    if len(pairs) != full_pair_count:
        print(
            f"Note: Full pair count would be {full_pair_count}; filters/prefilter/max-pairs reduced it."
        )
    print("Fast mode: Yahoo data was downloaded once and reused for every pair.")

    meta_map = metadata_lookup(meta)
    rows: list[dict] = []
    errors: list[dict] = []
    pair_loop_start = time.perf_counter()

    for i, (stock1, stock2) in enumerate(pairs, start=1):
        if i == 1 or i % args.progress_every == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] {stock1} vs {stock2}")
        try:
            rows.append(analyze_one_pair(module, prices, stock1, stock2, args, meta_map))
        except Exception as exc:
            errors.append(
                {
                    "stock1": stock1,
                    "stock2": stock2,
                    "company1": meta_value(meta_map, stock1, "company"),
                    "company2": meta_value(meta_map, stock2, "company"),
                    "sector1": meta_value(meta_map, stock1, "sector"),
                    "sector2": meta_value(meta_map, stock2, "sector"),
                    "error": str(exc),
                }
            )

    pair_loop_elapsed = time.perf_counter() - pair_loop_start

    results = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)
    if not results.empty:
        results = results.sort_values(
            by=["good_pair", "backtest_sharpe_ratio", "quality_score", "abs_signal_zscore"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    output_dir = create_run_output_dir(args.output_dir)

    all_path = output_dir / "sp500_pair_results.csv"
    good_path = output_dir / "sp500_good_pairs.csv"
    errors_path = output_dir / "sp500_errors.csv"
    tickers_path = output_dir / "sp500_tickers_used.csv"
    meta_path = output_dir / "sp500_metadata_used.csv"

    results.to_csv(all_path, index=False)
    if not results.empty:
        results[results["good_pair"]].to_csv(good_path, index=False)
    else:
        pd.DataFrame().to_csv(good_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    pd.DataFrame({"ticker": tickers}).to_csv(tickers_path, index=False)
    meta.to_csv(meta_path, index=False)

    total_elapsed = time.perf_counter() - total_start

    print("\nRuntime:")
    print(f"- Pair analysis time: {format_elapsed(pair_loop_elapsed)}")
    print(f"- Total runtime including download/output: {format_elapsed(total_elapsed)}")

    print("\nSaved files:")
    print(f"- {all_path}")
    print(f"- {good_path}")
    print(f"- {errors_path}")
    print(f"- {tickers_path}")
    print(f"- {meta_path}")

    if not results.empty:
        cols = [
            "stock1",
            "stock2",
            "sector1",
            "sector2",
            "market_ticker",
            "good_pair",
            "quality_score",
            "coint_pvalue",
            "adf_pvalue",
            "half_life",
            "spread_std",
            "beta_pair",
            "beta_market",
            "r_squared",
            "beta_pair_stability_ratio",
            "beta_market_stability_ratio",
            "signal_zscore",
            "trade_signal",
            "backtest_gross_total_return",
            "backtest_total_return",
            "backtest_transaction_cost_drag",
            "backtest_sharpe_ratio",
            "backtest_max_drawdown",
            "backtest_number_of_trades",
            "backtest_win_rate",
            "backtest_average_holding_days",
            "backtest_stop_loss_exits",
            "backtest_max_holding_exits",
            "backtest_relationship_break_exits",
            "backtest_total_transaction_cost",
        ]
        cols = [col for col in cols if col in results.columns]
        print(f"\nTop {args.top} pairs:")
        print(results[cols].head(args.top).to_string(index=False))

    return results, errors_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast S&P 500 pair screener using one shared Yahoo download."
    )

    parser.add_argument(
        "--analyzer-path",
        default=default_analyzer_path(),
        help="Path to updated pair_trading_signal_analyzer.py",
    )
    parser.add_argument(
        "--ticker-source",
        choices=["wikipedia", "fallback"],
        default="wikipedia",
        help="Default: wikipedia current S&P 500 table.",
    )
    parser.add_argument(
        "--tickers", default=None, help="Comma-separated tickers. Example: AAPL,MSFT,NVDA,JPM"
    )
    parser.add_argument(
        "--tickers-file",
        default=None,
        help="Optional .txt or .csv file of tickers. CSV can have a 'ticker' column.",
    )
    parser.add_argument(
        "--sector",
        default=None,
        help="Optional exact GICS sector filter. Example: Financials, Information Technology, Health Care",
    )
    parser.add_argument(
        "--sub-industry-contains",
        default=None,
        help="Optional text filter for GICS sub-industry. Example: Bank, Semiconductor",
    )

    parser.add_argument("--period", default="max", help="Yahoo Finance period. Default: max")
    parser.add_argument("--start", default="2024-01-01", help="Walk-forward backtest start date.")
    parser.add_argument(
        "--train-end",
        default=None,
        help="Walk-forward backtest end date. If omitted, signal date/latest data is used.",
    )
    parser.add_argument(
        "--signal-date",
        default=None,
        help="Signal date. If omitted, latest available data is used.",
    )
    parser.add_argument("--market-ticker", default="SPY", help="Market ticker. Default: SPY")
    parser.add_argument(
        "--rolling-window", type=int, default=126, help="Rolling beta window in trading days."
    )
    parser.add_argument(
        "--rolling-years",
        type=float,
        default=2.0,
        help="Daily walk-forward training window in calendar years. Default: 2",
    )
    parser.add_argument(
        "--exit-zscore", type=float, default=0.0, help="Exit threshold toward the mean. Default: 0"
    )
    parser.add_argument(
        "--buy-transaction-cost",
        type=float,
        default=0.002,
        help="Cost applied to bought notional. Default: 0.002 = 0.2%%",
    )
    parser.add_argument(
        "--sell-transaction-cost",
        type=float,
        default=0.002,
        help="Cost applied to sold notional. Default: 0.002 = 0.2%%",
    )
    parser.add_argument(
        "--gross-exposure",
        type=float,
        default=1.0,
        help="Total absolute pair exposure relative to equity. Default: 1.0",
    )
    parser.add_argument(
        "--min-training-observations",
        type=int,
        default=252,
        help="Minimum rows required inside each rolling-year training window.",
    )
    parser.add_argument(
        "--saved-model-during-trade",
        dest="use_saved_model_during_trade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the entry model for exit decisions while a trade is open. Default: enabled.",
    )
    parser.add_argument(
        "--trade-market-leg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade the market factor as a third hedge leg. Default: enabled.",
    )
    parser.add_argument(
        "--rebalance-daily",
        action="store_true",
        help="Restore target weights daily while a trade is open. Default: hold drifting weights until exit.",
    )
    parser.add_argument(
        "--max-holding-half-lives",
        type=float,
        default=2.0,
        help="Maximum holding period as a multiple of the entry model half-life. Use 0 to disable. Default: 2.",
    )
    parser.add_argument(
        "--stop-loss-fraction",
        type=float,
        default=0.10,
        help="Exit when net trade loss reaches this fraction. Use 0 to disable. Default: 0.10.",
    )
    parser.add_argument(
        "--relationship-break-r-squared",
        type=float,
        default=0.60,
        help="Exit if the fresh rolling model R-squared falls below this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--relationship-break-beta-change",
        type=float,
        default=0.50,
        help="Exit if pair or market beta changes by more than this relative amount. Use 0 to disable.",
    )

    parser.add_argument("--coint-threshold", type=float, default=0.05)
    parser.add_argument("--adf-threshold", type=float, default=0.05)
    parser.add_argument("--min-half-life", type=float, default=2.0)
    parser.add_argument("--max-half-life", type=float, default=60.0)
    parser.add_argument("--max-beta-stability", type=float, default=0.50)
    parser.add_argument("--min-r-squared", type=float, default=0.60)
    parser.add_argument("--entry-zscore", type=float, default=2.0)

    parser.add_argument(
        "--min-validation-days",
        type=int,
        default=252,
        help="Remove tickers with fewer valid price rows than this.",
    )
    parser.add_argument(
        "--download-chunk-size",
        type=int,
        default=100,
        help="Download tickers in chunks to reduce Yahoo failures.",
    )
    parser.add_argument(
        "--download-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between Yahoo download chunks.",
    )
    parser.add_argument(
        "--prefilter-top-corr",
        type=int,
        default=None,
        help="Optional speed filter: keep each ticker's top-N correlated names only.",
    )
    parser.add_argument(
        "--skip-market-leg",
        action="store_true",
        default=True,
        help="Skip any ticker equal to the market ticker.",
    )
    parser.add_argument(
        "--include-market-leg",
        dest="skip_market_leg",
        action="store_false",
        help="Allow pairs containing the market ticker.",
    )

    parser.add_argument(
        "--top", type=int, default=100, help="How many top pairs to print. Default: 100"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None, help="Optional testing limit for debugging."
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Parent folder for timestamped run folders. Default: results",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    screen_pairs(args)


if __name__ == "__main__":
    main()
