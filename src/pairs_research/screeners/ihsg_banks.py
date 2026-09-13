"""
Fast IHSG Banking Pair Screener
================================

This version is faster because it downloads all prices once, then reuses that
single price table for every bank-stock pair.

Required file in the same folder:
    pair_trading_signal_analyzer.py

Run:
    python ihsg_bank_pair_screener_fast.py --start 2024-01-01 --rolling-window 126 --top 20
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pandas as pd

DEFAULT_BANK_TICKERS = [
    "AGRO.JK",
    "AGRS.JK",
    "AMAR.JK",
    "ARTO.JK",
    "BABP.JK",
    "BACA.JK",
    "BANK.JK",
    "BBCA.JK",
    "BBHI.JK",
    "BBKP.JK",
    "BBMD.JK",
    "BBNI.JK",
    "BBRI.JK",
    "BBTN.JK",
    "BBYB.JK",
    "BCIC.JK",
    "BDMN.JK",
    "BEKS.JK",
    "BGTG.JK",
    "BINA.JK",
    "BJBR.JK",
    "BJTM.JK",
    "BKSW.JK",
    "BMAS.JK",
    "BMRI.JK",
    "BNBA.JK",
    "BNGA.JK",
    "BNII.JK",
    "BSIM.JK",
    "BTPN.JK",
    "BVIC.JK",
    "DNAR.JK",
    "INPC.JK",
    "MASB.JK",
    "MAYA.JK",
    "MCOR.JK",
    "MEGA.JK",
    "NISP.JK",
    "NOBU.JK",
    "PNBN.JK",
    "SDRA.JK",
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


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not ticker:
        return ""
    if ticker.startswith("^"):
        return ticker
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    return ticker


def normalize_tickers(raw_tickers: list[str]) -> list[str]:
    clean = [normalize_ticker(ticker) for ticker in raw_tickers]
    clean = [ticker for ticker in clean if ticker]
    return list(dict.fromkeys(clean))


def load_tickers(tickers_arg: str | None, tickers_file: str | None) -> list[str]:
    if tickers_arg:
        return normalize_tickers(tickers_arg.split(","))

    if tickers_file:
        path = Path(tickers_file)
        if not path.exists():
            raise FileNotFoundError(f"Ticker file not found: {path}")
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            raw = (
                df["ticker"].astype(str).tolist()
                if "ticker" in df.columns
                else df.iloc[:, 0].astype(str).tolist()
            )
        else:
            raw = path.read_text(encoding="utf-8").replace("\n", ",").split(",")
        return normalize_tickers(raw)

    return DEFAULT_BANK_TICKERS.copy()


def default_analyzer_path() -> str:
    return str(Path(__file__).parents[1] / "analyzer.py")


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
            "Use the updated pair_trading_signal_analyzer.py I gave you."
        )
    return module


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


def analyze_one_pair(
    module: ModuleType, prices: pd.DataFrame, stock1: str, stock2: str, args: argparse.Namespace
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
        adf_threshold=args.adf_threshold,
        min_half_life=args.min_half_life,
        max_half_life=args.max_half_life,
        max_beta_stability=args.max_beta_stability,
        min_r_squared=args.min_r_squared,
        verbose=False,
    )
    return dict(strategy.summary)


def screen_pairs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    module = load_analyzer(args.analyzer_path)
    tickers = load_tickers(args.tickers, args.tickers_file)
    market_ticker = normalize_ticker(args.market_ticker)
    args.market_ticker = market_ticker

    download_tickers = list(dict.fromkeys(tickers + [market_ticker]))
    print(f"Downloading {len(download_tickers)} tickers once...")
    prices = module.download_price_data(
        download_tickers, period=args.period, interval="1d", progress=False, threads=True
    )
    tickers = validate_downloaded_tickers(prices, tickers, args.min_validation_days)

    pairs = list(itertools.combinations(tickers, 2))
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    print(f"Testing {len(tickers)} bank tickers = {len(pairs)} pairs")
    print("Fast mode: Yahoo data was downloaded once and reused for every pair.")

    rows: list[dict] = []
    errors: list[dict] = []
    for i, (stock1, stock2) in enumerate(pairs, start=1):
        if i == 1 or i % args.progress_every == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] {stock1} vs {stock2}")
        try:
            rows.append(analyze_one_pair(module, prices, stock1, stock2, args))
        except Exception as exc:
            errors.append({"stock1": stock1, "stock2": stock2, "error": str(exc)})

    results = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)
    if not results.empty:
        results = results.sort_values(
            by=["good_pair", "backtest_sharpe_ratio", "quality_score", "abs_signal_zscore"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    output_dir = create_run_output_dir(args.output_dir)
    all_path = output_dir / "ihsg_banking_pair_results.csv"
    good_path = output_dir / "ihsg_banking_good_pairs.csv"
    errors_path = output_dir / "ihsg_banking_errors.csv"
    results.to_csv(all_path, index=False)
    results[results["good_pair"]].to_csv(
        good_path, index=False
    ) if not results.empty else pd.DataFrame().to_csv(good_path, index=False)
    errors_df.to_csv(errors_path, index=False)

    print("\nSaved files:")
    print(f"- {all_path}")
    print(f"- {good_path}")
    print(f"- {errors_path}")

    if not results.empty:
        cols = [
            "stock1",
            "stock2",
            "market_ticker",
            "good_pair",
            "quality_score",
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
        print(f"\nTop {args.top} pairs:")
        print(results[cols].head(args.top).to_string(index=False))
    return results, errors_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast IHSG banking pair screener using one shared Yahoo download."
    )
    parser.add_argument(
        "--analyzer-path",
        default=default_analyzer_path(),
        help="Path to updated pair_trading_signal_analyzer.py",
    )
    parser.add_argument(
        "--tickers", default=None, help="Comma-separated tickers. Example: BBCA,BBRI,BMRI,BBNI"
    )
    parser.add_argument(
        "--tickers-file",
        default=None,
        help="Optional .txt or .csv file of tickers. CSV can have a 'ticker' column.",
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
    parser.add_argument("--market-ticker", default="^JKSE", help="Market ticker. Default: ^JKSE")
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
    parser.add_argument("--adf-threshold", type=float, default=0.05)
    parser.add_argument("--min-half-life", type=float, default=2.0)
    parser.add_argument("--max-half-life", type=float, default=60.0)
    parser.add_argument("--max-beta-stability", type=float, default=0.50)
    parser.add_argument("--min-r-squared", type=float, default=0.60)
    parser.add_argument("--entry-zscore", type=float, default=2.0)
    parser.add_argument("--min-validation-days", type=int, default=100)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--max-pairs", type=int, default=None, help="Optional testing limit for debugging."
    )
    parser.add_argument("--progress-every", type=int, default=25)
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
