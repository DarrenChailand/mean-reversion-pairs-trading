"""Research-grade tools for market-neutral pairs trading experiments."""

from .analyzer import (
    PairAnalysisConfig,
    PairAnalysisResult,
    PairBacktestConfig,
    PairBacktestResult,
    PairStrategyResult,
    analyze_pair_from_prices,
    analyze_pair_strategy_from_prices,
    download_price_data,
    run_daily_walk_forward_backtest_from_prices,
)

__all__ = [
    "PairAnalysisConfig",
    "PairAnalysisResult",
    "PairBacktestConfig",
    "PairBacktestResult",
    "PairStrategyResult",
    "analyze_pair_from_prices",
    "analyze_pair_strategy_from_prices",
    "download_price_data",
    "run_daily_walk_forward_backtest_from_prices",
]
