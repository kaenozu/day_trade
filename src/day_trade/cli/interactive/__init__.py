"""
対話型CLIパッケージ
後方互換性を保ちながら新しいモジュール構造を提供
"""

# 主要なクラスと関数をインポート（後方互換性のため）
from .cli import cli
from .interactive_mode import InteractiveMode

# 元のモジュールからの直接インポート（後方互換性）
from .helpers import get_watchlist_manager as _get_watchlist_manager
from .helpers import display_stock_details as _display_stock_details
from .helpers import display_historical_data as _display_historical_data
from .screening import run_interactive_backtest

# パブリックAPI
__all__ = [
    "cli",
    "InteractiveMode",
    "run_interactive_backtest",
    "_get_watchlist_manager",
    "_display_stock_details", 
    "_display_historical_data",
]