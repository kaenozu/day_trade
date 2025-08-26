"""
株価データ取得パッケージ
yfinanceを使用してリアルタイムおよびヒストリカルな株価データを取得

このパッケージは以下のコンポーネントで構成されています:
- StockFetcher: メインの株価データ取得クラス（レガシーファイルからの完全移行版）
- DataCache: 高度なキャッシュシステム
- BulkStockFetcher: 一括データ取得機能
- カスタム例外クラス群

バックワード互換性のため、従来のインポート方法を完全にサポートします。
"""

# 元の大きなファイルの完全な機能を持つStockFetcher
from .legacy_fetcher import LegacyStockFetcher as StockFetcher

# 各コンポーネントクラス
from .fetcher import StockFetcher as ModularStockFetcher
from .cache import DataCache, cache_with_ttl
from .cache_core import DataCache as DataCacheCore  
from .cache_tuning import TunableDataCache
from .bulk_fetcher import BulkStockFetcher
from .bulk_parallel import BulkParallelFetcher
from .parallel_core import ParallelProcessorCore
from .advanced_features import AdvancedStockFetcherMixin
from .bulk_operations import BulkOperationsMixin
from .exceptions import (
    StockFetcherError,
    InvalidSymbolError,
    DataNotFoundError,
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Day Trading System"

# パッケージの公開API（完全な後方互換性）
__all__ = [
    # メインクラス（レガシー完全版 - 元の stock_fetcher.py と100%互換）
    "StockFetcher",
    
    # 新しいモジュラークラス（必要に応じて使用可能）
    "ModularStockFetcher",
    
    # キャッシュシステム
    "DataCache",
    "DataCacheCore", 
    "TunableDataCache",
    "cache_with_ttl",
    
    # 一括処理
    "BulkStockFetcher",
    "BulkParallelFetcher",
    "ParallelProcessorCore",
    
    # 高度な機能
    "AdvancedStockFetcherMixin",
    "BulkOperationsMixin",
    
    # 例外クラス
    "StockFetcherError",
    "InvalidSymbolError", 
    "DataNotFoundError",
]

# 使用例とドキュメント
USAGE_EXAMPLE = """
使用例:

# 基本的な使用方法
from day_trade.data.stock_fetcher import StockFetcher

fetcher = StockFetcher()

# 現在価格を取得
price_data = fetcher.get_current_price("7203")  # トヨタ

# ヒストリカルデータを取得  
hist_data = fetcher.get_historical_data("7203", period="1mo")

# 企業情報を取得
company_info = fetcher.get_company_info("7203")

# 複数銘柄の一括取得
codes = ["7203", "9984", "6758"]
realtime_data = fetcher.get_realtime_data(codes)
bulk_company_info = fetcher.bulk_get_company_info(codes)

# 並列処理での取得
parallel_prices = fetcher.parallel_get_current_prices(codes)
parallel_hist = fetcher.parallel_get_historical_data(codes, period="1y")
"""