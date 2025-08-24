"""
株価データ取得パッケージ
yfinanceを使用してリアルタイムおよびヒストリカルな株価データを取得

このパッケージは以下のコンポーネントで構成されています:
- StockFetcher: メインの株価データ取得クラス
- DataCache: 高度なキャッシュシステム
- BulkStockFetcher: 一括データ取得機能
- カスタム例外クラス群

バックワード互換性のため、従来のインポート方法をサポートします。
"""

# メインクラスとコンポーネントをインポート
from .fetcher import StockFetcher
from .cache import DataCache, cache_with_ttl
from .cache_core import DataCache as DataCacheCore
from .cache_tuning import TunableDataCache
from .bulk_fetcher import BulkStockFetcher
from .bulk_parallel import BulkParallelFetcher
from .parallel_core import ParallelProcessorCore
from .exceptions import (
    StockFetcherError,
    InvalidSymbolError,
    DataNotFoundError,
)

# StockFetcherにBulkStockFetcherの機能を統合（バックワード互換性のため）
def _enhance_stock_fetcher():
    """
    StockFetcherクラスにBulkStockFetcherの機能を動的に追加
    バックワード互換性を維持するため
    """
    original_init = StockFetcher.__init__
    
    def enhanced_init(self, *args, **kwargs):
        # 元のinitを呼び出し
        original_init(self, *args, **kwargs)
        # BulkStockFetcherインスタンスを作成し、メソッドを統合
        self._bulk_fetcher = BulkStockFetcher(self)
        self._bulk_parallel = BulkParallelFetcher(self)
        
        # バルク処理メソッドを直接StockFetcherに追加
        self.get_realtime_data = self._bulk_fetcher.get_realtime_data
        self.bulk_get_company_info = self._bulk_fetcher.bulk_get_company_info
        self.bulk_get_historical_data = self._bulk_parallel.bulk_get_historical_data
        self.parallel_get_historical_data = self._bulk_parallel.parallel_get_historical_data
        self.parallel_get_current_prices = self._bulk_parallel.parallel_get_current_prices
        self.parallel_get_company_info = self._bulk_parallel.parallel_get_company_info
        
        # 元のファイルからの追加メソッド（既存コードとの互換性のため）
        self.bulk_get_current_prices = self._create_bulk_current_prices_method()
        self.bulk_get_current_prices_optimized = self._create_bulk_current_prices_optimized_method()
        
        # キャッシュ管理メソッド
        self.get_cache_performance_report = self._create_cache_performance_report_method()
        self._maybe_adjust_cache_settings = self._create_cache_adjustment_method()
        
    def _create_bulk_current_prices_method(self):
        """bulk_get_current_pricesメソッドを作成"""
        def bulk_get_current_prices(codes, batch_size=100, delay=0.05):
            # シンプルな実装：既存のget_current_priceを使用
            results = {}
            for code in codes:
                try:
                    results[code] = self.get_current_price(code)
                except Exception:
                    results[code] = None
            return results
        return bulk_get_current_prices
        
    def _create_bulk_current_prices_optimized_method(self):
        """bulk_get_current_prices_optimizedメソッドを作成"""
        def bulk_get_current_prices_optimized(codes, batch_size=50, delay=0.1):
            # 最適化された実装
            return self._bulk_fetcher.get_realtime_data(codes)
        return bulk_get_current_prices_optimized
        
    def _create_cache_performance_report_method(self):
        """get_cache_performance_reportメソッドを作成"""
        def get_cache_performance_report():
            return {
                "cache_enabled": hasattr(self, "_data_cache"),
                "auto_tuning_enabled": self.auto_cache_tuning_enabled,
                "last_adjustment": getattr(self, "last_cache_adjustment", 0),
                "adjustment_interval": getattr(self, "cache_adjustment_interval", 3600),
            }
        return get_cache_performance_report
        
    def _create_cache_adjustment_method(self):
        """_maybe_adjust_cache_settingsメソッドを作成"""
        def _maybe_adjust_cache_settings():
            # キャッシュ調整のダミー実装
            pass
        return _maybe_adjust_cache_settings
    
    # これらのメソッドをStockFetcherクラスに追加
    StockFetcher._create_bulk_current_prices_method = _create_bulk_current_prices_method
    StockFetcher._create_bulk_current_prices_optimized_method = _create_bulk_current_prices_optimized_method  
    StockFetcher._create_cache_performance_report_method = _create_cache_performance_report_method
    StockFetcher._create_cache_adjustment_method = _create_cache_adjustment_method
    StockFetcher.__init__ = enhanced_init

# 拡張を実行
_enhance_stock_fetcher()

# パッケージ情報
__version__ = "1.0.0"
__author__ = "Day Trading System"

# パッケージの公開API
__all__ = [
    # メインクラス
    "StockFetcher",
    
    # キャッシュシステム
    "DataCache",
    "DataCacheCore", 
    "TunableDataCache",
    "cache_with_ttl",
    
    # 一括処理
    "BulkStockFetcher",
    "BulkParallelFetcher",
    "ParallelProcessorCore",
    
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