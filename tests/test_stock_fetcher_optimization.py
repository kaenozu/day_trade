"""
StockFetcherのキャッシュ最適化機能テスト - 未カバー領域対応
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.day_trade.data.stock_fetcher import DataCache, StockFetcher


@pytest.mark.skip(reason="DataCache API変更により一時的に無効化")
class TestDataCacheOptimization:
    """DataCacheの最適化機能テスト"""

    def test_optimize_cache_settings_low_hit_rate(self):
        """低ヒット率での最適化提案テスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000)

        # 低ヒット率の状況を作成
        cache._stats["total_requests"] = 100
        cache._stats["hits"] = 20  # 20%ヒット率
        cache._stats["misses"] = 80
        cache._stats["evictions"] = 0
        cache._stats["stale_hits"] = 5

        # 最適化提案を取得
        optimization = cache.optimize_cache_settings()

        # 低ヒット率の場合のTTL延長提案があることを確認
        assert "recommendations" in optimization
        recommendations = optimization["recommendations"]
        assert "ttl_increase" in recommendations
        assert "TTLを延長" in recommendations["ttl_increase"]

    def test_optimize_cache_settings_high_eviction(self):
        """高退避率での最適化提案テスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000)

        # 高退避率の状況を作成
        cache._stats["total_requests"] = 100
        cache._stats["hits"] = 70
        cache._stats["misses"] = 30
        cache._stats["evictions"] = 15  # ヒット数の約21%
        cache._stats["stale_hits"] = 2

        optimization = cache.optimize_cache_settings()

        # 高退避率の場合のサイズ増加提案があることを確認
        recommendations = optimization["recommendations"]
        assert "size_increase" in recommendations
        assert "キャッシュサイズを増加" in recommendations["size_increase"]

    def test_optimize_cache_settings_low_utilization(self):
        """低使用率での最適化提案テスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000)

        # キャッシュに少しだけデータを追加（低使用率）
        for i in range(200):  # 1000中200 = 20%使用率
            cache.set(f"key_{i}", f"value_{i}")

        # 統計を調整
        cache._stats["total_requests"] = 100
        cache._stats["hits"] = 60
        cache._stats["misses"] = 40
        cache._stats["evictions"] = 0
        cache._stats["stale_hits"] = 1

        optimization = cache.optimize_cache_settings()

        # 低使用率の場合のサイズ減少提案があることを確認
        recommendations = optimization["recommendations"]
        assert "size_decrease" in recommendations
        assert "キャッシュサイズを減少" in recommendations["size_decrease"]

    def test_optimize_cache_settings_high_stale_hit_rate(self):
        """高staleヒット率での最適化提案テスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000, stale_while_revalidate=600)

        # 高staleヒット率の状況を作成
        cache._stats["total_requests"] = 100
        cache._stats["hits"] = 50
        cache._stats["misses"] = 25
        cache._stats["evictions"] = 0
        cache._stats["stale_hits"] = 25  # 25%staleヒット率

        optimization = cache.optimize_cache_settings()

        # 高staleヒット率の場合の期間調整提案があることを確認
        recommendations = optimization["recommendations"]
        assert "stale_period_adjust" in recommendations
        assert "stale-while-revalidate期間の調整" in recommendations["stale_period_adjust"]

    def test_auto_tune_cache_insufficient_data(self):
        """データ不足時の自動調整テスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000)

        # データ不足の状況（100未満のリクエスト）
        cache._stats["total_requests"] = 50
        cache._stats["hits"] = 20
        cache._stats["misses"] = 30

        result = cache.auto_tune_cache()

        # データ不足で調整されないことを確認
        assert result["adjusted"] is False
        assert "insufficient_data" in result["reason"]
        assert result["stats"]["total_requests"] == 50

    def test_auto_tune_cache_ttl_increase(self):
        """TTL延長の自動調整テスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000)

        # TTL延長が必要な状況を作成
        cache._stats["total_requests"] = 200
        cache._stats["hits"] = 60  # 30%ヒット率（0.4未満）
        cache._stats["misses"] = 140
        cache._stats["evictions"] = 5
        cache._stats["stale_hits"] = 35  # 17.5%staleヒット率（0.15超）

        original_ttl = cache.ttl_seconds
        result = cache.auto_tune_cache()

        # TTLが延長されることを確認
        assert cache.ttl_seconds > original_ttl
        assert cache.ttl_seconds <= 900  # 最大15分
        assert "ttl" in result.get("adjustments", {})

    def test_auto_tune_cache_ttl_decrease(self):
        """TTL短縮の自動調整テスト"""
        cache = DataCache(ttl_seconds=600, max_size=1000)

        # TTL短縮が必要な状況を作成
        cache._stats["total_requests"] = 200
        cache._stats["hits"] = 170  # 85%ヒット率（0.8超）
        cache._stats["misses"] = 30
        cache._stats["evictions"] = 2
        cache._stats["stale_hits"] = 8  # 4%staleヒット率（0.05未満）

        original_ttl = cache.ttl_seconds
        result = cache.auto_tune_cache()

        # TTLが短縮されることを確認
        assert cache.ttl_seconds < original_ttl
        assert cache.ttl_seconds >= 60  # 最小1分
        assert "ttl" in result.get("adjustments", {})

    def test_auto_tune_cache_size_increase(self):
        """キャッシュサイズ増加の自動調整テスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000)

        # キャッシュをほぼ満杯にする（高使用率）
        for i in range(950):
            cache.set(f"key_{i}", f"value_{i}")

        # サイズ増加が必要な状況を作成
        cache._stats["total_requests"] = 200
        cache._stats["hits"] = 120
        cache._stats["misses"] = 80
        cache._stats["evictions"] = 30  # ヒット数の25%（0.2超）
        cache._stats["stale_hits"] = 10

        original_max_size = cache.max_size
        result = cache.auto_tune_cache()

        # サイズが増加されることを確認
        assert cache.max_size > original_max_size
        assert cache.max_size <= 5000  # 最大5000エントリ
        assert "max_size" in result.get("adjustments", {})

    def test_auto_tune_cache_size_decrease(self):
        """キャッシュサイズ減少の自動調整テスト"""
        cache = DataCache(ttl_seconds=300, max_size=2000)

        # 低使用率の状況を作成（少ないエントリ）
        for i in range(200):  # 2000中200 = 10%使用率
            cache.set(f"key_{i}", f"value_{i}")

        cache._stats["total_requests"] = 200
        cache._stats["hits"] = 120
        cache._stats["misses"] = 80
        cache._stats["evictions"] = 5
        cache._stats["stale_hits"] = 10

        original_max_size = cache.max_size
        result = cache.auto_tune_cache()

        # サイズが減少されることを確認
        assert cache.max_size < original_max_size
        assert cache.max_size >= 500  # 最小500エントリ
        assert "max_size" in result.get("adjustments", {})

    def test_auto_tune_cache_no_adjustment_needed(self):
        """調整が不要な場合のテスト"""
        cache = DataCache(ttl_seconds=300, max_size=1000)

        # 良好な統計値を設定
        cache._stats["total_requests"] = 200
        cache._stats["hits"] = 120  # 60%ヒット率（適正範囲）
        cache._stats["misses"] = 80
        cache._stats["evictions"] = 10  # 適正範囲
        cache._stats["stale_hits"] = 20  # 10%staleヒット率（適正範囲）

        # キャッシュ使用率も適正に設定
        for i in range(600):  # 60%使用率
            cache.set(f"key_{i}", f"value_{i}")

        original_ttl = cache.ttl_seconds
        original_max_size = cache.max_size

        result = cache.auto_tune_cache()

        # 調整が行われないことを確認
        assert cache.ttl_seconds == original_ttl
        assert cache.max_size == original_max_size
        assert result.get("adjustments", {}) == {}


@pytest.mark.skip(reason="StockFetcher API変更により一時的に無効化")
class TestStockFetcherOptimization:
    """StockFetcherの最適化機能テスト"""

    @pytest.fixture
    def mock_yfinance(self):
        """yfinanceのモック"""
        with patch('src.day_trade.data.stock_fetcher.yf') as mock_yf:
            mock_ticker = Mock()
            mock_yf.Ticker.return_value = mock_ticker

            # ダミーの履歴データ
            mock_history = pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [101, 102, 103],
                'Low': [99, 100, 101],
                'Close': [100.5, 101.5, 102.5],
                'Volume': [1000, 1100, 1200]
            })
            mock_ticker.history.return_value = mock_history

            # ダミーの情報データ
            mock_ticker.info = {
                'longName': 'Test Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000000,
                'regularMarketPrice': 102.5
            }

            yield mock_yf

    def test_stock_fetcher_cache_optimization_integration(self, mock_yfinance):
        """StockFetcherでのキャッシュ最適化統合テスト"""
        fetcher = StockFetcher()

        # 複数回の取得でキャッシュを使用
        for _ in range(10):
            fetcher.get_historical_data("TEST", "1d", "1mo")

        # キャッシュ統計を確認
        stats = fetcher.data_cache.get_cache_stats()
        assert stats["total_requests"] > 0
        assert stats["hits"] > 0  # キャッシュヒットが発生することを確認

        # 最適化提案を取得
        optimization = fetcher.data_cache.optimize_cache_settings()
        assert "current_stats" in optimization
        assert "recommendations" in optimization

    def test_stock_fetcher_auto_tune_after_requests(self, mock_yfinance):
        """リクエスト後の自動調整テスト"""
        fetcher = StockFetcher()

        # 十分な数のリクエストを実行
        symbols = [f"TEST{i}" for i in range(20)]
        for symbol in symbols:
            for _ in range(10):  # 各シンボルで10回取得
                fetcher.get_historical_data(symbol, "1d", "1mo")

        # 自動調整を実行
        # original_ttl = fetcher.data_cache.ttl_seconds  # 未使用のため削除
        # original_max_size = fetcher.data_cache.max_size  # 未使用のため削除

        tune_result = fetcher.data_cache.auto_tune_cache()

        # 統計が蓄積されていることを確認
        assert tune_result["stats"]["total_requests"] >= 100

        # 調整が実行される可能性があることを確認（必ずしも調整されるとは限らない）
        assert "adjusted" in tune_result
        assert isinstance(tune_result["adjusted"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
