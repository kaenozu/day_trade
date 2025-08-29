#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Fetcher Data Tests
株式データ取得テスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# モッククラス
class MockStockDataFetcher:
    """株式データ取得モック"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5分
        self.rate_limit_delay = 0.1  # 100ms
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
        
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """現在価格取得"""
        self._rate_limit_check()
        
        try:
            # キャッシュチェック
            cache_key = f"price_{symbol}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # モック価格生成（実際の実装ではAPI呼び出し）
            price = self._generate_mock_price(symbol)
            
            # キャッシュ保存
            self._cache_data(cache_key, price)
            
            self.request_count += 1
            return price
            
        except Exception as e:
            self.error_count += 1
            return None
    
    def fetch_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """履歴データ取得"""
        self._rate_limit_check()
        
        try:
            # キャッシュチェック
            cache_key = f"history_{symbol}_{days}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                return cached_data
            
            # モック履歴データ生成
            historical_data = self._generate_mock_historical_data(symbol, days)
            
            # キャッシュ保存
            self._cache_data(cache_key, historical_data)
            
            self.request_count += 1
            return historical_data
            
        except Exception as e:
            self.error_count += 1
            return None
    
    def fetch_multiple_symbols(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """複数銘柄同時取得"""
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.fetch_current_price(symbol)
            
        return results
    
    def fetch_market_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """市場情報取得"""
        self._rate_limit_check()
        
        try:
            cache_key = f"info_{symbol}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # モック市場情報
            info = {
                'symbol': symbol,
                'name': f'Company {symbol}',
                'market': '東証1部' if symbol.startswith('7') else '東証2部',
                'sector': 'Technology',
                'market_cap': np.random.randint(1000, 100000) * 100000000,
                'volume': np.random.randint(100000, 10000000),
                'pe_ratio': np.random.uniform(5, 50),
                'pb_ratio': np.random.uniform(0.5, 5),
                'dividend_yield': np.random.uniform(0, 5),
                'last_updated': datetime.now()
            }
            
            self._cache_data(cache_key, info)
            self.request_count += 1
            return info
            
        except Exception as e:
            self.error_count += 1
            return None
    
    def _rate_limit_check(self):
        """レート制限チェック"""
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay)
        self.last_request_time = current_time
    
    def _get_cached_data(self, key: str):
        """キャッシュデータ取得"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                # 期限切れキャッシュ削除
                del self.cache[key]
        return None
    
    def _cache_data(self, key: str, data):
        """データキャッシュ"""
        self.cache[key] = (data, time.time())
    
    def _generate_mock_price(self, symbol: str) -> float:
        """モック価格生成"""
        # シンボルのハッシュをシードとして使用
        np.random.seed(hash(symbol) % 1000000)
        
        base_price = 1000 + (hash(symbol) % 3000)  # 1000-4000円
        variation = np.random.normal(0, 50)  # ±50円の変動
        
        return max(100, base_price + variation)  # 最低100円
    
    def _generate_mock_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """モック履歴データ生成"""
        np.random.seed(hash(symbol) % 1000000)
        
        # 開始価格
        start_price = self._generate_mock_price(symbol)
        
        # 日付範囲
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 価格系列生成（ランダムウォーク）
        returns = np.random.normal(0, 0.02, len(dates))  # 日次2%の標準偏差
        prices = start_price * np.exp(np.cumsum(returns))
        
        # OHLC生成
        opens = prices * (1 + np.random.normal(0, 0.005, len(dates)))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        
        # 出来高生成
        volumes = np.random.randint(100000, 5000000, len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        cache_hit_rate = 0.0
        if self.request_count > 0:
            cache_hits = len([k for k in self.cache.keys() if self._get_cached_data(k) is not None])
            cache_hit_rate = cache_hits / (self.request_count + len(self.cache))
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count + self.error_count),
            'cache_size': len(self.cache),
            'cache_hit_rate': cache_hit_rate
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.cache.clear()


class MockDataValidator:
    """データ検証モック"""
    
    @staticmethod
    def validate_price_data(price: float) -> bool:
        """価格データ検証"""
        if price is None:
            return False
        if not isinstance(price, (int, float)):
            return False
        if price <= 0:
            return False
        if price > 1000000:  # 100万円超えは異常値
            return False
        return True
    
    @staticmethod
    def validate_historical_data(data: pd.DataFrame) -> bool:
        """履歴データ検証"""
        if data is None or data.empty:
            return False
        
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # OHLC関係チェック
        for i in data.index:
            row = data.loc[i]
            if not (row['low'] <= row['open'] <= row['high']):
                return False
            if not (row['low'] <= row['close'] <= row['high']):
                return False
            if row['volume'] < 0:
                return False
        
        return True
    
    @staticmethod
    def validate_market_info(info: Dict[str, Any]) -> bool:
        """市場情報検証"""
        if not info:
            return False
        
        required_fields = ['symbol', 'name', 'market']
        if not all(field in info for field in required_fields):
            return False
        
        if not isinstance(info['symbol'], str) or len(info['symbol']) != 4:
            return False
        
        return True


class MockBulkDataFetcher:
    """バルクデータ取得モック"""
    
    def __init__(self, fetcher: MockStockDataFetcher):
        self.fetcher = fetcher
        self.batch_size = 50
        self.parallel_workers = 4
    
    def fetch_bulk_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """バルク価格取得"""
        results = {}
        
        # バッチ処理
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i+self.batch_size]
            batch_results = self.fetcher.fetch_multiple_symbols(batch)
            results.update(batch_results)
            
            # バッチ間の待機
            if i + self.batch_size < len(symbols):
                time.sleep(0.1)
        
        return results
    
    def fetch_bulk_historical(self, symbols: List[str], 
                            days: int = 30) -> Dict[str, Optional[pd.DataFrame]]:
        """バルク履歴データ取得"""
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.fetcher.fetch_historical_data(symbol, days)
        
        return results


class TestMockStockDataFetcher:
    """株式データ取得テストクラス"""
    
    @pytest.fixture
    def fetcher(self):
        """データ取得器フィクスチャ"""
        return MockStockDataFetcher()
    
    def test_fetcher_initialization(self, fetcher):
        """取得器初期化テスト"""
        assert fetcher.cache == {}
        assert fetcher.cache_ttl == 300
        assert fetcher.request_count == 0
        assert fetcher.error_count == 0
    
    def test_current_price_fetch(self, fetcher):
        """現在価格取得テスト"""
        symbol = "7203"  # トヨタ
        price = fetcher.fetch_current_price(symbol)
        
        assert price is not None
        assert isinstance(price, float)
        assert price > 0
        assert fetcher.request_count == 1
    
    def test_current_price_caching(self, fetcher):
        """現在価格キャッシュテスト"""
        symbol = "7203"
        
        # 初回取得
        price1 = fetcher.fetch_current_price(symbol)
        initial_requests = fetcher.request_count
        
        # 2回目取得（キャッシュヒット）
        price2 = fetcher.fetch_current_price(symbol)
        
        assert price1 == price2
        assert fetcher.request_count == initial_requests  # リクエスト数変化なし
    
    def test_historical_data_fetch(self, fetcher):
        """履歴データ取得テスト"""
        symbol = "7203"
        days = 30
        
        data = fetcher.fetch_historical_data(symbol, days)
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'date' in data.columns
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
        assert fetcher.request_count == 1
    
    def test_multiple_symbols_fetch(self, fetcher):
        """複数銘柄取得テスト"""
        symbols = ["7203", "9984", "6758"]
        
        results = fetcher.fetch_multiple_symbols(symbols)
        
        assert len(results) == len(symbols)
        for symbol in symbols:
            assert symbol in results
            assert results[symbol] is not None
            assert isinstance(results[symbol], float)
        
        assert fetcher.request_count == len(symbols)
    
    def test_market_info_fetch(self, fetcher):
        """市場情報取得テスト"""
        symbol = "7203"
        
        info = fetcher.fetch_market_info(symbol)
        
        assert info is not None
        assert isinstance(info, dict)
        assert info['symbol'] == symbol
        assert 'name' in info
        assert 'market' in info
        assert 'sector' in info
        assert 'market_cap' in info
        assert fetcher.request_count == 1
    
    def test_rate_limiting(self, fetcher):
        """レート制限テスト"""
        symbols = ["7203", "9984"]
        
        start_time = time.time()
        for symbol in symbols:
            fetcher.fetch_current_price(symbol)
        end_time = time.time()
        
        # レート制限により最低限の時間がかかる
        min_expected_time = len(symbols) * fetcher.rate_limit_delay
        assert end_time - start_time >= min_expected_time * 0.8  # 若干の誤差を許容
    
    def test_cache_expiry(self, fetcher):
        """キャッシュ期限切れテスト"""
        symbol = "7203"
        fetcher.cache_ttl = 0.1  # 100ms
        
        # 初回取得
        price1 = fetcher.fetch_current_price(symbol)
        
        # 期限切れまで待機
        time.sleep(0.2)
        
        # 再取得（キャッシュ期限切れ）
        price2 = fetcher.fetch_current_price(symbol)
        
        # リクエスト数が増加している
        assert fetcher.request_count == 2
    
    def test_stats_collection(self, fetcher):
        """統計情報収集テスト"""
        symbols = ["7203", "9984", "6758"]
        
        # いくつかリクエスト実行
        for symbol in symbols:
            fetcher.fetch_current_price(symbol)
        
        stats = fetcher.get_stats()
        
        assert stats['request_count'] == len(symbols)
        assert stats['error_count'] == 0
        assert stats['error_rate'] == 0.0
        assert stats['cache_size'] >= 0
        assert 'cache_hit_rate' in stats
    
    def test_cache_clear(self, fetcher):
        """キャッシュクリアテスト"""
        symbol = "7203"
        
        # データをキャッシュ
        fetcher.fetch_current_price(symbol)
        assert len(fetcher.cache) > 0
        
        # キャッシュクリア
        fetcher.clear_cache()
        assert len(fetcher.cache) == 0


class TestMockDataValidator:
    """データ検証テストクラス"""
    
    def test_valid_price_data(self):
        """有効価格データ検証テスト"""
        valid_prices = [100.0, 1500.5, 5000, 25000.25]
        
        for price in valid_prices:
            assert MockDataValidator.validate_price_data(price) is True
    
    def test_invalid_price_data(self):
        """無効価格データ検証テスト"""
        invalid_prices = [None, -100, 0, "1000", 2000000]
        
        for price in invalid_prices:
            assert MockDataValidator.validate_price_data(price) is False
    
    def test_valid_historical_data(self):
        """有効履歴データ検証テスト"""
        valid_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [1000, 1010, 1020, 1030, 1040],
            'high': [1050, 1060, 1070, 1080, 1090],
            'low': [950, 960, 970, 980, 990],
            'close': [1025, 1035, 1045, 1055, 1065],
            'volume': [100000, 150000, 120000, 180000, 110000]
        })
        
        assert MockDataValidator.validate_historical_data(valid_data) is True
    
    def test_invalid_historical_data(self):
        """無効履歴データ検証テスト"""
        # 空のデータ
        assert MockDataValidator.validate_historical_data(pd.DataFrame()) is False
        assert MockDataValidator.validate_historical_data(None) is False
        
        # 必要なカラムがない
        invalid_data = pd.DataFrame({'date': [1, 2, 3], 'price': [100, 200, 300]})
        assert MockDataValidator.validate_historical_data(invalid_data) is False
        
        # OHLC関係が不正
        invalid_ohlc = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=1),
            'open': [1000],
            'high': [900],  # high < open は不正
            'low': [950],
            'close': [1025],
            'volume': [100000]
        })
        assert MockDataValidator.validate_historical_data(invalid_ohlc) is False
    
    def test_valid_market_info(self):
        """有効市場情報検証テスト"""
        valid_info = {
            'symbol': '7203',
            'name': 'Toyota Motor Corp',
            'market': '東証1部',
            'sector': 'Automotive'
        }
        
        assert MockDataValidator.validate_market_info(valid_info) is True
    
    def test_invalid_market_info(self):
        """無効市場情報検証テスト"""
        invalid_infos = [
            None,
            {},
            {'symbol': '7203'},  # name, marketがない
            {'symbol': '720', 'name': 'Test', 'market': 'TSE'},  # symbol長さ不正
            {'symbol': 7203, 'name': 'Test', 'market': 'TSE'}   # symbol型不正
        ]
        
        for info in invalid_infos:
            assert MockDataValidator.validate_market_info(info) is False


class TestMockBulkDataFetcher:
    """バルクデータ取得テストクラス"""
    
    @pytest.fixture
    def fetcher(self):
        """基本フェッチャーフィクスチャ"""
        return MockStockDataFetcher()
    
    @pytest.fixture
    def bulk_fetcher(self, fetcher):
        """バルクフェッチャーフィクスチャ"""
        return MockBulkDataFetcher(fetcher)
    
    def test_bulk_price_fetch(self, bulk_fetcher):
        """バルク価格取得テスト"""
        symbols = ["7203", "9984", "6758", "4755", "8306"]
        
        results = bulk_fetcher.fetch_bulk_prices(symbols)
        
        assert len(results) == len(symbols)
        for symbol in symbols:
            assert symbol in results
            assert results[symbol] is not None
            assert isinstance(results[symbol], float)
    
    def test_bulk_historical_fetch(self, bulk_fetcher):
        """バルク履歴データ取得テスト"""
        symbols = ["7203", "9984", "6758"]
        days = 30
        
        results = bulk_fetcher.fetch_bulk_historical(symbols, days)
        
        assert len(results) == len(symbols)
        for symbol in symbols:
            assert symbol in results
            data = results[symbol]
            assert data is not None
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
    
    def test_batch_processing(self, bulk_fetcher):
        """バッチ処理テスト"""
        # バッチサイズを小さく設定
        bulk_fetcher.batch_size = 2
        
        symbols = ["7203", "9984", "6758", "4755", "8306"]
        
        start_time = time.time()
        results = bulk_fetcher.fetch_bulk_prices(symbols)
        end_time = time.time()
        
        assert len(results) == len(symbols)
        
        # バッチ処理により時間がかかる
        expected_batches = (len(symbols) + bulk_fetcher.batch_size - 1) // bulk_fetcher.batch_size
        min_expected_time = (expected_batches - 1) * 0.1  # バッチ間の待機時間
        assert end_time - start_time >= min_expected_time * 0.8


class TestDataIntegration:
    """データ統合テスト"""
    
    def test_complete_data_workflow(self):
        """完全データワークフローテスト"""
        # コンポーネント作成
        fetcher = MockStockDataFetcher()
        bulk_fetcher = MockBulkDataFetcher(fetcher)
        validator = MockDataValidator()
        
        symbols = ["7203", "9984", "6758"]
        
        # 1. バルク価格取得
        prices = bulk_fetcher.fetch_bulk_prices(symbols)
        
        # 2. データ検証
        for symbol, price in prices.items():
            assert validator.validate_price_data(price)
        
        # 3. 履歴データ取得と検証
        for symbol in symbols:
            historical_data = fetcher.fetch_historical_data(symbol)
            assert validator.validate_historical_data(historical_data)
        
        # 4. 市場情報取得と検証
        for symbol in symbols:
            market_info = fetcher.fetch_market_info(symbol)
            assert validator.validate_market_info(market_info)
        
        # 5. 統計確認
        stats = fetcher.get_stats()
        assert stats['error_count'] == 0
        assert stats['request_count'] > 0
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        fetcher = MockStockDataFetcher()
        
        # 異常なシンボルでのテスト
        invalid_symbols = ["", "INVALID", "12345"]
        
        for symbol in invalid_symbols:
            price = fetcher.fetch_current_price(symbol)
            # エラーが発生してもNoneが返される
            assert price is None or isinstance(price, float)
    
    def test_performance_characteristics(self):
        """パフォーマンス特性テスト"""
        fetcher = MockStockDataFetcher()
        symbols = [f"000{i}" for i in range(10)]  # 10銘柄
        
        # キャッシュなしの場合
        start_time = time.time()
        for symbol in symbols:
            fetcher.fetch_current_price(symbol)
        first_run_time = time.time() - start_time
        
        # キャッシュありの場合
        start_time = time.time()
        for symbol in symbols:
            fetcher.fetch_current_price(symbol)
        second_run_time = time.time() - start_time
        
        # キャッシュがある場合の方が高速
        assert second_run_time < first_run_time * 0.5
    
    def test_concurrent_access(self):
        """並行アクセステスト"""
        import threading
        
        fetcher = MockStockDataFetcher()
        symbols = ["7203", "9984", "6758"]
        results = {}
        errors = []
        
        def fetch_symbol(symbol):
            try:
                price = fetcher.fetch_current_price(symbol)
                results[symbol] = price
            except Exception as e:
                errors.append(e)
        
        # 並行実行
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=fetch_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        # 完了待ち
        for thread in threads:
            thread.join()
        
        # 全てが成功
        assert len(errors) == 0
        assert len(results) == len(symbols)
        for price in results.values():
            assert price is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])