#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider テストスイート - Issue #852対応
テストコードの本体からの分離完了版
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml
import sys
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from real_data_provider import (
    RealDataProvider,
    RealDataAnalysisEngine,
    RealStockData,
    DataFetchResult,
    DataFetchError,
    DataValidationError,
    ConfigurationError
)


class TestRealDataProvider:
    """RealDataProviderクラスのテストスイート"""

    @pytest.fixture
    def temp_config_file(self):
        """一時的な設定ファイルを作成"""
        config_data = {
            'symbols': {
                'TEST001.T': 'テスト銘柄1',
                'TEST002.T': 'テスト銘柄2'
            },
            'api_settings': {
                'cache_duration': 30,
                'request_interval': 0.1,
                'max_requests_per_minute': 120
            },
            'data_validation': {
                'min_price': 10.0,
                'max_price_change_percent': 20.0
            },
            'error_handling': {
                'max_retries': 2,
                'retry_delay': 0.5,
                'consecutive_failure_threshold': 3
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            return f.name

    @pytest.fixture
    def provider(self, temp_config_file):
        """テスト用プロバイダーインスタンス"""
        return RealDataProvider(config_path=temp_config_file)

    def test_initialization_with_config(self, provider):
        """設定ファイルからの初期化テスト"""
        assert len(provider.target_symbols) == 2
        assert 'TEST001.T' in provider.target_symbols
        assert provider.cache_duration == 30
        assert provider.max_retries == 2

    def test_initialization_without_config(self):
        """設定ファイルなしでの初期化テスト"""
        with patch('pathlib.Path.exists', return_value=False):
            provider = RealDataProvider(config_path="nonexistent.yaml")
            # デフォルト設定が使用されることを確認
            assert provider.cache_duration == 60
            assert provider.max_retries == 3

    def test_configuration_error_handling(self):
        """設定ファイルエラーハンドリングテスト"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(ConfigurationError):
                RealDataProvider(config_path="invalid.yaml")

    @patch('yfinance.Ticker')
    def test_successful_data_fetch(self, mock_ticker, provider):
        """正常なデータ取得テスト"""
        # モックデータ設定
        mock_hist = pd.DataFrame({
            'Open': [1000, 1010],
            'High': [1020, 1030],
            'Low': [990, 1000],
            'Close': [1010, 1025],
            'Volume': [100000, 110000]
        }, index=pd.date_range('2023-01-01', periods=2))

        mock_info = {
            'marketCap': 1000000000,
            'trailingPE': 15.5,
            'dividendYield': 0.02
        }

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        # テスト実行
        result = provider.get_real_stock_data('TEST001.T')

        # 結果検証
        assert result.success == True
        assert result.data.symbol == 'TEST001.T'
        assert result.data.current_price == 1025.0
        assert result.data.previous_close == 1010.0
        assert result.error is None

    @patch('yfinance.Ticker')
    def test_data_validation_error(self, mock_ticker, provider):
        """データ検証エラーテスト"""
        # 無効なデータ（高値 < 安値）
        mock_hist = pd.DataFrame({
            'Open': [1000],
            'High': [990],  # 高値が安値より低い
            'Low': [1000],
            'Close': [995],
            'Volume': [100000]
        }, index=pd.date_range('2023-01-01', periods=1))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker_instance.info = {}
        mock_ticker.return_value = mock_ticker_instance

        # テスト実行
        result = provider.get_real_stock_data('TEST001.T')

        # 結果検証
        assert result.success == False
        assert "High price is less than low price" in result.error

    @patch('yfinance.Ticker')
    def test_api_error_handling(self, mock_ticker, provider):
        """API エラーハンドリングテスト"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance

        # テスト実行
        result = provider.get_real_stock_data('TEST001.T')

        # 結果検証
        assert result.success == False
        assert "Yahoo Finance API error" in result.error

    @patch('yfinance.Ticker')
    def test_consecutive_failure_tracking(self, mock_ticker, provider):
        """連続失敗追跡テスト"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("Persistent Error")
        mock_ticker.return_value = mock_ticker_instance

        # 連続失敗を発生させる
        for _ in range(provider.consecutive_failure_threshold + 1):
            result = provider.get_real_stock_data('TEST001.T')
            assert result.success == False

        # 閾値を超えた後のテスト
        result = provider.get_real_stock_data('TEST001.T')
        assert "exceeded consecutive failure threshold" in result.error

    def test_cache_functionality(self, provider):
        """キャッシュ機能テスト"""
        # モックデータでキャッシュに直接追加
        cached_data = RealStockData(
            symbol='TEST001.T',
            current_price=1000.0,
            open_price=990.0,
            high_price=1010.0,
            low_price=985.0,
            volume=100000,
            previous_close=995.0
        )
        cached_result = DataFetchResult(True, cached_data, None, 'TEST001.T')

        # キャッシュキーを計算して設定
        import time
        cache_key = f"TEST001.T_{int(time.time() / provider.cache_duration)}"
        provider.data_cache[cache_key] = cached_result

        # キャッシュからの取得テスト
        result = provider.get_real_stock_data('TEST001.T')
        assert result.success == True
        assert result.data.symbol == 'TEST001.T'

    @pytest.mark.asyncio
    async def test_multiple_stocks_data_fetch(self, provider):
        """複数銘柄データ取得テスト"""
        with patch.object(provider, 'get_real_stock_data') as mock_get_data:
            # モック設定
            mock_get_data.return_value = DataFetchResult(
                True,
                RealStockData(
                    symbol='TEST001.T',
                    current_price=1000.0,
                    open_price=990.0,
                    high_price=1010.0,
                    low_price=985.0,
                    volume=100000,
                    previous_close=995.0
                ),
                None,
                'TEST001.T'
            )

            # テスト実行
            results = await provider.get_multiple_stocks_data(['TEST001.T', 'TEST002.T'])

            # 結果検証
            assert len(results) == 2
            assert 'TEST001.T' in results
            assert 'TEST002.T' in results

    def test_successful_data_only_extraction(self, provider):
        """成功データのみ抽出テスト"""
        results = {
            'SUCCESS.T': DataFetchResult(True, Mock(), None, 'SUCCESS.T'),
            'FAIL.T': DataFetchResult(False, None, "Error", 'FAIL.T')
        }

        successful_only = provider.get_successful_data_only(results)

        assert len(successful_only) == 1
        assert 'SUCCESS.T' in successful_only
        assert 'FAIL.T' not in successful_only

    @patch('yfinance.Ticker')
    def test_market_status_check(self, mock_ticker, provider):
        """市場状況確認テスト"""
        mock_hist = pd.DataFrame({
            'Close': [25000]
        }, index=pd.date_range('2023-01-01', periods=1))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance

        status = provider.get_market_status()

        assert 'market_open' in status
        assert 'last_update' in status

    @patch('yfinance.Ticker')
    def test_historical_data_fetch(self, mock_ticker, provider):
        """履歴データ取得テスト"""
        mock_hist = pd.DataFrame({
            'Open': [1000, 1010, 1020],
            'High': [1020, 1030, 1040],
            'Low': [990, 1000, 1010],
            'Close': [1010, 1025, 1035],
            'Volume': [100000, 110000, 120000]
        }, index=pd.date_range('2023-01-01', periods=3))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance

        result = provider.get_historical_data('TEST001.T', '3d')

        assert result is not None
        assert len(result) == 3


class TestRealDataAnalysisEngine:
    """RealDataAnalysisEngineクラスのテストスイート"""

    @pytest.fixture
    def analysis_engine(self):
        """テスト用分析エンジンインスタンス"""
        mock_provider = Mock()
        return RealDataAnalysisEngine(data_provider=mock_provider)

    @pytest.mark.asyncio
    async def test_daytrading_analysis(self, analysis_engine):
        """デイトレード分析テスト"""
        # モックデータ設定
        mock_data = RealStockData(
            symbol='TEST001.T',
            current_price=1000.0,
            open_price=990.0,
            high_price=1010.0,
            low_price=985.0,
            volume=1000000,
            previous_close=995.0
        )

        mock_results = {
            'TEST001.T': DataFetchResult(True, mock_data, None, 'TEST001.T')
        }

        analysis_engine.data_provider.get_multiple_stocks_data.return_value = mock_results
        analysis_engine.data_provider.get_successful_data_only.return_value = {
            'TEST001.T': mock_data
        }
        analysis_engine.data_provider.get_historical_data.return_value = pd.DataFrame({
            'Close': [990, 995, 1000]
        })

        # テスト実行
        recommendations = await analysis_engine.analyze_daytrading_opportunities(limit=5)

        # 結果検証
        assert len(recommendations) > 0
        assert recommendations[0]['symbol'] == 'TEST001'  # .T が除去されている

    def test_basic_indicators_calculation(self, analysis_engine):
        """基本テクニカル指標計算テスト"""
        # テストデータ作成
        hist_data = pd.DataFrame({
            'Close': [100, 102, 101, 105, 103, 107, 110, 108, 112, 115]
        }, index=pd.date_range('2023-01-01', periods=10))

        indicators = analysis_engine._calculate_basic_indicators(hist_data)

        assert 'ma5' in indicators
        assert isinstance(indicators['ma5'], float)

    def test_trading_score_calculation(self, analysis_engine):
        """取引スコア計算テスト"""
        mock_data = RealStockData(
            symbol='TEST001.T',
            current_price=1000.0,
            open_price=990.0,
            high_price=1010.0,
            low_price=985.0,
            volume=2000000,  # 高出来高
            previous_close=995.0
        )

        indicators = {
            'volatility': 3.0,  # 適度なボラティリティ
            'rsi': 50.0,  # 中立RSI
            'ma5': 1005.0,
            'ma20': 1000.0
        }

        score = analysis_engine._calculate_real_trading_score(mock_data, indicators)

        assert 0 <= score <= 100
        assert score > 50  # 良い条件なので高スコア期待


class TestDataValidation:
    """データ検証のテストスイート"""

    def test_price_validation(self):
        """価格データ検証テスト"""
        provider = RealDataProvider()

        # 正常データ
        valid_data = pd.DataFrame({
            'Open': [1000],
            'High': [1020],
            'Low': [990],
            'Close': [1010],
            'Volume': [100000]
        })

        assert provider._validate_stock_data(valid_data, 'TEST') == True

    def test_invalid_ohlc_relationship(self):
        """OHLC関係無効データテスト"""
        provider = RealDataProvider()

        # 無効データ（高値 < 安値）
        invalid_data = pd.DataFrame({
            'Open': [1000],
            'High': [990],  # 高値 < 安値
            'Low': [1000],
            'Close': [995],
            'Volume': [100000]
        })

        with pytest.raises(DataValidationError):
            provider._validate_stock_data(invalid_data, 'TEST')

    def test_missing_columns(self):
        """必要カラム不足テスト"""
        provider = RealDataProvider()

        # Volumeカラムなし
        incomplete_data = pd.DataFrame({
            'Open': [1000],
            'High': [1020],
            'Low': [990],
            'Close': [1010]
        })

        with pytest.raises(DataValidationError):
            provider._validate_stock_data(incomplete_data, 'TEST')


def run_comprehensive_test():
    """包括的テスト実行"""
    print("=== Real Data Provider 包括テスト実行 ===")

    try:
        # pytest実行
        exit_code = pytest.main([
            __file__,
            '-v',
            '--tb=short',
            '--maxfail=5'
        ])

        if exit_code == 0:
            print("✅ 全テストが正常に完了しました")
        else:
            print(f"❌ テストで問題が発生しました (exit code: {exit_code})")

    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生しました: {e}")


if __name__ == "__main__":
    run_comprehensive_test()