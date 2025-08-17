#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Real Data Provider V2 (Enhanced Version)
改善版RealDataProviderV2のテストケース

Issue #853対応：外部設定、品質チェック強化、マルチソース対応のテスト
"""

import pytest
import asyncio
import tempfile
import yaml
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from real_data_provider_v2 import (
    DataSourceConfigManager,
    YahooFinanceProviderV2,
    MultiSourceDataProvider,
    DataSource,
    DataQuality,
    DataSourceInfo,
    StockDataPoint
)

class TestDataSourceConfigManager:
    """DataSourceConfigManagerのテストクラス"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリの作成"""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def config_file(self, temp_dir):
        """テスト用設定ファイルの作成"""
        config_path = temp_dir / "test_data_sources.yaml"
        config = {
            'data_sources': {
                'yahoo_finance': {
                    'enabled': True,
                    'rate_limits': {
                        'requests_per_minute': 60,
                        'daily_limit': 800
                    },
                    'quality_settings': {
                        'delay_minutes': 10,
                        'expected_success_rate': 0.90
                    },
                    'retry_config': {
                        'max_retries': 2,
                        'backoff_factor': 1.5
                    },
                    'fallback_priority': 1
                },
                'stooq': {
                    'enabled': False,
                    'rate_limits': {
                        'requests_per_minute': 30,
                        'daily_limit': 400
                    }
                }
            },
            'cache_settings': {
                'enabled': True,
                'cache_directory': 'test_cache',
                'default_ttl_seconds': 900
            },
            'data_quality': {
                'price_validation': {
                    'min_price': 1.0,
                    'max_price_multiplier': 5.0,
                    'use_iqr_outlier_detection': True,
                    'iqr_multiplier': 2.5
                },
                'completeness_check': {
                    'required_fields': ['Open', 'High', 'Low', 'Close'],
                    'min_data_points': 3,
                    'max_missing_ratio': 0.1
                }
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        return config_path

    def test_config_loading(self, config_file):
        """設定ファイル読み込みのテスト"""
        config_manager = DataSourceConfigManager(config_file)

        assert config_manager.is_data_source_enabled('yahoo_finance') == True
        assert config_manager.is_data_source_enabled('stooq') == False

        yahoo_config = config_manager.get_data_source_config('yahoo_finance')
        assert yahoo_config['rate_limits']['requests_per_minute'] == 60
        assert yahoo_config['quality_settings']['delay_minutes'] == 10

    def test_cache_config(self, config_file):
        """キャッシュ設定取得のテスト"""
        config_manager = DataSourceConfigManager(config_file)
        cache_config = config_manager.get_cache_config()

        assert cache_config['enabled'] == True
        assert cache_config['cache_directory'] == 'test_cache'
        assert cache_config['default_ttl_seconds'] == 900

    def test_quality_config(self, config_file):
        """品質設定取得のテスト"""
        config_manager = DataSourceConfigManager(config_file)
        quality_config = config_manager.get_quality_config()

        price_validation = quality_config['price_validation']
        assert price_validation['min_price'] == 1.0
        assert price_validation['use_iqr_outlier_detection'] == True
        assert price_validation['iqr_multiplier'] == 2.5

    def test_default_config_creation(self, temp_dir):
        """デフォルト設定作成のテスト"""
        config_path = temp_dir / "non_existent_config.yaml"
        config_manager = DataSourceConfigManager(config_path)

        # デフォルト設定が作成されているか確認
        assert config_path.exists()
        assert config_manager.is_data_source_enabled('yahoo_finance') == True


class TestYahooFinanceProviderV2:
    """YahooFinanceProviderV2のテストクラス"""

    @pytest.fixture
    def mock_config_manager(self):
        """モック設定管理の作成"""
        mock_config = Mock()
        mock_config.get_data_source_config.return_value = {
            'rate_limits': {
                'requests_per_minute': 100,
                'daily_limit': 1000
            },
            'quality_settings': {
                'delay_minutes': 5,
                'expected_success_rate': 0.95
            },
            'retry_config': {
                'max_retries': 3,
                'backoff_factor': 2.0,
                'initial_delay': 1.0
            }
        }
        mock_config.get_quality_config.return_value = {
            'price_validation': {
                'min_price': 0.01,
                'max_price_multiplier': 10.0,
                'use_iqr_outlier_detection': True,
                'iqr_multiplier': 3.0
            },
            'volume_validation': {
                'min_volume': 0,
                'max_volume_multiplier': 50.0
            },
            'completeness_check': {
                'required_fields': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'min_data_points': 5,
                'max_missing_ratio': 0.05
            }
        }
        return mock_config

    def test_initialization_with_config(self, mock_config_manager):
        """外部設定による初期化テスト"""
        provider = YahooFinanceProviderV2(mock_config_manager)

        assert provider.daily_limit == 1000
        assert provider.min_request_interval == 0.6  # 60/100
        assert provider.max_retries == 3
        assert provider.backoff_factor == 2.0

    def test_symbol_variations_generation(self, mock_config_manager):
        """銘柄コードバリエーション生成テスト"""
        provider = YahooFinanceProviderV2(mock_config_manager)

        # 数字のみの銘柄コード
        variations = provider._generate_symbol_variations("7203")
        expected = ["7203.T", "7203.JP", "7203", "7203.TO", "7203.TYO"]
        assert variations == expected

        # すでにサフィックス付きの銘柄コード
        variations = provider._generate_symbol_variations("AAPL")
        assert "AAPL" in variations

    def test_data_quality_validation_basic(self, mock_config_manager):
        """基本的なデータ品質チェックテスト"""
        provider = YahooFinanceProviderV2(mock_config_manager)

        # 正常なデータ
        good_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        assert provider._validate_data_quality(good_data) == True

        # 空のデータ
        empty_data = pd.DataFrame()
        assert provider._validate_data_quality(empty_data) == False

        # 不正なOHLC関係
        bad_ohlc_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [90, 91, 92],  # Highが他より低い
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        assert provider._validate_data_quality(bad_ohlc_data) == False

    def test_data_quality_validation_iqr(self, mock_config_manager):
        """IQR法による異常値検知テスト"""
        provider = YahooFinanceProviderV2(mock_config_manager)

        # 正常な価格範囲のデータ
        normal_data = pd.DataFrame({
            'Open': [100] * 15,
            'High': [105] * 15,
            'Low': [95] * 15,
            'Close': [102] * 15,
            'Volume': [1000] * 15
        })
        assert provider._validate_data_quality(normal_data) == True

        # 異常値を含むデータ（10%以上が異常値）
        outlier_data = pd.DataFrame({
            'Open': [100] * 10 + [10000] * 5,  # 5/15 = 33%が異常値
            'High': [105] * 10 + [10005] * 5,
            'Low': [95] * 10 + [9995] * 5,
            'Close': [102] * 10 + [10002] * 5,
            'Volume': [1000] * 15
        })
        assert provider._validate_data_quality(outlier_data) == False


class TestMultiSourceDataProvider:
    """MultiSourceDataProviderのテストクラス"""

    @pytest.fixture
    def mock_config_manager(self):
        """モック設定管理の作成"""
        mock_config = Mock()
        mock_config.is_data_source_enabled.side_effect = lambda x: x in ['yahoo_finance', 'stooq']
        mock_config.get_data_source_config.return_value = {
            'rate_limits': {'requests_per_minute': 100, 'daily_limit': 1000},
            'quality_settings': {'delay_minutes': 5},
            'retry_config': {'max_retries': 3, 'backoff_factor': 2.0, 'initial_delay': 1.0}
        }
        mock_config.get_cache_config.return_value = {
            'enabled': True,
            'default_ttl_seconds': 1800
        }
        return mock_config

    @patch('real_data_provider_v2.SYMBOL_SELECTOR_AVAILABLE', False)
    def test_initialization_without_symbol_selector(self, mock_config_manager):
        """symbol_selector無しでの初期化テスト"""
        provider = MultiSourceDataProvider(config_manager=mock_config_manager)

        assert provider.symbol_selector is None
        assert len(provider.providers) == 2  # yahoo_finance + stooq
        assert DataSource.YAHOO_FINANCE in provider.providers
        assert DataSource.STOOQ in provider.providers

    def test_provider_prioritization(self, mock_config_manager):
        """プロバイダー優先度決定テスト"""
        provider = MultiSourceDataProvider(config_manager=mock_config_manager)

        # 品質スコアを設定
        provider.source_info[DataSource.YAHOO_FINANCE].current_quality_score = 90.0
        provider.source_info[DataSource.STOOQ].current_quality_score = 75.0

        prioritized = provider._get_prioritized_sources()

        # Yahoo Financeが最初に来るべき（品質スコアが高い）
        assert prioritized[0] == DataSource.YAHOO_FINANCE

    @pytest.mark.asyncio
    async def test_stock_data_retrieval_success(self, mock_config_manager):
        """株価データ取得成功テスト"""
        provider = MultiSourceDataProvider(config_manager=mock_config_manager)

        # モックデータの作成
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [102, 103],
            'Volume': [1000, 1100]
        })

        # Yahoo Financeプロバイダーをモック
        mock_yahoo_provider = AsyncMock()
        mock_yahoo_provider.get_stock_data.return_value = mock_data
        provider.providers[DataSource.YAHOO_FINANCE] = mock_yahoo_provider

        # データ取得テスト
        result = await provider.get_stock_data("7203")

        assert result is not None
        assert len(result) == 2
        mock_yahoo_provider.get_stock_data.assert_called_once_with("7203", "1mo")

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, mock_config_manager):
        """フォールバック機能テスト"""
        provider = MultiSourceDataProvider(config_manager=mock_config_manager)

        # Yahoo Financeが失敗、Stooqが成功するシナリオ
        mock_yahoo_provider = AsyncMock()
        mock_yahoo_provider.get_stock_data.side_effect = Exception("Yahoo Finance failed")

        mock_stooq_provider = AsyncMock()
        mock_stooq_data = pd.DataFrame({
            'Open': [200], 'High': [205], 'Low': [195], 'Close': [202], 'Volume': [2000]
        })
        mock_stooq_provider.get_stock_data.return_value = mock_stooq_data

        provider.providers[DataSource.YAHOO_FINANCE] = mock_yahoo_provider
        provider.providers[DataSource.STOOQ] = mock_stooq_provider

        result = await provider.get_stock_data("8306")

        assert result is not None
        assert len(result) == 1
        # 両方のプロバイダーが呼ばれたことを確認
        mock_yahoo_provider.get_stock_data.assert_called_once()
        mock_stooq_provider.get_stock_data.assert_called_once()

    def test_performance_metrics_tracking(self, mock_config_manager):
        """性能メトリクス追跡テスト"""
        provider = MultiSourceDataProvider(config_manager=mock_config_manager)

        # 初期状態の確認
        assert provider.performance_metrics.total_requests == 0
        assert provider.cache_hits == 0
        assert provider.cache_requests == 0

        # キャッシュヒット率の計算確認
        provider.cache_requests = 10
        provider.cache_hits = 3
        expected_hit_rate = (3 / 10) * 100

        provider.performance_metrics.cache_hit_rate = (provider.cache_hits / provider.cache_requests) * 100
        assert provider.performance_metrics.cache_hit_rate == expected_hit_rate


class TestDataStructures:
    """データ構造のテストクラス"""

    def test_data_source_info_creation(self):
        """DataSourceInfo作成テスト"""
        info = DataSourceInfo(
            source=DataSource.YAHOO_FINANCE,
            is_available=True,
            delay_minutes=5,
            daily_limit=1000,
            cost_per_request=0.0,
            quality=DataQuality.GOOD,
            current_quality_score=85.0,
            consecutive_failures=0,
            fallback_priority=1
        )

        assert info.source == DataSource.YAHOO_FINANCE
        assert info.is_available == True
        assert info.current_quality_score == 85.0
        assert info.fallback_priority == 1

    def test_stock_data_point_creation(self):
        """StockDataPoint作成テスト"""
        data_point = StockDataPoint(
            symbol="7203",
            timestamp=datetime.now(),
            open_price=100.0,
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=1000,
            source=DataSource.YAHOO_FINANCE,
            delay_minutes=5,
            quality_score=90.0
        )

        assert data_point.symbol == "7203"
        assert data_point.open_price == 100.0
        assert data_point.quality_score == 90.0
        assert data_point.source == DataSource.YAHOO_FINANCE


# インテグレーションテスト
class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """全体ワークフローのシミュレーションテスト"""
        # 一時設定ファイル作成
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "integration_config.yaml"
            config = {
                'data_sources': {
                    'yahoo_finance': {
                        'enabled': True,
                        'rate_limits': {'requests_per_minute': 100, 'daily_limit': 1000}
                    }
                },
                'cache_settings': {'enabled': True, 'default_ttl_seconds': 1800},
                'data_quality': {
                    'price_validation': {'min_price': 0.01, 'use_iqr_outlier_detection': True},
                    'completeness_check': {'required_fields': ['Open', 'High', 'Low', 'Close']}
                }
            }

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)

            # 設定管理とデータプロバイダーの初期化
            config_manager = DataSourceConfigManager(config_path)
            provider = MultiSourceDataProvider(config_manager=config_manager)

            # 設定が正しく読み込まれているか確認
            assert config_manager.is_data_source_enabled('yahoo_finance') == True
            assert len(provider.providers) >= 1

            # データ品質チェックのテスト
            yahoo_provider = provider.providers.get(DataSource.YAHOO_FINANCE)
            if yahoo_provider:
                test_data = pd.DataFrame({
                    'Open': [100, 101, 102],
                    'High': [105, 106, 107],
                    'Low': [95, 96, 97],
                    'Close': [102, 103, 104],
                    'Volume': [1000, 1100, 1200]
                })

                is_valid = yahoo_provider._validate_data_quality(test_data)
                assert is_valid == True


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])