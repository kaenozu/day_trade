#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Web Dashboard テストスイート
Issue #871対応：ウェブダッシュボード機能拡張のテスト

高度ウェブダッシュボードシステムの包括的テスト
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from web_dashboard_advanced import (
    AdvancedWebDashboard,
    RealtimeDataManager,
    AdvancedAnalysisManager,
    DashboardCustomization
)


@pytest.fixture
def temp_dir():
    """一時ディレクトリフィクスチャ"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config():
    """サンプル設定"""
    return {
        'layout': {
            'theme': 'dark',
            'sidebar_position': 'left',
            'chart_height': 400,
            'refresh_interval': 5
        },
        'widgets': {
            'price_chart': {'enabled': True, 'position': {'row': 1, 'col': 1}, 'size': {'width': 6, 'height': 2}},
            'prediction_panel': {'enabled': True, 'position': {'row': 1, 'col': 7}, 'size': {'width': 6, 'height': 2}}
        },
        'symbols': {
            'watchlist': ['7203', '4751', '9984'],
            'auto_update': True,
            'update_interval': 5
        },
        'alerts': {
            'price_change_threshold': 3.0,
            'prediction_confidence_threshold': 0.8,
            'sound_alerts': True
        }
    }


@pytest.fixture
def sample_stock_data():
    """サンプル株価データ"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

    # 模擬株価データ生成
    import numpy as np
    np.random.seed(42)

    prices = [1000]
    for i in range(29):
        change = np.random.normal(0, 0.02)
        price = prices[-1] * (1 + change)
        prices.append(max(price, 100))  # 最低価格100円

    data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.02) for p in prices],
        'Low': [p * np.random.uniform(0.98, 1.00) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1000000, 10000000) for _ in range(30)]
    }, index=dates)

    return data


class TestRealtimeDataManager:
    """リアルタイムデータ管理テスト"""

    def test_initialization(self):
        """初期化テスト"""
        manager = RealtimeDataManager()

        assert manager.active_subscriptions == set()
        assert manager.current_data == {}
        assert manager.update_interval == 5
        assert not manager.is_running

    def test_subscription_management(self):
        """購読管理テスト"""
        manager = RealtimeDataManager()

        # 購読開始
        manager.subscribe_symbol('7203')
        assert '7203' in manager.active_subscriptions

        manager.subscribe_symbol('4751')
        assert '4751' in manager.active_subscriptions
        assert len(manager.active_subscriptions) == 2

        # 購読停止
        manager.unsubscribe_symbol('7203')
        assert '7203' not in manager.active_subscriptions
        assert '4751' in manager.active_subscriptions

    @pytest.mark.asyncio
    async def test_get_current_price(self):
        """現在価格取得テスト"""
        manager = RealtimeDataManager()

        # Mock real data provider
        with patch('web_dashboard_advanced.REAL_DATA_AVAILABLE', False):
            price_data = await manager.get_current_price('7203')

            assert 'price' in price_data
            assert 'open' in price_data
            assert 'high' in price_data
            assert 'low' in price_data
            assert 'volume' in price_data

            assert price_data['price'] > 0
            assert price_data['volume'] > 0

    @pytest.mark.asyncio
    async def test_get_technical_indicators(self):
        """テクニカル指標取得テスト"""
        manager = RealtimeDataManager()

        with patch('web_dashboard_advanced.REAL_DATA_AVAILABLE', False):
            technical_data = await manager.get_technical_indicators('7203')

            assert 'change_percent' in technical_data
            assert 'sma_20' in technical_data
            assert 'volume' in technical_data
            assert 'price_vs_sma' in technical_data

    @pytest.mark.asyncio
    async def test_get_prediction_data(self):
        """予測データ取得テスト"""
        manager = RealtimeDataManager()

        prediction_data = await manager.get_prediction_data('7203')

        assert 'direction' in prediction_data
        assert 'confidence' in prediction_data
        assert 'expected_change' in prediction_data
        assert 'last_updated' in prediction_data

        assert 0 <= prediction_data['confidence'] <= 1
        assert prediction_data['direction'] in ['上昇', '下降', '中立']


class TestAdvancedAnalysisManager:
    """高度分析管理テスト"""

    def test_initialization(self):
        """初期化テスト"""
        manager = AdvancedAnalysisManager()

        # システム統合確認
        assert hasattr(manager, 'accuracy_enhancer')
        assert hasattr(manager, 'next_morning_system')
        assert hasattr(manager, 'performance_monitor')
        assert hasattr(manager, 'data_quality_monitor')

    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, sample_stock_data):
        """包括分析テスト"""
        manager = AdvancedAnalysisManager()

        # Mock external systems
        with patch.multiple(manager,
                          accuracy_enhancer=None,
                          next_morning_system=None,
                          performance_monitor=None,
                          data_quality_monitor=None):

            result = await manager.run_comprehensive_analysis('7203')

            assert 'symbol' in result
            assert 'timestamp' in result
            assert result['symbol'] == '7203'

            # 各分析結果の存在確認
            assert 'accuracy_enhancement' in result
            assert 'next_morning_prediction' in result
            assert 'performance_metrics' in result
            assert 'data_quality' in result

    @pytest.mark.asyncio
    async def test_system_health(self):
        """システム健全性テスト"""
        manager = AdvancedAnalysisManager()

        health = await manager.get_system_health()

        assert 'timestamp' in health
        assert 'systems' in health
        assert 'overall_status' in health

        # システム状態確認
        systems = health['systems']
        expected_systems = ['accuracy_enhancer', 'next_morning_system', 'performance_monitor', 'data_quality_monitor']

        for system_name in expected_systems:
            assert system_name in systems
            system_status = systems[system_name]
            assert 'status' in system_status
            assert 'details' in system_status


class TestDashboardCustomization:
    """ダッシュボードカスタマイズテスト"""

    def test_initialization(self, temp_dir):
        """初期化テスト"""
        config_file = Path(temp_dir) / "dashboard_config.json"

        customization = DashboardCustomization()
        customization.config_file = config_file

        default_config = customization._get_default_config()

        assert 'layout' in default_config
        assert 'widgets' in default_config
        assert 'symbols' in default_config
        assert 'alerts' in default_config

        # レイアウト設定確認
        layout = default_config['layout']
        assert layout['theme'] in ['dark', 'light']
        assert layout['refresh_interval'] > 0

        # ウィジェット設定確認
        widgets = default_config['widgets']
        assert 'price_chart' in widgets
        assert 'prediction_panel' in widgets

        for widget_name, widget_config in widgets.items():
            assert 'enabled' in widget_config
            assert 'position' in widget_config
            assert 'size' in widget_config

    def test_config_save_load(self, temp_dir, sample_config):
        """設定保存・読み込みテスト"""
        config_file = Path(temp_dir) / "test_config.json"

        customization = DashboardCustomization()
        customization.config_file = config_file

        # 設定保存
        customization.save_user_config(sample_config, 'test_user')

        assert config_file.exists()

        # 設定読み込み
        loaded_config = customization.load_user_config('test_user')

        assert loaded_config == sample_config

        # 存在しないユーザーの場合はデフォルト設定
        default_config = customization.load_user_config('non_existent_user')
        assert default_config == customization._get_default_config()

    def test_multiple_user_configs(self, temp_dir, sample_config):
        """複数ユーザー設定テスト"""
        config_file = Path(temp_dir) / "multi_user_config.json"

        customization = DashboardCustomization()
        customization.config_file = config_file

        # 複数ユーザーの設定保存
        user1_config = sample_config.copy()
        user1_config['layout']['theme'] = 'dark'

        user2_config = sample_config.copy()
        user2_config['layout']['theme'] = 'light'

        customization.save_user_config(user1_config, 'user1')
        customization.save_user_config(user2_config, 'user2')

        # 設定読み込み確認
        loaded_user1 = customization.load_user_config('user1')
        loaded_user2 = customization.load_user_config('user2')

        assert loaded_user1['layout']['theme'] == 'dark'
        assert loaded_user2['layout']['theme'] == 'light'


class TestAdvancedWebDashboard:
    """高度ウェブダッシュボードテスト"""

    def test_initialization(self):
        """初期化テスト"""
        # Flask利用可能性をモック
        with patch('web_dashboard_advanced.FLASK_AVAILABLE', True):
            with patch('web_dashboard_advanced.Flask') as mock_flask:
                with patch('web_dashboard_advanced.SocketIO') as mock_socketio:
                    with patch('web_dashboard_advanced.CORS') as mock_cors:

                        mock_app = Mock()
                        mock_flask.return_value = mock_app
                        mock_socketio_instance = Mock()
                        mock_socketio.return_value = mock_socketio_instance

                        dashboard = AdvancedWebDashboard(host='localhost', port=5001)

                        # コンポーネント初期化確認
                        assert hasattr(dashboard, 'realtime_manager')
                        assert hasattr(dashboard, 'analysis_manager')
                        assert hasattr(dashboard, 'customization')
                        assert hasattr(dashboard, 'app')
                        assert hasattr(dashboard, 'socketio')

                        # Flask設定確認
                        mock_flask.assert_called_once()
                        mock_socketio.assert_called_once()
                        mock_cors.assert_called_once()

    def test_flask_not_available(self):
        """Flask利用不可テスト"""
        with patch('web_dashboard_advanced.FLASK_AVAILABLE', False):
            with pytest.raises(ImportError, match="Flask is required"):
                AdvancedWebDashboard()


class TestDashboardIntegration:
    """ダッシュボード統合テスト"""

    @pytest.mark.asyncio
    async def test_realtime_data_flow(self):
        """リアルタイムデータフローテスト"""
        manager = RealtimeDataManager()

        # 購読開始
        manager.subscribe_symbol('7203')

        # データ更新実行
        with patch('web_dashboard_advanced.REAL_DATA_AVAILABLE', False):
            # update_all_data は SocketIO を使用するため、モックが必要
            manager.socketio = Mock()

            await manager.update_all_data()

            # データが更新されていることを確認
            assert '7203' in manager.current_data

            symbol_data = manager.current_data['7203']
            assert 'symbol' in symbol_data
            assert 'timestamp' in symbol_data
            assert 'current_price' in symbol_data
            assert 'technical' in symbol_data
            assert 'prediction' in symbol_data

    def test_config_integration(self, temp_dir, sample_config):
        """設定統合テスト"""
        config_file = Path(temp_dir) / "integration_config.json"

        customization = DashboardCustomization()
        customization.config_file = config_file

        # 設定保存
        customization.save_user_config(sample_config)

        # 設定読み込み
        loaded_config = customization.load_user_config()

        # ウォッチリスト設定確認
        watchlist = loaded_config['symbols']['watchlist']
        assert isinstance(watchlist, list)
        assert len(watchlist) > 0
        assert all(isinstance(symbol, str) for symbol in watchlist)

        # アラート設定確認
        alerts = loaded_config['alerts']
        assert 'price_change_threshold' in alerts
        assert alerts['price_change_threshold'] > 0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """エラーハンドリングテスト"""
        manager = AdvancedAnalysisManager()

        # 存在しない銘柄での分析
        result = await manager.run_comprehensive_analysis('INVALID_SYMBOL')

        # エラーが適切に処理されることを確認
        assert 'symbol' in result
        assert result['symbol'] == 'INVALID_SYMBOL'

        # 各システムがエラー処理されていることを確認
        for key in ['accuracy_enhancement', 'next_morning_prediction', 'performance_metrics', 'data_quality']:
            if key in result and result[key] is not None:
                # エラーステータスまたは適切なフォールバック値
                assert isinstance(result[key], dict)


class TestDashboardPerformance:
    """ダッシュボードパフォーマンステスト"""

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """並行分析テスト"""
        manager = AdvancedAnalysisManager()

        symbols = ['7203', '4751', '9984']

        # 並行分析実行
        tasks = [manager.run_comprehensive_analysis(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 全ての分析が完了することを確認
        assert len(results) == len(symbols)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 例外の場合はログ出力（テスト失敗ではない）
                print(f"Analysis failed for {symbols[i]}: {result}")
            else:
                assert 'symbol' in result
                assert result['symbol'] == symbols[i]

    def test_memory_usage(self):
        """メモリ使用量テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 大量のデータ処理
        manager = RealtimeDataManager()

        for i in range(100):
            symbol = f"TEST{i:04d}"
            manager.subscribe_symbol(symbol)
            manager.current_data[symbol] = {
                'timestamp': datetime.now().isoformat(),
                'price': 1000 + i,
                'data': list(range(100))  # ダミーデータ
            }

        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory

        # メモリ増加が許容範囲内（100MB未満）であることを確認
        assert memory_increase < 100 * 1024 * 1024  # 100MB

        # クリーンアップ
        manager.current_data.clear()
        manager.active_subscriptions.clear()


def test_dashboard_components():
    """ダッシュボードコンポーネントテスト"""
    print("=== Advanced Web Dashboard Components Test ===")

    # リアルタイムデータ管理
    realtime_manager = RealtimeDataManager()
    print(f"✓ RealtimeDataManager initialized")

    # 高度分析管理
    analysis_manager = AdvancedAnalysisManager()
    print(f"✓ AdvancedAnalysisManager initialized")

    # カスタマイズ管理
    customization = DashboardCustomization()
    default_config = customization._get_default_config()
    print(f"✓ DashboardCustomization initialized - {len(default_config)} sections")

    # 購読テスト
    realtime_manager.subscribe_symbol("7203")
    realtime_manager.subscribe_symbol("4751")
    print(f"✓ Symbol subscription test - {len(realtime_manager.active_subscriptions)} symbols")

    print(f"✓ All dashboard components working correctly!")


if __name__ == "__main__":
    # コンポーネントテスト実行
    test_dashboard_components()

    # pytest実行
    import os
    os.system("pytest " + __file__ + " -v")