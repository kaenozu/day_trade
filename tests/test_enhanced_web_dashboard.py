#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Enhanced Web Dashboard
拡張ウェブダッシュボードのテストケース

Issue #871対応：リアルタイム・分析・予測・モニタリング・カスタマイズ機能のテスト
"""

import pytest
import asyncio
import tempfile
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Webライブラリが利用可能かチェック
try:
    from enhanced_web_dashboard import (
        EnhancedWebDashboard,
        RealTimeDataManager,
        AdvancedVisualization,
        AlertManager,
        DashboardConfig,
        AlertConfig,
        UserPreferences,
        DashboardTheme,
        UpdateFrequency,
        create_enhanced_web_dashboard
    )
    WEB_TESTS_AVAILABLE = True
except ImportError as e:
    WEB_TESTS_AVAILABLE = False
    print(f"Web dashboard tests skipped: {e}")


@pytest.mark.skipif(not WEB_TESTS_AVAILABLE, reason="Web dependencies not available")
class TestDashboardConfig:
    """DashboardConfigのテストクラス"""

    def test_default_config_creation(self):
        """デフォルト設定作成テスト"""
        config = DashboardConfig()

        assert config.theme == DashboardTheme.FINANCIAL
        assert config.update_frequency == UpdateFrequency.MEDIUM
        assert config.auto_refresh == True
        assert len(config.default_symbols) == 3
        assert config.charts_per_row == 2

    def test_config_with_custom_values(self):
        """カスタム値での設定作成テスト"""
        config = DashboardConfig(
            theme=DashboardTheme.DARK,
            update_frequency=UpdateFrequency.HIGH,
            auto_refresh=False,
            charts_per_row=3
        )

        assert config.theme == DashboardTheme.DARK
        assert config.update_frequency == UpdateFrequency.HIGH
        assert config.auto_refresh == False
        assert config.charts_per_row == 3


@pytest.mark.skipif(not WEB_TESTS_AVAILABLE, reason="Web dependencies not available")
class TestRealTimeDataManager:
    """RealTimeDataManagerのテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return DashboardConfig(update_frequency=UpdateFrequency.HIGH)

    @pytest.fixture
    def manager(self, config):
        """RealTimeDataManagerインスタンス"""
        return RealTimeDataManager(config)

    @pytest.fixture
    def mock_socketio(self):
        """モックSocketIO"""
        mock_io = MagicMock()
        mock_io.emit = MagicMock()
        return mock_io

    def test_initialization(self, manager):
        """初期化テスト"""
        assert manager.config.update_frequency == UpdateFrequency.HIGH
        assert isinstance(manager.active_subscriptions, set)
        assert isinstance(manager.data_cache, dict)

    @pytest.mark.asyncio
    async def test_subscribe_symbol(self, manager, mock_socketio):
        """銘柄購読テスト"""
        symbol = "7203"

        with patch.object(manager, '_send_initial_data', new_callable=AsyncMock) as mock_send:
            await manager.subscribe_symbol(symbol, mock_socketio)

            assert symbol in manager.active_subscriptions
            mock_send.assert_called_once_with(symbol, mock_socketio)

    @pytest.mark.asyncio
    async def test_unsubscribe_symbol(self, manager, mock_socketio):
        """銘柄購読停止テスト"""
        symbol = "7203"
        manager.active_subscriptions.add(symbol)

        await manager.unsubscribe_symbol(symbol)

        assert symbol not in manager.active_subscriptions

    def test_get_update_interval(self, manager):
        """更新間隔取得テスト"""
        interval = manager._get_update_interval()
        assert interval == 5.0  # HIGH frequency

    @pytest.mark.asyncio
    async def test_send_initial_data_with_data_provider(self, manager, mock_socketio):
        """データプロバイダーありでの初回データ送信テスト"""
        symbol = "7203"

        # モックデータ作成
        mock_data = pd.DataFrame({
            'Close': [100, 102, 101],
            'Volume': [1000, 1100, 1050]
        })

        mock_provider = AsyncMock()
        mock_provider.get_stock_data.return_value = mock_data
        manager.data_provider = mock_provider

        await manager._send_initial_data(symbol, mock_socketio)

        # SocketIOのemitが呼ばれたことを確認
        mock_socketio.emit.assert_called_once()
        call_args = mock_socketio.emit.call_args
        assert call_args[0][0] == 'price_update'
        assert call_args[0][1]['symbol'] == symbol
        assert call_args[0][1]['price'] == 101  # 最新価格

    @pytest.mark.asyncio
    async def test_send_initial_data_without_data_provider(self, manager, mock_socketio):
        """データプロバイダーなしでの初回データ送信テスト"""
        symbol = "7203"
        manager.data_provider = None

        await manager._send_initial_data(symbol, mock_socketio)

        # データプロバイダーがないのでemitは呼ばれない
        mock_socketio.emit.assert_not_called()


@pytest.mark.skipif(not WEB_TESTS_AVAILABLE, reason="Web dependencies not available")
class TestAdvancedVisualization:
    """AdvancedVisualizationのテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return DashboardConfig(theme=DashboardTheme.FINANCIAL)

    @pytest.fixture
    def visualization(self, config):
        """AdvancedVisualizationインスタンス"""
        return AdvancedVisualization(config)

    @pytest.fixture
    def sample_data(self):
        """サンプル株価データ"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')

        data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(100, 110, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(1000, 10000, 50),
            'SMA_20': np.random.uniform(98, 102, 50),
            'SMA_50': np.random.uniform(99, 101, 50),
            'RSI': np.random.uniform(30, 70, 50),
            'BB_Upper': np.random.uniform(105, 110, 50),
            'BB_Lower': np.random.uniform(90, 95, 50)
        }, index=dates)

        return data

    def test_create_enhanced_candlestick_chart(self, visualization, sample_data):
        """拡張ローソク足チャート作成テスト"""
        symbol = "7203"
        indicators = ["RSI", "MACD"]

        result = visualization.create_enhanced_candlestick_chart(
            sample_data, symbol, indicators
        )

        assert result['success'] == True
        assert 'chart' in result
        assert isinstance(result['chart'], dict)

    def test_create_enhanced_candlestick_chart_error_handling(self, visualization):
        """チャート作成エラーハンドリングテスト"""
        # 不正なデータでテスト
        bad_data = pd.DataFrame({'bad_column': [1, 2, 3]})
        symbol = "TEST"

        result = visualization.create_enhanced_candlestick_chart(bad_data, symbol)

        assert result['success'] == False
        assert 'error' in result

    def test_create_prediction_chart(self, visualization, sample_data):
        """予測チャート作成テスト"""
        symbol = "7203"
        predictions = {
            'predictions': [105, 106, 104, 107, 105],
            'confidence_intervals': {
                'upper': [110, 111, 109, 112, 110],
                'lower': [100, 101, 99, 102, 100]
            }
        }

        result = visualization.create_prediction_chart(
            sample_data, predictions, symbol
        )

        assert result['success'] == True
        assert 'chart' in result

    def test_create_performance_dashboard(self, visualization):
        """パフォーマンスダッシュボード作成テスト"""
        performance_data = {
            'accuracy_history': [0.85, 0.87, 0.86, 0.88, 0.89],
            'confidence_distribution': [0.7, 0.8, 0.9, 0.85, 0.75],
            'feature_importance': {
                'RSI': 0.3,
                'MACD': 0.25,
                'Volume': 0.2,
                'Price': 0.25
            },
            'data_quality': {
                'Completeness': 0.95,
                'Accuracy': 0.92,
                'Timeliness': 0.88
            }
        }

        result = visualization.create_performance_dashboard(performance_data)

        assert result['success'] == True
        assert 'chart' in result


@pytest.mark.skipif(not WEB_TESTS_AVAILABLE, reason="Web dependencies not available")
class TestAlertManager:
    """AlertManagerのテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return DashboardConfig(alerts_enabled=True)

    @pytest.fixture
    def alert_manager(self, config):
        """AlertManagerインスタンス"""
        return AlertManager(config)

    @pytest.fixture
    def sample_alerts(self):
        """サンプルアラート設定"""
        return [
            AlertConfig(
                alert_id="alert_1",
                symbol="7203",
                alert_type="price_threshold",
                condition=">",
                threshold=150.0,
                enabled=True
            ),
            AlertConfig(
                alert_id="alert_2",
                symbol="7203",
                alert_type="price_change_percent",
                condition=">",
                threshold=5.0,
                enabled=True
            )
        ]

    @pytest.mark.asyncio
    async def test_check_alerts_price_threshold(self, alert_manager, sample_alerts):
        """価格閾値アラートチェックテスト"""
        current_data = {
            'symbol': '7203',
            'price': 155.0,
            'change': 5.0,
            'change_percent': 3.3
        }

        triggered_alerts = await alert_manager.check_alerts(
            "7203", current_data, sample_alerts
        )

        # 価格閾値アラートがトリガーされるはず
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0]['alert_id'] == 'alert_1'
        assert triggered_alerts[0]['type'] == 'price_threshold'

    @pytest.mark.asyncio
    async def test_check_alerts_change_percent(self, alert_manager, sample_alerts):
        """価格変動率アラートチェックテスト"""
        current_data = {
            'symbol': '7203',
            'price': 140.0,
            'change': -8.0,
            'change_percent': -6.5
        }

        triggered_alerts = await alert_manager.check_alerts(
            "7203", current_data, sample_alerts
        )

        # 価格変動率アラートがトリガーされるはず
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0]['alert_id'] == 'alert_2'
        assert triggered_alerts[0]['type'] == 'price_change_percent'

    @pytest.mark.asyncio
    async def test_check_alerts_no_trigger(self, alert_manager, sample_alerts):
        """アラートトリガーなしのテスト"""
        current_data = {
            'symbol': '7203',
            'price': 145.0,
            'change': 2.0,
            'change_percent': 1.4
        }

        triggered_alerts = await alert_manager.check_alerts(
            "7203", current_data, sample_alerts
        )

        # アラートはトリガーされないはず
        assert len(triggered_alerts) == 0

    @pytest.mark.asyncio
    async def test_check_alert_condition_price_threshold(self, alert_manager):
        """価格閾値条件チェックテスト"""
        alert = AlertConfig(
            alert_id="test",
            symbol="7203",
            alert_type="price_threshold",
            condition=">",
            threshold=150.0
        )

        # 条件を満たすデータ
        data_above = {'price': 155.0}
        result = await alert_manager._check_alert_condition(alert, data_above)
        assert result == True

        # 条件を満たさないデータ
        data_below = {'price': 145.0}
        result = await alert_manager._check_alert_condition(alert, data_below)
        assert result == False

    def test_generate_alert_message(self, alert_manager):
        """アラートメッセージ生成テスト"""
        alert = AlertConfig(
            alert_id="test",
            symbol="7203",
            alert_type="price_threshold",
            condition=">",
            threshold=150.0
        )

        data = {'price': 155.50}
        message = alert_manager._generate_alert_message(alert, data)

        assert "7203" in message
        assert "150" in message
        assert "155.50" in message

    def test_get_alert_severity(self, alert_manager):
        """アラート重要度判定テスト"""
        alert = AlertConfig(
            alert_id="test",
            symbol="7203",
            alert_type="price_change_percent",
            condition=">",
            threshold=5.0
        )

        # 大きな変動
        data_critical = {'change_percent': 12.0}
        severity = alert_manager._get_alert_severity(alert, data_critical)
        assert severity == 'critical'

        # 中程度の変動
        data_warning = {'change_percent': 7.0}
        severity = alert_manager._get_alert_severity(alert, data_warning)
        assert severity == 'warning'

        # 小さな変動
        data_info = {'change_percent': 3.0}
        severity = alert_manager._get_alert_severity(alert, data_info)
        assert severity == 'info'


@pytest.mark.skipif(not WEB_TESTS_AVAILABLE, reason="Web dependencies not available")
class TestEnhancedWebDashboard:
    """EnhancedWebDashboardのテストクラス"""

    @pytest.fixture
    def temp_config_file(self):
        """一時設定ファイル"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'theme': 'dark',
                'update_frequency': 'high',
                'auto_refresh': True,
                'default_symbols': ['7203', '8306'],
                'alerts_enabled': True
            }
            yaml.dump(config, f)
            yield Path(f.name)
            Path(f.name).unlink()

    @patch('enhanced_web_dashboard.WEB_AVAILABLE', True)
    def test_dashboard_initialization(self, temp_config_file):
        """ダッシュボード初期化テスト"""
        with patch('enhanced_web_dashboard.Flask'), \
             patch('enhanced_web_dashboard.SocketIO'):

            dashboard = EnhancedWebDashboard(
                config_path=temp_config_file,
                port=8081
            )

            assert dashboard.port == 8081
            assert dashboard.config.theme.value == 'dark'
            assert dashboard.config.update_frequency.value == 'high'

    def test_configuration_loading(self, temp_config_file):
        """設定読み込みテスト"""
        with patch('enhanced_web_dashboard.Flask'), \
             patch('enhanced_web_dashboard.SocketIO'):

            dashboard = EnhancedWebDashboard(config_path=temp_config_file)
            config = dashboard._load_configuration(temp_config_file)

            assert config.theme.value == 'dark'
            assert config.update_frequency.value == 'high'
            assert config.auto_refresh == True

    def test_create_enhanced_web_dashboard_factory(self, temp_config_file):
        """ファクトリー関数テスト"""
        with patch('enhanced_web_dashboard.Flask'), \
             patch('enhanced_web_dashboard.SocketIO'):

            dashboard = create_enhanced_web_dashboard(
                config_path=str(temp_config_file),
                port=8082
            )

            assert isinstance(dashboard, EnhancedWebDashboard)
            assert dashboard.port == 8082


# インテグレーションテスト
@pytest.mark.skipif(not WEB_TESTS_AVAILABLE, reason="Web dependencies not available")
class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    async def test_realtime_data_flow(self):
        """リアルタイムデータフローテスト"""
        config = DashboardConfig(update_frequency=UpdateFrequency.HIGH)
        manager = RealTimeDataManager(config)

        # モックSocketIO
        mock_socketio = MagicMock()
        mock_socketio.emit = MagicMock()

        # モックデータプロバイダー
        mock_data = pd.DataFrame({
            'Close': [100, 102],
            'Volume': [1000, 1100]
        })

        mock_provider = AsyncMock()
        mock_provider.get_stock_data.return_value = mock_data
        manager.data_provider = mock_provider

        # データ更新テスト
        symbol = "7203"
        await manager.subscribe_symbol(symbol, mock_socketio)
        await manager._update_symbol_data(symbol, mock_socketio)

        # 期待される呼び出しの確認
        assert mock_socketio.emit.call_count >= 1
        assert symbol in manager.active_subscriptions

    def test_alert_visualization_integration(self):
        """アラート・ビジュアライゼーション統合テスト"""
        config = DashboardConfig(alerts_enabled=True)
        alert_manager = AlertManager(config)
        visualization = AdvancedVisualization(config)

        # アラート設定
        alert = AlertConfig(
            alert_id="integration_test",
            symbol="7203",
            alert_type="price_threshold",
            condition=">",
            threshold=150.0
        )

        # サンプルデータでチャート作成
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [102] * 10,
            'Volume': [1000] * 10
        }, index=dates)

        chart_result = visualization.create_enhanced_candlestick_chart(
            data, "7203"
        )

        assert chart_result['success'] == True
        assert alert.symbol == "7203"


if __name__ == "__main__":
    # テスト実行
    if WEB_TESTS_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("Web dashboard tests require Flask, SocketIO, and Plotly")
        print("Install with: pip install flask flask-socketio plotly")