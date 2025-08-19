#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor テストスイート
Issue #857対応：pytest統合されたテストフレームワーク

MLモデル性能監視システムの包括的テスト
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sqlite3
import json
import yaml

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.day_trade.monitoring.model_performance_monitor import (
    ModelPerformanceMonitor,
    PerformanceMetrics,
    PerformanceAlert,
    RetrainingTrigger,
    PerformanceStatus,
    AlertLevel
)


@pytest.fixture
def temp_dir():
    """一時ディレクトリフィクスチャ"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """テスト用設定ファイル"""
    config_data = {
        'system': {
            'name': 'テスト用モデル性能監視システム',
            'version': '1.0.0',
            'enabled': True,
            'debug_mode': True
        },
        'performance_thresholds': {
            'accuracy': {
                'minimum_threshold': 0.75,
                'warning_threshold': 0.80,
                'target_threshold': 0.90,
                'critical_threshold': 0.70
            },
            'confidence': {
                'minimum_threshold': 0.65,
                'warning_threshold': 0.75,
                'target_threshold': 0.85
            }
        },
        'monitoring_symbols': {
            'primary_symbols': ['7203', '8306', '9984'],
            'secondary_symbols': ['4751', '2914'],
            'dynamic_selection': {
                'enabled': False
            }
        },
        'monitoring': {
            'intervals': {
                'real_time_seconds': 1,  # テスト用に短縮
                'periodic_minutes': 1
            }
        },
        'retraining': {
            'triggers': {
                'accuracy_degradation': {
                    'enabled': True,
                    'threshold_drop': 0.05,
                    'consecutive_failures': 2  # テスト用に短縮
                }
            },
            'strategy': {
                'granularity': 'selective',
                'batch_size': 2,
                'max_concurrent': 1
            }
        },
        'models': {
            'supported_models': ['RandomForestClassifier', 'XGBClassifier']
        },
        'database': {
            'connection': {
                'type': 'sqlite',
                'path': f'{temp_dir}/test_performance.db'
            }
        }
    }

    config_path = Path(temp_dir) / "test_performance_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    return config_path


@pytest.fixture
async def monitor(test_config):
    """モニターインスタンス"""
    monitor = ModelPerformanceMonitor(test_config)
    yield monitor
    # クリーンアップ
    if monitor.monitoring_active:
        await monitor.stop_monitoring()


class TestModelPerformanceMonitor:
    """ModelPerformanceMonitorテストクラス"""

    def test_initialization(self, monitor):
        """初期化テスト"""
        assert monitor is not None
        assert monitor.config is not None
        assert monitor.db_path.exists()
        assert not monitor.monitoring_active
        assert monitor.last_check_time is None

    def test_configuration_loading(self, monitor):
        """設定読み込みテスト"""
        config = monitor.config

        # 基本設定確認
        assert config['system']['name'] == 'テスト用モデル性能監視システム'
        assert config['performance_thresholds']['accuracy']['minimum_threshold'] == 0.75

        # 監視銘柄確認
        symbols = monitor.get_monitoring_symbols()
        assert '7203' in symbols
        assert '8306' in symbols
        assert len(symbols) >= 3

    def test_database_initialization(self, monitor):
        """データベース初期化テスト"""
        # テーブル存在確認
        with sqlite3.connect(monitor.db_path) as conn:
            cursor = conn.cursor()

            # テーブル一覧取得
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            assert 'model_performance_history' in tables
            assert 'performance_alerts' in tables
            assert 'retraining_history' in tables

    @pytest.mark.asyncio
    async def test_performance_metrics_generation(self, monitor):
        """性能指標生成テスト"""
        # 模擬性能データ生成
        metrics = monitor._generate_mock_performance('7203', 'RandomForestClassifier')

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.symbol == '7203'
        assert metrics.model_type == 'RandomForestClassifier'
        assert 0.0 <= metrics.accuracy <= 1.0
        assert isinstance(metrics.status, PerformanceStatus)

    def test_performance_status_determination(self, monitor):
        """性能ステータス判定テスト"""
        # 優秀レベル
        metrics = PerformanceMetrics(
            symbol='7203', model_type='test', timestamp=datetime.now(), accuracy=0.95
        )
        status = monitor._determine_performance_status(metrics)
        assert status == PerformanceStatus.EXCELLENT

        # 警告レベル
        metrics.accuracy = 0.82
        status = monitor._determine_performance_status(metrics)
        assert status == PerformanceStatus.GOOD

        # 注意レベル
        metrics.accuracy = 0.77
        status = monitor._determine_performance_status(metrics)
        assert status == PerformanceStatus.WARNING

        # 緊急レベル
        metrics.accuracy = 0.65
        status = monitor._determine_performance_status(metrics)
        assert status == PerformanceStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_performance_metrics_saving(self, monitor):
        """性能指標保存テスト"""
        metrics = PerformanceMetrics(
            symbol='7203',
            model_type='TestModel',
            timestamp=datetime.now(),
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            confidence_avg=0.80,
            sample_size=100,
            status=PerformanceStatus.GOOD
        )

        await monitor._save_performance_metrics(metrics)

        # データベース確認
        with sqlite3.connect(monitor.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM model_performance_history WHERE symbol = ? AND model_type = ?",
                (metrics.symbol, metrics.model_type)
            )
            row = cursor.fetchone()

            assert row is not None
            assert row[1] == '7203'  # symbol
            assert row[2] == 'TestModel'  # model_type
            assert abs(row[4] - 0.85) < 0.001  # accuracy

    @pytest.mark.asyncio
    async def test_threshold_checking(self, monitor):
        """閾値チェックテスト"""
        # 緊急閾値を下回る性能
        metrics = PerformanceMetrics(
            symbol='7203',
            model_type='TestModel',
            timestamp=datetime.now(),
            accuracy=0.65,  # 緊急閾値(0.70)を下回る
            confidence_avg=0.60  # 最低閾値(0.65)を下回る
        )

        alerts = monitor._check_thresholds(metrics)

        assert len(alerts) == 2  # accuracy + confidence

        # 精度アラート確認
        accuracy_alert = next((a for a in alerts if a.metric_name == 'accuracy'), None)
        assert accuracy_alert is not None
        assert accuracy_alert.alert_level == AlertLevel.CRITICAL
        assert accuracy_alert.current_value == 0.65

        # 信頼度アラート確認
        confidence_alert = next((a for a in alerts if a.metric_name == 'confidence'), None)
        assert confidence_alert is not None
        assert confidence_alert.alert_level == AlertLevel.WARNING

    @pytest.mark.asyncio
    async def test_performance_history_checking(self, monitor):
        """性能履歴チェックテスト"""
        key = '7203_TestModel'

        # 履歴データ作成
        base_accuracy = 0.85
        for i in range(3):
            metrics = PerformanceMetrics(
                symbol='7203',
                model_type='TestModel',
                timestamp=datetime.now() - timedelta(hours=i),
                accuracy=base_accuracy - (i * 0.01)  # 徐々に悪化
            )
            monitor.performance_history[key].append(metrics)

        # 大幅に悪化した現在の性能
        current_metrics = PerformanceMetrics(
            symbol='7203',
            model_type='TestModel',
            timestamp=datetime.now(),
            accuracy=0.75  # 0.10ポイント低下
        )

        alerts = monitor._check_performance_history(key, current_metrics)

        assert len(alerts) >= 1
        degradation_alert = alerts[0]
        assert degradation_alert.alert_level == AlertLevel.ERROR
        assert degradation_alert.metric_name == 'accuracy_degradation'

    @pytest.mark.asyncio
    async def test_alert_saving(self, monitor):
        """アラート保存テスト"""
        alert = PerformanceAlert(
            id='test_alert_001',
            timestamp=datetime.now(),
            symbol='7203',
            model_type='TestModel',
            alert_level=AlertLevel.WARNING,
            metric_name='accuracy',
            current_value=0.72,
            threshold_value=0.75,
            message='テストアラート'
        )

        await monitor._save_alert(alert)

        # データベース確認
        with sqlite3.connect(monitor.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM performance_alerts WHERE id = ?", (alert.id,))
            row = cursor.fetchone()

            assert row is not None
            assert row[0] == 'test_alert_001'  # id
            assert row[2] == '7203'  # symbol
            assert row[4] == 'warning'  # alert_level

        # アクティブアラート確認
        assert alert.id in monitor.active_alerts

    @pytest.mark.asyncio
    async def test_retraining_trigger_creation(self, monitor):
        """再学習トリガー作成テスト"""
        # 重要なアラート作成
        critical_alerts = [
            PerformanceAlert(
                id='critical_001',
                timestamp=datetime.now(),
                symbol='7203',
                model_type='TestModel',
                alert_level=AlertLevel.CRITICAL,
                metric_name='accuracy',
                current_value=0.65,
                threshold_value=0.70,
                message='緊急：精度が大幅低下'
            ),
            PerformanceAlert(
                id='error_001',
                timestamp=datetime.now(),
                symbol='8306',
                model_type='TestModel',
                alert_level=AlertLevel.ERROR,
                metric_name='accuracy_degradation',
                current_value=0.70,
                threshold_value=0.75,
                message='性能劣化検出'
            )
        ]

        trigger = await monitor.trigger_retraining(critical_alerts)

        assert trigger is not None
        assert isinstance(trigger, RetrainingTrigger)
        assert len(trigger.affected_symbols) == 2
        assert '7203' in trigger.affected_symbols
        assert '8306' in trigger.affected_symbols
        assert trigger.severity == AlertLevel.CRITICAL

        # データベース確認
        with sqlite3.connect(monitor.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM retraining_history WHERE id = ?", (trigger.trigger_id,))
            row = cursor.fetchone()

            assert row is not None
            assert row[2] == 'selective'  # trigger_type

    @pytest.mark.asyncio
    async def test_get_latest_model_performance(self, monitor):
        """最新モデル性能取得テスト"""
        performance_data = await monitor.get_latest_model_performance()

        assert isinstance(performance_data, dict)
        assert len(performance_data) > 0

        # いくつかの性能データを確認
        for key, metrics in performance_data.items():
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.symbol in monitor.get_monitoring_symbols()
            assert 0.0 <= metrics.accuracy <= 1.0
            assert isinstance(metrics.status, PerformanceStatus)

    @pytest.mark.asyncio
    async def test_performance_check_workflow(self, monitor):
        """性能チェックワークフローテスト"""
        # 性能チェック実行
        await monitor.run_performance_check()

        # 最終チェック時刻が更新されていることを確認
        assert monitor.last_check_time is not None
        assert (datetime.now() - monitor.last_check_time).total_seconds() < 10

        # 履歴にデータが蓄積されていることを確認
        assert len(monitor.performance_history) > 0

    @pytest.mark.asyncio
    async def test_monitoring_status(self, monitor):
        """監視ステータステスト"""
        status = monitor.get_monitoring_status()

        assert isinstance(status, dict)
        assert 'monitoring_active' in status
        assert 'monitoring_symbols_count' in status
        assert 'config_loaded' in status
        assert 'external_systems' in status

        assert status['monitoring_active'] == False
        assert status['monitoring_symbols_count'] > 0
        assert status['config_loaded'] == True

    @pytest.mark.asyncio
    async def test_short_term_monitoring(self, monitor):
        """短期監視テスト"""
        # 監視開始
        monitoring_task = asyncio.create_task(monitor.start_monitoring())

        # 少し待機
        await asyncio.sleep(2)

        # 監視停止
        await monitor.stop_monitoring()

        # タスク完了待機
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        # 監視が実行されたことを確認
        assert monitor.last_check_time is not None
        assert not monitor.monitoring_active

    @pytest.mark.asyncio
    async def test_error_handling(self, monitor):
        """エラーハンドリングテスト"""
        # 無効な銘柄での性能評価
        metrics = await monitor._evaluate_model_performance('INVALID', 'TestModel')

        # エラーが発生しても、None が返されるか模擬データが返される
        assert metrics is None or isinstance(metrics, PerformanceMetrics)

        # 空のアラートリストでの再学習トリガー
        trigger = await monitor.trigger_retraining([])
        assert trigger is None

    @pytest.mark.asyncio
    async def test_configuration_fallback(self, temp_dir):
        """設定フォールバックテスト"""
        # 存在しない設定ファイルパス
        non_existent_config = Path(temp_dir) / "non_existent.yaml"

        # デフォルト設定が使用されることを確認
        monitor = ModelPerformanceMonitor(non_existent_config)
        assert monitor.config is not None

        # デフォルト設定の内容確認
        assert 'performance_thresholds' in monitor.config
        assert 'monitoring_symbols' in monitor.config

    def test_mock_performance_generation(self, monitor):
        """模擬性能生成テスト"""
        # 複数回生成して値の範囲を確認
        accuracies = []
        for _ in range(10):
            metrics = monitor._generate_mock_performance('7203', 'TestModel')
            accuracies.append(metrics.accuracy)

        # 値が適切な範囲内にあることを確認
        assert all(0.6 <= acc <= 0.95 for acc in accuracies)

        # ある程度のバリエーションがあることを確認
        assert len(set(round(acc, 2) for acc in accuracies)) > 1


@pytest.mark.asyncio
async def test_integration_with_external_systems():
    """外部システム統合テスト"""
    # MLPredictionModelsが利用できない環境でのテスト
    with patch('model_performance_monitor.ML_MODELS_AVAILABLE', False):
        monitor = ModelPerformanceMonitor()
        assert monitor.ml_models is None

        # 性能チェックが正常に動作することを確認
        await monitor.run_performance_check()


def test_windows_console_integration():
    """Windows環境統合テスト"""
    # Windows文字化け対策が正常に動作することを確認
    try:
        from model_performance_monitor import ModelPerformanceMonitor
        monitor = ModelPerformanceMonitor()

        # 日本語文字列のログ出力テスト
        monitor.logger.info("テスト: 日本語文字列の正常出力")

        assert True  # エラーが発生しなければ成功
    except Exception as e:
        pytest.fail(f"Windows文字化け対策でエラー: {e}")


if __name__ == "__main__":
    # pytest実行
    import os
    os.system("pytest " + __file__ + " -v")