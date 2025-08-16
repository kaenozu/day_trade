#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Model Performance Monitor (Improved Version)
改善版ModelPerformanceMonitorのテストケース

pytestフレームワークを使用した構造化テスト
"""

import pytest
import asyncio
import sqlite3
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model_performance_monitor import (
    ModelPerformanceMonitor,
    PerformanceMetrics,
    RetrainingResult,
    create_performance_monitor
)

class TestModelPerformanceMonitor:
    """ModelPerformanceMonitorのテストクラス"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリの作成"""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def config_file(self, temp_dir):
        """テスト用設定ファイルの作成"""
        config_path = temp_dir / "test_config.yaml"
        config = {
            'performance_thresholds': {
                'accuracy': 85.0,
                'prediction_accuracy': 80.0
            },
            'monitoring': {
                'default_symbols': ["TEST1", "TEST2"],
                'validation_hours': 24
            },
            'retraining': {
                'granular_mode': True,
                'global_threshold': 75.0,
                'cooldown_hours': 1
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def monitor(self, config_file, temp_dir):
        """テスト用MonitorPerformanceMonitorインスタンス"""
        db_path = temp_dir / "test.db"

        with patch('model_performance_monitor.PredictionAccuracyValidator'):
            monitor = ModelPerformanceMonitor(
                config_path=config_file,
                upgrade_db_path=db_path
            )
            return monitor

    def test_init_and_config_loading(self, monitor):
        """初期化と設定読み込みのテスト"""
        assert monitor.thresholds['accuracy'] == 85.0
        assert monitor.thresholds['prediction_accuracy'] == 80.0
        assert monitor.monitoring_symbols == ["TEST1", "TEST2"]
        assert monitor.validation_hours == 24

    def test_database_initialization(self, monitor):
        """データベース初期化のテスト"""
        # データベースが正しく作成されているかチェック
        with sqlite3.connect(monitor.upgrade_db_path) as conn:
            cursor = conn.cursor()

            # テーブル存在確認
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='performance_thresholds'
            """)
            assert cursor.fetchone() is not None

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='performance_history'
            """)
            assert cursor.fetchone() is not None

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='retraining_history'
            """)
            assert cursor.fetchone() is not None

    def test_get_monitoring_symbols(self, monitor):
        """監視対象銘柄取得のテスト"""
        symbols = monitor.get_monitoring_symbols()
        assert symbols == ["TEST1", "TEST2"]

    @pytest.mark.asyncio
    async def test_get_latest_model_performance_no_validator(self, monitor):
        """バリデータがない場合の性能取得テスト"""
        monitor.accuracy_validator = None
        performance = await monitor.get_latest_model_performance()
        assert performance.accuracy == 0.0

    @pytest.mark.asyncio
    async def test_get_latest_model_performance_with_mock_validator(self, monitor):
        """モックバリデータを使用した性能取得テスト"""
        # モックバリデータの設定
        mock_validator = Mock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 92.5

        mock_validator.validate_current_system_accuracy = AsyncMock(
            return_value=mock_metrics
        )
        monitor.accuracy_validator = mock_validator

        performance = await monitor.get_latest_model_performance(["TEST1"])

        assert performance.accuracy == 92.5
        assert isinstance(performance.timestamp, datetime)

    def test_determine_retraining_scope(self, monitor):
        """再学習範囲決定のテスト"""
        # グローバル閾値を下回る場合
        performance = PerformanceMetrics(accuracy=70.0)
        scope = monitor._determine_retraining_scope(performance)
        assert scope == "global"

        # 部分的再学習範囲の場合
        performance = PerformanceMetrics(accuracy=78.0)
        scope = monitor._determine_retraining_scope(performance)
        assert scope == "partial"

        # 標準的再学習範囲の場合
        performance = PerformanceMetrics(accuracy=82.0)
        scope = monitor._determine_retraining_scope(performance)
        assert scope == "standard"

        # 再学習不要の場合
        performance = PerformanceMetrics(accuracy=90.0)
        scope = monitor._determine_retraining_scope(performance)
        assert scope == "none"

    def test_check_cooldown_period(self, monitor):
        """冷却期間チェックのテスト"""
        # 冷却期間なしの場合
        assert monitor._check_cooldown_period("global") == True

        # 最近実行済みの場合
        monitor.last_retraining["global"] = datetime.now()
        assert monitor._check_cooldown_period("global") == False

        # 冷却期間経過後の場合
        monitor.last_retraining["global"] = datetime.now() - timedelta(hours=2)
        assert monitor._check_cooldown_period("global") == True

    @pytest.mark.asyncio
    async def test_record_performance_history(self, monitor):
        """性能履歴記録のテスト"""
        symbols = ["TEST1", "TEST2"]
        performance = PerformanceMetrics(accuracy=88.5)

        await monitor._record_performance_history(symbols, performance)

        # データベースから確認
        with sqlite3.connect(monitor.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM performance_history
                WHERE metric_name = 'accuracy'
            """)
            count = cursor.fetchone()[0]
            assert count >= 3  # 各銘柄 + 全体

    @pytest.mark.asyncio
    async def test_record_retraining_result(self, monitor):
        """再学習結果記録のテスト"""
        result = RetrainingResult(
            triggered=True,
            scope="test",
            affected_symbols=["TEST1"],
            improvement=5.2,
            duration=120.0
        )

        await monitor._record_retraining_result(result)

        # データベースから確認
        with sqlite3.connect(monitor.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT scope, improvement, duration, status
                FROM retraining_history
                ORDER BY trigger_time DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            assert row[0] == "test"
            assert row[1] == 5.2
            assert row[2] == 120.0
            assert row[3] == "success"

    @pytest.mark.asyncio
    async def test_execute_retraining_no_system(self, monitor):
        """再学習システムがない場合のテスト"""
        # ml_upgrade_systemがNoneの場合
        with patch('model_performance_monitor.ml_upgrade_system', None):
            result = await monitor._execute_retraining(
                "global", ["TEST1"], PerformanceMetrics(accuracy=70.0)
            )
            assert result.triggered == False
            assert result.error is not None

    def test_get_performance_summary(self, monitor):
        """性能サマリー取得のテスト"""
        summary = monitor.get_performance_summary()

        assert 'current_thresholds' in summary
        assert 'monitoring_symbols' in summary
        assert 'recent_performance' in summary
        assert 'recent_retraining' in summary
        assert 'config_path' in summary

    @pytest.mark.asyncio
    async def test_check_and_trigger_retraining_no_performance(self, monitor):
        """性能データなしでの再学習チェックテスト"""
        monitor.accuracy_validator = None

        result = await monitor.check_and_trigger_retraining()

        assert result.triggered == False
        assert result.scope == "none"

    def test_create_performance_monitor_function(self, config_file):
        """ユーティリティ関数のテスト"""
        with patch('model_performance_monitor.PredictionAccuracyValidator'):
            monitor = create_performance_monitor(str(config_file))
            assert isinstance(monitor, ModelPerformanceMonitor)

    @pytest.mark.asyncio
    async def test_symbol_specific_performance_analysis(self, monitor):
        """銘柄別性能分析のテスト"""
        # モックバリデータの設定
        mock_validator = Mock()

        def mock_validate(symbols, hours):
            # 銘柄によって異なる性能を返す
            if symbols == ["TEST1"]:
                mock_result = Mock()
                mock_result.overall_accuracy = 85.0
                return mock_result
            elif symbols == ["TEST2"]:
                mock_result = Mock()
                mock_result.overall_accuracy = 95.0
                return mock_result
            return Mock(overall_accuracy=90.0)

        mock_validator.validate_current_system_accuracy = AsyncMock(side_effect=mock_validate)
        monitor.accuracy_validator = mock_validator

        symbols = ["TEST1", "TEST2"]
        performances = await monitor._analyze_symbol_specific_performance(symbols)

        assert len(performances) == 2
        assert performances["TEST1"] == 85.0
        assert performances["TEST2"] == 95.0

    def test_identify_underperforming_symbols(self, monitor):
        """性能不足銘柄特定のテスト"""
        symbol_performances = {
            "GOOD1": 95.0,
            "BAD1": 75.0,
            "GOOD2": 92.0,
            "BAD2": 80.0
        }

        # 閾値85.0を設定
        monitor.thresholds["accuracy"] = 85.0

        underperforming = monitor._identify_underperforming_symbols(symbol_performances)

        assert len(underperforming) == 2
        assert "BAD1" in underperforming
        assert "BAD2" in underperforming
        assert "GOOD1" not in underperforming
        assert "GOOD2" not in underperforming

    @pytest.mark.asyncio
    async def test_get_monitoring_symbols_with_selector(self, monitor, temp_dir):
        """symbol_selector連携での監視銘柄取得テスト"""
        # symbol_selector連携設定を追加
        monitor.config["symbol_selector"] = {
            "enabled": True,
            "strategy": "test_strategy",
            "limit": 5,
            "config_path": str(temp_dir / "test_selector_config.yaml"),
            "db_path": str(temp_dir / "test_selector.db")
        }

        # モックsymbol_selectorの設定
        with patch('src.day_trade.data.symbol_selector.create_symbol_selector') as mock_create:
            mock_selector = Mock()
            mock_selector.get_symbols_by_strategy.return_value = ["DYN1", "DYN2", "DYN3"]
            mock_create.return_value = mock_selector

            symbols = monitor.get_monitoring_symbols()

            # 基本銘柄と動的銘柄が結合されているか確認
            assert "TEST1" in symbols  # 基本銘柄
            assert "TEST2" in symbols  # 基本銘柄
            assert "DYN1" in symbols   # 動的銘柄
            assert "DYN2" in symbols   # 動的銘柄
            assert "DYN3" in symbols   # 動的銘柄

    def test_default_config_creation(self, temp_dir):
        """デフォルト設定作成のテスト"""
        config_path = temp_dir / "non_existent_config.yaml"

        with patch('model_performance_monitor.PredictionAccuracyValidator'):
            monitor = ModelPerformanceMonitor(config_path=config_path)

        # デフォルト設定が作成されているか確認
        assert config_path.exists()

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            assert 'performance_thresholds' in config
            assert 'monitoring' in config

class TestPerformanceMetrics:
    """PerformanceMetricsデータクラスのテスト"""

    def test_performance_metrics_creation(self):
        """PerformanceMetrics作成テスト"""
        metrics = PerformanceMetrics(accuracy=85.5)
        assert metrics.accuracy == 85.5
        assert metrics.prediction_accuracy == 0.0
        assert isinstance(metrics.timestamp, datetime)

    def test_performance_metrics_with_timestamp(self):
        """タイムスタンプ指定でのPerformanceMetrics作成テスト"""
        test_time = datetime(2025, 1, 1, 12, 0, 0)
        metrics = PerformanceMetrics(accuracy=90.0, timestamp=test_time)
        assert metrics.timestamp == test_time

class TestRetrainingResult:
    """RetrainingResultデータクラスのテスト"""

    def test_retraining_result_creation(self):
        """RetrainingResult作成テスト"""
        result = RetrainingResult(
            triggered=True,
            scope="global",
            affected_symbols=["7203", "8306"],
            improvement=3.2,
            duration=180.5
        )

        assert result.triggered == True
        assert result.scope == "global"
        assert result.affected_symbols == ["7203", "8306"]
        assert result.improvement == 3.2
        assert result.duration == 180.5
        assert result.error is None

# インテグレーションテスト
class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self, temp_dir):
        """全体ワークフローのシミュレーションテスト"""
        # 設定ファイル作成
        config_path = temp_dir / "integration_config.yaml"
        config = {
            'performance_thresholds': {'accuracy': 90.0},
            'monitoring': {'default_symbols': ["INT1", "INT2"]},
            'retraining': {'cooldown_hours': 0.01}  # 短い冷却時間
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        # モニター作成
        with patch('model_performance_monitor.PredictionAccuracyValidator'):
            monitor = ModelPerformanceMonitor(
                config_path=config_path,
                upgrade_db_path=temp_dir / "integration.db"
            )

        # モックバリデータ設定
        mock_validator = Mock()
        mock_metrics = Mock()
        mock_metrics.overall_accuracy = 85.0  # 閾値以下

        mock_validator.validate_current_system_accuracy = AsyncMock(
            return_value=mock_metrics
        )
        monitor.accuracy_validator = mock_validator

        # 再学習チェック実行
        with patch('model_performance_monitor.ml_upgrade_system') as mock_system:
            mock_report = Mock()
            mock_report.overall_improvement = 7.5
            mock_system.run_complete_system_upgrade = AsyncMock(
                return_value=mock_report
            )
            mock_system.integrate_best_models = AsyncMock(
                return_value="integration_success"
            )

            result = await monitor.check_and_trigger_retraining()

        # 結果確認
        assert result.triggered == True
        assert result.improvement == 7.5
        assert result.scope == "standard"

        # サマリー確認
        summary = monitor.get_performance_summary()
        assert len(summary['recent_retraining']) >= 1

if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])