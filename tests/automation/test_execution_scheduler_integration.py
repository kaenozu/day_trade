#!/usr/bin/env python3
"""
Issue #755 Phase 4: ExecutionScheduler統合テスト

ExecutionSchedulerとメインシステムの統合検証
- SmartSymbolSelector統合テスト
- EnsembleSystem統合テスト
- DataFetcher統合テスト
- エンドツーエンド自動化ワークフローテスト
"""

import unittest
import pytest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.automation.execution_scheduler import (
        ExecutionScheduler,
        ScheduledTask,
        ExecutionResult,
        ScheduleType,
        ExecutionStatus,
        smart_stock_analysis_task,
        create_default_automation_tasks
    )
    from src.day_trade.automation.smart_symbol_selector import (
        SmartSymbolSelector,
        SymbolMetrics,
        SelectionCriteria
    )
    from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig
    from src.day_trade.data_fetcher import DataFetcher

except ImportError as e:
    print(f"統合テスト用インポートエラー: {e}")
    sys.exit(1)


class TestExecutionSchedulerSmartSelectorIntegration(unittest.TestCase):
    """ExecutionScheduler - SmartSymbolSelector統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()
        self.smart_selector = SmartSymbolSelector()

        # モックデータ準備
        self.mock_market_data = self._create_mock_market_data()

    def _create_mock_market_data(self) -> pd.DataFrame:
        """モック市場データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        symbols = ['7203.T', '6758.T', '9984.T']

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(2000000, 4000000)
                })

        return pd.DataFrame(data)

    @patch('src.day_trade.automation.smart_symbol_selector.get_smart_selected_symbols')
    def test_scheduled_smart_analysis_task(self, mock_get_symbols):
        """スケジュール化されたスマート分析タスクテスト"""
        # SmartSymbolSelectorのモック設定
        mock_get_symbols.return_value = ['7203.T', '6758.T', '9984.T', '4519.T']

        # スマート分析タスク作成
        analysis_task = ScheduledTask(
            task_id="smart_analysis_integration",
            name="Smart Analysis Integration Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=lambda: asyncio.run(smart_stock_analysis_task(target_count=4))
        )

        # タスク追加・実行
        self.scheduler.add_task(analysis_task)
        self.scheduler._execute_task(analysis_task)

        # 実行結果確認
        self.assertEqual(analysis_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(analysis_task.success_count, 1)

        # 実行履歴確認
        self.assertEqual(len(self.scheduler.execution_history), 1)
        history = self.scheduler.execution_history[0]
        self.assertEqual(history.status, ExecutionStatus.SUCCESS)

        # 結果データ確認
        result_data = history.result_data
        self.assertIsInstance(result_data, dict)
        self.assertIn('selected_symbols', result_data)
        self.assertIn('predictions', result_data)
        self.assertEqual(len(result_data['selected_symbols']), 4)

    @patch('yfinance.Ticker')
    def test_real_time_symbol_selection_scheduling(self, mock_ticker):
        """リアルタイム銘柄選択スケジューリングテスト"""
        # yfinanceモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 5000000000000}
        mock_ticker_instance.history.return_value = self._create_ticker_data()
        mock_ticker.return_value = mock_ticker_instance

        selection_results = []

        async def symbol_selection_task():
            """銘柄選択タスク"""
            try:
                selector = SmartSymbolSelector()
                # テスト用に小さなシンボルプールを使用
                selector.symbol_pool = {
                    '7203.T': 'トヨタ自動車',
                    '6758.T': 'ソニーグループ',
                    '9984.T': 'ソフトバンクグループ'
                }

                selected = await selector.select_optimal_symbols(
                    SelectionCriteria(target_symbols=2)
                )

                selection_results.append({
                    'timestamp': datetime.now(),
                    'symbols': selected,
                    'count': len(selected)
                })

                return selected

            except Exception as e:
                selection_results.append({
                    'timestamp': datetime.now(),
                    'error': str(e)
                })
                raise

        # リアルタイム選択タスク作成
        realtime_task = ScheduledTask(
            task_id="realtime_selection",
            name="Real-time Symbol Selection",
            schedule_type=ScheduleType.CONTINUOUS,
            target_function=lambda: asyncio.run(symbol_selection_task()),
            interval_minutes=1
        )

        # タスク実行
        self.scheduler.add_task(realtime_task)
        self.scheduler._execute_task(realtime_task)

        # 結果確認
        self.assertEqual(realtime_task.status, ExecutionStatus.SUCCESS)
        self.assertGreater(len(selection_results), 0)

        # 選択結果詳細確認
        if selection_results and 'symbols' in selection_results[0]:
            result = selection_results[0]
            self.assertIsInstance(result['symbols'], list)
            self.assertLessEqual(result['count'], 2)

    def _create_ticker_data(self) -> pd.DataFrame:
        """ティッカーデータ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(2000000, 4000000, 60)
        }

        return pd.DataFrame(data, index=dates)

    @patch('src.day_trade.automation.smart_symbol_selector.get_smart_selected_symbols')
    def test_market_hours_conditional_execution(self, mock_get_symbols):
        """市場時間条件実行テスト"""
        mock_get_symbols.return_value = ['7203.T', '6758.T']

        execution_log = []

        def market_hours_task():
            """市場時間タスク"""
            current_time = datetime.now()
            execution_log.append({
                'execution_time': current_time,
                'hour': current_time.hour,
                'minute': current_time.minute
            })
            return f"executed_at_{current_time.hour}:{current_time.minute:02d}"

        # 市場時間タスク作成
        market_task = ScheduledTask(
            task_id="market_hours_test",
            name="Market Hours Conditional Task",
            schedule_type=ScheduleType.MARKET_HOURS,
            target_function=market_hours_task
        )

        # 次回実行時刻を手動設定（テスト用）
        market_task.next_execution = datetime.now() - timedelta(seconds=1)

        # タスク実行
        self.scheduler.add_task(market_task)
        self.scheduler._execute_task(market_task)

        # 実行確認
        self.assertEqual(market_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(execution_log), 1)

        # 次回実行時刻が設定されていることを確認
        self.assertIsNotNone(market_task.next_execution)


class TestExecutionSchedulerEnsembleIntegration(unittest.TestCase):
    """ExecutionScheduler - EnsembleSystem統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()

    def test_ensemble_system_initialization(self):
        """EnsembleSystem初期化統合テスト"""
        # EnsembleSystem統合確認
        self.assertIsNotNone(self.scheduler.ensemble_system)
        self.assertIsInstance(self.scheduler.ensemble_system, EnsembleSystem)

        # 設定確認
        config = self.scheduler.ensemble_system.config
        self.assertTrue(config.use_xgboost)
        self.assertTrue(config.use_catboost)
        self.assertTrue(config.use_random_forest)

        # パラメータ確認
        self.assertIn('n_estimators', config.xgboost_params)
        self.assertIn('iterations', config.catboost_params)
        self.assertIn('n_estimators', config.random_forest_params)

    def test_ensemble_prediction_task_integration(self):
        """アンサンブル予測タスク統合テスト"""
        prediction_results = []

        def ensemble_prediction_task(symbols: List[str]):
            """アンサンブル予測タスク"""
            # テスト用特徴量・ターゲット生成
            n_samples = 100
            n_features = 15

            X_train = np.random.randn(n_samples, n_features)
            y_train = np.random.randn(n_samples)
            X_test = np.random.randn(20, n_features)

            # EnsembleSystem予測
            ensemble = self.scheduler.ensemble_system
            ensemble.fit(X_train, y_train)
            predictions = ensemble.predict(X_test)

            # 結果記録
            result = {
                'symbols': symbols,
                'predictions': predictions.final_predictions.tolist(),
                'confidence': predictions.ensemble_confidence.tolist() if hasattr(predictions, 'ensemble_confidence') else None,
                'timestamp': datetime.now()
            }

            prediction_results.append(result)
            return result

        # アンサンブル予測タスク作成
        prediction_task = ScheduledTask(
            task_id="ensemble_prediction",
            name="Ensemble Prediction Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=lambda: ensemble_prediction_task(['7203.T', '6758.T', '9984.T'])
        )

        # タスク実行
        self.scheduler.add_task(prediction_task)
        self.scheduler._execute_task(prediction_task)

        # 実行結果確認
        self.assertEqual(prediction_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(prediction_results), 1)

        # 予測結果確認
        result = prediction_results[0]
        self.assertIn('symbols', result)
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), 20)
        self.assertTrue(all(isinstance(p, (int, float)) for p in result['predictions']))

    def test_ensemble_retraining_schedule(self):
        """アンサンブル再訓練スケジュールテスト"""
        retraining_log = []

        def retrain_ensemble_task():
            """アンサンブル再訓練タスク"""
            # 新しい訓練データ生成（実際の実装では市場データを使用）
            n_samples = 200
            n_features = 15

            X_new = np.random.randn(n_samples, n_features)
            y_new = np.random.randn(n_samples)

            # アンサンブル再訓練
            ensemble = self.scheduler.ensemble_system
            ensemble.fit(X_new, y_new)

            # 再訓練ログ記録
            retraining_log.append({
                'timestamp': datetime.now(),
                'samples': n_samples,
                'features': n_features,
                'status': 'completed'
            })

            return {'retrained': True, 'samples': n_samples}

        # 再訓練タスク作成（日次実行）
        retrain_task = ScheduledTask(
            task_id="ensemble_retrain",
            name="Ensemble Retraining Task",
            schedule_type=ScheduleType.DAILY,
            target_function=retrain_ensemble_task,
            schedule_time="20:00"  # 市場終了後
        )

        # タスク実行
        self.scheduler.add_task(retrain_task)
        self.scheduler._execute_task(retrain_task)

        # 再訓練結果確認
        self.assertEqual(retrain_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(retraining_log), 1)

        # 次回実行時刻確認
        self.assertIsNotNone(retrain_task.next_execution)
        self.assertEqual(retrain_task.next_execution.hour, 20)
        self.assertEqual(retrain_task.next_execution.minute, 0)


class TestExecutionSchedulerDataIntegration(unittest.TestCase):
    """ExecutionScheduler - DataFetcher統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()

    @patch('src.day_trade.data_fetcher.DataFetcher.fetch_data')
    def test_data_fetcher_scheduled_task(self, mock_fetch_data):
        """DataFetcher スケジュール化タスクテスト"""
        # DataFetcherのモック設定
        mock_market_data = self._create_mock_market_data()
        mock_fetch_data.return_value = mock_market_data

        data_collection_results = []

        def data_collection_task(symbols: List[str]):
            """データ収集タスク"""
            fetcher = DataFetcher()
            market_data = fetcher.fetch_data(symbols)

            # データ処理・保存（模擬）
            processed_data = {
                'symbols': symbols,
                'data_points': len(market_data) if market_data is not None else 0,
                'timestamp': datetime.now(),
                'status': 'success' if market_data is not None else 'failed'
            }

            data_collection_results.append(processed_data)
            return processed_data

        # データ収集タスク作成
        data_task = ScheduledTask(
            task_id="data_collection",
            name="Market Data Collection",
            schedule_type=ScheduleType.HOURLY,
            target_function=lambda: data_collection_task(['7203.T', '6758.T'])
        )

        # タスク実行
        self.scheduler.add_task(data_task)
        self.scheduler._execute_task(data_task)

        # 実行結果確認
        self.assertEqual(data_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(data_collection_results), 1)

        # データ収集結果確認
        result = data_collection_results[0]
        self.assertEqual(result['status'], 'success')
        self.assertGreater(result['data_points'], 0)
        self.assertEqual(len(result['symbols']), 2)

    def _create_mock_market_data(self) -> pd.DataFrame:
        """モック市場データ作成"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        symbols = ['7203.T', '6758.T']

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(1000000, 5000000)
                })

        return pd.DataFrame(data)

    def test_real_time_data_monitoring_task(self):
        """リアルタイムデータ監視タスクテスト"""
        monitoring_log = []

        def data_monitoring_task():
            """データ監視タスク"""
            # 模擬的なデータ監視
            current_time = datetime.now()

            # 市場時間チェック
            is_market_hours = (9 <= current_time.hour < 15 and
                             not (11 <= current_time.hour < 13))  # 昼休み除外

            monitoring_result = {
                'timestamp': current_time,
                'market_open': is_market_hours,
                'monitoring_active': True,
                'data_quality': 'good'
            }

            monitoring_log.append(monitoring_result)
            return monitoring_result

        # データ監視タスク作成
        monitoring_task = ScheduledTask(
            task_id="data_monitoring",
            name="Real-time Data Monitoring",
            schedule_type=ScheduleType.CONTINUOUS,
            target_function=data_monitoring_task,
            interval_minutes=5
        )

        # タスク実行
        self.scheduler.add_task(monitoring_task)
        self.scheduler._execute_task(monitoring_task)

        # 監視結果確認
        self.assertEqual(monitoring_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(monitoring_log), 1)

        # 監視データ確認
        result = monitoring_log[0]
        self.assertIn('timestamp', result)
        self.assertIn('market_open', result)
        self.assertTrue(result['monitoring_active'])


class TestExecutionSchedulerEndToEnd(unittest.TestCase):
    """エンドツーエンド自動化ワークフローテスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()
        self.workflow_results = []

    @patch('src.day_trade.automation.smart_symbol_selector.get_smart_selected_symbols')
    @patch('src.day_trade.data_fetcher.DataFetcher.fetch_data')
    def test_complete_automation_workflow(self, mock_fetch_data, mock_get_symbols):
        """完全自動化ワークフローテスト"""
        # モック設定
        mock_get_symbols.return_value = ['7203.T', '6758.T', '9984.T']
        mock_fetch_data.return_value = self._create_workflow_data()

        def complete_workflow_task():
            """完全ワークフロータスク"""
            workflow_result = {
                'start_time': datetime.now(),
                'steps': []
            }

            try:
                # Step 1: 銘柄選択
                workflow_result['steps'].append('symbol_selection_started')
                selected_symbols = asyncio.run(
                    self._async_symbol_selection()
                )
                workflow_result['selected_symbols'] = selected_symbols
                workflow_result['steps'].append('symbol_selection_completed')

                # Step 2: データ収集
                workflow_result['steps'].append('data_collection_started')
                market_data = self._collect_market_data(selected_symbols)
                workflow_result['data_points'] = len(market_data)
                workflow_result['steps'].append('data_collection_completed')

                # Step 3: 特徴量エンジニアリング
                workflow_result['steps'].append('feature_engineering_started')
                features = self._engineer_features(market_data)
                workflow_result['feature_count'] = features.shape[1] if len(features) > 0 else 0
                workflow_result['steps'].append('feature_engineering_completed')

                # Step 4: アンサンブル予測
                if len(features) > 20:
                    workflow_result['steps'].append('prediction_started')
                    predictions = self._make_predictions(features)
                    workflow_result['predictions'] = predictions[:5].tolist()  # 最初の5件
                    workflow_result['steps'].append('prediction_completed')

                workflow_result['end_time'] = datetime.now()
                workflow_result['duration'] = (workflow_result['end_time'] - workflow_result['start_time']).total_seconds()
                workflow_result['status'] = 'success'

            except Exception as e:
                workflow_result['error'] = str(e)
                workflow_result['status'] = 'failed'
                workflow_result['end_time'] = datetime.now()

            self.workflow_results.append(workflow_result)
            return workflow_result

        # 完全ワークフロータスク作成
        workflow_task = ScheduledTask(
            task_id="complete_workflow",
            name="Complete Automation Workflow",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=complete_workflow_task
        )

        # ワークフロー実行
        self.scheduler.add_task(workflow_task)
        self.scheduler._execute_task(workflow_task)

        # 実行結果確認
        self.assertEqual(workflow_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(self.workflow_results), 1)

        # ワークフロー詳細確認
        result = self.workflow_results[0]
        self.assertEqual(result['status'], 'success')
        self.assertIn('symbol_selection_completed', result['steps'])
        self.assertIn('data_collection_completed', result['steps'])
        self.assertIn('feature_engineering_completed', result['steps'])

        # 実行時間確認
        self.assertLess(result['duration'], 30.0, "ワークフロー実行時間が長すぎます")

    async def _async_symbol_selection(self) -> List[str]:
        """非同期銘柄選択"""
        # 模擬的な非同期処理
        await asyncio.sleep(0.1)
        return ['7203.T', '6758.T', '9984.T']

    def _collect_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """市場データ収集"""
        # DataFetcher使用（モック化済み）
        fetcher = DataFetcher()
        return fetcher.fetch_data(symbols)

    def _engineer_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """特徴量エンジニアリング"""
        if market_data.empty:
            return np.array([])

        # 簡単な特徴量生成
        features = []
        for symbol in market_data['Symbol'].unique():
            symbol_data = market_data[market_data['Symbol'] == symbol]
            if len(symbol_data) > 10:
                # 基本統計特徴量
                symbol_features = [
                    symbol_data['Close'].mean(),
                    symbol_data['Close'].std(),
                    symbol_data['Volume'].mean(),
                    symbol_data['High'].max() - symbol_data['Low'].min(),
                    len(symbol_data)
                ]
                features.append(symbol_features)

        return np.array(features) if features else np.array([])

    def _make_predictions(self, features: np.ndarray) -> np.ndarray:
        """アンサンブル予測"""
        if len(features) == 0:
            return np.array([])

        # 簡単な予測（実際はEnsembleSystemを使用）
        n_samples = len(features)
        # ランダム予測（テスト用）
        return np.random.randn(n_samples)

    def _create_workflow_data(self) -> pd.DataFrame:
        """ワークフロー用データ作成"""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        symbols = ['7203.T', '6758.T', '9984.T']

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(2000000, 4000000)
                })

        return pd.DataFrame(data)

    def test_multi_schedule_coordination(self):
        """複数スケジュール協調テスト"""
        coordination_log = []

        def morning_prep_task():
            coordination_log.append(f"morning_prep_{datetime.now().isoformat()}")
            return "morning_completed"

        def analysis_task():
            coordination_log.append(f"analysis_{datetime.now().isoformat()}")
            return "analysis_completed"

        def reporting_task():
            coordination_log.append(f"reporting_{datetime.now().isoformat()}")
            return "reporting_completed"

        # 協調タスク群作成
        tasks = [
            ScheduledTask(
                task_id="morning_prep",
                name="Morning Preparation",
                schedule_type=ScheduleType.ON_DEMAND,
                target_function=morning_prep_task
            ),
            ScheduledTask(
                task_id="main_analysis",
                name="Main Analysis",
                schedule_type=ScheduleType.ON_DEMAND,
                target_function=analysis_task
            ),
            ScheduledTask(
                task_id="final_reporting",
                name="Final Reporting",
                schedule_type=ScheduleType.ON_DEMAND,
                target_function=reporting_task
            )
        ]

        # 全タスク追加・順次実行
        for task in tasks:
            task.next_execution = datetime.now() - timedelta(seconds=1)
            self.scheduler.add_task(task)
            self.scheduler._execute_task(task)

        # 協調実行確認
        self.assertEqual(len(coordination_log), 3)

        # 全タスク成功確認
        for task in tasks:
            status = self.scheduler.get_task_status(task.task_id)
            self.assertEqual(status['status'], ExecutionStatus.SUCCESS.value)


if __name__ == '__main__':
    # テストスイート設定
    test_suite = unittest.TestSuite()

    # SmartSelector統合テスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerSmartSelectorIntegration))

    # Ensemble統合テスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerEnsembleIntegration))

    # データ統合テスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerDataIntegration))

    # エンドツーエンドテスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerEndToEnd))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n{'='*75}")
    print(f"ExecutionScheduler統合テスト完了")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*75}")