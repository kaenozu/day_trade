"""
Issue #638: 適応レベル選択ロジックの改善とパラメータ化のテスト

改善された適応レベル選択機能の包括的テスト:
- AdaptiveLevelConfigの設定テスト
- システムリソース情報収集のテスト
- 特殊条件検出のテスト
- 負荷レベル判定のテスト
- 最適レベル選択のテスト
- GPU検出機能のテスト
- 詳細ログ出力のテスト
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List

from src.day_trade.core.optimization_strategy import (
    OptimizationLevel,
    OptimizationConfig,
    AdaptiveLevelConfig,
    OptimizationStrategy,
    OptimizationStrategyFactory
)


class MockOptimizationStrategy(OptimizationStrategy):
    """テスト用のモック戦略"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)

    def execute(self, *args, **kwargs):
        return "mock_result"

    def get_strategy_name(self) -> str:
        return "MockStrategy"


class TestAdaptiveLevelConfig(unittest.TestCase):
    """AdaptiveLevelConfigの設定テスト"""

    def test_default_config_creation(self):
        """デフォルト設定の作成テスト"""
        config = AdaptiveLevelConfig()

        # デフォルト閾値の確認
        self.assertEqual(config.high_load_cpu_threshold, 80.0)
        self.assertEqual(config.high_load_memory_threshold, 80.0)
        self.assertEqual(config.medium_load_cpu_threshold, 60.0)
        self.assertEqual(config.medium_load_memory_threshold, 60.0)

        # GPU設定の確認
        self.assertTrue(config.enable_gpu_detection)
        self.assertEqual(config.gpu_memory_threshold, 70.0)

        # デバッグモード設定の確認
        self.assertTrue(config.enable_debug_mode_detection)
        self.assertIn("DEBUG", config.debug_mode_env_vars)
        self.assertIn("DAYTRADE_DEBUG", config.debug_mode_env_vars)

        # フォールバック設定の確認
        self.assertEqual(config.fallback_level, OptimizationLevel.STANDARD)

    def test_custom_config_creation(self):
        """カスタム設定の作成テスト"""
        config = AdaptiveLevelConfig(
            high_load_cpu_threshold=90.0,
            medium_load_memory_threshold=50.0,
            enable_gpu_detection=False,
            fallback_level=OptimizationLevel.DEBUG
        )

        self.assertEqual(config.high_load_cpu_threshold, 90.0)
        self.assertEqual(config.medium_load_memory_threshold, 50.0)
        self.assertFalse(config.enable_gpu_detection)
        self.assertEqual(config.fallback_level, OptimizationLevel.DEBUG)

    def test_preference_lists(self):
        """優先順位リストのテスト"""
        config = AdaptiveLevelConfig()

        # 高負荷時の優先順位
        expected_high = [OptimizationLevel.STANDARD, OptimizationLevel.DEBUG]
        self.assertEqual(config.high_load_preference, expected_high)

        # 中負荷時の優先順位
        expected_medium = [OptimizationLevel.OPTIMIZED, OptimizationLevel.STANDARD]
        self.assertEqual(config.medium_load_preference, expected_medium)

        # 低負荷時の優先順位
        expected_low = [
            OptimizationLevel.GPU_ACCELERATED,
            OptimizationLevel.OPTIMIZED,
            OptimizationLevel.STANDARD
        ]
        self.assertEqual(config.low_load_preference, expected_low)


class TestSystemResourceGathering(unittest.TestCase):
    """システムリソース情報収集のテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.factory = OptimizationStrategyFactory
        self.adaptive_config = AdaptiveLevelConfig()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.getloadavg')
    def test_gather_system_resources_success(self, mock_getloadavg, mock_virtual_memory, mock_cpu_percent):
        """正常なシステムリソース収集テスト"""
        # psutilのモック設定
        mock_cpu_percent.return_value = 75.0

        mock_memory = Mock()
        mock_memory.percent = 65.0
        mock_memory.available = 8 * (1024**3)  # 8GB
        mock_virtual_memory.return_value = mock_memory

        mock_getloadavg.return_value = [1.5, 1.0, 0.8]

        # GPU検出を無効にしてテスト
        self.adaptive_config.enable_gpu_detection = False

        result = self.factory._gather_system_resources(self.adaptive_config)

        # 結果の確認
        self.assertEqual(result['cpu_percent'], 75.0)
        self.assertEqual(result['memory_percent'], 65.0)
        self.assertEqual(result['memory_available_gb'], 8.0)
        self.assertEqual(result['load_average'], 1.5)
        self.assertFalse(result.get('gpu_available', False))

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.getloadavg')
    def test_gather_system_resources_with_gpu(self, mock_getloadavg, mock_virtual_memory, mock_cpu_percent):
        """GPU検出付きリソース収集テスト"""
        # 基本リソースのモック
        mock_cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 40.0
        mock_memory.available = 16 * (1024**3)
        mock_virtual_memory.return_value = mock_memory
        mock_getloadavg.return_value = [0.5, 0.4, 0.3]

        # GPU検出メソッドのモック
        with patch.object(self.factory, '_check_gpu_availability', return_value=True), \
             patch.object(self.factory, '_get_gpu_memory_usage', return_value=30.0):

            result = self.factory._gather_system_resources(self.adaptive_config)

            self.assertTrue(result['gpu_available'])
            self.assertEqual(result['gpu_memory_percent'], 30.0)

    @patch('psutil.cpu_percent')
    def test_gather_system_resources_error_handling(self, mock_cpu_percent):
        """エラー時のフォールバック処理テスト"""
        # psutilでエラーが発生する状況をシミュレート
        mock_cpu_percent.side_effect = Exception("CPU監視エラー")

        result = self.factory._gather_system_resources(self.adaptive_config)

        # デフォルト値が設定されることを確認
        self.assertEqual(result['cpu_percent'], 50.0)
        self.assertEqual(result['memory_percent'], 50.0)
        self.assertFalse(result['gpu_available'])
        self.assertEqual(result['load_average'], 1.0)


class TestSpecialConditions(unittest.TestCase):
    """特殊条件検出のテスト"""

    def setUp(self):
        self.factory = OptimizationStrategyFactory
        self.adaptive_config = AdaptiveLevelConfig()
        self.available_levels = [
            OptimizationLevel.STANDARD,
            OptimizationLevel.OPTIMIZED,
            OptimizationLevel.DEBUG,
            OptimizationLevel.GPU_ACCELERATED
        ]

    def test_debug_mode_detection(self):
        """デバッグモード検出テスト"""
        # DEBUG環境変数を設定
        with patch.dict(os.environ, {'DEBUG': 'true'}):
            conditions = self.factory._check_special_conditions(
                self.adaptive_config, self.available_levels
            )
            self.assertTrue(conditions['debug_mode'])

        # デバッグ環境変数をクリア
        with patch.dict(os.environ, {}, clear=True):
            conditions = self.factory._check_special_conditions(
                self.adaptive_config, self.available_levels
            )
            self.assertFalse(conditions['debug_mode'])

    def test_ci_environment_detection(self):
        """CI環境検出テスト"""
        # CI環境変数を設定
        with patch.dict(os.environ, {'CI': 'true'}):
            conditions = self.factory._check_special_conditions(
                self.adaptive_config, self.available_levels
            )
            self.assertTrue(conditions['ci_environment'])

        # CI環境変数をクリア
        with patch.dict(os.environ, {}, clear=True):
            conditions = self.factory._check_special_conditions(
                self.adaptive_config, self.available_levels
            )
            self.assertFalse(conditions['ci_environment'])

    def test_available_level_conditions(self):
        """利用可能レベル条件テスト"""
        conditions = self.factory._check_special_conditions(
            self.adaptive_config, self.available_levels
        )

        self.assertTrue(conditions['gpu_level_available'])
        self.assertTrue(conditions['debug_level_available'])
        self.assertTrue(conditions['optimized_level_available'])

        # 限定的な利用可能レベルでテスト
        limited_levels = [OptimizationLevel.STANDARD]
        conditions = self.factory._check_special_conditions(
            self.adaptive_config, limited_levels
        )

        self.assertFalse(conditions['gpu_level_available'])
        self.assertFalse(conditions['debug_level_available'])
        self.assertFalse(conditions['optimized_level_available'])


class TestLoadLevelDetermination(unittest.TestCase):
    """負荷レベル判定のテスト"""

    def setUp(self):
        self.factory = OptimizationStrategyFactory
        self.adaptive_config = AdaptiveLevelConfig()

    def test_high_load_detection(self):
        """高負荷検出テスト"""
        # CPU高負荷
        resource_info = {'cpu_percent': 85.0, 'memory_percent': 50.0}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'high')

        # メモリ高負荷
        resource_info = {'cpu_percent': 50.0, 'memory_percent': 85.0}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'high')

        # 両方高負荷
        resource_info = {'cpu_percent': 90.0, 'memory_percent': 90.0}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'high')

    def test_medium_load_detection(self):
        """中負荷検出テスト"""
        # CPU中負荷
        resource_info = {'cpu_percent': 70.0, 'memory_percent': 40.0}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'medium')

        # メモリ中負荷
        resource_info = {'cpu_percent': 40.0, 'memory_percent': 70.0}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'medium')

    def test_low_load_detection(self):
        """低負荷検出テスト"""
        resource_info = {'cpu_percent': 30.0, 'memory_percent': 40.0}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'low')

        # 非常に低い負荷
        resource_info = {'cpu_percent': 10.0, 'memory_percent': 20.0}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'low')

    def test_missing_resource_info(self):
        """リソース情報欠如時のテスト"""
        resource_info = {}
        load_level = self.factory._determine_load_level(resource_info, self.adaptive_config)
        self.assertEqual(load_level, 'low')  # デフォルト値50%は低負荷


class TestOptimalLevelSelection(unittest.TestCase):
    """最適レベル選択のテスト"""

    def setUp(self):
        self.factory = OptimizationStrategyFactory
        self.adaptive_config = AdaptiveLevelConfig()
        self.available_levels = [
            OptimizationLevel.STANDARD,
            OptimizationLevel.OPTIMIZED,
            OptimizationLevel.DEBUG,
            OptimizationLevel.GPU_ACCELERATED
        ]

    def test_debug_mode_priority(self):
        """デバッグモード優先テスト"""
        special_conditions = {
            'debug_mode': True,
            'debug_level_available': True,
            'ci_environment': False
        }

        selected = self.factory._select_optimal_level(
            'low', special_conditions, self.available_levels, self.adaptive_config
        )
        self.assertEqual(selected, OptimizationLevel.DEBUG)

    def test_ci_environment_priority(self):
        """CI環境優先テスト"""
        special_conditions = {
            'debug_mode': False,
            'ci_environment': True
        }

        selected = self.factory._select_optimal_level(
            'low', special_conditions, self.available_levels, self.adaptive_config
        )
        self.assertEqual(selected, OptimizationLevel.STANDARD)

    def test_load_based_selection(self):
        """負荷ベース選択テスト"""
        special_conditions = {
            'debug_mode': False,
            'ci_environment': False
        }

        # 高負荷時
        selected = self.factory._select_optimal_level(
            'high', special_conditions, self.available_levels, self.adaptive_config
        )
        self.assertEqual(selected, OptimizationLevel.STANDARD)

        # 中負荷時
        selected = self.factory._select_optimal_level(
            'medium', special_conditions, self.available_levels, self.adaptive_config
        )
        self.assertEqual(selected, OptimizationLevel.OPTIMIZED)

        # 低負荷時
        selected = self.factory._select_optimal_level(
            'low', special_conditions, self.available_levels, self.adaptive_config
        )
        self.assertEqual(selected, OptimizationLevel.GPU_ACCELERATED)

    def test_fallback_when_unavailable(self):
        """利用不可時のフォールバックテスト"""
        # GPUレベルが利用できない場合
        limited_levels = [OptimizationLevel.STANDARD, OptimizationLevel.OPTIMIZED]
        special_conditions = {'debug_mode': False, 'ci_environment': False}

        selected = self.factory._select_optimal_level(
            'low', special_conditions, limited_levels, self.adaptive_config
        )
        self.assertEqual(selected, OptimizationLevel.OPTIMIZED)

        # 何も利用できない場合
        empty_levels = []
        selected = self.factory._select_optimal_level(
            'low', special_conditions, empty_levels, self.adaptive_config
        )
        self.assertEqual(selected, self.adaptive_config.fallback_level)


class TestGPUDetection(unittest.TestCase):
    """GPU検出機能のテスト"""

    def setUp(self):
        self.factory = OptimizationStrategyFactory

    @patch('subprocess.run')
    def test_gpu_availability_check_success(self, mock_run):
        """GPU利用可能性チェック成功テスト"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = self.factory._check_gpu_availability()
        self.assertTrue(result)

        mock_run.assert_called_once_with(
            ['nvidia-smi'], capture_output=True, text=True, timeout=2
        )

    @patch('subprocess.run')
    def test_gpu_availability_check_failure(self, mock_run):
        """GPU利用可能性チェック失敗テスト"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = self.factory._check_gpu_availability()
        self.assertFalse(result)

    @patch('subprocess.run')
    def test_gpu_availability_check_exception(self, mock_run):
        """GPU利用可能性チェック例外テスト"""
        mock_run.side_effect = FileNotFoundError()

        result = self.factory._check_gpu_availability()
        self.assertFalse(result)

    @patch('subprocess.run')
    def test_gpu_memory_usage_success(self, mock_run):
        """GPUメモリ使用率取得成功テスト"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "2048, 8192"
        mock_run.return_value = mock_result

        result = self.factory._get_gpu_memory_usage()
        expected = (2048 / 8192) * 100.0  # 25%
        self.assertEqual(result, expected)

    @patch('subprocess.run')
    def test_gpu_memory_usage_failure(self, mock_run):
        """GPUメモリ使用率取得失敗テスト"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = self.factory._get_gpu_memory_usage()
        self.assertEqual(result, 0.0)


class TestIntegratedAdaptiveSelection(unittest.TestCase):
    """統合適応選択テスト"""

    def setUp(self):
        self.factory = OptimizationStrategyFactory
        # テスト用の戦略を登録
        self.factory.register_strategy("test_component", OptimizationLevel.STANDARD, MockOptimizationStrategy)
        self.factory.register_strategy("test_component", OptimizationLevel.OPTIMIZED, MockOptimizationStrategy)
        self.factory.register_strategy("test_component", OptimizationLevel.DEBUG, MockOptimizationStrategy)

        self.adaptive_config = AdaptiveLevelConfig()
        self.optimization_config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            adaptive_config=self.adaptive_config
        )

    @patch.dict(os.environ, {}, clear=True)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.getloadavg')
    def test_low_load_gpu_selection(self, mock_getloadavg, mock_virtual_memory, mock_cpu_percent):
        """低負荷時のGPU選択テスト"""
        # 低負荷状況をシミュレート
        mock_cpu_percent.return_value = 30.0
        mock_memory = Mock()
        mock_memory.percent = 35.0
        mock_memory.available = 16 * (1024**3)
        mock_virtual_memory.return_value = mock_memory
        mock_getloadavg.return_value = [0.5, 0.4, 0.3]

        # GPU利用可能な戦略を追加
        self.factory.register_strategy("test_component", OptimizationLevel.GPU_ACCELERATED, MockOptimizationStrategy)

        with patch.object(self.factory, '_check_gpu_availability', return_value=True):
            selected_level = self.factory._select_adaptive_level("test_component", self.optimization_config)
            self.assertEqual(selected_level, OptimizationLevel.GPU_ACCELERATED)

    @patch.dict(os.environ, {'DEBUG': 'true'}, clear=True)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.getloadavg')
    def test_debug_mode_selection(self, mock_getloadavg, mock_virtual_memory, mock_cpu_percent):
        """デバッグモード選択テスト"""
        # 任意の負荷状況
        mock_cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * (1024**3)
        mock_virtual_memory.return_value = mock_memory
        mock_getloadavg.return_value = [1.0, 1.0, 1.0]

        selected_level = self.factory._select_adaptive_level("test_component", self.optimization_config)
        self.assertEqual(selected_level, OptimizationLevel.DEBUG)

    @patch.dict(os.environ, {'CI': 'true'}, clear=True)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.getloadavg')
    def test_ci_environment_selection(self, mock_getloadavg, mock_virtual_memory, mock_cpu_percent):
        """CI環境選択テスト"""
        # 低負荷でもCI環境では安定した実装を選択
        mock_cpu_percent.return_value = 20.0
        mock_memory = Mock()
        mock_memory.percent = 25.0
        mock_memory.available = 16 * (1024**3)
        mock_virtual_memory.return_value = mock_memory
        mock_getloadavg.return_value = [0.3, 0.2, 0.1]

        selected_level = self.factory._select_adaptive_level("test_component", self.optimization_config)
        self.assertEqual(selected_level, OptimizationLevel.STANDARD)

    def test_error_handling_fallback(self):
        """エラー処理フォールバックテスト"""
        # より深刻なエラーを発生させる - 存在しないコンポーネントで利用可能レベル取得時にエラー
        with patch.object(self.factory, '_get_available_levels', side_effect=Exception("重大なエラー")):
            selected_level = self.factory._select_adaptive_level("test_component", self.optimization_config)
            self.assertEqual(selected_level, self.adaptive_config.fallback_level)


if __name__ == '__main__':
    unittest.main(verbosity=2)