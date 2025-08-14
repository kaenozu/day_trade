#!/usr/bin/env python3
"""
Issue #760: 基本機能テスト

プロジェクトの主要コンポーネントの基本機能をテスト
"""

import unittest
import sys
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestBasicImports(unittest.TestCase):
    """基本インポートテスト"""

    def test_project_structure(self):
        """プロジェクト構造の確認"""
        src_dir = project_root / 'src' / 'day_trade'
        self.assertTrue(src_dir.exists(), "src/day_trade ディレクトリが存在しない")

        # 主要ディレクトリの確認
        important_dirs = [
            'analysis', 'ml', 'data', 'config', 'utils'
        ]

        for dir_name in important_dirs:
            dir_path = src_dir / dir_name
            self.assertTrue(dir_path.exists(), f"{dir_name} ディレクトリが存在しない")

    def test_basic_imports(self):
        """基本的なインポートテスト"""
        try:
            from src.day_trade.models.enums import Signal, TradingAction
            self.assertIsNotNone(Signal)
            self.assertIsNotNone(TradingAction)
        except ImportError as e:
            self.fail(f"基本モデルのインポートに失敗: {e}")

    def test_config_import(self):
        """設定モジュールのインポートテスト"""
        try:
            from src.day_trade.config.config_manager import ConfigManager
            config = ConfigManager()
            self.assertIsNotNone(config)
        except ImportError as e:
            self.fail(f"設定管理のインポートに失敗: {e}")

    def test_utils_import(self):
        """ユーティリティのインポートテスト"""
        try:
            from src.day_trade.utils.exceptions import DayTradeError
            self.assertIsNotNone(DayTradeError)
        except ImportError as e:
            self.fail(f"ユーティリティのインポートに失敗: {e}")

class TestBasicFunctionality(unittest.TestCase):
    """基本機能テスト"""

    def test_signal_enum(self):
        """シグナル列挙型のテスト"""
        from src.day_trade.models.enums import Signal

        # 基本的なシグナルの存在確認
        self.assertTrue(hasattr(Signal, 'BUY'))
        self.assertTrue(hasattr(Signal, 'SELL'))
        self.assertTrue(hasattr(Signal, 'HOLD'))

    def test_trading_action_enum(self):
        """トレーディングアクション列挙型のテスト"""
        from src.day_trade.models.enums import TradingAction

        # 基本的なアクションの存在確認
        self.assertTrue(hasattr(TradingAction, 'BUY'))
        self.assertTrue(hasattr(TradingAction, 'SELL'))
        self.assertTrue(hasattr(TradingAction, 'HOLD'))

    def test_config_manager_basic(self):
        """設定管理の基本テスト"""
        from src.day_trade.config.config_manager import ConfigManager

        config = ConfigManager()
        self.assertIsNotNone(config)

        # 基本的な設定の確認
        self.assertTrue(hasattr(config, 'load_config'))

    def test_exception_hierarchy(self):
        """例外階層のテスト"""
        from src.day_trade.utils.exceptions import (
            DayTradeError,
            DataError,
            ModelError
        )

        # 例外階層の確認
        self.assertTrue(issubclass(DataError, DayTradeError))
        self.assertTrue(issubclass(ModelError, DayTradeError))

class TestDataModels(unittest.TestCase):
    """データモデルテスト"""

    def test_stock_model_import(self):
        """Stock モデルのインポートテスト"""
        try:
            from src.day_trade.models.stock import Stock
            self.assertIsNotNone(Stock)
        except ImportError as e:
            self.fail(f"Stockモデルのインポートに失敗: {e}")

    def test_stock_model_basic(self):
        """Stock モデルの基本機能テスト"""
        from src.day_trade.models.stock import Stock

        # 基本的なStock作成
        try:
            stock = Stock(
                symbol="1234",
                name="テスト株式",
                price=1000.0,
                volume=100000
            )
            self.assertEqual(stock.symbol, "1234")
            self.assertEqual(stock.name, "テスト株式")
            self.assertEqual(stock.price, 1000.0)
            self.assertEqual(stock.volume, 100000)
        except Exception as e:
            self.fail(f"Stock作成に失敗: {e}")

class TestMLComponents(unittest.TestCase):
    """ML関連コンポーネントテスト"""

    def test_base_model_import(self):
        """ベースモデルのインポートテスト"""
        try:
            from src.day_trade.ml.base_models import BaseModelInterface
            self.assertIsNotNone(BaseModelInterface)
        except ImportError as e:
            self.skip(f"ベースモデルのインポートに失敗: {e}")

    def test_ensemble_import(self):
        """アンサンブルシステムのインポートテスト"""
        try:
            from src.day_trade.analysis.ensemble import EnsembleSystem
            self.assertIsNotNone(EnsembleSystem)
        except ImportError as e:
            self.skip(f"アンサンブルシステムのインポートに失敗: {e}")

class TestSystemIntegration(unittest.TestCase):
    """システム統合テスト"""

    def test_project_python_path(self):
        """プロジェクトのPythonパステスト"""
        import sys

        # プロジェクトルートがパスに含まれているか確認
        project_root_str = str(project_root)
        self.assertIn(project_root_str, sys.path)

    def test_basic_system_workflow(self):
        """基本的なシステムワークフローテスト"""
        try:
            # 基本的な設定の読み込み
            from src.day_trade.config.config_manager import ConfigManager
            config = ConfigManager()

            # 基本的なモデルの作成
            from src.day_trade.models.stock import Stock
            stock = Stock(symbol="TEST", name="テスト", price=100.0, volume=1000)

            # 基本的な例外処理
            from src.day_trade.utils.exceptions import DayTradeError

            self.assertIsNotNone(config)
            self.assertIsNotNone(stock)
            self.assertIsNotNone(DayTradeError)

        except Exception as e:
            self.fail(f"基本システムワークフローに失敗: {e}")

def run_basic_tests():
    """基本テストの実行"""
    print("="*60)
    print("Day Trade 基本機能テスト実行")
    print("="*60)

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # テストクラスを追加
    test_classes = [
        TestBasicImports,
        TestBasicFunctionality,
        TestDataModels,
        TestMLComponents,
        TestSystemIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print("\n" + "="*60)
    print("テスト結果サマリー")
    print("="*60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    print(f"スキップ: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"成功率: {success_rate:.1f}%")

    if result.failures:
        print(f"\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print("- " + str(test) + ": " + (traceback.split('\n')[-2] if '\n' in traceback else traceback))

    print("="*60)

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_basic_tests()
    exit(0 if success else 1)