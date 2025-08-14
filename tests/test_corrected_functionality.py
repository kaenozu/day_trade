#!/usr/bin/env python3
"""
Issue #760: 正しい基本機能テスト

実際のコードベース構造に基づく基本機能テスト
"""

import unittest
import sys
from pathlib import Path
from decimal import Decimal

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestCorrectImports(unittest.TestCase):
    """正しいインポートテスト"""

    def test_project_structure(self):
        """プロジェクト構造の確認"""
        src_dir = project_root / 'src' / 'day_trade'
        self.assertTrue(src_dir.exists(), "src/day_trade ディレクトリが存在しない")

        # 主要ディレクトリの確認
        important_dirs = [
            'analysis', 'ml', 'data', 'config', 'utils', 'models'
        ]

        for dir_name in important_dirs:
            dir_path = src_dir / dir_name
            self.assertTrue(dir_path.exists(), f"{dir_name} ディレクトリが存在しない")

    def test_enums_import(self):
        """列挙型のインポートテスト"""
        try:
            from src.day_trade.models.enums import AlertType, TradeType
            self.assertIsNotNone(AlertType)
            self.assertIsNotNone(TradeType)
        except ImportError as e:
            self.fail(f"列挙型のインポートに失敗: {e}")

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
            from src.day_trade.utils.exceptions import DayTradeError, DataError
            self.assertIsNotNone(DayTradeError)
            self.assertIsNotNone(DataError)
        except ImportError as e:
            self.fail(f"ユーティリティのインポートに失敗: {e}")

class TestCorrectEnums(unittest.TestCase):
    """正しい列挙型テスト"""

    def test_alert_type_enum(self):
        """AlertType列挙型のテスト"""
        from src.day_trade.models.enums import AlertType

        # 基本的なアラートタイプの存在確認
        self.assertTrue(hasattr(AlertType, 'PRICE_ABOVE'))
        self.assertTrue(hasattr(AlertType, 'PRICE_BELOW'))
        self.assertTrue(hasattr(AlertType, 'VOLUME_SPIKE'))

    def test_trade_type_enum(self):
        """TradeType列挙型のテスト"""
        from src.day_trade.models.enums import TradeType

        # 基本的なトレードタイプの存在確認
        self.assertTrue(hasattr(TradeType, 'BUY'))
        self.assertTrue(hasattr(TradeType, 'SELL'))

        # 値の確認
        self.assertEqual(TradeType.BUY.value, "BUY")
        self.assertEqual(TradeType.SELL.value, "SELL")

class TestCorrectModels(unittest.TestCase):
    """正しいモデルテスト"""

    def test_stock_model_import(self):
        """Stock モデルのインポートテスト"""
        try:
            from src.day_trade.models.stock import Stock, PriceData, Trade
            self.assertIsNotNone(Stock)
            self.assertIsNotNone(PriceData)
            self.assertIsNotNone(Trade)
        except ImportError as e:
            self.fail(f"Stockモデルのインポートに失敗: {e}")

    def test_stock_model_attributes(self):
        """Stock モデルの属性テスト"""
        from src.day_trade.models.stock import Stock

        # 基本的な属性の存在確認
        self.assertTrue(hasattr(Stock, 'code'))
        self.assertTrue(hasattr(Stock, 'name'))
        self.assertTrue(hasattr(Stock, 'market'))
        self.assertTrue(hasattr(Stock, 'sector'))

    def test_trade_model_attributes(self):
        """Trade モデルの属性テスト"""
        from src.day_trade.models.stock import Trade
        from src.day_trade.models.enums import TradeType

        # 基本的な属性の存在確認
        self.assertTrue(hasattr(Trade, 'stock_code'))
        self.assertTrue(hasattr(Trade, 'trade_type'))
        self.assertTrue(hasattr(Trade, 'quantity'))
        self.assertTrue(hasattr(Trade, 'price'))

    def test_trade_total_amount_method(self):
        """Trade モデルのtotal_amountメソッドテスト"""
        from src.day_trade.models.stock import Trade
        from src.day_trade.models.enums import TradeType

        # Tradeインスタンスを作成（モックデータ）
        trade = Trade()
        trade.trade_type = TradeType.BUY
        trade.quantity = 100
        trade.price = Decimal("1000.50")
        trade.commission = Decimal("500")

        # total_amountの計算テスト
        expected_total = Decimal("100050.00")  # (1000.50 * 100) + 500
        self.assertEqual(trade.total_amount, expected_total)

class TestCorrectUtils(unittest.TestCase):
    """正しいユーティリティテスト"""

    def test_exception_hierarchy(self):
        """例外階層のテスト"""
        from src.day_trade.utils.exceptions import DayTradeError, DataError

        # 例外階層の確認
        self.assertTrue(issubclass(DataError, DayTradeError))

        # 基本的な例外作成テスト
        error = DayTradeError("テストエラー")
        self.assertEqual(str(error), "テストエラー")

    def test_logging_config_import(self):
        """ログ設定のインポートテスト"""
        try:
            from src.day_trade.utils.logging_config import get_logger
            logger = get_logger(__name__)
            self.assertIsNotNone(logger)
        except ImportError as e:
            self.fail(f"ログ設定のインポートに失敗: {e}")

class TestConfigSystem(unittest.TestCase):
    """設定システムテスト"""

    def test_config_manager_creation(self):
        """ConfigManager作成テスト"""
        from src.day_trade.config.config_manager import ConfigManager

        config = ConfigManager()
        self.assertIsNotNone(config)

        # 基本的なメソッドの存在確認
        self.assertTrue(hasattr(config, 'load_config'))

    def test_config_file_exists(self):
        """設定ファイルの存在確認"""
        config_dir = project_root / 'config'
        self.assertTrue(config_dir.exists(), "configディレクトリが存在しない")

        # 主要設定ファイルの確認
        settings_file = config_dir / 'settings.json'
        if settings_file.exists():
            self.assertTrue(settings_file.is_file(), "settings.jsonが正しくない")

class TestBasicMLComponents(unittest.TestCase):
    """基本ML関連コンポーネントテスト"""

    def test_base_model_import(self):
        """ベースモデルのインポートテスト"""
        try:
            from src.day_trade.ml.base_models import BaseModelInterface
            self.assertIsNotNone(BaseModelInterface)
        except ImportError:
            self.skipTest("ベースモデルのインポートに失敗（オプション機能）")

    def test_ensemble_components_import(self):
        """アンサンブル関連コンポーネントのインポートテスト"""
        try:
            # 実際に存在するクラス名を確認
            from src.day_trade.analysis import ensemble
            self.assertIsNotNone(ensemble)
        except ImportError:
            self.skipTest("アンサンブルコンポーネントのインポートに失敗（オプション機能）")

class TestDatabaseModels(unittest.TestCase):
    """データベースモデルテスト"""

    def test_base_model_import(self):
        """ベースモデルのインポートテスト"""
        try:
            from src.day_trade.models.base import BaseModel
            self.assertIsNotNone(BaseModel)
        except ImportError as e:
            self.fail(f"ベースモデルのインポートに失敗: {e}")

    def test_database_model_inheritance(self):
        """データベースモデルの継承テスト"""
        from src.day_trade.models.base import BaseModel
        from src.day_trade.models.stock import Stock, PriceData, Trade

        # 継承関係の確認
        self.assertTrue(issubclass(Stock, BaseModel))
        self.assertTrue(issubclass(PriceData, BaseModel))
        self.assertTrue(issubclass(Trade, BaseModel))

def run_corrected_tests():
    """正しいテストの実行"""
    print("="*70)
    print("Day Trade 正しい基本機能テスト実行")
    print("="*70)

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # テストクラスを追加
    test_classes = [
        TestCorrectImports,
        TestCorrectEnums,
        TestCorrectModels,
        TestCorrectUtils,
        TestConfigSystem,
        TestBasicMLComponents,
        TestDatabaseModels
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print("\n" + "="*70)
    print("テスト結果サマリー")
    print("="*70)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    # スキップされたテストの数を計算
    skipped_count = 0
    if hasattr(result, 'skipped'):
        skipped_count = len(result.skipped)

    print(f"スキップ: {skipped_count}")

    success_count = result.testsRun - len(result.failures) - len(result.errors)
    success_rate = (success_count / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"成功率: {success_rate:.1f}%")

    # 詳細情報
    if result.failures:
        print(f"\n失敗したテスト ({len(result.failures)}件):")
        for test, traceback in result.failures:
            print(f"- {test}")
            # エラーメッセージの抽出
            if 'AssertionError:' in traceback:
                error_msg = traceback.split('AssertionError:')[-1].strip()
                print(f"  理由: {error_msg}")

    if result.errors:
        print(f"\nエラーが発生したテスト ({len(result.errors)}件):")
        for test, traceback in result.errors:
            print(f"- {test}")
            # エラーメッセージの抽出
            lines = traceback.strip().split('\n')
            for line in reversed(lines):
                if ': ' in line and ('Error' in line or 'Exception' in line):
                    print(f"  エラー: {line}")
                    break

    if skipped_count > 0:
        print(f"\nスキップされたテスト ({skipped_count}件):")
        if hasattr(result, 'skipped'):
            for test, reason in result.skipped:
                print(f"- {test}: {reason}")

    # 評価
    if success_rate >= 90:
        print(f"\n[評価] 優秀! システムの基本機能は正常に動作しています。")
    elif success_rate >= 70:
        print(f"\n[評価] 良好! 基本機能は動作していますが、いくつかの改善点があります。")
    elif success_rate >= 50:
        print(f"\n[評価] 注意が必要! 基本機能に問題がある可能性があります。")
    else:
        print(f"\n[評価] 重要な問題があります。基本機能の修正が必要です。")

    print("="*70)

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_corrected_tests()
    exit(0 if success else 1)