#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Symbol Selector テストスイート
Issue #854対応：動的銘柄選択システムの包括的テスト

テスト項目：
1. 設定管理機能のテスト
2. データベースプロバイダーインターフェースのテスト
3. SQLクエリビルダーのテスト
4. 銘柄選択ロジックのテスト
5. エラーハンドリングとリトライ機能のテスト
6. 統合シナリオのテスト
"""

import unittest
import tempfile
import shutil
import sqlite3
import yaml
from pathlib import Path
from unittest.mock import patch

# Windows環境対応
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    from src.day_trade.data.symbol_selector import (
        DynamicSymbolSelector,
        ConfigurationManager,
        QueryBuilder,
        TOPIX500DatabaseProvider,
        SymbolSelectionCriteria,
        create_symbol_selector
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestConfigurationManager(unittest.TestCase):
    """設定管理クラステスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_symbol_config.yaml"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_default_config_creation(self):
        """デフォルト設定ファイル作成テスト"""
        config_manager = ConfigurationManager(self.config_path)

        # 設定が読み込まれること
        self.assertIsNotNone(config_manager.config)
        self.assertIn('system', config_manager.config)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_custom_config_loading(self):
        """カスタム設定読み込みテスト"""
        custom_config = {
            'system': {
                'log_level': 'DEBUG',
                'db_path': 'test_db.db'
            },
            'default_criteria': {
                'min_market_cap': 200_000_000_000,
                'max_symbols': 30
            },
            'strategies': {
                'test_strategy': {
                    'min_market_cap': 500_000_000_000,
                    'max_symbols': 10,
                    'excluded_sectors': ['REIT']
                }
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(custom_config, f)

        config_manager = ConfigurationManager(self.config_path)

        # カスタム設定が読み込まれること
        self.assertEqual(
            config_manager.get_system_setting('db_path'),
            'test_db.db'
        )

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_strategy_criteria_creation(self):
        """戦略別基準作成テスト"""
        config_manager = ConfigurationManager(self.config_path)

        # デフォルト戦略の取得
        criteria = config_manager.get_criteria_for_strategy('nonexistent')
        self.assertIsInstance(criteria, SymbolSelectionCriteria)
        self.assertEqual(criteria.max_symbols, 50)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 存在しないパスでの初期化
        invalid_path = Path("/invalid/path/config.yaml")
        config_manager = ConfigurationManager(invalid_path)

        # デフォルト設定で動作すること
        self.assertIsNotNone(config_manager.config)
        self.assertIn('system', config_manager.config)


class TestQueryBuilder(unittest.TestCase):
    """SQLクエリビルダーテスト"""

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_basic_query_building(self):
        """基本的なクエリ構築テスト"""
        builder = QueryBuilder()

        query, params = (builder
                         .reset()
                         .add_base_filters()
                         .set_limit(10)
                         .build())

        self.assertIn("SELECT", query)
        self.assertIn("FROM topix500_master", query)
        self.assertIn("is_active = TRUE", query)
        self.assertIn("LIMIT 10", query)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_market_cap_filtering(self):
        """時価総額フィルターテスト"""
        builder = QueryBuilder()

        query, params = (builder
                         .reset()
                         .add_market_cap_filter(100_000_000_000, 500_000_000_000)
                         .build())

        self.assertIn("market_cap >= ?", query)
        self.assertIn("market_cap <= ?", query)
        self.assertEqual(params, [100_000_000_000, 500_000_000_000])

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_sector_filtering(self):
        """セクターフィルターテスト"""
        builder = QueryBuilder()

        excluded = ['REIT', 'ETF']
        preferred = ['Technology', 'Healthcare']

        query, params = (builder
                         .reset()
                         .add_sector_filters(excluded, preferred)
                         .build())

        self.assertIn("sector_code NOT IN", query)
        self.assertIn("sector_code IN", query)
        self.assertEqual(params, excluded + preferred)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_custom_filter(self):
        """カスタムフィルターテスト"""
        builder = QueryBuilder()

        query, params = (builder
                         .reset()
                         .add_custom_filter("volume > ?", [1000000])
                         .build())

        self.assertIn("volume > ?", query)
        self.assertEqual(params, [1000000])

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_order_setting(self):
        """ソート設定テスト"""
        builder = QueryBuilder()

        query, params = (builder
                         .reset()
                         .set_order("market_cap_desc")
                         .build())

        self.assertIn("ORDER BY market_cap DESC", query)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_complex_query(self):
        """複合クエリテスト"""
        builder = QueryBuilder()

        query, params = (builder
                         .reset()
                         .add_base_filters()
                         .add_market_cap_filter(100_000_000_000, None)
                         .add_sector_filters(['REIT'], ['Technology'])
                         .set_order("topix_weight_desc")
                         .set_limit(20)
                         .build())

        # クエリ構造の確認
        self.assertIn("is_active = TRUE", query)
        self.assertIn("market_cap >= ?", query)
        self.assertIn("sector_code NOT IN", query)
        self.assertIn("sector_code IN", query)
        self.assertIn("ORDER BY topix_weight DESC", query)
        self.assertIn("LIMIT 20", query)

        # パラメータ順序の確認
        expected_params = [100_000_000_000, 'REIT', 'Technology']
        self.assertEqual(params, expected_params)


class TestTOPIX500DatabaseProvider(unittest.TestCase):
    """TOPIX500データベースプロバイダーテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_topix.db"

        # テスト用データベース作成
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE topix500_master (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    market_cap REAL,
                    topix_weight REAL,
                    sector_code TEXT,
                    sector_name TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # テストデータ挿入
            test_data = [
                ('7203', 'トヨタ自動車', 300_000_000_000, 3.5, 'AUTO', '自動車', True),
                ('8306', '三菱UFJ', 200_000_000_000, 2.1, 'BANK', '銀行', True),
                ('4751', 'サイバーエージェント', 50_000_000_000, 0.5, 'IT', 'IT', True),
                ('9999', 'テスト銘柄', 10_000_000_000, 0.1, 'OTHER', 'その他', False),
            ]

            conn.executemany("""
                INSERT INTO topix500_master
                (code, name, market_cap, topix_weight, sector_code, sector_name, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, test_data)

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_database_connection(self):
        """データベース接続テスト"""
        provider = TOPIX500DatabaseProvider(str(self.db_path))

        self.assertEqual(provider.db_path, str(self.db_path))

        with provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM topix500_master")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 4)


class TestDynamicSymbolSelector(unittest.TestCase):
    """動的銘柄選択システムテスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_selector.db"
        self.config_path = self.temp_dir / "test_selector_config.yaml"

        # テスト用データベース作成
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE topix500_master (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    market_cap REAL,
                    topix_weight REAL,
                    sector_code TEXT,
                    sector_name TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # 多様なテストデータ
            test_data = [
                ('7203', 'トヨタ自動車', 300_000_000_000, 3.5, 'AUTO', '自動車', True),
                ('8306', '三菱UFJ', 200_000_000_000, 2.1, 'BANK', '銀行', True),
                ('4751', 'サイバーエージェント', 150_000_000_000, 1.5, 'IT', 'IT', True),
                ('6758', 'ソニー', 180_000_000_000, 1.8, 'ELEC', '電機', True),
                ('9984', 'ソフトバンクG', 100_000_000_000, 1.0, 'COMM', '通信', True),
                ('7974', '任天堂', 120_000_000_000, 1.2, 'GAME', 'ゲーム', True),
                ('9999', '非アクティブ銘柄', 50_000_000_000, 0.5, 'OTHER', 'その他', False),
            ]

            conn.executemany("""
                INSERT INTO topix500_master
                (code, name, market_cap, topix_weight, sector_code, sector_name, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, test_data)

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_basic_symbol_selection(self):
        """基本的な銘柄選択テスト"""
        # テスト用設定
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        criteria = SymbolSelectionCriteria(
            min_market_cap=100_000_000_000,
            max_symbols=5
        )

        symbols = selector.select_symbols_by_criteria(criteria)

        # 結果検証
        self.assertIsInstance(symbols, list)
        self.assertLessEqual(len(symbols), 5)
        self.assertIn('7203', symbols)  # トヨタは含まれるはず

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_sector_filtering(self):
        """セクターフィルタリングテスト"""
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        criteria = SymbolSelectionCriteria(
            excluded_sectors=['AUTO', 'BANK'],
            max_symbols=10
        )

        symbols = selector.select_symbols_by_criteria(criteria)

        # トヨタと三菱UFJは除外されるはず
        self.assertNotIn('7203', symbols)  # トヨタ（AUTO）
        self.assertNotIn('8306', symbols)  # 三菱UFJ（BANK）

        # IT銘柄は含まれるはず
        self.assertIn('4751', symbols)  # サイバーエージェント（IT）

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_market_cap_filtering(self):
        """時価総額フィルタリングテスト"""
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        criteria = SymbolSelectionCriteria(
            min_market_cap=150_000_000_000,
            max_symbols=10
        )

        symbols = selector.select_symbols_by_criteria(criteria)

        # 低時価総額銘柄は除外されるはず
        self.assertNotIn('9984', symbols)  # ソフトバンクG（100億）

        # 高時価総額銘柄は含まれるはず
        self.assertIn('7203', symbols)  # トヨタ（3000億）

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_liquid_symbols_method(self):
        """高流動性銘柄取得テスト"""
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        symbols = selector.get_liquid_symbols(limit=3)

        self.assertIsInstance(symbols, list)
        self.assertLessEqual(len(symbols), 3)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_sector_diversified_symbols(self):
        """セクター分散銘柄選択テスト"""
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        symbols = selector.get_sector_diversified_symbols(limit=10)

        self.assertIsInstance(symbols, list)
        self.assertLessEqual(len(symbols), 10)

        # セクター分散が実現されているか確認
        if len(symbols) > 1:
            # セクター情報を取得して分散性をチェック
            sectors = selector._get_sectors_for_symbols(symbols)
            self.assertGreater(len(sectors), 1)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_validation_functionality(self):
        """銘柄選択検証機能テスト"""
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        # 有効な銘柄リストの検証
        valid_symbols = ['7203', '8306', '4751']
        validation_result = selector.validate_symbol_selection(valid_symbols)

        self.assertTrue(validation_result['valid'])
        self.assertEqual(validation_result['valid_symbols'], 3)
        self.assertEqual(len(validation_result['invalid_symbols']), 0)

        # 無効な銘柄を含むリストの検証
        mixed_symbols = ['7203', 'INVALID', '8306']
        validation_result = selector.validate_symbol_selection(mixed_symbols)

        self.assertFalse(validation_result['valid'])
        self.assertEqual(validation_result['valid_symbols'], 2)
        self.assertIn('INVALID', validation_result['invalid_symbols'])

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    @patch('src.day_trade.data.symbol_selector.DynamicSymbolSelector._execute_query_with_retry')
    def test_error_handling_and_fallback(self, mock_query):
        """エラーハンドリングとフォールバック機能テスト"""
        # クエリ実行エラーをシミュレート
        mock_query.side_effect = Exception("Database connection failed")

        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        criteria = SymbolSelectionCriteria(max_symbols=5)
        symbols = selector.select_symbols_by_criteria(criteria)

        # フォールバック銘柄が返されること
        self.assertEqual(symbols, ["7203", "8306", "4751"])

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_strategy_based_selection(self):
        """戦略別銘柄選択テスト"""
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

        # 戦略別選択（存在しない戦略）
        symbols = selector.get_symbols_by_strategy('nonexistent_strategy', limit=5)

        self.assertIsInstance(symbols, list)
        self.assertLessEqual(len(symbols), 5)


class TestFactoryFunction(unittest.TestCase):
    """ファクトリー関数テスト"""

    def setUp(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "factory_test_config.yaml"
        self.db_path = self.temp_dir / "factory_test.db"

    def tearDown(self):
        """テスト後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_create_symbol_selector(self):
        """create_symbol_selector関数テスト"""
        # パラメータ指定での作成
        selector = create_symbol_selector(
            config_path=str(self.config_path),
            db_path=str(self.db_path)
        )

        self.assertIsInstance(selector, DynamicSymbolSelector)

        # パラメータなしでの作成
        default_selector = create_symbol_selector()
        self.assertIsInstance(default_selector, DynamicSymbolSelector)


def run_dynamic_symbol_selector_tests():
    """動的銘柄選択システムテスト実行"""
    print("=== Dynamic Symbol Selector テスト開始 ===")

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_classes = [
        TestConfigurationManager,
        TestQueryBuilder,
        TestTOPIX500DatabaseProvider,
        TestDynamicSymbolSelector,
        TestFactoryFunction
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    print(f"スキップ: {len(result.skipped)}")

    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Windows環境対応
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    success = run_dynamic_symbol_selector_tests()
    sys.exit(0 if success else 1)
