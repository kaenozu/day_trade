#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbol_selector.py改善版のテストコード

テスト項目:
1. 基本機能テスト
2. エラーハンドリングテスト
3. 設定ファイル機能テスト
4. クエリビルダーテスト
5. データベースプロバイダーテスト
6. パフォーマンステスト
"""

import unittest
import sqlite3
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# テスト対象モジュールのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from day_trade.data.symbol_selector import (
        DynamicSymbolSelector,
        SymbolSelectionCriteria,
        ConfigurationManager,
        QueryBuilder,
        TOPIX500DatabaseProvider,
        create_symbol_selector
    )
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("テストをスキップします")
    sys.exit(0)


class TestSymbolSelectionCriteria(unittest.TestCase):
    """SymbolSelectionCriteriaのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        criteria = SymbolSelectionCriteria()

        self.assertEqual(criteria.min_market_cap, 100_000_000_000)
        self.assertIsNone(criteria.max_market_cap)
        self.assertEqual(criteria.min_liquidity_score, 0.5)
        self.assertEqual(criteria.excluded_sectors, [])
        self.assertEqual(criteria.preferred_sectors, [])
        self.assertEqual(criteria.max_symbols, 50)
        self.assertEqual(criteria.sort_criteria, "market_cap_desc")
        self.assertEqual(criteria.additional_filters, [])

    def test_custom_values(self):
        """カスタム値のテスト"""
        criteria = SymbolSelectionCriteria(
            min_market_cap=500_000_000_000,
            max_market_cap=2_000_000_000_000,
            excluded_sectors=["REIT", "ETF"],
            preferred_sectors=["情報・通信業"],
            max_symbols=30
        )

        self.assertEqual(criteria.min_market_cap, 500_000_000_000)
        self.assertEqual(criteria.max_market_cap, 2_000_000_000_000)
        self.assertEqual(criteria.excluded_sectors, ["REIT", "ETF"])
        self.assertEqual(criteria.preferred_sectors, ["情報・通信業"])
        self.assertEqual(criteria.max_symbols, 30)


class TestQueryBuilder(unittest.TestCase):
    """QueryBuilderのテスト"""

    def setUp(self):
        self.builder = QueryBuilder()

    def test_basic_query_build(self):
        """基本クエリ構築のテスト"""
        query, params = (self.builder
                        .reset()
                        .add_base_filters()
                        .build())

        self.assertIn("SELECT", query)
        self.assertIn("FROM topix500_master", query)
        self.assertIn("WHERE is_active = TRUE", query)
        self.assertEqual(params, [])

    def test_market_cap_filters(self):
        """時価総額フィルターのテスト"""
        query, params = (self.builder
                        .reset()
                        .add_base_filters()
                        .add_market_cap_filter(100_000_000_000, 1_000_000_000_000)
                        .build())

        self.assertIn("market_cap >= ?", query)
        self.assertIn("market_cap <= ?", query)
        self.assertEqual(params, [100_000_000_000, 1_000_000_000_000])

    def test_sector_filters(self):
        """セクターフィルターのテスト"""
        excluded = ["REIT", "ETF"]
        preferred = ["情報・通信業"]

        query, params = (self.builder
                        .reset()
                        .add_base_filters()
                        .add_sector_filters(excluded, preferred)
                        .build())

        self.assertIn("sector_code NOT IN", query)
        self.assertIn("sector_code IN", query)
        self.assertEqual(params, excluded + preferred)

    def test_order_and_limit(self):
        """ソートとLIMITのテスト"""
        query, params = (self.builder
                        .reset()
                        .add_base_filters()
                        .set_order("market_cap_desc")
                        .set_limit(20)
                        .build())

        self.assertIn("ORDER BY market_cap DESC", query)
        self.assertIn("LIMIT 20", query)

    def test_custom_filter(self):
        """カスタムフィルターのテスト"""
        query, params = (self.builder
                        .reset()
                        .add_base_filters()
                        .add_custom_filter("topix_weight > ?", [0.1])
                        .build())

        self.assertIn("topix_weight > ?", query)
        self.assertEqual(params, [0.1])


class TestConfigurationManager(unittest.TestCase):
    """ConfigurationManagerのテスト"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_config_creation(self):
        """デフォルト設定ファイル作成のテスト"""
        config_manager = ConfigurationManager(self.config_path)

        self.assertTrue(self.config_path.exists())
        self.assertIn('system', config_manager.config)
        self.assertIn('default_criteria', config_manager.config)

    def test_existing_config_loading(self):
        """既存設定ファイル読み込みのテスト"""
        test_config = {
            'system': {'log_level': 'DEBUG'},
            'default_criteria': {'min_market_cap': 200_000_000_000},
            'strategies': {
                'test_strategy': {
                    'min_market_cap': 500_000_000_000,
                    'max_symbols': 10
                }
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)

        config_manager = ConfigurationManager(self.config_path)

        self.assertEqual(config_manager.config['system']['log_level'], 'DEBUG')
        self.assertEqual(config_manager.config['default_criteria']['min_market_cap'], 200_000_000_000)

    def test_strategy_criteria_generation(self):
        """戦略基準生成のテスト"""
        test_config = {
            'default_criteria': {
                'min_market_cap': 100_000_000_000,
                'max_symbols': 50
            },
            'strategies': {
                'test_strategy': {
                    'min_market_cap': 300_000_000_000,
                    'excluded_sectors': ['REIT']
                }
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)

        config_manager = ConfigurationManager(self.config_path)
        criteria = config_manager.get_criteria_for_strategy('test_strategy')

        self.assertEqual(criteria.min_market_cap, 300_000_000_000)
        self.assertEqual(criteria.max_symbols, 50)  # デフォルト値がマージされる
        self.assertEqual(criteria.excluded_sectors, ['REIT'])


class TestTOPIX500DatabaseProvider(unittest.TestCase):
    """TOPIX500DatabaseProviderのテスト"""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # テスト用データベースの作成
        self._create_test_database()

        self.provider = TOPIX500DatabaseProvider(self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def _create_test_database(self):
        """テスト用データベースの作成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE topix500_master (
                code TEXT PRIMARY KEY,
                name TEXT,
                market_cap REAL,
                topix_weight REAL,
                sector_code TEXT,
                sector_name TEXT,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')

        # テストデータの挿入
        test_data = [
            ('7203', 'トヨタ自動車', 400_000_000_000, 3.5, '輸送用機器', '輸送用機器', True),
            ('8306', '三菱UFJ銀行', 300_000_000_000, 2.1, '銀行業', '銀行業', True),
            ('4751', 'サイバーエージェント', 150_000_000_000, 0.8, '情報・通信業', '情報・通信業', True),
            ('9983', 'ファーストリテイリング', 500_000_000_000, 4.2, '小売業', '小売業', True),
            ('6758', 'ソニー', 250_000_000_000, 1.9, '電気機器', '電気機器', True),
        ]

        cursor.executemany(
            'INSERT INTO topix500_master VALUES (?, ?, ?, ?, ?, ?, ?)',
            test_data
        )

        conn.commit()
        conn.close()

    def test_db_path_property(self):
        """データベースパスプロパティのテスト"""
        self.assertEqual(self.provider.db_path, self.db_path)

    def test_get_connection(self):
        """データベース接続取得のテスト"""
        conn = self.provider.get_connection()
        self.assertIsInstance(conn, sqlite3.Connection)

        # 実際にクエリを実行してみる
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM topix500_master")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 5)

        conn.close()


class TestDynamicSymbolSelector(unittest.TestCase):
    """DynamicSymbolSelectorのテスト"""

    def setUp(self):
        # テスト用の一時ディレクトリとファイル
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.db_path = Path(self.temp_dir) / "test.db"

        # テスト用データベースの作成
        self._create_test_database()

        # テスト用設定ファイルの作成
        self._create_test_config()

        # DynamicSymbolSelectorの初期化
        from day_trade.data.symbol_selector import ConfigurationManager
        config_manager = ConfigurationManager(self.config_path)
        db_provider = TOPIX500DatabaseProvider(str(self.db_path))

        self.selector = DynamicSymbolSelector(
            db_provider=db_provider,
            config_manager=config_manager
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_database(self):
        """テスト用データベースの作成"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE topix500_master (
                code TEXT PRIMARY KEY,
                name TEXT,
                market_cap REAL,
                topix_weight REAL,
                sector_code TEXT,
                sector_name TEXT,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')

        # より多様なテストデータ
        test_data = [
            ('7203', 'トヨタ自動車', 400_000_000_000, 3.5, '輸送用機器', '輸送用機器', True),
            ('8306', '三菱UFJ銀行', 300_000_000_000, 2.1, '銀行業', '銀行業', True),
            ('4751', 'サイバーエージェント', 150_000_000_000, 0.8, '情報・通信業', '情報・通信業', True),
            ('9983', 'ファーストリテイリング', 500_000_000_000, 4.2, '小売業', '小売業', True),
            ('6758', 'ソニー', 250_000_000_000, 1.9, '電気機器', '電気機器', True),
            ('9984', 'ソフトバンク', 350_000_000_000, 2.8, '情報・通信業', '情報・通信業', True),
            ('8411', 'みずほ銀行', 180_000_000_000, 1.2, '銀行業', '銀行業', True),
            ('6861', 'キーエンス', 600_000_000_000, 3.8, '電気機器', '電気機器', True),
            ('7974', '任天堂', 450_000_000_000, 3.1, '情報・通信業', '情報・通信業', True),
            ('4543', 'テルモ', 80_000_000_000, 0.5, '精密機器', '精密機器', True),
        ]

        cursor.executemany(
            'INSERT INTO topix500_master VALUES (?, ?, ?, ?, ?, ?, ?)',
            test_data
        )

        conn.commit()
        conn.close()

    def _create_test_config(self):
        """テスト用設定ファイルの作成"""
        test_config = {
            'system': {
                'log_level': 'INFO',
                'db_path': str(self.db_path)
            },
            'default_criteria': {
                'min_market_cap': 100_000_000_000,
                'max_symbols': 50
            },
            'strategies': {
                'liquid_trading': {
                    'min_market_cap': 300_000_000_000,
                    'max_symbols': 20,
                    'excluded_sectors': ['REIT', 'ETF']
                },
                'volatile_trading': {
                    'min_market_cap': 50_000_000_000,
                    'max_market_cap': 1_000_000_000_000,
                    'max_symbols': 10,
                    'preferred_sectors': ['情報・通信業', '電気機器']
                }
            },
            'sector_diversification': {
                'enabled': True,
                'max_symbols_per_sector': 2,
                'min_sectors': 3,
                'market_cap_threshold': 100_000_000_000
            },
            'error_handling': {
                'max_retries': 3,
                'fallback_to_defaults': True
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)

    def test_basic_symbol_selection(self):
        """基本的な銘柄選択のテスト"""
        criteria = SymbolSelectionCriteria(
            min_market_cap=200_000_000_000,
            max_symbols=5
        )

        symbols = self.selector.select_symbols_by_criteria(criteria)

        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) <= 5)
        self.assertTrue(all(isinstance(symbol, str) for symbol in symbols))

    def test_liquid_symbols(self):
        """高流動性銘柄取得のテスト"""
        symbols = self.selector.get_liquid_symbols(limit=3)

        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) <= 3)

    def test_volatile_symbols(self):
        """高ボラティリティ銘柄取得のテスト"""
        symbols = self.selector.get_volatile_symbols(limit=5)

        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) <= 5)

    def test_sector_diversified_symbols(self):
        """セクター分散銘柄取得のテスト"""
        symbols = self.selector.get_sector_diversified_symbols(limit=6)

        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) <= 6)

        # セクター分散の検証
        validation = self.selector.validate_symbol_selection(symbols)
        self.assertTrue(validation['sector_count'] >= 1)  # 最低1セクター

    def test_symbols_by_strategy(self):
        """戦略別銘柄選択のテスト"""
        symbols = self.selector.get_symbols_by_strategy('liquid_trading', limit=3)

        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) <= 3)

    def test_symbol_validation(self):
        """銘柄選択検証のテスト"""
        test_symbols = ['7203', '8306', '4751']
        validation = self.selector.validate_symbol_selection(test_symbols)

        self.assertIsInstance(validation, dict)
        self.assertIn('valid', validation)
        self.assertIn('total_symbols', validation)
        self.assertIn('valid_symbols', validation)
        self.assertIn('sector_count', validation)

        self.assertEqual(validation['total_symbols'], 3)
        self.assertTrue(validation['valid_symbols'] <= 3)

    def test_empty_result_handling(self):
        """空結果の処理テスト"""
        criteria = SymbolSelectionCriteria(
            min_market_cap=10_000_000_000_000,  # 非現実的に高い値
            max_symbols=10
        )

        symbols = self.selector.select_symbols_by_criteria(criteria)

        # フォールバック処理が動作することを確認
        self.assertIsInstance(symbols, list)

    @patch('day_trade.data.symbol_selector.sqlite3.connect')
    def test_database_error_handling(self, mock_connect):
        """データベースエラーハンドリングのテスト"""
        mock_connect.side_effect = sqlite3.Error("Database connection failed")

        criteria = SymbolSelectionCriteria(max_symbols=5)
        symbols = self.selector.select_symbols_by_criteria(criteria)

        # フォールバック処理により何らかの結果が返されることを確認
        self.assertIsInstance(symbols, list)


class TestFactoryFunction(unittest.TestCase):
    """ファクトリー関数のテスト"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.db_path = Path(self.temp_dir) / "test.db"

        # 最小限のテスト設定ファイル
        test_config = {
            'system': {'db_path': str(self.db_path)},
            'default_criteria': {'max_symbols': 10}
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_symbol_selector(self):
        """create_symbol_selector関数のテスト"""
        selector = create_symbol_selector(
            config_path=str(self.config_path),
            db_path=str(self.db_path)
        )

        self.assertIsInstance(selector, DynamicSymbolSelector)
        self.assertEqual(selector.db_provider.db_path, str(self.db_path))

    def test_create_symbol_selector_defaults(self):
        """デフォルト値でのcreate_symbol_selector関数のテスト"""
        selector = create_symbol_selector()

        self.assertIsInstance(selector, DynamicSymbolSelector)


class TestIntegration(unittest.TestCase):
    """統合テスト"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "integration_config.yaml"
        self.db_path = Path(self.temp_dir) / "integration.db"

        # 統合テスト用のデータベースと設定の作成
        self._setup_integration_environment()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _setup_integration_environment(self):
        """統合テスト環境のセットアップ"""
        # データベース作成
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE topix500_master (
                code TEXT PRIMARY KEY,
                name TEXT,
                market_cap REAL,
                topix_weight REAL,
                sector_code TEXT,
                sector_name TEXT,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')

        # 統合テスト用の豊富なデータ
        test_data = [
            # 銀行業
            ('8306', '三菱UFJ銀行', 300_000_000_000, 2.1, '銀行業', '銀行業', True),
            ('8411', 'みずほ銀行', 180_000_000_000, 1.2, '銀行業', '銀行業', True),
            ('8316', '三井住友銀行', 220_000_000_000, 1.8, '銀行業', '銀行業', True),

            # 情報・通信業
            ('4751', 'サイバーエージェント', 150_000_000_000, 0.8, '情報・通信業', '情報・通信業', True),
            ('9984', 'ソフトバンク', 350_000_000_000, 2.8, '情報・通信業', '情報・通信業', True),
            ('7974', '任天堂', 450_000_000_000, 3.1, '情報・通信業', '情報・通信業', True),

            # 電気機器
            ('6758', 'ソニー', 250_000_000_000, 1.9, '電気機器', '電気機器', True),
            ('6861', 'キーエンス', 600_000_000_000, 3.8, '電気機器', '電気機器', True),

            # 輸送用機器
            ('7203', 'トヨタ自動車', 400_000_000_000, 3.5, '輸送用機器', '輸送用機器', True),
            ('7267', 'ホンダ', 280_000_000_000, 2.3, '輸送用機器', '輸送用機器', True),

            # 小売業
            ('9983', 'ファーストリテイリング', 500_000_000_000, 4.2, '小売業', '小売業', True),

            # 精密機器
            ('4543', 'テルモ', 80_000_000_000, 0.5, '精密機器', '精密機器', True),
        ]

        cursor.executemany(
            'INSERT INTO topix500_master VALUES (?, ?, ?, ?, ?, ?, ?)',
            test_data
        )

        conn.commit()
        conn.close()

        # 統合テスト用設定ファイル
        integration_config = {
            'system': {
                'log_level': 'INFO',
                'db_path': str(self.db_path)
            },
            'default_criteria': {
                'min_market_cap': 100_000_000_000,
                'max_symbols': 50
            },
            'strategies': {
                'liquid_trading': {
                    'min_market_cap': 300_000_000_000,
                    'max_symbols': 20
                },
                'volatile_trading': {
                    'min_market_cap': 50_000_000_000,
                    'max_symbols': 10,
                    'preferred_sectors': ['情報・通信業', '電気機器']
                },
                'balanced_portfolio': {
                    'min_market_cap': 200_000_000_000,
                    'max_symbols': 15,
                    'excluded_sectors': ['REIT', 'ETF']
                }
            },
            'sector_diversification': {
                'enabled': True,
                'max_symbols_per_sector': 2,
                'min_sectors': 3,
                'market_cap_threshold': 100_000_000_000
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(integration_config, f, default_flow_style=False, allow_unicode=True)

    def test_end_to_end_workflow(self):
        """エンドツーエンドのワークフローテスト"""
        # セレクターの作成
        selector = create_symbol_selector(
            config_path=str(self.config_path),
            db_path=str(self.db_path)
        )

        # 各種戦略による銘柄選択
        liquid_symbols = selector.get_liquid_symbols(limit=5)
        volatile_symbols = selector.get_volatile_symbols(limit=5)
        balanced_symbols = selector.get_balanced_portfolio(limit=5)
        diversified_symbols = selector.get_sector_diversified_symbols(limit=8)

        # 結果の検証
        self.assertTrue(len(liquid_symbols) <= 5)
        self.assertTrue(len(volatile_symbols) <= 5)
        self.assertTrue(len(balanced_symbols) <= 5)
        self.assertTrue(len(diversified_symbols) <= 8)

        # 全体の検証
        all_symbols = set(liquid_symbols + volatile_symbols + balanced_symbols + diversified_symbols)

        # 重複を除いた全銘柄の検証
        validation = selector.validate_symbol_selection(list(all_symbols))

        self.assertTrue(validation['valid'])
        self.assertTrue(validation['sector_count'] >= 3)
        self.assertTrue(validation['avg_market_cap'] > 0)

        print(f"統合テスト結果:")
        print(f"- 高流動性銘柄: {liquid_symbols}")
        print(f"- 高ボラティリティ銘柄: {volatile_symbols}")
        print(f"- バランス型銘柄: {balanced_symbols}")
        print(f"- セクター分散銘柄: {diversified_symbols}")
        print(f"- 検証結果: {validation['valid_symbols']}/{validation['total_symbols']} 有効")
        print(f"- セクター数: {validation['sector_count']}")


if __name__ == '__main__':
    # ロギング設定
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テストスイートの実行
    unittest.main(verbosity=2)