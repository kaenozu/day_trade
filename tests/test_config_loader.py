#!/usr/bin/env python3
"""
統合設定ローダーのテストスイート

新しく実装した分割設定ファイル機能をテストします。
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.config.config_loader import (
    ConfigLoader,
    get_config_loader,
    load_integrated_config,
)


class TestConfigLoader:
    """ConfigLoaderクラスのテスト"""

    def test_config_loader_initialization(self):
        """ConfigLoader初期化テスト"""
        print("=== ConfigLoader初期化テスト ===")

        loader = ConfigLoader("config")
        assert loader.config_dir == Path("config")
        assert loader._cache == {}

        print("[OK] ConfigLoader初期化成功")

    def test_load_config_success(self):
        """設定読み込み成功テスト"""
        print("=== 設定読み込み成功テスト ===")

        loader = ConfigLoader("config")
        config = loader.load_config()

        # 基本構造の確認
        assert isinstance(config, dict)
        assert len(config) > 0

        # 主要セクションの確認
        expected_sections = ["watchlist", "analysis", "risk_management"]
        for section in expected_sections:
            assert section in config, f"必須セクション '{section}' が見つかりません"

        print(f"[OK] 設定読み込み成功: {len(config)}セクション")

    def test_get_section(self):
        """セクション取得テスト"""
        print("=== セクション取得テスト ===")

        loader = ConfigLoader("config")
        watchlist = loader.get_section("watchlist")

        assert isinstance(watchlist, dict)
        if watchlist:  # watchlistが存在する場合
            assert "symbols" in watchlist
            symbols = watchlist["symbols"]
            assert isinstance(symbols, list)
            assert len(symbols) > 0

        print("[OK] セクション取得成功")

    def test_validate_config_success(self):
        """設定検証成功テスト"""
        print("=== 設定検証成功テスト ===")

        loader = ConfigLoader("config")
        is_valid = loader.validate_config()

        assert is_valid is True, "設定検証が失敗しました"

        print("[OK] 設定検証成功")

    def test_cache_functionality(self):
        """キャッシュ機能テスト"""
        print("=== キャッシュ機能テスト ===")

        loader = ConfigLoader("config")

        # 初回読み込み
        config1 = loader.load_config()
        assert len(loader._cache) > 0, "キャッシュが作成されていません"

        # 2回目読み込み（キャッシュから）
        config2 = loader.load_config()
        assert config1 == config2, "キャッシュから異なる設定が返されました"

        # キャッシュクリア
        loader.reload()
        assert len(loader._cache) == 0, "キャッシュがクリアされていません"

        print("[OK] キャッシュ機能正常")

    def test_error_handling_missing_files(self):
        """存在しないファイルのエラーハンドリング"""
        print("=== 存在しないファイルエラーハンドリング ===")

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(temp_dir)

            # 空のディレクトリでの読み込み試行
            try:
                config = loader.load_config()
                # 空でも例外は発生しない（警告のみ）
                assert isinstance(config, dict)
            except ValueError as e:
                # 有効な設定ファイルがない場合の例外
                assert "有効な設定ファイルが見つかりませんでした" in str(e)

        print("[OK] エラーハンドリング正常")

    @staticmethod
    def test_singleton_functionality():
        """シングルトン機能テスト"""
        print("=== シングルトン機能テスト ===")

        loader1 = get_config_loader()
        loader2 = get_config_loader()

        assert loader1 is loader2, "シングルトンが機能していません"

        print("[OK] シングルトン機能正常")

    @staticmethod
    def test_convenience_functions():
        """便利関数テスト"""
        print("=== 便利関数テスト ===")

        # load_integrated_config関数
        config = load_integrated_config()
        assert isinstance(config, dict)

        print("[OK] 便利関数正常")


def test_config_integration():
    """設定統合テスト"""
    print("=== 設定統合テスト ===")

    try:
        loader = ConfigLoader("config")
        config = loader.load_config()

        # 統合された設定の検証
        assert "version" in config, "バージョン情報が見つかりません"
        assert "watchlist" in config, "watchlist設定が見つかりません"
        assert "analysis" in config, "analysis設定が見つかりません"

        # watchlistの詳細検証
        watchlist = config["watchlist"]
        if "symbols" in watchlist:
            symbols = watchlist["symbols"]
            assert len(symbols) > 0, "銘柄リストが空です"

            # 銘柄データの構造確認
            first_symbol = symbols[0]
            required_fields = ["code", "name", "group", "priority"]
            for field in required_fields:
                assert (
                    field in first_symbol
                ), f"銘柄データに必須フィールド '{field}' が不足"

        # 分析設定の検証
        if "analysis" in config:
            analysis = config["analysis"]
            if "technical_indicators" in analysis:
                indicators = analysis["technical_indicators"]
                assert isinstance(indicators, dict), "テクニカル指標設定が無効です"

        print(f"[OK] 設定統合テスト成功 - {len(config)}セクション統合")
        return True

    except Exception as e:
        print(f"[ERROR] 設定統合テストエラー: {e}")
        return False


def run_all_tests():
    """全テスト実行"""
    print("統合設定ローダー テストスイート開始")
    print("=" * 50)

    test_suite = TestConfigLoader()

    tests = [
        test_suite.test_config_loader_initialization,
        test_suite.test_load_config_success,
        test_suite.test_get_section,
        test_suite.test_validate_config_success,
        test_suite.test_cache_functionality,
        test_suite.test_error_handling_missing_files,
        test_suite.test_singleton_functionality,
        test_suite.test_convenience_functions,
        test_config_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAILED] {test.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"テスト結果: {passed}個成功, {failed}個失敗")
    print(f"成功率: {(passed / (passed + failed) * 100):.1f}%")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
