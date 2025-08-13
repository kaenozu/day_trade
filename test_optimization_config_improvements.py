#!/usr/bin/env python3
"""
Issues #634, #635 テストケース

optimization_strategy.pyのOptimizationConfig改善をテスト
"""

import sys
import json
import tempfile
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.core.optimization_strategy import OptimizationConfig, OptimizationLevel

def test_issue_634_consolidate_default_values():
    """Issue #634: デフォルト値統合テスト"""
    print("=== Issue #634: デフォルト値統合テスト ===")

    # get_default_values()メソッドのテスト
    defaults = OptimizationConfig.get_default_values()

    expected_keys = [
        'level', 'auto_fallback', 'performance_monitoring',
        'cache_enabled', 'parallel_processing', 'batch_size',
        'timeout_seconds', 'memory_limit_mb', 'ci_test_mode'
    ]

    all_present = True
    for key in expected_keys:
        if key in defaults:
            print(f"  [PASS] デフォルト値キー '{key}' が存在: {defaults[key]}")
        else:
            print(f"  [FAIL] デフォルト値キー '{key}' が見つかりません")
            all_present = False

    # from_env()でのデフォルト値使用確認
    config_env = OptimizationConfig.from_env()

    if (config_env.level == defaults['level'] and
        config_env.batch_size == defaults['batch_size'] and
        config_env.timeout_seconds == defaults['timeout_seconds']):
        print("  [PASS] from_env()でデフォルト値が統合使用されています")
    else:
        print("  [FAIL] from_env()でのデフォルト値統合に問題があります")

    if all_present:
        print("  [PASS] デフォルト値が統合されました")
    print()

def test_issue_635_type_conversion_from_file():
    """Issue #635: from_file型変換テスト"""
    print("=== Issue #635: from_file型変換テスト ===")

    # 文字列型の数値を含む設定ファイルを作成
    test_config = {
        "level": "optimized",
        "auto_fallback": "true",
        "performance_monitoring": "false",
        "cache_enabled": True,
        "parallel_processing": "1",
        "batch_size": "250",      # 文字列型の数値
        "timeout_seconds": "45",  # 文字列型の数値
        "memory_limit_mb": "1024", # 文字列型の数値
        "ci_test_mode": "0"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        temp_path = f.name

    try:
        # from_fileでの読み込みテスト
        config = OptimizationConfig.from_file(temp_path)

        # 型変換の確認
        type_tests = [
            ("level", config.level, OptimizationLevel, OptimizationLevel.OPTIMIZED),
            ("auto_fallback", config.auto_fallback, bool, True),
            ("performance_monitoring", config.performance_monitoring, bool, False),
            ("parallel_processing", config.parallel_processing, bool, True),
            ("batch_size", config.batch_size, int, 250),
            ("timeout_seconds", config.timeout_seconds, int, 45),
            ("memory_limit_mb", config.memory_limit_mb, int, 1024),
            ("ci_test_mode", config.ci_test_mode, bool, False),
        ]

        all_passed = True
        for field_name, actual_value, expected_type, expected_value in type_tests:
            if isinstance(actual_value, expected_type) and actual_value == expected_value:
                print(f"  [PASS] {field_name}: {actual_value} ({type(actual_value).__name__})")
            else:
                print(f"  [FAIL] {field_name}: 期待値 {expected_value} ({expected_type.__name__}), 実際 {actual_value} ({type(actual_value).__name__})")
                all_passed = False

        if all_passed:
            print("  [PASS] 全ての型変換が正常に完了しました")

    finally:
        Path(temp_path).unlink()

    print()

def test_invalid_config_handling():
    """無効な設定値のハンドリングテスト"""
    print("=== 無効な設定値ハンドリングテスト ===")

    # 無効な値を含む設定ファイル
    invalid_config = {
        "level": "invalid_level",
        "batch_size": "not_a_number",
        "timeout_seconds": -10,  # 負の値
        "memory_limit_mb": 99999999,  # 範囲外の値
        "auto_fallback": "maybe"  # 無効なbool値
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_config, f)
        temp_path = f.name

    try:
        config = OptimizationConfig.from_file(temp_path)
        defaults = OptimizationConfig.get_default_values()

        # デフォルト値へのフォールバック確認
        fallback_tests = [
            ("level", config.level, defaults['level']),
            ("batch_size", config.batch_size, defaults['batch_size']),
            ("timeout_seconds", config.timeout_seconds, 1),  # 最小値制限
            ("memory_limit_mb", config.memory_limit_mb, 16384),  # 最大値制限
            ("auto_fallback", config.auto_fallback, defaults['auto_fallback']),
        ]

        all_handled = True
        for field_name, actual_value, expected_fallback in fallback_tests:
            if actual_value == expected_fallback:
                print(f"  [PASS] {field_name}: 無効値が適切にフォールバック -> {actual_value}")
            else:
                print(f"  [FAIL] {field_name}: フォールバック失敗 期待 {expected_fallback}, 実際 {actual_value}")
                all_handled = False

        if all_handled:
            print("  [PASS] 無効な設定値が適切にハンドリングされました")

    finally:
        Path(temp_path).unlink()

    print()

def test_safe_conversion_methods():
    """安全な変換メソッドの単体テスト"""
    print("=== 安全な変換メソッドテスト ===")

    # _safe_str_conversion テスト
    str_tests = [
        (None, "default", "default"),
        ("test", "default", "test"),
        (123, "default", "123"),
        (True, "default", "True"),
    ]

    for value, default, expected in str_tests:
        result = OptimizationConfig._safe_str_conversion(value, default)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"  {status} str変換: {value} -> {result} (期待値: {expected})")

    # _safe_bool_conversion テスト
    bool_tests = [
        (None, False, False),
        (True, False, True),
        ("true", False, True),
        ("false", False, False),
        ("1", False, True),
        ("0", False, False),
        (1, False, True),
        (0, False, False),
        ("invalid", True, True),  # デフォルト値使用
    ]

    for value, default, expected in bool_tests:
        result = OptimizationConfig._safe_bool_conversion(value, default)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"  {status} bool変換: {value} -> {result} (期待値: {expected})")

    # _safe_int_conversion テスト
    int_tests = [
        (None, 10, None, None, 10),
        (42, 10, None, None, 42),
        ("100", 10, None, None, 100),
        ("50.0", 10, None, None, 50),
        (-5, 10, 0, 100, 0),    # 最小値制限
        (150, 10, 0, 100, 100), # 最大値制限
        ("invalid", 10, None, None, 10), # デフォルト値使用
    ]

    for value, default, min_val, max_val, expected in int_tests:
        result = OptimizationConfig._safe_int_conversion(value, default, min_val, max_val)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"  {status} int変換: {value} -> {result} (期待値: {expected})")

    print()

def test_integration():
    """統合テスト"""
    print("=== 統合テスト ===")

    try:
        # 環境変数設定
        os.environ['DAYTRADE_OPTIMIZATION_LEVEL'] = 'optimized'
        os.environ['DAYTRADE_BATCH_SIZE'] = '500'

        # from_envテスト
        config_env = OptimizationConfig.from_env()
        print(f"  環境変数config: level={config_env.level.value}, batch_size={config_env.batch_size}")

        # デフォルト設定テスト
        config_default = OptimizationConfig()
        defaults = OptimizationConfig.get_default_values()
        print(f"  デフォルトconfig: level={config_default.level.value}, batch_size={config_default.batch_size}")

        # 設定値の一貫性確認
        if (config_default.level == defaults['level'] and
            config_default.batch_size == defaults['batch_size']):
            print("  [PASS] デフォルト設定とget_default_values()の一貫性OK")
        else:
            print("  [FAIL] デフォルト設定の一貫性に問題があります")

        print("  [PASS] 統合テストが成功しました")

    finally:
        # 環境変数クリーンアップ
        os.environ.pop('DAYTRADE_OPTIMIZATION_LEVEL', None)
        os.environ.pop('DAYTRADE_BATCH_SIZE', None)

    print()

def run_all_tests():
    """全テストを実行"""
    print("optimization_strategy.py 改善テスト開始\\n")

    test_issue_634_consolidate_default_values()
    test_issue_635_type_conversion_from_file()
    test_invalid_config_handling()
    test_safe_conversion_methods()
    test_integration()

    print("全テスト完了")

if __name__ == "__main__":
    run_all_tests()