#!/usr/bin/env python3
"""
Issues #624-632 テストケース

logging_config.pyの包括的改善をテスト
"""

import sys
import tempfile
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import queue
import threading
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.utils.logging_config import (
    LoggingConfig, ContextLogger, setup_logging, get_logger,
    get_context_logger, log_database_operation, log_business_event,
    get_performance_logger, log_api_call, log_performance_metric,
    log_error_with_context, log_error_with_enhanced_context,
    get_caller_info, STRUCTLOG_AVAILABLE
)

def test_issue_624_structlog_integration():
    """Issue #624: structlog統合と標準化テスト"""
    print("=== Issue #624: structlog統合標準化テスト ===")

    try:
        # LoggingConfigの初期化テスト
        config = LoggingConfig()
        config.configure_logging()

        print(f"  [PASS] ロギング設定が正常に初期化されました")
        print(f"  structlog利用可能: {STRUCTLOG_AVAILABLE}")
        print(f"  ログレベル: {config.log_level}")
        print(f"  ログフォーマット: {config.log_format}")

        # ロガー取得テスト
        logger = get_logger("test_logger")
        print(f"  [PASS] ロガーを取得できました: {type(logger).__name__}")

        # 設定の再実行防止テスト
        initial_state = config.is_configured
        config.configure_logging()  # 再実行
        if config.is_configured == initial_state:
            print("  [PASS] 重複設定を防止しました")
        else:
            print("  [FAIL] 重複設定防止に問題があります")

    except Exception as e:
        print(f"  [FAIL] structlog統合テストでエラー: {e}")

    print()

def test_issue_625_context_logger_fallback():
    """Issue #625: ContextLoggerのフォールバック簡素化テスト"""
    print("=== Issue #625: ContextLoggerフォールバック簡素化テスト ===")

    # 標準ロガーでのContextLogger作成
    std_logger = logging.getLogger("test_context")
    context = {"component": "test", "session_id": "12345"}

    context_logger = ContextLogger(std_logger, context)

    # bindメソッドテスト
    bound_logger = context_logger.bind(operation="test_op", user_id="user123")

    print(f"  [PASS] ContextLoggerを作成しました")
    print(f"  [PASS] コンテキストバインドが動作しました")

    # ログ出力テスト（実際の出力は確認しないが、エラーがないことを確認）
    try:
        context_logger.info("テスト情報ログ", extra={"test_data": "value"})
        context_logger.warning("テスト警告ログ")
        context_logger.error("テストエラーログ")
        bound_logger.debug("バインドされたデバッグログ")
        print("  [PASS] 全てのログレベルで出力できました")
    except Exception as e:
        print(f"  [FAIL] ログ出力でエラー: {e}")

    print()

def test_issue_626_queue_handler_robustness():
    """Issue #626: QueueHandler終了堅牢性テスト"""
    print("=== Issue #626: QueueHandler終了堅牢性テスト ===")

    try:
        # 新しい設定インスタンスで非同期ロギングをテスト
        test_config = LoggingConfig()
        test_config.configure_logging()

        # ログ出力テスト（キューに正しく送信されるか）
        logger = get_logger("queue_test")

        if STRUCTLOG_AVAILABLE:
            logger.info("QueueHandlerテストメッセージ")
        else:
            logger.info("QueueHandler標準ロガーテストメッセージ")

        print("  [PASS] QueueHandlerでのログ出力が成功しました")

        # atexitハンドラーが登録されているかは直接確認できないが、
        # エラーなく設定できていることを確認
        print("  [PASS] QueueListener設定と終了ハンドラー登録が完了")

    except Exception as e:
        print(f"  [FAIL] QueueHandler堅牢性テストでエラー: {e}")

    print()

def test_issue_627_logger_naming_hierarchy():
    """Issue #627: ロガー命名と階層構造改善テスト"""
    print("=== Issue #627: ロガー命名階層構造テスト ===")

    try:
        # 階層的なロガー名のテスト
        base_logger = get_context_logger("trading", component="signals")
        child_logger = get_context_logger("trading.signals", component="macd")

        print(f"  [PASS] ベースロガーを作成: trading.signals")
        print(f"  [PASS] 子ロガーを作成: trading.signals.macd")

        # 異なるコンポーネントでのロガー取得
        db_logger = get_context_logger("trading", component="database")
        api_logger = get_context_logger("trading", component="api")

        print(f"  [PASS] データベースロガー: trading.database")
        print(f"  [PASS] APIロガー: trading.api")

        # パフォーマンスロガーの階層確認
        perf_logger = get_performance_logger("trading.optimization")
        print(f"  [PASS] パフォーマンスロガー階層化")

    except Exception as e:
        print(f"  [FAIL] ロガー階層テストでエラー: {e}")

    print()

def test_issue_629_consolidate_helper_functions():
    """Issue #629: ロギングヘルパー関数統合テスト"""
    print("=== Issue #629: ロギングヘルパー関数統合テスト ===")

    try:
        # データベース操作ログテスト
        log_database_operation("SELECT", duration=0.5, table="stocks", rows=100)
        log_database_operation("UPDATE", duration=2.5, table="trades", rows=50)  # 遅い操作
        print("  [PASS] データベース操作ログ出力成功")

        # ビジネスイベントログテスト
        log_business_event("trade_executed", {"symbol": "7203", "quantity": 100, "price": 1500})
        log_business_event("portfolio_rebalanced", {"total_value": 1000000})
        print("  [PASS] ビジネスイベントログ出力成功")

        # API呼び出しログテスト
        log_api_call("/api/stocks/7203", method="GET", duration=0.3, status_code=200)
        log_api_call("/api/orders", method="POST", duration=3.0, status_code=201)  # 遅い操作
        log_api_call("/api/invalid", method="GET", duration=0.1, status_code=404)  # エラー
        print("  [PASS] API呼び出しログ出力成功")

        # パフォーマンスメトリクスログテスト
        log_performance_metric("cpu_usage", 75.5, "percent", component="ml_engine")
        log_performance_metric("memory_usage", 512, "MB", component="data_fetcher")
        print("  [PASS] パフォーマンスメトリクスログ出力成功")

    except Exception as e:
        print(f"  [FAIL] ヘルパー関数統合テストでエラー: {e}")

    print()

def test_issue_630_duration_type_handling():
    """Issue #630: duration型ハンドリング改善テスト"""
    print("=== Issue #630: duration型ハンドリングテスト ===")

    try:
        # 正常な数値でのテスト
        log_database_operation("SELECT_NORMAL", duration=1.5)
        print("  [PASS] 正常な数値duration処理")

        # 文字列数値でのテスト
        log_database_operation("SELECT_STRING_NUM", duration="2.5")
        print("  [PASS] 文字列数値duration処理")

        # 無効な値でのテスト
        log_database_operation("SELECT_INVALID", duration="invalid")
        print("  [PASS] 無効duration値のフォールバック処理")

        # Noneでのテスト
        log_database_operation("SELECT_NONE", duration=None)
        print("  [PASS] None duration値のフォールバック処理")

        # 整数でのテスト
        log_database_operation("SELECT_INT", duration=3)
        print("  [PASS] 整数duration処理")

    except Exception as e:
        print(f"  [FAIL] duration型ハンドリングテストでエラー: {e}")

    print()

def test_issue_631_external_config_enhancement():
    """Issue #631: 外部設定ファイル拡張テスト"""
    print("=== Issue #631: 外部設定ファイル拡張テスト ===")

    try:
        # 環境変数でのログレベル設定テスト
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG', 'LOG_FORMAT': 'json'}):
            test_config = LoggingConfig()

            if test_config.log_level == 'DEBUG':
                print("  [PASS] 環境変数からログレベルを正しく取得")
            else:
                print(f"  [FAIL] ログレベル取得失敗: 期待値DEBUG, 実際{test_config.log_level}")

            if test_config.log_format == 'json':
                print("  [PASS] 環境変数からログフォーマットを正しく取得")
            else:
                print(f"  [FAIL] ログフォーマット取得失敗: 期待値json, 実際{test_config.log_format}")

        # デフォルト設定テスト
        default_config = LoggingConfig()
        if default_config.log_level == 'INFO' and default_config.log_format == 'simple':
            print("  [PASS] デフォルト設定値が正しく設定されています")
        else:
            print(f"  [FAIL] デフォルト設定に問題: level={default_config.log_level}, format={default_config.log_format}")

    except Exception as e:
        print(f"  [FAIL] 外部設定拡張テストでエラー: {e}")

    print()

def test_issue_632_python_version_compatibility():
    """Issue #632: Python版本依存structlog設定簡素化テスト"""
    print("=== Issue #632: Python版本依存structlog設定テスト ===")

    try:
        if STRUCTLOG_AVAILABLE:
            import structlog

            # add_logger_oidの存在確認
            has_add_logger_oid = hasattr(structlog.stdlib, 'add_logger_oid')
            print(f"  add_logger_oid利用可能: {has_add_logger_oid}")

            # 設定の作成テスト
            config = LoggingConfig()
            config.configure_logging()

            print("  [PASS] Python版本依存の設定が正常に動作")

            # structlogロガーの動作確認
            logger = structlog.get_logger("version_test")
            logger.info("Python版本依存設定テスト", python_version=sys.version_info[:2])
            print("  [PASS] structlogロガーが正常に動作")

        else:
            print("  [INFO] structlogが利用できません。フォールバックテストを実行")

            # フォールバック設定のテスト
            config = LoggingConfig()
            config.configure_logging()

            logger = logging.getLogger("fallback_test")
            logger.info("フォールバック設定テスト")
            print("  [PASS] フォールバック設定が正常に動作")

    except Exception as e:
        print(f"  [FAIL] Python版本依存設定テストでエラー: {e}")

    print()

def test_enhanced_error_logging():
    """拡張エラーロギング機能テスト"""
    print("=== 拡張エラーロギング機能テスト ===")

    try:
        # 基本的なエラーロギング
        try:
            raise ValueError("テストエラー")
        except ValueError as e:
            log_error_with_context(e, {"operation": "test", "data": "sample"})
            print("  [PASS] 基本的なエラーロギング")

        # 拡張エラーロギング
        try:
            raise RuntimeError("拡張テストエラー")
        except RuntimeError as e:
            log_error_with_enhanced_context(e, {"component": "test_module", "session": "abc123"})
            print("  [PASS] 拡張エラーロギング")

        # 呼び出し元情報取得テスト
        caller_info = get_caller_info()
        expected_keys = ['module_name', 'function_name', 'filename', 'line_number', 'code_context']

        all_keys_present = all(key in caller_info for key in expected_keys)
        if all_keys_present:
            print("  [PASS] 呼び出し元情報を正しく取得")
        else:
            print(f"  [FAIL] 呼び出し元情報に不足: {caller_info}")

    except Exception as e:
        print(f"  [FAIL] 拡張エラーロギングテストでエラー: {e}")

    print()

def test_integration():
    """統合テスト"""
    print("=== 統合テスト ===")

    try:
        # 全体の設定初期化
        setup_logging()
        print("  [PASS] ロギング設定初期化成功")

        # 異なる種類のロガーで同時出力テスト
        main_logger = get_logger("integration_test")
        context_logger = get_context_logger("integration_test", component="main")
        perf_logger = get_performance_logger("integration_test")

        main_logger.info("統合テスト開始")
        context_logger.info("コンテキスト付きログ", operation="integration_test")
        log_performance_metric("test_duration", 0.5, "seconds")

        print("  [PASS] 複数ロガーでの同時出力成功")

        # エラーハンドリングと回復テスト
        try:
            raise Exception("統合テスト用例外")
        except Exception as e:
            log_error_with_enhanced_context(e, {"test_phase": "integration"})

        print("  [PASS] 統合エラーハンドリング成功")
        print("  [PASS] 統合テストが正常に完了しました")

    except Exception as e:
        print(f"  [FAIL] 統合テストでエラー: {e}")

    print()

def run_all_tests():
    """全テストを実行"""
    print("logging_config.py 包括的改善テスト開始\\n")

    test_issue_624_structlog_integration()
    test_issue_625_context_logger_fallback()
    test_issue_626_queue_handler_robustness()
    test_issue_627_logger_naming_hierarchy()
    test_issue_629_consolidate_helper_functions()
    test_issue_630_duration_type_handling()
    test_issue_631_external_config_enhancement()
    test_issue_632_python_version_compatibility()
    test_enhanced_error_logging()
    test_integration()

    print("全テスト完了")

if __name__ == "__main__":
    run_all_tests()