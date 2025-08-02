"""
構造化ロギング機能テスト
Issue 186: 構造化ロギングの導入
"""

import os
import sys
import time
from datetime import datetime

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.day_trade.utils.logging_config import (
    setup_logging,
    setup_logging_with_alerts,
    get_logger,
    get_context_logger,
    log_business_event,
    log_performance_metric,
    log_api_call,
    log_database_operation,
    log_security_event,
    log_user_action,
    log_system_health,
    enhanced_log_error_with_context,
    enhanced_log_performance_metric
)

def test_basic_logging():
    """基本ロギング機能テスト"""

    print("=== 基本ロギング機能テスト ===")

    # ロギング設定
    setup_logging()

    # 基本ロガー取得
    logger = get_logger("test_module")

    # 各ログレベルのテスト
    logger.debug("デバッグメッセージ", module="test", function="test_basic_logging")
    logger.info("情報メッセージ", status="testing", progress=25)
    logger.warning("警告メッセージ", issue="minor_issue", action="continue")
    logger.error("エラーメッセージ", error_code="E001", retry_count=1)

    print("✓ 基本ロギング機能テスト完了")

def test_context_logging():
    """コンテキストロギング機能テスト"""

    print("=== コンテキストロギング機能テスト ===")

    # コンテキスト付きロガー
    logger = get_context_logger("test_context",
                               component="trading_system",
                               version="1.0.0")

    # 基本コンテキスト付きロギング
    logger.info("取引処理開始", symbol="7203", quantity=100, price=2500)

    # 動的コンテキスト追加
    trade_logger = logger.bind(trade_id="T001", user_id="user123")
    trade_logger.info("取引実行", action="buy", amount=250000)
    trade_logger.info("取引完了", result="success", execution_time=0.15)

    print("✓ コンテキストロギング機能テスト完了")

def test_business_event_logging():
    """ビジネスイベントロギングテスト"""

    print("=== ビジネスイベントロギングテスト ===")

    # 各種ビジネスイベント
    log_business_event("trade_executed",
                      symbol="7203",
                      quantity=100,
                      price=2500,
                      trade_type="buy",
                      commission=250)

    log_business_event("portfolio_updated",
                      symbol="7203",
                      old_quantity=0,
                      new_quantity=100,
                      portfolio_value=250000)

    log_business_event("signal_generated",
                      symbol="7203",
                      signal_type="BUY",
                      confidence=85.0,
                      strategy="sma_crossover")

    print("✓ ビジネスイベントロギングテスト完了")

def test_performance_logging():
    """パフォーマンスロギングテスト"""

    print("=== パフォーマンスロギングテスト ===")

    # 処理時間測定のシミュレーション
    start_time = time.time()
    time.sleep(0.1)  # 処理のシミュレーション
    execution_time = (time.time() - start_time) * 1000

    # パフォーマンスメトリクス記録
    log_performance_metric("signal_generation_time", execution_time, "ms",
                          symbols_processed=5,
                          indicators_calculated=10)

    log_performance_metric("api_response_time", 250, "ms",
                          api_endpoint="/api/stock_data",
                          success=True)

    log_performance_metric("memory_usage", 45.6, "percentage",
                          component="data_processor")

    # 高負荷アラート用（閾値テスト）
    enhanced_log_performance_metric("response_time", 6000, "ms")  # 閾値超過
    enhanced_log_performance_metric("memory_usage", 85.0, "percentage")  # 閾値超過

    print("✓ パフォーマンスロギングテスト完了")

def test_system_logging():
    """システム関連ロギングテスト"""

    print("=== システム関連ロギングテスト ===")

    # API呼び出しログ
    log_api_call("yahoo_finance", "GET", "https://query1.finance.yahoo.com/v7/finance/quote",
                status_code=200, response_time=180, symbols=["7203", "9984"])

    # データベース操作ログ
    log_database_operation("INSERT", "trades",
                         rows_affected=1,
                         execution_time=45,
                         query_size=256)

    # システムヘルス情報
    log_system_health("database", "healthy",
                     connection_count=5,
                     response_time=12,
                     cpu_usage=25.3)

    log_system_health("trading_engine", "operational",
                     active_strategies=3,
                     processed_signals=15,
                     memory_usage=67.8)

    print("✓ システム関連ロギングテスト完了")

def test_security_logging():
    """セキュリティロギングテスト"""

    print("=== セキュリティロギングテスト ===")

    # ユーザーアクション
    log_user_action("login", "user123",
                   ip_address="192.168.1.100",
                   user_agent="DayTradeApp/1.0")

    log_user_action("trade_execution", "user123",
                   symbol="7203",
                   action="buy",
                   amount=250000)

    # セキュリティイベント
    log_security_event("failed_login_attempt", "warning",
                      ip_address="10.0.0.1",
                      attempts=3,
                      user="unknown")

    log_security_event("suspicious_activity", "critical",
                      user_id="user456",
                      activity="large_trade",
                      amount=10000000,
                      pattern="unusual")

    print("✓ セキュリティロギングテスト完了")

def test_error_logging():
    """エラーロギングテスト"""

    print("=== エラーロギングテスト ===")

    # 通常のエラーログ
    try:
        raise ValueError("テスト用エラー: 無効な銘柄コード")
    except ValueError as e:
        enhanced_log_error_with_context(e, {
            "operation": "validate_symbol",
            "symbol": "INVALID",
            "user_input": "INVALID123",
            "validation_rules": ["length", "format", "exists"]
        })

    # ビジネスロジックエラー
    try:
        raise RuntimeError("売却数量が保有数量を超過")
    except RuntimeError as e:
        enhanced_log_error_with_context(e, {
            "operation": "sell_stock",
            "symbol": "7203",
            "requested_quantity": 200,
            "available_quantity": 100,
            "user_id": "user123"
        })

    print("✓ エラーロギングテスト完了")

def test_alert_functionality():
    """アラート機能テスト"""

    print("=== アラート機能テスト ===")

    # アラート機能付きロギング設定
    os.environ["ENABLE_LOGGING_ALERTS"] = "true"
    os.environ["ALERT_ERROR_THRESHOLD"] = "3"
    os.environ["ALERT_RESPONSE_TIME_THRESHOLD"] = "1000"

    setup_logging_with_alerts()

    # エラー閾値テスト
    for i in range(5):
        try:
            raise Exception(f"テストエラー {i+1}")
        except Exception as e:
            enhanced_log_error_with_context(e, {"test_iteration": i+1})

    # パフォーマンス閾値テスト
    enhanced_log_performance_metric("response_time", 1500, "ms")  # 閾値超過
    enhanced_log_performance_metric("response_time", 500, "ms")   # 正常

    print("✓ アラート機能テスト完了")

def test_json_output():
    """JSON出力テスト"""

    print("=== JSON出力テスト ===")

    # 本番環境設定でJSONフォーマット
    os.environ["ENVIRONMENT"] = "production"
    setup_logging()

    logger = get_context_logger("json_test")

    # 複雑なデータ構造のログ
    complex_data = {
        "trade_data": {
            "symbol": "7203",
            "quantity": 100,
            "prices": [2500, 2510, 2505],
            "metadata": {
                "strategy": "sma_crossover",
                "confidence": 85.5,
                "market_conditions": ["bullish", "high_volume"]
            }
        },
        "execution_context": {
            "timestamp": datetime.now().isoformat(),
            "user_id": "user123",
            "session_id": "sess_abc123"
        }
    }

    logger.info("複雑なデータ構造ログテスト", **complex_data)

    # 開発環境に戻す
    os.environ["ENVIRONMENT"] = "development"
    setup_logging()

    print("✓ JSON出力テスト完了")

def demonstrate_logging_hierarchy():
    """ロギング階層の実演"""

    print("=== ロギング階層実演 ===")

    # 異なるコンポーネントのロガー
    components = [
        "trading_engine",
        "data_fetcher",
        "signal_generator",
        "portfolio_manager",
        "risk_manager"
    ]

    for component in components:
        logger = get_context_logger(f"day_trade.{component}")
        logger.info(f"{component}が初期化されました",
                   component=component,
                   status="initialized",
                   startup_time=0.5)

    # 階層的な処理フロー
    main_logger = get_context_logger("day_trade.main")
    main_logger.info("自動取引処理開始", session_id="session_001")

    # サブコンポーネントの処理
    signal_logger = main_logger.bind(component="signal_generator")
    signal_logger.info("シグナル生成開始", symbols=["7203", "9984"])
    signal_logger.info("シグナル生成完了", generated_signals=2)

    trade_logger = main_logger.bind(component="trading_engine")
    trade_logger.info("取引実行開始", signals_to_process=2)
    trade_logger.info("取引実行完了", successful_trades=2, failed_trades=0)

    main_logger.info("自動取引処理完了", duration=2.5, total_trades=2)

    print("✓ ロギング階層実演完了")

if __name__ == "__main__":
    print("構造化ロギング機能テスト開始")
    print("=" * 50)

    tests = [
        ("基本ロギング機能", test_basic_logging),
        ("コンテキストロギング機能", test_context_logging),
        ("ビジネスイベントロギング", test_business_event_logging),
        ("パフォーマンスロギング", test_performance_logging),
        ("システム関連ロギング", test_system_logging),
        ("セキュリティロギング", test_security_logging),
        ("エラーロギング", test_error_logging),
        ("アラート機能", test_alert_functionality),
        ("JSON出力", test_json_output),
        ("ロギング階層実演", demonstrate_logging_hierarchy),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: エラー - {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"テスト結果:")
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    print(f"成功率: {passed}/{passed+failed} ({passed/(passed+failed)*100:.1f}%)")

    if failed == 0:
        print("✓ すべての構造化ロギングテストが成功しました")
    else:
        print("✗ 一部のテストが失敗しました")
