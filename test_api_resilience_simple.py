"""
API耐障害性機能テスト（シンプル版）
Issue 185: 外部API通信の耐障害性強化
"""

import os
import sys
import time

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher
from src.day_trade.utils.api_resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
)
from src.day_trade.utils.logging_config import get_context_logger, setup_logging


def test_circuit_breaker():
    """サーキットブレーカーテスト"""
    print("=== サーキットブレーカーテスト ===")

    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0,
        monitor_window=60.0
    )

    circuit_breaker = CircuitBreaker(config)

    # 初期状態
    assert circuit_breaker.state == CircuitState.CLOSED
    print("OK 初期状態CLOSED")

    # 失敗を記録
    for _i in range(config.failure_threshold):
        circuit_breaker.record_failure()

    assert circuit_breaker.state == CircuitState.OPEN
    print("OK OPEN状態移行")

    # タイムアウト後
    time.sleep(config.timeout + 0.1)
    assert circuit_breaker.can_execute()
    assert circuit_breaker.state == CircuitState.HALF_OPEN
    print("OK HALF_OPEN状態移行")

    # 成功記録
    for _i in range(config.success_threshold):
        circuit_breaker.record_success()

    assert circuit_breaker.state == CircuitState.CLOSED
    print("OK CLOSED状態復帰")

    return True


def test_retry_config():
    """リトライ設定テスト"""
    print("=== リトライ設定テスト ===")

    retry_config = RetryConfig(
        max_attempts=5,
        base_delay=0.1,
        max_delay=2.0,
        exponential_base=2.0
    )

    assert retry_config.max_attempts == 5
    assert retry_config.base_delay == 0.1
    print("OK リトライ設定確認")

    return True


def test_enhanced_fetcher():
    """拡張フェッチャーテスト"""
    print("=== 拡張フェッチャーテスト ===")

    fetcher = EnhancedStockFetcher(
        enable_fallback=True,
        enable_circuit_breaker=True,
        enable_health_monitoring=False
    )

    assert fetcher.enable_fallback
    assert fetcher.enable_circuit_breaker
    print("OK 初期化設定")

    # システム状態
    status = fetcher.get_system_status()
    assert "service" in status
    assert status["service"] == "enhanced_stock_fetcher"
    print("OK システム状態取得")

    # デフォルトデータ生成
    default_price = fetcher._generate_default_price_data("TEST")
    assert default_price["data_quality"] == "degraded"
    print("OK デフォルトデータ生成")

    return True


def test_validation():
    """検証機能テスト"""
    print("=== 検証機能テスト ===")

    fetcher = EnhancedStockFetcher(enable_health_monitoring=False)

    # 有効なレスポンス
    valid_response = {
        "currentPrice": 1500.0,
        "previousClose": 1480.0,
        "volume": 1000000
    }
    assert fetcher._validate_api_response(valid_response)
    print("OK 有効レスポンス検証")

    # 無効なレスポンス
    assert not fetcher._validate_api_response({})
    assert not fetcher._validate_api_response(None)
    print("OK 無効レスポンス検証")

    # 価格データ抽出
    price_data = fetcher._extract_and_validate_price_data(valid_response, "TEST.T")
    assert price_data["symbol"] == "TEST.T"
    assert price_data["current_price"] == 1500.0
    assert price_data["data_quality"] == "standard"
    print("OK 価格データ抽出")

    return True


def test_health_check():
    """ヘルスチェックテスト"""
    print("=== ヘルスチェックテスト ===")

    fetcher = EnhancedStockFetcher(enable_health_monitoring=False)

    health = fetcher.health_check()

    assert "timestamp" in health
    assert "status" in health
    assert "checks" in health
    assert "response_time_ms" in health
    print("OK ヘルスチェック構造")

    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    print(f"OK ヘルスステータス: {health['status']}")

    return True


def run_all_tests():
    """全テスト実行"""
    print("API耐障害性機能テスト開始")
    print("=" * 50)

    setup_logging()
    logger = get_context_logger(__name__)

    tests = [
        ("サーキットブレーカー", test_circuit_breaker),
        ("リトライ設定", test_retry_config),
        ("拡張フェッチャー", test_enhanced_fetcher),
        ("検証機能", test_validation),
        ("ヘルスチェック", test_health_check),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"PASS {test_name}")
            else:
                failed += 1
                print(f"FAIL {test_name}")
        except Exception as e:
            failed += 1
            print(f"ERROR {test_name}: {e}")
            logger.error(f"テスト失敗: {test_name}", error=str(e))

        print()

    print("=" * 50)
    print(f"結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("すべてのテストが成功しました")
        logger.info("すべてのAPI耐障害性テストが成功")
        return True
    else:
        print("一部のテストが失敗しました")
        logger.warning("一部のAPI耐障害性テストが失敗", failed_count=failed)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
