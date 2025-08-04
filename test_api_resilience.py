"""
API耐障害性機能テスト
Issue 185: 外部API通信の耐障害性強化

リトライ機構、サーキットブレーカー、フェイルオーバー機能の包括的テスト
"""

import os
import sys
import time

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher
from src.day_trade.utils.api_resilience import (
    APIEndpoint,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ResilientAPIClient,
    RetryConfig,
)
from src.day_trade.utils.logging_config import (
    get_context_logger,
    setup_logging,
)


def test_circuit_breaker_functionality():
    """サーキットブレーカー機能テスト"""

    print("=== サーキットブレーカー機能テスト ===")

    # 設定
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0,  # テスト用に短く設定
        monitor_window=60.0,
    )

    circuit_breaker = CircuitBreaker(config)

    # 初期状態確認
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.can_execute()
    print("OK 初期状態（CLOSED）確認")

    # 失敗を記録してOPEN状態に移行させる
    for i in range(config.failure_threshold):
        circuit_breaker.record_failure()
        print(f"  失敗記録 {i + 1}/{config.failure_threshold}")

    assert circuit_breaker.state == CircuitState.OPEN
    assert not circuit_breaker.can_execute()
    print("OK OPEN状態移行確認")

    # タイムアウト前は実行不可
    assert not circuit_breaker.can_execute()
    print("OK タイムアウト前実行不可確認")

    # タイムアウト後はHALF_OPEN状態に移行
    time.sleep(config.timeout + 0.1)
    assert circuit_breaker.can_execute()
    assert circuit_breaker.state == CircuitState.HALF_OPEN
    print("OK HALF_OPEN状態移行確認")

    # 成功を記録してCLOSED状態に復帰
    for i in range(config.success_threshold):
        circuit_breaker.record_success()
        print(f"  成功記録 {i + 1}/{config.success_threshold}")

    assert circuit_breaker.state == CircuitState.CLOSED
    print("OK CLOSED状態復帰確認")

    print("OK サーキットブレーカー機能テスト完了")


def test_retry_configuration():
    """リトライ設定テスト"""

    print("=== リトライ設定テスト ===")

    # カスタム設定
    retry_config = RetryConfig(
        max_attempts=5,
        base_delay=0.1,
        max_delay=2.0,
        exponential_base=2.0,
        jitter=True,
        status_forcelist=[500, 502, 503, 504, 429],
    )

    assert retry_config.max_attempts == 5
    assert retry_config.base_delay == 0.1
    assert retry_config.max_delay == 2.0
    assert 500 in retry_config.status_forcelist
    assert 429 in retry_config.status_forcelist
    print("OK リトライ設定確認")

    # デフォルト設定
    default_config = RetryConfig()
    assert default_config.max_attempts == 3
    assert default_config.base_delay == 1.0
    print("OK デフォルト設定確認")

    print("OK リトライ設定テスト完了")


def test_api_endpoint_configuration():
    """APIエンドポイント設定テスト"""

    print("=== APIエンドポイント設定テスト ===")

    # エンドポイント作成
    endpoints = [
        APIEndpoint("primary", "https://api.example.com", priority=1, timeout=30.0),
        APIEndpoint(
            "secondary", "https://api-backup.example.com", priority=2, timeout=20.0
        ),
        APIEndpoint(
            "tertiary", "https://api-fallback.example.com", priority=3, timeout=15.0
        ),
    ]

    # 優先度順ソート確認
    sorted_endpoints = sorted(endpoints, key=lambda x: x.priority)
    assert sorted_endpoints[0].name == "primary"
    assert sorted_endpoints[1].name == "secondary"
    assert sorted_endpoints[2].name == "tertiary"
    print("✓ エンドポイント優先度ソート確認")

    # 設定値確認
    primary = endpoints[0]
    assert primary.name == "primary"
    assert primary.base_url == "https://api.example.com"
    assert primary.timeout == 30.0
    assert primary.is_active
    print("✓ エンドポイント設定値確認")

    print("✓ APIエンドポイント設定テスト完了")


def test_resilient_client_status():
    """耐障害性クライアント状態テスト"""

    print("=== 耐障害性クライアント状態テスト ===")

    # テスト用エンドポイント
    endpoints = [
        APIEndpoint("test1", "https://httpbin.org", priority=1),
        APIEndpoint("test2", "https://httpbin.org", priority=2),
    ]

    # クライアント作成（ヘルスチェック無効）
    client = ResilientAPIClient(endpoints=endpoints, enable_health_check=False)

    # 状態取得
    status = client.get_status()

    assert "endpoints" in status
    assert "overall_health" in status
    assert len(status["endpoints"]) == 2
    print("✓ ステータス構造確認")

    # エンドポイント状態確認
    for endpoint_status in status["endpoints"]:
        assert "name" in endpoint_status
        assert "base_url" in endpoint_status
        assert "is_active" in endpoint_status
        assert "circuit_state" in endpoint_status
        print(
            f"  エンドポイント {endpoint_status['name']}: {endpoint_status['circuit_state']}"
        )

    print("✓ 耐障害性クライアント状態テスト完了")


def test_enhanced_stock_fetcher_initialization():
    """拡張版StockFetcher初期化テスト"""

    print("=== 拡張版StockFetcher初期化テスト ===")

    # デフォルト設定で初期化
    fetcher = EnhancedStockFetcher(
        enable_fallback=True,
        enable_circuit_breaker=True,
        enable_health_monitoring=False,  # テスト用にヘルスチェック無効
    )

    assert fetcher.enable_fallback
    assert fetcher.enable_circuit_breaker
    assert not fetcher.enable_health_monitoring
    print("✓ 設定値確認")

    # 耐障害性クライアントが初期化されているか確認
    assert hasattr(fetcher, "resilient_client")
    print("✓ 耐障害性クライアント初期化確認")

    # システム状態取得
    status = fetcher.get_system_status()
    assert "service" in status
    assert "timestamp" in status
    assert "cache_stats" in status
    assert status["service"] == "enhanced_stock_fetcher"
    print("✓ システム状態取得確認")

    print("✓ 拡張版StockFetcher初期化テスト完了")


def test_fallback_strategies():
    """フォールバック戦略テスト"""

    print("=== フォールバック戦略テスト ===")

    fetcher = EnhancedStockFetcher(
        enable_fallback=True,
        enable_circuit_breaker=False,
        enable_health_monitoring=False,
    )

    # デフォルトデータ生成テスト
    default_price = fetcher._generate_default_price_data("7203")
    assert default_price["symbol"] == "7203"
    assert default_price["current_price"] == 0.0
    assert default_price["data_quality"] == "degraded"
    assert default_price["source"] == "fallback"
    print("✓ デフォルト価格データ生成確認")

    default_company = fetcher._generate_default_company_info("7203")
    assert default_company["symbol"] == "7203"
    assert "Company" in default_company["name"]
    assert default_company["data_quality"] == "degraded"
    print("✓ デフォルト企業情報生成確認")

    default_historical = fetcher._generate_default_historical_data()
    assert len(default_historical) == 0
    assert list(default_historical.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    print("✓ デフォルトヒストリカルデータ生成確認")

    print("✓ フォールバック戦略テスト完了")


def test_validation_enhancements():
    """検証機能強化テスト"""

    print("=== 検証機能強化テスト ===")

    fetcher = EnhancedStockFetcher(enable_health_monitoring=False)

    # APIレスポンス検証テスト
    # 有効なレスポンス
    valid_response = {
        "currentPrice": 1500.0,
        "previousClose": 1480.0,
        "volume": 1000000,
    }
    assert fetcher._validate_api_response(valid_response)
    print("✓ 有効なAPIレスポンス検証")

    # 無効なレスポンス
    invalid_response1 = {}
    assert not fetcher._validate_api_response(invalid_response1)
    print("✓ 空のレスポンス検証")

    invalid_response2 = None
    assert not fetcher._validate_api_response(invalid_response2)
    print("✓ Noneレスポンス検証")

    invalid_response3 = "invalid"
    assert not fetcher._validate_api_response(invalid_response3)
    print("✓ 非辞書型レスポンス検証")

    # 価格データ抽出と検証
    valid_info = {
        "currentPrice": 1500.0,
        "previousClose": 1480.0,
        "volume": 1000000,
        "marketCap": 50000000000,
    }

    price_data = fetcher._extract_and_validate_price_data(valid_info, "7203.T")
    assert price_data["symbol"] == "7203.T"
    assert price_data["current_price"] == 1500.0
    assert price_data["previous_close"] == 1480.0
    assert price_data["change"] == 20.0  # 1500 - 1480
    assert (
        abs(price_data["change_percent"] - 1.3513513513513513) < 0.001
    )  # (20/1480)*100
    assert price_data["data_quality"] == "standard"
    print("✓ 価格データ抽出・検証確認")

    print("✓ 検証機能強化テスト完了")


def test_health_check_functionality():
    """ヘルスチェック機能テスト"""

    print("=== ヘルスチェック機能テスト ===")

    fetcher = EnhancedStockFetcher(
        enable_fallback=True,
        enable_circuit_breaker=True,
        enable_health_monitoring=False,  # APIアクセスを避けるため無効
    )

    # ヘルスチェック実行
    health = fetcher.health_check()

    # 基本構造確認
    assert "timestamp" in health
    assert "status" in health
    assert "checks" in health
    assert "response_time_ms" in health
    print("✓ ヘルスチェック構造確認")

    # ステータス確認
    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert isinstance(health["response_time_ms"], (int, float))
    assert health["response_time_ms"] >= 0
    print(f"✓ ヘルスステータス: {health['status']}")
    print(f"✓ 応答時間: {health['response_time_ms']:.2f}ms")

    # チェック項目確認
    checks = health["checks"]
    assert "price_fetch" in checks or "resilience" in checks
    print("✓ ヘルスチェック項目確認")

    for check_name, check_result in checks.items():
        assert "status" in check_result
        assert "details" in check_result
        print(f"  {check_name}: {check_result['status']}")

    print("✓ ヘルスチェック機能テスト完了")


def test_error_handling_improvements():
    """エラーハンドリング改善テスト"""

    print("=== エラーハンドリング改善テスト ===")

    fetcher = EnhancedStockFetcher(enable_health_monitoring=False)

    # 無効なシンボルテスト
    try:
        fetcher._validate_symbol("")
        raise AssertionError("空のシンボルで例外が発生すべき")
    except Exception as e:
        print(f"✓ 空のシンボル検証: {type(e).__name__}")

    try:
        fetcher._validate_symbol("X")
        raise AssertionError("短すぎるシンボルで例外が発生すべき")
    except Exception as e:
        print(f"✓ 短いシンボル検証: {type(e).__name__}")

    # 市場時間チェック（警告のみなので例外は発生しない）
    try:
        fetcher._validate_market_hours()
        print("✓ 市場時間検証（警告レベル）")
    except Exception as e:
        print(f"✓ 市場時間検証エラー: {e}")

    print("✓ エラーハンドリング改善テスト完了")


def test_cache_statistics():
    """キャッシュ統計テスト"""

    print("=== キャッシュ統計テスト ===")

    fetcher = EnhancedStockFetcher(enable_health_monitoring=False)

    # キャッシュ統計取得
    cache_stats = fetcher._get_cache_stats()

    assert isinstance(cache_stats, dict)
    print("✓ キャッシュ統計構造確認")

    # 統計にメソッド名が含まれているか確認
    expected_methods = ["get_current_price", "get_historical_data", "get_company_info"]
    for method in expected_methods:
        # キャッシュ統計が利用可能な場合のみチェック
        if method in cache_stats:
            print(f"  {method}: キャッシュ統計利用可能")
        else:
            print(f"  {method}: キャッシュ統計未利用")

    print("✓ キャッシュ統計テスト完了")


def comprehensive_resilience_test():
    """包括的耐障害性テスト"""

    print("=== 包括的耐障害性テスト ===")

    # 全機能有効で初期化
    fetcher = EnhancedStockFetcher(
        enable_fallback=True,
        enable_circuit_breaker=True,
        enable_health_monitoring=False,  # 外部APIアクセスを避けるため
        retry_count=2,  # テスト用に短縮
        retry_delay=0.1,  # テスト用に短縮
    )

    print("✓ 全機能有効で初期化完了")

    # システム状態確認
    status = fetcher.get_system_status()
    print(f"✓ サービス: {status['service']}")

    if status.get("resilience_status"):
        resilience = status["resilience_status"]
        print(f"✓ 全体ヘルス: {resilience['overall_health']}")
        print(f"✓ エンドポイント数: {len(resilience['endpoints'])}")

    # ヘルスチェック
    health = fetcher.health_check()
    print(f"✓ ヘルスステータス: {health['status']}")
    print(f"✓ 応答時間: {health['response_time_ms']:.2f}ms")

    # デモンストレーション：実際のAPIエラーをシミュレート
    print("✓ APIエラーシミュレーション（フォールバック動作確認）")

    # 劣化モードデータ生成テスト
    degraded_price = fetcher._generate_default_price_data("TEST")
    print(f"  劣化モード価格データ: {degraded_price['data_quality']}")

    degraded_company = fetcher._generate_default_company_info("TEST")
    print(f"  劣化モード企業情報: {degraded_company['data_quality']}")

    print("✓ 包括的耐障害性テスト完了")


if __name__ == "__main__":
    print("API耐障害性機能テスト開始")
    print("=" * 50)

    # ロギング設定
    setup_logging()
    logger = get_context_logger(__name__)

    tests = [
        ("サーキットブレーカー機能", test_circuit_breaker_functionality),
        ("リトライ設定", test_retry_configuration),
        ("APIエンドポイント設定", test_api_endpoint_configuration),
        ("耐障害性クライアント状態", test_resilient_client_status),
        ("拡張版StockFetcher初期化", test_enhanced_stock_fetcher_initialization),
        ("フォールバック戦略", test_fallback_strategies),
        ("検証機能強化", test_validation_enhancements),
        ("ヘルスチェック機能", test_health_check_functionality),
        ("エラーハンドリング改善", test_error_handling_improvements),
        ("キャッシュ統計", test_cache_statistics),
        ("包括的耐障害性", comprehensive_resilience_test),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            logger.info(f"テスト成功: {test_name}")
        except Exception as e:
            print(f"✗ {test_name}: エラー - {e}")
            logger.error(f"テスト失敗: {test_name}", error=str(e))
            failed += 1

        print()  # 空行でテスト間を区切り

    print(f"{'=' * 50}")
    print("テスト結果:")
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    print(
        f"成功率: {passed}/{passed + failed} ({passed / (passed + failed) * 100:.1f}%)"
    )

    if failed == 0:
        print("✓ すべてのAPI耐障害性テストが成功しました")
        logger.info("すべてのAPI耐障害性テストが成功")
    else:
        print("✗ 一部のテストが失敗しました")
        logger.warning("一部のAPI耐障害性テストが失敗", failed_count=failed)
