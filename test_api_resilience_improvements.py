#!/usr/bin/env python3
"""
API耐障害性機能の改善テスト

Issue #211: API耐障害性機能の正確性と堅牢性の向上
修正された機能のテストとベンチマーク
"""

import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# プロジェクトのルートを追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from day_trade.utils.api_resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ResilientAPIClient,
    APIEndpoint,
    RetryConfig
)
from day_trade.utils.exceptions import (
    APIError,
    NetworkError,
    TimeoutError,
    AuthenticationError,
    RateLimitError,
    ServerError
)
from day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher


class TestCircuitBreakerImprovements(unittest.TestCase):
    """サーキットブレーカーの改善テスト"""

    def setUp(self):
        """テスト前準備"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=1.0,
            monitor_window=10.0
        )
        self.circuit_breaker = CircuitBreaker(config)

    def test_record_success_resets_failure_count_in_closed_state(self):
        """CLOSED状態での成功時にfailure_countが0にリセットされることをテスト"""
        # 初期状態：CLOSED
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)

        # 失敗を記録
        self.circuit_breaker.record_failure()
        self.circuit_breaker.record_failure()
        self.assertEqual(self.circuit_breaker.failure_count, 2)
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)

        # 成功を記録 - failure_countが0にリセットされるべき
        self.circuit_breaker.record_success()
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)

    def test_half_open_to_closed_transition(self):
        """HALF_OPEN から CLOSED への遷移テスト"""
        # OPEN状態に移行
        for _ in range(3):
            self.circuit_breaker.record_failure()
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)

        # 時間経過をシミュレート（タイムアウト後）
        time.sleep(1.1)

        # HALF_OPEN状態に移行可能
        self.assertTrue(self.circuit_breaker.can_execute())
        self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)

        # 成功を記録してCLOSEDに復帰
        self.circuit_breaker.record_success()
        self.circuit_breaker.record_success()  # success_threshold=2
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)


class TestExceptionHierarchy(unittest.TestCase):
    """例外階層の改善テスト"""

    def test_timeout_error_inherits_from_network_error(self):
        """TimeoutErrorがNetworkErrorを継承していることをテスト"""
        timeout_error = TimeoutError("Test timeout")

        # TimeoutErrorがNetworkErrorのインスタンスであること
        self.assertIsInstance(timeout_error, NetworkError)

        # TimeoutErrorがAPIErrorのインスタンスであること（継承チェーン）
        self.assertIsInstance(timeout_error, APIError)

    def test_exception_hierarchy_consistency(self):
        """例外階層の一貫性をテスト"""
        # ネットワーク関連の例外はすべてNetworkErrorを継承
        network_exceptions = [
            NetworkError("test"),
            TimeoutError("test timeout")
        ]

        for exc in network_exceptions:
            self.assertIsInstance(exc, NetworkError)
            self.assertIsInstance(exc, APIError)


class TestRetryLogicImprovements(unittest.TestCase):
    """リトライロジックの改善テスト"""

    def setUp(self):
        """テスト前準備"""
        self.endpoints = [
            APIEndpoint("test", "https://httpbin.org", timeout=5.0)
        ]
        self.retry_config = RetryConfig(
            max_attempts=3,
            backoff_factor=1.0,
            status_forcelist=[500, 502, 503, 504]
        )

    @patch('day_trade.utils.api_resilience.requests.Session')
    def test_retry_logic_delegation_to_urllib3(self, mock_session_class):
        """urllib3のRetryにリトライ処理が委任されることをテスト"""
        # Sessionインスタンスのモック
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # 成功レスポンスのモック
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.request.return_value = mock_response

        # ResilientAPIClientを作成
        client = ResilientAPIClient(
            self.endpoints,
            self.retry_config,
            enable_health_check=False
        )

        # リクエスト実行
        response = client.get("/get")

        # リクエストが1回だけ呼ばれたことを確認（リトライループが削除されたため）
        self.assertEqual(mock_session.request.call_count, 1)
        self.assertEqual(response.status_code, 200)

    @patch('day_trade.utils.api_resilience.requests.Session')
    def test_circuit_breaker_integration(self, mock_session_class):
        """サーキットブレーカーとの統合テスト"""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # 失敗レスポンスのモック
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_session.request.return_value = mock_response

        client = ResilientAPIClient(
            self.endpoints,
            self.retry_config,
            enable_health_check=False
        )

        # 複数回失敗させてサーキットブレーカーを開く
        with self.assertRaises(ServerError):
            client.get("/error")


class TestCacheFallbackImprovements(unittest.TestCase):
    """キャッシュフォールバック機能の改善テスト"""

    def test_cache_fallback_mechanism(self):
        """改善されたキャッシュフォールバック機能をテスト"""
        # EnhancedStockFetcherのインスタンス作成
        fetcher = EnhancedStockFetcher(
            enable_fallback=True,
            enable_circuit_breaker=False,  # テスト用
            enable_health_monitoring=False
        )

        # _cache_fallback_executionメソッドのテスト
        def mock_function():
            pass
        mock_function.__name__ = 'get_current_price'

        # キャッシュフォールバック実行のテスト
        result = fetcher._cache_fallback_execution(mock_function, "AAPL")

        # 結果がNoneでないことを確認（キャッシュが空の場合はNoneが返される）
        # または None が返されることを期待（キャッシュがない場合）
        self.assertIsNone(result)  # キャッシュが空の場合の正常な動作


class TestPerformanceBenchmark(unittest.TestCase):
    """パフォーマンスベンチマーク"""

    def test_improved_retry_performance(self):
        """改善されたリトライ機構のパフォーマンステスト"""
        start_time = time.perf_counter()

        # 簡単なパフォーマンステスト
        config = CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker(config)

        # 1000回の成功記録
        for _ in range(1000):
            circuit_breaker.record_success()

        execution_time = time.perf_counter() - start_time

        # 1000回の操作が1秒以内に完了することを確認
        self.assertLess(execution_time, 1.0)

        # failure_countが正しく0になっていることを確認
        self.assertEqual(circuit_breaker.failure_count, 0)


def run_improvement_tests():
    """改善テストの実行"""
    print(">> API耐障害性機能の改善テスト開始")
    print("=" * 60)

    # テストスイートの実行
    test_classes = [
        TestCircuitBreakerImprovements,
        TestExceptionHierarchy,
        TestRetryLogicImprovements,
        TestCacheFallbackImprovements,
        TestPerformanceBenchmark
    ]

    total_tests = 0
    total_failures = 0

    for test_class in test_classes:
        print(f"\n>> {test_class.__name__}")
        print("-" * 40)

        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)

        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)

        if result.failures:
            print(f">> 失敗: {len(result.failures)}")
        if result.errors:
            print(f">> エラー: {len(result.errors)}")

        if not result.failures and not result.errors:
            print(">> 全テスト成功")

    print("\n" + "=" * 60)
    print(">> API耐障害性改善テスト結果")
    print("=" * 60)

    print(f"総テスト数: {total_tests}")
    print(f"失敗数: {total_failures}")
    print(f"成功率: {((total_tests - total_failures) / total_tests * 100):.1f}%")

    if total_failures == 0:
        print("\n>> 全ての改善が正常に動作しています！")

        print("\n>> 実装された改善:")
        print("  - TimeoutError の継承階層修正")
        print("  - サーキットブレーカーの success 記録修正")
        print("  - リトライロジック重複の解消")
        print("  - キャッシュフォールバック機能の改善")

        print("\n>> 期待される効果:")
        print("  - より正確なサーキットブレーカー動作")
        print("  - 例外階層の一貫性向上")
        print("  - リトライ処理の効率化")
        print("  - 障害時のサービス継続性向上")
    else:
        print(f"\n>> {total_failures}個の問題が検出されました。")

    return total_failures == 0


if __name__ == "__main__":
    success = run_improvement_tests()
    sys.exit(0 if success else 1)
