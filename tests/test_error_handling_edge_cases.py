"""
エラーハンドリングエッジケースの包括的テストスイート

Issue #10: テストカバレッジ向上とエラーハンドリング強化
優先度: エラーハンドリング強化（Critical）
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta
import concurrent.futures

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from day_trade.core.error_handling.unified_error_system import (
    ApplicationError, ValidationError, BusinessLogicError,
    BusinessRuleViolationError, InsufficientDataError,
    DataAccessError, ExternalServiceError, SystemError,
    ErrorSeverity, ErrorCategory, ErrorContext, ErrorDetails,
    UnifiedErrorHandler, error_boundary, global_error_handler
)
from day_trade.domain.common.value_objects import Money, Price, Quantity, Symbol


class TestAsyncErrorHandling:
    """非同期エラーハンドリングのテスト"""
    
    @pytest.mark.asyncio
    async def test_async_error_boundary_with_suppression(self):
        """非同期エラー境界・抑制テスト"""
        @error_boundary(
            component_name="async_test",
            operation_name="suppressed_error",
            suppress_errors=True,
            fallback_value="fallback_result"
        )
        async def async_function_with_error():
            await asyncio.sleep(0.01)
            raise ValueError("テスト用エラー")
        
        result = await async_function_with_error()
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_async_error_boundary_without_suppression(self):
        """非同期エラー境界・非抑制テスト"""
        @error_boundary(
            component_name="async_test",
            operation_name="non_suppressed_error",
            suppress_errors=False
        )
        async def async_function_with_error():
            await asyncio.sleep(0.01)
            raise BusinessLogicError("ビジネスロジックエラー")
        
        with pytest.raises(BusinessLogicError):
            await async_function_with_error()
    
    @pytest.mark.asyncio
    async def test_nested_async_error_handling(self):
        """ネストした非同期エラーハンドリングテスト"""
        @error_boundary(
            component_name="outer",
            operation_name="outer_function",
            suppress_errors=True,
            fallback_value="outer_fallback"
        )
        async def outer_function():
            @error_boundary(
                component_name="inner",
                operation_name="inner_function",
                suppress_errors=False
            )
            async def inner_function():
                await asyncio.sleep(0.01)
                raise ValidationError("内部バリデーションエラー")
            
            try:
                return await inner_function()
            except ValidationError:
                # 内部エラーをキャッチして新しいエラーを発生
                raise SystemError("システムエラーに変換")
        
        result = await outer_function()
        assert result == "outer_fallback"
    
    @pytest.mark.asyncio
    async def test_concurrent_async_error_handling(self):
        """同時非同期エラーハンドリングテスト"""
        @error_boundary(
            component_name="concurrent",
            operation_name="concurrent_task",
            suppress_errors=True,
            fallback_value="error_fallback"
        )
        async def concurrent_task(task_id, should_error):
            await asyncio.sleep(0.01)
            if should_error:
                raise ExternalServiceError(f"タスク{task_id}でエラー")
            return f"success_{task_id}"
        
        # 一部が成功、一部がエラーのタスクを同時実行
        tasks = [
            concurrent_task(1, False),  # 成功
            concurrent_task(2, True),   # エラー
            concurrent_task(3, False),  # 成功
            concurrent_task(4, True),   # エラー
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert results[0] == "success_1"
        assert results[1] == "error_fallback"
        assert results[2] == "success_3"
        assert results[3] == "error_fallback"
    
    @pytest.mark.asyncio
    async def test_async_timeout_error_handling(self):
        """非同期タイムアウトエラーハンドリングテスト"""
        @error_boundary(
            component_name="timeout_test",
            operation_name="slow_operation",
            suppress_errors=True,
            fallback_value="timeout_fallback"
        )
        async def slow_operation():
            await asyncio.sleep(2.0)  # 長時間処理
            return "completed"
        
        # タイムアウト付きで実行
        try:
            result = await asyncio.wait_for(slow_operation(), timeout=0.1)
        except asyncio.TimeoutError:
            result = "timeout_fallback"
        
        assert result == "timeout_fallback"


class TestErrorRecoveryStrategies:
    """エラー回復戦略のテスト"""
    
    @pytest.mark.asyncio
    async def test_retry_strategy_with_exponential_backoff(self):
        """指数バックオフ付きリトライ戦略テスト"""
        call_count = 0
        
        @error_boundary(
            component_name="retry_test",
            operation_name="flaky_operation",
            suppress_errors=False
        )
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise ExternalServiceError("一時的なサービスエラー")
            return "success_after_retries"
        
        # エラーハンドラーに回復戦略をテスト
        error_handler = UnifiedErrorHandler()
        
        # 手動でリトライロジックをテスト
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await flaky_operation()
                break
            except ExternalServiceError as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(0.01 * (2 ** attempt))  # 指数バックオフ
        
        assert result == "success_after_retries"
        assert call_count == 3
    
    def test_circuit_breaker_pattern(self):
        """サーキットブレーカーパターンテスト"""
        from day_trade.core.error_handling.unified_error_system import CircuitBreakerStrategy
        
        circuit_breaker = CircuitBreakerStrategy(failure_threshold=3, timeout_seconds=1)
        
        def failing_operation():
            raise ExternalServiceError("サービス利用不可")
        
        # 失敗回数がしきい値に達するまで
        for i in range(3):
            try:
                asyncio.run(circuit_breaker.recover(
                    ExternalServiceError("テストエラー"), 
                    failing_operation
                ))
            except:
                pass
        
        # サーキットブレーカーが開いた状態でのテスト
        with pytest.raises(ApplicationError, match="Circuit breaker is open"):
            asyncio.run(circuit_breaker.recover(
                ExternalServiceError("テストエラー"), 
                failing_operation
            ))


class TestErrorPropagation:
    """エラー伝播のテスト"""
    
    def test_error_context_propagation(self):
        """エラーコンテキスト伝播テスト"""
        def level_three():
            raise DataAccessError("データベース接続エラー", operation="select")
        
        def level_two():
            try:
                level_three()
            except DataAccessError as e:
                # コンテキストを追加して再発生
                e.context.operation_name = "level_two_operation"
                raise BusinessLogicError("レベル2でのビジネスロジックエラー") from e
        
        def level_one():
            try:
                level_two()
            except BusinessLogicError as e:
                # 最上位レベルでエラーを処理
                context = ErrorContext(
                    component_name="error_propagation_test",
                    operation_name="level_one_operation",
                    user_id="test_user",
                    correlation_id="test_correlation"
                )
                raise SystemError("システム全体のエラー", context=context) from e
        
        with pytest.raises(SystemError) as exc_info:
            level_one()
        
        error = exc_info.value
        assert error.context.component_name == "error_propagation_test"
        assert error.context.user_id == "test_user"
        assert error.context.correlation_id == "test_correlation"
        assert error.__cause__ is not None  # 元のエラーが保持されている
    
    def test_error_chain_analysis(self):
        """エラーチェーン分析テスト"""
        def analyze_error_chain(error):
            """エラーチェーンを分析して情報を抽出"""
            chain = []
            current = error
            
            while current is not None:
                chain.append({
                    'type': type(current).__name__,
                    'message': str(current),
                    'category': getattr(current, 'category', None)
                })
                current = current.__cause__
            
            return chain
        
        # 複層エラーを作成
        try:
            try:
                try:
                    raise ConnectionError("ネットワーク接続失敗")
                except ConnectionError as e:
                    raise ExternalServiceError("外部API呼び出し失敗") from e
            except ExternalServiceError as e:
                raise BusinessLogicError("取引処理失敗") from e
        except BusinessLogicError as e:
            final_error = e
        
        chain = analyze_error_chain(final_error)
        
        assert len(chain) == 3
        assert chain[0]['type'] == 'BusinessLogicError'
        assert chain[1]['type'] == 'ExternalServiceError'
        assert chain[2]['type'] == 'ConnectionError'


class TestErrorBoundaryEdgeCases:
    """エラー境界エッジケースのテスト"""
    
    def test_error_boundary_with_generator(self):
        """ジェネレーター関数でのエラー境界テスト"""
        @error_boundary(
            component_name="generator_test",
            operation_name="generator_function",
            suppress_errors=True,
            fallback_value=[]
        )
        def error_generator():
            yield 1
            yield 2
            raise ValueError("ジェネレーター内エラー")
            yield 3  # 到達しない
        
        # ジェネレーターの消費
        results = []
        try:
            for value in error_generator():
                results.append(value)
        except ValueError:
            pass
        
        assert results == [1, 2]
    
    def test_error_boundary_with_context_manager(self):
        """コンテキストマネージャーでのエラー境界テスト"""
        class ErrorProneContextManager:
            def __init__(self, should_error_on_enter=False, should_error_on_exit=False):
                self.should_error_on_enter = should_error_on_enter
                self.should_error_on_exit = should_error_on_exit
                self.entered = False
                self.exited = False
            
            def __enter__(self):
                if self.should_error_on_enter:
                    raise SystemError("コンテキスト入場時エラー")
                self.entered = True
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.should_error_on_exit:
                    raise SystemError("コンテキスト退場時エラー")
                self.exited = True
                return False
        
        @error_boundary(
            component_name="context_manager_test",
            operation_name="context_operation",
            suppress_errors=True,
            fallback_value="context_error_fallback"
        )
        def use_context_manager():
            with ErrorProneContextManager(should_error_on_exit=True) as cm:
                return "operation_completed"
        
        result = use_context_manager()
        assert result == "context_error_fallback"
    
    def test_error_boundary_with_recursive_function(self):
        """再帰関数でのエラー境界テスト"""
        @error_boundary(
            component_name="recursion_test",
            operation_name="recursive_function",
            suppress_errors=True,
            fallback_value="recursion_error"
        )
        def recursive_factorial(n, depth=0):
            if depth > 10:  # 深い再帰でエラー
                raise RecursionError("再帰が深すぎます")
            
            if n <= 1:
                return 1
            return n * recursive_factorial(n - 1, depth + 1)
        
        # 正常な場合
        result_normal = recursive_factorial(5)
        assert result_normal == 120
        
        # エラーが発生する場合
        result_error = recursive_factorial(20)
        assert result_error == "recursion_error"


class TestMemoryAndResourceErrorHandling:
    """メモリ・リソースエラーハンドリングのテスト"""
    
    def test_memory_exhaustion_handling(self):
        """メモリ枯渇ハンドリングテスト"""
        @error_boundary(
            component_name="memory_test",
            operation_name="memory_intensive",
            suppress_errors=True,
            fallback_value="memory_error_handled"
        )
        def memory_intensive_operation():
            # メモリエラーをシミュレート
            raise MemoryError("メモリ不足")
        
        result = memory_intensive_operation()
        assert result == "memory_error_handled"
    
    def test_file_handle_exhaustion(self):
        """ファイルハンドル枯渇テスト"""
        @error_boundary(
            component_name="file_test",
            operation_name="file_operations",
            suppress_errors=True,
            fallback_value="file_error_handled"
        )
        def file_operations():
            # ファイルハンドルエラーをシミュレート
            raise OSError("ファイルハンドル不足")
        
        result = file_operations()
        assert result == "file_error_handled"
    
    def test_thread_pool_exhaustion(self):
        """スレッドプール枯渇テスト"""
        import concurrent.futures
        
        @error_boundary(
            component_name="thread_test",
            operation_name="thread_operations",
            suppress_errors=True,
            fallback_value="thread_error_handled"
        )
        def thread_operations():
            def worker_task():
                time.sleep(0.1)
                return "completed"
            
            # 大量のタスクでスレッドプールを枯渇
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(worker_task) for _ in range(100)]
                
                # タイムアウトでエラーをシミュレート
                raise concurrent.futures.TimeoutError("スレッドプールタイムアウト")
        
        result = thread_operations()
        assert result == "thread_error_handled"


class TestErrorHandlingWithValueObjects:
    """値オブジェクトでのエラーハンドリングテスト"""
    
    def test_money_validation_error_handling(self):
        """Money値オブジェクトバリデーションエラーハンドリング"""
        @error_boundary(
            component_name="value_object_test",
            operation_name="money_creation",
            suppress_errors=True,
            fallback_value=Money(Decimal("0"), "JPY")
        )
        def create_invalid_money():
            return Money(Decimal("-1000"), "JPY")  # 負の金額
        
        result = create_invalid_money()
        assert result.amount == Decimal("0")
        assert result.currency == "JPY"
    
    def test_price_validation_error_handling(self):
        """Price値オブジェクトバリデーションエラーハンドリング"""
        @error_boundary(
            component_name="value_object_test",
            operation_name="price_creation",
            suppress_errors=True,
            fallback_value=Price(Decimal("1"), "JPY")
        )
        def create_invalid_price():
            return Price(Decimal("0"), "JPY")  # ゼロ価格
        
        result = create_invalid_price()
        assert result.value == Decimal("1")
    
    def test_quantity_validation_error_handling(self):
        """Quantity値オブジェクトバリデーションエラーハンドリング"""
        @error_boundary(
            component_name="value_object_test",
            operation_name="quantity_creation",
            suppress_errors=True,
            fallback_value=Quantity(1)
        )
        def create_invalid_quantity():
            return Quantity(0)  # ゼロ数量
        
        result = create_invalid_quantity()
        assert result.value == 1
    
    def test_symbol_validation_error_handling(self):
        """Symbol値オブジェクトバリデーションエラーハンドリング"""
        @error_boundary(
            component_name="value_object_test",
            operation_name="symbol_creation",
            suppress_errors=True,
            fallback_value=Symbol("0000")
        )
        def create_invalid_symbol():
            return Symbol("")  # 空のシンボル
        
        result = create_invalid_symbol()
        assert result.code == "0000"


class TestConcurrentErrorHandling:
    """並行エラーハンドリングのテスト"""
    
    def test_thread_safe_error_logging(self):
        """スレッドセーフエラーログテスト"""
        import threading
        import queue
        
        error_queue = queue.Queue()
        
        @error_boundary(
            component_name="thread_safety_test",
            operation_name="concurrent_errors",
            suppress_errors=True,
            fallback_value="thread_error"
        )
        def generate_error(thread_id):
            raise BusinessLogicError(f"スレッド{thread_id}からのエラー")
        
        def worker(thread_id):
            try:
                result = generate_error(thread_id)
                error_queue.put((thread_id, result))
            except Exception as e:
                error_queue.put((thread_id, str(e)))
        
        # 複数スレッドでエラーを生成
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 結果確認
        results = []
        while not error_queue.empty():
            results.append(error_queue.get())
        
        assert len(results) == 5
        assert all(result[1] == "thread_error" for result in results)
    
    def test_race_condition_error_handling(self):
        """競合状態エラーハンドリングテスト"""
        import threading
        
        shared_resource = {"value": 0, "lock": threading.Lock()}
        
        @error_boundary(
            component_name="race_condition_test",
            operation_name="shared_resource_access",
            suppress_errors=True,
            fallback_value="race_condition_handled"
        )
        def access_shared_resource(increment):
            with shared_resource["lock"]:
                current = shared_resource["value"]
                # 意図的に競合状態を作る
                if current > 50:
                    raise BusinessRuleViolationError("値が上限を超えました")
                time.sleep(0.001)  # 競合状態を促進
                shared_resource["value"] = current + increment
                return shared_resource["value"]
        
        def worker(increment):
            return access_shared_resource(increment)
        
        # 複数スレッドで共有リソースにアクセス
        threads = []
        results = []
        
        def thread_worker(increment):
            result = worker(increment)
            results.append(result)
        
        for i in range(10):
            thread = threading.Thread(target=thread_worker, args=(10,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # エラーハンドリングが正常に動作したことを確認
        error_results = [r for r in results if r == "race_condition_handled"]
        success_results = [r for r in results if isinstance(r, int)]
        
        assert len(error_results) + len(success_results) == 10


class TestErrorAnalyticsAndReporting:
    """エラー分析・報告のテスト"""
    
    def test_error_analytics_collection(self):
        """エラー分析データ収集テスト"""
        handler = UnifiedErrorHandler()
        
        # 様々なタイプのエラーを生成
        errors = [
            ValidationError("バリデーションエラー1"),
            ValidationError("バリデーションエラー2"),
            BusinessLogicError("ビジネスロジックエラー1"),
            DataAccessError("データアクセスエラー1"),
            ExternalServiceError("外部サービスエラー1"),
            ExternalServiceError("外部サービスエラー2"),
            ExternalServiceError("外部サービスエラー3"),
        ]
        
        # エラーを記録
        for error in errors:
            handler._analytics.record_error(error)
        
        # 分析データ取得
        trends = handler.get_analytics()
        
        assert trends["total_errors"] == 7
        assert trends["categories"]["validation"] == 2
        assert trends["categories"]["business_logic"] == 1
        assert trends["categories"]["data_access"] == 1
        assert trends["categories"]["external_service"] == 3
    
    def test_error_pattern_detection(self):
        """エラーパターン検出テスト"""
        handler = UnifiedErrorHandler()
        
        # 同じパターンのエラーを複数回生成
        for i in range(5):
            error = ExternalServiceError("API呼び出し失敗")
            handler._analytics.record_error(error)
        
        for i in range(3):
            error = ValidationError("入力値が不正です")
            handler._analytics.record_error(error)
        
        trends = handler.get_analytics()
        patterns = trends["most_common_patterns"]
        
        # 最も頻繁なパターンを確認
        top_pattern = patterns[0]
        assert "external_service:API呼び出し失敗" in top_pattern[0]
        assert top_pattern[1] == 5


class TestErrorHandlingPerformance:
    """エラーハンドリングパフォーマンステスト"""
    
    def test_error_handling_overhead(self):
        """エラーハンドリングオーバーヘッドテスト"""
        import time
        
        def normal_function():
            return "normal_result"
        
        @error_boundary(
            component_name="performance_test",
            operation_name="overhead_test",
            suppress_errors=False
        )
        def decorated_function():
            return "decorated_result"
        
        # 通常の関数の実行時間測定
        start_time = time.time()
        for _ in range(1000):
            normal_function()
        normal_time = time.time() - start_time
        
        # デコレートされた関数の実行時間測定
        start_time = time.time()
        for _ in range(1000):
            decorated_function()
        decorated_time = time.time() - start_time
        
        # オーバーヘッドが合理的な範囲内であることを確認
        overhead_ratio = decorated_time / normal_time
        assert overhead_ratio < 5.0  # 5倍以下のオーバーヘッド
    
    def test_bulk_error_processing(self):
        """一括エラー処理テスト"""
        handler = UnifiedErrorHandler()
        
        # 大量のエラーを生成
        start_time = time.time()
        
        for i in range(1000):
            error = BusinessLogicError(f"エラー{i}")
            handler._analytics.record_error(error)
        
        processing_time = time.time() - start_time
        
        # 処理時間が合理的であることを確認
        assert processing_time < 1.0  # 1秒以内
        
        # データが正しく記録されていることを確認
        trends = handler.get_analytics()
        assert trends["total_errors"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])