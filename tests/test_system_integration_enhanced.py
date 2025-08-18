#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
システム統合テスト（拡張版）

新しく実装した機能の統合テスト
"""

import pytest
import time
import threading
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.utils.memory_optimizer import (
    MemoryOptimizer, get_memory_optimizer, get_memory_stats
)
from src.day_trade.utils.error_handler import (
    ErrorHandler, get_error_handler, handle_errors, 
    safe_execute, CircuitBreaker, ErrorSeverity
)
from src.day_trade.utils.enhanced_logging import (
    EnhancedLogger, get_enhanced_logger, performance_timer
)


class TestMemoryOptimizer:
    """メモリ最適化システムのテスト"""
    
    def test_memory_stats_collection(self):
        """メモリ統計収集のテスト"""
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_stats()
        
        assert stats.total_mb > 0
        assert stats.available_mb > 0
        assert stats.usage_percent >= 0
        assert stats.usage_percent <= 100
        assert stats.process_mb > 0
    
    def test_garbage_collection(self):
        """ガベージコレクションのテスト"""
        optimizer = MemoryOptimizer()
        
        # 大量のオブジェクトを作成
        large_list = [i for i in range(100000)]
        del large_list
        
        # ガベージコレクション実行
        result = optimizer.force_garbage_collection()
        
        assert isinstance(result, dict)
        assert 'generation_0' in result
        assert 'generation_1' in result
        assert 'generation_2' in result
    
    def test_cache_registration(self):
        """キャッシュ登録のテスト"""
        optimizer = MemoryOptimizer()
        
        # テスト用キャッシュオブジェクト
        class TestCache:
            def __init__(self):
                self.data = {}
            
            def clear(self):
                self.data.clear()
        
        cache = TestCache()
        cache.data['test'] = 'value'
        
        optimizer.register_cached_object(cache)
        
        # キャッシュクリア実行
        optimizer.clear_caches()
        
        assert len(cache.data) == 0
    
    def test_memory_monitoring(self):
        """メモリ監視のテスト"""
        optimizer = MemoryOptimizer(
            warning_threshold=50.0,
            critical_threshold=90.0,
            cleanup_interval=1
        )
        
        # コールバック呼び出し確認用
        warning_called = threading.Event()
        
        def test_warning_callback(stats):
            warning_called.set()
        
        optimizer.register_warning_callback(test_warning_callback)
        
        # 短時間監視を開始
        optimizer.start_monitoring()
        time.sleep(2)  # 監視実行
        optimizer.stop_monitoring()
        
        # 監視が正常に動作したことを確認
        assert not optimizer.monitoring


class TestErrorHandler:
    """エラーハンドリングシステムのテスト"""
    
    def test_error_registration(self):
        """エラー登録のテスト"""
        handler = ErrorHandler()
        
        try:
            raise ValueError("テストエラー")
        except Exception as e:
            error_id = handler.register_error(
                e, 
                context={"test": True},
                severity=ErrorSeverity.MEDIUM
            )
        
        assert error_id in handler.error_history
        error_info = handler.error_history[error_id]
        assert error_info.error_type == "ValueError"
        assert error_info.message == "テストエラー"
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.context["test"] is True
    
    def test_retry_policy(self):
        """リトライポリシーのテスト"""
        handler = ErrorHandler()
        
        # ConnectionErrorのリトライポリシーをテスト
        conn_error = ConnectionError("接続エラー")
        policy = handler.get_retry_policy(conn_error)
        
        assert policy.max_retries == 5
        assert policy.base_delay == 2.0
        
        # リトライ判定テスト
        assert handler.should_retry(conn_error, 0) is True
        assert handler.should_retry(conn_error, 3) is True
        assert handler.should_retry(conn_error, 5) is False
    
    def test_handle_errors_decorator(self):
        """handle_errorsデコレータのテスト"""
        call_count = 0
        
        @handle_errors(
            retry=True,
            fallback_value="fallback",
            context={"test_function": True}
        )
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 最初の2回は失敗
                raise ConnectionError("テストエラー")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 3  # リトライにより3回呼ばれる
    
    def test_safe_execute(self):
        """safe_execute関数のテスト"""
        def failing_function():
            raise ValueError("テストエラー")
        
        def success_function():
            return "success"
        
        # エラー時のデフォルト値返却
        result = safe_execute(
            failing_function,
            default_value="default"
        )
        assert result == "default"
        
        # 成功時の通常値返却
        result = safe_execute(
            success_function,
            default_value="default"
        )
        assert result == "success"
    
    def test_circuit_breaker(self):
        """サーキットブレーカーのテスト"""
        failure_count = 0
        
        @CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        def test_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("テストエラー")
            return "success"
        
        # 最初の3回は失敗
        for _ in range(3):
            with pytest.raises(Exception):
                test_function()
        
        # 4回目はサーキットブレーカーが開く
        with pytest.raises(Exception, match="サーキットブレーカーが開いています"):
            test_function()
    
    def test_error_stats(self):
        """エラー統計のテスト"""
        handler = ErrorHandler()
        
        # 複数のエラーを登録
        errors = [
            ValueError("エラー1"),
            ConnectionError("エラー2"),
            ValueError("エラー3"),
        ]
        
        for error in errors:
            handler.register_error(error, severity=ErrorSeverity.HIGH)
        
        stats = handler.get_error_stats()
        
        assert stats["total_errors"] == 3
        assert stats["error_counts_by_type"]["ValueError"] == 2
        assert stats["error_counts_by_type"]["ConnectionError"] == 1
        assert stats["severity_counts"]["high"] == 3


class TestEnhancedLogging:
    """拡張ログシステムのテスト"""
    
    def test_logger_creation(self):
        """ロガー作成のテスト"""
        logger = EnhancedLogger(
            name="test_logger",
            log_dir="test_logs",
            async_logging=False  # テスト用に同期ログ使用
        )
        
        assert logger.name == "test_logger"
        assert logger.log_dir.name == "test_logs"
        assert len(logger.logger.handlers) > 0
    
    def test_metrics_logging(self):
        """メトリクス付きログのテスト"""
        logger = EnhancedLogger(
            name="test_metrics",
            async_logging=False
        )
        
        # メトリクス付きログ出力
        logger.log_with_metrics(
            logging.INFO,
            "テストメッセージ",
            context={"test": True},
            duration_ms=123.45,
            memory_mb=67.89
        )
        
        # メトリクス確認
        metrics = logger.get_metrics(1)
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.message == "テストメッセージ"
        assert metric.duration_ms == 123.45
        assert metric.memory_mb == 67.89
        assert metric.context["test"] is True
    
    def test_performance_timer_decorator(self):
        """パフォーマンスタイマーデコレータのテスト"""
        
        @performance_timer
        def test_function():
            time.sleep(0.1)  # 100ms待機
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # ログが記録されたことを確認
        logger = get_enhanced_logger()
        metrics = logger.get_metrics(1)
        
        # パフォーマンスログが記録されているはず
        perf_metrics = [m for m in metrics if "パフォーマンス" in m.message]
        assert len(perf_metrics) > 0
        
        perf_metric = perf_metrics[0]
        assert perf_metric.duration_ms >= 100  # 最低100ms
    
    def test_performance_stats(self):
        """パフォーマンス統計のテスト"""
        logger = EnhancedLogger(async_logging=False)
        
        # 複数のログを記録
        logger.log_with_metrics(logging.INFO, "メッセージ1", duration_ms=100)
        logger.log_with_metrics(logging.WARNING, "メッセージ2", duration_ms=200)
        logger.log_with_metrics(logging.ERROR, "メッセージ3", duration_ms=300)
        
        stats = logger.get_performance_stats()
        
        assert stats["total_logs_1h"] == 3
        assert stats["average_duration_ms"] == 200.0
        assert stats["max_duration_ms"] == 300.0
        assert stats["level_counts"]["INFO"] == 1
        assert stats["level_counts"]["WARNING"] == 1
        assert stats["level_counts"]["ERROR"] == 1


class TestSystemIntegration:
    """システム統合テスト"""
    
    def test_global_instances(self):
        """グローバルインスタンスの取得テスト"""
        # 各システムのグローバルインスタンスが正常に取得できることを確認
        memory_optimizer = get_memory_optimizer()
        error_handler = get_error_handler()
        enhanced_logger = get_enhanced_logger()
        
        assert isinstance(memory_optimizer, MemoryOptimizer)
        assert isinstance(error_handler, ErrorHandler)
        assert isinstance(enhanced_logger, EnhancedLogger)
        
        # 同じインスタンスが返されることを確認（シングルトン）
        assert get_memory_optimizer() is memory_optimizer
        assert get_error_handler() is error_handler
        assert get_enhanced_logger() is enhanced_logger
    
    def test_integration_workflow(self):
        """統合ワークフローのテスト"""
        # 実際のワークフローに近いテスト
        
        # 1. メモリ統計取得
        stats = get_memory_stats()
        assert stats.total_mb > 0
        
        # 2. エラー処理付き関数実行
        @handle_errors(fallback_value="error_fallback")
        def test_workflow():
            # 何らかの処理をシミュレート
            time.sleep(0.05)
            return "workflow_success"
        
        result = test_workflow()
        assert result == "workflow_success"
        
        # 3. パフォーマンス計測付き関数実行
        @performance_timer
        def performance_workflow():
            time.sleep(0.05)
            return "performance_success"
        
        result = performance_workflow()
        assert result == "performance_success"
        
        # 4. ログメトリクス確認
        logger = get_enhanced_logger()
        metrics = logger.get_metrics(1)
        assert len(metrics) > 0
    
    def test_error_recovery_integration(self):
        """エラー回復統合テスト"""
        attempt_count = 0
        
        @handle_errors(retry=True, fallback_value="recovered")
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count <= 2:
                raise ConnectionError("一時的なエラー")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3  # 3回目で成功
        
        # エラーハンドラーにエラーが記録されていることを確認
        handler = get_error_handler()
        stats = handler.get_error_stats()
        assert stats["total_errors"] >= 2  # 最低2回のエラーが記録される


if __name__ == "__main__":
    pytest.main([__file__, "-v"])