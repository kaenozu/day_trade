"""
次世代システム包括的テスト

全アーキテクチャコンポーネントの統合テスト、単体テスト、パフォーマンステスト。
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# テスト対象のインポート
from day_trade.core.unified_system_error import (
    UnifiedSystemError, ValidationError, SecurityError, ErrorRecoveryAction
)
from day_trade.core.common.service_instance import (
    BaseServiceInstance, ServiceStatus, ServiceType
)
from day_trade.core.security.security_manager import (
    SecurityManager, InputValidator, SecurityLevel
)
from day_trade.core.performance.performance_optimizer import (
    PerformanceOptimizer, PerformanceMonitor, MetricType, PerformanceMetric
)

# テストユーティリティ
class TestUtils:
    """テストユーティリティ"""
    
    @staticmethod
    def create_test_service_instance(service_name: str = "test-service") -> BaseServiceInstance:
        """テスト用サービスインスタンス作成"""
        return BaseServiceInstance(
            service_name=service_name,
            host="localhost",
            port=8080,
            version="1.0.0",
            status=ServiceStatus.HEALTHY
        )
    
    @staticmethod
    def create_test_performance_metric(
        metric_type: MetricType = MetricType.RESPONSE_TIME,
        value: float = 100.0
    ) -> PerformanceMetric:
        """テスト用パフォーマンスメトリクス作成"""
        return PerformanceMetric(
            metric_type=metric_type,
            value=value,
            component="test-component"
        )


# ================================
# 統一エラーシステムテスト
# ================================

class TestUnifiedErrorSystem:
    """統一エラーシステムテスト"""
    
    def test_unified_system_error_creation(self):
        """統一システムエラー作成テスト"""
        error = UnifiedSystemError(
            message="テストエラー",
            error_code="TEST_ERROR_001"
        )
        
        assert error.message == "テストエラー"
        assert error.error_code == "TEST_ERROR_001"
        assert isinstance(error.timestamp, datetime)
        
        error_dict = error.to_dict()
        assert error_dict['message'] == "テストエラー"
        assert error_dict['error_code'] == "TEST_ERROR_001"
    
    def test_validation_error(self):
        """検証エラーテスト"""
        error = ValidationError(
            message="無効な入力です",
            field_name="email"
        )
        
        assert isinstance(error, UnifiedSystemError)
        assert error.details['field_name'] == "email"
        assert error.recovery_action == ErrorRecoveryAction.LOG_ONLY
    
    def test_security_error(self):
        """セキュリティエラーテスト"""
        error = SecurityError(
            message="不正アクセス検出",
            security_context="login_attempt"
        )
        
        assert isinstance(error, UnifiedSystemError)
        assert error.details['security_context'] == "login_attempt"
        assert error.should_alert() == True


# ================================
# サービスインスタンステスト
# ================================

class TestServiceInstance:
    """サービスインスタンステスト"""
    
    def test_service_instance_creation(self):
        """サービスインスタンス作成テスト"""
        instance = TestUtils.create_test_service_instance("test-service")
        
        assert instance.service_name == "test-service"
        assert instance.host == "localhost"
        assert instance.port == 8080
        assert instance.status == ServiceStatus.HEALTHY
        assert instance.is_healthy()
    
    def test_service_instance_serialization(self):
        """サービスインスタンスシリアライゼーションテスト"""
        instance = TestUtils.create_test_service_instance()
        
        # 辞書変換テスト
        instance_dict = instance.to_dict()
        assert instance_dict['service_name'] == instance.service_name
        assert instance_dict['host'] == instance.host
        assert instance_dict['port'] == instance.port
        
        # JSON変換テスト
        instance_json = instance.to_json()
        assert isinstance(instance_json, str)
        
        # 辞書からの復元テスト
        restored_instance = BaseServiceInstance.from_dict(instance_dict)
        assert restored_instance.service_name == instance.service_name
        assert restored_instance.host == instance.host
    
    def test_service_instance_health_check(self):
        """サービスインスタンスヘルスチェックテスト"""
        instance = TestUtils.create_test_service_instance()
        
        # 健全性確認
        assert instance.is_healthy()
        
        # ステータス変更
        instance.status = ServiceStatus.UNHEALTHY
        assert not instance.is_healthy()
        
        # ハートビート更新
        instance.status = ServiceStatus.HEALTHY
        instance.update_heartbeat()
        assert instance.is_healthy()


# ================================
# セキュリティシステムテスト
# ================================

class TestSecuritySystem:
    """セキュリティシステムテスト"""
    
    def test_input_validator_email(self):
        """入力検証 - メールアドレステスト"""
        # 有効なメールアドレス
        assert InputValidator.validate_email("test@example.com") == True
        assert InputValidator.validate_email("user.name@domain.co.jp") == True
        
        # 無効なメールアドレス
        assert InputValidator.validate_email("invalid-email") == False
        assert InputValidator.validate_email("@example.com") == False
        assert InputValidator.validate_email("test@") == False
    
    def test_input_validator_symbol(self):
        """入力検証 - 銘柄コードテスト"""
        # 有効な銘柄コード
        assert InputValidator.validate_symbol("AAPL") == True
        assert InputValidator.validate_symbol("GOOGL") == True
        assert InputValidator.validate_symbol("MSFT") == True
        
        # 無効な銘柄コード
        assert InputValidator.validate_symbol("invalid_symbol") == False
        assert InputValidator.validate_symbol("123") == False
        assert InputValidator.validate_symbol("") == False
    
    def test_input_validator_numeric(self):
        """入力検証 - 数値テスト"""
        # 有効な数値
        assert InputValidator.validate_numeric("123") == True
        assert InputValidator.validate_numeric("123.45") == True
        assert InputValidator.validate_numeric("0") == True
        
        # 無効な数値
        assert InputValidator.validate_numeric("abc") == False
        assert InputValidator.validate_numeric("12.34.56") == False
        assert InputValidator.validate_numeric("") == False
        
        # 範囲チェック
        assert InputValidator.validate_numeric("50", min_value=0, max_value=100) == True
        assert InputValidator.validate_numeric("150", min_value=0, max_value=100) == False
    
    def test_threat_detection(self):
        """脅威検出テスト"""
        # SQLインジェクション検出
        threats = InputValidator.detect_threats("SELECT * FROM users WHERE id = 1; DROP TABLE users;")
        assert "sql_injection" in threats
        
        # XSS検出
        threats = InputValidator.detect_threats("<script>alert('XSS')</script>")
        assert "xss" in threats
        
        # コマンドインジェクション検出
        threats = InputValidator.detect_threats("ls -la; rm -rf /")
        assert "command_injection" in threats
        
        # 安全な文字列
        threats = InputValidator.detect_threats("This is a safe string")
        assert len(threats) == 0
    
    def test_string_sanitization(self):
        """文字列サニタイゼーションテスト"""
        # HTML エスケープ
        result = InputValidator.sanitize_string("<script>alert('test')</script>")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result
        
        # 長さ制限
        long_string = "a" * 2000
        result = InputValidator.sanitize_string(long_string, max_length=100)
        assert len(result) == 100
    
    @pytest.mark.asyncio
    async def test_security_manager(self):
        """セキュリティマネージャーテスト"""
        security_manager = SecurityManager()
        
        # データ検証・サニタイゼーション
        test_data = {
            "email": "test@example.com",
            "symbol": "AAPL",
            "quantity": "100",
            "description": "<p>Valid description</p>"
        }
        
        field_validations = {
            "email": InputValidator.validate_email,
            "symbol": InputValidator.validate_symbol,
            "quantity": lambda x: InputValidator.validate_numeric(x, min_value=1)
        }
        
        sanitized_data = security_manager.validate_and_sanitize(test_data, field_validations)
        
        assert sanitized_data["email"] == test_data["email"]
        assert sanitized_data["symbol"] == test_data["symbol"]
        assert "&lt;p&gt;" in sanitized_data["description"]


# ================================
# パフォーマンス最適化テスト
# ================================

class TestPerformanceOptimization:
    """パフォーマンス最適化テスト"""
    
    def test_performance_monitor(self):
        """パフォーマンスモニターテスト"""
        monitor = PerformanceMonitor()
        
        # メトリクス記録
        metric = TestUtils.create_test_performance_metric()
        monitor.record_metric(metric)
        
        # メトリクス取得
        metrics = monitor.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.RESPONSE_TIME
        assert metrics[0].value == 100.0
        
        # フィルタリング
        cpu_metric = PerformanceMetric(
            metric_type=MetricType.CPU_USAGE,
            value=50.0,
            component="system"
        )
        monitor.record_metric(cpu_metric)
        
        cpu_metrics = monitor.get_metrics(metric_type=MetricType.CPU_USAGE)
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0].metric_type == MetricType.CPU_USAGE
    
    def test_performance_optimizer(self):
        """パフォーマンス最適化テスト"""
        optimizer = PerformanceOptimizer()
        
        # 最適化バッファ作成
        buffer = optimizer.create_optimized_buffer("test-buffer", max_size=10)
        
        # データ追加
        for i in range(15):  # バッファサイズを超える
            buffer.add(f"item-{i}")
        
        stats = buffer.get_stats()
        assert stats['current_size'] == 10  # maxlenに制限される
        assert stats['total_added'] == 15
        assert stats['total_dropped'] == 5
        
        # 適応型キャッシュ作成
        cache = optimizer.create_adaptive_cache("test-cache", initial_size=5)
        
        # キャッシュ操作
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key3") is None
        
        cache_stats = cache.get_stats()
        assert cache_stats['hits'] == 1
        assert cache_stats['misses'] == 1
    
    @pytest.mark.asyncio
    async def test_performance_optimized_decorator(self):
        """パフォーマンス最適化デコレーターテスト"""
        from day_trade.core.performance.performance_optimizer import performance_optimized
        
        @performance_optimized(cache_enabled=True, cache_size=5)
        async def expensive_calculation(x: int, y: int) -> int:
            # 重い計算をシミュレート
            await asyncio.sleep(0.01)
            return x * y + x ** y
        
        # 初回呼び出し（キャッシュミス）
        result1 = await expensive_calculation(2, 3)
        assert result1 == 14  # 2*3 + 2^3 = 6 + 8 = 14
        
        # 同じ引数での呼び出し（キャッシュヒット）
        result2 = await expensive_calculation(2, 3)
        assert result2 == 14
        
        # パフォーマンス統計確認
        stats = expensive_calculation.get_performance_stats()
        assert 'function_name' in stats
        assert 'cache_stats' in stats
        assert stats['cache_stats']['hits'] >= 1


# ================================
# 統合テスト
# ================================

class TestSystemIntegration:
    """システム統合テスト"""
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        # セキュリティエラーが統一エラーシステムで処理されることを確認
        try:
            raise SecurityError("テストセキュリティエラー")
        except UnifiedSystemError as e:
            assert isinstance(e, SecurityError)
            assert e.should_alert()
            assert "セキュリティ" in e.category.value
    
    def test_service_instance_with_security(self):
        """サービスインスタンスとセキュリティの統合テスト"""
        instance = TestUtils.create_test_service_instance("secure-service")
        
        # セキュリティレベル設定
        instance.metadata['security_level'] = SecurityLevel.CONFIDENTIAL.value
        
        # シリアライゼーション時の機密情報フィルタリング
        safe_dict = instance.to_dict(include_sensitive=False)
        confidential_dict = instance.to_dict(include_sensitive=True)
        
        assert 'security_level' not in safe_dict['metadata']
        assert 'security_level' in confidential_dict['metadata']
    
    @pytest.mark.asyncio
    async def test_performance_with_error_handling(self):
        """パフォーマンス監視とエラーハンドリングの統合テスト"""
        monitor = PerformanceMonitor()
        
        try:
            # エラーを発生させる
            raise ValidationError("テスト検証エラー")
        except UnifiedSystemError as e:
            # エラーメトリクス記録
            error_metric = PerformanceMetric(
                metric_type=MetricType.ERROR_RATE,
                value=1.0,
                component="test-integration",
                tags={'error_type': type(e).__name__}
            )
            monitor.record_metric(error_metric)
        
        # エラーメトリクスが記録されていることを確認
        error_metrics = monitor.get_metrics(metric_type=MetricType.ERROR_RATE)
        assert len(error_metrics) == 1
        assert error_metrics[0].tags['error_type'] == 'ValidationError'


# ================================
# ベンチマークテスト
# ================================

class TestBenchmarks:
    """ベンチマークテスト"""
    
    @pytest.mark.asyncio
    async def test_service_instance_serialization_benchmark(self):
        """サービスインスタンスシリアライゼーションベンチマーク"""
        import time
        
        instance = TestUtils.create_test_service_instance()
        
        # シリアライゼーション性能測定
        iterations = 1000
        start_time = time.time()
        
        for _ in range(iterations):
            instance.to_dict()
        
        elapsed_time = time.time() - start_time
        ops_per_second = iterations / elapsed_time
        
        # 最低限の性能要件チェック（1秒間に1000回以上）
        assert ops_per_second > 1000, f"シリアライゼーション性能が不十分: {ops_per_second} ops/sec"
    
    def test_input_validation_benchmark(self):
        """入力検証ベンチマーク"""
        import time
        
        test_strings = [
            "valid@example.com",
            "<script>alert('xss')</script>",
            "SELECT * FROM users",
            "normal text string",
            "AAPL"
        ]
        
        iterations = 10000
        start_time = time.time()
        
        for _ in range(iterations):
            for test_string in test_strings:
                InputValidator.validate_safe_input(test_string)
                InputValidator.detect_threats(test_string)
                InputValidator.sanitize_string(test_string)
        
        elapsed_time = time.time() - start_time
        ops_per_second = (iterations * len(test_strings)) / elapsed_time
        
        # 最低限の性能要件チェック（1秒間に10000回以上）
        assert ops_per_second > 10000, f"入力検証性能が不十分: {ops_per_second} ops/sec"


# ================================
# エンドツーエンドテスト
# ================================

class TestEndToEnd:
    """エンドツーエンドテスト"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """完全ワークフローテスト"""
        # 1. サービスインスタンス作成
        service = TestUtils.create_test_service_instance("trading-service")
        
        # 2. セキュリティマネージャー初期化
        security_manager = SecurityManager()
        
        # 3. パフォーマンス最適化初期化
        optimizer = PerformanceOptimizer()
        monitor = optimizer.monitor
        
        # 4. 取引データ検証
        trade_data = {
            "symbol": "AAPL",
            "quantity": "100",
            "price": "150.50",
            "direction": "buy"
        }
        
        field_validations = {
            "symbol": InputValidator.validate_symbol,
            "quantity": lambda x: InputValidator.validate_numeric(x, min_value=1),
            "price": lambda x: InputValidator.validate_numeric(x, min_value=0.01)
        }
        
        # 5. データ検証実行
        validated_data = security_manager.validate_and_sanitize(trade_data, field_validations)
        
        # 6. パフォーマンスメトリクス記録
        processing_metric = PerformanceMetric(
            metric_type=MetricType.RESPONSE_TIME,
            value=50.0,
            component=service.service_name
        )
        monitor.record_metric(processing_metric)
        
        # 7. 結果検証
        assert validated_data["symbol"] == "AAPL"
        assert validated_data["quantity"] == "100"
        assert service.is_healthy()
        
        metrics = monitor.get_metrics(component=service.service_name)
        assert len(metrics) == 1
        assert metrics[0].value == 50.0


# ================================
# テスト設定とフィクスチャ
# ================================

@pytest.fixture
def sample_service_instance():
    """サンプルサービスインスタンス"""
    return TestUtils.create_test_service_instance()

@pytest.fixture
def security_manager():
    """セキュリティマネージャー"""
    return SecurityManager()

@pytest.fixture
def performance_optimizer():
    """パフォーマンス最適化"""
    return PerformanceOptimizer()


# ================================
# テスト実行関数
# ================================

def run_error_tests():
    """エラーシステムテスト実行"""
    from .test_unified_error_system import run_error_tests as run_errors
    return run_errors()

def run_performance_tests():
    """パフォーマンステスト実行"""
    from .test_performance_system import run_performance_tests as run_perf
    return run_perf()

def run_monitoring_tests():
    """監視システムテスト実行"""
    from .test_unified_monitoring_system import run_monitoring_tests as run_monitoring
    return run_monitoring()

async def run_all_tests():
    """全テスト実行"""
    import sys
    import traceback
    
    test_classes = [
        TestUnifiedErrorSystem,
        TestServiceInstance, 
        TestSecuritySystem,
        TestPerformanceOptimization,
        TestSystemIntegration,
        TestBenchmarks,
        TestEndToEnd
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n=== {test_class.__name__} ===")
        test_instance = test_class()
        
        # テストメソッドを取得
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            test_method = getattr(test_instance, test_method_name)
            
            try:
                print(f"実行中: {test_method_name}...", end=" ")
                
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                print("✅ PASSED")
                passed_tests += 1
                
            except Exception as e:
                print("❌ FAILED")
                print(f"  エラー: {str(e)}")
                traceback.print_exc()
                failed_tests += 1
    
    # 個別テストスイート実行
    print("\n=== 個別テストスイート実行 ===")
    
    # エラーシステムテスト
    print("エラーシステムテスト実行中...")
    error_result = run_error_tests()
    
    # パフォーマンステスト
    print("パフォーマンステスト実行中...")
    perf_result = run_performance_tests()
    
    # 監視システムテスト
    print("監視システムテスト実行中...")
    monitoring_result = run_monitoring_tests()
    
    print(f"\n=== テスト結果 ===")
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed_tests}")
    print(f"失敗: {failed_tests}")
    print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
    print(f"\n個別テストスイート結果:")
    print(f"エラーシステム: {'✅' if error_result else '❌'}")
    print(f"パフォーマンス: {'✅' if perf_result else '❌'}")
    print(f"監視システム: {'✅' if monitoring_result else '❌'}")
    
    return passed_tests == total_tests and error_result and perf_result and monitoring_result


if __name__ == "__main__":
    # テスト実行
    asyncio.run(run_all_tests())