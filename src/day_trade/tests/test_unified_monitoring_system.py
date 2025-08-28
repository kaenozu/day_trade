"""
統合監視システムテスト

包括的なテストスイートで監視システムの動作を検証。
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from day_trade.core.monitoring import (
    MetricType, AlertLevel, MonitoringScope,
    MetricValue, Metric, Alert, HealthCheck,
    ThresholdAlertRule, SystemMetricsCollector,
    MetricsStorage, AlertManager, HealthCheckManager,
    UnifiedMonitoringSystem, get_global_monitoring_system,
    monitor_performance, IntegrationConfig,
    MonitoringIntegrationManager, initialize_integrated_monitoring,
    SecurityMiddleware, RateLimiter, AuthenticationMiddleware
)


class TestMetricsStorage:
    """メトリクスストレージテスト"""
    
    def test_create_metric(self):
        """メトリクス作成テスト"""
        storage = MetricsStorage()
        
        metric = storage.create_metric(
            "test.metric", MetricType.GAUGE, "テストメトリクス", "unit"
        )
        
        assert metric.name == "test.metric"
        assert metric.type == MetricType.GAUGE
        assert metric.description == "テストメトリクス"
        assert metric.unit == "unit"
        
    def test_record_metric(self):
        """メトリクス記録テスト"""
        storage = MetricsStorage()
        storage.create_metric("test.metric", MetricType.COUNTER, "テスト")
        
        success = storage.record_metric("test.metric", 100)
        assert success is True
        
        metric = storage.get_metric("test.metric")
        assert len(metric.values) == 1
        assert metric.values[0].value == 100
        
    def test_record_nonexistent_metric(self):
        """存在しないメトリクス記録テスト"""
        storage = MetricsStorage()
        
        success = storage.record_metric("nonexistent", 100)
        assert success is False
        
    def test_cleanup_old_metrics(self):
        """古いメトリクスクリーンアップテスト"""
        storage = MetricsStorage(retention_period=timedelta(seconds=1))
        storage.create_metric("test.metric", MetricType.GAUGE, "テスト")
        
        # 古いメトリクス追加
        old_time = datetime.now() - timedelta(seconds=2)
        metric = storage.get_metric("test.metric")
        metric.values.append(MetricValue(100, old_time))
        
        # 新しいメトリクス追加
        storage.record_metric("test.metric", 200)
        
        # クリーンアップ実行
        storage.cleanup_old_metrics()
        
        # 古いメトリクスが削除され、新しいものは残る
        assert len(metric.values) == 1
        assert metric.values[0].value == 200


class TestThresholdAlertRule:
    """閾値アラートルールテスト"""
    
    def test_greater_than_rule(self):
        """より大きい条件テスト"""
        rule = ThresholdAlertRule("CPU高", AlertLevel.WARNING, "cpu.usage", 80.0, "gt")
        
        # メトリクス作成
        metric = Metric("cpu.usage", MetricType.GAUGE, "CPU使用率")
        metric.values.append(MetricValue(85.0, datetime.now()))
        
        alert = rule.evaluate(metric)
        assert alert is not None
        assert alert.name == "CPU高"
        assert alert.level == AlertLevel.WARNING
        
    def test_less_than_rule(self):
        """より小さい条件テスト"""
        rule = ThresholdAlertRule("CPU正常", AlertLevel.INFO, "cpu.usage", 50.0, "lt")
        
        metric = Metric("cpu.usage", MetricType.GAUGE, "CPU使用率")
        metric.values.append(MetricValue(30.0, datetime.now()))
        
        alert = rule.evaluate(metric)
        assert alert is not None
        
    def test_no_alert_condition(self):
        """アラート条件不一致テスト"""
        rule = ThresholdAlertRule("CPU高", AlertLevel.WARNING, "cpu.usage", 80.0, "gt")
        
        metric = Metric("cpu.usage", MetricType.GAUGE, "CPU使用率")
        metric.values.append(MetricValue(70.0, datetime.now()))
        
        alert = rule.evaluate(metric)
        assert alert is None


class TestAlertManager:
    """アラート管理器テスト"""
    
    def test_add_alert_rule(self):
        """アラートルール追加テスト"""
        manager = AlertManager()
        rule = ThresholdAlertRule("テスト", AlertLevel.WARNING, "test.metric", 100.0)
        
        manager.add_alert_rule(rule)
        assert len(manager.alert_rules) == 1
        
    def test_alert_handler_execution(self):
        """アラートハンドラー実行テスト"""
        manager = AlertManager()
        handler_called = False
        
        def test_handler(alert):
            nonlocal handler_called
            handler_called = True
            
        manager.add_alert_handler(AlertLevel.WARNING, test_handler)
        
        # アラート発火
        alert = Alert("test", "テスト", AlertLevel.WARNING, "テストメッセージ", "test")
        manager._trigger_alert(alert)
        
        assert handler_called is True
        assert len(manager.active_alerts) == 1
        
    def test_resolve_alert(self):
        """アラート解決テスト"""
        manager = AlertManager()
        alert = Alert("test", "テスト", AlertLevel.WARNING, "テストメッセージ", "test")
        
        manager._trigger_alert(alert)
        assert len(manager.active_alerts) == 1
        
        resolved = manager.resolve_alert(alert.alert_id)
        assert resolved is True
        assert len(manager.active_alerts) == 0
        assert len(manager.resolved_alerts) == 1


class TestSystemMetricsCollector:
    """システムメトリクス収集器テスト"""
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self):
        """システムメトリクス収集テスト"""
        collector = SystemMetricsCollector()
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net:
                
            # モックデータ設定
            mock_memory.return_value = Mock(percent=60.0, available=1024*1024*1024)
            mock_disk.return_value = Mock(used=500*1024*1024*1024, total=1000*1024*1024*1024)
            mock_net.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
            
            metrics = await collector.collect_system_metrics()
            
            assert "system.cpu.usage" in metrics
            assert "system.memory.usage" in metrics
            assert "system.disk.usage" in metrics
            assert "system.network.bytes_sent" in metrics
            assert "system.network.bytes_recv" in metrics
            
            assert metrics["system.cpu.usage"].value == 50.0
            assert metrics["system.memory.usage"].value == 60.0


class TestHealthCheckManager:
    """ヘルスチェック管理器テスト"""
    
    def test_register_health_check(self):
        """ヘルスチェック登録テスト"""
        manager = HealthCheckManager()
        
        manager.register_health_check("test_service", "http://localhost:8080/health")
        
        assert "test_service" in manager.health_checks
        health_check = manager.health_checks["test_service"]
        assert health_check.name == "test_service"
        assert health_check.endpoint == "http://localhost:8080/health"
        
    @pytest.mark.asyncio
    async def test_perform_health_check_success(self):
        """ヘルスチェック成功テスト"""
        manager = HealthCheckManager()
        manager.register_health_check("test_service", "http://localhost:8080/health")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await manager.perform_health_check("test_service")
            
            assert result is True
            health_check = manager.health_checks["test_service"]
            assert health_check.status is True
            
    @pytest.mark.asyncio
    async def test_perform_health_check_failure(self):
        """ヘルスチェック失敗テスト"""
        manager = HealthCheckManager()
        manager.register_health_check("test_service", "http://localhost:8080/health")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("接続エラー")
            
            result = await manager.perform_health_check("test_service")
            
            assert result is False
            health_check = manager.health_checks["test_service"]
            assert health_check.status is False
            assert "接続エラー" in health_check.error_message


class TestUnifiedMonitoringSystem:
    """統合監視システムテスト"""
    
    @pytest.fixture
    def monitoring_system(self):
        """監視システムフィクスチャ"""
        return UnifiedMonitoringSystem()
        
    def test_initialization(self, monitoring_system):
        """初期化テスト"""
        assert monitoring_system.running is False
        assert len(monitoring_system.metrics_storage.metrics) > 0
        assert len(monitoring_system.alert_manager.alert_rules) > 0
        
    @pytest.mark.asyncio
    async def test_start_stop(self, monitoring_system):
        """開始・停止テスト"""
        await monitoring_system.start()
        assert monitoring_system.running is True
        assert len(monitoring_system.tasks) > 0
        
        await monitoring_system.stop()
        assert monitoring_system.running is False
        assert len(monitoring_system.tasks) == 0
        
    def test_record_business_metric(self, monitoring_system):
        """ビジネスメトリクス記録テスト"""
        monitoring_system.record_business_metric("business.test", 100.0)
        
        metric = monitoring_system.metrics_storage.get_metric("business.test")
        assert metric is None  # メトリクスが存在しない場合
        
        # メトリクス作成後
        monitoring_system.metrics_storage.create_metric(
            "business.test", MetricType.GAUGE, "テスト"
        )
        monitoring_system.record_business_metric("business.test", 200.0)
        
        metric = monitoring_system.metrics_storage.get_metric("business.test")
        assert len(metric.values) == 1
        assert metric.values[0].value == 200.0
        
    def test_get_monitoring_status(self, monitoring_system):
        """監視状態取得テスト"""
        status = monitoring_system.get_monitoring_status()
        
        assert "running" in status
        assert "uptime" in status
        assert "metrics_collected" in status
        assert "alerts_triggered" in status
        assert "system_overview" in status
        
    def test_export_metrics(self, monitoring_system):
        """メトリクスエクスポートテスト"""
        # テストメトリクス作成
        monitoring_system.metrics_storage.create_metric(
            "test.export", MetricType.COUNTER, "エクスポートテスト"
        )
        monitoring_system.record_business_metric("test.export", 42.0)
        
        exported = monitoring_system.export_metrics("json")
        
        assert isinstance(exported, str)
        assert "test.export" in exported
        assert "42" in exported


class TestMonitorPerformanceDecorator:
    """パフォーマンス監視デコレータテスト"""
    
    def test_monitor_performance_decorator(self):
        """パフォーマンス監視デコレータテスト"""
        monitoring_system = get_global_monitoring_system()
        
        # テスト用のメトリクス作成
        monitoring_system.metrics_storage.create_metric(
            "function.test_function.duration", MetricType.HISTOGRAM, "テスト関数実行時間"
        )
        
        @monitor_performance("test_function")
        def test_function():
            time.sleep(0.01)  # 10ms
            return "result"
            
        result = test_function()
        
        assert result == "result"
        
        # メトリクスが記録されたことを確認
        metric = monitoring_system.metrics_storage.get_metric("function.test_function.duration")
        assert len(metric.values) == 1
        assert metric.values[0].value >= 10  # 少なくとも10ms
        
    def test_monitor_performance_with_exception(self):
        """例外時のパフォーマンス監視テスト"""
        monitoring_system = get_global_monitoring_system()
        
        # エラーメトリクス作成
        monitoring_system.metrics_storage.create_metric(
            "function.error_function.errors", MetricType.COUNTER, "エラー関数エラー数"
        )
        
        @monitor_performance("error_function")
        def error_function():
            raise ValueError("テストエラー")
            
        with pytest.raises(ValueError):
            error_function()
            
        # エラーメトリクスが記録されたことを確認
        metric = monitoring_system.metrics_storage.get_metric("function.error_function.errors")
        assert len(metric.values) == 1
        assert metric.values[0].value == 1


class TestIntegrationManager:
    """統合管理器テスト"""
    
    @pytest.fixture
    def integration_config(self):
        """統合設定フィクスチャ"""
        return IntegrationConfig(
            enable_dashboard=False,  # テスト時はダッシュボードを無効
            enable_microservices_monitoring=False,
            enable_event_monitoring=False
        )
        
    def test_integration_config(self, integration_config):
        """統合設定テスト"""
        assert integration_config.enable_dashboard is False
        assert integration_config.enable_microservices_monitoring is False
        assert integration_config.enable_event_monitoring is False
        assert integration_config.enable_performance_integration is True
        
    @pytest.mark.asyncio
    async def test_initialize_integration_manager(self, integration_config):
        """統合管理器初期化テスト"""
        manager = MonitoringIntegrationManager(integration_config)
        
        await manager.initialize()
        
        status = manager.get_integration_status()
        assert status["status"]["monitoring_system"] is True
        assert status["active_integrations"] > 0
        
        await manager.shutdown()


class TestMonitoringSystemIntegration:
    """監視システム統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """フルシステム統合テスト"""
        config = IntegrationConfig(
            enable_dashboard=False,  # テスト時は無効
            enable_microservices_monitoring=False,
            enable_event_monitoring=False,
            auto_start=False
        )
        
        manager = await initialize_integrated_monitoring(config)
        
        # システム状態確認
        status = manager.get_integration_status()
        assert status["status"]["monitoring_system"] is True
        
        # メトリクス記録テスト
        manager.monitoring_system.record_business_metric("integration.test", 100.0)
        
        # 少し待つ
        await asyncio.sleep(0.1)
        
        # 統合システム停止
        await manager.shutdown()
        
        # 停止状態確認
        status = manager.get_integration_status()
        assert all(not v for v in status["status"].values())


class TestSecurityMiddleware:
    """セキュリティミドルウェアテスト"""
    
    def test_rate_limiter_creation(self):
        """レート制限器作成テスト"""
        rate_limiter = RateLimiter()
        
        assert rate_limiter.default_rule.requests_per_minute == 1000
        assert len(rate_limiter.clients) == 0
        
    def test_rate_limiter_allow_requests(self):
        """レート制限器リクエスト許可テスト"""
        from day_trade.core.monitoring.security_middleware import RateLimitRule
        
        rate_limiter = RateLimiter(RateLimitRule(requests_per_minute=5))
        client_ip = "192.168.1.1"
        
        # 最初の5リクエストは許可
        for i in range(5):
            assert rate_limiter.is_allowed(client_ip) is True
            
        # 6番目は拒否
        assert rate_limiter.is_allowed(client_ip) is False
        
        # 統計確認
        stats = rate_limiter.get_client_stats(client_ip)
        assert stats["requests_in_window"] == 5
        assert stats["total_requests"] == 6  # 拒否されたものもカウント
        
    def test_authentication_middleware_api_key(self):
        """認証ミドルウェアAPIキーテスト"""
        auth = AuthenticationMiddleware(require_auth=True)
        
        # APIキー追加
        test_key = "test-api-key-123"
        auth.add_api_key(test_key)
        
        # 正しいAPIキー
        assert auth.verify_api_key(test_key) is True
        
        # 間違ったAPIキー
        assert auth.verify_api_key("wrong-key") is False
        
        # 空のAPIキー
        assert auth.verify_api_key("") is False
        
    def test_authentication_middleware_session(self):
        """認証ミドルウェアセッションテスト"""
        auth = AuthenticationMiddleware()
        
        # セッション作成
        session_id = auth.create_session("user123", ["read", "write"])
        assert isinstance(session_id, str)
        assert len(session_id) == 64  # SHA256ハッシュの長さ
        
        # セッション検証
        session = auth.verify_session(session_id)
        assert session is not None
        assert session["user_id"] == "user123"
        assert "read" in session["permissions"]
        assert "write" in session["permissions"]
        
        # 無効なセッション
        invalid_session = auth.verify_session("invalid-session-id")
        assert invalid_session is None
        
    def test_security_middleware_creation(self):
        """セキュリティミドルウェア作成テスト"""
        config = {
            "require_authentication": True,
            "rate_limit_requests_per_minute": 100
        }
        
        middleware = SecurityMiddleware(config)
        
        assert middleware.config == config
        assert middleware.auth.require_auth is True
        assert middleware.rate_limiter.default_rule.requests_per_minute == 100
        
        # セキュリティヘッダー確認
        assert "X-Content-Type-Options" in middleware.security_headers
        assert "X-Frame-Options" in middleware.security_headers
        assert middleware.security_headers["X-Frame-Options"] == "DENY"


def run_monitoring_tests():
    """監視システムテスト実行"""
    print("統合監視システムテストを開始...")
    
    # pytest実行
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    if exit_code == 0:
        print("✓ 統合監視システムテストが全て成功しました")
    else:
        print("✗ 統合監視システムテストで失敗がありました")
        
    return exit_code == 0


if __name__ == "__main__":
    run_monitoring_tests()