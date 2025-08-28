#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Application Tests
コアアプリケーションテスト
"""

import pytest
import unittest.mock as mock
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import asyncio
from datetime import datetime
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# モッククラスを定義
class MockApplication:
    """モックアプリケーション"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.running = False
        self.services = {}
        self.config = {
            'database_url': 'sqlite:///test.db',
            'debug': True,
            'log_level': 'INFO'
        }
    
    def initialize(self):
        """アプリケーション初期化"""
        if self.initialized:
            raise RuntimeError("Already initialized")
            
        self.logger.info("Initializing application")
        
        # サービス初期化
        self.services['database'] = Mock()
        self.services['cache'] = Mock()
        self.services['trade_manager'] = Mock()
        
        self.initialized = True
        return True
    
    def start(self):
        """アプリケーション開始"""
        if not self.initialized:
            self.initialize()
            
        self.logger.info("Starting application")
        self.running = True
        return True
    
    def stop(self):
        """アプリケーション停止"""
        self.logger.info("Stopping application")
        self.running = False
        
        # サービスクリーンアップ
        for service_name, service in self.services.items():
            if hasattr(service, 'close'):
                service.close()
        
        return True
    
    def restart(self):
        """アプリケーション再起動"""
        self.stop()
        return self.start()
    
    def get_status(self):
        """アプリケーション状態取得"""
        return {
            'initialized': self.initialized,
            'running': self.running,
            'services': list(self.services.keys()),
            'uptime': datetime.now().isoformat()
        }
    
    def get_service(self, name):
        """サービス取得"""
        return self.services.get(name)
    
    def register_service(self, name, service):
        """サービス登録"""
        self.services[name] = service
    
    def health_check(self):
        """ヘルスチェック"""
        if not self.running:
            return {'status': 'stopped', 'healthy': False}
        
        # 各サービスのヘルスチェック
        service_health = {}
        for name, service in self.services.items():
            if hasattr(service, 'health_check'):
                service_health[name] = service.health_check()
            else:
                service_health[name] = {'status': 'unknown'}
        
        return {
            'status': 'running',
            'healthy': True,
            'services': service_health
        }


class MockAsyncApplication:
    """非同期モックアプリケーション"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.running = False
        self.services = {}
        self.tasks = []
    
    async def initialize(self):
        """非同期初期化"""
        if self.initialized:
            raise RuntimeError("Already initialized")
            
        self.logger.info("Async initializing application")
        
        # 非同期サービス初期化
        self.services['async_database'] = Mock()
        self.services['websocket_manager'] = Mock()
        
        await asyncio.sleep(0.1)  # 初期化時間をシミュレート
        self.initialized = True
        return True
    
    async def start(self):
        """非同期開始"""
        if not self.initialized:
            await self.initialize()
            
        self.logger.info("Async starting application")
        
        # バックグラウンドタスク開始
        self.tasks.append(asyncio.create_task(self._background_task()))
        
        self.running = True
        return True
    
    async def stop(self):
        """非同期停止"""
        self.logger.info("Async stopping application")
        
        # バックグラウンドタスク停止
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.running = False
        return True
    
    async def _background_task(self):
        """バックグラウンドタスク"""
        try:
            while self.running:
                await asyncio.sleep(1)
                self.logger.debug("Background task running")
        except asyncio.CancelledError:
            self.logger.info("Background task cancelled")
            raise


class TestMockApplication:
    """モックアプリケーションテストクラス"""
    
    @pytest.fixture
    def app(self):
        """アプリケーションフィクスチャ"""
        return MockApplication()
    
    def test_initialization(self, app):
        """初期化テスト"""
        assert app.initialized is False
        assert app.running is False
        
        result = app.initialize()
        assert result is True
        assert app.initialized is True
        assert len(app.services) > 0
    
    def test_double_initialization(self, app):
        """重複初期化テスト"""
        app.initialize()
        
        with pytest.raises(RuntimeError):
            app.initialize()
    
    def test_start_stop_cycle(self, app):
        """開始停止サイクルテスト"""
        # 開始
        result = app.start()
        assert result is True
        assert app.running is True
        assert app.initialized is True
        
        # 停止
        result = app.stop()
        assert result is True
        assert app.running is False
    
    def test_restart(self, app):
        """再起動テスト"""
        app.start()
        assert app.running is True
        
        result = app.restart()
        assert result is True
        assert app.running is True
    
    def test_status_retrieval(self, app):
        """状態取得テスト"""
        status = app.get_status()
        assert 'initialized' in status
        assert 'running' in status
        assert 'services' in status
        assert 'uptime' in status
        
        app.start()
        status = app.get_status()
        assert status['running'] is True
        assert len(status['services']) > 0
    
    def test_service_management(self, app):
        """サービス管理テスト"""
        app.initialize()
        
        # サービス取得
        trade_manager = app.get_service('trade_manager')
        assert trade_manager is not None
        
        # サービス登録
        new_service = Mock()
        app.register_service('new_service', new_service)
        
        retrieved_service = app.get_service('new_service')
        assert retrieved_service is new_service
    
    def test_health_check_stopped(self, app):
        """停止状態ヘルスチェック"""
        health = app.health_check()
        assert health['status'] == 'stopped'
        assert health['healthy'] is False
    
    def test_health_check_running(self, app):
        """実行状態ヘルスチェック"""
        app.start()
        health = app.health_check()
        
        assert health['status'] == 'running'
        assert health['healthy'] is True
        assert 'services' in health
    
    def test_configuration(self, app):
        """設定テスト"""
        assert app.config['debug'] is True
        assert app.config['log_level'] == 'INFO'
        assert 'database_url' in app.config


class TestMockAsyncApplication:
    """非同期アプリケーションテストクラス"""
    
    @pytest.fixture
    def async_app(self):
        """非同期アプリケーションフィクスチャ"""
        return MockAsyncApplication()
    
    @pytest.mark.asyncio
    async def test_async_initialization(self, async_app):
        """非同期初期化テスト"""
        assert async_app.initialized is False
        
        result = await async_app.initialize()
        assert result is True
        assert async_app.initialized is True
    
    @pytest.mark.asyncio
    async def test_async_start_stop(self, async_app):
        """非同期開始停止テスト"""
        # 開始
        result = await async_app.start()
        assert result is True
        assert async_app.running is True
        
        # 少し待機
        await asyncio.sleep(0.1)
        
        # 停止
        result = await async_app.stop()
        assert result is True
        assert async_app.running is False
    
    @pytest.mark.asyncio
    async def test_background_tasks(self, async_app):
        """バックグラウンドタスクテスト"""
        await async_app.start()
        
        # バックグラウンドタスクが開始されている
        assert len(async_app.tasks) > 0
        
        # 少し待機してタスクが実行されることを確認
        await asyncio.sleep(0.2)
        
        await async_app.stop()
        
        # タスクが停止されている
        for task in async_app.tasks:
            assert task.cancelled()


class TestApplicationLifecycle:
    """アプリケーションライフサイクルテスト"""
    
    def test_complete_lifecycle(self):
        """完全ライフサイクルテスト"""
        app = MockApplication()
        
        # 初期状態
        assert app.initialized is False
        assert app.running is False
        
        # 初期化
        app.initialize()
        assert app.initialized is True
        assert app.running is False
        
        # 開始
        app.start()
        assert app.running is True
        
        # ヘルスチェック
        health = app.health_check()
        assert health['healthy'] is True
        
        # 再起動
        app.restart()
        assert app.running is True
        
        # 停止
        app.stop()
        assert app.running is False
    
    def test_service_lifecycle(self):
        """サービスライフサイクルテスト"""
        app = MockApplication()
        
        # サービスが存在しない状態
        service = app.get_service('nonexistent')
        assert service is None
        
        # 初期化でサービスが作成される
        app.initialize()
        trade_manager = app.get_service('trade_manager')
        assert trade_manager is not None
        
        # カスタムサービス追加
        custom_service = Mock()
        app.register_service('custom', custom_service)
        
        retrieved = app.get_service('custom')
        assert retrieved is custom_service


class TestApplicationError:
    """アプリケーションエラーテスト"""
    
    def test_initialization_error(self):
        """初期化エラーテスト"""
        app = MockApplication()
        
        # 正常初期化
        app.initialize()
        
        # 重複初期化エラー
        with pytest.raises(RuntimeError, match="Already initialized"):
            app.initialize()
    
    def test_service_error_handling(self):
        """サービスエラーハンドリングテスト"""
        app = MockApplication()
        
        # エラーを起こすサービスを作成
        error_service = Mock()
        error_service.close.side_effect = Exception("Service error")
        
        app.initialize()
        app.register_service('error_service', error_service)
        
        # 停止時にエラーが発生しても継続する
        try:
            app.stop()
        except Exception:
            pytest.fail("Stop should handle service errors gracefully")


class TestApplicationIntegration:
    """アプリケーション統合テスト"""
    
    @pytest.mark.asyncio
    async def test_sync_async_integration(self):
        """同期・非同期統合テスト"""
        # 同期アプリケーション
        sync_app = MockApplication()
        sync_app.start()
        
        # 非同期アプリケーション
        async_app = MockAsyncApplication()
        await async_app.start()
        
        # 両方が実行中
        assert sync_app.running is True
        assert async_app.running is True
        
        # 停止
        sync_app.stop()
        await async_app.stop()
        
        assert sync_app.running is False
        assert async_app.running is False
    
    def test_multiple_applications(self):
        """複数アプリケーションテスト"""
        apps = [MockApplication() for _ in range(3)]
        
        # 全て開始
        for app in apps:
            app.start()
            assert app.running is True
        
        # 全て停止
        for app in apps:
            app.stop()
            assert app.running is False
    
    def test_application_monitoring(self):
        """アプリケーション監視テスト"""
        app = MockApplication()
        app.start()
        
        # 状態監視
        status = app.get_status()
        assert status['running'] is True
        
        # ヘルス監視
        health = app.health_check()
        assert health['healthy'] is True
        
        # サービス監視
        services = list(app.services.keys())
        assert len(services) > 0
        
        for service_name in services:
            service = app.get_service(service_name)
            assert service is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])