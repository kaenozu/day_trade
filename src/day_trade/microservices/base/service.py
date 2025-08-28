#!/usr/bin/env python3
"""
Base Microservice Implementation
Hexagonal Architecture + マイクロサービス基底クラス
"""

import asyncio
import logging
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID, uuid4
from enum import Enum

import uvloop
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx


class ServiceStatus(Enum):
    """サービスステータス"""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class HealthCheckResult(BaseModel):
    """ヘルスチェック結果"""
    status: ServiceStatus
    checks: Dict[str, bool]
    metrics: Dict[str, Any]
    timestamp: datetime
    response_time_ms: float


@dataclass
class ServiceConfig:
    """サービス設定"""
    name: str
    version: str
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Service Discovery
    registry_url: Optional[str] = None
    heartbeat_interval: int = 30
    
    # Resilience
    circuit_breaker_enabled: bool = True
    retry_max_attempts: int = 3
    timeout_seconds: int = 30
    
    # Observability
    tracing_enabled: bool = True
    metrics_enabled: bool = True
    log_level: str = "INFO"
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    database_url: Optional[str] = None
    message_broker_url: Optional[str] = None


class ServiceHealth:
    """サービスヘルス管理"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.status = ServiceStatus.STARTING
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.metrics: Dict[str, Any] = {}
        self.last_check_time = datetime.utcnow()
    
    def add_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """ヘルスチェック追加"""
        self.health_checks[name] = check_func
    
    async def check_health(self) -> HealthCheckResult:
        """ヘルスチェック実行"""
        start_time = time.time()
        
        checks = {}
        all_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                result = await asyncio.to_thread(check_func) if asyncio.iscoroutinefunction(check_func) else check_func()
                checks[name] = result
                if not result:
                    all_healthy = False
            except Exception as e:
                logging.error(f"Health check '{name}' failed: {e}")
                checks[name] = False
                all_healthy = False
        
        # ステータス決定
        if all_healthy:
            self.status = ServiceStatus.HEALTHY
        elif any(checks.values()):
            self.status = ServiceStatus.DEGRADED
        else:
            self.status = ServiceStatus.UNHEALTHY
        
        response_time = (time.time() - start_time) * 1000
        self.last_check_time = datetime.utcnow()
        
        return HealthCheckResult(
            status=self.status,
            checks=checks,
            metrics=self.metrics.copy(),
            timestamp=self.last_check_time,
            response_time_ms=response_time
        )
    
    def update_metric(self, name: str, value: Any) -> None:
        """メトリクス更新"""
        self.metrics[name] = value


class BaseService(ABC):
    """
    マイクロサービス基底クラス
    Hexagonal Architecture + 分散システム対応
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.service_id = str(uuid4())
        self.start_time = datetime.utcnow()
        
        # FastAPI アプリケーション
        self.app = FastAPI(
            title=config.name,
            version=config.version,
            debug=config.debug
        )
        
        # Health Management
        self.health = ServiceHealth(config.name)
        
        # Service Registry Client
        self.registry_client: Optional[httpx.AsyncClient] = None
        
        # Graceful Shutdown
        self.shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
        
        # Default Routes
        self._setup_default_routes()
        
        # Default Health Checks
        self._setup_default_health_checks()
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(config.name)
    
    @abstractmethod
    async def initialize(self) -> None:
        """サービス初期化（各サービスで実装）"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """サービス終了処理（各サービスで実装）"""
        pass
    
    async def start(self) -> None:
        """サービス開始"""
        try:
            # uvloop設定
            if not isinstance(asyncio.get_event_loop_policy(), uvloop.EventLoopPolicy):
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            
            self.logger.info(f"Starting service: {self.config.name}")
            
            # サービス初期化
            await self.initialize()
            
            # サービス登録
            await self._register_service()
            
            # ヘルスチェック開始
            asyncio.create_task(self._health_check_loop())
            
            # ハートビート開始
            asyncio.create_task(self._heartbeat_loop())
            
            self.health.status = ServiceStatus.HEALTHY
            self.logger.info(f"Service {self.config.name} started successfully")
            
            # シャットダウンまで待機
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            self.health.status = ServiceStatus.UNHEALTHY
            raise
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """サービス停止"""
        self.health.status = ServiceStatus.SHUTTING_DOWN
        self.logger.info(f"Shutting down service: {self.config.name}")
        
        try:
            # サービス登録解除
            await self._deregister_service()
            
            # クリーンアップ
            await self.cleanup()
            
            # クライアント終了
            if self.registry_client:
                await self.registry_client.aclose()
            
            self.health.status = ServiceStatus.STOPPED
            self.logger.info(f"Service {self.config.name} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during service shutdown: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _setup_default_routes(self) -> None:
        """デフォルトルート設定"""
        
        @self.app.get("/health")
        async def health_check():
            """ヘルスチェックエンドポイント"""
            result = await self.health.check_health()
            status_code = 200 if result.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED] else 503
            return result.dict()
        
        @self.app.get("/info")
        async def service_info():
            """サービス情報エンドポイント"""
            uptime = datetime.utcnow() - self.start_time
            return {
                "service_name": self.config.name,
                "service_id": self.service_id,
                "version": self.config.version,
                "status": self.health.status.value,
                "uptime_seconds": uptime.total_seconds(),
                "start_time": self.start_time.isoformat()
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """メトリクスエンドポイント"""
            return self.health.metrics
        
        @self.app.post("/shutdown")
        async def shutdown():
            """シャットダウンエンドポイント"""
            self.shutdown_event.set()
            return {"message": "Shutdown initiated"}
    
    def _setup_default_health_checks(self) -> None:
        """デフォルトヘルスチェック設定"""
        
        def basic_health_check() -> bool:
            """基本ヘルスチェック"""
            return self.health.status != ServiceStatus.UNHEALTHY
        
        def memory_check() -> bool:
            """メモリチェック"""
            try:
                import psutil
                process = psutil.Process()
                memory_percent = process.memory_percent()
                self.health.update_metric("memory_percent", memory_percent)
                return memory_percent < 90.0  # 90%未満
            except ImportError:
                return True
        
        def uptime_check() -> bool:
            """稼働時間チェック"""
            uptime = datetime.utcnow() - self.start_time
            self.health.update_metric("uptime_seconds", uptime.total_seconds())
            return True
        
        self.health.add_health_check("basic", basic_health_check)
        self.health.add_health_check("memory", memory_check)
        self.health.add_health_check("uptime", uptime_check)
    
    async def _register_service(self) -> None:
        """サービス登録"""
        if not self.config.registry_url:
            return
        
        try:
            self.registry_client = httpx.AsyncClient()
            
            registration_data = {
                "service_id": self.service_id,
                "service_name": self.config.name,
                "version": self.config.version,
                "host": self.config.host,
                "port": self.config.port,
                "health_check_url": f"http://{self.config.host}:{self.config.port}/health",
                "metadata": {
                    "start_time": self.start_time.isoformat(),
                    "dependencies": self.config.dependencies
                }
            }
            
            response = await self.registry_client.post(
                f"{self.config.registry_url}/services/register",
                json=registration_data,
                timeout=10.0
            )
            response.raise_for_status()
            
            self.logger.info(f"Service registered with registry: {self.service_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to register service: {e}")
    
    async def _deregister_service(self) -> None:
        """サービス登録解除"""
        if not self.registry_client or not self.config.registry_url:
            return
        
        try:
            await self.registry_client.delete(
                f"{self.config.registry_url}/services/{self.service_id}",
                timeout=10.0
            )
            self.logger.info(f"Service deregistered: {self.service_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service: {e}")
    
    async def _health_check_loop(self) -> None:
        """ヘルスチェックループ"""
        while not self.shutdown_event.is_set():
            try:
                await self.health.check_health()
                await asyncio.sleep(10)  # 10秒間隔
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _heartbeat_loop(self) -> None:
        """ハートビートループ"""
        if not self.registry_client or not self.config.registry_url:
            return
        
        while not self.shutdown_event.is_set():
            try:
                heartbeat_data = {
                    "service_id": self.service_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": self.health.status.value
                }
                
                await self.registry_client.put(
                    f"{self.config.registry_url}/services/{self.service_id}/heartbeat",
                    json=heartbeat_data,
                    timeout=5.0
                )
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)


# Utility functions
async def run_service(service: BaseService) -> None:
    """サービス実行ユーティリティ"""
    try:
        await service.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Service failed: {e}")
        raise


def create_service_config(
    name: str,
    version: str = "1.0.0",
    port: int = 8000,
    **kwargs
) -> ServiceConfig:
    """サービス設定作成ユーティリティ"""
    return ServiceConfig(
        name=name,
        version=version,
        port=port,
        **kwargs
    )