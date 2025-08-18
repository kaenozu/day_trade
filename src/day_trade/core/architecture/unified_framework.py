"""
統一アーキテクチャフレームワーク

システム全体の一貫したアーキテクチャパターンを提供
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
import logging
from contextlib import contextmanager
import asyncio

# 型変数定義
T = TypeVar('T')
TRequest = TypeVar('TRequest')
TResponse = TypeVar('TResponse')
TEntity = TypeVar('TEntity')

logger = logging.getLogger(__name__)


class ComponentLifecycle(Enum):
    """コンポーネントライフサイクル状態"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class ComponentType(Enum):
    """コンポーネントタイプ"""
    SERVICE = "service"
    REPOSITORY = "repository"
    ENGINE = "engine"
    ANALYZER = "analyzer"
    PROCESSOR = "processor"
    MANAGER = "manager"
    ADAPTER = "adapter"


@dataclass
class ComponentMetrics:
    """コンポーネントメトリクス"""
    requests_processed: int = 0
    errors_occurred: int = 0
    average_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_activity: Optional[datetime] = None


@dataclass
class OperationContext:
    """操作コンテキスト"""
    operation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OperationResult(Generic[T]):
    """操作結果の統一インターフェース"""

    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    @classmethod
    def success(cls, data: T, metadata: Optional[Dict[str, Any]] = None) -> 'OperationResult[T]':
        """成功結果作成"""
        return cls(True, data=data, metadata=metadata)

    @classmethod
    def failure(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> 'OperationResult[T]':
        """失敗結果作成"""
        return cls(False, error=error, metadata=metadata)

    def is_success(self) -> bool:
        """成功判定"""
        return self.success

    def get_data_or_raise(self) -> T:
        """データ取得（失敗時は例外）"""
        if not self.success:
            raise RuntimeError(f"Operation failed: {self.error}")
        return self.data


class BaseComponent(ABC):
    """基底コンポーネントクラス"""

    def __init__(self, name: str, component_type: ComponentType):
        self.name = name
        self.component_type = component_type
        self.lifecycle = ComponentLifecycle.INITIALIZING
        self.metrics = ComponentMetrics()
        self._config: Dict[str, Any] = {}

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """コンポーネント初期化"""
        if config:
            self._config.update(config)

        await self._initialize_internal()
        self.lifecycle = ComponentLifecycle.ACTIVE
        logger.info(f"Component {self.name} initialized successfully")

    @abstractmethod
    async def _initialize_internal(self) -> None:
        """内部初期化処理"""
        pass

    async def shutdown(self) -> None:
        """コンポーネント終了"""
        self.lifecycle = ComponentLifecycle.SHUTTING_DOWN
        await self._shutdown_internal()
        self.lifecycle = ComponentLifecycle.TERMINATED
        logger.info(f"Component {self.name} shut down successfully")

    @abstractmethod
    async def _shutdown_internal(self) -> None:
        """内部終了処理"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "lifecycle": self.lifecycle.value,
            "metrics": {
                "requests_processed": self.metrics.requests_processed,
                "errors_occurred": self.metrics.errors_occurred,
                "error_rate": (
                    self.metrics.errors_occurred / max(self.metrics.requests_processed, 1)
                ),
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            }
        }

    def update_metrics(self, response_time_ms: float, error: bool = False) -> None:
        """メトリクス更新"""
        self.metrics.requests_processed += 1
        if error:
            self.metrics.errors_occurred += 1

        # 移動平均でレスポンス時間更新
        if self.metrics.requests_processed == 1:
            self.metrics.average_response_time_ms = response_time_ms
        else:
            self.metrics.average_response_time_ms = (
                self.metrics.average_response_time_ms * 0.9 + response_time_ms * 0.1
            )

        self.metrics.last_activity = datetime.now()


class AsyncServiceProtocol(Protocol[TRequest, TResponse]):
    """非同期サービスプロトコル"""

    async def process(self, request: TRequest, context: OperationContext) -> OperationResult[TResponse]:
        """リクエスト処理"""
        ...


class CacheableService(ABC, Generic[TRequest, TResponse]):
    """キャッシュ対応サービス基底クラス"""

    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def generate_cache_key(self, request: TRequest, context: OperationContext) -> str:
        """キャッシュキー生成"""
        pass

    @abstractmethod
    async def process_uncached(self, request: TRequest, context: OperationContext) -> OperationResult[TResponse]:
        """キャッシュなし処理"""
        pass

    async def process(self, request: TRequest, context: OperationContext) -> OperationResult[TResponse]:
        """キャッシュ考慮処理"""
        cache_key = self.generate_cache_key(request, context)

        # キャッシュ確認
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl_seconds:
                return OperationResult.success(cached_data, {"from_cache": True})

        # 実処理
        result = await self.process_uncached(request, context)

        # キャッシュ保存
        if result.is_success():
            self._cache[cache_key] = (result.data, datetime.now())

        return result


class EventDrivenComponent(BaseComponent):
    """イベント駆動コンポーネント"""

    def __init__(self, name: str, component_type: ComponentType):
        super().__init__(name, component_type)
        self._event_handlers: Dict[str, List[callable]] = {}

    def subscribe(self, event_type: str, handler: callable) -> None:
        """イベント購読"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def publish_event(self, event_type: str, data: Any) -> None:
        """イベント発行"""
        if event_type in self._event_handlers:
            tasks = []
            for handler in self._event_handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(data))
                else:
                    # 同期関数を非同期で実行
                    tasks.append(asyncio.create_task(asyncio.to_thread(handler, data)))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class ResourceManager:
    """リソース管理"""

    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._cleanup_handlers: Dict[str, callable] = {}

    def register_resource(self, name: str, resource: Any, cleanup_handler: Optional[callable] = None) -> None:
        """リソース登録"""
        self._resources[name] = resource
        if cleanup_handler:
            self._cleanup_handlers[name] = cleanup_handler

    def get_resource(self, name: str) -> Any:
        """リソース取得"""
        return self._resources.get(name)

    async def cleanup_all(self) -> None:
        """全リソースクリーンアップ"""
        for name, cleanup_handler in self._cleanup_handlers.items():
            try:
                if asyncio.iscoroutinefunction(cleanup_handler):
                    await cleanup_handler(self._resources[name])
                else:
                    cleanup_handler(self._resources[name])
            except Exception as e:
                logger.error(f"Failed to cleanup resource {name}: {e}")


class ComponentRegistry:
    """コンポーネント登録管理"""

    def __init__(self):
        self._components: Dict[str, BaseComponent] = {}
        self._dependencies: Dict[str, List[str]] = {}

    def register(self, component: BaseComponent, dependencies: Optional[List[str]] = None) -> None:
        """コンポーネント登録"""
        self._components[component.name] = component
        if dependencies:
            self._dependencies[component.name] = dependencies

    def get_component(self, name: str) -> Optional[BaseComponent]:
        """コンポーネント取得"""
        return self._components.get(name)

    async def initialize_all(self) -> None:
        """全コンポーネント初期化（依存関係順）"""
        initialized = set()

        async def initialize_component(name: str):
            if name in initialized:
                return

            # 依存関係を先に初期化
            for dep in self._dependencies.get(name, []):
                await initialize_component(dep)

            component = self._components[name]
            await component.initialize()
            initialized.add(name)

        for name in self._components:
            await initialize_component(name)

    async def shutdown_all(self) -> None:
        """全コンポーネント終了"""
        # 逆順で終了
        for component in reversed(list(self._components.values())):
            await component.shutdown()


class ConfigurationProvider:
    """設定プロバイダー"""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._watchers: List[callable] = []

    def load_config(self, config: Dict[str, Any]) -> None:
        """設定読み込み"""
        old_config = self._config.copy()
        self._config.update(config)

        # 変更通知
        for watcher in self._watchers:
            try:
                watcher(old_config, self._config)
            except Exception as e:
                logger.error(f"Config watcher error: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """設定値取得"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def watch_changes(self, watcher: callable) -> None:
        """設定変更監視"""
        self._watchers.append(watcher)


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self):
        self._profiles: Dict[str, List[float]] = {}

    @contextmanager
    def profile(self, operation_name: str):
        """操作プロファイル"""
        start_time = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            if operation_name not in self._profiles:
                self._profiles[operation_name] = []
            self._profiles[operation_name].append(duration)

    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """統計情報取得"""
        if operation_name not in self._profiles:
            return {}

        times = self._profiles[operation_name]
        return {
            "count": len(times),
            "average_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p95_ms": sorted(times)[int(len(times) * 0.95)] if times else 0
        }


# グローバルインスタンス
component_registry = ComponentRegistry()
resource_manager = ResourceManager()
config_provider = ConfigurationProvider()
performance_profiler = PerformanceProfiler()


class UnifiedApplication:
    """統一アプリケーション"""

    def __init__(self, name: str):
        self.name = name
        self.registry = component_registry
        self.resource_manager = resource_manager
        self.config_provider = config_provider
        self.profiler = performance_profiler

    async def start(self, config: Dict[str, Any]) -> None:
        """アプリケーション開始"""
        logger.info(f"Starting application: {self.name}")

        # 設定読み込み
        self.config_provider.load_config(config)

        # コンポーネント初期化
        await self.registry.initialize_all()

        logger.info(f"Application {self.name} started successfully")

    async def stop(self) -> None:
        """アプリケーション停止"""
        logger.info(f"Stopping application: {self.name}")

        # コンポーネント終了
        await self.registry.shutdown_all()

        # リソースクリーンアップ
        await self.resource_manager.cleanup_all()

        logger.info(f"Application {self.name} stopped successfully")

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        component_healths = {}
        for name, component in self.registry._components.items():
            component_healths[name] = await component.health_check()

        return {
            "application": self.name,
            "status": "healthy",
            "components": component_healths,
            "timestamp": datetime.now().isoformat()
        }