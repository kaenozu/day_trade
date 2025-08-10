#!/usr/bin/env python3
"""
Dependency Injection Container
依存性注入コンテナー

軽量DIコンテナーによるサービス管理とライフサイクル制御
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from enum import Enum
from dataclasses import dataclass
import functools
import threading
from abc import ABC, abstractmethod

from ..exceptions.risk_exceptions import ConfigurationError

T = TypeVar('T')

class ServiceLifetime(Enum):
    """サービスライフタイム"""
    SINGLETON = "singleton"      # 単一インスタンス
    TRANSIENT = "transient"      # 毎回新規作成
    SCOPED = "scoped"           # スコープ内で単一

@dataclass
class ServiceDescriptor:
    """サービス記述子"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: List[Type] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class IServiceScope(ABC):
    """サービススコープインターフェース"""

    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """サービス取得"""
        pass

    @abstractmethod
    def dispose(self) -> None:
        """リソース解放"""
        pass

class ServiceScope(IServiceScope):
    """サービススコープ実装"""

    def __init__(self, container: 'DIContainer'):
        self._container = container
        self._scoped_instances: Dict[Type, Any] = {}
        self._disposed = False

    def get_service(self, service_type: Type[T]) -> T:
        """スコープ内サービス取得"""
        if self._disposed:
            raise RuntimeError("Service scope has been disposed")

        descriptor = self._container._get_service_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type not in self._scoped_instances:
                self._scoped_instances[service_type] = self._container._create_instance(descriptor, self)
            return self._scoped_instances[service_type]
        else:
            return self._container._create_instance(descriptor, self)

    def dispose(self) -> None:
        """スコープ内インスタンス解放"""
        if self._disposed:
            return

        for instance in self._scoped_instances.values():
            if hasattr(instance, 'dispose'):
                try:
                    instance.dispose()
                except Exception:
                    pass

        self._scoped_instances.clear()
        self._disposed = True

class DIContainer:
    """依存性注入コンテナー"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()

        # 自身をコンテナーとして登録
        self.register_instance(DIContainer, self)

    def register_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None
    ) -> 'DIContainer':
        """一時的サービス登録"""
        return self._register_service(
            service_type,
            implementation_type or service_type,
            ServiceLifetime.TRANSIENT
        )

    def register_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None
    ) -> 'DIContainer':
        """シングルトンサービス登録"""
        return self._register_service(
            service_type,
            implementation_type or service_type,
            ServiceLifetime.SINGLETON
        )

    def register_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None
    ) -> 'DIContainer':
        """スコープサービス登録"""
        return self._register_service(
            service_type,
            implementation_type or service_type,
            ServiceLifetime.SCOPED
        )

    def register_instance(
        self,
        service_type: Type[T],
        instance: T
    ) -> 'DIContainer':
        """インスタンス登録"""
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                instance=instance,
                lifetime=ServiceLifetime.SINGLETON
            )
            self._singletons[service_type] = instance
        return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    ) -> 'DIContainer':
        """ファクトリー関数登録"""
        with self._lock:
            dependencies = self._analyze_dependencies(factory)
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime,
                dependencies=dependencies
            )
        return self

    def get_service(self, service_type: Type[T]) -> T:
        """サービス取得"""
        descriptor = self._get_service_descriptor(service_type)
        return self._create_instance(descriptor)

    def get_required_service(self, service_type: Type[T]) -> T:
        """必須サービス取得（見つからない場合は例外）"""
        if service_type not in self._services:
            raise ConfigurationError(
                f"Service of type {service_type.__name__} is not registered",
                config_key=f"service.{service_type.__name__}"
            )
        return self.get_service(service_type)

    def get_services(self, service_type: Type[T]) -> List[T]:
        """同一タイプの全サービス取得"""
        # 簡略化実装：単一サービスのみ対応
        try:
            service = self.get_service(service_type)
            return [service] if service else []
        except:
            return []

    def create_scope(self) -> ServiceScope:
        """新しいサービススコープ作成"""
        return ServiceScope(self)

    def is_registered(self, service_type: Type) -> bool:
        """サービス登録状態確認"""
        return service_type in self._services

    def _register_service(
        self,
        service_type: Type[T],
        implementation_type: Type[T],
        lifetime: ServiceLifetime
    ) -> 'DIContainer':
        """サービス登録内部実装"""
        with self._lock:
            dependencies = self._analyze_dependencies(implementation_type.__init__)
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                lifetime=lifetime,
                dependencies=dependencies
            )
        return self

    def _get_service_descriptor(self, service_type: Type) -> ServiceDescriptor:
        """サービス記述子取得"""
        if service_type not in self._services:
            # 自動登録を試行
            if inspect.isclass(service_type) and not inspect.isabstract(service_type):
                self.register_transient(service_type)
            else:
                raise ConfigurationError(
                    f"Service of type {service_type.__name__} is not registered",
                    config_key=f"service.{service_type.__name__}"
                )

        return self._services[service_type]

    def _create_instance(
        self,
        descriptor: ServiceDescriptor,
        scope: Optional[ServiceScope] = None
    ) -> Any:
        """インスタンス作成"""

        # 既存インスタンス使用
        if descriptor.instance is not None:
            return descriptor.instance

        # シングルトンチェック
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.service_type in self._singletons:
                return self._singletons[descriptor.service_type]

        # インスタンス作成
        if descriptor.factory:
            instance = self._create_from_factory(descriptor, scope)
        elif descriptor.implementation_type:
            instance = self._create_from_type(descriptor, scope)
        else:
            raise ConfigurationError(
                f"Cannot create instance of {descriptor.service_type.__name__}",
                config_key=f"service.{descriptor.service_type.__name__}"
            )

        # シングルトンキャッシュ
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            with self._lock:
                self._singletons[descriptor.service_type] = instance

        return instance

    def _create_from_factory(
        self,
        descriptor: ServiceDescriptor,
        scope: Optional[ServiceScope] = None
    ) -> Any:
        """ファクトリーからインスタンス作成"""

        # 依存関係解決
        dependencies = {}
        for dep_type in descriptor.dependencies:
            if scope and dep_type in self._services:
                dep_descriptor = self._services[dep_type]
                if dep_descriptor.lifetime == ServiceLifetime.SCOPED:
                    dependencies[self._get_parameter_name(dep_type)] = scope.get_service(dep_type)
                else:
                    dependencies[self._get_parameter_name(dep_type)] = self.get_service(dep_type)
            else:
                dependencies[self._get_parameter_name(dep_type)] = self.get_service(dep_type)

        return descriptor.factory(**dependencies)

    def _create_from_type(
        self,
        descriptor: ServiceDescriptor,
        scope: Optional[ServiceScope] = None
    ) -> Any:
        """型からインスタンス作成"""

        # コンストラクター依存関係解決
        constructor = descriptor.implementation_type.__init__
        sig = inspect.signature(constructor)

        args = []
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # 型アノテーションから依存関係を特定
            if param.annotation != param.empty:
                service_type = param.annotation

                if scope and service_type in self._services:
                    dep_descriptor = self._services[service_type]
                    if dep_descriptor.lifetime == ServiceLifetime.SCOPED:
                        kwargs[param_name] = scope.get_service(service_type)
                    else:
                        kwargs[param_name] = self.get_service(service_type)
                else:
                    if self.is_registered(service_type):
                        kwargs[param_name] = self.get_service(service_type)
                    elif param.default != param.empty:
                        kwargs[param_name] = param.default
                    else:
                        # オプション依存関係として None を設定
                        kwargs[param_name] = None

        return descriptor.implementation_type(**kwargs)

    def _analyze_dependencies(self, func: Callable) -> List[Type]:
        """関数の依存関係分析"""
        dependencies = []

        if func is None:
            return dependencies

        sig = inspect.signature(func)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            if param.annotation != param.empty:
                dependencies.append(param.annotation)

        return dependencies

    def _get_parameter_name(self, service_type: Type) -> str:
        """型からパラメーター名を推定"""
        return service_type.__name__.lower().replace('i', '', 1) if service_type.__name__.startswith('I') else service_type.__name__.lower()

# デコレーター

def inject_dependencies(container: Optional[DIContainer] = None):
    """依存関係注入デコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal container
            if container is None:
                container = get_global_container()

            sig = inspect.signature(func)

            # 依存関係解決
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param.annotation != param.empty:
                    service_type = param.annotation
                    if container.is_registered(service_type):
                        kwargs[param_name] = container.get_service(service_type)

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal container
            if container is None:
                container = get_global_container()

            sig = inspect.signature(func)

            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param.annotation != param.empty:
                    service_type = param.annotation
                    if container.is_registered(service_type):
                        kwargs[param_name] = container.get_service(service_type)

            return await func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator

def singleton(cls: Type[T]) -> Type[T]:
    """シングルトンクラスデコレーター"""
    container = get_global_container()
    container.register_singleton(cls, cls)
    return cls

def transient(cls: Type[T]) -> Type[T]:
    """一時的クラスデコレーター"""
    container = get_global_container()
    container.register_transient(cls, cls)
    return cls

# グローバルコンテナー

_global_container: Optional[DIContainer] = None

def get_global_container() -> DIContainer:
    """グローバルDIコンテナー取得"""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container

def set_global_container(container: DIContainer):
    """グローバルDIコンテナー設定"""
    global _global_container
    _global_container = container

def configure_services(configure_func: Callable[[DIContainer], None]) -> DIContainer:
    """サービス設定"""
    container = get_global_container()
    configure_func(container)
    return container
