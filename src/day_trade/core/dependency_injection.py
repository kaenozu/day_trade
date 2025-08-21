#!/usr/bin/env python3
"""
依存性注入システム - Dependency Injection Container
Issue #918 項目3対応: 依存性注入パターンの導入

このモジュールは以下の機能を提供します:
1. 軽量なDIコンテナ
2. インターフェース定義
3. シングルトン管理
4. ライフサイクル管理
"""

from abc import ABC, abstractmethod
from typing import Type, TypeVar, Dict, Any, Optional, Callable, Protocol
from enum import Enum
import threading
from ..utils.logging_config import get_context_logger

T = TypeVar('T')

class LifecycleType(Enum):
    """ライフサイクルタイプ"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class Injectable(Protocol):
    """インジェクタブルなクラスのプロトコル"""
    pass


class DIContainer:
    """軽量な依存性注入コンテナ"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._lifecycles: Dict[str, LifecycleType] = {}
        self._lock = threading.RLock()
        self.logger = get_context_logger(__name__, "DIContainer")

    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """シングルトンとして登録"""
        key = self._get_key(interface)
        with self._lock:
            self._services[key] = implementation
            self._lifecycles[key] = LifecycleType.SINGLETON
        self.logger.debug(f"Registered singleton: {key}")
        return self

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """トランジェントとして登録（毎回新しいインスタンス）"""
        key = self._get_key(interface)
        with self._lock:
            self._services[key] = implementation
            self._lifecycles[key] = LifecycleType.TRANSIENT
        self.logger.debug(f"Registered transient: {key}")
        return self

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'DIContainer':
        """ファクトリーとして登録"""
        key = self._get_key(interface)
        with self._lock:
            self._factories[key] = factory
            self._lifecycles[key] = LifecycleType.TRANSIENT
        self.logger.debug(f"Registered factory: {key}")
        return self

    def register_instance(self, interface: Type[T], instance: T) -> 'DIContainer':
        """インスタンスを直接登録（シングルトン扱い）"""
        key = self._get_key(interface)
        with self._lock:
            self._singletons[key] = instance
            self._lifecycles[key] = LifecycleType.SINGLETON
        self.logger.debug(f"Registered instance: {key}")
        return self

    def resolve(self, interface: Type[T]) -> T:
        """依存性を解決してインスタンスを取得"""
        key = self._get_key(interface)

        # シングルトンキャッシュをチェック
        with self._lock:
            if key in self._singletons:
                return self._singletons[key]

        # ファクトリーをチェック
        if key in self._factories:
            instance = self._factories[key]()
            lifecycle = self._lifecycles.get(key, LifecycleType.TRANSIENT)

            if lifecycle == LifecycleType.SINGLETON:
                with self._lock:
                    self._singletons[key] = instance

            return instance

        # 登録されたサービスをチェック
        if key in self._services:
            service_class = self._services[key]
            instance = self._create_instance(service_class)
            lifecycle = self._lifecycles.get(key, LifecycleType.TRANSIENT)

            if lifecycle == LifecycleType.SINGLETON:
                with self._lock:
                    self._singletons[key] = instance

            return instance

        # 登録されていない場合は例外を発生
        raise ValueError(f"Service not registered: {interface}")

    def _create_instance(self, service_class: Type[T]) -> T:
        """インスタンスを作成（簡易版）"""
        try:
            # コンストラクターの引数を分析して依存性注入
            import inspect
            sig = inspect.signature(service_class.__init__)
            params = list(sig.parameters.values())[1:]  # selfを除く

            kwargs = {}
            for param in params:
                if param.annotation != param.empty:
                    try:
                        kwargs[param.name] = self.resolve(param.annotation)
                    except ValueError:
                        # 解決できない依存性はデフォルト値を使用
                        if param.default != param.empty:
                            kwargs[param.name] = param.default
                        else:
                            self.logger.warning(f"Cannot resolve dependency: {param.annotation}")

            return service_class(**kwargs)
        except Exception as e:
            # フォールバック: 引数なしで作成
            self.logger.warning(f"Fallback to no-arg constructor for {service_class}: {e}")
            return service_class()

    def _get_key(self, interface: Type) -> str:
        """インターフェースからキーを生成"""
        return f"{interface.__module__}.{interface.__name__}"

    def is_registered(self, interface: Type) -> bool:
        """サービスが登録されているかチェック"""
        key = self._get_key(interface)
        return key in self._services or key in self._factories or key in self._singletons

    def clear(self):
        """全ての登録をクリア"""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._singletons.clear()
            self._lifecycles.clear()
        self.logger.info("Container cleared")


# グローバルコンテナインスタンス
_container = DIContainer()

def get_container() -> DIContainer:
    """グローバルDIコンテナを取得"""
    return _container


# 便利なデコレーター
def injectable(cls: Type[T]) -> Type[T]:
    """インジェクタブルマーカーデコレーター"""
    cls._injectable = True
    return cls


def singleton(interface: Optional[Type] = None):
    """シングルトン登録デコレーター"""
    def decorator(cls: Type[T]) -> Type[T]:
        target_interface = interface or cls
        get_container().register_singleton(target_interface, cls)
        return cls
    return decorator


def transient(interface: Optional[Type] = None):
    """トランジェント登録デコレーター"""
    def decorator(cls: Type[T]) -> Type[T]:
        target_interface = interface or cls
        get_container().register_transient(target_interface, cls)
        return cls
    return decorator


# サービス抽象クラス
class IConfigurationService(ABC):
    """設定サービスインターフェース"""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_analysis_config(self) -> Dict[str, Any]:
        pass


class ILoggingService(ABC):
    """ログサービスインターフェース"""

    @abstractmethod
    def get_logger(self, name: str, context: str = None):
        pass


class IAnalyzerService(ABC):
    """分析サービスインターフェース"""

    @abstractmethod
    def analyze(self, symbol: str, **kwargs):
        pass


class IDashboardService(ABC):
    """ダッシュボードサービスインターフェース"""

    @abstractmethod
    def start_dashboard(self, **kwargs):
        pass


class IDataProviderService(ABC):
    """データプロバイダーサービスインターフェース"""

    @abstractmethod
    def get_stock_data(self, symbol: str, period: str):
        pass