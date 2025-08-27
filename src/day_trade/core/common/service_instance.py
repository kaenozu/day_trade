"""
統一サービスインスタンス定義

全アーキテクチャコンポーネント共通のサービスインスタンス基底クラス。
重複したServiceInstanceクラスを統合し、一貫性のあるサービス管理を提供。
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import threading


# ================================
# サービス状態とメタデータ
# ================================

class ServiceStatus(Enum):
    """サービス状態"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class ServiceType(Enum):
    """サービスタイプ"""
    WEB_SERVICE = "web_service"
    API_GATEWAY = "api_gateway"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    MICROSERVICE = "microservice"
    BACKGROUND_SERVICE = "background_service"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ServiceEndpoint:
    """サービスエンドポイント"""
    path: str
    method: str = "GET"
    description: str = ""
    timeout: int = 30
    auth_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'method': self.method,
            'description': self.description,
            'timeout': self.timeout,
            'auth_required': self.auth_required
        }


@dataclass
class ServiceCapability:
    """サービス機能"""
    name: str
    version: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'parameters': self.parameters
        }


@dataclass
class ServiceHealthMetrics:
    """サービスヘルスメトリクス"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    request_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'response_time': self.response_time,
            'error_rate': self.error_rate,
            'request_count': self.request_count,
            'last_updated': self.last_updated.isoformat()
        }


# ================================
# 統一サービスインスタンス基底クラス
# ================================

@dataclass
class BaseServiceInstance:
    """統一サービスインスタンス基底クラス"""
    
    # 基本識別情報
    service_name: str
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    service_type: ServiceType = ServiceType.MICROSERVICE
    
    # ネットワーク情報
    host: str = "localhost"
    port: int = 8080
    protocol: str = "http"
    
    # バージョンと設定
    version: str = "1.0.0"
    build_info: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # 状態管理
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_heartbeat: datetime = field(default_factory=datetime.now)
    start_time: datetime = field(default_factory=datetime.now)
    
    # 機能とエンドポイント
    capabilities: List[ServiceCapability] = field(default_factory=list)
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    
    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # ヘルスとメトリクス
    health_check_url: str = ""
    health_metrics: ServiceHealthMetrics = field(default_factory=ServiceHealthMetrics)
    
    def __post_init__(self):
        if not self.health_check_url:
            self.health_check_url = f"{self.protocol}://{self.host}:{self.port}/health"
    
    # ================================
    # 基本操作
    # ================================
    
    def get_full_url(self, path: str = "") -> str:
        """完全URL取得"""
        base_url = f"{self.protocol}://{self.host}:{self.port}"
        if path:
            path = path.lstrip('/')
            return f"{base_url}/{path}"
        return base_url
    
    def update_heartbeat(self):
        """ハートビート更新"""
        self.last_heartbeat = datetime.now()
    
    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """健全性チェック"""
        if self.status != ServiceStatus.HEALTHY:
            return False
        
        # ハートビートタイムアウトチェック
        cutoff_time = datetime.now() - timedelta(seconds=timeout_seconds)
        return self.last_heartbeat > cutoff_time
    
    def get_uptime(self) -> timedelta:
        """稼働時間取得"""
        return datetime.now() - self.start_time
    
    def add_capability(self, capability: ServiceCapability):
        """機能追加"""
        self.capabilities.append(capability)
    
    def add_endpoint(self, endpoint: ServiceEndpoint):
        """エンドポイント追加"""
        self.endpoints.append(endpoint)
    
    def add_tag(self, tag: str):
        """タグ追加"""
        self.tags.add(tag)
    
    def has_capability(self, capability_name: str) -> bool:
        """機能チェック"""
        return any(cap.name == capability_name for cap in self.capabilities)
    
    def get_endpoint(self, path: str, method: str = "GET") -> Optional[ServiceEndpoint]:
        """エンドポイント取得"""
        for endpoint in self.endpoints:
            if endpoint.path == path and endpoint.method == method:
                return endpoint
        return None
    
    # ================================
    # メトリクス更新
    # ================================
    
    def update_health_metrics(
        self,
        cpu_usage: float = None,
        memory_usage: float = None,
        response_time: float = None,
        error_rate: float = None,
        request_count: int = None
    ):
        """ヘルスメトリクス更新"""
        if cpu_usage is not None:
            self.health_metrics.cpu_usage = cpu_usage
        if memory_usage is not None:
            self.health_metrics.memory_usage = memory_usage
        if response_time is not None:
            self.health_metrics.response_time = response_time
        if error_rate is not None:
            self.health_metrics.error_rate = error_rate
        if request_count is not None:
            self.health_metrics.request_count = request_count
        
        self.health_metrics.last_updated = datetime.now()
    
    def increment_request_count(self):
        """リクエストカウント増加"""
        self.health_metrics.request_count += 1
    
    # ================================
    # シリアライゼーション
    # ================================
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            'service_name': self.service_name,
            'instance_id': self.instance_id,
            'service_type': self.service_type.value,
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'version': self.version,
            'build_info': self.build_info,
            'status': self.status.value,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': self.get_uptime().total_seconds(),
            'health_check_url': self.health_check_url,
            'capabilities': [cap.to_dict() for cap in self.capabilities],
            'endpoints': [ep.to_dict() for ep in self.endpoints],
            'tags': list(self.tags),
            'health_metrics': self.health_metrics.to_dict()
        }
        
        # 機密情報を含める場合
        if include_sensitive:
            result['configuration'] = self.configuration
            result['metadata'] = self.metadata
        else:
            # 機密情報をフィルタリング
            safe_config = {k: v for k, v in self.configuration.items() 
                          if not any(sensitive in k.lower() 
                                   for sensitive in ['password', 'secret', 'key', 'token'])}
            result['configuration'] = safe_config
            result['metadata'] = {k: v for k, v in self.metadata.items()
                                if not any(sensitive in k.lower()
                                         for sensitive in ['password', 'secret', 'key', 'token'])}
        
        return result
    
    def to_json(self, include_sensitive: bool = False) -> str:
        """JSON形式に変換"""
        return json.dumps(self.to_dict(include_sensitive), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseServiceInstance':
        """辞書から作成"""
        instance = cls(
            service_name=data['service_name'],
            instance_id=data.get('instance_id', str(uuid.uuid4())),
            service_type=ServiceType(data.get('service_type', 'microservice')),
            host=data.get('host', 'localhost'),
            port=data.get('port', 8080),
            protocol=data.get('protocol', 'http'),
            version=data.get('version', '1.0.0'),
            build_info=data.get('build_info', ''),
            status=ServiceStatus(data.get('status', 'unknown')),
            configuration=data.get('configuration', {}),
            metadata=data.get('metadata', {}),
            tags=set(data.get('tags', [])),
            health_check_url=data.get('health_check_url', '')
        )
        
        # タイムスタンプの復元
        if 'last_heartbeat' in data:
            instance.last_heartbeat = datetime.fromisoformat(data['last_heartbeat'])
        if 'start_time' in data:
            instance.start_time = datetime.fromisoformat(data['start_time'])
        
        # 機能とエンドポイントの復元
        if 'capabilities' in data:
            instance.capabilities = [
                ServiceCapability(**cap) for cap in data['capabilities']
            ]
        if 'endpoints' in data:
            instance.endpoints = [
                ServiceEndpoint(**ep) for ep in data['endpoints']
            ]
        
        # ヘルスメトリクスの復元
        if 'health_metrics' in data:
            metrics_data = data['health_metrics']
            instance.health_metrics = ServiceHealthMetrics(
                cpu_usage=metrics_data.get('cpu_usage', 0.0),
                memory_usage=metrics_data.get('memory_usage', 0.0),
                response_time=metrics_data.get('response_time', 0.0),
                error_rate=metrics_data.get('error_rate', 0.0),
                request_count=metrics_data.get('request_count', 0),
                last_updated=datetime.fromisoformat(metrics_data.get('last_updated', datetime.now().isoformat()))
            )
        
        return instance
    
    # ================================
    # 比較とハッシュ
    # ================================
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseServiceInstance):
            return False
        return (self.service_name == other.service_name and 
                self.instance_id == other.instance_id)
    
    def __hash__(self) -> int:
        return hash((self.service_name, self.instance_id))
    
    def __str__(self) -> str:
        return f"{self.service_name}#{self.instance_id}@{self.host}:{self.port}"
    
    def __repr__(self) -> str:
        return (f"BaseServiceInstance(service_name='{self.service_name}', "
                f"instance_id='{self.instance_id}', "
                f"host='{self.host}', port={self.port}, "
                f"status='{self.status.value}')")


# ================================
# 特化したサービスインスタンス
# ================================

class MicroserviceInstance(BaseServiceInstance):
    """マイクロサービスインスタンス"""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            service_name=service_name,
            service_type=ServiceType.MICROSERVICE,
            **kwargs
        )
        
        # デフォルトエンドポイント追加
        self.add_endpoint(ServiceEndpoint("/health", "GET", "Health check"))
        self.add_endpoint(ServiceEndpoint("/metrics", "GET", "Metrics"))
        self.add_endpoint(ServiceEndpoint("/info", "GET", "Service info"))


class ApiGatewayInstance(BaseServiceInstance):
    """APIゲートウェイインスタンス"""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            service_name=service_name,
            service_type=ServiceType.API_GATEWAY,
            **kwargs
        )
        
        # APIゲートウェイ特有の機能
        self.add_capability(ServiceCapability("routing", "1.0.0", "Request routing"))
        self.add_capability(ServiceCapability("load_balancing", "1.0.0", "Load balancing"))
        self.add_capability(ServiceCapability("rate_limiting", "1.0.0", "Rate limiting"))
        self.add_capability(ServiceCapability("circuit_breaking", "1.0.0", "Circuit breaking"))


class DatabaseInstance(BaseServiceInstance):
    """データベースインスタンス"""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            service_name=service_name,
            service_type=ServiceType.DATABASE,
            **kwargs
        )
        
        # データベース特有の機能
        self.add_capability(ServiceCapability("transactions", "1.0.0", "Transaction support"))
        self.add_capability(ServiceCapability("replication", "1.0.0", "Data replication"))
        self.add_capability(ServiceCapability("backup", "1.0.0", "Backup and restore"))


# ================================
# サービスインスタンスファクトリー
# ================================

class ServiceInstanceFactory:
    """サービスインスタンスファクトリー"""
    
    _instance_classes = {
        ServiceType.MICROSERVICE: MicroserviceInstance,
        ServiceType.API_GATEWAY: ApiGatewayInstance,
        ServiceType.DATABASE: DatabaseInstance,
        ServiceType.WEB_SERVICE: BaseServiceInstance,
        ServiceType.CACHE: BaseServiceInstance,
        ServiceType.MESSAGE_QUEUE: BaseServiceInstance,
        ServiceType.BACKGROUND_SERVICE: BaseServiceInstance,
        ServiceType.EXTERNAL_SERVICE: BaseServiceInstance
    }
    
    @classmethod
    def create_instance(
        cls, 
        service_type: ServiceType, 
        service_name: str, 
        **kwargs
    ) -> BaseServiceInstance:
        """サービスインスタンス作成"""
        instance_class = cls._instance_classes.get(service_type, BaseServiceInstance)
        return instance_class(service_name=service_name, **kwargs)
    
    @classmethod
    def register_instance_class(cls, service_type: ServiceType, instance_class: type):
        """インスタンスクラス登録"""
        cls._instance_classes[service_type] = instance_class


# エクスポート
__all__ = [
    # エナムと基本クラス
    'ServiceStatus', 'ServiceType',
    'ServiceEndpoint', 'ServiceCapability', 'ServiceHealthMetrics',
    
    # メインクラス
    'BaseServiceInstance',
    
    # 特化クラス
    'MicroserviceInstance', 'ApiGatewayInstance', 'DatabaseInstance',
    
    # ファクトリー
    'ServiceInstanceFactory'
]