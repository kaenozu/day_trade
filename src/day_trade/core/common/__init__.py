"""
共通コンポーネント

全アーキテクチャで共有される基底クラスとユーティリティ。
"""

from .service_instance import (
    ServiceStatus, ServiceType, ServiceEndpoint, ServiceCapability, 
    ServiceHealthMetrics, BaseServiceInstance, MicroserviceInstance,
    ApiGatewayInstance, DatabaseInstance, ServiceInstanceFactory
)

__all__ = [
    'ServiceStatus', 'ServiceType', 'ServiceEndpoint', 'ServiceCapability',
    'ServiceHealthMetrics', 'BaseServiceInstance', 'MicroserviceInstance',
    'ApiGatewayInstance', 'DatabaseInstance', 'ServiceInstanceFactory'
]