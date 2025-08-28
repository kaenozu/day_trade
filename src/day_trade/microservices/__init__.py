#!/usr/bin/env python3
"""
Next-Generation Microservices Architecture
分散システム対応の次世代マイクロサービスアーキテクチャ
"""

__version__ = "4.0.0"
__description__ = "Distributed Microservices with Hexagonal Architecture"

# Service Registry
from .registry import ServiceRegistry, ServiceDiscovery
from .gateway import APIGateway, RequestRouter
from .communication import MessageBroker, EventBus, ServiceMesh

# Core Services
from .trading_engine import TradingEngineService
from .market_data import MarketDataService
from .portfolio import PortfolioService
from .risk_management import RiskManagementService
from .notification import NotificationService
from .analytics import AnalyticsService

# Infrastructure Services
from .config import ConfigurationService
from .monitoring import MonitoringService
from .security import SecurityService

__all__ = [
    # Service Infrastructure
    'ServiceRegistry', 'ServiceDiscovery', 'APIGateway', 'RequestRouter',
    'MessageBroker', 'EventBus', 'ServiceMesh',
    
    # Domain Services
    'TradingEngineService', 'MarketDataService', 'PortfolioService',
    'RiskManagementService', 'NotificationService', 'AnalyticsService',
    
    # Infrastructure Services
    'ConfigurationService', 'MonitoringService', 'SecurityService'
]