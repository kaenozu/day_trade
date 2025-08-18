#!/usr/bin/env python3
"""
データベースサービス - 依存性注入版データベース最適化
Issue #918 項目7対応: データベースアクセスとクエリの最適化

データベース最適化サービスの実装
"""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator, AsyncGenerator
from enum import Enum
import threading

from sqlalchemy import create_engine, event, text, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from .dependency_injection import (
    IConfigurationService, injectable, singleton, get_container
)
from ..models.database import DatabaseConfig, DatabaseManager
from ..utils.logging_config import get_context_logger


class CacheStrategy(Enum):
    """キャッシュ戦略"""
    LRU = "lru"
    LFU = "lfu" 
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ConnectionPoolStrategy(Enum):
    """接続プール戦略"""
    STANDARD = "standard"
    OPTIMIZED = "optimized"
    HIGH_PERFORMANCE = "high_performance"


@dataclass
class DatabasePerformanceMetrics:
    """データベースパフォーマンス指標"""
    query_count: int = 0
    avg_query_time: float = 0.0
    slow_query_count: int = 0
    cache_hit_rate: float = 0.0
    connection_pool_usage: float = 0.0
    active_connections: int = 0
    total_connections: int = 0
    deadlock_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QueryOptimizationResult:
    """クエリ最適化結果"""
    original_query: str
    optimized_query: str
    optimization_techniques: List[str]
    performance_improvement: float
    execution_time_before: float
    execution_time_after: float


@dataclass
class IndexRecommendation:
    """インデックス推奨"""
    table_name: str
    column_names: List[str]
    index_type: str
    estimated_benefit: float
    creation_cost: float
    maintenance_overhead: float


class IDatabaseService(ABC):
    """データベースサービスインターフェース"""

    @abstractmethod
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """セッション取得"""
        pass

    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """クエリ実行"""
        pass

    @abstractmethod
    def get_performance_metrics(self) -> DatabasePerformanceMetrics:
        """パフォーマンス指標取得"""
        pass


class IQueryOptimizerService(ABC):
    """クエリ最適化サービスインターフェース"""

    @abstractmethod
    def optimize_query(self, query: str) -> QueryOptimizationResult:
        """クエリ最適化"""
        pass

    @abstractmethod
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """クエリパフォーマンス分析"""
        pass

    @abstractmethod
    def get_index_recommendations(self, table_name: str) -> List[IndexRecommendation]:
        """インデックス推奨取得"""
        pass


class ICacheService(ABC):
    """キャッシュサービスインターフェース"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """キャッシュ設定"""
        pass

    @abstractmethod
    def clear(self):
        """キャッシュクリア"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        pass


@singleton(IDatabaseService)
@injectable
class OptimizedDatabaseService(IDatabaseService):
    """最適化データベースサービス実装"""

    def __init__(self, 
                 config_service: IConfigurationService,
                 cache_service: Optional['ICacheService'] = None):
        self.config_service = config_service
        self.cache_service = cache_service
        self.logger = get_context_logger(__name__, "OptimizedDatabaseService")
        
        # パフォーマンス指標
        self._metrics = DatabasePerformanceMetrics()
        self._query_times = []
        self._slow_query_threshold = 1.0  # 1秒
        
        # データベース設定
        self._db_config = self._create_optimized_config()
        self._db_manager = DatabaseManager(self._db_config)
        
        # 接続プール監視
        self._setup_pool_monitoring()
        
        self.logger.info("OptimizedDatabaseService initialized")

    def _create_optimized_config(self) -> DatabaseConfig:
        """最適化されたデータベース設定を作成"""
        config = self.config_service.get_config()
        db_settings = config.get('database', {})
        
        # 最適化されたプール設定
        pool_settings = db_settings.get('pool', {})
        
        return DatabaseConfig(
            database_url=db_settings.get('url', 'sqlite:///data/trading.db'),
            pool_size=pool_settings.get('size', 20),  # 増量
            max_overflow=pool_settings.get('max_overflow', 30),  # 増量
            pool_timeout=pool_settings.get('timeout', 60),
            pool_recycle=pool_settings.get('recycle', 3600),
            connect_args={
                'check_same_thread': False,
                'timeout': 20,
                'isolation_level': None  # autocommit mode
            }
        )

    def _setup_pool_monitoring(self):
        """接続プール監視設定"""
        if self._db_manager.engine:
            @event.listens_for(self._db_manager.engine, "connect")
            def on_connect(dbapi_conn, connection_record):
                self._metrics.total_connections += 1
                
                # SQLite最適化
                if 'sqlite' in str(self._db_manager.engine.url):
                    dbapi_conn.execute("PRAGMA journal_mode=WAL")
                    dbapi_conn.execute("PRAGMA synchronous=NORMAL")
                    dbapi_conn.execute("PRAGMA cache_size=10000")
                    dbapi_conn.execute("PRAGMA temp_store=memory")
                    dbapi_conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            @event.listens_for(self._db_manager.engine, "checkout")
            def on_checkout(dbapi_conn, connection_record, connection_proxy):
                self._metrics.active_connections += 1

            @event.listens_for(self._db_manager.engine, "checkin") 
            def on_checkin(dbapi_conn, connection_record):
                if self._metrics.active_connections > 0:
                    self._metrics.active_connections -= 1

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """最適化されたセッション取得"""
        session = self._db_manager.session_factory()
        session_start = time.time()
        
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session_time = time.time() - session_start
            self._update_metrics(session_time)
            session.close()

    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """最適化クエリ実行"""
        start_time = time.time()
        
        # キャッシュチェック
        cache_key = None
        if self.cache_service and 'SELECT' in query.upper():
            cache_key = f"query:{hash(query)}:{hash(str(params or {}))}"
            cached_result = self.cache_service.get(cache_key)
            if cached_result:
                self.logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        try:
            with self.get_session() as session:
                if params:
                    result = session.execute(text(query), params)
                else:
                    result = session.execute(text(query))
                
                # 結果の取得方法を最適化
                if result.returns_rows:
                    data = result.fetchall()
                else:
                    data = result.rowcount
                
                # キャッシュに保存
                if cache_key and self.cache_service:
                    self.cache_service.set(cache_key, data, ttl=300)  # 5分
                
                execution_time = time.time() - start_time
                self._log_query_performance(query, execution_time)
                
                return data
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Query execution failed: {query[:50]}..., Error: {e}")
            self._log_query_performance(query, execution_time, error=True)
            raise

    def _update_metrics(self, session_time: float):
        """パフォーマンス指標更新"""
        self._metrics.query_count += 1
        self._query_times.append(session_time)
        
        # 平均クエリ時間更新
        if len(self._query_times) > 100:
            self._query_times.pop(0)
        
        self._metrics.avg_query_time = sum(self._query_times) / len(self._query_times)
        
        # 遅いクエリカウント
        if session_time > self._slow_query_threshold:
            self._metrics.slow_query_count += 1
        
        # 接続プール使用率
        pool = getattr(self._db_manager.engine.pool, '_creator', None)
        if hasattr(self._db_manager.engine.pool, 'size'):
            pool_size = self._db_manager.engine.pool.size()
            checked_out = getattr(self._db_manager.engine.pool, 'checkedout', lambda: 0)()
            self._metrics.connection_pool_usage = checked_out / pool_size if pool_size > 0 else 0.0

    def _log_query_performance(self, query: str, execution_time: float, error: bool = False):
        """クエリパフォーマンスログ"""
        if execution_time > self._slow_query_threshold or error:
            self.logger.warning(
                f"Slow query detected: {query[:100]}...",
                extra={
                    'execution_time': execution_time,
                    'threshold': self._slow_query_threshold,
                    'error': error
                }
            )

    def get_performance_metrics(self) -> DatabasePerformanceMetrics:
        """パフォーマンス指標取得"""
        self._metrics.timestamp = datetime.now()
        
        # キャッシュヒット率
        if self.cache_service:
            cache_stats = self.cache_service.get_stats()
            self._metrics.cache_hit_rate = cache_stats.get('hit_rate', 0.0)
        
        return self._metrics


@singleton(IQueryOptimizerService)
@injectable 
class QueryOptimizerService(IQueryOptimizerService):
    """クエリ最適化サービス実装"""

    def __init__(self, db_service: IDatabaseService):
        self.db_service = db_service
        self.logger = get_context_logger(__name__, "QueryOptimizerService")
        self._optimization_rules = self._load_optimization_rules()
        
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """最適化ルール読み込み"""
        return {
            'select_optimization': {
                'avoid_select_star': True,
                'use_limit_for_testing': True,
                'prefer_exists_over_in': True
            },
            'join_optimization': {
                'prefer_inner_join': True,
                'use_proper_join_conditions': True,
                'avoid_cartesian_products': True
            },
            'index_hints': {
                'suggest_covering_indexes': True,
                'recommend_composite_indexes': True
            }
        }

    def optimize_query(self, query: str) -> QueryOptimizationResult:
        """クエリ最適化実装"""
        start_time = time.time()
        original_query = query.strip()
        optimized_query = original_query
        techniques = []
        
        try:
            # 1. SELECT * の最適化
            if 'SELECT *' in original_query.upper():
                optimized_query = self._optimize_select_star(optimized_query)
                techniques.append('select_star_removal')
            
            # 2. JOIN最適化
            if 'JOIN' in original_query.upper():
                optimized_query = self._optimize_joins(optimized_query)
                techniques.append('join_optimization')
            
            # 3. WHERE句最適化
            if 'WHERE' in original_query.upper():
                optimized_query = self._optimize_where_clause(optimized_query)
                techniques.append('where_optimization')
            
            # 4. ORDER BY最適化
            if 'ORDER BY' in original_query.upper():
                optimized_query = self._optimize_order_by(optimized_query)
                techniques.append('order_by_optimization')
            
            optimization_time = time.time() - start_time
            
            return QueryOptimizationResult(
                original_query=original_query,
                optimized_query=optimized_query,
                optimization_techniques=techniques,
                performance_improvement=self._estimate_improvement(techniques),
                execution_time_before=0.0,  # 実測が必要
                execution_time_after=optimization_time
            )
            
        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            return QueryOptimizationResult(
                original_query=original_query,
                optimized_query=original_query,
                optimization_techniques=[],
                performance_improvement=0.0,
                execution_time_before=0.0,
                execution_time_after=0.0
            )

    def _optimize_select_star(self, query: str) -> str:
        """SELECT * の最適化"""
        # 実際の実装では、テーブルスキーマを取得して必要なカラムのみ選択
        self.logger.info("Optimizing SELECT * query")
        return query.replace('SELECT *', 'SELECT id, created_at, updated_at')  # 例

    def _optimize_joins(self, query: str) -> str:
        """JOIN最適化"""
        # INNER JOINの明示化、適切な結合条件の確認など
        return query

    def _optimize_where_clause(self, query: str) -> str:
        """WHERE句最適化"""
        # インデックスを活用できる条件への変更など
        return query

    def _optimize_order_by(self, query: str) -> str:
        """ORDER BY最適化"""
        # インデックスを活用できるソート順への変更など
        return query

    def _estimate_improvement(self, techniques: List[str]) -> float:
        """パフォーマンス改善推定"""
        improvement_map = {
            'select_star_removal': 10.0,
            'join_optimization': 25.0,
            'where_optimization': 30.0,
            'order_by_optimization': 15.0
        }
        return sum(improvement_map.get(tech, 0.0) for tech in techniques)

    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """クエリパフォーマンス分析"""
        analysis = {
            'query': query,
            'estimated_cost': self._estimate_query_cost(query),
            'suggested_indexes': self._suggest_indexes(query),
            'potential_issues': self._identify_issues(query),
            'optimization_recommendations': self._get_recommendations(query)
        }
        return analysis

    def _estimate_query_cost(self, query: str) -> float:
        """クエリコスト推定"""
        # 簡易版: クエリの複雑度に基づく推定
        cost = 1.0
        if 'JOIN' in query.upper():
            cost *= 2.0
        if 'ORDER BY' in query.upper():
            cost *= 1.5
        if 'GROUP BY' in query.upper():
            cost *= 1.3
        return cost

    def _suggest_indexes(self, query: str) -> List[str]:
        """インデックス提案"""
        suggestions = []
        # WHERE句の条件からインデックス提案を生成
        if 'WHERE' in query.upper():
            suggestions.append("Consider adding index on WHERE clause columns")
        return suggestions

    def _identify_issues(self, query: str) -> List[str]:
        """問題点特定"""
        issues = []
        if 'SELECT *' in query.upper():
            issues.append("Using SELECT * can be inefficient")
        if query.count('JOIN') > 3:
            issues.append("Complex joins may need optimization")
        return issues

    def _get_recommendations(self, query: str) -> List[str]:
        """最適化推奨"""
        recommendations = []
        if 'SELECT *' in query.upper():
            recommendations.append("Specify only needed columns instead of SELECT *")
        return recommendations

    def get_index_recommendations(self, table_name: str) -> List[IndexRecommendation]:
        """インデックス推奨取得"""
        # テーブル解析に基づくインデックス推奨
        recommendations = []
        
        try:
            with self.db_service.get_session() as session:
                # テーブル構造とクエリパターンを分析してインデックスを推奨
                inspector = inspect(session.bind)
                columns = inspector.get_columns(table_name)
                
                # よく使用される条件に基づくインデックス推奨
                for column in columns:
                    if column['name'] in ['created_at', 'updated_at', 'status']:
                        recommendations.append(IndexRecommendation(
                            table_name=table_name,
                            column_names=[column['name']],
                            index_type='btree',
                            estimated_benefit=15.0,
                            creation_cost=2.0,
                            maintenance_overhead=5.0
                        ))
                        
        except Exception as e:
            self.logger.error(f"Failed to get index recommendations for {table_name}: {e}")
            
        return recommendations


@singleton(ICacheService)
@injectable
class InMemoryCacheService(ICacheService):
    """インメモリキャッシュサービス実装"""

    def __init__(self):
        self.logger = get_context_logger(__name__, "InMemoryCacheService") 
        self._cache = {}
        self._ttl_cache = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        self._lock = threading.RLock()
        self._max_size = 1000  # 最大キャッシュ数

    def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        with self._lock:
            # TTL確認
            if key in self._ttl_cache:
                if datetime.now() > self._ttl_cache[key]:
                    del self._cache[key]
                    del self._ttl_cache[key]
                    self._stats['misses'] += 1
                    return None
            
            if key in self._cache:
                self._stats['hits'] += 1
                return self._cache[key]
            else:
                self._stats['misses'] += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """キャッシュ設定"""
        with self._lock:
            # サイズ制限チェック
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = value
            
            if ttl:
                self._ttl_cache[key] = datetime.now() + timedelta(seconds=ttl)
            elif key in self._ttl_cache:
                del self._ttl_cache[key]
            
            self._stats['sets'] += 1

    def _evict_lru(self):
        """LRU退避"""
        # 簡易版: 古いエントリを削除
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key] 
            if oldest_key in self._ttl_cache:
                del self._ttl_cache[oldest_key]

    def clear(self):
        """キャッシュクリア"""
        with self._lock:
            self._cache.clear()
            self._ttl_cache.clear()
            self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache),
                'ttl_entries': len(self._ttl_cache)
            }


def register_database_services():
    """データベースサービスを登録"""
    container = get_container()
    
    # キャッシュサービス
    if not container.is_registered(ICacheService):
        container.register_singleton(ICacheService, InMemoryCacheService)
    
    # データベースサービス  
    if not container.is_registered(IDatabaseService):
        container.register_singleton(IDatabaseService, OptimizedDatabaseService)
    
    # クエリ最適化サービス
    if not container.is_registered(IQueryOptimizerService):
        container.register_singleton(IQueryOptimizerService, QueryOptimizerService)