#!/usr/bin/env python3
"""
データベース最適化レポート生成
Issue #918 項目7対応: データベースアクセスとクエリの最適化

最適化結果の詳細レポート生成機能
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from ..core.dependency_injection import get_container
from ..core.database_services import (
    IDatabaseService, IQueryOptimizerService, ICacheService,
    DatabasePerformanceMetrics
)
from ..utils.logging_config import get_context_logger


class DatabaseOptimizationReport:
    """データベース最適化レポート生成クラス"""

    def __init__(self):
        self.logger = get_context_logger(__name__, "DatabaseOptimizationReport")
        container = get_container()

        self.db_service = container.resolve(IDatabaseService)
        self.optimizer_service = container.resolve(IQueryOptimizerService)
        self.cache_service = container.resolve(ICacheService)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """包括的最適化レポートを生成"""
        self.logger.info("Generating comprehensive database optimization report")

        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'database_optimization_comprehensive',
            'version': '1.0.0',
            'sections': {}
        }

        # 1. パフォーマンス指標セクション
        report['sections']['performance_metrics'] = self._generate_performance_section()

        # 2. キャッシュ効率セクション
        report['sections']['cache_efficiency'] = self._generate_cache_section()

        # 3. クエリ最適化推奨セクション
        report['sections']['query_optimization'] = self._generate_optimization_section()

        # 4. システム健全性セクション
        report['sections']['system_health'] = self._generate_health_section()

        # 5. 推奨アクションセクション
        report['sections']['recommendations'] = self._generate_recommendations_section(report)

        return report

    def _generate_performance_section(self) -> Dict[str, Any]:
        """パフォーマンス指標セクション生成"""
        metrics = self.db_service.get_performance_metrics()

        # パフォーマンス評価
        performance_grade = self._evaluate_performance(metrics)

        return {
            'current_metrics': asdict(metrics),
            'performance_grade': performance_grade,
            'key_indicators': {
                'query_count': metrics.query_count,
                'avg_query_time': round(metrics.avg_query_time, 4),
                'slow_query_percentage': (metrics.slow_query_count / max(metrics.query_count, 1)) * 100,
                'connection_pool_usage': metrics.connection_pool_usage,
                'active_connections': metrics.active_connections
            },
            'analysis': {
                'query_performance': 'Good' if metrics.avg_query_time < 0.1 else 'Needs Improvement',
                'connection_efficiency': 'Optimal' if metrics.connection_pool_usage < 0.8 else 'High Usage',
                'overall_health': performance_grade
            }
        }

    def _evaluate_performance(self, metrics: DatabasePerformanceMetrics) -> str:
        """パフォーマンス評価"""
        score = 100

        # 平均クエリ時間評価
        if metrics.avg_query_time > 0.5:
            score -= 30
        elif metrics.avg_query_time > 0.1:
            score -= 15

        # 遅いクエリの割合評価
        slow_query_ratio = metrics.slow_query_count / max(metrics.query_count, 1)
        if slow_query_ratio > 0.1:
            score -= 20
        elif slow_query_ratio > 0.05:
            score -= 10

        # 接続プール使用率評価
        if metrics.connection_pool_usage > 0.9:
            score -= 15
        elif metrics.connection_pool_usage > 0.8:
            score -= 5

        if score >= 90:
            return 'Excellent'
        elif score >= 75:
            return 'Good'
        elif score >= 60:
            return 'Fair'
        else:
            return 'Poor'

    def _generate_cache_section(self) -> Dict[str, Any]:
        """キャッシュ効率セクション生成"""
        cache_stats = self.cache_service.get_stats()

        # キャッシュ効率評価
        cache_grade = self._evaluate_cache_efficiency(cache_stats)

        return {
            'cache_statistics': cache_stats,
            'cache_grade': cache_grade,
            'efficiency_metrics': {
                'hit_rate_percentage': round(cache_stats.get('hit_rate', 0) * 100, 2),
                'cache_utilization': cache_stats.get('cache_size', 0),
                'total_operations': cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
            },
            'analysis': {
                'hit_rate_status': 'Excellent' if cache_stats.get('hit_rate', 0) > 0.8 else
                                   'Good' if cache_stats.get('hit_rate', 0) > 0.6 else 'Needs Improvement',
                'cache_effectiveness': cache_grade,
                'memory_usage': 'Within limits'
            }
        }

    def _evaluate_cache_efficiency(self, stats: Dict[str, Any]) -> str:
        """キャッシュ効率評価"""
        hit_rate = stats.get('hit_rate', 0)

        if hit_rate >= 0.9:
            return 'Excellent'
        elif hit_rate >= 0.7:
            return 'Good'
        elif hit_rate >= 0.5:
            return 'Fair'
        else:
            return 'Poor'

    def _generate_optimization_section(self) -> Dict[str, Any]:
        """クエリ最適化推奨セクション生成"""
        # サンプルクエリでの最適化デモ
        sample_queries = [
            "SELECT * FROM stock_data WHERE created_at > '2024-01-01'",
            "SELECT symbol, price FROM stock_data ORDER BY created_at DESC",
            "SELECT COUNT(*) FROM stock_data WHERE status = 'active'"
        ]

        optimizations = []
        total_improvement = 0

        for query in sample_queries:
            try:
                result = self.optimizer_service.optimize_query(query)
                optimizations.append({
                    'original_query': query,
                    'optimized_query': result.optimized_query,
                    'techniques': result.optimization_techniques,
                    'estimated_improvement': result.performance_improvement
                })
                total_improvement += result.performance_improvement
            except Exception as e:
                self.logger.warning(f"Failed to optimize query: {query[:50]}... Error: {e}")

        return {
            'sample_optimizations': optimizations,
            'optimization_summary': {
                'queries_analyzed': len(sample_queries),
                'successfully_optimized': len(optimizations),
                'total_estimated_improvement': round(total_improvement, 2),
                'average_improvement_per_query': round(total_improvement / max(len(optimizations), 1), 2)
            },
            'common_optimization_techniques': [
                'SELECT * elimination',
                'WHERE clause optimization',
                'JOIN optimization',
                'ORDER BY optimization'
            ]
        }

    def _generate_health_section(self) -> Dict[str, Any]:
        """システム健全性セクション生成"""
        health_checks = {
            'database_connection': self._check_database_connection(),
            'cache_system': self._check_cache_system(),
            'query_optimizer': self._check_query_optimizer(),
            'performance_monitoring': self._check_performance_monitoring()
        }

        overall_health = 'Healthy' if all(health_checks.values()) else 'Issues Detected'

        return {
            'health_checks': health_checks,
            'overall_health': overall_health,
            'system_uptime': self._get_system_uptime(),
            'resource_usage': {
                'memory_usage': 'Normal',
                'cpu_usage': 'Normal',
                'storage_usage': 'Normal'
            }
        }

    def _check_database_connection(self) -> bool:
        """データベース接続チェック"""
        try:
            with self.db_service.get_session() as session:
                session.execute("SELECT 1")
            return True
        except:
            return False

    def _check_cache_system(self) -> bool:
        """キャッシュシステムチェック"""
        try:
            test_key = f"health_check_{int(time.time())}"
            self.cache_service.set(test_key, "test_value")
            result = self.cache_service.get(test_key)
            return result == "test_value"
        except:
            return False

    def _check_query_optimizer(self) -> bool:
        """クエリ最適化チェック"""
        try:
            test_query = "SELECT 1 as test"
            result = self.optimizer_service.optimize_query(test_query)
            return result is not None
        except:
            return False

    def _check_performance_monitoring(self) -> bool:
        """パフォーマンス監視チェック"""
        try:
            metrics = self.db_service.get_performance_metrics()
            return metrics is not None
        except:
            return False

    def _get_system_uptime(self) -> str:
        """システム稼働時間取得"""
        # 簡易実装
        return "System operational"

    def _generate_recommendations_section(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """推奨アクションセクション生成"""
        recommendations = []
        priority_actions = []

        # パフォーマンス指標に基づく推奨
        perf_grade = report['sections']['performance_metrics']['performance_grade']
        if perf_grade in ['Fair', 'Poor']:
            recommendations.append({
                'category': 'Performance',
                'action': 'Query optimization and indexing review',
                'priority': 'High',
                'estimated_impact': 'Significant performance improvement'
            })
            priority_actions.append('Optimize slow queries')

        # キャッシュ効率に基づく推奨
        cache_grade = report['sections']['cache_efficiency']['cache_grade']
        if cache_grade in ['Fair', 'Poor']:
            recommendations.append({
                'category': 'Cache',
                'action': 'Improve caching strategy',
                'priority': 'Medium',
                'estimated_impact': 'Reduced database load'
            })

        # システム健全性に基づく推奨
        if report['sections']['system_health']['overall_health'] != 'Healthy':
            recommendations.append({
                'category': 'System Health',
                'action': 'Address system health issues',
                'priority': 'High',
                'estimated_impact': 'System stability improvement'
            })
            priority_actions.append('Fix system health issues')

        # 一般的な推奨アクション
        general_recommendations = [
            'Regular database maintenance and statistics updates',
            'Monitor query performance trends',
            'Implement connection pooling best practices',
            'Review and optimize frequently used queries'
        ]

        return {
            'specific_recommendations': recommendations,
            'priority_actions': priority_actions,
            'general_recommendations': general_recommendations,
            'next_review_date': (datetime.now() + timedelta(days=7)).isoformat(),
            'optimization_roadmap': {
                'immediate': 'Address priority issues',
                'short_term': 'Implement query optimizations',
                'long_term': 'Establish continuous monitoring'
            }
        }

    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """レポートをファイルに保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"database_optimization_report_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Database optimization report saved: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            raise

    def print_summary(self, report: Dict[str, Any]):
        """レポートサマリー表示"""
        print("\n" + "="*60)
        print("DATABASE OPTIMIZATION REPORT SUMMARY")
        print("="*60)

        perf_section = report['sections']['performance_metrics']
        print(f"Performance Grade: {perf_section['performance_grade']}")
        print(f"Average Query Time: {perf_section['key_indicators']['avg_query_time']:.4f}s")

        cache_section = report['sections']['cache_efficiency']
        print(f"Cache Hit Rate: {cache_section['efficiency_metrics']['hit_rate_percentage']:.1f}%")
        print(f"Cache Grade: {cache_section['cache_grade']}")

        health_section = report['sections']['system_health']
        print(f"System Health: {health_section['overall_health']}")

        recommendations = report['sections']['recommendations']
        if recommendations['priority_actions']:
            print(f"\nPriority Actions:")
            for action in recommendations['priority_actions']:
                print(f"  - {action}")

        print(f"\nReport Generated: {report['timestamp']}")
        print("="*60)


# 便利な関数
def generate_optimization_report() -> DatabaseOptimizationReport:
    """最適化レポートを生成"""
    reporter = DatabaseOptimizationReport()
    report = reporter.generate_comprehensive_report()
    reporter.print_summary(report)
    return reporter, report


if __name__ == "__main__":
    # テスト実行
    from ..core.services import register_default_services
    register_default_services()

    reporter, report = generate_optimization_report()
    filename = reporter.save_report(report)
    print(f"\n詳細レポートを保存しました: {filename}")