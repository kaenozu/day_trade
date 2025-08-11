"""
パフォーマンス最適化推奨システム

システムメトリクス、ログデータ、アプリケーション性能を分析し、
具体的な最適化提案を自動生成するインテリジェントシステム。
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .alert_system import get_alert_manager
from .metrics_collection_system import get_metrics_system


class OptimizationType(Enum):
    """最適化種別"""

    RESOURCE_OPTIMIZATION = "resource_optimization"  # リソース最適化
    PERFORMANCE_TUNING = "performance_tuning"  # パフォーマンスチューニング
    ARCHITECTURE_IMPROVEMENT = "architecture_improvement"  # アーキテクチャ改善
    CODE_OPTIMIZATION = "code_optimization"  # コード最適化
    DATABASE_OPTIMIZATION = "database_optimization"  # データベース最適化
    CACHE_OPTIMIZATION = "cache_optimization"  # キャッシュ最適化
    SCALING_RECOMMENDATION = "scaling_recommendation"  # スケーリング推奨


class Priority(Enum):
    """優先度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ImpactLevel(Enum):
    """影響レベル"""

    MINIMAL = "minimal"  # 最小限の影響
    MODERATE = "moderate"  # 中程度の影響
    SIGNIFICANT = "significant"  # 重大な影響
    TRANSFORMATIVE = "transformative"  # 変革的影響


@dataclass
class OptimizationRecommendation:
    """最適化推奨事項"""

    id: str
    title: str
    description: str
    optimization_type: OptimizationType
    priority: Priority
    impact_level: ImpactLevel
    estimated_improvement: Dict[str, float]  # メトリクス名 -> 改善率
    implementation_effort: int  # 1-10のスケール
    implementation_steps: List[str]
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    evidence: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class OptimizationOpportunity:
    """最適化機会"""

    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percentage: float
    trend_direction: str  # increasing, decreasing, stable
    confidence_score: float
    data_points: int


class PerformanceAnalyzer:
    """パフォーマンス分析器"""

    def __init__(self):
        self.metrics_system = get_metrics_system()
        self.alert_manager = get_alert_manager()
        self.analysis_window = timedelta(hours=24)

    async def analyze_system_performance(self) -> Dict[str, Any]:
        """システムパフォーマンス分析"""
        end_time = datetime.utcnow()
        start_time = end_time - self.analysis_window

        analysis_results = {
            "cpu_analysis": await self._analyze_cpu_performance(start_time, end_time),
            "memory_analysis": await self._analyze_memory_performance(
                start_time, end_time
            ),
            "disk_analysis": await self._analyze_disk_performance(start_time, end_time),
            "network_analysis": await self._analyze_network_performance(
                start_time, end_time
            ),
            "application_analysis": await self._analyze_application_performance(
                start_time, end_time
            ),
        }

        return analysis_results

    async def _analyze_cpu_performance(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """CPU性能分析"""
        cpu_data = self.metrics_system.query_metrics(
            "cpu_usage_percent", start_time, end_time
        )

        if not cpu_data:
            return {"status": "no_data"}

        values = [value for _, value, _ in cpu_data]

        if NUMPY_AVAILABLE:
            avg_usage = np.mean(values)
            max_usage = np.max(values)
            std_usage = np.std(values)
            p95_usage = np.percentile(values, 95)
        else:
            avg_usage = sum(values) / len(values)
            max_usage = max(values)
            std_usage = (sum((v - avg_usage) ** 2 for v in values) / len(values)) ** 0.5
            sorted_values = sorted(values)
            p95_usage = sorted_values[int(0.95 * len(sorted_values))]

        analysis = {
            "average_usage": avg_usage,
            "max_usage": max_usage,
            "p95_usage": p95_usage,
            "variability": std_usage,
            "data_points": len(values),
        }

        # 問題の特定
        issues = []
        if avg_usage > 70:
            issues.append("high_average_cpu")
        if max_usage > 90:
            issues.append("cpu_spikes")
        if std_usage > 20:
            issues.append("high_cpu_variability")

        analysis["issues"] = issues
        return analysis

    async def _analyze_memory_performance(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """メモリ性能分析"""
        memory_data = self.metrics_system.query_metrics(
            "memory_usage_percent", start_time, end_time
        )
        memory_bytes_data = self.metrics_system.query_metrics(
            "memory_usage_bytes", start_time, end_time
        )

        if not memory_data:
            return {"status": "no_data"}

        values = [value for _, value, _ in memory_data]

        if NUMPY_AVAILABLE:
            avg_usage = np.mean(values)
            max_usage = np.max(values)
            trend_slope = self._calculate_trend(values)
        else:
            avg_usage = sum(values) / len(values)
            max_usage = max(values)
            trend_slope = self._calculate_simple_trend(values)

        analysis = {
            "average_usage": avg_usage,
            "max_usage": max_usage,
            "trend_slope": trend_slope,
            "data_points": len(values),
        }

        # 問題の特定
        issues = []
        if avg_usage > 80:
            issues.append("high_memory_usage")
        if trend_slope > 0.1:
            issues.append("memory_leak_suspected")
        if max_usage > 95:
            issues.append("memory_pressure")

        analysis["issues"] = issues
        return analysis

    async def _analyze_disk_performance(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """ディスク性能分析"""
        disk_usage_data = self.metrics_system.query_metrics(
            "disk_usage_percent", start_time, end_time
        )
        disk_read_data = self.metrics_system.query_metrics(
            "disk_io_read_bytes", start_time, end_time
        )
        disk_write_data = self.metrics_system.query_metrics(
            "disk_io_write_bytes", start_time, end_time
        )

        analysis = {}
        issues = []

        # ディスク使用率分析
        if disk_usage_data:
            usage_values = [value for _, value, _ in disk_usage_data]
            avg_usage = sum(usage_values) / len(usage_values)
            max_usage = max(usage_values)

            analysis["disk_usage"] = {"average": avg_usage, "maximum": max_usage}

            if avg_usage > 85:
                issues.append("high_disk_usage")
            if max_usage > 95:
                issues.append("disk_space_critical")

        # I/O性能分析
        if disk_read_data and disk_write_data:
            read_values = [value for _, value, _ in disk_read_data]
            write_values = [value for _, value, _ in disk_write_data]

            # I/O率の計算（簡易版）
            total_io = sum(read_values) + sum(write_values)
            analysis["io_activity"] = {
                "total_read_bytes": sum(read_values),
                "total_write_bytes": sum(write_values),
                "total_io_bytes": total_io,
            }

        analysis["issues"] = issues
        return analysis

    async def _analyze_network_performance(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """ネットワーク性能分析"""
        sent_data = self.metrics_system.query_metrics(
            "network_io_sent_bytes", start_time, end_time
        )
        recv_data = self.metrics_system.query_metrics(
            "network_io_recv_bytes", start_time, end_time
        )

        analysis = {}
        issues = []

        if sent_data and recv_data:
            sent_values = [value for _, value, _ in sent_data]
            recv_values = [value for _, value, _ in recv_data]

            total_sent = sum(sent_values)
            total_recv = sum(recv_values)

            analysis["network_usage"] = {
                "total_sent_bytes": total_sent,
                "total_received_bytes": total_recv,
                "total_traffic_bytes": total_sent + total_recv,
            }

            # 異常な通信量の検出
            if total_sent + total_recv > 1e9:  # 1GB以上
                issues.append("high_network_usage")

        analysis["issues"] = issues
        return analysis

    async def _analyze_application_performance(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """アプリケーション性能分析"""
        request_data = self.metrics_system.query_metrics(
            "http_requests_total", start_time, end_time
        )
        response_time_data = self.metrics_system.query_metrics(
            "http_request_duration_seconds", start_time, end_time
        )
        error_data = self.metrics_system.query_metrics(
            "http_errors_total", start_time, end_time
        )

        analysis = {}
        issues = []

        # リクエスト率分析
        if request_data:
            request_values = [value for _, value, _ in request_data]
            total_requests = sum(request_values)
            avg_requests_per_minute = total_requests / (len(request_values) or 1)

            analysis["request_metrics"] = {
                "total_requests": total_requests,
                "average_requests_per_minute": avg_requests_per_minute,
            }

        # レスポンス時間分析
        if response_time_data:
            response_values = [value for _, value, _ in response_time_data]

            if NUMPY_AVAILABLE:
                avg_response_time = np.mean(response_values)
                p95_response_time = np.percentile(response_values, 95)
                p99_response_time = np.percentile(response_values, 99)
            else:
                avg_response_time = sum(response_values) / len(response_values)
                sorted_responses = sorted(response_values)
                p95_response_time = sorted_responses[int(0.95 * len(sorted_responses))]
                p99_response_time = sorted_responses[int(0.99 * len(sorted_responses))]

            analysis["response_time_metrics"] = {
                "average": avg_response_time,
                "p95": p95_response_time,
                "p99": p99_response_time,
            }

            if avg_response_time > 2.0:
                issues.append("slow_response_time")
            if p95_response_time > 5.0:
                issues.append("response_time_spikes")

        # エラー率分析
        if error_data and request_data:
            error_values = [value for _, value, _ in error_data]
            total_errors = sum(error_values)
            error_rate = total_errors / (sum(request_values) or 1)

            analysis["error_metrics"] = {
                "total_errors": total_errors,
                "error_rate": error_rate,
            }

            if error_rate > 0.01:  # 1%以上のエラー率
                issues.append("high_error_rate")

        analysis["issues"] = issues
        return analysis

    def _calculate_trend(self, values: List[float]) -> float:
        """トレンド計算（NumPy版）"""
        if not NUMPY_AVAILABLE or len(values) < 2:
            return self._calculate_simple_trend(values)

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)

    def _calculate_simple_trend(self, values: List[float]) -> float:
        """トレンド計算（シンプル版）"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0


class OptimizationEngine:
    """最適化エンジン"""

    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.recommendation_rules = self._initialize_recommendation_rules()

    def _initialize_recommendation_rules(self) -> Dict[str, Any]:
        """推奨ルール初期化"""
        return {
            "cpu_optimization": {
                "high_average_cpu": {
                    "title": "CPU使用率最適化",
                    "description": "平均CPU使用率が高すぎます。プロセス最適化またはスケールアウトを検討してください。",
                    "type": OptimizationType.RESOURCE_OPTIMIZATION,
                    "priority": Priority.HIGH,
                    "impact": ImpactLevel.SIGNIFICANT,
                    "steps": [
                        "CPU集約的プロセスの特定",
                        "不要なバックグラウンドタスクの停止",
                        "アプリケーションの並列処理最適化",
                        "水平スケーリングの検討",
                    ],
                    "effort": 6,
                },
                "cpu_spikes": {
                    "title": "CPUスパイク対策",
                    "description": "CPU使用率のスパイクが発生しています。負荷分散とリソース制限を実装してください。",
                    "type": OptimizationType.PERFORMANCE_TUNING,
                    "priority": Priority.MEDIUM,
                    "impact": ImpactLevel.MODERATE,
                    "steps": [
                        "スパイク発生タイミングの分析",
                        "リソース制限の設定",
                        "負荷分散の実装",
                        "キューイングシステムの導入",
                    ],
                    "effort": 4,
                },
            },
            "memory_optimization": {
                "high_memory_usage": {
                    "title": "メモリ使用量最適化",
                    "description": "メモリ使用量が高すぎます。メモリリークの確認とメモリ効率の改善が必要です。",
                    "type": OptimizationType.RESOURCE_OPTIMIZATION,
                    "priority": Priority.HIGH,
                    "impact": ImpactLevel.SIGNIFICANT,
                    "steps": [
                        "メモリプロファイリングの実行",
                        "メモリリークの特定と修正",
                        "オブジェクトプールの導入",
                        "ガベージコレクション最適化",
                    ],
                    "effort": 7,
                },
                "memory_leak_suspected": {
                    "title": "メモリリーク対策",
                    "description": "メモリ使用量が継続的に増加しています。メモリリークの可能性があります。",
                    "type": OptimizationType.CODE_OPTIMIZATION,
                    "priority": Priority.CRITICAL,
                    "impact": ImpactLevel.TRANSFORMATIVE,
                    "steps": [
                        "メモリプロファイラーによる詳細分析",
                        "循環参照の検出と修正",
                        "適切なリソース解放の実装",
                        "メモリ監視の強化",
                    ],
                    "effort": 8,
                },
            },
            "application_optimization": {
                "slow_response_time": {
                    "title": "レスポンス時間改善",
                    "description": "平均レスポンス時間が遅すぎます。アプリケーションの最適化が必要です。",
                    "type": OptimizationType.PERFORMANCE_TUNING,
                    "priority": Priority.HIGH,
                    "impact": ImpactLevel.SIGNIFICANT,
                    "steps": [
                        "ボトルネック箇所の特定",
                        "データベースクエリの最適化",
                        "キャッシングの実装",
                        "非同期処理の導入",
                    ],
                    "effort": 6,
                },
                "high_error_rate": {
                    "title": "エラー率改善",
                    "description": "エラー率が高すぎます。エラーハンドリングと信頼性の向上が必要です。",
                    "type": OptimizationType.CODE_OPTIMIZATION,
                    "priority": Priority.CRITICAL,
                    "impact": ImpactLevel.TRANSFORMATIVE,
                    "steps": [
                        "エラーパターンの分析",
                        "エラーハンドリングの改善",
                        "リトライ機構の実装",
                        "監視とアラートの強化",
                    ],
                    "effort": 5,
                },
            },
            "disk_optimization": {
                "high_disk_usage": {
                    "title": "ディスク容量最適化",
                    "description": "ディスク使用量が高すぎます。ストレージ管理の改善が必要です。",
                    "type": OptimizationType.RESOURCE_OPTIMIZATION,
                    "priority": Priority.MEDIUM,
                    "impact": ImpactLevel.MODERATE,
                    "steps": [
                        "不要ファイルのクリーンアップ",
                        "ログローテーション設定",
                        "データアーカイブ戦略の実装",
                        "ストレージ容量の拡張検討",
                    ],
                    "effort": 3,
                }
            },
        }

    async def generate_recommendations(self) -> List[OptimizationRecommendation]:
        """最適化推奨事項を生成"""
        performance_analysis = (
            await self.performance_analyzer.analyze_system_performance()
        )
        recommendations = []

        for category, analysis in performance_analysis.items():
            if "issues" in analysis:
                for issue in analysis["issues"]:
                    recommendation = await self._generate_recommendation_for_issue(
                        category, issue, analysis
                    )
                    if recommendation:
                        recommendations.append(recommendation)

        # 優先度順でソート
        recommendations.sort(
            key=lambda x: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.priority.value],
                -x.impact_level.value.count(
                    "t"
                ),  # transformative > significant > moderate > minimal
            )
        )

        return recommendations

    async def _generate_recommendation_for_issue(
        self, category: str, issue: str, analysis_data: Dict[str, Any]
    ) -> Optional[OptimizationRecommendation]:
        """個別問題に対する推奨事項生成"""
        category_key = category.replace("_analysis", "_optimization")

        if category_key not in self.recommendation_rules:
            return None

        rule = self.recommendation_rules[category_key].get(issue)
        if not rule:
            return None

        # 推定改善効果の計算
        estimated_improvement = self._calculate_estimated_improvement(
            category, issue, analysis_data
        )

        recommendation = OptimizationRecommendation(
            id=f"{category}_{issue}_{int(datetime.utcnow().timestamp())}",
            title=rule["title"],
            description=rule["description"],
            optimization_type=rule["type"],
            priority=rule["priority"],
            impact_level=rule["impact"],
            estimated_improvement=estimated_improvement,
            implementation_effort=rule["effort"],
            implementation_steps=rule["steps"],
            evidence=analysis_data,
            tags=[category, issue],
        )

        # 成功指標の設定
        recommendation.success_metrics = self._generate_success_metrics(category, issue)

        # リスクの設定
        recommendation.risks = self._generate_risks(rule["type"], rule["effort"])

        return recommendation

    def _calculate_estimated_improvement(
        self, category: str, issue: str, analysis_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """推定改善効果の計算"""
        improvements = {}

        if category == "cpu_analysis":
            if issue == "high_average_cpu":
                current_avg = analysis_data.get("average_usage", 0)
                improvements["cpu_usage_reduction"] = min(30.0, current_avg - 50.0)
            elif issue == "cpu_spikes":
                improvements["cpu_spike_reduction"] = 40.0

        elif category == "memory_analysis":
            if issue == "high_memory_usage":
                current_avg = analysis_data.get("average_usage", 0)
                improvements["memory_usage_reduction"] = min(25.0, current_avg - 60.0)
            elif issue == "memory_leak_suspected":
                improvements["memory_stability_improvement"] = 80.0

        elif category == "application_analysis":
            if issue == "slow_response_time":
                improvements["response_time_improvement"] = 50.0
            elif issue == "high_error_rate":
                improvements["error_rate_reduction"] = 70.0

        elif category == "disk_analysis":
            if issue == "high_disk_usage":
                improvements["disk_space_freed"] = 20.0

        return improvements

    def _generate_success_metrics(self, category: str, issue: str) -> List[str]:
        """成功指標の生成"""
        metrics = []

        if category == "cpu_analysis":
            metrics = ["CPU使用率が70%以下", "CPUスパイクが90%以下"]
        elif category == "memory_analysis":
            metrics = ["メモリ使用率が80%以下", "メモリ使用量が安定"]
        elif category == "application_analysis":
            metrics = ["平均レスポンス時間が2秒以下", "エラー率が1%以下"]
        elif category == "disk_analysis":
            metrics = ["ディスク使用率が85%以下"]

        return metrics

    def _generate_risks(
        self, optimization_type: OptimizationType, effort: int
    ) -> List[str]:
        """リスクの生成"""
        risks = []

        if effort > 7:
            risks.append("実装に長期間を要する可能性")

        if optimization_type in [
            OptimizationType.ARCHITECTURE_IMPROVEMENT,
            OptimizationType.CODE_OPTIMIZATION,
        ]:
            risks.append("既存機能への影響のリスク")
            risks.append("テスト工数の増加")

        if optimization_type == OptimizationType.SCALING_RECOMMENDATION:
            risks.append("インフラストラクチャコストの増加")

        return risks


class OptimizationManager:
    """最適化管理システム"""

    def __init__(self, db_path: str = "optimization.db"):
        self.db_path = db_path
        self.optimization_engine = OptimizationEngine()
        self._initialize_database()

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_recommendations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    optimization_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    impact_level TEXT NOT NULL,
                    estimated_improvement TEXT,
                    implementation_effort INTEGER,
                    implementation_steps TEXT,
                    prerequisites TEXT,
                    risks TEXT,
                    success_metrics TEXT,
                    created_at DATETIME NOT NULL,
                    evidence TEXT,
                    tags TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (recommendation_id) REFERENCES optimization_recommendations (id)
                )
            """
            )

            conn.commit()

    async def analyze_and_recommend(self) -> List[OptimizationRecommendation]:
        """分析と推奨事項生成"""
        recommendations = await self.optimization_engine.generate_recommendations()

        # データベースに保存
        for recommendation in recommendations:
            self._save_recommendation(recommendation)

        return recommendations

    def _save_recommendation(self, recommendation: OptimizationRecommendation):
        """推奨事項をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO optimization_recommendations
                (id, title, description, optimization_type, priority, impact_level,
                 estimated_improvement, implementation_effort, implementation_steps,
                 prerequisites, risks, success_metrics, created_at, evidence, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    recommendation.id,
                    recommendation.title,
                    recommendation.description,
                    recommendation.optimization_type.value,
                    recommendation.priority.value,
                    recommendation.impact_level.value,
                    json.dumps(recommendation.estimated_improvement),
                    recommendation.implementation_effort,
                    json.dumps(recommendation.implementation_steps),
                    json.dumps(recommendation.prerequisites),
                    json.dumps(recommendation.risks),
                    json.dumps(recommendation.success_metrics),
                    recommendation.created_at.isoformat(),
                    json.dumps(recommendation.evidence),
                    json.dumps(recommendation.tags),
                ),
            )
            conn.commit()

    def get_recommendations(
        self,
        priority: Optional[Priority] = None,
        optimization_type: Optional[OptimizationType] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """推奨事項を取得"""
        query = "SELECT * FROM optimization_recommendations WHERE 1=1"
        params = []

        if priority:
            query += " AND priority = ?"
            params.append(priority.value)

        if optimization_type:
            query += " AND optimization_type = ?"
            params.append(optimization_type.value)

        query += " ORDER BY priority, impact_level DESC LIMIT ?"
        params.append(limit)

        recommendations = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                recommendations.append(
                    {
                        "id": row[0],
                        "title": row[1],
                        "description": row[2],
                        "optimization_type": row[3],
                        "priority": row[4],
                        "impact_level": row[5],
                        "estimated_improvement": json.loads(row[6]) if row[6] else {},
                        "implementation_effort": row[7],
                        "implementation_steps": json.loads(row[8]) if row[8] else [],
                        "prerequisites": json.loads(row[9]) if row[9] else [],
                        "risks": json.loads(row[10]) if row[10] else [],
                        "success_metrics": json.loads(row[11]) if row[11] else [],
                        "created_at": row[12],
                        "evidence": json.loads(row[13]) if row[13] else {},
                        "tags": json.loads(row[14]) if row[14] else [],
                        "status": row[15] if len(row) > 15 else "pending",
                    }
                )

        return recommendations

    def mark_recommendation_implemented(self, recommendation_id: str, notes: str = ""):
        """推奨事項を実装済みとしてマーク"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE optimization_recommendations
                SET status = 'implemented'
                WHERE id = ?
            """,
                (recommendation_id,),
            )

            conn.execute(
                """
                INSERT INTO optimization_history
                (recommendation_id, action, timestamp, notes)
                VALUES (?, 'implemented', ?, ?)
            """,
                (recommendation_id, datetime.utcnow().isoformat(), notes),
            )

            conn.commit()

    def get_optimization_report(self) -> Dict[str, Any]:
        """最適化レポートを生成"""
        with sqlite3.connect(self.db_path) as conn:
            # 推奨事項統計
            cursor = conn.execute(
                """
                SELECT
                    optimization_type,
                    priority,
                    COUNT(*) as count
                FROM optimization_recommendations
                WHERE status = 'pending'
                GROUP BY optimization_type, priority
            """
            )

            statistics = {}
            for row in cursor.fetchall():
                opt_type = row[0]
                priority = row[1]
                count = row[2]

                if opt_type not in statistics:
                    statistics[opt_type] = {}
                statistics[opt_type][priority] = count

            # 実装済み統計
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM optimization_recommendations
                WHERE status = 'implemented'
            """
            )
            implemented_count = cursor.fetchone()[0]

            # 総推奨事項数
            cursor = conn.execute("SELECT COUNT(*) FROM optimization_recommendations")
            total_count = cursor.fetchone()[0]

        return {
            "total_recommendations": total_count,
            "pending_recommendations": total_count - implemented_count,
            "implemented_recommendations": implemented_count,
            "implementation_rate": (implemented_count / total_count * 100)
            if total_count > 0
            else 0,
            "statistics_by_type": statistics,
            "generated_at": datetime.utcnow().isoformat(),
        }


# グローバルインスタンス
_optimization_manager = None


def get_optimization_manager() -> OptimizationManager:
    """グローバル最適化管理を取得"""
    global _optimization_manager
    if _optimization_manager is None:
        _optimization_manager = OptimizationManager()
    return _optimization_manager
