"""
キャッシュ統計・パフォーマンス監視

キャッシュのパフォーマンス追跡と分析機能
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
from collections import defaultdict

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class CacheStats:
    """キャッシュ統計クラス"""

    def __init__(self):
        """初期化"""
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """統計をリセット"""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.fallbacks = 0
            self.errors = 0
            self.start_time = time.time()
            self.last_reset_time = datetime.now()

    def record_hit(self) -> None:
        """キャッシュヒットを記録"""
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        """キャッシュミスを記録"""
        with self._lock:
            self.misses += 1

    def record_set(self) -> None:
        """キャッシュ設定を記録"""
        with self._lock:
            self.sets += 1

    def record_fallback(self) -> None:
        """フォールバック使用を記録"""
        with self._lock:
            self.fallbacks += 1

    def record_error(self) -> None:
        """エラーを記録"""
        with self._lock:
            self.errors += 1

    def to_dict(self) -> Dict[str, Any]:
        """統計を辞書形式で取得"""
        with self._lock:
            total_requests = self.hits + self.misses + self.fallbacks
            uptime = time.time() - self.start_time

            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "fallbacks": self.fallbacks,
                "errors": self.errors,
                "total_requests": total_requests,
                "hit_rate": self.hits / total_requests if total_requests > 0 else 0.0,
                "miss_rate": self.misses / total_requests if total_requests > 0 else 0.0,
                "fallback_rate": self.fallbacks / total_requests if total_requests > 0 else 0.0,
                "error_rate": self.errors / (total_requests + self.errors) if total_requests + self.errors > 0 else 0.0,
                "requests_per_second": total_requests / uptime if uptime > 0 else 0.0,
                "uptime_seconds": uptime,
                "last_reset": self.last_reset_time.isoformat(),
            }


class CachePerformanceMonitor:
    """キャッシュパフォーマンス監視クラス"""

    def __init__(self, window_size: int = 300, sample_interval: int = 10):
        """
        初期化

        Args:
            window_size: 監視ウィンドウサイズ（秒）
            sample_interval: サンプリング間隔（秒）
        """
        self.window_size = window_size
        self.sample_interval = sample_interval
        self._lock = threading.Lock()
        
        # 時系列データ
        self.performance_history: List[Dict[str, Any]] = []
        self.function_stats: Dict[str, CacheStats] = defaultdict(CacheStats)
        
        # アラート設定
        self.alert_thresholds = {
            "hit_rate_min": 0.5,
            "error_rate_max": 0.05,
            "response_time_max": 5.0,
        }
        
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info(f"パフォーマンス監視初期化: ウィンドウ={window_size}s, 間隔={sample_interval}s")

    def record_function_call(
        self, 
        function_name: str, 
        cache_hit: bool, 
        response_time: float,
        error: Optional[Exception] = None
    ) -> None:
        """
        関数呼び出しを記録

        Args:
            function_name: 関数名
            cache_hit: キャッシュヒットかどうか
            response_time: 応答時間（秒）
            error: エラー（ある場合）
        """
        with self._lock:
            stats = self.function_stats[function_name]
            
            if cache_hit:
                stats.record_hit()
            else:
                stats.record_miss()
            
            if error:
                stats.record_error()

            # パフォーマンス履歴に追加
            performance_point = {
                "timestamp": datetime.now(),
                "function": function_name,
                "cache_hit": cache_hit,
                "response_time": response_time,
                "error": str(error) if error else None,
            }
            
            self.performance_history.append(performance_point)
            
            # ウィンドウサイズ制限
            cutoff_time = datetime.now() - timedelta(seconds=self.window_size)
            self.performance_history = [
                p for p in self.performance_history 
                if p["timestamp"] > cutoff_time
            ]

    def get_function_performance(self, function_name: str) -> Dict[str, Any]:
        """
        特定関数のパフォーマンス取得

        Args:
            function_name: 関数名

        Returns:
            パフォーマンス統計
        """
        with self._lock:
            stats = self.function_stats[function_name].to_dict()
            
            # 最近のパフォーマンスデータ
            recent_calls = [
                p for p in self.performance_history
                if p["function"] == function_name
            ]
            
            if recent_calls:
                response_times = [p["response_time"] for p in recent_calls]
                stats.update({
                    "recent_calls": len(recent_calls),
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                })
            else:
                stats.update({
                    "recent_calls": 0,
                    "avg_response_time": 0.0,
                    "min_response_time": 0.0,
                    "max_response_time": 0.0,
                })

            return stats

    def get_overall_performance(self) -> Dict[str, Any]:
        """全体パフォーマンス統計を取得"""
        with self._lock:
            # 全関数の統計を集計
            total_stats = CacheStats()
            
            for stats in self.function_stats.values():
                total_stats.hits += stats.hits
                total_stats.misses += stats.misses
                total_stats.sets += stats.sets
                total_stats.fallbacks += stats.fallbacks
                total_stats.errors += stats.errors

            overall_stats = total_stats.to_dict()
            
            # 追加の全体統計
            if self.performance_history:
                response_times = [p["response_time"] for p in self.performance_history]
                cache_hits = [p for p in self.performance_history if p["cache_hit"]]
                
                overall_stats.update({
                    "total_functions": len(self.function_stats),
                    "recent_calls": len(self.performance_history),
                    "avg_response_time": sum(response_times) / len(response_times),
                    "recent_hit_rate": len(cache_hits) / len(self.performance_history),
                    "active_alerts": len(self.alerts),
                })

            return overall_stats

    def get_top_functions(self, limit: int = 10, sort_by: str = "hits") -> List[Dict[str, Any]]:
        """
        トップ関数リストを取得

        Args:
            limit: 取得件数
            sort_by: ソート基準（hits, misses, errors, calls）

        Returns:
            トップ関数リスト
        """
        with self._lock:
            function_performances = []
            
            for func_name, stats in self.function_stats.items():
                perf = self.get_function_performance(func_name)
                perf["function_name"] = func_name
                function_performances.append(perf)

            # ソート
            if sort_by == "hits":
                function_performances.sort(key=lambda x: x["hits"], reverse=True)
            elif sort_by == "misses":
                function_performances.sort(key=lambda x: x["misses"], reverse=True)
            elif sort_by == "errors":
                function_performances.sort(key=lambda x: x["errors"], reverse=True)
            elif sort_by == "calls":
                function_performances.sort(key=lambda x: x["total_requests"], reverse=True)
            else:
                function_performances.sort(key=lambda x: x.get("avg_response_time", 0), reverse=True)

            return function_performances[:limit]

    def check_alerts(self) -> List[Dict[str, Any]]:
        """アラート条件をチェック"""
        new_alerts = []
        current_time = datetime.now()

        with self._lock:
            overall_perf = self.get_overall_performance()
            
            # ヒット率アラート
            if overall_perf.get("hit_rate", 0) < self.alert_thresholds["hit_rate_min"]:
                new_alerts.append({
                    "type": "low_hit_rate",
                    "message": f"キャッシュヒット率が低下: {overall_perf['hit_rate']:.2%}",
                    "threshold": self.alert_thresholds["hit_rate_min"],
                    "current_value": overall_perf["hit_rate"],
                    "timestamp": current_time,
                })

            # エラー率アラート
            if overall_perf.get("error_rate", 0) > self.alert_thresholds["error_rate_max"]:
                new_alerts.append({
                    "type": "high_error_rate",
                    "message": f"エラー率が上昇: {overall_perf['error_rate']:.2%}",
                    "threshold": self.alert_thresholds["error_rate_max"],
                    "current_value": overall_perf["error_rate"],
                    "timestamp": current_time,
                })

            # 応答時間アラート
            if overall_perf.get("avg_response_time", 0) > self.alert_thresholds["response_time_max"]:
                new_alerts.append({
                    "type": "slow_response",
                    "message": f"平均応答時間が遅延: {overall_perf['avg_response_time']:.2f}s",
                    "threshold": self.alert_thresholds["response_time_max"],
                    "current_value": overall_perf["avg_response_time"],
                    "timestamp": current_time,
                })

            # 新しいアラートを追加
            self.alerts.extend(new_alerts)
            
            # 古いアラートを削除（24時間以上前）
            cutoff_time = current_time - timedelta(hours=24)
            self.alerts = [
                alert for alert in self.alerts 
                if alert["timestamp"] > cutoff_time
            ]

        if new_alerts:
            logger.warning(f"新しいパフォーマンスアラート: {len(new_alerts)}件")

        return new_alerts

    def get_performance_trend(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """
        パフォーマンストレンドを取得

        Args:
            duration_minutes: 分析期間（分）

        Returns:
            トレンド分析結果
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            recent_history = [
                p for p in self.performance_history
                if p["timestamp"] > cutoff_time
            ]

            if not recent_history:
                return {"trend": "insufficient_data", "data_points": 0}

            # 時間軸での分析
            time_buckets = defaultdict(list)
            bucket_size = duration_minutes // 10  # 10個のバケットに分割
            
            for point in recent_history:
                minutes_ago = (datetime.now() - point["timestamp"]).total_seconds() / 60
                bucket_index = int(minutes_ago // bucket_size)
                if bucket_index < 10:
                    time_buckets[bucket_index].append(point)

            # トレンド計算
            hit_rates = []
            response_times = []
            
            for i in range(10):
                bucket_data = time_buckets.get(i, [])
                if bucket_data:
                    hits = sum(1 for p in bucket_data if p["cache_hit"])
                    hit_rate = hits / len(bucket_data)
                    avg_response = sum(p["response_time"] for p in bucket_data) / len(bucket_data)
                    
                    hit_rates.append(hit_rate)
                    response_times.append(avg_response)

            trend_analysis = {
                "data_points": len(recent_history),
                "time_buckets": len([b for b in time_buckets.values() if b]),
                "hit_rate_trend": self._calculate_trend(hit_rates),
                "response_time_trend": self._calculate_trend(response_times),
                "hit_rates": hit_rates,
                "response_times": response_times,
            }

            return trend_analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """値の配列からトレンドを計算"""
        if len(values) < 3:
            return "insufficient_data"
        
        # 単純な線形回帰的なトレンド判定
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        change_rate = (second_half - first_half) / first_half if first_half != 0 else 0
        
        if change_rate > 0.1:
            return "increasing"
        elif change_rate < -0.1:
            return "decreasing"
        else:
            return "stable"

    def generate_performance_report(self) -> str:
        """パフォーマンスレポートを生成"""
        overall_perf = self.get_overall_performance()
        top_functions = self.get_top_functions(5)
        trend_analysis = self.get_performance_trend()
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("キャッシュパフォーマンスレポート")
        report_lines.append("=" * 60)
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        report_lines.append("")
        
        # 全体統計
        report_lines.append("【全体統計】")
        report_lines.append(f"総リクエスト数: {overall_perf.get('total_requests', 0):,}")
        report_lines.append(f"キャッシュヒット率: {overall_perf.get('hit_rate', 0):.2%}")
        report_lines.append(f"エラー率: {overall_perf.get('error_rate', 0):.2%}")
        report_lines.append(f"平均応答時間: {overall_perf.get('avg_response_time', 0):.3f}s")
        report_lines.append(f"アクティブアラート: {overall_perf.get('active_alerts', 0)}件")
        report_lines.append("")
        
        # トップ関数
        if top_functions:
            report_lines.append("【最もよく使用される関数】")
            for i, func in enumerate(top_functions, 1):
                report_lines.append(
                    f"{i}. {func['function_name']} - "
                    f"ヒット: {func['hits']}, ミス: {func['misses']}, "
                    f"ヒット率: {func['hit_rate']:.2%}"
                )
            report_lines.append("")
        
        # トレンド分析
        if trend_analysis["data_points"] > 0:
            report_lines.append("【トレンド分析】")
            report_lines.append(f"分析データ数: {trend_analysis['data_points']}")
            report_lines.append(f"ヒット率トレンド: {trend_analysis['hit_rate_trend']}")
            report_lines.append(f"応答時間トレンド: {trend_analysis['response_time_trend']}")
            report_lines.append("")
        
        # アクティブアラート
        recent_alerts = [a for a in self.alerts if (datetime.now() - a["timestamp"]).total_seconds() < 3600]
        if recent_alerts:
            report_lines.append("【直近のアラート】")
            for alert in recent_alerts[-5:]:  # 最新5件
                report_lines.append(
                    f"- {alert['message']} "
                    f"({alert['timestamp'].strftime('%H:%M:%S')})"
                )
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

    def reset_stats(self, function_name: Optional[str] = None) -> None:
        """統計をリセット"""
        with self._lock:
            if function_name:
                if function_name in self.function_stats:
                    self.function_stats[function_name].reset()
                    logger.info(f"関数統計リセット: {function_name}")
            else:
                for stats in self.function_stats.values():
                    stats.reset()
                self.performance_history.clear()
                self.alerts.clear()
                logger.info("全統計リセット完了")

    def set_alert_threshold(self, metric: str, value: float) -> None:
        """アラート閾値を設定"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = value
            logger.info(f"アラート閾値更新: {metric} = {value}")
        else:
            logger.warning(f"未知のメトリック: {metric}")

    def export_performance_data(self) -> Dict[str, Any]:
        """パフォーマンスデータをエクスポート"""
        with self._lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "overall_performance": self.get_overall_performance(),
                "function_stats": {
                    name: self.get_function_performance(name)
                    for name in self.function_stats.keys()
                },
                "recent_alerts": self.alerts,
                "alert_thresholds": self.alert_thresholds,
                "performance_history": [
                    {**p, "timestamp": p["timestamp"].isoformat()}
                    for p in self.performance_history
                ],
            }
            
            logger.info(f"パフォーマンスデータエクスポート完了: {len(export_data)}項目")
            return export_data