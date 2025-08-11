#!/usr/bin/env python3
"""
リアルタイムリスク監視システム
Real-time Risk Monitoring System

24時間連続リスク監視・自動対応・アラート管理
"""

import asyncio
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..data.stock_fetcher_v2 import StockFetcherV2
from ..realtime.alert_system import AlertLevel, AlertManager

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from .risk_coordinator import RiskAnalysisCoordinator, RiskAssessmentSummary

logger = get_context_logger(__name__)


@dataclass
class RiskMonitoringConfig:
    """リスク監視設定"""

    monitoring_interval_seconds: int = 5
    batch_analysis_interval_minutes: int = 15
    alert_cooldown_minutes: int = 10
    max_concurrent_analyses: int = 20
    risk_threshold_critical: float = 0.85
    risk_threshold_high: float = 0.7
    risk_threshold_medium: float = 0.5
    enable_auto_response: bool = True
    max_daily_alerts: int = 100


@dataclass
class MonitoringMetrics:
    """監視メトリクス"""

    timestamp: datetime
    active_monitors: int
    total_analyses_today: int
    alerts_sent_today: int
    average_risk_score: float
    critical_alerts_count: int
    system_health_status: str
    processing_queue_size: int
    memory_usage_mb: float


class RealTimeRiskMonitor:
    """リアルタイムリスク監視システム"""

    def __init__(self, config: Optional[RiskMonitoringConfig] = None):
        self.config = config or RiskMonitoringConfig()

        # 核となるコンポーネント
        self.risk_coordinator = RiskAnalysisCoordinator()
        self.alert_manager = AlertManager()
        self.stock_fetcher = StockFetcherV2()

        # 監視状態
        self.is_running = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.analysis_queue: asyncio.Queue = asyncio.Queue()

        # データ管理
        self.active_symbols: List[str] = []
        self.monitoring_metrics: List[MonitoringMetrics] = []
        self.daily_statistics = {}

        # アラート管理
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_cooldown: Dict[str, datetime] = {}

        # 自動応答ハンドラー
        self.response_handlers: Dict[str, Callable] = {}

        # パフォーマンス統計
        self.performance_stats = {
            "start_time": None,
            "total_monitoring_cycles": 0,
            "total_risk_analyses": 0,
            "total_alerts_sent": 0,
            "avg_cycle_time": 0.0,
            "error_count": 0,
            "uptime_seconds": 0,
        }

        logger.info("リアルタイムリスク監視システム初期化完了")

    async def start_monitoring(
        self, symbols: List[str], custom_handlers: Optional[Dict[str, Callable]] = None
    ):
        """監視開始"""

        if self.is_running:
            logger.warning("監視システムは既に稼働中です")
            return

        self.active_symbols = symbols
        self.is_running = True
        self.performance_stats["start_time"] = datetime.now()

        # カスタムハンドラー登録
        if custom_handlers:
            self.response_handlers.update(custom_handlers)

        logger.info(f"リアルタイム監視開始: {len(symbols)}銘柄")

        # 監視タスク起動
        self.monitoring_tasks = [
            asyncio.create_task(self._continuous_monitoring_loop()),
            asyncio.create_task(self._analysis_worker()),
            asyncio.create_task(self._metrics_collector_loop()),
            asyncio.create_task(self._batch_analysis_loop()),
            asyncio.create_task(self._alert_manager_loop()),
        ]

        # タスク実行
        try:
            await asyncio.gather(*self.monitoring_tasks)
        except asyncio.CancelledError:
            logger.info("監視タスクがキャンセルされました")
        except Exception as e:
            logger.error(f"監視システムエラー: {e}")
            await self.stop_monitoring()

    async def stop_monitoring(self):
        """監視停止"""

        if not self.is_running:
            return

        logger.info("リアルタイム監視停止中...")
        self.is_running = False

        # すべてのタスクをキャンセル
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()

        # タスク完了待機
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.monitoring_tasks.clear()

        # 統計更新
        if self.performance_stats["start_time"]:
            uptime = datetime.now() - self.performance_stats["start_time"]
            self.performance_stats["uptime_seconds"] = uptime.total_seconds()

        logger.info("監視システム停止完了")

    async def _continuous_monitoring_loop(self):
        """連続監視ループ"""

        logger.info("連続監視ループ開始")

        while self.is_running:
            cycle_start = time.time()

            try:
                # 市場データ取得
                market_data = await self._fetch_market_data()

                # リスク分析が必要な取引/状況を検出
                risk_events = await self._detect_risk_events(market_data)

                # 分析キューに追加
                for event in risk_events:
                    await self.analysis_queue.put(event)

                # 監視サイクル統計更新
                cycle_time = time.time() - cycle_start
                self.performance_stats["total_monitoring_cycles"] += 1

                # 平均サイクル時間更新
                cycles = self.performance_stats["total_monitoring_cycles"]
                old_avg = self.performance_stats["avg_cycle_time"]
                self.performance_stats["avg_cycle_time"] = (
                    old_avg * (cycles - 1) + cycle_time
                ) / cycles

                # 次のサイクルまで待機
                await asyncio.sleep(self.config.monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                self.performance_stats["error_count"] += 1
                await asyncio.sleep(1)  # エラー時は短時間待機

    async def _analysis_worker(self):
        """分析ワーカー"""

        logger.info("リスク分析ワーカー開始")

        while self.is_running:
            try:
                # キューからイベント取得（タイムアウト付き）
                event = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)

                # リスク分析実行
                assessment = await self._analyze_risk_event(event)

                if assessment:
                    # アラート評価・送信
                    await self._process_risk_assessment(assessment)
                    self.performance_stats["total_risk_analyses"] += 1

                # キュータスク完了マーク
                self.analysis_queue.task_done()

            except asyncio.TimeoutError:
                # タイムアウトは正常（キューが空の状態）
                continue
            except Exception as e:
                logger.error(f"分析ワーカーエラー: {e}")
                self.performance_stats["error_count"] += 1

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """市場データ取得"""

        try:
            market_data = {}

            # 監視対象銘柄のデータ取得
            for symbol in self.active_symbols:
                try:
                    data = await self.stock_fetcher.fetch_realtime_data(symbol)
                    if data:
                        market_data[symbol] = data
                except Exception as e:
                    logger.warning(f"銘柄 {symbol} データ取得エラー: {e}")

            return market_data

        except Exception as e:
            logger.error(f"市場データ取得エラー: {e}")
            return {}

    async def _detect_risk_events(
        self, market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """リスクイベント検出"""

        risk_events = []

        try:
            for symbol, data in market_data.items():
                # 価格変動チェック
                price_change = data.get("price_change_percent", 0)
                if abs(price_change) > 5:  # 5%以上の変動
                    risk_events.append(
                        {
                            "type": "price_volatility",
                            "symbol": symbol,
                            "severity": "high" if abs(price_change) > 10 else "medium",
                            "data": data,
                            "timestamp": datetime.now(),
                        }
                    )

                # 取引量チェック
                volume_ratio = data.get("volume_ratio", 1.0)
                if volume_ratio > 3.0:  # 通常の3倍以上
                    risk_events.append(
                        {
                            "type": "volume_spike",
                            "symbol": symbol,
                            "severity": "medium",
                            "data": data,
                            "timestamp": datetime.now(),
                        }
                    )

                # テクニカル指標チェック
                rsi = data.get("rsi", 50)
                if rsi > 80 or rsi < 20:  # 過熱状態
                    risk_events.append(
                        {
                            "type": "technical_signal",
                            "symbol": symbol,
                            "severity": "low",
                            "data": data,
                            "timestamp": datetime.now(),
                        }
                    )

        except Exception as e:
            logger.error(f"リスクイベント検出エラー: {e}")

        return risk_events

    async def _analyze_risk_event(
        self, event: Dict[str, Any]
    ) -> Optional[RiskAssessmentSummary]:
        """リスクイベント分析"""

        try:
            # イベントを取引データ形式に変換
            transaction_data = {
                "symbol": event["symbol"],
                "type": "market_event",
                "amount": 1000000,  # デフォルト金額
                "timestamp": event["timestamp"].isoformat(),
                "event_type": event["type"],
                "severity": event["severity"],
                "market_conditions": event["data"],
            }

            # 包括的リスク分析
            assessment = await self.risk_coordinator.comprehensive_risk_assessment(
                transaction_data,
                market_context=event["data"],
                enable_ai_analysis=True,
                enable_fraud_detection=False,  # 市場イベントでは不正検知は無効
            )

            return assessment

        except Exception as e:
            logger.error(f"リスクイベント分析エラー: {e}")
            return None

    async def _process_risk_assessment(self, assessment: RiskAssessmentSummary):
        """リスク評価処理"""

        try:
            risk_score = assessment.overall_risk_score

            # アラート判定
            if risk_score >= self.config.risk_threshold_critical:
                await self._handle_critical_risk(assessment)
            elif risk_score >= self.config.risk_threshold_high:
                await self._handle_high_risk(assessment)
            elif risk_score >= self.config.risk_threshold_medium:
                await self._handle_medium_risk(assessment)

            # 自動応答実行
            if self.config.enable_auto_response:
                await self._execute_auto_response(assessment)

        except Exception as e:
            logger.error(f"リスク評価処理エラー: {e}")

    async def _handle_critical_risk(self, assessment: RiskAssessmentSummary):
        """重要リスク処理"""

        # クールダウンチェック
        if not self._check_alert_cooldown(assessment.request_id, "critical"):
            return

        await self.alert_manager.create_alert(
            title=f"🚨 重要リスクアラート: {assessment.risk_category.upper()}",
            message=f"銘柄: {assessment.component_results.get('symbol', 'N/A')}\n"
            f"リスクスコア: {assessment.overall_risk_score:.3f}\n"
            f"推定損失: ¥{assessment.estimated_loss_potential:,.0f}\n"
            f"緊急対応が必要です",
            level=AlertLevel.CRITICAL,
            source="RealTimeMonitor",
            metadata=asdict(assessment),
        )

        self.performance_stats["total_alerts_sent"] += 1
        logger.critical(f"重要リスクアラート送信: {assessment.request_id}")

    async def _handle_high_risk(self, assessment: RiskAssessmentSummary):
        """高リスク処理"""

        if not self._check_alert_cooldown(assessment.request_id, "high"):
            return

        await self.alert_manager.create_alert(
            title="⚠️ 高リスクアラート",
            message=f"リスクスコア: {assessment.overall_risk_score:.3f}\n"
            f"監視強化が推奨されます",
            level=AlertLevel.HIGH,
            source="RealTimeMonitor",
            metadata=asdict(assessment),
        )

        self.performance_stats["total_alerts_sent"] += 1

    async def _handle_medium_risk(self, assessment: RiskAssessmentSummary):
        """中程度リスク処理"""

        if not self._check_alert_cooldown(assessment.request_id, "medium"):
            return

        await self.alert_manager.create_alert(
            title="📊 中程度リスクアラート",
            message=f"リスクスコア: {assessment.overall_risk_score:.3f}",
            level=AlertLevel.MEDIUM,
            source="RealTimeMonitor",
            metadata=asdict(assessment),
        )

    def _check_alert_cooldown(self, request_id: str, risk_level: str) -> bool:
        """アラートクールダウンチェック"""

        cooldown_key = f"{request_id}_{risk_level}"
        now = datetime.now()

        if cooldown_key in self.alert_cooldown:
            last_alert = self.alert_cooldown[cooldown_key]
            if now - last_alert < timedelta(minutes=self.config.alert_cooldown_minutes):
                return False

        self.alert_cooldown[cooldown_key] = now

        # 古いクールダウンデータ削除
        cutoff = now - timedelta(hours=1)
        self.alert_cooldown = {
            k: v for k, v in self.alert_cooldown.items() if v > cutoff
        }

        return True

    async def _execute_auto_response(self, assessment: RiskAssessmentSummary):
        """自動応答実行"""

        try:
            risk_category = assessment.risk_category

            if risk_category in self.response_handlers:
                handler = self.response_handlers[risk_category]
                await handler(assessment)
                logger.info(f"自動応答実行: {risk_category} - {assessment.request_id}")

        except Exception as e:
            logger.error(f"自動応答エラー: {e}")

    async def _metrics_collector_loop(self):
        """メトリクス収集ループ"""

        while self.is_running:
            try:
                # システムメトリクス収集
                metrics = MonitoringMetrics(
                    timestamp=datetime.now(),
                    active_monitors=len(self.active_symbols),
                    total_analyses_today=self.performance_stats["total_risk_analyses"],
                    alerts_sent_today=self.performance_stats["total_alerts_sent"],
                    average_risk_score=self._calculate_average_risk_score(),
                    critical_alerts_count=self._count_critical_alerts_today(),
                    system_health_status=self._assess_system_health(),
                    processing_queue_size=self.analysis_queue.qsize(),
                    memory_usage_mb=self._get_memory_usage(),
                )

                # メトリクス履歴に追加
                self.monitoring_metrics.append(metrics)
                if len(self.monitoring_metrics) > 1440:  # 24時間分（1分毎）
                    self.monitoring_metrics = self.monitoring_metrics[-720:]

                await asyncio.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"メトリクス収集エラー: {e}")
                await asyncio.sleep(5)

    async def _batch_analysis_loop(self):
        """バッチ分析ループ"""

        while self.is_running:
            try:
                # 定期的な包括分析実行
                await self._run_comprehensive_batch_analysis()

                # 指定間隔で実行
                await asyncio.sleep(self.config.batch_analysis_interval_minutes * 60)

            except Exception as e:
                logger.error(f"バッチ分析ループエラー: {e}")
                await asyncio.sleep(60)

    async def _alert_manager_loop(self):
        """アラート管理ループ"""

        while self.is_running:
            try:
                # 期限切れアラートのクリーンアップ
                await self._cleanup_expired_alerts()

                # アラート統計更新
                await self._update_alert_statistics()

                await asyncio.sleep(300)  # 5分間隔

            except Exception as e:
                logger.error(f"アラート管理エラー: {e}")
                await asyncio.sleep(60)

    def _calculate_average_risk_score(self) -> float:
        """平均リスクスコア計算"""

        recent_assessments = self.risk_coordinator.get_recent_assessments(50)
        if not recent_assessments:
            return 0.0

        scores = [a.overall_risk_score for a in recent_assessments]
        return np.mean(scores)

    def _count_critical_alerts_today(self) -> int:
        """本日の重要アラート数カウント"""

        today = datetime.now().date()
        critical_count = 0

        for alert_data in self.alert_history:
            alert_date = datetime.fromisoformat(alert_data["timestamp"]).date()
            if alert_date == today and alert_data.get("level") == "critical":
                critical_count += 1

        return critical_count

    def _assess_system_health(self) -> str:
        """システムヘルス評価"""

        error_rate = self.performance_stats["error_count"] / max(
            1, self.performance_stats["total_monitoring_cycles"]
        )

        queue_size = self.analysis_queue.qsize()

        if error_rate > 0.1 or queue_size > 100:
            return "critical"
        elif error_rate > 0.05 or queue_size > 50:
            return "warning"
        else:
            return "healthy"

    def _get_memory_usage(self) -> float:
        """メモリ使用量取得"""

        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

    async def _run_comprehensive_batch_analysis(self):
        """包括バッチ分析実行"""

        logger.info("包括バッチ分析開始")

        try:
            # 全監視銘柄の最新データを取得
            market_data = await self._fetch_market_data()

            # バッチリスク分析用データ作成
            transactions = []
            for symbol, data in market_data.items():
                transactions.append(
                    {
                        "symbol": symbol,
                        "type": "batch_analysis",
                        "amount": 1000000,
                        "timestamp": datetime.now().isoformat(),
                        "market_conditions": data,
                    }
                )

            # バッチ分析実行
            if transactions:
                results = await self.risk_coordinator.batch_risk_assessment(
                    transactions, concurrent_limit=self.config.max_concurrent_analyses
                )

                logger.info(f"バッチ分析完了: {len(results)}件処理")

        except Exception as e:
            logger.error(f"バッチ分析エラー: {e}")

    async def _cleanup_expired_alerts(self):
        """期限切れアラート削除"""

        cutoff = datetime.now() - timedelta(days=7)  # 7日前

        self.alert_history = [
            alert
            for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]

    async def _update_alert_statistics(self):
        """アラート統計更新"""

        today = datetime.now().date()

        if today not in self.daily_statistics:
            self.daily_statistics[today] = {
                "alerts_sent": 0,
                "critical_alerts": 0,
                "high_alerts": 0,
                "medium_alerts": 0,
                "low_alerts": 0,
            }

    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状況取得"""

        uptime = 0
        if self.performance_stats["start_time"]:
            uptime = (
                datetime.now() - self.performance_stats["start_time"]
            ).total_seconds()

        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "active_symbols": len(self.active_symbols),
            "performance_stats": self.performance_stats,
            "current_metrics": self.monitoring_metrics[-1]
            if self.monitoring_metrics
            else None,
            "alert_queue_size": self.analysis_queue.qsize(),
            "system_health": self._assess_system_health(),
            "daily_statistics": dict(
                list(self.daily_statistics.items())[-7:]
            ),  # 最新7日分
        }

    def register_response_handler(self, risk_level: str, handler: Callable):
        """自動応答ハンドラー登録"""

        self.response_handlers[risk_level] = handler
        logger.info(f"自動応答ハンドラー登録: {risk_level}")


# テスト・デモ用関数
async def test_realtime_monitor():
    """リアルタイム監視テスト"""

    # テスト用設定
    config = RiskMonitoringConfig(
        monitoring_interval_seconds=2,
        batch_analysis_interval_minutes=5,
        alert_cooldown_minutes=1,
    )

    monitor = RealTimeRiskMonitor(config)

    # テスト用応答ハンドラー
    async def critical_response_handler(assessment):
        print(f"🚨 重要リスク自動応答: {assessment.request_id}")

    async def high_response_handler(assessment):
        print(f"⚠️ 高リスク自動応答: {assessment.request_id}")

    # ハンドラー登録
    monitor.register_response_handler("critical", critical_response_handler)
    monitor.register_response_handler("high", high_response_handler)

    # テスト銘柄
    test_symbols = ["7203", "6758", "9984"]  # トヨタ、ソニー、ソフトバンク

    print("🖥️ リアルタイムリスク監視システムテスト開始")
    print(f"📊 監視銘柄: {', '.join(test_symbols)}")
    print("⏱️ テスト時間: 30秒")

    # 監視開始（30秒間）
    monitor_task = asyncio.create_task(monitor.start_monitoring(test_symbols))

    try:
        await asyncio.wait_for(monitor_task, timeout=30)
    except asyncio.TimeoutError:
        await monitor.stop_monitoring()

    # 結果表示
    status = monitor.get_monitoring_status()
    print("\n📈 監視結果:")
    print(f"  総監視サイクル: {status['performance_stats']['total_monitoring_cycles']}")
    print(f"  総分析数: {status['performance_stats']['total_risk_analyses']}")
    print(f"  送信アラート数: {status['performance_stats']['total_alerts_sent']}")
    print(f"  エラー数: {status['performance_stats']['error_count']}")
    print(f"  システムヘルス: {status['system_health']}")


if __name__ == "__main__":
    asyncio.run(test_realtime_monitor())
