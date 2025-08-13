#!/usr/bin/env python3
"""
Model Deployment Manager - ロールアウト/ロールバックシステム

Issue #733対応: MLモデルの段階的デプロイメントとロールバック機能
安全で制御されたモデル配信を実現

主要機能:
- 段階的ロールアウト（Canary Deployment）
- 自動ロールバック（メトリクス監視ベース）
- デプロイメント履歴管理
- リアルタイム監視・アラート
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import numpy as np

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class DeploymentStatus(Enum):
    """デプロイメントステータス"""
    DRAFT = "draft"  # 下書き
    ROLLOUT_STARTING = "rollout_starting"  # ロールアウト開始
    ROLLOUT_IN_PROGRESS = "rollout_in_progress"  # ロールアウト中
    ROLLOUT_COMPLETED = "rollout_completed"  # ロールアウト完了
    ROLLOUT_PAUSED = "rollout_paused"  # ロールアウト一時停止
    ROLLBACK_TRIGGERED = "rollback_triggered"  # ロールバック実行中
    ROLLBACK_COMPLETED = "rollback_completed"  # ロールバック完了
    FAILED = "failed"  # 失敗


class DeploymentStrategy(Enum):
    """デプロイメント戦略"""
    BLUE_GREEN = "blue_green"  # Blue-Green デプロイメント
    CANARY = "canary"  # カナリアデプロイメント
    ROLLING = "rolling"  # ローリングデプロイメント
    IMMEDIATE = "immediate"  # 即座に全切り替え


class RollbackTrigger(Enum):
    """ロールバックトリガー"""
    MANUAL = "manual"  # 手動
    ERROR_RATE_THRESHOLD = "error_rate_threshold"  # エラー率閾値
    LATENCY_THRESHOLD = "latency_threshold"  # レイテンシ閾値
    ACCURACY_THRESHOLD = "accuracy_threshold"  # 精度閾値
    CUSTOM_METRIC = "custom_metric"  # カスタムメトリクス


@dataclass
class ModelVersion:
    """モデルバージョン情報"""
    version_id: str
    name: str
    description: str
    model_path: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """デプロイメント設定"""
    deployment_id: str
    name: str
    source_version: ModelVersion  # デプロイするバージョン
    target_environment: str = "production"
    strategy: DeploymentStrategy = DeploymentStrategy.CANARY

    # カナリアデプロイメント設定
    initial_traffic_percentage: float = 0.05  # 初期トラフィック（5%）
    rollout_increments: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0])
    increment_duration_minutes: int = 30  # 各段階の持続時間

    # 監視・ロールバック設定
    monitoring_window_minutes: int = 15  # 監視ウィンドウ
    error_rate_threshold: float = 0.05  # エラー率閾値（5%）
    latency_p99_threshold_ms: float = 1000.0  # 99パーセンタイルレイテンシ閾値
    accuracy_threshold: float = 0.8  # 精度閾値
    auto_rollback_enabled: bool = True

    # 通知設定
    notification_webhooks: List[str] = field(default_factory=list)
    alert_email_addresses: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """デプロイメントメトリクス"""
    timestamp: datetime
    deployment_id: str
    traffic_percentage: float
    request_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    accuracy: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DeploymentEvent:
    """デプロイメントイベント"""
    event_id: str
    deployment_id: str
    timestamp: datetime
    event_type: str  # "rollout_started", "traffic_increased", "rollback_triggered" etc.
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # "info", "warning", "error", "critical"


@dataclass
class DeploymentState:
    """デプロイメント状態"""
    deployment_id: str
    status: DeploymentStatus
    current_traffic_percentage: float = 0.0
    current_stage: int = 0
    total_stages: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated_at: datetime = field(default_factory=datetime.now)
    rollback_reason: Optional[str] = None
    events: List[DeploymentEvent] = field(default_factory=list)
    metrics_history: List[DeploymentMetrics] = field(default_factory=list)


class DeploymentHealthChecker:
    """デプロイメントヘルスチェッカー"""

    def __init__(self, config: DeploymentConfig):
        """初期化"""
        self.config = config
        self.metrics_buffer: List[DeploymentMetrics] = []
        self.monitoring_window_size = max(1, config.monitoring_window_minutes)

    def add_metrics(self, metrics: DeploymentMetrics):
        """メトリクスの追加"""
        self.metrics_buffer.append(metrics)

        # ウィンドウサイズを超えた古いメトリクスを削除
        cutoff_time = datetime.now() - timedelta(minutes=self.monitoring_window_size)
        self.metrics_buffer = [
            m for m in self.metrics_buffer
            if m.timestamp > cutoff_time
        ]

    def check_health(self) -> Tuple[bool, List[str]]:
        """
        ヘルスチェックの実行

        Returns:
            (健全かどうか, 問題の詳細リスト)
        """
        if not self.metrics_buffer:
            return True, []

        issues = []

        # 最新メトリクスを取得
        latest_metrics = self.metrics_buffer[-1]

        # エラー率チェック
        if latest_metrics.error_rate > self.config.error_rate_threshold:
            issues.append(
                f"エラー率が閾値を超過: {latest_metrics.error_rate:.3f} > {self.config.error_rate_threshold:.3f}"
            )

        # レイテンシチェック
        if latest_metrics.p99_latency_ms > self.config.latency_p99_threshold_ms:
            issues.append(
                f"P99レイテンシが閾値を超過: {latest_metrics.p99_latency_ms:.1f}ms > {self.config.latency_p99_threshold_ms:.1f}ms"
            )

        # 精度チェック
        if (latest_metrics.accuracy is not None and
            latest_metrics.accuracy < self.config.accuracy_threshold):
            issues.append(
                f"精度が閾値を下回る: {latest_metrics.accuracy:.3f} < {self.config.accuracy_threshold:.3f}"
            )

        # 複数メトリクスでのトレンド分析
        if len(self.metrics_buffer) >= 3:
            self._check_trends(issues)

        return len(issues) == 0, issues

    def _check_trends(self, issues: List[str]):
        """トレンド分析"""
        if len(self.metrics_buffer) < 3:
            return

        recent_metrics = self.metrics_buffer[-3:]

        # エラー率の増加トレンド
        error_rates = [m.error_rate for m in recent_metrics]
        if len(error_rates) == 3 and error_rates[0] < error_rates[1] < error_rates[2]:
            if error_rates[2] - error_rates[0] > 0.01:  # 1%以上の増加
                issues.append(f"エラー率が連続増加: {error_rates[0]:.3f} → {error_rates[2]:.3f}")

        # レイテンシの増加トレンド
        latencies = [m.avg_latency_ms for m in recent_metrics]
        if len(latencies) == 3 and latencies[0] < latencies[1] < latencies[2]:
            increase_rate = (latencies[2] - latencies[0]) / latencies[0]
            if increase_rate > 0.2:  # 20%以上の増加
                issues.append(f"レイテンシが連続増加: {latencies[0]:.1f}ms → {latencies[2]:.1f}ms")


class NotificationManager:
    """通知管理"""

    def __init__(self, config: DeploymentConfig):
        """初期化"""
        self.config = config
        self.notification_callbacks: List[Callable] = []

    def add_notification_callback(self, callback: Callable[[str, str, Dict], None]):
        """通知コールバックの追加"""
        self.notification_callbacks.append(callback)

    async def send_notification(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """通知の送信"""
        details = details or {}

        # ログ出力
        logger.info(f"Deployment Notification [{event_type}]: {message}")

        # 登録されたコールバックを実行
        for callback in self.notification_callbacks:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, callback, event_type, message, details
                )
            except Exception as e:
                logger.error(f"通知コールバック実行エラー: {e}")

        # Webhook通知（実装省略 - 実際のプロジェクトでは HTTP POST 等）
        for webhook_url in self.config.notification_webhooks:
            logger.info(f"Webhook通知 ({webhook_url}): {message}")

        # メール通知（実装省略 - 実際のプロジェクトでは SMTP 等）
        for email in self.config.alert_email_addresses:
            logger.info(f"メール通知 ({email}): {message}")


class ModelDeploymentManager:
    """モデルデプロイメント管理"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        初期化

        Args:
            storage_path: デプロイメント情報保存パス
        """
        self.storage_path = Path(storage_path) if storage_path else Path("data/model_deployment")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.active_deployments: Dict[str, DeploymentState] = {}
        self.health_checkers: Dict[str, DeploymentHealthChecker] = {}
        self.notification_managers: Dict[str, NotificationManager] = {}
        self.model_registry: Dict[str, ModelVersion] = {}

        # バックグラウンドタスク用
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

        logger.info(f"モデルデプロイメント管理システム初期化完了: {self.storage_path}")

    async def register_model_version(self, version: ModelVersion) -> bool:
        """
        モデルバージョンの登録

        Args:
            version: モデルバージョン情報

        Returns:
            登録成功かどうか
        """
        try:
            if version.version_id in self.model_registry:
                logger.warning(f"モデルバージョン {version.version_id} は既に登録されています")
                return False

            self.model_registry[version.version_id] = version

            # 永続化
            await self._save_model_version(version)

            logger.info(f"モデルバージョン {version.version_id} ({version.name}) を登録しました")
            return True

        except Exception as e:
            logger.error(f"モデルバージョン登録エラー: {e}")
            return False

    async def create_deployment(self, config: DeploymentConfig) -> bool:
        """
        新しいデプロイメントの作成

        Args:
            config: デプロイメント設定

        Returns:
            作成成功かどうか
        """
        try:
            # 設定検証
            if not self._validate_deployment_config(config):
                return False

            # デプロイメント状態の初期化
            state = DeploymentState(
                deployment_id=config.deployment_id,
                status=DeploymentStatus.DRAFT,
                total_stages=len(config.rollout_increments)
            )

            # ヘルスチェッカーと通知管理の初期化
            health_checker = DeploymentHealthChecker(config)
            notification_manager = NotificationManager(config)

            # 登録
            self.active_deployments[config.deployment_id] = state
            self.health_checkers[config.deployment_id] = health_checker
            self.notification_managers[config.deployment_id] = notification_manager

            # 設定の永続化
            await self._save_deployment_config(config)
            await self._save_deployment_state(state)

            logger.info(f"デプロイメント {config.deployment_id} ({config.name}) を作成しました")
            return True

        except Exception as e:
            logger.error(f"デプロイメント作成エラー: {e}")
            return False

    async def start_deployment(self, deployment_id: str) -> bool:
        """
        デプロイメントの開始

        Args:
            deployment_id: デプロイメントID

        Returns:
            開始成功かどうか
        """
        try:
            if deployment_id not in self.active_deployments:
                logger.error(f"デプロイメント {deployment_id} が見つかりません")
                return False

            state = self.active_deployments[deployment_id]

            if state.status != DeploymentStatus.DRAFT:
                logger.error(f"デプロイメント {deployment_id} は既に開始されています")
                return False

            # ステータス更新
            state.status = DeploymentStatus.ROLLOUT_STARTING
            state.started_at = datetime.now()
            state.current_stage = 0

            # イベント記録
            await self._record_event(
                deployment_id, "rollout_started", "デプロイメントロールアウト開始"
            )

            # 通知送信
            notification_manager = self.notification_managers[deployment_id]
            await notification_manager.send_notification(
                "deployment_started",
                f"デプロイメント {deployment_id} のロールアウトを開始しました"
            )

            # バックグラウンド監視タスクの開始
            self.monitoring_tasks[deployment_id] = asyncio.create_task(
                self._monitor_deployment(deployment_id)
            )

            logger.info(f"デプロイメント {deployment_id} のロールアウトを開始しました")
            return True

        except Exception as e:
            logger.error(f"デプロイメント開始エラー: {e}")
            return False

    async def record_metrics(self, deployment_id: str, metrics: DeploymentMetrics) -> bool:
        """
        デプロイメントメトリクスの記録

        Args:
            deployment_id: デプロイメントID
            metrics: メトリクスデータ

        Returns:
            記録成功かどうか
        """
        try:
            if deployment_id not in self.active_deployments:
                return False

            state = self.active_deployments[deployment_id]
            health_checker = self.health_checkers[deployment_id]

            # メトリクスの記録
            state.metrics_history.append(metrics)
            health_checker.add_metrics(metrics)

            # ヘルスチェック
            if state.status in [DeploymentStatus.ROLLOUT_IN_PROGRESS]:
                is_healthy, issues = health_checker.check_health()

                if not is_healthy and self._get_config(deployment_id).auto_rollback_enabled:
                    # 自動ロールバック実行
                    rollback_reason = f"ヘルスチェック失敗: {'; '.join(issues)}"
                    await self._trigger_rollback(deployment_id, rollback_reason)

            return True

        except Exception as e:
            logger.error(f"メトリクス記録エラー: {e}")
            return False

    async def trigger_manual_rollback(self, deployment_id: str, reason: str) -> bool:
        """
        手動ロールバックの実行

        Args:
            deployment_id: デプロイメントID
            reason: ロールバック理由

        Returns:
            実行成功かどうか
        """
        return await self._trigger_rollback(deployment_id, reason)

    async def pause_deployment(self, deployment_id: str) -> bool:
        """
        デプロイメントの一時停止

        Args:
            deployment_id: デプロイメントID

        Returns:
            一時停止成功かどうか
        """
        try:
            if deployment_id not in self.active_deployments:
                return False

            state = self.active_deployments[deployment_id]

            if state.status == DeploymentStatus.ROLLOUT_IN_PROGRESS:
                state.status = DeploymentStatus.ROLLOUT_PAUSED

                await self._record_event(
                    deployment_id, "deployment_paused", "デプロイメント一時停止"
                )

                logger.info(f"デプロイメント {deployment_id} を一時停止しました")
                return True

            return False

        except Exception as e:
            logger.error(f"デプロイメント一時停止エラー: {e}")
            return False

    async def resume_deployment(self, deployment_id: str) -> bool:
        """
        デプロイメントの再開

        Args:
            deployment_id: デプロイメントID

        Returns:
            再開成功かどうか
        """
        try:
            if deployment_id not in self.active_deployments:
                return False

            state = self.active_deployments[deployment_id]

            if state.status == DeploymentStatus.ROLLOUT_PAUSED:
                state.status = DeploymentStatus.ROLLOUT_IN_PROGRESS

                await self._record_event(
                    deployment_id, "deployment_resumed", "デプロイメント再開"
                )

                logger.info(f"デプロイメント {deployment_id} を再開しました")
                return True

            return False

        except Exception as e:
            logger.error(f"デプロイメント再開エラー: {e}")
            return False

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentState]:
        """
        デプロイメント状態の取得

        Args:
            deployment_id: デプロイメントID

        Returns:
            デプロイメント状態
        """
        return self.active_deployments.get(deployment_id)

    def list_active_deployments(self) -> List[str]:
        """
        アクティブなデプロイメント一覧

        Returns:
            デプロイメントIDリスト
        """
        return list(self.active_deployments.keys())

    async def _monitor_deployment(self, deployment_id: str):
        """デプロイメント監視バックグラウンドタスク"""
        try:
            config = self._get_config(deployment_id)
            state = self.active_deployments[deployment_id]

            # 初期段階の設定
            await self._advance_to_next_stage(deployment_id)

            while (state.status in [DeploymentStatus.ROLLOUT_STARTING,
                                   DeploymentStatus.ROLLOUT_IN_PROGRESS] and
                   state.current_stage < state.total_stages):

                # 段階待機時間
                await asyncio.sleep(config.increment_duration_minutes * 60)

                # 一時停止中は待機
                while state.status == DeploymentStatus.ROLLOUT_PAUSED:
                    await asyncio.sleep(30)

                # ロールバックが実行された場合は終了
                if state.status == DeploymentStatus.ROLLBACK_TRIGGERED:
                    break

                # 次の段階に進む
                if state.current_stage < state.total_stages:
                    await self._advance_to_next_stage(deployment_id)

            # 完了処理
            if (state.status == DeploymentStatus.ROLLOUT_IN_PROGRESS and
                state.current_stage >= state.total_stages):
                await self._complete_deployment(deployment_id)

        except Exception as e:
            logger.error(f"デプロイメント監視エラー: {e}")

    async def _advance_to_next_stage(self, deployment_id: str):
        """次の段階への進行"""
        config = self._get_config(deployment_id)
        state = self.active_deployments[deployment_id]

        if state.current_stage < len(config.rollout_increments):
            # トラフィック割合の更新
            state.current_traffic_percentage = config.rollout_increments[state.current_stage]
            state.current_stage += 1
            state.status = DeploymentStatus.ROLLOUT_IN_PROGRESS

            await self._record_event(
                deployment_id,
                "traffic_increased",
                f"トラフィック {state.current_traffic_percentage*100:.1f}% に増加 (段階 {state.current_stage}/{state.total_stages})"
            )

            # 通知送信
            notification_manager = self.notification_managers[deployment_id]
            await notification_manager.send_notification(
                "traffic_increased",
                f"デプロイメント {deployment_id}: トラフィック {state.current_traffic_percentage*100:.1f}% に増加"
            )

            logger.info(f"デプロイメント {deployment_id}: 段階 {state.current_stage}/{state.total_stages} (トラフィック {state.current_traffic_percentage*100:.1f}%)")

    async def _complete_deployment(self, deployment_id: str):
        """デプロイメント完了処理"""
        state = self.active_deployments[deployment_id]
        state.status = DeploymentStatus.ROLLOUT_COMPLETED
        state.completed_at = datetime.now()

        await self._record_event(
            deployment_id, "rollout_completed", "デプロイメントロールアウト完了"
        )

        # 通知送信
        notification_manager = self.notification_managers[deployment_id]
        await notification_manager.send_notification(
            "deployment_completed",
            f"デプロイメント {deployment_id} のロールアウトが完了しました"
        )

        logger.info(f"デプロイメント {deployment_id} のロールアウトが完了しました")

    async def _trigger_rollback(self, deployment_id: str, reason: str) -> bool:
        """ロールバック実行"""
        try:
            state = self.active_deployments[deployment_id]
            state.status = DeploymentStatus.ROLLBACK_TRIGGERED
            state.rollback_reason = reason

            # トラフィックを0%に戻す
            state.current_traffic_percentage = 0.0

            await self._record_event(
                deployment_id, "rollback_triggered", f"ロールバック実行: {reason}"
            )

            # 通知送信
            notification_manager = self.notification_managers[deployment_id]
            await notification_manager.send_notification(
                "rollback_triggered",
                f"デプロイメント {deployment_id} のロールバックを実行: {reason}",
                {"reason": reason}
            )

            # ロールバック完了
            state.status = DeploymentStatus.ROLLBACK_COMPLETED
            state.completed_at = datetime.now()

            await self._record_event(
                deployment_id, "rollback_completed", "ロールバック完了"
            )

            logger.warning(f"デプロイメント {deployment_id} をロールバックしました: {reason}")
            return True

        except Exception as e:
            logger.error(f"ロールバック実行エラー: {e}")
            return False

    async def _record_event(self, deployment_id: str, event_type: str, message: str,
                          details: Dict[str, Any] = None, severity: str = "info"):
        """イベントの記録"""
        if deployment_id not in self.active_deployments:
            return

        event = DeploymentEvent(
            event_id=str(uuid.uuid4()),
            deployment_id=deployment_id,
            timestamp=datetime.now(),
            event_type=event_type,
            message=message,
            details=details or {},
            severity=severity
        )

        state = self.active_deployments[deployment_id]
        state.events.append(event)
        state.last_updated_at = datetime.now()

        # 定期的な永続化
        if len(state.events) % 10 == 0:
            await self._save_deployment_state(state)

    def _validate_deployment_config(self, config: DeploymentConfig) -> bool:
        """デプロイメント設定の検証"""
        # ソースバージョンの存在チェック
        if config.source_version.version_id not in self.model_registry:
            logger.error(f"ソースモデルバージョン {config.source_version.version_id} が見つかりません")
            return False

        # トラフィック配分の検証
        if not config.rollout_increments or config.rollout_increments[-1] != 1.0:
            logger.error("ロールアウト段階の最終値は1.0である必要があります")
            return False

        # 閾値の検証
        if config.error_rate_threshold <= 0 or config.error_rate_threshold >= 1:
            logger.error("エラー率閾値は0-1の範囲である必要があります")
            return False

        return True

    def _get_config(self, deployment_id: str) -> DeploymentConfig:
        """デプロイメント設定の取得"""
        # 実装簡略化のため、ここでは基本設定を返す
        # 実際のプロジェクトでは永続化された設定を読み込む
        return DeploymentConfig(
            deployment_id=deployment_id,
            name=f"Deployment {deployment_id}",
            source_version=ModelVersion(
                version_id="dummy",
                name="Dummy Model",
                description="Dummy",
                model_path="/dummy/path"
            )
        )

    async def _save_model_version(self, version: ModelVersion):
        """モデルバージョンの永続化"""
        version_path = self.storage_path / f"model_version_{version.version_id}.json"

        with open(version_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(version), f, ensure_ascii=False, indent=2, default=str)

    async def _save_deployment_config(self, config: DeploymentConfig):
        """デプロイメント設定の永続化"""
        config_path = self.storage_path / f"deployment_{config.deployment_id}_config.json"

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2, default=str)

    async def _save_deployment_state(self, state: DeploymentState):
        """デプロイメント状態の永続化"""
        state_path = self.storage_path / f"deployment_{state.deployment_id}_state.json"

        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2, default=str)


# ユーティリティ関数
def create_canary_deployment(
    deployment_name: str,
    model_version: ModelVersion,
    canary_stages: List[float] = None,
    stage_duration_minutes: int = 30
) -> DeploymentConfig:
    """
    カナリアデプロイメント設定の作成ヘルパー

    Args:
        deployment_name: デプロイメント名
        model_version: デプロイするモデルバージョン
        canary_stages: カナリア段階設定
        stage_duration_minutes: 各段階の持続時間

    Returns:
        デプロイメント設定
    """
    deployment_id = f"canary_deploy_{uuid.uuid4().hex[:8]}"
    canary_stages = canary_stages or [0.05, 0.1, 0.25, 0.5, 1.0]

    return DeploymentConfig(
        deployment_id=deployment_id,
        name=deployment_name,
        source_version=model_version,
        strategy=DeploymentStrategy.CANARY,
        rollout_increments=canary_stages,
        increment_duration_minutes=stage_duration_minutes
    )


async def main():
    """デモンストレーション"""
    logger.info("モデルデプロイメント管理システム デモンストレーション開始")

    # デプロイメント管理システムの初期化
    deployment_manager = ModelDeploymentManager("data/deployment_demo")

    # ダミーモデルバージョンの作成・登録
    model_version = ModelVersion(
        version_id="rf_v2.1.0",
        name="RandomForest v2.1.0",
        description="精度改善版RandomForestモデル",
        model_path="/models/random_forest_v2.1.0.joblib",
        config={"n_estimators": 150, "max_depth": 20}
    )

    await deployment_manager.register_model_version(model_version)

    # カナリアデプロイメント設定の作成
    deployment_config = create_canary_deployment(
        deployment_name="RandomForest v2.1.0 Canary Rollout",
        model_version=model_version,
        canary_stages=[0.05, 0.25, 1.0],  # 5% → 25% → 100%
        stage_duration_minutes=1  # デモ用に短時間設定
    )

    # デプロイメントの作成
    success = await deployment_manager.create_deployment(deployment_config)
    if success:
        logger.info(f"カナリアデプロイメント {deployment_config.deployment_id} を作成しました")

        # デプロイメントの開始
        await deployment_manager.start_deployment(deployment_config.deployment_id)

        # シミュレートされたメトリクス記録
        for i in range(5):
            metrics = DeploymentMetrics(
                timestamp=datetime.now(),
                deployment_id=deployment_config.deployment_id,
                traffic_percentage=0.05 * (i + 1),
                request_count=1000 + i * 100,
                error_count=5 + i,
                error_rate=0.005 + 0.001 * i,
                avg_latency_ms=50.0 + i * 5,
                p99_latency_ms=200.0 + i * 10,
                accuracy=0.95 - 0.001 * i
            )

            await deployment_manager.record_metrics(deployment_config.deployment_id, metrics)
            await asyncio.sleep(1)

        # デプロイメント状態の確認
        state = deployment_manager.get_deployment_status(deployment_config.deployment_id)
        if state:
            logger.info(f"デプロイメント状態: {state.status.value}")
            logger.info(f"現在のトラフィック: {state.current_traffic_percentage*100:.1f}%")
            logger.info(f"段階: {state.current_stage}/{state.total_stages}")

    logger.info("モデルデプロイメント管理システム デモンストレーション完了")


if __name__ == "__main__":
    asyncio.run(main())