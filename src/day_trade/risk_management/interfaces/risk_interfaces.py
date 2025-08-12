#!/usr/bin/env python3
"""
Risk Management Core Interfaces
リスク管理コアインターフェース

依存関係逆転の原則(DIP)を適用したメインインターフェース定義
循環依存を防ぎ、テスタビリティと拡張性を向上
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# 前方参照用の型定義
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..models.unified_models import UnifiedRiskRequest, UnifiedRiskResult


class RiskLevel(Enum):
    """リスクレベル定義"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisStatus(Enum):
    """分析ステータス"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RiskAnalyzerMetadata:
    """リスク分析器メタデータ"""

    name: str
    version: str
    description: str
    supported_request_types: List[str]
    expected_processing_time_ms: float
    confidence_threshold: float
    weight: float = 1.0


class IRiskAnalyzer(ABC):
    """リスク分析器インターフェース"""

    @abstractmethod
    def get_metadata(self) -> RiskAnalyzerMetadata:
        """分析器メタデータ取得"""
        pass

    @abstractmethod
    async def analyze_risk(self, request: "UnifiedRiskRequest") -> "UnifiedRiskResult":
        """リスク分析実行"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        pass

    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """設定取得"""
        pass

    @abstractmethod
    async def update_configuration(self, config: Dict[str, Any]) -> bool:
        """設定更新"""
        pass


class IAlertManager(ABC):
    """アラート管理インターフェース"""

    @abstractmethod
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """アラート送信"""
        pass

    @abstractmethod
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """アクティブアラート取得"""
        pass

    @abstractmethod
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """アラート承認"""
        pass

    @abstractmethod
    async def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""
        pass

    @abstractmethod
    def subscribe_to_alerts(
        self, callback: callable, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> str:
        """アラート購読"""
        pass


class ICacheManager(ABC):
    """キャッシュ管理インターフェース"""

    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        """キャッシュ取得"""
        pass

    @abstractmethod
    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """キャッシュ設定"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """キャッシュ削除"""
        pass

    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """キャッシュクリア"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """キャッシュ存在確認"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        pass


class IMetricsCollector(ABC):
    """メトリクス収集インターフェース"""

    @abstractmethod
    def record_counter(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """カウンターメトリクス記録"""
        pass

    @abstractmethod
    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """ヒストグラムメトリクス記録"""
        pass

    @abstractmethod
    def record_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """ゲージメトリクス記録"""
        pass

    @abstractmethod
    async def get_metrics(self, name_pattern: Optional[str] = None) -> Dict[str, Any]:
        """メトリクス取得"""
        pass

    @abstractmethod
    def create_timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """タイマーコンテキスト作成"""
        pass


class IConfigManager(ABC):
    """設定管理インターフェース"""

    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """設定取得"""
        pass

    @abstractmethod
    async def set_config(self, key: str, value: Any) -> bool:
        """設定更新"""
        pass

    @abstractmethod
    async def get_all_configs(self) -> Dict[str, Any]:
        """全設定取得"""
        pass

    @abstractmethod
    async def reload_config(self) -> bool:
        """設定リロード"""
        pass

    @abstractmethod
    def subscribe_to_changes(
        self, callback: callable, key_pattern: Optional[str] = None
    ) -> str:
        """設定変更購読"""
        pass


class IRiskAnalysisOrchestrator(ABC):
    """リスク分析オーケストレーターインターフェース"""

    @abstractmethod
    async def add_analyzer(self, analyzer: IRiskAnalyzer) -> bool:
        """分析器追加"""
        pass

    @abstractmethod
    async def remove_analyzer(self, analyzer_name: str) -> bool:
        """分析器削除"""
        pass

    @abstractmethod
    async def execute_analysis(
        self, request: "UnifiedRiskRequest", analyzer_names: Optional[List[str]] = None
    ) -> "UnifiedRiskResult":
        """統合分析実行"""
        pass

    @abstractmethod
    async def get_analyzer_status(self) -> Dict[str, Dict[str, Any]]:
        """分析器ステータス取得"""
        pass


class IEventBus(ABC):
    """イベントバスインターフェース"""

    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> bool:
        """イベント発行"""
        pass

    @abstractmethod
    def subscribe(self, event_type: str, handler: callable) -> str:
        """イベント購読"""
        pass

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """イベント購読解除"""
        pass


# ヘルパー関数とユーティリティ


def validate_risk_level(level: str) -> bool:
    """リスクレベル検証"""
    try:
        RiskLevel(level.lower())
        return True
    except ValueError:
        return False


def calculate_weighted_risk_score(
    scores: List[Tuple[float, float]],  # (score, weight) のリスト
    normalization_factor: float = 1.0,
) -> float:
    """重み付きリスクスコア計算"""
    if not scores:
        return 0.0

    weighted_sum = sum(score * weight for score, weight in scores)
    total_weight = sum(weight for _, weight in scores)

    if total_weight == 0:
        return 0.0

    return min(1.0, max(0.0, (weighted_sum / total_weight) * normalization_factor))


def create_analysis_context(
    request_id: str,
    timestamp: datetime,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """分析コンテキスト作成"""
    context = {
        "request_id": request_id,
        "timestamp": timestamp.isoformat(),
        "created_at": datetime.now().isoformat(),
    }

    if additional_metadata:
        context.update(additional_metadata)

    return context
