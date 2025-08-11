"""
プラグインインターフェース定義

カスタムリスク分析器の統一インターフェース・データクラス定義
"""

import abc
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class PluginType(Enum):
    """プラグインタイプ"""

    RISK_ANALYZER = "risk_analyzer"
    MARKET_MONITOR = "market_monitor"
    COMPLIANCE_CHECK = "compliance_check"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TECHNICAL_INDICATOR = "technical_indicator"


class RiskLevel(Enum):
    """リスクレベル"""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class PluginInfo:
    """プラグイン情報"""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    config_schema: Dict[str, Any]
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class RiskAnalysisResult:
    """リスク分析結果"""

    plugin_name: str
    timestamp: datetime
    risk_level: RiskLevel
    risk_score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    message: str
    details: Dict[str, Any]
    recommendations: List[str]
    alerts: List[str]
    metadata: Dict[str, Any]


@dataclass
class MarketData:
    """市場データ"""

    symbol: str
    timestamp: datetime
    price_data: pd.DataFrame
    volume_data: pd.DataFrame
    fundamentals: Optional[Dict[str, Any]] = None
    technical_indicators: Optional[Dict[str, float]] = None
    news_data: Optional[List[Dict[str, Any]]] = None
    sentiment_data: Optional[Dict[str, Any]] = None


@dataclass
class PortfolioData:
    """ポートフォリオデータ"""

    positions: Dict[str, Dict[str, Any]]
    total_value: float
    available_cash: float
    allocation: Dict[str, float]
    performance_metrics: Dict[str, float]
    historical_data: Optional[pd.DataFrame] = None


class IRiskPlugin(abc.ABC):
    """リスクプラグインインターフェース"""

    def __init__(self, config: Dict[str, Any]):
        """
        プラグイン初期化

        Args:
            config: プラグイン設定辞書
        """
        self.config = config
        self._info = None
        self._initialized = False

    @abc.abstractmethod
    def get_info(self) -> PluginInfo:
        """
        プラグイン情報取得

        Returns:
            プラグイン情報
        """
        pass

    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        プラグイン初期化実行

        Returns:
            初期化成功フラグ
        """
        pass

    @abc.abstractmethod
    def analyze_risk(
        self,
        market_data: Union[MarketData, List[MarketData]],
        portfolio_data: Optional[PortfolioData] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RiskAnalysisResult:
        """
        リスク分析実行

        Args:
            market_data: 市場データ
            portfolio_data: ポートフォリオデータ
            context: 追加コンテキスト

        Returns:
            リスク分析結果
        """
        pass

    @abc.abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        設定検証

        Args:
            config: 設定辞書

        Returns:
            検証成功フラグ
        """
        pass

    def cleanup(self) -> None:
        """プラグインクリーンアップ"""
        pass

    @property
    def is_initialized(self) -> bool:
        """初期化状態"""
        return self._initialized

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        設定更新

        Args:
            new_config: 新設定

        Returns:
            更新成功フラグ
        """
        if self.validate_config(new_config):
            self.config.update(new_config)
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """
        プラグイン状態取得

        Returns:
            状態情報
        """
        return {
            "initialized": self._initialized,
            "config": self.config,
            "last_analysis": getattr(self, "_last_analysis_time", None),
        }


class IMarketMonitorPlugin(IRiskPlugin):
    """市場監視プラグインインターフェース"""

    @abc.abstractmethod
    def monitor_market(
        self, symbols: List[str], monitoring_config: Dict[str, Any]
    ) -> List[RiskAnalysisResult]:
        """
        市場監視実行

        Args:
            symbols: 監視対象銘柄
            monitoring_config: 監視設定

        Returns:
            監視結果リスト
        """
        pass


class ICompliancePlugin(IRiskPlugin):
    """コンプライアンスプラグインインターフェース"""

    @abc.abstractmethod
    def check_compliance(
        self,
        trade_data: Dict[str, Any],
        portfolio_data: PortfolioData,
        regulations: Dict[str, Any],
    ) -> RiskAnalysisResult:
        """
        コンプライアンスチェック

        Args:
            trade_data: 取引データ
            portfolio_data: ポートフォリオデータ
            regulations: 規制情報

        Returns:
            コンプライアンス結果
        """
        pass


class ISentimentPlugin(IRiskPlugin):
    """センチメント分析プラグインインターフェース"""

    @abc.abstractmethod
    def analyze_sentiment(
        self,
        news_data: List[Dict[str, Any]],
        social_data: Optional[List[Dict[str, Any]]] = None,
        historical_sentiment: Optional[pd.DataFrame] = None,
    ) -> RiskAnalysisResult:
        """
        センチメント分析実行

        Args:
            news_data: ニュースデータ
            social_data: ソーシャルデータ
            historical_sentiment: 過去センチメント

        Returns:
            センチメント分析結果
        """
        pass
