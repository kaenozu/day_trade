#!/usr/bin/env python3
"""
Unified Data Models
統一データモデル

リスク管理システム全体で使用される標準化されたデータモデル
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

# 基本データ型定義


class AssetType(Enum):
    """資産タイプ"""

    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    BOND = "bond"
    DERIVATIVE = "derivative"


class PositionType(Enum):
    """ポジションタイプ"""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class RiskLevel(Enum):
    """リスクレベル"""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class AnalysisStatus(Enum):
    """分析ステータス"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 基本エンティティモデル


@dataclass
class Asset:
    """資産情報"""

    symbol: str
    asset_type: AssetType
    exchange: Optional[str] = None
    currency: str = "USD"
    name: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.symbol = self.symbol.upper()


@dataclass
class MarketData:
    """市場データ"""

    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Optional[Decimal] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    open_price: Optional[Decimal] = None
    close_price: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """ポジション情報"""

    position_id: str
    asset: Asset
    position_type: PositionType
    quantity: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> Decimal:
        """時価総額"""
        price = self.current_price or self.entry_price
        return self.quantity * price

    @property
    def unrealized_pnl(self) -> Optional[Decimal]:
        """含み損益"""
        if self.current_price is None:
            return None

        price_diff = self.current_price - self.entry_price
        if self.position_type == PositionType.SHORT:
            price_diff = -price_diff

        return self.quantity * price_diff


@dataclass
class Portfolio:
    """ポートフォリオ"""

    portfolio_id: str
    name: str
    positions: List[Position] = field(default_factory=list)
    cash: Decimal = Decimal("0")
    currency: str = "USD"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_market_value(self) -> Decimal:
        """総時価額"""
        return sum(pos.market_value for pos in self.positions) + self.cash

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """総含み損益"""
        return sum(pos.unrealized_pnl or Decimal("0") for pos in self.positions)


# リスク分析関連モデル


@dataclass
class RiskMetrics:
    """リスクメトリクス"""

    risk_level: RiskLevel
    confidence_score: float  # 0.0 - 1.0
    value_at_risk: Optional[Decimal] = None
    expected_shortfall: Optional[Decimal] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")


@dataclass
class RiskFactor:
    """リスク要因"""

    factor_id: str
    name: str
    description: str
    category: str
    impact_level: RiskLevel
    probability: float  # 0.0 - 1.0
    impact_score: float  # 0.0 - 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0.0 and 1.0")
        if not 0.0 <= self.impact_score <= 1.0:
            raise ValueError("impact_score must be between 0.0 and 1.0")

    @property
    def risk_score(self) -> float:
        """リスクスコア計算"""
        return self.probability * self.impact_score


@dataclass
class RiskAlert:
    """リスクアラート"""

    alert_id: str
    risk_level: RiskLevel
    title: str
    message: str
    asset: Optional[Asset] = None
    position: Optional[Position] = None
    portfolio: Optional[Portfolio] = None
    risk_factors: List[RiskFactor] = field(default_factory=list)
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """分析結果"""

    analysis_id: str
    analyzer_name: str
    status: AnalysisStatus
    risk_metrics: Optional[RiskMetrics] = None
    risk_factors: List[RiskFactor] = field(default_factory=list)
    alerts: List[RiskAlert] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def processing_time(self) -> Optional[float]:
        """処理時間（秒）"""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


# バッチ処理関連モデル


@dataclass
class BatchJob:
    """バッチジョブ"""

    job_id: str
    job_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: AnalysisStatus = AnalysisStatus.PENDING
    progress: float = 0.0  # 0.0 - 1.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingStats:
    """処理統計"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0
    peak_memory_usage: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# バリデーション関連


class ValidationResult:
    """バリデーション結果"""

    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, message: str):
        """エラー追加"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """警告追加"""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult"):
        """他の結果とマージ"""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


# データ変換ヘルパー


def asset_from_dict(data: Dict[str, Any]) -> Asset:
    """辞書からAssetオブジェクト作成"""
    return Asset(
        symbol=data["symbol"],
        asset_type=AssetType(data["asset_type"]),
        exchange=data.get("exchange"),
        currency=data.get("currency", "USD"),
        name=data.get("name"),
        sector=data.get("sector"),
        country=data.get("country"),
        metadata=data.get("metadata", {}),
    )


def position_from_dict(data: Dict[str, Any]) -> Position:
    """辞書からPositionオブジェクト作成"""
    return Position(
        position_id=data["position_id"],
        asset=asset_from_dict(data["asset"]),
        position_type=PositionType(data["position_type"]),
        quantity=Decimal(str(data["quantity"])),
        entry_price=Decimal(str(data["entry_price"])),
        current_price=Decimal(str(data["current_price"]))
        if data.get("current_price")
        else None,
        timestamp=datetime.fromisoformat(
            data.get("timestamp", datetime.now().isoformat())
        ),
        metadata=data.get("metadata", {}),
    )


def portfolio_from_dict(data: Dict[str, Any]) -> Portfolio:
    """辞書からPortfolioオブジェクト作成"""
    positions = [position_from_dict(pos_data) for pos_data in data.get("positions", [])]

    return Portfolio(
        portfolio_id=data["portfolio_id"],
        name=data["name"],
        positions=positions,
        cash=Decimal(str(data.get("cash", "0"))),
        currency=data.get("currency", "USD"),
        created_at=datetime.fromisoformat(
            data.get("created_at", datetime.now().isoformat())
        ),
        updated_at=datetime.fromisoformat(
            data.get("updated_at", datetime.now().isoformat())
        ),
        metadata=data.get("metadata", {}),
    )


# バリデーター関数


def validate_asset(asset: Asset) -> ValidationResult:
    """Asset検証"""
    result = ValidationResult()

    if not asset.symbol or len(asset.symbol.strip()) == 0:
        result.add_error("Asset symbol is required")

    if len(asset.symbol) > 20:
        result.add_error("Asset symbol too long (max 20 characters)")

    if asset.currency and len(asset.currency) != 3:
        result.add_warning("Currency should be 3-letter ISO code")

    return result


def validate_position(position: Position) -> ValidationResult:
    """Position検証"""
    result = ValidationResult()

    # Asset検証
    asset_result = validate_asset(position.asset)
    result.merge(asset_result)

    # 数量検証
    if position.quantity <= 0:
        result.add_error("Position quantity must be positive")

    # 価格検証
    if position.entry_price <= 0:
        result.add_error("Entry price must be positive")

    if position.current_price is not None and position.current_price <= 0:
        result.add_error("Current price must be positive")

    return result


def validate_portfolio(portfolio: Portfolio) -> ValidationResult:
    """Portfolio検証"""
    result = ValidationResult()

    if not portfolio.name or len(portfolio.name.strip()) == 0:
        result.add_error("Portfolio name is required")

    # ポジション検証
    for position in portfolio.positions:
        pos_result = validate_position(position)
        result.merge(pos_result)

    # 現金残高検証
    if portfolio.cash < 0:
        result.add_warning("Negative cash balance detected")

    return result


def validate_risk_metrics(metrics: RiskMetrics) -> ValidationResult:
    """RiskMetrics検証"""
    result = ValidationResult()

    if not 0.0 <= metrics.confidence_score <= 1.0:
        result.add_error("Confidence score must be between 0.0 and 1.0")

    if metrics.volatility is not None and metrics.volatility < 0:
        result.add_error("Volatility cannot be negative")

    if metrics.value_at_risk is not None and metrics.value_at_risk > 0:
        result.add_warning("VaR should typically be negative (indicating loss)")

    return result


# モデル変換ユーティリティ


class ModelConverter:
    """モデル変換ユーティリティ"""

    @staticmethod
    def to_dict(obj: Any) -> Dict[str, Any]:
        """オブジェクトを辞書に変換"""
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for field_name, field_info in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)

                if isinstance(value, Enum):
                    result[field_name] = value.value
                elif isinstance(value, datetime):
                    result[field_name] = value.isoformat()
                elif isinstance(value, Decimal):
                    result[field_name] = str(value)
                elif isinstance(value, list):
                    result[field_name] = [
                        ModelConverter.to_dict(item) for item in value
                    ]
                elif hasattr(value, "__dataclass_fields__"):
                    result[field_name] = ModelConverter.to_dict(value)
                else:
                    result[field_name] = value

            return result
        else:
            return obj

    @staticmethod
    def from_dict(data: Dict[str, Any], target_class: type):
        """辞書からオブジェクトを作成"""
        if target_class == Asset:
            return asset_from_dict(data)
        elif target_class == Position:
            return position_from_dict(data)
        elif target_class == Portfolio:
            return portfolio_from_dict(data)
        else:
            # 汎用的な変換（型アノテーション使用）
            return target_class(**data)
