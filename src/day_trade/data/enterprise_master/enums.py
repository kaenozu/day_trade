#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - 列挙型定義

このモジュールは、MDMシステムで使用される各種列挙型を定義します。
"""

from enum import Enum


class MasterDataType(Enum):
    """マスターデータ種別
    
    金融システムで扱う主要なマスターデータタイプを定義します。
    """
    
    FINANCIAL_INSTRUMENTS = "financial_instruments"
    MARKET_SEGMENTS = "market_segments"
    CURRENCY_CODES = "currency_codes"
    INDUSTRY_CODES = "industry_codes"
    EXCHANGE_CODES = "exchange_codes"
    COUNTRY_CODES = "country_codes"
    TRADING_HOLIDAYS = "trading_holidays"
    REFERENCE_RATES = "reference_rates"
    RISK_RATINGS = "risk_ratings"
    REGULATORY_DATA = "regulatory_data"


class DataGovernanceLevel(Enum):
    """データガバナンスレベル
    
    データの重要度に応じたガバナンス管理レベルを定義します。
    """
    
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    REGULATORY = "regulatory"


class ChangeType(Enum):
    """変更種別
    
    マスターデータに対する変更操作の種類を定義します。
    """
    
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    SPLIT = "split"
    ARCHIVE = "archive"


class ApprovalStatus(Enum):
    """承認状態
    
    データ変更リクエストの承認状態を定義します。
    """
    
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class QualityScoreLevel(Enum):
    """品質スコアレベル
    
    データ品質スコアの分類レベルを定義します。
    """
    
    POOR = "poor"          # 0-50
    FAIR = "fair"          # 51-70
    GOOD = "good"          # 71-85
    EXCELLENT = "excellent"  # 86-100


class RiskLevel(Enum):
    """リスクレベル
    
    データ変更に伴うリスクレベルを定義します。
    """
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EntityStatus(Enum):
    """エンティティ状態
    
    マスターデータエンティティの状態を定義します。
    """
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ValidationResult(Enum):
    """バリデーション結果
    
    データバリデーションの結果を定義します。
    """
    
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"


class HierarchyType(Enum):
    """階層タイプ
    
    データ階層の種別を定義します。
    """
    
    ORGANIZATIONAL = "organizational"
    FUNCTIONAL = "functional"
    GEOGRAPHICAL = "geographical"
    TEMPORAL = "temporal"
    CLASSIFICATION = "classification"


class AuditAction(Enum):
    """監査アクション
    
    監査ログで記録する操作の種類を定義します。
    """
    
    VIEW = "view"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    APPROVE = "approve"
    REJECT = "reject"
    EXPORT = "export"
    IMPORT = "import"


# 品質スコアのしきい値定数
QUALITY_THRESHOLDS = {
    QualityScoreLevel.POOR: (0, 50),
    QualityScoreLevel.FAIR: (51, 70),
    QualityScoreLevel.GOOD: (71, 85),
    QualityScoreLevel.EXCELLENT: (86, 100)
}


def get_quality_level(score: float) -> QualityScoreLevel:
    """品質スコアからレベルを取得
    
    Args:
        score: 品質スコア（0-100）
        
    Returns:
        QualityScoreLevel: 対応する品質レベル
    """
    if score <= 50:
        return QualityScoreLevel.POOR
    elif score <= 70:
        return QualityScoreLevel.FAIR
    elif score <= 85:
        return QualityScoreLevel.GOOD
    else:
        return QualityScoreLevel.EXCELLENT


def get_risk_level_from_score(score: float) -> RiskLevel:
    """品質スコアからリスクレベルを推定
    
    Args:
        score: 品質スコア（0-100）
        
    Returns:
        RiskLevel: 推定されるリスクレベル
    """
    if score >= 90:
        return RiskLevel.LOW
    elif score >= 75:
        return RiskLevel.MEDIUM
    elif score >= 50:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL