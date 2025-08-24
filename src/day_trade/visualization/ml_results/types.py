#!/usr/bin/env python3
"""
機械学習結果可視化システム - 基本型・設定・定数

Issue #315: 高度テクニカル指標・ML機能拡張
LSTM・GARCH・マルチタイムフレーム分析結果の包括的可視化システムの型定義
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """リスクレベル列挙型"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class TradingAction(Enum):
    """トレーディングアクション列挙型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CAUTION = "CAUTION"
    NORMAL = "NORMAL"


class SignalStrength(Enum):
    """シグナル強度列挙型"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class PositionSize(Enum):
    """ポジションサイズ列挙型"""
    SMALL = "SMALL"
    NEUTRAL = "NEUTRAL"
    MODERATE = "MODERATE"
    LARGE = "LARGE"


class TrendDirection(Enum):
    """トレンド方向列挙型"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


@dataclass
class ColorPalette:
    """カラーパレット設定"""
    price: str = "#1f77b4"
    prediction: str = "#ff7f0e"
    lstm: str = "#2ca02c"
    garch: str = "#d62728"
    vix: str = "#9467bd"
    support: str = "#8c564b"
    resistance: str = "#e377c2"
    bullish: str = "#2ca02c"
    bearish: str = "#d62728"
    neutral: str = "#7f7f7f"
    volatility: str = "#ff9999"
    confidence: str = "#66b3ff"


@dataclass
class LSTMResults:
    """LSTM予測結果データクラス"""
    predicted_prices: List[float]
    predicted_returns: List[float]
    prediction_dates: List[str]
    confidence_score: float
    model_metrics: Optional[Dict] = None


@dataclass
class CurrentMetrics:
    """現在の市場メトリクス"""
    realized_volatility: float
    vix_like_indicator: float
    volatility_regime: str


@dataclass
class EnsembleForecast:
    """アンサンブル予測結果"""
    ensemble_volatility: float
    ensemble_confidence: float
    individual_forecasts: Dict[str, float]


@dataclass
class RiskAssessment:
    """リスク評価結果"""
    risk_level: RiskLevel
    risk_score: int
    risk_factors: List[str]


@dataclass
class VolatilityResults:
    """ボラティリティ分析結果データクラス"""
    current_metrics: CurrentMetrics
    ensemble_forecast: EnsembleForecast
    risk_assessment: RiskAssessment
    investment_implications: Dict[str, List[str]]


@dataclass
class TechnicalIndicators:
    """テクニカル指標データクラス"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bb_position: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None


@dataclass
class TimeframeData:
    """時間軸別データ"""
    timeframe: str
    trend_direction: TrendDirection
    trend_strength: float
    technical_indicators: TechnicalIndicators
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


@dataclass
class IntegratedSignal:
    """統合シグナルデータ"""
    action: TradingAction
    strength: SignalStrength
    signal_score: float


@dataclass
class InvestmentRecommendation:
    """投資推奨データ"""
    recommendation: TradingAction
    position_size: PositionSize
    confidence: float
    reasons: List[str]
    stop_loss_suggestion: Optional[float] = None
    take_profit_suggestion: Optional[float] = None


@dataclass
class IntegratedAnalysis:
    """統合分析結果"""
    overall_trend: TrendDirection
    trend_confidence: float
    consistency_score: float
    integrated_signal: IntegratedSignal
    investment_recommendation: InvestmentRecommendation
    risk_assessment: Optional[RiskAssessment] = None


@dataclass
class MultiFrameResults:
    """マルチタイムフレーム分析結果データクラス"""
    timeframes: Dict[str, TimeframeData]
    integrated_analysis: IntegratedAnalysis


@dataclass
class VisualizationConfig:
    """可視化設定"""
    output_dir: Path
    dpi: int = 300
    figsize_small: Tuple[int, int] = (15, 12)
    figsize_large: Tuple[int, int] = (24, 16)
    color_palette: ColorPalette = field(default_factory=ColorPalette)
    show_confidence_bands: bool = True
    include_timestamps: bool = True
    
    def __post_init__(self):
        """初期化後処理"""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ChartType(Enum):
    """チャート種別列挙型"""
    LSTM_PREDICTION = "lstm_prediction"
    VOLATILITY_FORECAST = "volatility_forecast"
    MULTIFRAME_ANALYSIS = "multiframe_analysis"
    COMPREHENSIVE_DASHBOARD = "comprehensive_dashboard"
    INTERACTIVE_DASHBOARD = "interactive_dashboard"


@dataclass
class ChartMetadata:
    """チャート メタデータ"""
    chart_type: ChartType
    symbol: str
    creation_time: datetime
    file_path: Optional[str] = None
    file_size: Optional[int] = None


# 依存関係チェック結果
@dataclass
class DependencyStatus:
    """依存パッケージ状況"""
    matplotlib_available: bool = False
    plotly_available: bool = False
    seaborn_available: bool = False
    
    @property
    def can_create_static_charts(self) -> bool:
        """静的チャート作成可能かチェック"""
        return self.matplotlib_available
    
    @property
    def can_create_interactive_charts(self) -> bool:
        """インタラクティブチャート作成可能かチェック"""
        return self.plotly_available
    
    @property
    def all_available(self) -> bool:
        """全依存関係利用可能かチェック"""
        return all([
            self.matplotlib_available,
            self.plotly_available,
            self.seaborn_available
        ])


# デフォルト設定
DEFAULT_CONFIG = VisualizationConfig(
    output_dir="output/ml_visualizations",
    dpi=300,
    figsize_small=(15, 12),
    figsize_large=(24, 16)
)

# エラーメッセージ定数
ERROR_MESSAGES = {
    'MATPLOTLIB_UNAVAILABLE': 'matplotlib/seabornが利用できません',
    'PLOTLY_UNAVAILABLE': 'plotlyが利用できません',
    'DATA_INSUFFICIENT': 'データが不十分です',
    'CHART_CREATION_FAILED': 'チャート作成に失敗しました',
    'FILE_SAVE_FAILED': 'ファイル保存に失敗しました',
    'INVALID_RESULTS': '分析結果が無効です'
}

# 日本語ラベル定義
JAPANESE_LABELS = {
    'price': '価格',
    'prediction': '予測',
    'lstm_prediction': 'LSTM予測',
    'volatility': 'ボラティリティ',
    'risk_level': 'リスクレベル',
    'trend_strength': 'トレンド強度',
    'confidence': '信頼度',
    'signal': 'シグナル',
    'recommendation': '推奨',
    'timeframe': '時間軸',
    'technical_indicators': 'テクニカル指標',
    'risk_assessment': 'リスク評価',
    'investment_implications': '投資への示唆'
}

# 時間軸定義
TIMEFRAMES = {
    'daily': '日足',
    'weekly': '週足',
    'monthly': '月足',
    'hourly': '時間足',
    'minute': '分足'
}

# チャート設定
CHART_SETTINGS = {
    'font_family': 'DejaVu Sans',
    'grid_alpha': 0.3,
    'line_width': 2,
    'marker_size': 4,
    'confidence_alpha': 0.2,
    'bar_alpha': 0.7
}


def validate_lstm_results(results: Dict) -> bool:
    """LSTM結果の妥当性チェック
    
    Args:
        results: LSTM分析結果辞書
        
    Returns:
        妥当性チェック結果
    """
    required_keys = ['predicted_prices', 'confidence_score']
    return all(key in results for key in required_keys)


def validate_volatility_results(results: Dict) -> bool:
    """ボラティリティ結果の妥当性チェック
    
    Args:
        results: ボラティリティ分析結果辞書
        
    Returns:
        妥当性チェック結果
    """
    return 'current_metrics' in results or 'ensemble_forecast' in results


def validate_multiframe_results(results: Dict) -> bool:
    """マルチタイムフレーム結果の妥当性チェック
    
    Args:
        results: マルチタイムフレーム分析結果辞書
        
    Returns:
        妥当性チェック結果
    """
    return 'timeframes' in results and isinstance(results['timeframes'], dict)


def get_risk_color(risk_level: Union[str, RiskLevel]) -> str:
    """リスクレベルに応じた色を取得
    
    Args:
        risk_level: リスクレベル
        
    Returns:
        対応する色コード
    """
    if isinstance(risk_level, str):
        risk_level = RiskLevel(risk_level)
    
    color_map = {
        RiskLevel.LOW: "#2ca02c",
        RiskLevel.MEDIUM: "#7f7f7f",
        RiskLevel.HIGH: "#d62728",
        RiskLevel.UNKNOWN: "#666666"
    }
    
    return color_map.get(risk_level, "#666666")


def get_signal_color(action: Union[str, TradingAction]) -> str:
    """シグナルアクションに応じた色を取得
    
    Args:
        action: トレーディングアクション
        
    Returns:
        対応する色コード
    """
    if isinstance(action, str):
        action = TradingAction(action)
    
    color_map = {
        TradingAction.BUY: "#2ca02c",
        TradingAction.SELL: "#d62728",
        TradingAction.HOLD: "#7f7f7f",
        TradingAction.CAUTION: "#ff7f0e",
        TradingAction.NORMAL: "#1f77b4"
    }
    
    return color_map.get(action, "#7f7f7f")


# バージョン情報
__version__ = "1.0.0"
__author__ = "ML Results Visualization System"
__description__ = "機械学習結果可視化システムの型定義・設定モジュール"