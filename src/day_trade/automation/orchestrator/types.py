"""
データ型・設定クラス定義
次世代AIオーケストレーター用のデータクラスと設定管理
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AIAnalysisResult:
    """AI分析結果"""
    
    symbol: str
    timestamp: datetime
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    technical_signals: Dict[str, Any]
    ml_features: Dict[str, Any]
    performance_metrics: Dict[str, float]
    data_quality: float
    recommendation: str
    risk_assessment: Dict[str, Any]


@dataclass
class ExecutionReport:
    """実行レポート（拡張版）"""
    
    start_time: datetime
    end_time: datetime
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    generated_signals: List[Dict[str, Any]]
    triggered_alerts: List[Dict[str, Any]]
    ai_analysis_results: List[AIAnalysisResult] = field(default_factory=list)
    portfolio_summary: Optional[Dict[str, Any]] = None
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    system_health: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初期化後処理"""
        if self.errors is None:
            self.errors = []
        if self.ai_analysis_results is None:
            self.ai_analysis_results = []


@dataclass
class OrchestrationConfig:
    """オーケストレーション設定"""
    
    max_workers: int = 8
    max_thread_workers: int = 12  # I/Oバウンド用
    max_process_workers: int = 4  # CPUバウンド用
    enable_ml_engine: bool = True
    enable_advanced_batch: bool = True
    enable_performance_monitoring: bool = True
    enable_fault_tolerance: bool = True
    enable_parallel_optimization: bool = True  # 新並列システム
    prediction_horizon: int = 5  # 予測期間（日）
    confidence_threshold: float = 0.7
    data_quality_threshold: float = 80.0
    timeout_seconds: int = 300
    cache_enabled: bool = True
    retry_attempts: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で返す"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OrchestrationConfig':
        """辞書から設定オブジェクトを作成"""
        return cls(**config_dict)