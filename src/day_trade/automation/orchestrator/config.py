"""
Orchestra配置・データクラス定義モジュール
Next-Gen AI Trading Engine の設定とデータ構造を管理
"""

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# CI/CI環境チェック
CI_MODE = os.getenv("CI", "false").lower() == "true"


@dataclass
class AIAnalysisResult:
    """
    AI分析結果データクラス
    
    Attributes:
        symbol: 銘柄コード
        timestamp: 分析実行時刻
        predictions: 予測結果辞書
        confidence_scores: 信頼度スコア辞書  
        technical_signals: テクニカル分析シグナル
        ml_features: 機械学習特徴量
        performance_metrics: パフォーマンス指標
        data_quality: データ品質スコア(0-100)
        recommendation: 推奨アクション
        risk_assessment: リスク評価結果
    """
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
    """
    実行レポート（拡張版）
    
    Attributes:
        start_time: 実行開始時刻
        end_time: 実行終了時刻
        total_symbols: 対象銘柄総数
        successful_symbols: 成功した銘柄数
        failed_symbols: 失敗した銘柄数
        generated_signals: 生成されたシグナルリスト
        triggered_alerts: 発動したアラートリスト
        ai_analysis_results: AI分析結果リスト
        portfolio_summary: ポートフォリオ概要
        performance_stats: パフォーマンス統計
        system_health: システムヘルス情報
        errors: エラーメッセージリスト
    """
    start_time: datetime
    end_time: datetime
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    generated_signals: List[Dict[str, Any]]
    triggered_alerts: List[Dict[str, Any]]
    ai_analysis_results: List[AIAnalysisResult] = None
    portfolio_summary: Optional[Dict[str, Any]] = None
    performance_stats: Dict[str, Any] = None
    system_health: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        """初期化後処理"""
        if self.errors is None:
            self.errors = []
        if self.ai_analysis_results is None:
            self.ai_analysis_results = []


@dataclass
class OrchestrationConfig:
    """
    オーケストレーション設定
    
    Attributes:
        max_workers: 最大ワーカー数
        max_thread_workers: I/Oバウンドタスク用最大スレッド数
        max_process_workers: CPUバウンドタスク用最大プロセス数
        enable_ml_engine: ML エンジン有効化
        enable_advanced_batch: 高度バッチ処理有効化
        enable_performance_monitoring: パフォーマンス監視有効化
        enable_fault_tolerance: フォールトトレラント有効化
        enable_parallel_optimization: 並列処理最適化有効化
        prediction_horizon: 予測期間（日）
        confidence_threshold: 信頼度閾値
        data_quality_threshold: データ品質閾値
        timeout_seconds: タイムアウト時間（秒）
        cache_enabled: キャッシュ有効化
        retry_attempts: リトライ回数
    """
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
        """設定を辞書形式で取得"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OrchestrationConfig':
        """辞書から設定オブジェクトを作成"""
        return cls(**config_dict)

    def update_for_ci_mode(self) -> None:
        """CI環境向けの軽量化設定"""
        if CI_MODE:
            self.enable_ml_engine = False
            self.enable_advanced_batch = False
            self.max_workers = min(self.max_workers, 4)
            self.max_thread_workers = min(self.max_thread_workers, 6)
            self.max_process_workers = min(self.max_process_workers, 2)
            self.timeout_seconds = min(self.timeout_seconds, 60)


def get_default_config() -> OrchestrationConfig:
    """
    デフォルト設定を取得
    
    Returns:
        OrchestrationConfig: デフォルト設定オブジェクト
    """
    config = OrchestrationConfig()
    config.update_for_ci_mode()
    return config


def validate_config(config: OrchestrationConfig) -> List[str]:
    """
    設定値の妥当性チェック
    
    Args:
        config: チェック対象の設定オブジェクト
        
    Returns:
        List[str]: 検証エラーメッセージのリスト
    """
    errors = []
    
    # ワーカー数の妥当性チェック
    if config.max_workers <= 0:
        errors.append("max_workers は正の整数である必要があります")
    
    if config.max_thread_workers <= 0:
        errors.append("max_thread_workers は正の整数である必要があります")
        
    if config.max_process_workers <= 0:
        errors.append("max_process_workers は正の整数である必要があります")
    
    # 閾値の妥当性チェック
    if not (0.0 <= config.confidence_threshold <= 1.0):
        errors.append("confidence_threshold は 0.0 から 1.0 の範囲である必要があります")
        
    if not (0.0 <= config.data_quality_threshold <= 100.0):
        errors.append("data_quality_threshold は 0.0 から 100.0 の範囲である必要があります")
    
    # 時間・回数の妥当性チェック
    if config.prediction_horizon <= 0:
        errors.append("prediction_horizon は正の整数である必要があります")
        
    if config.timeout_seconds <= 0:
        errors.append("timeout_seconds は正の整数である必要があります")
        
    if config.retry_attempts < 0:
        errors.append("retry_attempts は 0 以上の整数である必要があります")
    
    return errors