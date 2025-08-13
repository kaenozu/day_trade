#!/usr/bin/env python3
"""
Issue #473対応: AdvancedMLEngine インターフェース定義

AdvancedMLEngineの役割とインターフェースを明確化し、
アンサンブルシステム内での責任分担を明確にする
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from enum import Enum

from ..data.advanced_ml_engine import PredictionResult


class AdvancedModelType(Enum):
    """高度モデルの種類"""
    LSTM_TRANSFORMER = "lstm_transformer"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    ATTENTION_LSTM = "attention_lstm"
    CUSTOM_HYBRID = "custom_hybrid"


@dataclass
class AdvancedModelCapabilities:
    """高度モデルの能力"""
    supports_sequence_prediction: bool = True
    supports_multivariate_input: bool = True
    supports_uncertainty_quantification: bool = True
    supports_attention_weights: bool = False
    supports_transfer_learning: bool = False
    supports_online_learning: bool = False
    min_sequence_length: int = 10
    max_sequence_length: int = 1000
    preferred_feature_count: int = 50
    inference_time_target_ms: float = 100.0


@dataclass
class AdvancedModelMetrics:
    """高度モデルの性能指標"""
    accuracy: float = 0.0
    mse: float = float('inf')
    mae: float = float('inf')
    r2_score: float = -float('inf')
    sharpe_ratio: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    training_time_minutes: float = 0.0
    model_complexity_score: float = 0.0


class AdvancedMLEngineInterface(ABC):
    """
    Issue #473対応: AdvancedMLEngine インターフェース
    
    高度なML手法（LSTM-Transformer等）を提供するエンジンの
    統一インターフェース定義
    
    責任範囲:
    1. 時系列データの深層学習ベース予測
    2. 不確実性定量化付き予測
    3. アテンション機構による解釈可能性
    4. 高速推論最適化
    5. モデル状態管理
    """

    @abstractmethod
    def get_model_type(self) -> AdvancedModelType:
        """モデルタイプの取得"""
        pass

    @abstractmethod
    def get_capabilities(self) -> AdvancedModelCapabilities:
        """モデル能力の取得"""
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        """訓練済み状態の確認"""
        pass

    @abstractmethod
    def prepare_data(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """データ前処理"""
        pass

    @abstractmethod
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> AdvancedModelMetrics:
        """モデル訓練"""
        pass

    @abstractmethod
    def predict(self, 
                X: np.ndarray, 
                return_confidence: bool = True,
                return_attention: bool = False) -> PredictionResult:
        """予測実行"""
        pass

    @abstractmethod
    def get_model_metrics(self) -> AdvancedModelMetrics:
        """モデル性能指標の取得"""
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """モデル保存"""
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """モデル読み込み"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """特徴量重要度取得（可能な場合）"""
        pass

    @abstractmethod
    def validate_input_shape(self, X: np.ndarray) -> bool:
        """入力形状の検証"""
        pass

    @abstractmethod
    def optimize_for_inference(self) -> bool:
        """推論最適化"""
        pass

    @abstractmethod
    def get_memory_usage(self) -> float:
        """メモリ使用量取得（MB）"""
        pass


class LSTMTransformerEngine(AdvancedMLEngineInterface):
    """
    Issue #473対応: LSTM-Transformer 専用エンジンクラス
    
    AdvancedMLEngineを LSTM-Transformer に特化した実装として
    役割を明確化する
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Args:
            config: ModelConfig または互換設定
        """
        # 遅延インポートで AdvancedMLEngine をラップ
        from ..data.advanced_ml_engine import AdvancedMLEngine, ModelConfig
        
        self._engine = AdvancedMLEngine(config)
        self._capabilities = AdvancedModelCapabilities(
            supports_sequence_prediction=True,
            supports_multivariate_input=True,
            supports_uncertainty_quantification=True,
            supports_attention_weights=True,
            supports_transfer_learning=False,
            supports_online_learning=False,
            min_sequence_length=10,
            max_sequence_length=500,
            preferred_feature_count=50,
            inference_time_target_ms=100.0
        )

    def get_model_type(self) -> AdvancedModelType:
        return AdvancedModelType.LSTM_TRANSFORMER

    def get_capabilities(self) -> AdvancedModelCapabilities:
        return self._capabilities

    def is_trained(self) -> bool:
        return self._engine.model is not None

    def prepare_data(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """データ前処理の委譲"""
        return self._engine.prepare_data(data, **kwargs)

    def train(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> AdvancedModelMetrics:
        """訓練の実行と性能指標の統一"""
        # 実際の訓練を実行
        result = self._engine.train(X, y, validation_data, **kwargs)
        
        # メトリクスを統一形式に変換
        return self._convert_metrics(result)

    def predict(self, 
                X: np.ndarray, 
                return_confidence: bool = True,
                return_attention: bool = False) -> PredictionResult:
        """予測実行"""
        return self._engine.predict(X, return_confidence)

    def get_model_metrics(self) -> AdvancedModelMetrics:
        """現在のモデル性能指標"""
        if not self.is_trained():
            return AdvancedModelMetrics()
        
        # エンジンからメトリクスを取得して変換
        engine_metrics = getattr(self._engine, 'model_metadata', {}).get('performance', {})
        return self._convert_dict_to_metrics(engine_metrics)

    def save_model(self, filepath: str) -> bool:
        """モデル保存"""
        try:
            self._engine.save_model(filepath)
            return True
        except Exception:
            return False

    def load_model(self, filepath: str) -> bool:
        """モデル読み込み"""
        try:
            self._engine.load_model(filepath)
            return True
        except Exception:
            return False

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """特徴量重要度（LSTM-Transformerでは限定的）"""
        # LSTM-Transformerでは直接的な特徴量重要度は困難
        # アテンション重みベースの重要度を将来実装予定
        return None

    def validate_input_shape(self, X: np.ndarray) -> bool:
        """入力形状検証"""
        if X.ndim != 3:
            return False
        
        seq_len = X.shape[1]
        return (self._capabilities.min_sequence_length <= seq_len <= 
                self._capabilities.max_sequence_length)

    def optimize_for_inference(self) -> bool:
        """推論最適化"""
        # 現在の実装では基本的な最適化のみ
        if hasattr(self._engine.model, 'eval'):
            self._engine.model.eval()
            return True
        return False

    def get_memory_usage(self) -> float:
        """メモリ使用量推定"""
        if not self.is_trained():
            return 0.0
        
        # 概算計算（実装依存）
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # psutil がない場合は概算
            return 100.0  # デフォルト値

    def _convert_metrics(self, engine_result: Any) -> AdvancedModelMetrics:
        """エンジン結果をメトリクスに変換"""
        if isinstance(engine_result, dict):
            return self._convert_dict_to_metrics(engine_result)
        else:
            # デフォルトメトリクス
            return AdvancedModelMetrics()

    def _convert_dict_to_metrics(self, metrics_dict: Dict[str, Any]) -> AdvancedModelMetrics:
        """辞書形式メトリクスを統一形式に変換"""
        return AdvancedModelMetrics(
            accuracy=metrics_dict.get('accuracy', 0.0),
            mse=metrics_dict.get('mse', float('inf')),
            mae=metrics_dict.get('mae', float('inf')),
            r2_score=metrics_dict.get('r2_score', -float('inf')),
            sharpe_ratio=metrics_dict.get('sharpe_ratio', 0.0),
            inference_time_ms=metrics_dict.get('inference_time', 0.0) * 1000,
            memory_usage_mb=metrics_dict.get('memory_usage_mb', 0.0),
            training_time_minutes=metrics_dict.get('training_time', 0.0) / 60,
            model_complexity_score=metrics_dict.get('complexity_score', 0.0)
        )


def create_advanced_ml_engine(model_type: AdvancedModelType = AdvancedModelType.LSTM_TRANSFORMER,
                            config: Optional[Any] = None) -> AdvancedMLEngineInterface:
    """
    Issue #473対応: Advanced MLエンジンファクトリー
    
    Args:
        model_type: 作成するモデルタイプ
        config: モデル設定
        
    Returns:
        統一インターフェースを実装したエンジン
    """
    if model_type == AdvancedModelType.LSTM_TRANSFORMER:
        return LSTMTransformerEngine(config)
    else:
        raise NotImplementedError(f"モデルタイプ {model_type} は未実装")


def get_available_model_types() -> List[AdvancedModelType]:
    """利用可能なモデルタイプの取得"""
    return [AdvancedModelType.LSTM_TRANSFORMER]


def compare_model_capabilities(models: List[AdvancedMLEngineInterface]) -> Dict[str, Any]:
    """複数モデルの能力比較"""
    comparison = {
        'models': [],
        'capabilities_matrix': {},
        'recommended_use_cases': {}
    }
    
    for model in models:
        model_type = model.get_model_type()
        capabilities = model.get_capabilities()
        
        comparison['models'].append(model_type.value)
        comparison['capabilities_matrix'][model_type.value] = {
            'sequence_prediction': capabilities.supports_sequence_prediction,
            'multivariate_input': capabilities.supports_multivariate_input,
            'uncertainty_quantification': capabilities.supports_uncertainty_quantification,
            'attention_weights': capabilities.supports_attention_weights,
            'transfer_learning': capabilities.supports_transfer_learning,
            'online_learning': capabilities.supports_online_learning,
            'min_seq_len': capabilities.min_sequence_length,
            'max_seq_len': capabilities.max_sequence_length,
            'inference_target_ms': capabilities.inference_time_target_ms
        }
        
        # 推奨用途の判定
        use_cases = []
        if capabilities.supports_sequence_prediction and capabilities.supports_uncertainty_quantification:
            use_cases.append("高精度時系列予測")
        if capabilities.supports_attention_weights:
            use_cases.append("解釈可能予測")
        if capabilities.inference_time_target_ms <= 100:
            use_cases.append("リアルタイム予測")
            
        comparison['recommended_use_cases'][model_type.value] = use_cases
    
    return comparison