#!/usr/bin/env python3
"""
Base Model Interface for Ensemble Learning

統一されたインターフェースによりアンサンブル学習での
異なるモデルの管理を簡素化
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class ModelPrediction:
    """モデル予測結果"""
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_name: str = "unknown"
    processing_time: float = 0.0


@dataclass 
class ModelMetrics:
    """モデル評価指標"""
    mse: float
    rmse: float
    mae: float
    r2_score: float
    hit_rate: float  # 方向性予測精度
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    

class BaseModelInterface(ABC):
    """
    すべてのベースモデルが実装すべき統一インターフェース
    
    アンサンブル学習での一貫性を保つため、すべてのモデルが
    この基底クラスを継承し、共通のメソッドを実装する必要がある
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            model_name: モデル名
            config: モデル設定辞書
        """
        self.model_name = model_name
        self.config = config or {}
        self.is_trained = False
        self.model = None
        self.training_metrics = {}
        self.feature_names = []
        
        logger.info(f"{model_name}モデル初期化")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        モデル学習
        
        Args:
            X: 訓練データの特徴量
            y: 訓練データの目標変数
            validation_data: 検証データ (X_val, y_val)
            
        Returns:
            学習結果辞書（メトリクス、学習時間等）
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        予測実行
        
        Args:
            X: 予測対象の特徴量
            
        Returns:
            ModelPrediction: 予測結果とメタ情報
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度取得
        
        Returns:
            特徴量名と重要度のマッピング
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        モデル評価
        
        Args:
            X: 評価データの特徴量
            y: 評価データの目標変数
            
        Returns:
            ModelMetrics: 評価指標
        """
        try:
            prediction = self.predict(X)
            y_pred = prediction.predictions
            
            # 基本指標計算
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))
            
            # R²スコア
            y_mean = np.mean(y)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            # Hit Rate（方向性予測精度）
            y_diff = np.diff(y)  # 前日比
            pred_diff = np.diff(y_pred)  # 予測前日比
            
            if len(y_diff) > 0:
                # 上昇/下降の方向が一致する割合
                direction_match = np.sign(y_diff) == np.sign(pred_diff)
                hit_rate = np.mean(direction_match)
            else:
                hit_rate = 0.5  # デフォルト値
                
            return ModelMetrics(
                mse=mse,
                rmse=rmse, 
                mae=mae,
                r2_score=r2_score,
                hit_rate=hit_rate
            )
            
        except Exception as e:
            logger.error(f"{self.model_name}評価エラー: {e}")
            # エラー時はデフォルト値を返す
            return ModelMetrics(
                mse=float('inf'),
                rmse=float('inf'),
                mae=float('inf'),
                r2_score=-1.0,
                hit_rate=0.5
            )
    
    def save_model(self, filepath: str) -> bool:
        """
        モデル保存
        
        Args:
            filepath: 保存先パス
            
        Returns:
            保存成功フラグ
        """
        try:
            import pickle
            
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"{self.model_name}モデル保存完了: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"{self.model_name}モデル保存エラー: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        モデル読み込み
        
        Args:
            filepath: 読み込み元パス
            
        Returns:
            読み込み成功フラグ
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data.get('model')
            self.model_name = model_data.get('model_name', self.model_name)
            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_metrics = model_data.get('training_metrics', {})
            self.feature_names = model_data.get('feature_names', [])
            
            logger.info(f"{self.model_name}モデル読み込み完了: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"{self.model_name}モデル読み込みエラー: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報取得
        
        Returns:
            モデル情報辞書
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names[:10]  # 最初の10個のみ
        }
    
    def set_feature_names(self, feature_names: List[str]):
        """特徴量名設定"""
        self.feature_names = feature_names
        logger.debug(f"{self.model_name}特徴量名設定: {len(feature_names)}個")
        
    def __str__(self) -> str:
        """文字列表現"""
        status = "Trained" if self.is_trained else "Untrained"
        return f"{self.model_name} ({status})"
    
    def __repr__(self) -> str:
        """詳細文字列表現"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', trained={self.is_trained})"