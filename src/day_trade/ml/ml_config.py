"""
ML Prediction システム設定とパラメータ管理

ml_prediction_models_improved.py からのリファクタリング抽出
機械学習予測システムの設定クラスと列挙型定義
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelType(Enum):
    """モデルタイプ（拡張版）"""
    RANDOM_FOREST = "Random Forest"
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    ENSEMBLE = "Ensemble"


class PredictionTask(Enum):
    """予測タスク（拡張版）"""
    PRICE_DIRECTION = "価格方向予測"
    PRICE_REGRESSION = "価格回帰予測"
    VOLATILITY = "ボラティリティ予測"
    TREND_STRENGTH = "トレンド強度予測"


class DataQuality(Enum):
    """データ品質レベル"""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    INSUFFICIENT = 1
    
    def __lt__(self, other):
        if isinstance(other, DataQuality):
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, DataQuality):
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, DataQuality):
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, DataQuality):
            return self.value >= other.value
        return NotImplemented


@dataclass
class TrainingConfig:
    """訓練設定（強化版）"""
    # データ分割
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

    # クロスバリデーション
    cv_folds: int = 5
    enable_cross_validation: bool = True

    # 特徴量選択
    feature_selection: bool = False
    feature_selection_method: str = "SelectKBest"
    max_features: Optional[int] = None

    # モデル保存
    save_model: bool = True
    save_metadata: bool = True

    # ハイパーパラメータ最適化
    use_optimized_params: bool = True
    optimization_method: str = "grid_search"
    optimization_budget: int = 50

    # 前処理
    enable_scaling: bool = True
    handle_missing_values: bool = True
    outlier_detection: bool = False

    # 品質管理
    min_data_quality: DataQuality = DataQuality.FAIR
    performance_threshold: float = 0.6

    # その他
    n_jobs: int = -1
    verbose: bool = False