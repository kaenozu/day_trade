"""
ML Prediction システム例外クラス

ml_prediction_models_improved.py からのリファクタリング抽出
機械学習予測システムで使用するカスタム例外クラス群
"""


class MLPredictionError(Exception):
    """ML予測システムの基底例外"""
    pass


class DataPreparationError(MLPredictionError):
    """データ準備エラー"""
    pass


class ModelTrainingError(MLPredictionError):
    """モデル訓練エラー"""
    pass


class ModelMetadataError(MLPredictionError):
    """モデルメタデータエラー"""
    pass


class PredictionError(MLPredictionError):
    """予測実行エラー"""
    pass