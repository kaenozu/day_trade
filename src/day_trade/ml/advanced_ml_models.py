#!/usr/bin/env python3
"""
Advanced ML Models System
Issue #315 Phase 3: Advanced ML models実装

LSTM時系列予測、アンサンブル学習、高度特徴量エンジニアリングによる
投資判断精度向上システム
"""

import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import mlflow
    import mlflow.keras
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn(
        "MLflow未インストール - MLOps機能利用不可", ImportWarning, stacklevel=2
    )

try:

    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def __init__(self, **kwargs):
            pass

        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"advanced_ml_{hash(str(args) + str(kwargs))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 依存関係チェック
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta未インストール - 技術指標計算が制限されます")

try:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn未インストール - ML機能利用不可")

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow未インストール - LSTM機能利用不可")


@dataclass
class LSTMPredictionResult:
    """LSTM予測結果"""

    symbol: str
    predictions: np.ndarray
    confidence: float
    trend_direction: str  # 'up', 'down', 'sideways'
    volatility_forecast: float
    model_score: float
    feature_importance: Dict[str, float]
    processing_time: float


@dataclass
class EnsembleModelResult:
    """アンサンブルモデル結果"""

    symbol: str
    ensemble_prediction: float
    individual_predictions: Dict[str, float]
    weighted_confidence: float
    model_weights: Dict[str, float]
    consensus_strength: float
    outlier_detection: List[str]  # 外れ値となったモデル
    processing_time: float


@dataclass
class AdvancedFeatureSet:
    """高度特徴量セット"""

    symbol: str
    technical_features: Dict[str, float]
    price_patterns: Dict[str, float]
    volatility_features: Dict[str, float]
    momentum_features: Dict[str, float]
    volume_features: Dict[str, float]
    multiframe_features: Dict[str, float]
    feature_count: int
    processing_time: float


class AdvancedMLModels:
    """
    Advanced ML Models System

    LSTM時系列予測、アンサンブル学習、自動特徴量エンジニアリング
    統合最適化基盤（Issues #322-325）をフル活用
    """

    def __init__(
        self,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        enable_lstm: bool = True,
        lstm_sequence_length: int = 60,
        ensemble_models: int = 5,
        max_concurrent: int = 10,
        model_save_path: str = "models",
    ):
        """
        Advanced ML Models初期化

        Args:
            enable_cache: キャッシュ有効化
            enable_parallel: 並列処理有効化
            enable_lstm: LSTM機能有効化
            lstm_sequence_length: LSTM入力系列長
            ensemble_models: アンサンブルモデル数
            max_concurrent: 最大並行処理数
        """
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        self.enable_lstm = enable_lstm and TF_AVAILABLE
        self.lstm_sequence_length = lstm_sequence_length
        self.ensemble_models = ensemble_models
        self.max_concurrent = max_concurrent
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # 統合最適化基盤初期化
        if self.enable_cache:
            self.cache_manager = UnifiedCacheManager(
                l1_memory_mb=128, l2_memory_mb=512, l3_disk_mb=1024
            )
            logger.info("統合キャッシュシステム有効化（Issue #324統合）")

        if self.enable_parallel:
            self.parallel_engine = AdvancedParallelMLEngine(
                cpu_workers=max_concurrent, enable_monitoring=True
            )
            logger.info("高度並列処理システム有効化（Issue #323統合）")

        # モデル初期化
        self.trained_models = {}
        self.scalers = {}
        self.feature_selectors = {}

        # パフォーマンス統計
        self.stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "lstm_predictions": 0,
            "ensemble_predictions": 0,
            "avg_processing_time": 0.0,
            "accuracy_scores": [],
        }

        logger.info("Advanced ML Models System（統合最適化版）初期化完了")
        logger.info(f"  - 統合キャッシュ: {self.enable_cache}")
        logger.info(f"  - 並列処理: {self.enable_parallel}")
        logger.info(f"  - LSTM機能: {self.enable_lstm}")
        logger.info(f"  - LSTM系列長: {self.lstm_sequence_length}")
        logger.info(f"  - アンサンブル数: {self.ensemble_models}")
        logger.info(f"  - 最大並行数: {self.max_concurrent}")

    async def train_model(self, symbol: str):
        """指定された銘柄のMLモデルを訓練・保存"""
        logger.info(f"MLモデル訓練開始: {symbol}")
        start_time = time.time()

        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("DayTrade ML Model Training")
            with mlflow.start_run(run_name=f"Train Model for {symbol}") as run:
                mlflow.log_param("symbol", symbol)
                mlflow.log_param("lstm_enabled", self.enable_lstm)
                mlflow.log_param("ensemble_models", self.ensemble_models)

                # 訓練データをロード (DVC連携を想定)
                from src.day_trade.data.training_data_loader import (
                    load_training_data_for_symbol,
                )

                try:
                    data = load_training_data_for_symbol(symbol)
                    if (
                        data.empty or len(data) < self.lstm_sequence_length + 20
                    ):  # LSTMに必要な最低データ長
                        logger.warning(
                            f"訓練データが不足しているか無効です: {symbol}. 訓練をスキップします。"
                        )
                        mlflow.log_param("training_status", "skipped_data_insufficient")
                        return
                    mlflow.log_param("data_length", len(data))
                except NotImplementedError as e:
                    logger.error(f"訓練データロードエラー: {e}")
                    mlflow.log_param("training_status", "skipped_data_load_error")
                    return
                except Exception as e:
                    logger.error(f"訓練データロード中に予期せぬエラー: {e}")
                    mlflow.log_param("training_status", "skipped_data_load_error")
                    traceback.print_exc()
                    return

                # LSTMモデルの訓練と記録
                if self.enable_lstm:
                    try:
                        lstm_data = await self._prepare_lstm_data(
                            data, self._create_default_feature_set(symbol)
                        )
                        model = self._build_lstm_model()
                        self._train_lstm_model(
                            model,
                            lstm_data["X_train"],
                            lstm_data["y_train"],
                            lstm_data["X_test"],
                            lstm_data["y_test"],
                        )

                        lstm_score = await self._calculate_model_score(
                            model, lstm_data["X_test"], lstm_data["y_test"]
                        )
                        mlflow.log_metric("lstm_r2_score", lstm_score)
                        mlflow.keras.log_model(model, "lstm_model")
                        logger.info(
                            f"MLflow: LSTMモデル (R2: {lstm_score:.3f}) 記録完了"
                        )

                        if symbol not in self.trained_models:
                            self.trained_models[symbol] = {}
                        self.trained_models[symbol]["lstm"] = model
                        self._save_model(model, symbol, "lstm")
                        mlflow.log_param("lstm_train_status", "success")

                    except Exception as e:
                        logger.error(f"LSTMモデル訓練エラー: {symbol} - {e}")
                        mlflow.log_param("lstm_train_status", "failed")
                        traceback.print_exc()
                else:
                    mlflow.log_param("lstm_train_status", "skipped")

                # アンサンブルモデルの訓練と記録
                if SKLEARN_AVAILABLE:
                    try:
                        feature_set = await self.extract_advanced_features(data, symbol)
                        X, y = await self._prepare_ensemble_data(data, feature_set)

                        train_size = int(
                            len(X) * 0.8
                        )  # 過去80%を訓練、残りの20%をテスト
                        X_train, X_test = X[:train_size], X[train_size:]
                        y_train, y_test = y[:train_size], y[train_size:]

                        ensemble_models = await self._train_ensemble_models_internal(
                            X_train, y_train, symbol
                        )

                        # 各アンサンブルモデルを記録
                        for model_name, model_obj in ensemble_models.items():
                            mlflow.sklearn.log_model(
                                model_obj, f"ensemble_model_{model_name}"
                            )
                            # 各モデルのスコアも記録
                            y_pred = model_obj.predict(X_test)
                            model_score = max(0.0, r2_score(y_test, y_pred))
                            mlflow.log_metric(
                                f"ensemble_{model_name}_r2_score", model_score
                            )

                        # アンサンブル全体の重みとスコアを記録
                        model_weights = await self._calculate_model_weights(
                            ensemble_models, X_test, y_test
                        )
                        for k, v in model_weights.items():
                            mlflow.log_param(f"ensemble_weight_{k}", v)

                        ensemble_prediction_result = await self.ensemble_prediction(
                            data, symbol, feature_set
                        )  # 予測を実行して総合スコアを取得
                        mlflow.log_metric(
                            "ensemble_weighted_confidence",
                            ensemble_prediction_result.weighted_confidence,
                        )

                        logger.info("MLflow: アンサンブルモデル記録完了")

                        if symbol not in self.trained_models:
                            self.trained_models[symbol] = {}
                        self.trained_models[symbol][
                            "ensemble"
                        ] = ensemble_models  # アンサンブルモデル全体を管理するキーを追加
                        mlflow.log_param("ensemble_train_status", "success")

                    except Exception as e:
                        logger.error(f"アンサンブルモデル訓練エラー: {symbol} - {e}")
                        mlflow.log_param("ensemble_train_status", "failed")
                        traceback.print_exc()
                else:
                    mlflow.log_param("ensemble_train_status", "skipped")

                mlflow.log_param(
                    "total_training_time_seconds", time.time() - start_time
                )
                logger.info(f"MLflow実験終了: Run ID {run.info.run_id}")
        else:
            # MLflowが無効な場合でも、従来の訓練ロジックは実行
            # 訓練データをロード (DVC連携を想定)
            from src.day_trade.data.training_data_loader import (
                load_training_data_for_symbol,
            )

            try:
                data = load_training_data_for_symbol(symbol)
                if (
                    data.empty or len(data) < self.lstm_sequence_length + 20
                ):  # LSTMに必要な最低データ長
                    logger.warning(
                        f"訓練データが不足しているか無効です: {symbol}. 訓練をスキップします。"
                    )
                    return
            except NotImplementedError as e:
                logger.error(f"訓練データロードエラー: {e}")
                return
            except Exception as e:
                logger.error(f"訓練データロード中に予期せぬエラー: {e}")
                traceback.print_exc()
                return

            # LSTMモデルの訓練
            if self.enable_lstm:
                try:
                    lstm_data = await self._prepare_lstm_data(
                        data, self._create_default_feature_set(symbol)
                    )
                    model = self._build_lstm_model()
                    self._train_lstm_model(
                        model,
                        lstm_data["X_train"],
                        lstm_data["y_train"],
                        lstm_data["X_test"],
                        lstm_data["y_test"],
                    )
                    if symbol not in self.trained_models:
                        self.trained_models[symbol] = {}
                    self.trained_models[symbol]["lstm"] = model
                    self._save_model(model, symbol, "lstm")
                    logger.info(f"LSTMモデル訓練完了: {symbol}")
                except Exception as e:
                    logger.error(f"LSTMモデル訓練エラー: {symbol} - {e}")
                    traceback.print_exc()

            # アンサンブルモデルの訓練
            if SKLEARN_AVAILABLE:
                try:
                    feature_set = await self.extract_advanced_features(data, symbol)
                    X, y = await self._prepare_ensemble_data(data, feature_set)

                    # 訓練・テストデータ分割 (時間ベース)
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    models = await self._train_ensemble_models_internal(
                        X_train, y_train, symbol
                    )

                    # メモリに保存（必要であれば）
                    if symbol not in self.trained_models:
                        self.trained_models[symbol] = {}
                    self.trained_models[symbol][
                        "ensemble"
                    ] = models  # アンサンブルモデル全体を管理するキーを追加

                    logger.info(f"アンサンブルモデル訓練完了: {symbol}")

                except Exception as e:
                    logger.error(f"アンサンブルモデル訓練エラー: {symbol} - {e}")
                    traceback.print_exc()
        logger.info(f"MLモデル訓練完了: {symbol} ({time.time() - start_time:.2f}s)")
