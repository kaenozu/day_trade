#!/usr/bin/env python3
"""
機械学習モデルモジュール

ボラティリティ予測のための機械学習モデル訓練・予測:
- RandomForestRegressor
- GradientBoostingRegressor
- モデル評価・診断
- アンサンブル学習
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from .base import VolatilityEngineBase, SKLEARN_AVAILABLE
from .ml_features import MLFeatureGenerator
from .realized_volatility import RealizedVolatilityCalculator
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

if SKLEARN_AVAILABLE:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import joblib


class MLVolatilityPredictor(VolatilityEngineBase):
    """
    機械学習ボラティリティ予測クラス

    複数のMLモデルを用いたボラティリティ予測と評価を提供します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        super().__init__(model_cache_dir)
        
        self.feature_generator = MLFeatureGenerator(model_cache_dir)
        self.rv_calculator = RealizedVolatilityCalculator(model_cache_dir)
        
        self.supported_models = ["random_forest", "gradient_boosting"]
        
        logger.info("ML ボラティリティ予測器初期化完了")

    def train_volatility_ml_model(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        target_horizon: int = 5,
        model_types: Optional[List[str]] = None,
        test_size: float = 0.25,
    ) -> Optional[Dict]:
        """
        ボラティリティ予測機械学習モデル訓練

        Args:
            data: 価格データ
            symbol: 銘柄コード
            target_horizon: 予測ホライゾン（日数）
            model_types: 使用するモデルタイプリスト
            test_size: テストデータ割合

        Returns:
            訓練結果辞書
        """
        if not self._validate_dependencies(["sklearn"]):
            return None

        if model_types is None:
            model_types = self.supported_models

        try:
            # 特徴量・ターゲット準備
            features, target = self._prepare_training_data(data, target_horizon)
            if features is None or target is None:
                return None

            # 訓練・テストデータ分割
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=42, shuffle=False
            )

            # 特徴量正規化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # モデル訓練・評価
            models = {}
            results = {}

            for model_type in model_types:
                model, result = self._train_single_model(
                    model_type, X_train, X_test, y_train, y_test,
                    X_train_scaled, X_test_scaled
                )
                if model and result:
                    models[model_type] = model
                    results[model_type] = result

            if not results:
                logger.error("全てのモデル訓練が失敗しました")
                return None

            # 最良モデル選択
            best_model_name = min(results.keys(), key=lambda x: results[x]["mse"])
            
            # モデル・スケーラー保存
            self.ml_models[symbol] = {
                "models": models,
                "scaler": scaler,
                "feature_names": features.columns.tolist(),
                "best_model_name": best_model_name,
                "target_horizon": target_horizon,
            }

            training_result = {
                "symbol": symbol,
                "target_horizon": target_horizon,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(features.columns),
                "models": results,
                "best_model": best_model_name,
                "best_performance": results[best_model_name],
                "feature_names": features.columns.tolist(),
                "training_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"ML訓練完了: {symbol} - 最良モデル: {best_model_name} "
                f"(R²={results[best_model_name]['r2']:.3f})"
            )
            return training_result

        except Exception as e:
            logger.error(f"ML訓練エラー ({symbol}): {e}")
            return None

    def _prepare_training_data(
        self, data: pd.DataFrame, target_horizon: int
    ) -> tuple:
        """
        訓練データの準備

        Args:
            data: 価格データ
            target_horizon: 予測ホライゾン

        Returns:
            (特徴量DataFrame, ターゲットSeries)
        """
        try:
            # 特徴量準備
            features = self.feature_generator.prepare_ml_features_for_volatility(data)

            if features.empty:
                logger.error("特徴量準備に失敗")
                return None, None

            # ターゲット変数: target_horizon日先の実現ボラティリティ
            target_vol = self.rv_calculator.calculate_realized_volatility(
                data, window=5, annualize=True
            )
            target = target_vol.shift(-target_horizon)

            # 有効データのみ抽出
            valid_mask = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_mask]
            y = target[valid_mask]

            if len(X) < 100:
                logger.warning(f"訓練データ不足: {len(X)}行 (最低100行必要)")
                return None, None

            return X, y

        except Exception as e:
            logger.error(f"訓練データ準備エラー: {e}")
            return None, None

    def _train_single_model(
        self, 
        model_type: str,
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series, 
        y_test: pd.Series,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray
    ) -> tuple:
        """
        単一モデルの訓練

        Args:
            model_type: モデルタイプ
            X_train: 訓練特徴量
            X_test: テスト特徴量
            y_train: 訓練ターゲット
            y_test: テストターゲット
            X_train_scaled: 正規化済み訓練特徴量
            X_test_scaled: 正規化済みテスト特徴量

        Returns:
            (モデル, 結果辞書)
        """
        try:
            if model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                )
                # RandomForestは特徴量正規化不要
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            elif model_type == "gradient_boosting":
                model = GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    subsample=0.8,
                    random_state=42,
                )
                # GradientBoostingも特徴量正規化不要
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            else:
                logger.warning(f"サポートされていないモデルタイプ: {model_type}")
                return None, None

            # 評価メトリクス計算
            result = self._calculate_model_metrics(
                y_test, predictions, model, X_train, y_train
            )

            logger.info(f"{model_type} 訓練完了 - R²: {result['r2']:.3f}")
            return model, result

        except Exception as e:
            logger.error(f"{model_type} 訓練エラー: {e}")
            return None, None

    def _calculate_model_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        model, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict:
        """
        モデル評価メトリクス計算

        Args:
            y_true: 実際の値
            y_pred: 予測値
            model: モデルオブジェクト
            X_train: 訓練特徴量
            y_train: 訓練ターゲット

        Returns:
            評価メトリクス辞書
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 交差検証スコア
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring='r2'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_mean, cv_std = None, None

        result = {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mae),
            "r2": float(r2),
            "cv_r2_mean": float(cv_mean) if cv_mean is not None else None,
            "cv_r2_std": float(cv_std) if cv_std is not None else None,
        }

        # 特徴量重要度（利用可能な場合）
        if hasattr(model, 'feature_importances_'):
            result["feature_importance"] = model.feature_importances_.tolist()

        return result

    def predict_volatility_ml(
        self, data: pd.DataFrame, symbol: str, horizon: int = 5
    ) -> Optional[Dict]:
        """
        機械学習によるボラティリティ予測

        Args:
            data: 価格データ
            symbol: 銘柄コード
            horizon: 予測ホライゾン

        Returns:
            予測結果辞書
        """
        if not self._validate_dependencies(["sklearn"]) or symbol not in self.ml_models:
            logger.error(f"MLモデルが利用できません: {symbol}")
            return None

        try:
            model_info = self.ml_models[symbol]
            models = model_info["models"]
            scaler = model_info["scaler"]
            feature_names = model_info["feature_names"]
            best_model_name = model_info["best_model_name"]

            # 特徴量準備
            features = self.feature_generator.prepare_ml_features_for_volatility(data)

            if features.empty or len(features) == 0:
                logger.error("予測用特徴量準備に失敗")
                return None

            # 最新データポイントを使用
            latest_features = features.iloc[-1:][feature_names]

            if latest_features.isnull().any().any():
                logger.warning("最新特徴量にNaN値を含む")
                latest_features = latest_features.fillna(0)

            # 予測実行
            predictions = {}
            for model_name, model in models.items():
                pred = model.predict(latest_features)[0]
                predictions[model_name] = float(max(pred, 0.001))  # 負の値を避ける

            # 結果処理
            result = self._process_prediction_results(
                predictions, best_model_name, data, symbol, horizon
            )

            logger.info(f"ML予測完了: {symbol} - 予測ボラティリティ: {result['predicted_volatility']:.1f}%")
            return result

        except Exception as e:
            logger.error(f"ML予測エラー ({symbol}): {e}")
            return None

    def _process_prediction_results(
        self, 
        predictions: Dict, 
        best_model_name: str, 
        data: pd.DataFrame, 
        symbol: str, 
        horizon: int
    ) -> Dict:
        """
        予測結果の処理

        Args:
            predictions: モデル別予測結果
            best_model_name: 最良モデル名
            data: 価格データ
            symbol: 銘柄コード
            horizon: 予測ホライゾン

        Returns:
            処理済み予測結果
        """
        # 最良モデルの予測
        best_pred = predictions[best_model_name]

        # 信頼区間の簡易計算（予測の不確実性）
        prediction_std = np.std(list(predictions.values()))
        confidence_interval = {
            "lower": max(best_pred - 1.96 * prediction_std, 0.001),
            "upper": best_pred + 1.96 * prediction_std,
        }

        # 現在の実現ボラティリティとの比較
        current_vol = self.rv_calculator.calculate_realized_volatility(
            data, window=20, annualize=True
        ).iloc[-1]

        return {
            "symbol": symbol,
            "prediction_horizon": horizon,
            "predicted_volatility": float(best_pred),
            "current_volatility": (
                float(current_vol) if pd.notna(current_vol) else None
            ),
            "volatility_change": (
                float(best_pred - current_vol) if pd.notna(current_vol) else None
            ),
            "confidence_interval": confidence_interval,
            "model_predictions": predictions,
            "best_model": best_model_name,
            "prediction_date": datetime.now().strftime("%Y-%m-%d"),
            "prediction_timestamp": datetime.now().isoformat(),
        }

    def get_model_diagnostics(self, symbol: str) -> Optional[Dict]:
        """
        MLモデルの診断情報取得

        Args:
            symbol: 銘柄コード

        Returns:
            診断情報辞書
        """
        if symbol not in self.ml_models:
            logger.error(f"MLモデルが存在しません: {symbol}")
            return None

        try:
            model_info = self.ml_models[symbol]
            models = model_info["models"]
            feature_names = model_info["feature_names"]

            diagnostics = {
                "symbol": symbol,
                "model_count": len(models),
                "feature_count": len(feature_names),
                "best_model": model_info["best_model_name"],
                "target_horizon": model_info["target_horizon"],
                "feature_names": feature_names,
                "model_details": {},
            }

            # 各モデルの詳細情報
            for model_name, model in models.items():
                model_details = {"model_type": model_name}
                
                # 特徴量重要度（利用可能な場合）
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(
                        feature_names, 
                        model.feature_importances_
                    ))
                    # 上位10特徴量
                    top_features = sorted(
                        importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                    model_details["top_features"] = top_features

                diagnostics["model_details"][model_name] = model_details

            return diagnostics

        except Exception as e:
            logger.error(f"ML診断情報取得エラー ({symbol}): {e}")
            return {"symbol": symbol, "error": str(e)}

    def save_model(self, symbol: str, filepath: Optional[str] = None) -> bool:
        """
        モデルを保存

        Args:
            symbol: 銘柄コード
            filepath: 保存先パス

        Returns:
            保存成功フラグ
        """
        if symbol not in self.ml_models:
            logger.error(f"保存するモデルが存在しません: {symbol}")
            return False

        try:
            if filepath is None:
                filepath = self.model_cache_dir / f"ml_model_{symbol}.joblib"

            joblib.dump(self.ml_models[symbol], filepath)
            logger.info(f"モデル保存完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"モデル保存エラー ({symbol}): {e}")
            return False

    def load_model(self, symbol: str, filepath: str) -> bool:
        """
        モデルを読み込み

        Args:
            symbol: 銘柄コード
            filepath: 読み込み元パス

        Returns:
            読み込み成功フラグ
        """
        try:
            model_info = joblib.load(filepath)
            self.ml_models[symbol] = model_info
            logger.info(f"モデル読み込み完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"モデル読み込みエラー ({symbol}): {e}")
            return False