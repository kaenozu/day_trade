#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced ML Ensemble System - 高度な機械学習アンサンブルシステム
Issue #939 対応: LightGBM、CatBoost、XGBoostによる予測精度向上
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import json

# 基本的な機械学習ライブラリ
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    cb = None

# Sklearn
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# カスタムモジュール
try:
    from performance_monitor import performance_monitor, track_performance
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False
    def track_performance(func):
        return func

try:
    from audit_logger import audit_logger
    HAS_AUDIT_LOGGER = True
except ImportError:
    HAS_AUDIT_LOGGER = False

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class ModelConfiguration:
    """モデル設定クラス"""
    name: str
    enabled: bool = True
    hyperparameters: Dict[str, Any] = None
    weight: float = 1.0
    optimization_target: str = 'mae'  # mae, rmse, r2


@dataclass
class PredictionResult:
    """予測結果クラス"""
    symbol: str
    predictions: Dict[str, float]
    ensemble_prediction: float
    confidence_score: float
    model_contributions: Dict[str, float]
    feature_importance: Dict[str, float]
    timestamp: datetime
    processing_time_ms: float


class EnhancedMLEnsembleSystem:
    """高度な機械学習アンサンブルシステム"""

    def __init__(self, model_dir: str = "ml_models_data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # モデル設定
        self.model_configs = self._initialize_model_configs()

        # 学習済みモデル
        self.trained_models = {}
        self.scalers = {}
        self.feature_names = []

        # パフォーマンス追跡
        self.prediction_history = []
        self.model_performance = {}

        # キャッシュ
        self.prediction_cache = {}
        self.cache_expiry = {}

        self._load_existing_models()

    def _initialize_model_configs(self) -> Dict[str, ModelConfiguration]:
        """モデル設定を初期化"""
        configs = {}

        # XGBoost設定
        if HAS_XGBOOST:
            configs['xgboost'] = ModelConfiguration(
                name='XGBoost',
                enabled=True,
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                },
                weight=1.2,
                optimization_target='mae'
            )

        # LightGBM設定
        if HAS_LIGHTGBM:
            configs['lightgbm'] = ModelConfiguration(
                name='LightGBM',
                enabled=True,
                hyperparameters={
                    'n_estimators': 300,
                    'max_depth': 7,
                    'learning_rate': 0.08,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                },
                weight=1.3,
                optimization_target='mae'
            )

        # CatBoost設定
        if HAS_CATBOOST:
            configs['catboost'] = ModelConfiguration(
                name='CatBoost',
                enabled=True,
                hyperparameters={
                    'iterations': 200,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_seed': 42,
                    'verbose': False,
                    'thread_count': -1
                },
                weight=1.1,
                optimization_target='mae'
            )

        # Random Forest設定
        configs['random_forest'] = ModelConfiguration(
            name='Random Forest',
            enabled=True,
            hyperparameters={
                'n_estimators': 150,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            weight=0.8,
            optimization_target='mae'
        )

        return configs

    def _load_existing_models(self):
        """既存モデルの読み込み"""
        try:
            # モデルファイルの存在確認
            for model_name in self.model_configs.keys():
                model_path = self.model_dir / f"{model_name}_model.joblib"
                scaler_path = self.model_dir / f"{model_name}_scaler.joblib"

                if model_path.exists() and scaler_path.exists():
                    self.trained_models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)

                    if HAS_AUDIT_LOGGER:
                        audit_logger.log_business_event(
                            "model_loaded",
                            {"model": model_name, "path": str(model_path)}
                        )

            # 特徴量名の読み込み
            features_path = self.model_dir / "feature_names.json"
            if features_path.exists():
                with open(features_path, 'r', encoding='utf-8') as f:
                    self.feature_names = json.load(f)

        except Exception as e:
            print(f"既存モデル読み込みエラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "model_loading"})

    @track_performance
    def train_ensemble_models(self,
                            training_data: pd.DataFrame,
                            target_column: str,
                            feature_columns: List[str] = None,
                            test_size: float = 0.2) -> Dict[str, Any]:
        """アンサンブルモデルの学習"""
        start_time = time.time()

        try:
            # データ準備
            if feature_columns is None:
                feature_columns = [col for col in training_data.columns if col != target_column]

            X = training_data[feature_columns].copy()
            y = training_data[target_column].copy()

            self.feature_names = feature_columns

            # 欠損値処理
            X = X.fillna(X.mean())

            # 無限値の処理
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())

            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            training_results = {}

            # 各モデルの学習
            for model_name, config in self.model_configs.items():
                if not config.enabled:
                    continue

                try:
                    print(f"学習中: {config.name}")

                    # スケーラー準備
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # モデル作成と学習
                    model = self._create_model(model_name, config.hyperparameters)
                    model.fit(X_train_scaled, y_train)

                    # 予測と評価
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)

                    # メトリクス計算
                    train_metrics = self._calculate_metrics(y_train, y_pred_train)
                    test_metrics = self._calculate_metrics(y_test, y_pred_test)

                    # クロスバリデーション
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3,
                                              scoring='neg_mean_absolute_error', n_jobs=-1)

                    # 結果保存
                    training_results[model_name] = {
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'cv_mae_mean': -cv_scores.mean(),
                        'cv_mae_std': cv_scores.std(),
                        'feature_importance': self._get_feature_importance(model, feature_columns)
                    }

                    # モデル保存
                    self.trained_models[model_name] = model
                    self.scalers[model_name] = scaler
                    self.model_performance[model_name] = test_metrics

                    # ファイルに保存
                    model_path = self.model_dir / f"{model_name}_model.joblib"
                    scaler_path = self.model_dir / f"{model_name}_scaler.joblib"

                    joblib.dump(model, model_path)
                    joblib.dump(scaler, scaler_path)

                    print(f"✅ {config.name} 学習完了 - Test MAE: {test_metrics['mae']:.4f}")

                except Exception as e:
                    print(f"❌ {config.name} 学習エラー: {e}")
                    if HAS_AUDIT_LOGGER:
                        audit_logger.log_error_with_context(e, {"model": model_name})
                    continue

            # アンサンブル学習（スタッキング）
            try:
                ensemble_result = self._train_ensemble_meta_model(X_train, X_test, y_train, y_test)
                training_results['ensemble'] = ensemble_result
                print("✅ アンサンブル学習完了")
            except Exception as e:
                print(f"❌ アンサンブル学習エラー: {e}")
                if HAS_AUDIT_LOGGER:
                    audit_logger.log_error_with_context(e, {"context": "ensemble_training"})

            # 特徴量名を保存
            features_path = self.model_dir / "feature_names.json"
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_names, f, ensure_ascii=False, indent=2)

            training_time = (time.time() - start_time) * 1000

            # 学習結果サマリー
            summary = {
                'training_completed': datetime.now().isoformat(),
                'training_time_ms': training_time,
                'models_trained': len(training_results),
                'feature_count': len(feature_columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'results': training_results
            }

            # パフォーマンス記録
            if HAS_PERFORMANCE_MONITOR:
                performance_monitor.track_analysis_time(
                    symbol="ENSEMBLE_TRAINING",
                    duration=training_time / 1000,
                    analysis_type="ensemble_training"
                )

            if HAS_AUDIT_LOGGER:
                audit_logger.log_business_event(
                    "ensemble_training_completed",
                    {"models_count": len(training_results), "training_time_ms": training_time}
                )

            return summary

        except Exception as e:
            print(f"アンサンブル学習エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "ensemble_training"})
            raise

    def _create_model(self, model_name: str, hyperparams: Dict[str, Any]):
        """モデルインスタンスを作成"""
        if model_name == 'xgboost' and HAS_XGBOOST:
            return xgb.XGBRegressor(**hyperparams)

        elif model_name == 'lightgbm' and HAS_LIGHTGBM:
            return lgb.LGBMRegressor(**hyperparams)

        elif model_name == 'catboost' and HAS_CATBOOST:
            return cb.CatBoostRegressor(**hyperparams)

        elif model_name == 'random_forest':
            return RandomForestRegressor(**hyperparams)

        else:
            raise ValueError(f"サポートされていないモデル: {model_name}")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """評価メトリクスを計算"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else 0
        }

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度を取得"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                importances = model.get_feature_importance()
            else:
                return {}

            return {name: float(importance) for name, importance in zip(feature_names, importances)}

        except Exception:
            return {}

    def _train_ensemble_meta_model(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """スタッキング用メタモデルの学習"""
        if not self.trained_models:
            return {}

        # 各モデルの予測を作成
        train_predictions = []
        test_predictions = []
        model_names = []

        for model_name, model in self.trained_models.items():
            if model_name in self.scalers:
                scaler = self.scalers[model_name]
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                train_pred = model.predict(X_train_scaled).reshape(-1, 1)
                test_pred = model.predict(X_test_scaled).reshape(-1, 1)

                train_predictions.append(train_pred)
                test_predictions.append(test_pred)
                model_names.append(model_name)

        if not train_predictions:
            return {}

        # スタッキング用データ準備
        X_stack_train = np.hstack(train_predictions)
        X_stack_test = np.hstack(test_predictions)

        # メタモデル（線形回帰）学習
        meta_model = LinearRegression()
        meta_model.fit(X_stack_train, y_train)

        # 予測と評価
        y_pred_meta = meta_model.predict(X_stack_test)
        meta_metrics = self._calculate_metrics(y_test, y_pred_meta)

        # メタモデル保存
        self.trained_models['meta_model'] = meta_model
        meta_path = self.model_dir / "meta_model.joblib"
        joblib.dump(meta_model, meta_path)

        return {
            'meta_model_metrics': meta_metrics,
            'meta_model_coefficients': meta_model.coef_.tolist(),
            'model_names': model_names
        }

    @track_performance
    def predict(self, features: pd.DataFrame, symbol: str = "UNKNOWN") -> PredictionResult:
        """アンサンブル予測を実行"""
        start_time = time.time()

        try:
            # キャッシュチェック
            cache_key = self._generate_cache_key(features, symbol)
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
                    return cache_entry

            # 特徴量準備
            if not self.feature_names:
                raise ValueError("モデルが学習されていません")

            # 必要な特徴量のみ選択
            available_features = [f for f in self.feature_names if f in features.columns]
            if len(available_features) < len(self.feature_names) * 0.8:  # 80%以上の特徴量が必要
                raise ValueError(f"必要な特徴量が不足しています: {len(available_features)}/{len(self.feature_names)}")

            X = features[available_features].copy()

            # データ前処理
            X = X.fillna(X.mean())
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)

            # 各モデルで予測
            predictions = {}
            model_contributions = {}

            for model_name, model in self.trained_models.items():
                if model_name == 'meta_model':
                    continue

                if model_name not in self.scalers:
                    continue

                try:
                    scaler = self.scalers[model_name]
                    X_scaled = scaler.transform(X)

                    pred = model.predict(X_scaled)[0]
                    predictions[model_name] = float(pred)

                    # 重み付け貢献度
                    weight = self.model_configs[model_name].weight
                    model_contributions[model_name] = float(pred * weight)

                except Exception as e:
                    print(f"モデル {model_name} 予測エラー: {e}")
                    continue

            if not predictions:
                raise ValueError("すべてのモデルで予測に失敗しました")

            # アンサンブル予測（重み付き平均）
            total_weight = sum(self.model_configs[name].weight for name in predictions.keys())
            ensemble_prediction = sum(
                pred * self.model_configs[name].weight
                for name, pred in predictions.items()
            ) / total_weight

            # スタッキング予測（メタモデルがある場合）
            if 'meta_model' in self.trained_models and len(predictions) >= 2:
                try:
                    meta_model = self.trained_models['meta_model']
                    meta_input = np.array(list(predictions.values())).reshape(1, -1)
                    stacking_prediction = meta_model.predict(meta_input)[0]

                    # アンサンブル予測にスタッキング結果を加重平均
                    ensemble_prediction = (ensemble_prediction * 0.7 + stacking_prediction * 0.3)

                except Exception as e:
                    print(f"スタッキング予測エラー: {e}")

            # 信頼度スコア計算
            confidence_score = self._calculate_confidence_score(predictions)

            # 特徴量重要度の統合
            feature_importance = self._get_ensemble_feature_importance()

            processing_time = (time.time() - start_time) * 1000

            # 結果作成
            result = PredictionResult(
                symbol=symbol,
                predictions=predictions,
                ensemble_prediction=float(ensemble_prediction),
                confidence_score=float(confidence_score),
                model_contributions=model_contributions,
                feature_importance=feature_importance,
                timestamp=datetime.now(),
                processing_time_ms=processing_time
            )

            # キャッシュに保存（5分間有効）
            self.prediction_cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)

            # 履歴保存
            self.prediction_history.append(result)
            if len(self.prediction_history) > 1000:  # 最新1000件のみ保持
                self.prediction_history.pop(0)

            # ログ記録
            if HAS_AUDIT_LOGGER:
                audit_logger.log_analysis_request(
                    symbol=symbol,
                    source="ensemble_ml_system",
                    request_data={
                        "models_used": list(predictions.keys()),
                        "processing_time_ms": processing_time,
                        "confidence_score": confidence_score
                    }
                )

            return result

        except Exception as e:
            print(f"予測エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"symbol": symbol, "context": "prediction"})
            raise

    def _generate_cache_key(self, features: pd.DataFrame, symbol: str) -> str:
        """キャッシュキーを生成"""
        feature_hash = hash(tuple(features.iloc[0].values))
        return f"ensemble_pred_{symbol}_{feature_hash}"

    def _calculate_confidence_score(self, predictions: Dict[str, float]) -> float:
        """信頼度スコアを計算"""
        if len(predictions) < 2:
            return 0.5

        values = list(predictions.values())
        mean_pred = np.mean(values)
        std_pred = np.std(values)

        # 標準偏差が小さいほど信頼度が高い
        if std_pred == 0:
            return 1.0

        # 正規化された信頼度（0-1の範囲）
        confidence = 1.0 / (1.0 + std_pred / abs(mean_pred) if mean_pred != 0 else std_pred)
        return min(max(confidence, 0.0), 1.0)

    def _get_ensemble_feature_importance(self) -> Dict[str, float]:
        """アンサンブル特徴量重要度を取得"""
        if not self.trained_models or not self.feature_names:
            return {}

        # 各モデルの特徴量重要度を平均
        importance_sum = {name: 0.0 for name in self.feature_names}
        model_count = 0

        for model_name, model in self.trained_models.items():
            if model_name == 'meta_model':
                continue

            feature_imp = self._get_feature_importance(model, self.feature_names)
            if feature_imp:
                for feature_name in self.feature_names:
                    importance_sum[feature_name] += feature_imp.get(feature_name, 0.0)
                model_count += 1

        if model_count == 0:
            return {}

        # 平均化
        return {name: importance / model_count for name, importance in importance_sum.items()}

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """モデル性能サマリーを取得"""
        summary = {
            'models_available': list(self.trained_models.keys()),
            'model_performance': dict(self.model_performance),
            'feature_count': len(self.feature_names),
            'predictions_made': len(self.prediction_history),
            'last_prediction': self.prediction_history[-1].timestamp.isoformat() if self.prediction_history else None
        }

        # 最近の予測統計
        if self.prediction_history:
            recent_predictions = self.prediction_history[-100:]  # 最新100件
            avg_confidence = np.mean([p.confidence_score for p in recent_predictions])
            avg_processing_time = np.mean([p.processing_time_ms for p in recent_predictions])

            summary['recent_statistics'] = {
                'average_confidence': float(avg_confidence),
                'average_processing_time_ms': float(avg_processing_time),
                'predictions_analyzed': len(recent_predictions)
            }

        return summary

    def export_model_info(self) -> str:
        """モデル情報をエクスポート"""
        model_info = {
            'export_timestamp': datetime.now().isoformat(),
            'model_configurations': {
                name: {
                    'name': config.name,
                    'enabled': config.enabled,
                    'weight': config.weight,
                    'hyperparameters': config.hyperparameters
                }
                for name, config in self.model_configs.items()
            },
            'trained_models': list(self.trained_models.keys()),
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'performance_summary': self.get_model_performance_summary()
        }

        filename = self.model_dir / f"model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2, default=str)

        return str(filename)


# グローバルインスタンス
enhanced_ml_ensemble = EnhancedMLEnsembleSystem()


if __name__ == "__main__":
    # テスト実行
    print("Enhanced ML Ensemble System テスト開始")

    # テストデータ作成
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # 特徴量作成
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # ターゲット作成（複雑な非線形関係）
    y = (X[:, 0] * 2 + X[:, 1] ** 2 + np.sin(X[:, 2]) +
         np.random.randn(n_samples) * 0.1)

    # DataFrameに変換
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # アンサンブルシステムのテスト
    ensemble = EnhancedMLEnsembleSystem()

    print("モデル学習開始...")
    training_result = ensemble.train_ensemble_models(df, 'target', feature_names)

    print(f"学習完了: {training_result['models_trained']}個のモデルを学習")

    # 予測テスト
    test_features = df[feature_names].iloc[:1]
    prediction = ensemble.predict(test_features, symbol="TEST_SYMBOL")

    print(f"予測結果:")
    print(f"  アンサンブル予測: {prediction.ensemble_prediction:.4f}")
    print(f"  信頼度スコア: {prediction.confidence_score:.4f}")
    print(f"  処理時間: {prediction.processing_time_ms:.2f}ms")
    print(f"  使用モデル数: {len(prediction.predictions)}")

    # パフォーマンスサマリー
    print("\nモデル性能サマリー:")
    summary = ensemble.get_model_performance_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    # モデル情報エクスポート
    export_file = ensemble.export_model_info()
    print(f"\nモデル情報をエクスポート: {export_file}")

    print("テスト完了 ✅")