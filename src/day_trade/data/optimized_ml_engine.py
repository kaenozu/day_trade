#!/usr/bin/env python3
"""
最適化済みML投資助言エンジン

パフォーマンス重視版：85銘柄を10秒以下で処理
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 依存関係チェック
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
    logger.info("pandas-ta利用可能")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta未インストール - 技術指標計算が制限されます")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn利用可能")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn未インストール - ML機能利用不可")


class OptimizedMLEngine:
    """
    最適化済みML投資助言エンジン

    高速化手法:
    - 軽量モデル使用（LinearRegression + 小規模RandomForest）
    - モデル再利用によるキャッシング
    - 特徴量計算最適化
    - 不要な計算削除
    """

    def __init__(self, model_cache_dir: str = "model_cache"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)

        # 軽量化されたモデル設定
        self.models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=50,  # 200→50に削減
                max_depth=6,  # 10→6に削減
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        }

        # モデルキャッシュ
        self.trained_models = {}
        self.scalers = {}

        logger.info("最適化MLエンジン初期化完了")
        logger.info(f"  - モデル数: {len(self.models)}（軽量化）")
        logger.info(f"  - キャッシュディレクトリ: {self.model_cache_dir}")

    def prepare_lightweight_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        軽量特徴量準備（高速版）

        Args:
            data: 価格データ

        Returns:
            特徴量DataFrame
        """
        if not PANDAS_TA_AVAILABLE:
            return self._basic_features(data)

        try:
            df = data.copy()

            # 基本価格特徴量（高速）
            df["returns_1d"] = df["Close"].pct_change()
            df["returns_5d"] = df["Close"].pct_change(5)
            df["returns_20d"] = df["Close"].pct_change(20)

            # 移動平均（軽量化）
            df["sma_5"] = ta.sma(df["Close"], length=5)
            df["sma_20"] = ta.sma(df["Close"], length=20)
            df["sma_ratio"] = df["Close"] / df["sma_20"]

            # 必須指標のみ（RSI、MACD削除で高速化）
            df["bb_upper"], df["bb_mid"], df["bb_lower"] = ta.bbands(
                df["Close"], length=20
            ).values.T
            df["bb_position"] = (df["Close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

            # ボラティリティ（軽量版）
            df["volatility_5d"] = df["returns_1d"].rolling(5).std()
            df["volatility_20d"] = df["returns_1d"].rolling(20).std()

            # 特徴量選択（重要な7特徴量のみ）
            feature_cols = [
                "returns_1d",
                "returns_5d",
                "sma_ratio",
                "bb_position",
                "volatility_5d",
                "volatility_20d",
                "returns_20d",
            ]

            features = df[feature_cols].copy()
            return features.fillna(method="ffill").fillna(0)

        except Exception as e:
            logger.error(f"軽量特徴量計算エラー: {e}")
            return self._basic_features(data)

    def _basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量（pandas-ta無し）"""
        df = data.copy()

        # 価格変動率
        df["returns_1d"] = df["Close"].pct_change()
        df["returns_5d"] = df["Close"].pct_change(5)
        df["returns_20d"] = df["Close"].pct_change(20)

        # 移動平均
        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["sma_ratio"] = df["Close"] / df["sma_20"]

        # ボラティリティ
        df["volatility_5d"] = df["returns_1d"].rolling(5).std()

        feature_cols = [
            "returns_1d",
            "returns_5d",
            "returns_20d",
            "sma_ratio",
            "volatility_5d",
        ]

        return df[feature_cols].fillna(method="ffill").fillna(0)

    def fast_predict_trends(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        高速トレンド予測

        Args:
            symbol: 銘柄コード
            data: 価格データ

        Returns:
            予測結果辞書
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn未インストール")
            return None

        try:
            # 軽量特徴量計算
            features = self.prepare_lightweight_features(data)

            # ターゲット変数（3日後に短縮で高速化）
            target = data["Close"].pct_change(3).shift(-3) * 100

            # データ整合性チェック
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_indices]
            y = target[valid_indices]

            if len(X) < 10:  # 最小データ数を20→10に削減
                logger.warning(f"データ不足: {symbol} - {len(X)}行")
                return self._get_default_prediction(symbol)

            # 訓練・テスト分割（高速化のため70/30→80/20）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # モデルキャッシュチェック
            model_key = f"{symbol}_optimized"
            if model_key in self.trained_models:
                # キャッシュ済みモデル使用
                model_result = self.trained_models[model_key]
                best_model = model_result["model"]
                scaler = model_result.get("scaler")

                logger.debug(f"モデルキャッシュ使用: {symbol}")

            else:
                # 新規モデル訓練（軽量版）
                best_model, best_r2, scaler = self._train_lightweight_model(
                    X_train, X_test, y_train, y_test
                )

                # キャッシュ保存
                self.trained_models[model_key] = {
                    "model": best_model,
                    "scaler": scaler,
                    "r2_score": best_r2,
                    "symbol": symbol,
                }

                logger.debug(f"新規モデル訓練: {symbol} - R2: {best_r2:.3f}")

            # 最新データで最終予測
            latest_features = X.iloc[[-1]]
            if scaler:
                latest_scaled = scaler.transform(latest_features)
                final_prediction = best_model.predict(latest_scaled)[0]
            else:
                final_prediction = best_model.predict(latest_features)[0]

            return {
                "symbol": symbol,
                "prediction": final_prediction,
                "confidence": min(abs(final_prediction) * 10, 95),
                "model_type": "optimized_lightweight",
                "data_points": len(X),
            }

        except Exception as e:
            logger.error(f"高速予測エラー: {symbol} - {e}")
            return self._get_default_prediction(symbol)

    def _train_lightweight_model(self, X_train, X_test, y_train, y_test) -> Tuple:
        """軽量モデル訓練（2モデルのみ）"""

        # 1. Linear Regression（高速）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        linear_model = LinearRegression()
        linear_model.fit(X_train_scaled, y_train)
        linear_pred = linear_model.predict(X_test_scaled)
        linear_r2 = r2_score(y_test, linear_pred)

        # 2. Random Forest（軽量版）
        rf_model = self.models["random_forest"]
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)

        # ベストモデル選択
        if linear_r2 > rf_r2:
            return linear_model, linear_r2, scaler
        else:
            return rf_model, rf_r2, None

    def generate_fast_advice(
        self, symbol: str, data: pd.DataFrame, features: pd.DataFrame = None
    ) -> Dict:
        """
        高速投資助言生成

        Args:
            symbol: 銘柄コード
            data: 価格データ
            features: 特徴量（未使用、互換性のため）

        Returns:
            投資助言辞書
        """
        prediction_result = self.fast_predict_trends(symbol, data)

        if not prediction_result:
            return {
                "symbol": symbol,
                "advice": "HOLD",
                "confidence": 50.0,
                "risk_level": "MEDIUM",
                "reason": "データ不足により分析困難",
            }

        prediction = prediction_result["prediction"]
        base_confidence = prediction_result["confidence"]

        # 投資判定ロジック（単純化）
        if prediction > 2.0:
            advice = "BUY"
            risk_level = "LOW" if base_confidence > 80 else "MEDIUM"
            reason = f"{prediction:.1f}%の上昇予測"
        elif prediction < -2.0:
            advice = "SELL"
            risk_level = "HIGH" if abs(prediction) > 5 else "MEDIUM"
            reason = f"{abs(prediction):.1f}%の下落予測"
        else:
            advice = "HOLD"
            risk_level = "LOW"
            reason = "横ばいまたは小幅変動予測"

        return {
            "symbol": symbol,
            "advice": advice,
            "confidence": min(base_confidence, 95.0),
            "risk_level": risk_level,
            "reason": reason,
        }

    def _get_default_prediction(self, symbol: str) -> Dict:
        """デフォルト予測結果"""
        return {
            "symbol": symbol,
            "prediction": 0.0,
            "confidence": 50.0,
            "model_type": "default",
            "data_points": 0,
        }

    def clear_cache(self):
        """モデルキャッシュクリア"""
        self.trained_models.clear()
        logger.info("モデルキャッシュをクリアしました")

    def get_cache_stats(self) -> Dict:
        """キャッシュ統計"""
        return {
            "cached_models": len(self.trained_models),
            "cache_memory_usage": len(str(self.trained_models)) / 1024,  # KB概算
        }


if __name__ == "__main__":
    # テスト用
    print("最適化MLエンジンテスト")

    # サンプルデータ生成
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    sample_data = pd.DataFrame(
        {
            "Close": 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            "Open": 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            "High": 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            "Low": 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    engine = OptimizedMLEngine()

    try:
        import time

        start_time = time.time()

        advice = engine.generate_fast_advice("TEST", sample_data)

        elapsed = time.time() - start_time

        print("\n=== 高速助言テスト結果 ===")
        print(f"処理時間: {elapsed:.3f}秒")
        print(f"助言: {advice['advice']}")
        print(f"信頼度: {advice['confidence']:.1f}%")
        print(f"理由: {advice['reason']}")

        # キャッシュ統計
        stats = engine.get_cache_stats()
        print("\nキャッシュ統計:")
        print(f"  - キャッシュ数: {stats['cached_models']}")

    except Exception as e:
        print(f"テストエラー: {e}")
