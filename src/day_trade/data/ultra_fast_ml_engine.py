#!/usr/bin/env python3
"""
超高速ML投資助言エンジン

85銘柄を10秒以下で処理する極限最適化版
"""

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import RandomForestRegressor

    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn利用可能")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn未インストール")

PANDAS_TA_AVAILABLE = False  # 超高速版では使用しない
logger.info("pandas-ta未使用 - 超軽量処理")


class UltraFastMLEngine:
    """
    超高速ML投資助言エンジン

    極限最適化:
    - 単一軽量モデル（RandomForest 10 estimators）
    - 3特徴量のみ（returns_1d, returns_5d, sma_ratio）
    - 事前訓練済みモデルキャッシュ
    - 最小限データ処理
    """

    def __init__(self):
        """初期化"""
        self.cache_dir = Path("ultra_ml_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # 超軽量モデル（10 estimators のみ）
        self.base_model = (
            RandomForestRegressor(
                n_estimators=10,  # 極限まで削減
                max_depth=3,  # 浅い木
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            if SKLEARN_AVAILABLE
            else None
        )

        # モデルキャッシュ
        self.trained_models = {}
        self.feature_cache = {}

        logger.info("超高速MLエンジン初期化完了")
        logger.info("  - モデル: RandomForest(n_estimators=10)")
        logger.info("  - 特徴量: 3つのみ（最小限）")

    def ultra_fast_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        超高速特徴量計算（3特徴量のみ）

        Args:
            data: 価格データ

        Returns:
            最小限特徴量DataFrame
        """
        try:
            df = data.copy()

            # 必須3特徴量のみ
            df["returns_1d"] = df["Close"].pct_change()
            df["returns_5d"] = df["Close"].pct_change(5)
            df["sma_ratio"] = df["Close"] / df["Close"].rolling(10).mean()  # 10日に短縮

            # 最小限の3列のみ
            features = df[["returns_1d", "returns_5d", "sma_ratio"]].copy()
            return features.fillna(method="ffill").fillna(0)

        except Exception as e:
            logger.error(f"超高速特徴量エラー: {e}")
            # フォールバック
            df = data.copy()
            df["returns"] = df["Close"].pct_change()
            return df[["returns"]].fillna(0)

    def ultra_fast_predict(self, symbol: str, data: pd.DataFrame) -> float:
        """
        超高速予測（単一モデル、最小データ）

        Args:
            symbol: 銘柄コード
            data: 価格データ

        Returns:
            予測スコア（0-100）
        """
        if not SKLEARN_AVAILABLE:
            return 50.0

        try:
            # 特徴量キャッシュチェック
            data_hash = (
                hash(str(data["Close"].iloc[-10:].values)) if len(data) >= 10 else 0
            )
            cache_key = f"{symbol}_{data_hash}"

            if cache_key in self.feature_cache:
                features = self.feature_cache[cache_key]
            else:
                features = self.ultra_fast_features(data)
                self.feature_cache[cache_key] = features

            # ターゲット: 2日後（最短）
            target = data["Close"].pct_change(2).shift(-2) * 100

            # データ整合性（最小限）
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_indices]
            y = target[valid_indices]

            if len(X) < 5:  # 最小5点
                return 50.0

            # 事前訓練済みモデルチェック
            if symbol in self.trained_models:
                model = self.trained_models[symbol]
            else:
                # 高速訓練（最小データ）
                train_size = min(20, int(len(X) * 0.9))  # 最大20点のみ
                X_train = X.iloc[:train_size]
                y_train = y.iloc[:train_size]

                model = RandomForestRegressor(
                    n_estimators=5,  # さらに削減
                    max_depth=2,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)

                # キャッシュ保存
                self.trained_models[symbol] = model

            # 最新予測
            if len(X) > 0:
                prediction = model.predict(X.iloc[[-1]])[0]
                # スコア変換（-5〜+5 → 0〜100）
                score = max(0, min(100, 50 + prediction * 10))
                return score

            return 50.0

        except Exception as e:
            logger.debug(f"超高速予測エラー: {symbol} - {e}")
            return 50.0

    def ultra_fast_advice(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        超高速投資助言生成

        Args:
            symbol: 銘柄コード
            data: 価格データ

        Returns:
            投資助言辞書
        """
        try:
            # 予測実行
            prediction_score = self.ultra_fast_predict(symbol, data)

            # 超シンプル判定
            if prediction_score >= 70:
                return {
                    "symbol": symbol,
                    "advice": "BUY",
                    "confidence": min(95, prediction_score + 5),
                    "risk_level": "MEDIUM",
                    "reason": f"上昇予測 ({prediction_score:.0f})",
                    "processing_mode": "ultra_fast",
                }
            elif prediction_score <= 30:
                return {
                    "symbol": symbol,
                    "advice": "SELL",
                    "confidence": min(95, 105 - prediction_score),
                    "risk_level": "HIGH",
                    "reason": f"下落予測 ({prediction_score:.0f})",
                    "processing_mode": "ultra_fast",
                }
            else:
                return {
                    "symbol": symbol,
                    "advice": "HOLD",
                    "confidence": 70,
                    "risk_level": "LOW",
                    "reason": "横ばい予測",
                    "processing_mode": "ultra_fast",
                }

        except Exception as e:
            logger.error(f"超高速助言エラー: {symbol} - {e}")
            return {
                "symbol": symbol,
                "advice": "HOLD",
                "confidence": 50.0,
                "risk_level": "MEDIUM",
                "reason": "エラー発生",
                "processing_mode": "ultra_fast",
            }

    def batch_ultra_fast_analysis(
        self, stock_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        バッチ超高速分析

        Args:
            stock_data_dict: {銘柄: データ} 辞書

        Returns:
            {銘柄: 助言} 辞書
        """
        logger.info(f"バッチ超高速分析開始: {len(stock_data_dict)}銘柄")

        results = {}
        for symbol, data in stock_data_dict.items():
            if not data.empty:
                advice = self.ultra_fast_advice(symbol, data)
                results[symbol] = advice

        logger.info(f"バッチ超高速分析完了: {len(results)}銘柄")
        return results

    def clear_all_cache(self):
        """全キャッシュクリア"""
        self.trained_models.clear()
        self.feature_cache.clear()
        logger.info("全キャッシュクリア完了")

    def get_cache_info(self) -> Dict:
        """キャッシュ情報取得"""
        return {
            "trained_models": len(self.trained_models),
            "feature_cache": len(self.feature_cache),
            "memory_usage_kb": (
                len(str(self.trained_models)) + len(str(self.feature_cache))
            )
            / 1024,
        }


if __name__ == "__main__":
    # テスト用
    print("超高速MLエンジン テスト")

    # サンプルデータ生成
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sample_data = pd.DataFrame(
        {
            "Close": 1000 + np.cumsum(np.random.normal(0, 10, 100)),
            "Open": 1000 + np.cumsum(np.random.normal(0, 10, 100)),
            "High": 1000 + np.cumsum(np.random.normal(0, 10, 100)),
            "Low": 1000 + np.cumsum(np.random.normal(0, 10, 100)),
            "Volume": np.random.randint(100000, 1000000, 100),
        },
        index=dates,
    )

    engine = UltraFastMLEngine()

    # 単一テスト
    import time

    start = time.time()
    advice = engine.ultra_fast_advice("TEST", sample_data)
    elapsed = time.time() - start

    print("\n=== 超高速テスト結果 ===")
    print(f"処理時間: {elapsed:.4f}秒")
    print(f"助言: {advice['advice']}")
    print(f"信頼度: {advice['confidence']:.1f}%")
    print(f"理由: {advice['reason']}")

    # キャッシュ情報
    cache_info = engine.get_cache_info()
    print("\nキャッシュ情報:")
    print(f"  訓練済みモデル: {cache_info['trained_models']}個")
    print(f"  特徴量キャッシュ: {cache_info['feature_cache']}個")

    print(f"\n85銘柄推定処理時間: {elapsed * 85:.1f}秒")
