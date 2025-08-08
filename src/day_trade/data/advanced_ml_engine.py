#!/usr/bin/env python3
"""
高度な機械学習エンジン

pandas-ta + scikit-learnを使用したより本格的な技術分析と機械学習予測
"""

import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージの段階的読み込み
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
    logger.info("pandas-ta利用可能")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning(
        "pandas-ta未インストール - pip install pandas-taでインストールしてください"
    )

try:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn利用可能")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn未インストール - pip install scikit-learnでインストールしてください"
    )

# 警告の抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class AdvancedMLEngine:
    """
    高度な機械学習エンジンクラス

    pandas-ta + scikit-learnによる本格的な技術分析と予測モデル
    """

    def __init__(self, model_cache_path: Optional[str] = None):
        self.model_cache_path = model_cache_path or "data/ml_models.db"
        self.models = {}
        self._ensure_model_cache_db()

    def _ensure_model_cache_db(self):
        """モデルキャッシュDBの初期化"""
        cache_path = Path(self.model_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.model_cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_performance (
                    symbol TEXT,
                    model_type TEXT,
                    r2_score REAL,
                    mse REAL,
                    training_date TEXT,
                    feature_count INTEGER,
                    PRIMARY KEY (symbol, model_type)
                )
            """
            )

    def calculate_advanced_technical_indicators(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        pandas-taを使用した高度な技術指標計算

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame: 技術指標が追加されたDataFrame
        """
        if not PANDAS_TA_AVAILABLE or data is None or data.empty:
            logger.warning("pandas-ta利用不可またはデータなし - 基本指標のみ計算")
            return self._calculate_basic_indicators_fallback(data)

        try:
            df = data.copy()

            # pandas-taによる自動指標計算
            df.ta.strategy("All")

            # 主要指標の個別計算（念のため）
            df["SMA_20"] = ta.sma(df["Close"], length=20)
            df["EMA_12"] = ta.ema(df["Close"], length=12)
            df["EMA_26"] = ta.ema(df["Close"], length=26)
            df["RSI_14"] = ta.rsi(df["Close"], length=14)
            df["MACD"] = ta.macd(df["Close"])["MACD_12_26_9"]
            df["BB_Upper"] = ta.bbands(df["Close"])["BBU_20_2.0"]
            df["BB_Lower"] = ta.bbands(df["Close"])["BBL_20_2.0"]
            df["ADX"] = ta.adx(df["High"], df["Low"], df["Close"])["ADX_14"]
            df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"])
            df["Williams_R"] = ta.willr(df["High"], df["Low"], df["Close"])
            df["Stoch_K"] = ta.stoch(df["High"], df["Low"], df["Close"])[
                "STOCHk_14_3_3"
            ]

            # 出来高指標
            df["OBV"] = ta.obv(df["Close"], df["Volume"])
            df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"])
            df["MFI"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"])

            # トレンド指標
            df["PSAR"] = ta.psar(df["High"], df["Low"])["PSARl_0.02_0.2"]
            df["Supertrend"] = ta.supertrend(df["High"], df["Low"], df["Close"])[
                "SUPERT_7_3.0"
            ]

            logger.info(f"pandas-ta高度指標計算完了: {len(df.columns)}列")
            return df

        except Exception as e:
            logger.error(f"pandas-ta指標計算エラー: {e}")
            return self._calculate_basic_indicators_fallback(data)

    def _calculate_basic_indicators_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """フォールバック用基本指標計算"""
        if data is None or data.empty:
            return pd.DataFrame()

        try:
            df = data.copy()

            # 基本的な移動平均
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["EMA_12"] = df["Close"].ewm(span=12).mean()
            df["EMA_26"] = df["Close"].ewm(span=26).mean()

            # RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI_14"] = 100 - (100 / (1 + rs))

            # MACD
            df["MACD"] = df["EMA_12"] - df["EMA_26"]

            # ボリンジャーバンド
            df["BB_Upper"] = df["SMA_20"] + (df["Close"].rolling(window=20).std() * 2)
            df["BB_Lower"] = df["SMA_20"] - (df["Close"].rolling(window=20).std() * 2)

            logger.info("基本指標計算完了（フォールバック）")
            return df

        except Exception as e:
            logger.error(f"基本指標計算エラー: {e}")
            return data

    def prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        機械学習用特徴量準備

        Args:
            data: 技術指標付きDataFrame

        Returns:
            DataFrame: ML用特徴量DataFrame
        """
        try:
            features = pd.DataFrame(index=data.index)

            # 価格系特徴量
            features["price_change_1d"] = data["Close"].pct_change(1)
            features["price_change_5d"] = data["Close"].pct_change(5)
            features["price_change_20d"] = data["Close"].pct_change(20)
            features["volume_change_1d"] = data["Volume"].pct_change(1)
            features["high_low_ratio"] = (data["High"] - data["Low"]) / data["Close"]

            # 技術指標特徴量
            if "RSI_14" in data.columns:
                features["rsi"] = data["RSI_14"]
                features["rsi_oversold"] = (data["RSI_14"] < 30).astype(int)
                features["rsi_overbought"] = (data["RSI_14"] > 70).astype(int)

            if "MACD" in data.columns:
                features["macd"] = data["MACD"]
                features["macd_positive"] = (data["MACD"] > 0).astype(int)

            if "SMA_20" in data.columns:
                features["price_sma_ratio"] = data["Close"] / data["SMA_20"]

            if "BB_Upper" in data.columns and "BB_Lower" in data.columns:
                bb_width = data["BB_Upper"] - data["BB_Lower"]
                features["bb_position"] = (data["Close"] - data["BB_Lower"]) / bb_width
                features["bb_squeeze"] = (
                    bb_width < bb_width.rolling(20).mean()
                ).astype(int)

            # ボラティリティ特徴量
            features["volatility_5d"] = data["Close"].pct_change().rolling(5).std()
            features["volatility_20d"] = data["Close"].pct_change().rolling(20).std()

            # モメンタム特徴量
            if "ADX" in data.columns:
                features["adx"] = data["ADX"]
                features["adx_strong_trend"] = (data["ADX"] > 25).astype(int)

            if "CCI" in data.columns:
                features["cci"] = data["CCI"]

            if "Williams_R" in data.columns:
                features["williams_r"] = data["Williams_R"]

            # 出来高特徴量
            if "OBV" in data.columns:
                features["obv_change"] = data["OBV"].pct_change(5)

            if "MFI" in data.columns:
                features["mfi"] = data["MFI"]

            # NaN値を前向き埋め、それでも残る場合は0で埋める
            features = features.fillna(method="ffill").fillna(0)

            logger.info(f"ML特徴量準備完了: {len(features.columns)}特徴量")
            return features

        except Exception as e:
            logger.error(f"ML特徴量準備エラー: {e}")
            return pd.DataFrame()

    def train_trend_prediction_model(
        self, symbol: str, data: pd.DataFrame, features: pd.DataFrame
    ) -> Optional[Dict]:
        """
        トレンド予測モデル訓練

        Args:
            symbol: 銘柄コード
            data: 価格データ
            features: 特徴量データ

        Returns:
            Dict: 訓練結果とモデル情報
        """
        if not SKLEARN_AVAILABLE or features.empty:
            logger.warning("scikit-learn利用不可または特徴量なし")
            return None

        try:
            # ターゲット変数: 5日後のリターン
            target = data["Close"].pct_change(5).shift(-5) * 100  # 5日後のリターン率(%)

            # データの整合性チェック
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_indices]
            y = target[valid_indices]

            if len(X) < 20:  # 最小データ数チェック
                logger.warning(f"データ不足: {len(X)}行 (最小20行必要)")
                return None

            # 訓練・テストデータ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=False
            )

            # 特徴量正規化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # チューニング済みアンサンブルモデル構築
            models = {
                "linear": LinearRegression(),
                "random_forest": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    subsample=0.8,
                    random_state=42,
                ),
                "extra_trees": ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            }

            results = {}
            best_model = None
            best_r2 = -float("inf")

            for model_name, model in models.items():
                try:
                    # モデル訓練
                    if model_name == "linear":
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    # 評価指標計算
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    results[model_name] = {
                        "model": model,
                        "scaler": scaler if model_name == "linear" else None,
                        "r2_score": r2,
                        "mse": mse,
                        "feature_importance": getattr(
                            model, "feature_importances_", None
                        ),
                        "feature_count": len(X.columns)
                        if hasattr(X, "columns")
                        else X.shape[1]
                        if hasattr(X, "shape")
                        else 0,
                    }

                    # 最良モデル更新
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name

                    logger.info(f"{model_name}モデル - R2: {r2:.3f}, MSE: {mse:.3f}")

                except Exception as e:
                    logger.error(f"{model_name}モデル訓練エラー: {e}")
                    continue

            if not results:
                logger.error("全モデル訓練失敗")
                return None

            # モデル性能をDBに保存
            self._save_model_performance(symbol, results)

            # モデルをキャッシュに保存
            self.models[symbol] = results

            model_info = {
                "best_model": best_model,
                "best_r2": best_r2,
                "models": results,
                "feature_count": len(X.columns),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
            }

            logger.info(
                f"トレンド予測モデル訓練完了: {symbol} - 最良モデル: {best_model}"
            )
            return model_info

        except Exception as e:
            logger.error(f"トレンド予測モデル訓練エラー {symbol}: {e}")
            return None

    def _save_model_performance(self, symbol: str, results: Dict):
        """モデル性能をDBに保存"""
        try:
            with sqlite3.connect(self.model_cache_path) as conn:
                for model_type, result in results.items():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO model_performance
                        (symbol, model_type, r2_score, mse, training_date, feature_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symbol,
                            model_type,
                            result["r2_score"],
                            result["mse"],
                            datetime.now().isoformat(),
                            result.get("feature_count", 0),
                        ),
                    )
        except Exception as e:
            logger.error(f"モデル性能保存エラー: {e}")

    def predict_advanced_scores(
        self, symbol: str, data: pd.DataFrame, features: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        高度なMLスコア予測

        Args:
            symbol: 銘柄コード
            data: 価格データ
            features: 特徴量データ

        Returns:
            Tuple[float, float, float]: (トレンドスコア, ボラティリティスコア, パターンスコア)
        """
        try:
            # モデルが訓練済みでない場合は訓練実行
            if symbol not in self.models:
                model_info = self.train_trend_prediction_model(symbol, data, features)
                if model_info is None:
                    logger.warning(f"モデル訓練失敗 - フォールバック使用: {symbol}")
                    return self._calculate_fallback_scores(data)

            models = self.models[symbol]
            latest_features = features.iloc[-1:].fillna(0)  # 最新のデータポイント

            # トレンドスコア（アンサンブル予測）
            trend_predictions = []
            confidences = []

            for model_name, model_data in models.items():
                try:
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    r2_score = model_data["r2_score"]

                    # 予測実行
                    if scaler is not None:  # Linear model
                        scaled_features = scaler.transform(latest_features)
                        pred = model.predict(scaled_features)[0]
                    else:  # Tree-based models
                        pred = model.predict(latest_features)[0]

                    trend_predictions.append(pred)
                    confidences.append(max(0, r2_score))  # R²スコアを信頼度として使用

                except Exception as e:
                    logger.warning(f"{model_name}予測エラー: {e}")
                    continue

            if trend_predictions:
                # 信頼度重み付き平均
                if len(confidences) > 0 and np.sum(confidences) > 0:
                    weighted_pred = np.average(trend_predictions, weights=confidences)
                    avg_confidence = np.mean(confidences)
                else:
                    weighted_pred = np.mean(trend_predictions)
                    avg_confidence = 0.5  # デフォルト信頼度

                # 予測値を0-100スコアに変換
                trend_score = 50 + np.clip(weighted_pred, -50, 50)  # -50%~+50% → 0~100

                # ボラティリティスコア（特徴量ベース）
                volatility_score = self._calculate_volatility_score_advanced(features)

                # パターンスコア（特徴量ベース）
                pattern_score = self._calculate_pattern_score_advanced(features)

                logger.info(
                    f"高度MLスコア計算完了: {symbol} - 信頼度: {avg_confidence:.3f}"
                )
                return (
                    round(float(trend_score), 1),
                    round(float(volatility_score), 1),
                    round(float(pattern_score), 1),
                )

            else:
                logger.warning(f"全予測失敗 - フォールバック使用: {symbol}")
                return self._calculate_fallback_scores(data)

        except Exception as e:
            logger.error(f"高度MLスコア計算エラー {symbol}: {e}")
            return self._calculate_fallback_scores(data)

    def _calculate_volatility_score_advanced(self, features: pd.DataFrame) -> float:
        """高度なボラティリティスコア計算"""
        try:
            latest = features.iloc[-1]

            score = 50.0  # ベースライン

            # 短期・長期ボラティリティ比較
            if (
                "volatility_5d" in features.columns
                and "volatility_20d" in features.columns
            ):
                vol_5d = latest["volatility_5d"]
                vol_20d = latest["volatility_20d"]

                if vol_5d > 0 and vol_20d > 0:
                    vol_ratio = vol_5d / vol_20d
                    if vol_ratio > 1.5:
                        score += 25  # 短期ボラティリティ上昇
                    elif vol_ratio < 0.7:
                        score -= 15  # 短期ボラティリティ低下

            # RSIによるボラティリティ調整
            if "rsi" in features.columns:
                rsi = latest["rsi"]
                if rsi > 80 or rsi < 20:
                    score += 20  # 極端な水準は高ボラティリティ

            # ボリンジャーバンドスクイーズ
            if "bb_squeeze" in features.columns and latest["bb_squeeze"] > 0:
                score += 15  # スクイーズ後の拡張期待

            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"高度ボラティリティスコア計算エラー: {e}")
            return 50.0

    def _calculate_pattern_score_advanced(self, features: pd.DataFrame) -> float:
        """高度なパターンスコア計算"""
        try:
            latest = features.iloc[-1]

            score = 50.0  # ベースライン

            # トレンド強度
            if (
                "adx" in features.columns
                and "adx_strong_trend" in features.columns
                and latest["adx_strong_trend"] > 0
            ):
                score += 15  # 強いトレンド

            # 価格位置
            if "price_sma_ratio" in features.columns:
                price_ratio = latest["price_sma_ratio"]
                if price_ratio > 1.05:
                    score += 10  # 移動平均上
                elif price_ratio < 0.95:
                    score -= 10  # 移動平均下

            # ボリンジャーバンド位置
            if "bb_position" in features.columns:
                bb_pos = latest["bb_position"]
                if bb_pos > 0.8:
                    score += 10  # 上部バンド近く
                elif bb_pos < 0.2:
                    score -= 10  # 下部バンド近く

            # モメンタム
            if "cci" in features.columns:
                cci = latest["cci"]
                if cci > 100:
                    score += 15  # 強い買いモメンタム
                elif cci < -100:
                    score -= 15  # 強い売りモメンタム

            # 出来高確認
            if "obv_change" in features.columns:
                obv_change = latest["obv_change"]
                if obv_change > 0.1:
                    score += 10  # 出来高増加

            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"高度パターンスコア計算エラー: {e}")
            return 50.0

    def _calculate_fallback_scores(
        self, data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """フォールバック用簡易スコア計算"""
        try:
            if data is None or len(data) < 20:
                return (50.0, 50.0, 50.0)

            close_prices = data["Close"]

            # 簡易トレンドスコア
            sma_20 = close_prices.rolling(20).mean()
            current_price = close_prices.iloc[-1]
            sma_current = sma_20.iloc[-1]
            trend_score = 50 + ((current_price - sma_current) / sma_current * 100)
            trend_score = max(0, min(100, trend_score))

            # 簡易ボラティリティスコア
            returns = close_prices.pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            vol_score = min(100, volatility * 1000)  # スケール調整

            # 簡易パターンスコア
            recent_high = data["High"].tail(10).max()
            recent_low = data["Low"].tail(10).min()
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            pattern_score = price_position * 100

            return (
                round(float(trend_score), 1),
                round(float(vol_score), 1),
                round(float(pattern_score), 1),
            )

        except Exception as e:
            logger.error(f"フォールバックスコア計算エラー: {e}")
            return (50.0, 50.0, 50.0)

    def get_model_info(self, symbol: str) -> Optional[Dict]:
        """モデル情報取得"""
        if symbol in self.models:
            models = self.models[symbol]
            return {
                "available_models": list(models.keys()),
                "model_performance": {
                    name: {"r2": data["r2_score"], "mse": data["mse"]}
                    for name, data in models.items()
                },
            }
        return None

    def generate_investment_advice(
        self, symbol: str, data: pd.DataFrame, features: pd.DataFrame
    ) -> Dict:
        """
        機械学習に基づく投資助言生成

        Args:
            symbol: 銘柄コード
            data: 価格データ
            features: 特徴量データ

        Returns:
            Dict: 投資助言情報
        """
        try:
            # MLスコア取得
            trend_score, vol_score, pattern_score = self.predict_advanced_scores(
                symbol, data, features
            )

            # 現在価格情報
            current_price = data["Close"].iloc[-1] if not data.empty else 0
            price_change = (
                (
                    (current_price - data["Close"].iloc[-2])
                    / data["Close"].iloc[-2]
                    * 100
                )
                if len(data) >= 2
                else 0
            )

            # 投資助言判定
            advice = self._calculate_investment_signal(
                trend_score, vol_score, pattern_score, features
            )

            # 信頼度算出
            confidence = self._calculate_confidence(
                trend_score, vol_score, pattern_score, symbol
            )

            # リスク評価
            risk_level = self._calculate_risk_level(vol_score, features)

            return {
                "symbol": symbol,
                "advice": advice["action"],  # "BUY", "SELL", "HOLD"
                "confidence": confidence,  # 0-100
                "risk_level": risk_level,  # "LOW", "MEDIUM", "HIGH"
                "reason": advice["reason"],
                "scores": {
                    "trend": trend_score,
                    "volatility": vol_score,
                    "pattern": pattern_score,
                },
                "market_data": {
                    "current_price": current_price,
                    "price_change_pct": round(price_change, 2),
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"投資助言生成エラー: {e}")
            return self._get_default_advice(symbol)

    def _calculate_investment_signal(
        self,
        trend_score: float,
        vol_score: float,
        pattern_score: float,
        features: pd.DataFrame,
    ) -> Dict:
        """投資シグナル計算"""
        try:
            # 総合スコア算出（重み付き平均）
            composite_score = (
                trend_score * 0.4  # トレンドを重視
                + pattern_score * 0.35  # パターンも重要
                + vol_score * 0.25  # ボラティリティは参考
            )

            # RSI情報の考慮
            rsi = features["rsi"].iloc[-1] if "rsi" in features.columns else 50

            # MACD情報の考慮
            macd_signal = 0
            if "macd" in features.columns and "macd_signal" in features.columns:
                macd = features["macd"].iloc[-1]
                macd_sig = features["macd_signal"].iloc[-1]
                macd_signal = 1 if macd > macd_sig else -1

            # 投資判定ロジック
            if composite_score >= 70:
                if rsi < 80:  # オーバーボート回避
                    return {
                        "action": "BUY",
                        "reason": f"強いトレンド・パターン (スコア: {composite_score:.1f})",
                    }
                else:
                    return {
                        "action": "HOLD",
                        "reason": f"高スコアだがRSIオーバーボート ({rsi:.1f})",
                    }

            elif composite_score <= 30:
                if rsi > 20:  # オーバーソールド回避
                    return {
                        "action": "SELL",
                        "reason": f"弱いトレンド・パターン (スコア: {composite_score:.1f})",
                    }
                else:
                    return {
                        "action": "HOLD",
                        "reason": f"低スコアだがRSIオーバーソールド ({rsi:.1f})",
                    }

            elif composite_score >= 60 and macd_signal > 0:
                return {
                    "action": "BUY",
                    "reason": f"良好スコア+MACDゴールデンクロス (スコア: {composite_score:.1f})",
                }

            elif composite_score <= 40 and macd_signal < 0:
                return {
                    "action": "SELL",
                    "reason": f"低調スコア+MACDデッドクロス (スコア: {composite_score:.1f})",
                }

            else:
                return {
                    "action": "HOLD",
                    "reason": f"中立的な市況 (スコア: {composite_score:.1f})",
                }

        except Exception as e:
            logger.error(f"投資シグナル計算エラー: {e}")
            return {"action": "HOLD", "reason": "計算エラーのため保留"}

    def _calculate_confidence(
        self, trend_score: float, vol_score: float, pattern_score: float, symbol: str
    ) -> float:
        """信頼度計算"""
        try:
            # スコアの一貫性をチェック
            scores = [trend_score, vol_score, pattern_score]
            score_std = np.std(scores)

            # 一貫性が高いほど信頼度UP
            consistency = max(0, 100 - score_std * 2)

            # 極端なスコア（0付近、100付近）は信頼度DOWN
            extremity_penalty = 0
            for score in scores:
                if score < 10 or score > 90:
                    extremity_penalty += 10

            # モデル性能による調整
            model_bonus = 0
            if symbol in self.models and self.models[symbol]:
                # R²スコアの平均で判定
                r2_scores = [m.get("r2_score", 0) for m in self.models[symbol].values()]
                avg_r2 = np.mean(r2_scores) if r2_scores else 0
                model_bonus = max(0, avg_r2 * 20)  # 最大20ポイント

            confidence = consistency + model_bonus - extremity_penalty
            return max(0, min(100, confidence))

        except Exception as e:
            logger.error(f"信頼度計算エラー: {e}")
            return 50.0

    def _calculate_risk_level(self, vol_score: float, features: pd.DataFrame) -> str:
        """リスクレベル計算"""
        try:
            risk_factors = []

            # ボラティリティリスク
            if vol_score > 80:
                risk_factors.append("高ボラティリティ")
            elif vol_score < 20:
                risk_factors.append("低ボラティリティ")

            # RSIリスク
            if "rsi" in features.columns:
                rsi = features["rsi"].iloc[-1]
                if rsi > 80:
                    risk_factors.append("オーバーボート")
                elif rsi < 20:
                    risk_factors.append("オーバーソールド")

            # リスクレベル判定
            if len(risk_factors) >= 2:
                return "HIGH"
            elif len(risk_factors) == 1:
                return "MEDIUM"
            else:
                return "LOW"

        except Exception:
            return "MEDIUM"

    def _get_default_advice(self, symbol: str) -> Dict:
        """デフォルト助言"""
        return {
            "symbol": symbol,
            "advice": "HOLD",
            "confidence": 50.0,
            "risk_level": "MEDIUM",
            "reason": "データ不足のため保留",
            "scores": {"trend": 50.0, "volatility": 50.0, "pattern": 50.0},
            "market_data": {"current_price": 0, "price_change_pct": 0},
            "timestamp": datetime.now().isoformat(),
        }

    def tune_hyperparameters(
        self, symbol: str, data: pd.DataFrame, features: pd.DataFrame
    ) -> Optional[Dict]:
        """
        ハイパーパラメータチューニング実行

        Args:
            symbol: 銘柄コード
            data: 価格データ
            features: 特徴量データ

        Returns:
            Dict: チューニング結果
        """
        if not SKLEARN_AVAILABLE or features.empty:
            logger.warning("scikit-learn利用不可または特徴量なし - チューニング不可")
            return None

        try:
            # ターゲット変数: 5日後のリターン
            target = data["Close"].pct_change(5).shift(-5) * 100

            # データの整合性チェック
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_indices]
            y = target[valid_indices]

            if len(X) < 50:  # チューニングには更に多くのデータが必要
                logger.warning(f"チューニング用データ不足: {len(X)}行 (最小50行必要)")
                return None

            # パラメータグリッド定義
            param_grids = {
                "random_forest": {
                    "n_estimators": [100, 150, 200],
                    "max_depth": [8, 10, 12],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
                "gradient_boosting": {
                    "n_estimators": [100, 150, 200],
                    "learning_rate": [0.05, 0.1, 0.15],
                    "max_depth": [4, 6, 8],
                    "min_samples_split": [5, 10, 15],
                    "subsample": [0.8, 0.9, 1.0],
                },
            }

            tuning_results = {}

            logger.info(f"ハイパーパラメータチューニング開始: {symbol}")

            for model_name, param_grid in param_grids.items():
                try:
                    logger.info(f"{model_name}モデルのチューニング開始...")

                    if model_name == "random_forest":
                        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
                    elif model_name == "gradient_boosting":
                        base_model = GradientBoostingRegressor(random_state=42)
                    else:
                        continue

                    # グリッドサーチ実行（3-fold CV）
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0
                    )

                    grid_search.fit(X, y)

                    # 結果保存
                    tuning_results[model_name] = {
                        "best_params": grid_search.best_params_,
                        "best_score": grid_search.best_score_,
                        "best_model": grid_search.best_estimator_,
                    }

                    logger.info(
                        f"{model_name}チューニング完了 - 最良スコア: {grid_search.best_score_:.3f}"
                    )

                except Exception as e:
                    logger.error(f"{model_name}チューニングエラー: {e}")
                    continue

            if tuning_results:
                # チューニング結果を使ってモデルを更新
                self._update_models_with_tuned_params(symbol, tuning_results, X, y)

                logger.info(f"ハイパーパラメータチューニング完了: {symbol}")
                return tuning_results
            else:
                logger.warning(f"全モデルチューニング失敗: {symbol}")
                return None

        except Exception as e:
            logger.error(f"ハイパーパラメータチューニングエラー {symbol}: {e}")
            return None

    def _update_models_with_tuned_params(
        self, symbol: str, tuning_results: Dict, X: pd.DataFrame, y: pd.Series
    ):
        """チューニング結果でモデルを更新"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=False
            )

            updated_models = {}

            for model_name, tuning_data in tuning_results.items():
                best_model = tuning_data["best_model"]

                # テストデータで評価
                y_pred = best_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                updated_models[f"{model_name}_tuned"] = {
                    "model": best_model,
                    "scaler": None,
                    "r2_score": r2,
                    "mse": mse,
                    "mae": mae,
                    "best_params": tuning_data["best_params"],
                    "feature_importance": getattr(
                        best_model, "feature_importances_", None
                    ),
                }

                logger.info(
                    f"チューニング済み{model_name} - R2: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}"
                )

            # 既存モデルに追加
            if symbol not in self.models:
                self.models[symbol] = {}
            self.models[symbol].update(updated_models)

            # DBに保存
            self._save_model_performance(symbol, updated_models)

        except Exception as e:
            logger.error(f"チューニング済みモデル更新エラー: {e}")

    def evaluate_model_ensemble(
        self, symbol: str, data: pd.DataFrame, features: pd.DataFrame
    ) -> Optional[Dict]:
        """
        モデルアンサンブルの総合評価

        Returns:
            Dict: アンサンブル評価結果
        """
        if symbol not in self.models or features.empty:
            return None

        try:
            # ターゲット変数準備
            target = data["Close"].pct_change(5).shift(-5) * 100
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_indices]
            y = target[valid_indices]

            if len(X) < 20:
                return None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=False
            )

            models = self.models[symbol]
            ensemble_predictions = []
            individual_scores = {}

            # 各モデルの予測とスコア
            for model_name, model_data in models.items():
                try:
                    model = model_data["model"]
                    scaler = model_data.get("scaler")

                    if scaler is not None:
                        X_test_processed = scaler.transform(X_test)
                    else:
                        X_test_processed = X_test

                    y_pred = model.predict(X_test_processed)
                    ensemble_predictions.append(y_pred)

                    # 個別評価
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    individual_scores[model_name] = {
                        "r2_score": r2,
                        "mse": mse,
                        "mae": mae,
                        "weight": max(0, r2),  # 負のR²は0重み
                    }

                except Exception as e:
                    logger.warning(f"{model_name}評価エラー: {e}")
                    continue

            if len(ensemble_predictions) == 0:
                return None

            # アンサンブル予測（重み付き平均）
            weights = [individual_scores[name]["weight"] for name in individual_scores]
            if len(weights) > 0 and sum(weights) > 0:
                weights = np.array(weights) / sum(weights)
                ensemble_pred = np.average(
                    ensemble_predictions, weights=weights, axis=0
                )
            else:
                ensemble_pred = np.mean(ensemble_predictions, axis=0)

            # アンサンブル評価
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

            evaluation_result = {
                "ensemble_performance": {
                    "r2_score": ensemble_r2,
                    "mse": ensemble_mse,
                    "mae": ensemble_mae,
                },
                "individual_performance": individual_scores,
                "model_count": len(individual_scores),
                "best_individual_model": max(
                    individual_scores.keys(),
                    key=lambda x: individual_scores[x]["r2_score"],
                ),
                "ensemble_improvement": ensemble_r2
                - max(s["r2_score"] for s in individual_scores.values()),
            }

            logger.info(
                f"アンサンブル評価完了: {symbol} - アンサンブルR2: {ensemble_r2:.3f}"
            )
            return evaluation_result

        except Exception as e:
            logger.error(f"アンサンブル評価エラー {symbol}: {e}")
            return None


if __name__ == "__main__":
    # テスト実行
    from .real_market_data import RealMarketDataManager

    print("=== Advanced ML Engine チューニングテスト ===")

    # データ取得
    data_manager = RealMarketDataManager()
    ml_engine = AdvancedMLEngine()

    test_symbol = "7203"
    stock_data = data_manager.get_stock_data(test_symbol)

    if stock_data is not None:
        print(f"データ期間: {len(stock_data)}日")

        # 高度な技術指標計算
        advanced_data = ml_engine.calculate_advanced_technical_indicators(stock_data)
        print(f"高度指標計算完了: {len(advanced_data.columns)}列")

        # ML特徴量準備
        features = ml_engine.prepare_ml_features(advanced_data)
        print(f"ML特徴量: {len(features.columns)}特徴量")

        # 初期モデル訓練とスコア予測
        print("\n=== 初期モデル ===")
        trend_score, vol_score, pattern_score = ml_engine.predict_advanced_scores(
            test_symbol, stock_data, features
        )

        print("初期MLスコア:")
        print(f"  トレンド: {trend_score}")
        print(f"  ボラティリティ: {vol_score}")
        print(f"  パターン: {pattern_score}")

        # 初期モデル情報表示
        model_info = ml_engine.get_model_info(test_symbol)
        if model_info:
            print(f"初期モデル: {model_info['available_models']}")
            for name, perf in model_info["model_performance"].items():
                print(f"  {name}: R²={perf['r2']:.3f}, MSE={perf['mse']:.3f}")

        # ハイパーパラメータチューニング実行
        if len(features) >= 50:  # チューニング実行条件
            print("\n=== ハイパーパラメータチューニング ===")
            tuning_results = ml_engine.tune_hyperparameters(
                test_symbol, stock_data, features
            )

            if tuning_results:
                for model_name, results in tuning_results.items():
                    print(f"チューニング結果 - {model_name}:")
                    print(f"  最良パラメータ: {results['best_params']}")
                    print(f"  CVスコア: {results['best_score']:.3f}")

                # チューニング後のモデル情報表示
                updated_model_info = ml_engine.get_model_info(test_symbol)
                if updated_model_info:
                    print(
                        f"チューニング後モデル: {updated_model_info['available_models']}"
                    )
                    for name, perf in updated_model_info["model_performance"].items():
                        print(f"  {name}: R²={perf['r2']:.3f}, MSE={perf['mse']:.3f}")

                # アンサンブル評価
                print("\n=== アンサンブル評価 ===")
                ensemble_eval = ml_engine.evaluate_model_ensemble(
                    test_symbol, stock_data, features
                )
                if ensemble_eval:
                    print("アンサンブル性能:")
                    ens_perf = ensemble_eval["ensemble_performance"]
                    print(f"  R²: {ens_perf['r2_score']:.3f}")
                    print(f"  MSE: {ens_perf['mse']:.3f}")
                    print(f"  MAE: {ens_perf['mae']:.3f}")
                    print(f"最良単体モデル: {ensemble_eval['best_individual_model']}")
                    print(
                        f"アンサンブル改善: {ensemble_eval['ensemble_improvement']:+.3f}"
                    )

                # チューニング後の最終スコア
                print("\n=== チューニング後MLスコア ===")
                (
                    final_trend,
                    final_vol,
                    final_pattern,
                ) = ml_engine.predict_advanced_scores(test_symbol, stock_data, features)
                print("最終MLスコア:")
                print(f"  トレンド: {final_trend}")
                print(f"  ボラティリティ: {final_vol}")
                print(f"  パターン: {final_pattern}")

            else:
                print("チューニング実行できませんでした")
        else:
            print(
                f"\nチューニング実行には50行以上のデータが必要です（現在: {len(features)}行）"
            )

    else:
        print("データ取得に失敗しました")


class ParallelMLEngine:
    """
    並列ML処理エンジン

    複数銘柄のML分析を並列実行してパフォーマンス向上
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.ml_engine = AdvancedMLEngine()

    def analyze_multiple_symbols_parallel(
        self,
        stock_data_dict: Dict[str, pd.DataFrame],
        max_workers: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        複数銘柄のML分析を並列実行

        Args:
            stock_data_dict: {symbol: DataFrame} の辞書
            max_workers: 並列ワーカー数（None時はデフォルト使用）

        Returns:
            Dict[symbol, analysis_result]: 分析結果辞書
        """
        workers = max_workers or self.max_workers
        results = {}

        logger.info(f"並列ML分析開始: {len(stock_data_dict)}銘柄 (workers={workers})")

        def analyze_single_symbol(symbol_data_tuple):
            """単一銘柄の分析を実行"""
            symbol, data = symbol_data_tuple
            try:
                if data.empty:
                    return symbol, self.ml_engine._get_default_advice(symbol)

                # 特徴量準備
                features = self.ml_engine.prepare_ml_features(data)

                # ML予測実行
                (
                    trend_score,
                    vol_score,
                    pattern_score,
                ) = self.ml_engine.predict_advanced_scores(symbol, data, features)

                # 投資助言生成
                advice = self.ml_engine.generate_investment_advice(
                    symbol, data, features
                )

                return symbol, advice

            except Exception as e:
                logger.error(f"並列ML分析エラー ({symbol}): {e}")
                return symbol, self.ml_engine._get_default_advice(symbol)

        # 並列実行
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # タスク投入
            future_to_symbol = {
                executor.submit(analyze_single_symbol, item): item[0]
                for item in stock_data_dict.items()
            }

            # 結果収集
            completed_count = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result_symbol, analysis_result = future.result(
                        timeout=30
                    )  # 30秒タイムアウト
                    results[result_symbol] = analysis_result
                    completed_count += 1

                    if completed_count % 10 == 0:  # 10銘柄ごとに進捗表示
                        logger.info(
                            f"並列ML分析進捗: {completed_count}/{len(stock_data_dict)}"
                        )

                except Exception as e:
                    logger.error(f"並列処理結果取得エラー ({symbol}): {e}")
                    results[symbol] = self.ml_engine._get_default_advice(symbol)

        logger.info(f"並列ML分析完了: {len(results)}/{len(stock_data_dict)}銘柄")
        return results

    def batch_analyze_with_timing(
        self, stock_data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, Dict], float]:
        """
        タイミング測定付きバッチ分析

        Returns:
            Tuple[results, execution_time]: 分析結果と実行時間
        """
        import time

        start_time = time.time()
        results = self.analyze_multiple_symbols_parallel(stock_data_dict)
        execution_time = time.time() - start_time

        logger.info(f"バッチML分析完了: {execution_time:.2f}秒 ({len(results)}銘柄)")
        logger.info(f"平均処理時間: {execution_time/len(results):.3f}秒/銘柄")

        return results, execution_time
