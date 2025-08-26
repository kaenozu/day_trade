"""
市場レジーム検出モジュール
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble_market_regime")


class MarketRegimeDetector:
    """市場レジーム検出クラス"""

    def __init__(self, max_regime_history: int = 10):
        """
        Args:
            max_regime_history: 保持する履歴の最大数
        """
        self.max_regime_history = max_regime_history
        self.current_market_regime = "unknown"
        self.regime_history: List[str] = []

    def detect_market_regime(
        self,
        df: pd.DataFrame,
        indicators: Optional[pd.DataFrame] = None,
        meta_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        市場レジームを検出
        
        Args:
            df: 価格データのDataFrame
            indicators: テクニカル指標のDataFrame
            meta_features: メタ特徴量
            
        Returns:
            検出されたレジーム名
        """
        try:
            if len(df) < 50:
                return "insufficient_data"

            regime = self._analyze_regime(df, indicators, meta_features)
            
            # レジーム履歴を更新
            if len(self.regime_history) >= self.max_regime_history:
                self.regime_history.pop(0)
            self.regime_history.append(regime)
            self.current_market_regime = regime

            return regime

        except Exception as e:
            logger.error(f"市場レジーム検出エラー: {e}")
            return "unknown"

    def _analyze_regime(
        self,
        df: pd.DataFrame,
        indicators: Optional[pd.DataFrame] = None,
        meta_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """レジーム分析のメイン処理"""
        
        # ボラティリティベースのレジーム検出
        volatility_regime = self._detect_volatility_regime(df)
        
        # トレンドベースのレジーム検出  
        trend_regime = self._detect_trend_regime(df)
        
        # RSI/オシレーターベースのレジーム検出
        oscillator_regime = self._detect_oscillator_regime(df, indicators, meta_features)
        
        # 複合的なレジーム判定
        return self._combine_regimes(volatility_regime, trend_regime, oscillator_regime)

    def _detect_volatility_regime(self, df: pd.DataFrame) -> str:
        """ボラティリティベースのレジーム検出"""
        try:
            returns = df["Close"].pct_change().dropna()
            if len(returns) < 20:
                return "unknown"
                
            current_vol = returns.tail(20).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            if vol_ratio > 1.5:
                return "high_volatility"
            elif vol_ratio < 0.7:
                return "low_volatility"
            else:
                return "normal_volatility"
                
        except Exception as e:
            logger.warning(f"ボラティリティレジーム検出エラー: {e}")
            return "unknown"

    def _detect_trend_regime(self, df: pd.DataFrame) -> str:
        """トレンドベースのレジーム検出"""
        try:
            if len(df) < 50:
                return "unknown"
                
            sma_20 = df["Close"].rolling(20).mean()
            sma_50 = df["Close"].rolling(50).mean()
            
            if len(sma_20) == 0 or len(sma_50) == 0:
                return "unknown"
                
            trend_ratio = sma_20.iloc[-1] / sma_50.iloc[-1]
            
            # トレンドの強さを判定
            if trend_ratio > 1.05:
                return "uptrend"
            elif trend_ratio < 0.95:
                return "downtrend" 
            else:
                return "sideways"
                
        except Exception as e:
            logger.warning(f"トレンドレジーム検出エラー: {e}")
            return "unknown"

    def _detect_oscillator_regime(
        self,
        df: pd.DataFrame,
        indicators: Optional[pd.DataFrame] = None,
        meta_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """オシレーターベースのレジーム検出"""
        try:
            # メタ特徴量からRSIレベルを取得
            rsi_level = None
            if meta_features and "rsi_level" in meta_features:
                rsi_level = meta_features["rsi_level"]
            elif indicators is not None and "RSI" in indicators.columns:
                rsi_level = indicators["RSI"].iloc[-1]
            
            if rsi_level is None:
                return "unknown"
                
            if rsi_level > 70:
                return "overbought"
            elif rsi_level < 30:
                return "oversold"
            else:
                return "neutral"
                
        except Exception as e:
            logger.warning(f"オシレーターレジーム検出エラー: {e}")
            return "unknown"

    def _combine_regimes(
        self, volatility_regime: str, trend_regime: str, oscillator_regime: str
    ) -> str:
        """複数のレジーム分析結果を組み合わせる"""
        try:
            # 高ボラティリティの場合は最優先
            if volatility_regime == "high_volatility":
                if oscillator_regime == "overbought":
                    return "high_vol_overbought"
                elif oscillator_regime == "oversold":
                    return "high_vol_oversold"
                else:
                    return "high_volatility"
            
            # 低ボラティリティの場合
            elif volatility_regime == "low_volatility":
                return "low_volatility"
            
            # 通常ボラティリティの場合はトレンドを重視
            else:
                if trend_regime in ["uptrend", "downtrend"]:
                    return trend_regime
                elif trend_regime == "sideways":
                    return "sideways"
                else:
                    return "unknown"
                    
        except Exception as e:
            logger.warning(f"レジーム組み合わせエラー: {e}")
            return "unknown"

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """レジームに基づく戦略重みを取得"""
        weights = {
            "conservative_rsi": 0.2,
            "aggressive_momentum": 0.25,
            "trend_following": 0.25,
            "mean_reversion": 0.2,
            "default_integrated": 0.1,
        }
        
        try:
            if regime == "high_volatility":
                # 高ボラティリティ時は保守的戦略を重視
                weights.update({
                    "conservative_rsi": 0.4,
                    "aggressive_momentum": 0.1,
                    "trend_following": 0.2,
                    "mean_reversion": 0.2,
                    "default_integrated": 0.1,
                })
            elif regime in ["uptrend", "downtrend"]:
                # トレンド相場ではトレンドフォロー戦略を重視
                weights.update({
                    "conservative_rsi": 0.1,
                    "aggressive_momentum": 0.3,
                    "trend_following": 0.4,
                    "mean_reversion": 0.1,
                    "default_integrated": 0.1,
                })
            elif regime == "sideways":
                # レンジ相場では平均回帰戦略を重視
                weights.update({
                    "conservative_rsi": 0.2,
                    "aggressive_momentum": 0.1,
                    "trend_following": 0.1,
                    "mean_reversion": 0.5,
                    "default_integrated": 0.1,
                })
            elif regime in ["high_vol_overbought", "high_vol_oversold"]:
                # 高ボラティリティ+極値では保守的重視
                weights.update({
                    "conservative_rsi": 0.5,
                    "aggressive_momentum": 0.05,
                    "trend_following": 0.15,
                    "mean_reversion": 0.25,
                    "default_integrated": 0.05,
                })
            # その他の場合はデフォルト重み（balanced）を使用
                
        except Exception as e:
            logger.warning(f"レジーム重み取得エラー: {e}")
        
        return weights

    def get_regime_confidence_threshold(self, regime: str) -> float:
        """レジームに基づく信頼度閾値を取得"""
        try:
            if regime == "high_volatility":
                return 65.0
            elif regime in ["uptrend", "downtrend"]:
                return 40.0
            elif regime in ["high_vol_overbought", "high_vol_oversold"]:
                return 70.0  # 極値では高い閾値
            elif regime == "low_volatility":
                return 35.0  # 低ボラでは機会を逃さない
            else:
                return 50.0  # デフォルト
                
        except Exception as e:
            logger.warning(f"レジーム閾値取得エラー: {e}")
            return 50.0

    def get_regime_summary(self) -> Dict[str, Any]:
        """レジーム情報のサマリーを取得"""
        return {
            "current_regime": self.current_market_regime,
            "regime_history": self.regime_history.copy(),
            "regime_stability": self._calculate_regime_stability(),
            "regime_duration": self._calculate_regime_duration(),
        }

    def _calculate_regime_stability(self) -> float:
        """レジームの安定性を計算（同じレジームの連続性）"""
        if len(self.regime_history) < 2:
            return 0.0
            
        try:
            current_regime = self.regime_history[-1]
            consecutive_count = 0
            
            for regime in reversed(self.regime_history):
                if regime == current_regime:
                    consecutive_count += 1
                else:
                    break
                    
            return consecutive_count / len(self.regime_history)
            
        except Exception as e:
            logger.warning(f"レジーム安定性計算エラー: {e}")
            return 0.0

    def _calculate_regime_duration(self) -> int:
        """現在のレジームの継続期間を計算"""
        if not self.regime_history:
            return 0
            
        try:
            current_regime = self.regime_history[-1]
            duration = 0
            
            for regime in reversed(self.regime_history):
                if regime == current_regime:
                    duration += 1
                else:
                    break
                    
            return duration
            
        except Exception as e:
            logger.warning(f"レジーム継続期間計算エラー: {e}")
            return 0

    def reset_history(self) -> None:
        """レジーム履歴をリセット"""
        self.regime_history.clear()
        self.current_market_regime = "unknown"
        logger.info("レジーム履歴をリセットしました")