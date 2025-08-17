import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from simple_ml_prediction_system import SimpleMLPredictionSystem
from advanced_ml_prediction_system import AdvancedMLPredictionSystem

logger = logging.getLogger('daytrade')

@dataclass
class PredictionResult:
    confidence: float
    score: float
    signal: str
    risk_level: str
    ml_source: str
    backtest_score: Optional[float] = None
    model_consensus: Optional[Dict[str, Any]] = None
    feature_importance: Optional[List[str]] = None

class MLService:
    def __init__(self):
        self.ml_system = None
        self.ml_type = "none"
        self.ml_available = False

        try:
            self.ml_system = SimpleMLPredictionSystem()
            self.ml_type = "simple"
            self.ml_available = True
            logger.info("[OK] ML予測システム: シンプル版有効化")
        except ImportError:
            try:
                self.ml_system = AdvancedMLPredictionSystem()
                self.ml_type = "advanced"
                self.ml_available = True
                logger.info("[OK] ML予測システム: 高度版有効化")
            except ImportError:
                logger.warning("[WARNING] ML予測システム未対応 - フォールバックモード")

    async def get_prediction(self, symbol: str, use_random_fallback: bool = True) -> PredictionResult:
        if not self.ml_available:
            logger.warning(f"MLシステムが利用できません。フォールバックを適用します。 (Symbol: {symbol})")
            return self._get_fallback_prediction(symbol, use_random_fallback)

        try:
            if hasattr(self.ml_system, 'predict_symbol_movement'):
                prediction_result = await self.ml_system.predict_symbol_movement(symbol)
            else:
                raise Exception("ML prediction method not available in selected ML system")

            # シグナル強度計算（本番運用版）
            base_confidence = prediction_result.confidence * 100
            signal = self._determine_signal(prediction_result.prediction, base_confidence)
            risk_level = self._determine_risk_level(base_confidence, prediction_result.feature_values.get('volatility', 0.5))

            return PredictionResult(
                confidence=base_confidence,
                score=min(95, base_confidence + np.random.uniform(-3, 7)),  # 微小ランダム性
                signal=signal,
                risk_level=risk_level,
                ml_source=f"advanced_ml_{self.ml_type}",
                model_consensus=prediction_result.model_consensus,
                feature_importance=list(prediction_result.feature_values.keys())[:3]
            )

        except Exception as e:
            logger.error(f"ML予測エラー ({symbol}): {e}")
            return self._get_fallback_prediction(symbol, use_random_fallback)

    def _get_fallback_prediction(self, symbol: str, use_random_fallback: bool) -> PredictionResult:
        if use_random_fallback:
            logger.info(f"ランダムフォールバックを適用します。 (Symbol: {symbol})")
            np.random.seed(hash(symbol) % 1000)  # 銘柄コードでシード固定
            confidence = np.random.uniform(65, 85)
            signal_rand = np.random.random()
            if signal_rand > 0.7:
                signal = '買い'
            elif signal_rand > 0.4:
                signal = '検討'
            else:
                signal = '様子見'
            risk_level = '中' if confidence > 75 else '高'
            ml_source = 'random_fallback'
        else:
            logger.warning(f"非ランダムフォールバックを適用します。 (Symbol: {symbol})")
            confidence = 50.0 # 中立的な信頼度
            signal = '様子見' # 安全なシグナル
            risk_level = '中' # 中立的なリスク
            ml_source = 'non_random_fallback'

        return PredictionResult(
            confidence=confidence,
            score=confidence + np.random.uniform(-5, 10),
            signal=signal,
            risk_level=risk_level,
            ml_source=ml_source,
            backtest_score=None,
            model_consensus=None,
            feature_importance=None
        )

    def _determine_signal(self, prediction_value: int, confidence: float) -> str:
        if prediction_value == 1:  # 上昇予測
            if confidence > 85:
                return '強い買い'
            elif confidence > 75:
                return '買い'
            else:
                return '検討'
        else:  # 下降予測
            if confidence > 85:
                return '強い売り'
            elif confidence > 75:
                return '売り'
            else:
                return '様子見'

    def _determine_risk_level(self, confidence: float, volatility_risk: float) -> str:
        if volatility_risk > 0.7 or confidence < 70:
            return '高'
        elif volatility_risk > 0.4 or confidence < 80:
            return '中'
        else:
            return '低'
