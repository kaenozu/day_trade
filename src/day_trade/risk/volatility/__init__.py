#!/usr/bin/env python3
"""
Volatility Prediction Engine Package

ボラティリティ予測エンジンパッケージ

このパッケージは、高度なボラティリティ予測機能を提供します:

主要コンポーネント:
- RealizedVolatilityCalculator: 実現ボラティリティ計算
- GARCHModelEngine: GARCHモデル適合・予測
- VIXIndicatorCalculator: VIX風指標計算
- MLVolatilityPredictor: 機械学習ボラティリティ予測
- VolatilityEnsembleEngine: 統合アンサンブル予測
- VolatilityRiskAssessor: リスク評価・投資示唆
- VolatilityPredictionEngine: 統合インターフェース（バックワード互換性）

使用例:
    from day_trade.risk.volatility import VolatilityPredictionEngine
    
    engine = VolatilityPredictionEngine()
    forecast = engine.generate_comprehensive_volatility_forecast(data, "7203")
"""

from .base import VolatilityEngineBase, ARCH_AVAILABLE, SKLEARN_AVAILABLE
from .realized_volatility import RealizedVolatilityCalculator
from .garch_models import GARCHModelEngine
from .vix_indicator import VIXIndicatorCalculator
from .ml_features import MLFeatureGenerator
from .ml_models import MLVolatilityPredictor
from .ensemble import VolatilityEnsembleEngine
from .risk_assessment import VolatilityRiskAssessor

# バックワード互換性のための統合クラス
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class VolatilityPredictionEngine:
    """
    統合ボラティリティ予測エンジン（バックワード互換性）
    
    元のVolatilityPredictionEngineクラスと同等のインターフェースを提供し、
    内部的には新しいモジュール化されたコンポーネントを使用します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        self.model_cache_dir = model_cache_dir
        
        # 各コンポーネントの初期化
        self.rv_calculator = RealizedVolatilityCalculator(model_cache_dir)
        self.garch_engine = GARCHModelEngine(model_cache_dir)
        self.vix_calculator = VIXIndicatorCalculator(model_cache_dir)
        self.ml_predictor = MLVolatilityPredictor(model_cache_dir)
        self.ensemble_engine = VolatilityEnsembleEngine(model_cache_dir)
        self.risk_assessor = VolatilityRiskAssessor(model_cache_dir)
        
        # 元のクラスとの互換性のための属性
        self.garch_models = self.garch_engine.garch_models
        self.ml_models = self.ml_predictor.ml_models
        self.scalers = self.ml_predictor.scalers
        self.vix_params = self.vix_calculator.get_default_vix_params()
        
        logger.info("統合ボラティリティ予測エンジン初期化完了")

    def calculate_realized_volatility(self, *args, **kwargs):
        """実現ボラティリティ計算（バックワード互換性）"""
        return self.rv_calculator.calculate_realized_volatility(*args, **kwargs)

    def fit_garch_model(self, *args, **kwargs):
        """GARCHモデル適合（バックワード互換性）"""
        return self.garch_engine.fit_garch_model(*args, **kwargs)

    def predict_garch_volatility(self, *args, **kwargs):
        """GARCH予測（バックワード互換性）"""
        return self.garch_engine.predict_garch_volatility(*args, **kwargs)

    def calculate_vix_like_indicator(self, *args, **kwargs):
        """VIX風指標計算（バックワード互換性）"""
        return self.vix_calculator.calculate_vix_like_indicator(*args, **kwargs)

    def prepare_ml_features_for_volatility(self, *args, **kwargs):
        """ML特徴量準備（バックワード互換性）"""
        feature_generator = MLFeatureGenerator(self.model_cache_dir)
        return feature_generator.prepare_ml_features_for_volatility(*args, **kwargs)

    def train_volatility_ml_model(self, *args, **kwargs):
        """ML訓練（バックワード互換性）"""
        return self.ml_predictor.train_volatility_ml_model(*args, **kwargs)

    def predict_volatility_ml(self, *args, **kwargs):
        """ML予測（バックワード互換性）"""
        return self.ml_predictor.predict_volatility_ml(*args, **kwargs)

    def create_volatility_regime_classifier(self, *args, **kwargs):
        """ボラティリティレジーム分類（バックワード互換性）"""
        return self.rv_calculator.create_volatility_regime_classifier(*args, **kwargs)

    def generate_comprehensive_volatility_forecast(self, *args, **kwargs):
        """総合ボラティリティ予測（バックワード互換性）"""
        # アンサンブルエンジンに委譲
        result = self.ensemble_engine.generate_comprehensive_volatility_forecast(*args, **kwargs)
        
        # リスク評価を追加（元のクラスとの互換性）
        if "error" not in result and "current_metrics" in result:
            risk_metrics = self.risk_assessor.calculate_volatility_risk_metrics(result)
            result.update(risk_metrics)
        
        return result

    # 新しいメソッドの追加（拡張機能）
    def get_volatility_statistics(self, *args, **kwargs):
        """ボラティリティ統計（新機能）"""
        return self.rv_calculator.get_volatility_statistics(*args, **kwargs)

    def get_vix_statistics(self, *args, **kwargs):
        """VIX統計（新機能）"""
        return self.vix_calculator.calculate_vix_statistics(*args, **kwargs)

    def get_model_diagnostics(self, symbol: str, model_type: str = "ml"):
        """モデル診断（新機能）"""
        if model_type == "ml":
            return self.ml_predictor.get_model_diagnostics(symbol)
        elif model_type == "garch":
            return self.garch_engine.get_model_diagnostics(symbol)
        else:
            logger.error(f"サポートされていないモデルタイプ: {model_type}")
            return None

    def optimize_ensemble_weights(self, *args, **kwargs):
        """アンサンブル重み最適化（新機能）"""
        return self.ensemble_engine.optimize_ensemble_weights(*args, **kwargs)

    def get_ensemble_summary(self, *args, **kwargs):
        """アンサンブルサマリー（新機能）"""
        return self.ensemble_engine.get_ensemble_summary(*args, **kwargs)


# パッケージレベルでのエクスポート
__all__ = [
    # 基底クラス
    "VolatilityEngineBase",
    
    # 個別コンポーネント
    "RealizedVolatilityCalculator",
    "GARCHModelEngine", 
    "VIXIndicatorCalculator",
    "MLFeatureGenerator",
    "MLVolatilityPredictor",
    "VolatilityEnsembleEngine",
    "VolatilityRiskAssessor",
    
    # 統合インターフェース
    "VolatilityPredictionEngine",
    
    # 依存関係チェック
    "ARCH_AVAILABLE",
    "SKLEARN_AVAILABLE",
]

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Day Trade Sub Development Team"
__description__ = "Advanced volatility prediction engine with GARCH, ML, and ensemble methods"

# モジュール初期化時のログ
logger.info(f"Volatility Prediction Package loaded - Version: {__version__}")
if not ARCH_AVAILABLE:
    logger.warning("ARCH package not available - GARCH functionality will be limited")
if not SKLEARN_AVAILABLE:
    logger.warning("scikit-learn not available - ML functionality will be limited")