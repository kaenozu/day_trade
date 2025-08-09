#!/usr/bin/env python3
"""
Volatility Prediction System
Issue #315 Phase 4: ボラティリティ予測システム実装

GARCH モデル、VIX指標、動的リスク調整による
最大ドローダウン20%削減・シャープレシオ0.3向上システム
"""

import asyncio
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from ..data.advanced_parallel_ml_engine import AdvancedParallelMLEngine
    from ..utils.logging_config import get_context_logger
    from ..utils.performance_monitor import PerformanceMonitor
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
        return f"volatility_{hash(str(args) + str(kwargs))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 依存関係チェック
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未インストール - ML機能制限")

try:
    from arch import arch_model
    from arch.univariate import EGARCH, GARCH, GJR

    ARCH_AVAILABLE = True
    logger.info("arch利用可能 - GARCH モデル機能有効")
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch未インストール - GARCH機能制限（pip install arch）")


@dataclass
class GARCHVolatilityResult:
    """GARCH ボラティリティ予測結果"""

    symbol: str
    current_volatility: float
    forecast_volatility: float
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    confidence_interval: Tuple[float, float]
    model_type: str  # 'GARCH', 'EGARCH', 'GJR'
    model_params: Dict[str, float]
    forecast_horizon: int
    processing_time: float


@dataclass
class VIXRiskAssessment:
    """VIX リスク評価結果"""

    symbol: str
    vix_level: float
    risk_regime: str  # 'low', 'normal', 'high', 'extreme'
    market_fear_indicator: float  # 0-1スケール
    correlation_with_vix: float
    risk_adjustment_factor: float  # ポジションサイズ調整係数
    recommended_action: str  # 'increase', 'maintain', 'reduce', 'avoid'
    processing_time: float


@dataclass
class DynamicRiskMetrics:
    """動的リスク指標"""

    symbol: str
    current_var: float  # Value at Risk (1日、95%)
    expected_shortfall: float  # Conditional VaR
    max_drawdown_forecast: float
    sharpe_ratio_forecast: float
    position_size_multiplier: float  # 基準ポジションサイズへの乗数
    risk_budget_allocation: float  # リスク予算配分 (0-1)
    stop_loss_level: float
    take_profit_level: float
    processing_time: float


@dataclass
class IntegratedVolatilityForecast:
    """統合ボラティリティ予測"""

    symbol: str
    garch_result: GARCHVolatilityResult
    vix_assessment: VIXRiskAssessment
    risk_metrics: DynamicRiskMetrics
    final_volatility_forecast: float
    integrated_risk_score: float  # 0-1スケール（1が最高リスク）
    recommended_position_size: float  # 推奨ポジションサイズ（基準の%)
    risk_adjusted_return_forecast: float
    confidence_level: float
    processing_time: float


class VolatilityPredictionSystem:
    """
    Volatility Prediction System

    GARCH モデル、VIX指標、動的リスク調整による統合ボラティリティ予測
    統合最適化基盤（Issues #322-325）+ Phase 1-3システム統合
    """

    def __init__(
        self,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        garch_model_type: str = "GARCH",
        forecast_horizon: int = 5,
        var_confidence: float = 0.95,
        max_concurrent: int = 10,
    ):
        """
        Volatility Prediction System初期化

        Args:
            enable_cache: キャッシュ有効化
            enable_parallel: 並列処理有効化
            garch_model_type: GARCH モデル種類 ('GARCH', 'EGARCH', 'GJR')
            forecast_horizon: 予測期間（日数）
            var_confidence: VaR信頼水準
            max_concurrent: 最大並行処理数
        """
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        self.garch_model_type = garch_model_type
        self.forecast_horizon = forecast_horizon
        self.var_confidence = var_confidence
        self.max_concurrent = max_concurrent

        # 統合最適化基盤初期化
        if self.enable_cache:
            self.cache_manager = UnifiedCacheManager(
                l1_memory_mb=64, l2_memory_mb=256, l3_disk_mb=512
            )
            logger.info("統合キャッシュシステム有効化（Issue #324統合）")

        if self.enable_parallel:
            self.parallel_engine = AdvancedParallelMLEngine(
                cpu_workers=max_concurrent, enable_monitoring=True
            )
            logger.info("高度並列処理システム有効化（Issue #323統合）")

        # モデル・データ保存
        self.trained_garch_models = {}
        self.vix_data_cache = {}
        self.risk_parameters = {
            "base_position_size": 0.1,  # 基準ポジションサイズ (10%)
            "max_position_size": 0.25,  # 最大ポジションサイズ (25%)
            "min_position_size": 0.01,  # 最小ポジションサイズ (1%)
            "risk_free_rate": 0.001,  # リスクフリーレート (0.1%)
            "max_leverage": 2.0,  # 最大レバレッジ
        }

        # パフォーマンス統計
        self.stats = {
            "total_forecasts": 0,
            "cache_hits": 0,
            "garch_forecasts": 0,
            "vix_assessments": 0,
            "avg_processing_time": 0.0,
            "accuracy_scores": [],
            "risk_reduction_achieved": [],
        }

        logger.info("Volatility Prediction System（統合最適化版）初期化完了")
        logger.info(f"  - 統合キャッシュ: {self.enable_cache}")
        logger.info(f"  - 並列処理: {self.enable_parallel}")
        logger.info(f"  - GARCH モデル: {self.garch_model_type}")
        logger.info(f"  - 予測期間: {self.forecast_horizon}日")
        logger.info(f"  - VaR信頼水準: {self.var_confidence:.1%}")
        logger.info(f"  - 最大並行数: {self.max_concurrent}")

    async def predict_garch_volatility(
        self, data: pd.DataFrame, symbol: str
    ) -> GARCHVolatilityResult:
        """
        GARCH ボラティリティ予測

        Args:
            data: 価格データ
            symbol: 銘柄コード

        Returns:
            GARCHVolatilityResult: GARCH予測結果
        """
        start_time = time.time()

        # キャッシュチェック
        cache_key = None
        if self.enable_cache:
            cache_key = generate_unified_cache_key(
                "garch_volatility", symbol, str(len(data)), self.garch_model_type
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"GARCH予測キャッシュヒット: {symbol}")
                self.stats["cache_hits"] += 1
                return cached_result

        try:
            logger.info(
                f"GARCH ボラティリティ予測開始: {symbol} ({self.garch_model_type})"
            )

            # リターン計算
            returns = data["Close"].pct_change().dropna()

            if len(returns) < 50:
                logger.warning(f"データ不足: {symbol} - {len(returns)}日分")
                return self._create_default_garch_result(symbol)

            # GARCH モデル訓練
            if not ARCH_AVAILABLE:
                logger.warning("arch未インストール - 簡易ボラティリティ計算")
                return await self._simple_volatility_forecast(returns, symbol)

            # GARCH モデル構築
            model = await self._build_garch_model(returns, symbol)

            # 予測実行
            forecast = model.forecast(horizon=self.forecast_horizon)
            forecast_variance = forecast.variance.values[-1, :]
            forecast_volatility = np.sqrt(forecast_variance.mean()) * np.sqrt(
                252
            )  # 年率化

            # 現在ボラティリティ
            current_volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

            # トレンド分析
            volatility_trend = await self._analyze_volatility_trend(
                current_volatility, forecast_volatility
            )

            # 信頼区間
            confidence_interval = await self._calculate_confidence_interval(
                forecast_variance, self.var_confidence
            )

            # モデルパラメータ
            model_params = {
                "omega": float(model.params["omega"]),
                "alpha": float(model.params["alpha[1]"]),
                "beta": float(model.params["beta[1]"])
                if "beta[1]" in model.params
                else 0.0,
            }

            result = GARCHVolatilityResult(
                symbol=symbol,
                current_volatility=current_volatility,
                forecast_volatility=forecast_volatility,
                volatility_trend=volatility_trend,
                confidence_interval=confidence_interval,
                model_type=self.garch_model_type,
                model_params=model_params,
                forecast_horizon=self.forecast_horizon,
                processing_time=time.time() - start_time,
            )

            # キャッシュ保存
            if self.enable_cache and cache_key:
                self.cache_manager.put(cache_key, result)

            # モデル保存
            self.trained_garch_models[symbol] = model

            self.stats["garch_forecasts"] += 1
            logger.info(
                f"GARCH予測完了: {symbol} - {volatility_trend} ({forecast_volatility:.1%} vol) ({result.processing_time:.3f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"GARCH予測エラー: {symbol} - {e}")
            traceback.print_exc()
            return self._create_default_garch_result(symbol)

    async def assess_vix_risk(
        self, data: pd.DataFrame, symbol: str, vix_data: Optional[pd.DataFrame] = None
    ) -> VIXRiskAssessment:
        """
        VIX リスク評価

        Args:
            data: 価格データ
            symbol: 銘柄コード
            vix_data: VIXデータ（Noneの場合は模擬データ使用）

        Returns:
            VIXRiskAssessment: VIXリスク評価結果
        """
        start_time = time.time()

        # キャッシュチェック
        cache_key = None
        if self.enable_cache:
            cache_key = generate_unified_cache_key("vix_risk", symbol, str(len(data)))
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"VIXリスク評価キャッシュヒット: {symbol}")
                self.stats["cache_hits"] += 1
                return cached_result

        try:
            logger.info(f"VIXリスク評価開始: {symbol}")

            # VIXデータ準備（実際のVIXデータがない場合は模擬データ）
            if vix_data is None:
                vix_data = await self._generate_mock_vix_data(data)

            # 現在のVIXレベル
            current_vix = vix_data["Close"].iloc[-1] if len(vix_data) > 0 else 20.0

            # リスクレジーム判定
            risk_regime = await self._determine_risk_regime(current_vix)

            # 市場恐怖指数（0-1スケール）
            market_fear = await self._calculate_market_fear_indicator(current_vix)

            # VIXとの相関計算
            correlation = await self._calculate_vix_correlation(data, vix_data)

            # リスク調整係数
            risk_adjustment = await self._calculate_risk_adjustment_factor(
                risk_regime, market_fear, correlation
            )

            # 推奨アクション
            recommended_action = await self._determine_recommended_action(
                risk_regime, risk_adjustment
            )

            result = VIXRiskAssessment(
                symbol=symbol,
                vix_level=current_vix,
                risk_regime=risk_regime,
                market_fear_indicator=market_fear,
                correlation_with_vix=correlation,
                risk_adjustment_factor=risk_adjustment,
                recommended_action=recommended_action,
                processing_time=time.time() - start_time,
            )

            # キャッシュ保存
            if self.enable_cache and cache_key:
                self.cache_manager.put(cache_key, result)

            self.stats["vix_assessments"] += 1
            logger.info(
                f"VIXリスク評価完了: {symbol} - {risk_regime} regime (VIX: {current_vix:.1f}) ({result.processing_time:.3f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"VIXリスク評価エラー: {symbol} - {e}")
            traceback.print_exc()
            return self._create_default_vix_result(symbol)

    async def calculate_dynamic_risk_metrics(
        self,
        data: pd.DataFrame,
        symbol: str,
        garch_result: GARCHVolatilityResult,
        vix_result: VIXRiskAssessment,
    ) -> DynamicRiskMetrics:
        """
        動的リスク指標計算

        Args:
            data: 価格データ
            symbol: 銘柄コード
            garch_result: GARCH結果
            vix_result: VIX評価結果

        Returns:
            DynamicRiskMetrics: 動的リスク指標
        """
        start_time = time.time()

        try:
            logger.info(f"動的リスク指標計算開始: {symbol}")

            # リターン計算
            returns = data["Close"].pct_change().dropna()

            # Value at Risk (VaR)
            current_var = await self._calculate_var(returns, self.var_confidence)

            # Expected Shortfall (Conditional VaR)
            expected_shortfall = await self._calculate_expected_shortfall(
                returns, self.var_confidence
            )

            # 最大ドローダウン予測
            max_drawdown_forecast = await self._forecast_max_drawdown(
                returns, garch_result.forecast_volatility
            )

            # シャープレシオ予測
            sharpe_forecast = await self._forecast_sharpe_ratio(
                returns, garch_result.forecast_volatility
            )

            # ポジションサイズ調整
            position_multiplier = await self._calculate_position_size_multiplier(
                garch_result, vix_result, current_var
            )

            # リスク予算配分
            risk_budget = await self._calculate_risk_budget_allocation(
                garch_result.forecast_volatility, vix_result.risk_adjustment_factor
            )

            # ストップロス・テイクプロフィット
            stop_loss, take_profit = await self._calculate_exit_levels(
                data["Close"].iloc[-1], current_var, garch_result.forecast_volatility
            )

            result = DynamicRiskMetrics(
                symbol=symbol,
                current_var=current_var,
                expected_shortfall=expected_shortfall,
                max_drawdown_forecast=max_drawdown_forecast,
                sharpe_ratio_forecast=sharpe_forecast,
                position_size_multiplier=position_multiplier,
                risk_budget_allocation=risk_budget,
                stop_loss_level=stop_loss,
                take_profit_level=take_profit,
                processing_time=time.time() - start_time,
            )

            logger.info(
                f"動的リスク指標計算完了: {symbol} - VaR: {current_var:.1%}, Pos: {position_multiplier:.2f}x ({result.processing_time:.3f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"動的リスク指標計算エラー: {symbol} - {e}")
            traceback.print_exc()
            return self._create_default_risk_metrics(symbol)

    async def integrated_volatility_forecast(
        self, data: pd.DataFrame, symbol: str, vix_data: Optional[pd.DataFrame] = None
    ) -> IntegratedVolatilityForecast:
        """
        統合ボラティリティ予測

        Args:
            data: 価格データ
            symbol: 銘柄コード
            vix_data: VIXデータ（オプション）

        Returns:
            IntegratedVolatilityForecast: 統合予測結果
        """
        start_time = time.time()

        try:
            logger.info(f"統合ボラティリティ予測開始: {symbol}")

            # 並列実行で各コンポーネント計算
            if self.enable_parallel:
                # 並列実行
                tasks = [
                    self.predict_garch_volatility(data, symbol),
                    self.assess_vix_risk(data, symbol, vix_data),
                ]
                results = await asyncio.gather(*tasks)
                garch_result, vix_result = results
            else:
                # 順次実行
                garch_result = await self.predict_garch_volatility(data, symbol)
                vix_result = await self.assess_vix_risk(data, symbol, vix_data)

            # 動的リスク指標計算
            risk_metrics = await self.calculate_dynamic_risk_metrics(
                data, symbol, garch_result, vix_result
            )

            # 統合ボラティリティ予測
            final_volatility = await self._integrate_volatility_forecasts(
                garch_result, vix_result
            )

            # 統合リスクスコア（0-1スケール）
            integrated_risk = await self._calculate_integrated_risk_score(
                garch_result, vix_result, risk_metrics
            )

            # 推奨ポジションサイズ
            recommended_position = await self._calculate_recommended_position_size(
                risk_metrics.position_size_multiplier, integrated_risk
            )

            # リスク調整リターン予測
            risk_adjusted_return = await self._forecast_risk_adjusted_return(
                final_volatility, risk_metrics.sharpe_ratio_forecast
            )

            # 信頼度レベル
            confidence_level = await self._calculate_overall_confidence(
                garch_result, vix_result, len(data)
            )

            result = IntegratedVolatilityForecast(
                symbol=symbol,
                garch_result=garch_result,
                vix_assessment=vix_result,
                risk_metrics=risk_metrics,
                final_volatility_forecast=final_volatility,
                integrated_risk_score=integrated_risk,
                recommended_position_size=recommended_position,
                risk_adjusted_return_forecast=risk_adjusted_return,
                confidence_level=confidence_level,
                processing_time=time.time() - start_time,
            )

            self.stats["total_forecasts"] += 1
            logger.info(
                f"統合ボラティリティ予測完了: {symbol} - Vol: {final_volatility:.1%}, Risk: {integrated_risk:.3f}, Pos: {recommended_position:.1%} ({result.processing_time:.3f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"統合ボラティリティ予測エラー: {symbol} - {e}")
            traceback.print_exc()
            return self._create_default_integrated_result(symbol)

    async def _build_garch_model(self, returns: pd.Series, symbol: str):
        """GARCH モデル構築"""
        returns_percent = returns * 100  # パーセント変換

        if self.garch_model_type == "GARCH":
            model = arch_model(returns_percent, vol="GARCH", p=1, q=1)
        elif self.garch_model_type == "EGARCH":
            model = arch_model(returns_percent, vol="EGARCH", p=1, o=1, q=1)
        elif self.garch_model_type == "GJR":
            model = arch_model(returns_percent, vol="GARCH", p=1, o=1, q=1)
        else:
            model = arch_model(returns_percent, vol="GARCH", p=1, q=1)

        # モデル推定
        fitted_model = model.fit(disp="off")
        return fitted_model

    async def _simple_volatility_forecast(
        self, returns: pd.Series, symbol: str
    ) -> GARCHVolatilityResult:
        """簡易ボラティリティ予測（archパッケージなしの場合）"""
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        # 簡易予測（EWMA風）
        ewm_vol = returns.ewm(span=20).std().iloc[-1] * np.sqrt(252)
        forecast_vol = current_vol * 0.7 + ewm_vol * 0.3

        # トレンド判定
        vol_ma_short = returns.rolling(5).std().mean() * np.sqrt(252)
        vol_ma_long = returns.rolling(20).std().mean() * np.sqrt(252)

        if vol_ma_short > vol_ma_long * 1.1:
            trend = "increasing"
        elif vol_ma_short < vol_ma_long * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        return GARCHVolatilityResult(
            symbol=symbol,
            current_volatility=current_vol,
            forecast_volatility=forecast_vol,
            volatility_trend=trend,
            confidence_interval=(forecast_vol * 0.8, forecast_vol * 1.2),
            model_type="Simple",
            model_params={},
            forecast_horizon=self.forecast_horizon,
            processing_time=0.0,
        )

    async def _analyze_volatility_trend(
        self, current_vol: float, forecast_vol: float
    ) -> str:
        """ボラティリティトレンド分析"""
        change_ratio = (forecast_vol - current_vol) / current_vol

        if change_ratio > 0.1:
            return "increasing"
        elif change_ratio < -0.1:
            return "decreasing"
        else:
            return "stable"

    async def _calculate_confidence_interval(
        self, forecast_variance: np.ndarray, confidence: float
    ) -> Tuple[float, float]:
        """信頼区間計算"""
        vol_std = np.sqrt(forecast_variance.std())
        vol_mean = np.sqrt(forecast_variance.mean()) * np.sqrt(252)

        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * vol_std * np.sqrt(252)

        return (max(0.01, vol_mean - margin), vol_mean + margin)

    async def _generate_mock_vix_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """模擬VIXデータ生成"""
        returns = data["Close"].pct_change().dropna()

        # リターンのボラティリティからVIX風の指標を生成
        rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100

        # VIX風の調整（平均20、範囲10-50程度）
        mock_vix = rolling_vol.fillna(20)
        mock_vix = np.clip(mock_vix, 10, 50)

        vix_data = pd.DataFrame({"Close": mock_vix}, index=data.index[-len(mock_vix) :])

        return vix_data

    async def _determine_risk_regime(self, vix_level: float) -> str:
        """リスクレジーム判定"""
        if vix_level < 15:
            return "low"
        elif vix_level < 20:
            return "normal"
        elif vix_level < 30:
            return "high"
        else:
            return "extreme"

    async def _calculate_market_fear_indicator(self, vix_level: float) -> float:
        """市場恐怖指数計算（0-1スケール）"""
        # VIX 10-50を0-1にマッピング
        normalized_vix = (vix_level - 10) / 40
        return max(0.0, min(1.0, normalized_vix))

    async def _calculate_vix_correlation(
        self, price_data: pd.DataFrame, vix_data: pd.DataFrame
    ) -> float:
        """VIX相関計算"""
        try:
            price_returns = price_data["Close"].pct_change().dropna()
            vix_changes = vix_data["Close"].pct_change().dropna()

            # 共通期間での相関
            common_dates = price_returns.index.intersection(vix_changes.index)
            if len(common_dates) < 20:
                return -0.3  # デフォルト値（通常負の相関）

            correlation = price_returns[common_dates].corr(vix_changes[common_dates])
            return float(correlation) if not np.isnan(correlation) else -0.3
        except:
            return -0.3

    async def _calculate_risk_adjustment_factor(
        self, risk_regime: str, market_fear: float, correlation: float
    ) -> float:
        """リスク調整係数計算"""
        regime_factors = {
            "low": 1.2,  # リスク低時は少し攻撃的
            "normal": 1.0,  # 通常時はベース
            "high": 0.7,  # リスク高時は保守的
            "extreme": 0.3,  # 極端時は大幅縮小
        }

        base_factor = regime_factors.get(risk_regime, 1.0)

        # 市場恐怖度による調整
        fear_adjustment = 1.0 - (market_fear * 0.5)

        # 相関による調整（負の相関が強いほど少し有利）
        correlation_adjustment = 1.0 + (
            abs(correlation) * 0.1 if correlation < 0 else 0
        )

        final_factor = base_factor * fear_adjustment * correlation_adjustment
        return max(0.1, min(2.0, final_factor))

    async def _determine_recommended_action(
        self, risk_regime: str, risk_adjustment: float
    ) -> str:
        """推奨アクション決定"""
        if risk_adjustment > 1.1:
            return "increase"
        elif risk_adjustment > 0.8:
            return "maintain"
        elif risk_adjustment > 0.4:
            return "reduce"
        else:
            return "avoid"

    async def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Value at Risk計算"""
        if len(returns) < 20:
            return 0.05  # デフォルト5%

        return float(abs(returns.quantile(1 - confidence)))

    async def _calculate_expected_shortfall(
        self, returns: pd.Series, confidence: float
    ) -> float:
        """Expected Shortfall計算"""
        var_threshold = returns.quantile(1 - confidence)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return await self._calculate_var(returns, confidence) * 1.3

        return float(abs(tail_returns.mean()))

    async def _forecast_max_drawdown(
        self, returns: pd.Series, forecast_vol: float
    ) -> float:
        """最大ドローダウン予測"""
        # 過去の最大ドローダウンを参考に、予測ボラティリティで調整
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max

        historical_max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.2

        # ボラティリティ調整
        current_vol = returns.std() * np.sqrt(252)
        vol_adjustment = forecast_vol / max(current_vol, 0.01)

        forecast_dd = historical_max_dd * vol_adjustment
        return max(0.05, min(0.8, forecast_dd))  # 5%-80%の範囲

    async def _forecast_sharpe_ratio(
        self, returns: pd.Series, forecast_vol: float
    ) -> float:
        """シャープレシオ予測"""
        annual_return = returns.mean() * 252

        if forecast_vol <= 0:
            return 0.0

        forecast_sharpe = (
            annual_return - self.risk_parameters["risk_free_rate"]
        ) / forecast_vol
        return max(-2.0, min(3.0, forecast_sharpe))  # -2~3の範囲

    async def _calculate_position_size_multiplier(
        self,
        garch_result: GARCHVolatilityResult,
        vix_result: VIXRiskAssessment,
        current_var: float,
    ) -> float:
        """ポジションサイズ調整倍率計算"""
        base_multiplier = 1.0

        # GARCHボラティリティによる調整
        if garch_result.volatility_trend == "increasing":
            vol_adjustment = 0.8
        elif garch_result.volatility_trend == "decreasing":
            vol_adjustment = 1.2
        else:
            vol_adjustment = 1.0

        # VIXリスク調整
        vix_adjustment = vix_result.risk_adjustment_factor

        # VaR調整
        var_adjustment = max(0.5, 1.0 - (current_var - 0.02) * 5)

        multiplier = base_multiplier * vol_adjustment * vix_adjustment * var_adjustment
        return max(0.1, min(3.0, multiplier))

    async def _calculate_risk_budget_allocation(
        self, forecast_vol: float, risk_adjustment: float
    ) -> float:
        """リスク予算配分計算"""
        # 基準ボラティリティを20%とした相対的な配分
        base_vol = 0.20
        vol_ratio = base_vol / max(forecast_vol, 0.01)

        # リスク調整係数を適用
        allocation = vol_ratio * risk_adjustment * 0.1  # 基準10%

        return max(0.01, min(0.3, allocation))  # 1%-30%の範囲

    async def _calculate_exit_levels(
        self, current_price: float, var: float, forecast_vol: float
    ) -> Tuple[float, float]:
        """ストップロス・テイクプロフィット水準計算"""
        # ストップロス: VaRの1.5倍
        stop_loss = current_price * (1 - var * 1.5)

        # テイクプロフィット: リスクリワード比2:1を目標
        take_profit = current_price * (1 + var * 3.0)

        return stop_loss, take_profit

    async def _integrate_volatility_forecasts(
        self, garch_result: GARCHVolatilityResult, vix_result: VIXRiskAssessment
    ) -> float:
        """ボラティリティ予測統合"""
        garch_vol = garch_result.forecast_volatility

        # VIXレベルによる調整
        vix_adjustment = 1.0 + (vix_result.market_fear_indicator - 0.5) * 0.3

        integrated_vol = garch_vol * vix_adjustment
        return max(0.05, min(1.0, integrated_vol))  # 5%-100%の範囲

    async def _calculate_integrated_risk_score(
        self,
        garch_result: GARCHVolatilityResult,
        vix_result: VIXRiskAssessment,
        risk_metrics: DynamicRiskMetrics,
    ) -> float:
        """統合リスクスコア計算"""
        vol_score = min(1.0, garch_result.forecast_volatility / 0.5)  # 50%を上限
        vix_score = vix_result.market_fear_indicator
        var_score = min(1.0, risk_metrics.current_var / 0.1)  # 10%を上限

        # 重み付き平均
        integrated_score = vol_score * 0.4 + vix_score * 0.3 + var_score * 0.3
        return max(0.0, min(1.0, integrated_score))

    async def _calculate_recommended_position_size(
        self, multiplier: float, risk_score: float
    ) -> float:
        """推奨ポジションサイズ計算"""
        base_size = self.risk_parameters["base_position_size"]

        # リスクスコア調整
        risk_adjustment = 1.0 - risk_score * 0.5

        recommended_size = base_size * multiplier * risk_adjustment

        return max(
            self.risk_parameters["min_position_size"],
            min(self.risk_parameters["max_position_size"], recommended_size),
        )

    async def _forecast_risk_adjusted_return(
        self, volatility: float, sharpe_forecast: float
    ) -> float:
        """リスク調整リターン予測"""
        expected_return = (
            self.risk_parameters["risk_free_rate"] + sharpe_forecast * volatility
        )
        return max(-0.5, min(1.0, expected_return))  # -50%~100%の範囲

    async def _calculate_overall_confidence(
        self,
        garch_result: GARCHVolatilityResult,
        vix_result: VIXRiskAssessment,
        data_length: int,
    ) -> float:
        """総合信頼度計算"""
        data_score = min(1.0, data_length / 100)  # 100日で満点

        model_score = 0.9 if ARCH_AVAILABLE else 0.6

        # VIXデータの質（模擬データの場合は低下）
        vix_score = 0.7  # 実際のVIXデータがある場合は0.9

        overall_confidence = data_score * 0.4 + model_score * 0.4 + vix_score * 0.2
        return max(0.3, min(0.95, overall_confidence))

    def _create_default_garch_result(self, symbol: str) -> GARCHVolatilityResult:
        """デフォルトGARCH結果作成"""
        return GARCHVolatilityResult(
            symbol=symbol,
            current_volatility=0.25,
            forecast_volatility=0.25,
            volatility_trend="stable",
            confidence_interval=(0.20, 0.30),
            model_type="Default",
            model_params={},
            forecast_horizon=self.forecast_horizon,
            processing_time=0.0,
        )

    def _create_default_vix_result(self, symbol: str) -> VIXRiskAssessment:
        """デフォルトVIX結果作成"""
        return VIXRiskAssessment(
            symbol=symbol,
            vix_level=20.0,
            risk_regime="normal",
            market_fear_indicator=0.5,
            correlation_with_vix=-0.3,
            risk_adjustment_factor=1.0,
            recommended_action="maintain",
            processing_time=0.0,
        )

    def _create_default_risk_metrics(self, symbol: str) -> DynamicRiskMetrics:
        """デフォルトリスク指標作成"""
        return DynamicRiskMetrics(
            symbol=symbol,
            current_var=0.05,
            expected_shortfall=0.08,
            max_drawdown_forecast=0.15,
            sharpe_ratio_forecast=0.5,
            position_size_multiplier=1.0,
            risk_budget_allocation=0.1,
            stop_loss_level=0.0,
            take_profit_level=0.0,
            processing_time=0.0,
        )

    def _create_default_integrated_result(
        self, symbol: str
    ) -> IntegratedVolatilityForecast:
        """デフォルト統合結果作成"""
        garch_default = self._create_default_garch_result(symbol)
        vix_default = self._create_default_vix_result(symbol)
        risk_default = self._create_default_risk_metrics(symbol)

        return IntegratedVolatilityForecast(
            symbol=symbol,
            garch_result=garch_default,
            vix_assessment=vix_default,
            risk_metrics=risk_default,
            final_volatility_forecast=0.25,
            integrated_risk_score=0.5,
            recommended_position_size=0.1,
            risk_adjusted_return_forecast=0.05,
            confidence_level=0.5,
            processing_time=0.0,
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return {
            "total_forecasts": self.stats["total_forecasts"],
            "cache_hit_rate": self.stats["cache_hits"]
            / max(1, self.stats["total_forecasts"]),
            "garch_forecasts": self.stats["garch_forecasts"],
            "vix_assessments": self.stats["vix_assessments"],
            "avg_processing_time": self.stats["avg_processing_time"],
            "system_status": {
                "cache_enabled": self.enable_cache,
                "parallel_enabled": self.enable_parallel,
                "garch_model": self.garch_model_type,
                "arch_available": ARCH_AVAILABLE,
                "sklearn_available": SKLEARN_AVAILABLE,
            },
            "risk_parameters": self.risk_parameters,
            "optimization_benefits": {
                "max_drawdown_reduction": "20% 最大ドローダウン削減目標",
                "sharpe_improvement": "0.3 シャープレシオ向上目標",
                "risk_management": "動的ポジションサイジング",
                "volatility_forecasting": "GARCH + VIXリスク統合",
            },
        }
