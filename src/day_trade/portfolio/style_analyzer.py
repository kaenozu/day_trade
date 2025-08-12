#!/usr/bin/env python3
"""
投資スタイル分析システム
機械学習駆動投資スタイル分析・分類・適応システム

Features:
- ファクターベース投資スタイル分析
- 機械学習スタイル分類
- 動的スタイル適応
- リスク許容度分析
- パフォーマンス評価
- カスタムスタイル定義
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# オプショナル依存関係
SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, silhouette_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    pass


class InvestmentStyle(Enum):
    """投資スタイル"""

    GROWTH = "growth"
    VALUE = "value"
    BLEND = "blend"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    DIVIDEND = "dividend"
    SMALL_CAP = "small_cap"
    LARGE_CAP = "large_cap"
    INTERNATIONAL = "international"
    ESG = "esg"
    CUSTOM = "custom"


class RiskProfile(Enum):
    """リスクプロファイル"""

    CONSERVATIVE = "conservative"
    MODERATE_CONSERVATIVE = "moderate_conservative"
    MODERATE = "moderate"
    MODERATE_AGGRESSIVE = "moderate_aggressive"
    AGGRESSIVE = "aggressive"


class TimeHorizon(Enum):
    """投資期間"""

    SHORT_TERM = "short_term"  # < 1年
    MEDIUM_TERM = "medium_term"  # 1-5年
    LONG_TERM = "long_term"  # 5-10年
    VERY_LONG_TERM = "very_long_term"  # 10年+


@dataclass
class StyleConfiguration:
    """スタイル設定"""

    # 基本設定
    primary_style: InvestmentStyle = InvestmentStyle.BLEND
    secondary_styles: List[InvestmentStyle] = None
    risk_profile: RiskProfile = RiskProfile.MODERATE
    time_horizon: TimeHorizon = TimeHorizon.LONG_TERM

    # ファクター設定
    growth_weight: float = 0.2
    value_weight: float = 0.2
    momentum_weight: float = 0.2
    quality_weight: float = 0.2
    volatility_weight: float = 0.2

    # ESG設定
    esg_consideration: bool = False
    esg_weight: float = 0.1

    # 動的調整
    adaptive_style: bool = True
    rebalancing_frequency: int = 90  # 日数

    # 制約条件
    max_sector_weight: float = 0.25
    max_single_asset_weight: float = 0.10
    min_diversification: int = 20

    def __post_init__(self):
        if self.secondary_styles is None:
            self.secondary_styles = []


@dataclass
class StyleAnalysisResult:
    """スタイル分析結果"""

    detected_style: InvestmentStyle
    style_confidence: float
    style_scores: Dict[InvestmentStyle, float]

    # ファクター露出
    factor_exposures: Dict[str, float]

    # リスク分析
    risk_profile: RiskProfile
    volatility: float
    max_drawdown: float
    sharpe_ratio: float

    # 分散化メトリクス
    diversification_score: float
    sector_concentration: Dict[str, float]

    # 適合度
    style_consistency: float
    style_drift: float

    # パフォーマンス
    alpha: float
    beta: float
    information_ratio: float


class StyleAnalyzer:
    """投資スタイル分析システム"""

    def __init__(self, config: StyleConfiguration = None):
        self.config = config or StyleConfiguration()

        # スタイル定義
        self.style_definitions = self._initialize_style_definitions()

        # 機械学習モデル
        self.style_classifier = None
        self.pca_model = None
        self.scaler = None

        # 履歴管理
        self.analysis_history = []
        self.style_evolution = []

        # パフォーマンス追跡
        self.performance_metrics = {
            "classification_accuracy": [],
            "style_consistency": [],
            "adaptation_performance": [],
        }

        # 学習データ
        self.training_data = None
        self.is_trained = False

        logger.info("投資スタイル分析システム初期化完了")

    def _initialize_style_definitions(self) -> Dict[InvestmentStyle, Dict[str, Any]]:
        """スタイル定義初期化"""

        return {
            InvestmentStyle.GROWTH: {
                "characteristics": {
                    "revenue_growth": (0.15, 1.0),  # 15%+ 売上成長
                    "earnings_growth": (0.20, 1.0),  # 20%+ 利益成長
                    "pe_ratio": (20, 100),  # P/E 20-100
                    "price_momentum": (0.05, 1.0),  # 価格モメンタム
                    "beta": (1.0, 2.0),  # ベータ > 1
                },
                "sectors": ["Technology", "Healthcare", "Consumer Discretionary"],
                "typical_volatility": (0.20, 0.35),
            },
            InvestmentStyle.VALUE: {
                "characteristics": {
                    "pe_ratio": (5, 20),  # 低P/E
                    "pb_ratio": (0.5, 2.0),  # 低P/B
                    "dividend_yield": (0.02, 0.08),  # 配当利回り
                    "debt_to_equity": (0.0, 0.6),  # 低負債比率
                    "book_value_growth": (-0.05, 0.15),  # 簿価成長
                },
                "sectors": ["Financials", "Energy", "Utilities"],
                "typical_volatility": (0.12, 0.25),
            },
            InvestmentStyle.MOMENTUM: {
                "characteristics": {
                    "price_momentum_3m": (0.10, 1.0),  # 3ヶ月モメンタム
                    "price_momentum_12m": (0.20, 1.0),  # 12ヶ月モメンタム
                    "earnings_revision": (0.05, 1.0),  # 利益予想上方修正
                    "relative_strength": (0.7, 1.0),  # 相対強度
                    "volume_trend": (1.2, 3.0),  # 出来高増加
                },
                "sectors": ["All"],
                "typical_volatility": (0.18, 0.40),
            },
            InvestmentStyle.QUALITY: {
                "characteristics": {
                    "roe": (0.15, 1.0),  # ROE > 15%
                    "roa": (0.08, 1.0),  # ROA > 8%
                    "debt_to_equity": (0.0, 0.4),  # 低負債
                    "interest_coverage": (5.0, 100.0),  # 利息カバー率
                    "profit_margin": (0.10, 1.0),  # 利益率 > 10%
                },
                "sectors": ["Technology", "Healthcare", "Consumer Staples"],
                "typical_volatility": (0.15, 0.28),
            },
            InvestmentStyle.LOW_VOLATILITY: {
                "characteristics": {
                    "volatility": (0.08, 0.20),  # 低ボラティリティ
                    "beta": (0.3, 0.8),  # 低ベータ
                    "dividend_yield": (0.02, 0.06),  # 安定配当
                    "earnings_stability": (0.8, 1.0),  # 利益安定性
                    "debt_to_equity": (0.0, 0.5),  # 低負債
                },
                "sectors": ["Utilities", "Consumer Staples", "REITs"],
                "typical_volatility": (0.08, 0.18),
            },
            InvestmentStyle.DIVIDEND: {
                "characteristics": {
                    "dividend_yield": (0.03, 0.10),  # 高配当
                    "dividend_growth": (0.05, 0.20),  # 配当成長
                    "payout_ratio": (0.30, 0.80),  # 配当性向
                    "free_cash_flow": (0.05, 1.0),  # フリーキャッシュフロー
                    "debt_to_equity": (0.0, 0.6),  # 適度な負債
                },
                "sectors": ["Utilities", "Financials", "Consumer Staples"],
                "typical_volatility": (0.12, 0.22),
            },
        }

    def analyze_portfolio_style(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: pd.DataFrame = None,
        holdings_data: pd.DataFrame = None,
    ) -> StyleAnalysisResult:
        """ポートフォリオスタイル分析"""

        logger.info("ポートフォリオスタイル分析開始")

        try:
            # 基本統計計算
            returns = portfolio_data.pct_change().dropna()

            # ファクター露出計算
            factor_exposures = self._calculate_factor_exposures(portfolio_data, benchmark_data)

            # スタイル分類
            style_scores = self._classify_investment_style(factor_exposures, returns)
            detected_style = max(style_scores, key=style_scores.get)
            style_confidence = style_scores[detected_style]

            # リスクプロファイル分析
            risk_profile = self._analyze_risk_profile(returns)

            # パフォーマンス分析
            performance_metrics = self._calculate_performance_metrics(returns, benchmark_data)

            # 分散化分析
            diversification_metrics = self._analyze_diversification(holdings_data)

            # スタイル整合性
            style_consistency = self._calculate_style_consistency(returns, detected_style)

            result = StyleAnalysisResult(
                detected_style=detected_style,
                style_confidence=style_confidence,
                style_scores=style_scores,
                factor_exposures=factor_exposures,
                risk_profile=risk_profile,
                volatility=returns.std() * np.sqrt(252),
                max_drawdown=self._calculate_max_drawdown(portfolio_data),
                sharpe_ratio=performance_metrics.get("sharpe_ratio", 0),
                diversification_score=diversification_metrics.get("diversification_score", 0),
                sector_concentration=diversification_metrics.get("sector_concentration", {}),
                style_consistency=style_consistency,
                style_drift=self._calculate_style_drift(),
                alpha=performance_metrics.get("alpha", 0),
                beta=performance_metrics.get("beta", 1),
                information_ratio=performance_metrics.get("information_ratio", 0),
            )

            # 履歴更新
            self.analysis_history.append(result)
            self.style_evolution.append(
                {
                    "date": datetime.now(),
                    "style": detected_style,
                    "confidence": style_confidence,
                }
            )

            logger.info(
                f"スタイル分析完了: {detected_style.value} (信頼度: {style_confidence:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"スタイル分析エラー: {e}")
            raise

    def _calculate_factor_exposures(
        self, portfolio_data: pd.DataFrame, benchmark_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """ファクター露出計算"""

        returns = portfolio_data.pct_change().dropna()

        exposures = {}

        # 基本ファクター
        exposures["volatility"] = returns.std() * np.sqrt(252)
        exposures["skewness"] = returns.skew()
        exposures["kurtosis"] = returns.kurtosis()

        # モメンタムファクター
        if len(returns) >= 252:
            # 12ヶ月モメンタム
            momentum_12m = portfolio_data.iloc[-1] / portfolio_data.iloc[-252] - 1
            exposures["momentum_12m"] = momentum_12m

        if len(returns) >= 63:
            # 3ヶ月モメンタム
            momentum_3m = portfolio_data.iloc[-1] / portfolio_data.iloc[-63] - 1
            exposures["momentum_3m"] = momentum_3m

        # トレンドファクター
        if len(portfolio_data) >= 20:
            ma_20 = portfolio_data.rolling(20).mean()
            trend_strength = portfolio_data.iloc[-1] / ma_20.iloc[-1] - 1
            exposures["trend_strength"] = trend_strength

        # ベータ（ベンチマークとの共分散）
        if benchmark_data is not None and len(benchmark_data) == len(portfolio_data):
            benchmark_returns = benchmark_data.pct_change().dropna()
            if len(returns) == len(benchmark_returns) and len(returns) > 1:
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                exposures["beta"] = (
                    covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                )
            else:
                exposures["beta"] = 1.0
        else:
            exposures["beta"] = 1.0

        # 平均リバージョン
        if len(returns) >= 20:
            autocorr = returns.autocorr(lag=1)
            exposures["mean_reversion"] = -autocorr if not np.isnan(autocorr) else 0.0

        return exposures

    def _classify_investment_style(
        self, factor_exposures: Dict[str, float], returns: pd.Series
    ) -> Dict[InvestmentStyle, float]:
        """投資スタイル分類"""

        style_scores = {}

        # 各スタイルに対するスコア計算
        for style, definition in self.style_definitions.items():
            score = 0.0
            total_weight = 0.0

            characteristics = definition.get("characteristics", {})

            for char_name, (min_val, max_val) in characteristics.items():
                if char_name in factor_exposures:
                    exposure = factor_exposures[char_name]

                    # 特性範囲内かどうかでスコア計算
                    if min_val <= exposure <= max_val:
                        # 中央値に近いほど高スコア
                        center = (min_val + max_val) / 2
                        range_size = max_val - min_val
                        distance_from_center = abs(exposure - center) / (range_size / 2)
                        char_score = 1.0 - distance_from_center
                    else:
                        # 範囲外はペナルティ
                        if exposure < min_val:
                            char_score = max(0, 1.0 - (min_val - exposure) / min_val)
                        else:
                            char_score = max(0, 1.0 - (exposure - max_val) / max_val)

                    score += char_score
                    total_weight += 1.0

            # ボラティリティ適合性チェック
            portfolio_vol = factor_exposures.get("volatility", 0.2)
            typical_vol_range = definition.get("typical_volatility", (0.1, 0.3))

            if typical_vol_range[0] <= portfolio_vol <= typical_vol_range[1]:
                vol_score = 1.0
            else:
                vol_distance = min(
                    abs(portfolio_vol - typical_vol_range[0]),
                    abs(portfolio_vol - typical_vol_range[1]),
                )
                vol_score = max(0, 1.0 - vol_distance)

            score += vol_score
            total_weight += 1.0

            # 正規化
            style_scores[style] = score / total_weight if total_weight > 0 else 0.0

        # 追加スタイル（簡易実装）

        # Blend: 複数スタイルのバランス
        if len([s for s in style_scores.values() if s > 0.3]) >= 2:
            style_scores[InvestmentStyle.BLEND] = np.mean(list(style_scores.values())[:3])
        else:
            style_scores[InvestmentStyle.BLEND] = 0.0

        # Small-cap vs Large-cap（簡易版）
        beta = factor_exposures.get("beta", 1.0)
        volatility = factor_exposures.get("volatility", 0.2)

        if beta > 1.2 and volatility > 0.25:
            style_scores[InvestmentStyle.SMALL_CAP] = 0.8
            style_scores[InvestmentStyle.LARGE_CAP] = 0.2
        elif beta < 0.8 and volatility < 0.18:
            style_scores[InvestmentStyle.LARGE_CAP] = 0.8
            style_scores[InvestmentStyle.SMALL_CAP] = 0.2
        else:
            style_scores[InvestmentStyle.SMALL_CAP] = 0.5
            style_scores[InvestmentStyle.LARGE_CAP] = 0.5

        return style_scores

    def _analyze_risk_profile(self, returns: pd.Series) -> RiskProfile:
        """リスクプロファイル分析"""

        annual_vol = returns.std() * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown_from_returns(returns)

        # リスクレベル判定
        if annual_vol <= 0.10 and max_drawdown <= 0.05:
            return RiskProfile.CONSERVATIVE
        elif annual_vol <= 0.15 and max_drawdown <= 0.10:
            return RiskProfile.MODERATE_CONSERVATIVE
        elif annual_vol <= 0.20 and max_drawdown <= 0.15:
            return RiskProfile.MODERATE
        elif annual_vol <= 0.30 and max_drawdown <= 0.25:
            return RiskProfile.MODERATE_AGGRESSIVE
        else:
            return RiskProfile.AGGRESSIVE

    def _calculate_performance_metrics(
        self, returns: pd.Series, benchmark_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """パフォーマンス指標計算"""

        metrics = {}

        # 基本指標
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)

        # シャープレシオ
        risk_free_rate = 0.02  # 仮定
        metrics["sharpe_ratio"] = (
            (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        )

        # ベンチマーク比較
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna()

            if len(returns) == len(benchmark_returns) and len(returns) > 1:
                # ベータ
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                metrics["beta"] = beta

                # アルファ
                benchmark_annual_return = benchmark_returns.mean() * 252
                alpha = annual_return - (
                    risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
                )
                metrics["alpha"] = alpha

                # インフォメーションレシオ
                active_returns = returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                metrics["information_ratio"] = (
                    (annual_return - benchmark_annual_return) / tracking_error
                    if tracking_error > 0
                    else 0
                )
            else:
                metrics["beta"] = 1.0
                metrics["alpha"] = 0.0
                metrics["information_ratio"] = 0.0

        return metrics

    def _analyze_diversification(self, holdings_data: pd.DataFrame = None) -> Dict[str, Any]:
        """分散化分析"""

        if holdings_data is None:
            return {
                "diversification_score": 0.7,  # デフォルト値
                "sector_concentration": {},
            }

        # 保有銘柄数ベースの分散化スコア
        num_holdings = len(holdings_data)
        diversification_score = min(1.0, num_holdings / 50)  # 50銘柄で満点

        # セクター集中度
        sector_concentration = {}
        if "sector" in holdings_data.columns and "weight" in holdings_data.columns:
            sector_weights = holdings_data.groupby("sector")["weight"].sum()
            sector_concentration = sector_weights.to_dict()

        return {
            "diversification_score": diversification_score,
            "sector_concentration": sector_concentration,
        }

    def _calculate_style_consistency(
        self, returns: pd.Series, detected_style: InvestmentStyle
    ) -> float:
        """スタイル整合性計算"""

        if len(self.analysis_history) < 2:
            return 1.0

        # 過去の分析結果との一致度
        recent_styles = [result.detected_style for result in self.analysis_history[-5:]]
        consistency = recent_styles.count(detected_style) / len(recent_styles)

        return consistency

    def _calculate_style_drift(self) -> float:
        """スタイルドリフト計算"""

        if len(self.style_evolution) < 10:
            return 0.0

        # 過去10期間のスタイル変化頻度
        recent_styles = [entry["style"] for entry in self.style_evolution[-10:]]
        unique_styles = len(set(recent_styles))

        # ドリフト度合い（変化が多いほど高い値）
        drift = (unique_styles - 1) / 9  # 最大9回変化可能

        return drift

    def _calculate_max_drawdown(self, price_data: pd.DataFrame) -> float:
        """最大ドローダウン計算"""
        cumulative = (1 + price_data.pct_change().dropna()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """リターンから最大ドローダウン計算"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def train_style_classifier(
        self,
        training_portfolios: Dict[str, pd.DataFrame],
        style_labels: Dict[str, InvestmentStyle],
    ) -> Dict[str, Any]:
        """スタイル分類器訓練"""

        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn未利用のため、分類器訓練をスキップ")
            return {"status": "sklearn_unavailable"}

        logger.info("スタイル分類器訓練開始")

        try:
            # 特徴量抽出
            features = []
            labels = []

            for portfolio_name, portfolio_data in training_portfolios.items():
                if portfolio_name in style_labels:
                    # ファクター露出計算
                    factor_exp = self._calculate_factor_exposures(portfolio_data)
                    feature_vector = [
                        factor_exp.get("volatility", 0),
                        factor_exp.get("momentum_12m", 0),
                        factor_exp.get("momentum_3m", 0),
                        factor_exp.get("beta", 1),
                        factor_exp.get("trend_strength", 0),
                        factor_exp.get("mean_reversion", 0),
                        factor_exp.get("skewness", 0),
                        factor_exp.get("kurtosis", 0),
                    ]

                    features.append(feature_vector)
                    labels.append(style_labels[portfolio_name].value)

            if len(features) < 5:
                logger.warning("訓練データが不足しています")
                return {"status": "insufficient_data"}

            X = np.array(features)
            y = np.array(labels)

            # データ標準化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # 主成分分析
            self.pca_model = PCA(n_components=min(5, X.shape[1]))
            X_pca = self.pca_model.fit_transform(X_scaled)

            # 分類器訓練
            self.style_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
            self.style_classifier.fit(X_pca, y)

            # 性能評価
            train_accuracy = self.style_classifier.score(X_pca, y)

            # 特徴量重要度
            feature_names = [
                "volatility",
                "momentum_12m",
                "momentum_3m",
                "beta",
                "trend_strength",
                "mean_reversion",
                "skewness",
                "kurtosis",
            ]
            feature_importance = dict(
                zip(
                    feature_names[: X.shape[1]],
                    self.pca_model.explained_variance_ratio_[: X.shape[1]],
                )
            )

            self.is_trained = True

            training_result = {
                "status": "success",
                "train_accuracy": train_accuracy,
                "n_samples": len(features),
                "n_features": X.shape[1],
                "feature_importance": feature_importance,
                "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
            }

            logger.info(f"分類器訓練完了: 精度={train_accuracy:.3f}")
            return training_result

        except Exception as e:
            logger.error(f"分類器訓練エラー: {e}")
            return {"status": "error", "error": str(e)}

    def predict_style_ml(self, portfolio_data: pd.DataFrame) -> Tuple[InvestmentStyle, float]:
        """機械学習スタイル予測"""

        if not self.is_trained or not SKLEARN_AVAILABLE:
            # フォールバック: ルールベース
            factor_exposures = self._calculate_factor_exposures(portfolio_data)
            returns = portfolio_data.pct_change().dropna()
            style_scores = self._classify_investment_style(factor_exposures, returns)
            detected_style = max(style_scores, key=style_scores.get)
            confidence = style_scores[detected_style]
            return detected_style, confidence

        try:
            # 特徴量抽出
            factor_exp = self._calculate_factor_exposures(portfolio_data)
            feature_vector = np.array(
                [
                    [
                        factor_exp.get("volatility", 0),
                        factor_exp.get("momentum_12m", 0),
                        factor_exp.get("momentum_3m", 0),
                        factor_exp.get("beta", 1),
                        factor_exp.get("trend_strength", 0),
                        factor_exp.get("mean_reversion", 0),
                        factor_exp.get("skewness", 0),
                        factor_exp.get("kurtosis", 0),
                    ]
                ]
            )

            # 前処理
            X_scaled = self.scaler.transform(feature_vector)
            X_pca = self.pca_model.transform(X_scaled)

            # 予測
            prediction = self.style_classifier.predict(X_pca)[0]
            probabilities = self.style_classifier.predict_proba(X_pca)[0]
            confidence = np.max(probabilities)

            detected_style = InvestmentStyle(prediction)

            return detected_style, confidence

        except Exception as e:
            logger.error(f"ML予測エラー: {e}")
            # フォールバック
            factor_exposures = self._calculate_factor_exposures(portfolio_data)
            returns = portfolio_data.pct_change().dropna()
            style_scores = self._classify_investment_style(factor_exposures, returns)
            detected_style = max(style_scores, key=style_scores.get)
            confidence = style_scores[detected_style]
            return detected_style, confidence

    def recommend_style_adaptation(
        self,
        current_result: StyleAnalysisResult,
        market_conditions: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """スタイル適応推奨"""

        recommendations = {
            "maintain_current_style": True,
            "recommended_changes": [],
            "risk_adjustments": [],
            "rebalancing_priority": "low",
        }

        # スタイル整合性チェック
        if current_result.style_consistency < 0.7:
            recommendations["maintain_current_style"] = False
            recommendations["recommended_changes"].append(
                {
                    "type": "style_realignment",
                    "reason": "スタイル整合性が低い",
                    "suggestion": "より一貫したスタイルへの調整を検討",
                }
            )

        # リスクレベルチェック
        if current_result.volatility > 0.25:
            recommendations["risk_adjustments"].append(
                {
                    "type": "volatility_reduction",
                    "current_vol": current_result.volatility,
                    "target_vol": 0.20,
                    "method": "low_volatility_assets_increase",
                }
            )

        # パフォーマンス分析
        if current_result.sharpe_ratio < 0.5:
            recommendations["recommended_changes"].append(
                {
                    "type": "performance_improvement",
                    "reason": "シャープレシオが低い",
                    "suggestion": "リスク調整後リターンの改善が必要",
                }
            )
            recommendations["rebalancing_priority"] = "high"

        # 分散化チェック
        if current_result.diversification_score < 0.6:
            recommendations["recommended_changes"].append(
                {
                    "type": "diversification_improvement",
                    "reason": "分散化が不十分",
                    "suggestion": "より多くの銘柄・セクターへの分散を推奨",
                }
            )

        return recommendations

    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析概要取得"""

        if not self.analysis_history:
            return {"status": "分析未実行"}

        latest_result = self.analysis_history[-1]

        # スタイル進化
        style_evolution_summary = {}
        if self.style_evolution:
            recent_styles = [entry["style"].value for entry in self.style_evolution[-10:]]
            for style in set(recent_styles):
                style_evolution_summary[style] = recent_styles.count(style)

        # パフォーマンス統計
        avg_confidence = np.mean([result.style_confidence for result in self.analysis_history])
        avg_consistency = np.mean([result.style_consistency for result in self.analysis_history])

        return {
            "latest_analysis": {
                "detected_style": latest_result.detected_style.value,
                "confidence": latest_result.style_confidence,
                "risk_profile": latest_result.risk_profile.value,
                "portfolio_volatility": latest_result.volatility,
                "sharpe_ratio": latest_result.sharpe_ratio,
                "diversification_score": latest_result.diversification_score,
            },
            "historical_performance": {
                "total_analyses": len(self.analysis_history),
                "average_confidence": avg_confidence,
                "average_consistency": avg_consistency,
                "style_evolution": style_evolution_summary,
            },
            "ml_model_status": {
                "is_trained": self.is_trained,
                "sklearn_available": SKLEARN_AVAILABLE,
            },
            "configuration": {
                "primary_style": self.config.primary_style.value,
                "adaptive_style": self.config.adaptive_style,
                "risk_profile": self.config.risk_profile.value,
                "time_horizon": self.config.time_horizon.value,
            },
        }


# グローバルインスタンス
_style_analyzer = None


def get_style_analyzer(config: StyleConfiguration = None) -> StyleAnalyzer:
    """投資スタイル分析システム取得"""
    global _style_analyzer
    if _style_analyzer is None:
        _style_analyzer = StyleAnalyzer(config)
    return _style_analyzer
