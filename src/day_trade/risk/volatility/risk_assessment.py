#!/usr/bin/env python3
"""
リスク評価モジュール

ボラティリティ分析に基づく投資リスク評価と投資戦略示唆:
- ボラティリティリスクスコア計算
- 投資ポートフォリオへの示唆
- トレーディング戦略推奨
- マーケットタイミング分析
"""

import numpy as np
from typing import Dict, List, Any

from .base import VolatilityEngineBase
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class VolatilityRiskAssessor(VolatilityEngineBase):
    """
    ボラティリティリスク評価クラス

    ボラティリティ予測に基づく包括的なリスク評価と投資戦略示唆を提供します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        super().__init__(model_cache_dir)
        
        # リスク評価パラメータ
        self.risk_thresholds = {
            "volatility": {
                "extreme_high": 40,     # 40%以上
                "high": 25,            # 25-40%
                "normal": 15,          # 15-25%
                "low": 10,             # 10%未満（異常）
            },
            "vix": {
                "panic": 40,           # パニック水準
                "fear": 25,            # 恐怖水準
                "normal": 15,          # 通常水準
                "complacency": 15,     # 楽観水準未満
            },
            "forecast_dispersion": 10,  # モデル間予測分散閾値
        }
        
        logger.info("ボラティリティリスク評価器初期化完了")

    def assess_volatility_risk(
        self, current_vol: float, current_vix: float, ensemble: Dict
    ) -> Dict[str, Any]:
        """
        ボラティリティリスク評価

        Args:
            current_vol: 現在のボラティリティ
            current_vix: 現在のVIX値
            ensemble: アンサンブル予測結果

        Returns:
            リスク評価結果辞書
        """
        try:
            risk_score = 0  # 0-100, 高いほど高リスク
            risk_factors = []

            # ボラティリティレベル評価
            vol_risk = self._assess_volatility_level(current_vol, risk_factors)
            risk_score += vol_risk

            # VIXレベル評価
            vix_risk = self._assess_vix_level(current_vix, risk_factors)
            risk_score += vix_risk

            # 予測不確実性評価
            uncertainty_risk = self._assess_prediction_uncertainty(ensemble, risk_factors)
            risk_score += uncertainty_risk

            # ボラティリティ変化評価
            change_risk = self._assess_volatility_change(
                current_vol, ensemble, risk_factors
            )
            risk_score += change_risk

            # リスクレベル分類
            risk_level = self._classify_risk_level(risk_score)

            # ボラティリティ見通し
            outlook = self._determine_volatility_outlook(current_vol, ensemble)

            return {
                "risk_level": risk_level,
                "risk_score": min(int(risk_score), 100),  # 上限100
                "risk_factors": risk_factors,
                "volatility_outlook": outlook,
                "component_risks": {
                    "volatility_level": vol_risk,
                    "vix_level": vix_risk,
                    "prediction_uncertainty": uncertainty_risk,
                    "volatility_change": change_risk,
                },
            }

        except Exception as e:
            logger.error(f"ボラティリティリスク評価エラー: {e}")
            return {"risk_level": "UNKNOWN", "risk_score": 50, "error": str(e)}

    def _assess_volatility_level(
        self, current_vol: float, risk_factors: List[str]
    ) -> float:
        """
        ボラティリティレベル評価

        Args:
            current_vol: 現在のボラティリティ
            risk_factors: リスク要因リスト（更新）

        Returns:
            ボラティリティレベルリスクスコア
        """
        annual_vol_pct = current_vol * 100
        thresholds = self.risk_thresholds["volatility"]

        if annual_vol_pct > thresholds["extreme_high"]:
            risk_factors.append("極端に高いボラティリティ")
            return 30
        elif annual_vol_pct > thresholds["high"]:
            risk_factors.append("高ボラティリティ環境")
            return 15
        elif annual_vol_pct < thresholds["low"]:
            risk_factors.append("異常に低いボラティリティ（反発リスク）")
            return 10
        else:
            return 0

    def _assess_vix_level(
        self, current_vix: float, risk_factors: List[str]
    ) -> float:
        """
        VIXレベル評価

        Args:
            current_vix: 現在のVIX値
            risk_factors: リスク要因リスト（更新）

        Returns:
            VIXレベルリスクスコア
        """
        thresholds = self.risk_thresholds["vix"]

        if current_vix > thresholds["panic"]:
            risk_factors.append("パニック水準のVIX")
            return 25
        elif current_vix > thresholds["fear"]:
            risk_factors.append("恐怖水準のVIX")
            return 10
        elif current_vix < thresholds["complacency"]:
            risk_factors.append("楽観水準のVIX（反発リスク）")
            return 15
        else:
            return 0

    def _assess_prediction_uncertainty(
        self, ensemble: Dict, risk_factors: List[str]
    ) -> float:
        """
        予測不確実性評価

        Args:
            ensemble: アンサンブル予測結果
            risk_factors: リスク要因リスト（更新）

        Returns:
            予測不確実性リスクスコア
        """
        forecast_range = ensemble.get("forecast_range", {})
        forecast_std = ensemble.get("forecast_std", 0)

        range_size = (
            forecast_range.get("max", 0) - forecast_range.get("min", 0)
        )

        if range_size > self.risk_thresholds["forecast_dispersion"]:
            risk_factors.append("モデル間の予測分散が大きい")
            return 20
        elif forecast_std > 5:
            risk_factors.append("予測の不確実性が高い")
            return 10
        else:
            return 0

    def _assess_volatility_change(
        self, current_vol: float, ensemble: Dict, risk_factors: List[str]
    ) -> float:
        """
        ボラティリティ変化評価

        Args:
            current_vol: 現在のボラティリティ
            ensemble: アンサンブル予測結果
            risk_factors: リスク要因リスト（更新）

        Returns:
            ボラティリティ変化リスクスコア
        """
        ensemble_vol = ensemble.get("ensemble_volatility", current_vol * 100)
        vol_change = ensemble_vol - current_vol * 100

        if vol_change > 10:
            risk_factors.append("ボラティリティ急上昇予測")
            return 20
        elif vol_change > 5:
            risk_factors.append("ボラティリティ上昇予測")
            return 10
        elif vol_change < -10:
            risk_factors.append("ボラティリティ急降下予測（市場歪み可能性）")
            return 5
        else:
            return 0

    def _classify_risk_level(self, risk_score: float) -> str:
        """
        リスクレベル分類

        Args:
            risk_score: リスクスコア

        Returns:
            リスクレベル文字列
        """
        if risk_score >= 60:
            return "HIGH"
        elif risk_score >= 35:
            return "MEDIUM"
        else:
            return "LOW"

    def _determine_volatility_outlook(
        self, current_vol: float, ensemble: Dict
    ) -> str:
        """
        ボラティリティ見通し決定

        Args:
            current_vol: 現在のボラティリティ
            ensemble: アンサンブル予測結果

        Returns:
            見通し文字列
        """
        ensemble_vol = ensemble.get("ensemble_volatility", current_vol * 100)
        vol_change = ensemble_vol - current_vol * 100

        if vol_change > 2:
            return "increasing"
        elif vol_change < -2:
            return "decreasing"
        else:
            return "stable"

    def generate_volatility_implications(
        self, ensemble: Dict, risk_assessment: Dict, current_metrics: Dict
    ) -> Dict[str, Any]:
        """
        ボラティリティの投資への示唆

        Args:
            ensemble: アンサンブル予測結果
            risk_assessment: リスク評価結果
            current_metrics: 現在のメトリクス

        Returns:
            投資示唆辞書
        """
        try:
            implications = {
                "portfolio_adjustments": [],
                "trading_strategies": [],
                "risk_management": [],
                "market_timing": [],
                "options_strategies": [],
            }

            ensemble_vol = ensemble.get("ensemble_volatility", 20)
            risk_level = risk_assessment.get("risk_level", "MEDIUM")
            outlook = risk_assessment.get("volatility_outlook", "stable")
            current_vix = current_metrics.get("vix_like_indicator", 20)

            # ポートフォリオ調整示唆
            implications["portfolio_adjustments"] = self._generate_portfolio_implications(
                ensemble_vol, risk_level
            )

            # トレーディング戦略示唆
            implications["trading_strategies"] = self._generate_trading_implications(
                outlook, ensemble_vol
            )

            # リスク管理示唆
            implications["risk_management"] = self._generate_risk_management_implications(
                risk_level, ensemble_vol
            )

            # マーケットタイミング示唆
            implications["market_timing"] = self._generate_market_timing_implications(
                current_vix, outlook
            )

            # オプション戦略示唆
            implications["options_strategies"] = self._generate_options_implications(
                ensemble_vol, outlook, current_vix
            )

            return implications

        except Exception as e:
            logger.error(f"示唆生成エラー: {e}")
            return {
                "portfolio_adjustments": ["ボラティリティ情報を基に慎重な判断を"],
                "error": str(e),
            }

    def _generate_portfolio_implications(
        self, ensemble_vol: float, risk_level: str
    ) -> List[str]:
        """ポートフォリオ調整示唆"""
        implications = []

        if ensemble_vol > 30:
            implications.extend([
                "ポジションサイズの縮小を検討",
                "防御的資産（債券、金）の比重増加",
                "分散投資の強化",
                "キャッシュポジションの増加",
            ])
        elif ensemble_vol < 15:
            implications.extend([
                "リスク資産の比重増加を検討",
                "成長株への配分増加機会",
                "レバレッジ活用の検討",
                "集中投資の機会",
            ])
        else:
            implications.extend([
                "現在のポートフォリオバランス維持",
                "定期的なリバランス実施",
            ])

        return implications

    def _generate_trading_implications(
        self, outlook: str, ensemble_vol: float
    ) -> List[str]:
        """トレーディング戦略示唆"""
        implications = []

        if outlook == "increasing":
            implications.extend([
                "ボラティリティブレイクアウト戦略",
                "ストラドル・ストラングル戦略（オプション）",
                "短期トレーディングの機会増加",
                "トレンドフォロー戦略の有効性向上",
            ])
        elif outlook == "decreasing":
            implications.extend([
                "平均回帰戦略",
                "レンジトレーディング",
                "バイ・アンド・ホールド戦略",
                "ボラティリティ売り戦略",
            ])
        else:
            implications.extend([
                "中立的な戦略",
                "両サイドのポジション準備",
            ])

        return implications

    def _generate_risk_management_implications(
        self, risk_level: str, ensemble_vol: float
    ) -> List[str]:
        """リスク管理示唆"""
        implications = []

        if risk_level == "HIGH":
            implications.extend([
                "ストップロス幅の拡大",
                "より頻繁なポジション見直し",
                "ヘッジ比率の増加",
                "ポートフォリオ保険の検討",
            ])
        elif risk_level == "LOW":
            implications.extend([
                "ストップロス幅の最適化",
                "ポジション保有期間の延長可能",
                "コスト効率の重視",
            ])
        else:
            implications.extend([
                "標準的なリスク管理継続",
                "定期的な見直し",
            ])

        return implications

    def _generate_market_timing_implications(
        self, current_vix: float, outlook: str
    ) -> List[str]:
        """マーケットタイミング示唆"""
        implications = []

        if current_vix > 30 and outlook == "decreasing":
            implications.extend([
                "逆張り投資の絶好機",
                "恐怖売りでの押し目買い検討",
            ])
        elif current_vix < 20 and outlook == "increasing":
            implications.extend([
                "リスクオフの準備",
                "利確タイミングの検討",
            ])
        elif current_vix > 35:
            implications.extend([
                "極端な恐怖時の投資機会",
                "分割投資の検討",
            ])

        return implications

    def _generate_options_implications(
        self, ensemble_vol: float, outlook: str, current_vix: float
    ) -> List[str]:
        """オプション戦略示唆"""
        implications = []

        if outlook == "increasing" and current_vix < 25:
            implications.extend([
                "ボラティリティロング（ストラドル購入）",
                "プット購入によるヘッジ",
            ])
        elif outlook == "decreasing" and current_vix > 25:
            implications.extend([
                "ボラティリティショート（ストラングル売り）",
                "カバードコール戦略",
            ])

        if ensemble_vol > 30:
            implications.extend([
                "高ボラティリティ環境でのプレミアム収益",
                "アイアンコンドル戦略",
            ])

        return implications

    def calculate_volatility_risk_metrics(
        self, comprehensive_forecast: Dict
    ) -> Dict[str, Any]:
        """
        総合的なボラティリティリスクメトリクス計算

        Args:
            comprehensive_forecast: 総合予測結果

        Returns:
            リスクメトリクス辞書
        """
        try:
            current = comprehensive_forecast.get("current_metrics", {})
            ensemble = comprehensive_forecast.get("ensemble_forecast", {})

            risk_assessment = self.assess_volatility_risk(
                current.get("realized_volatility", 0.2),
                current.get("vix_like_indicator", 20),
                ensemble,
            )

            implications = self.generate_volatility_implications(
                ensemble, risk_assessment, current
            )

            return {
                "risk_assessment": risk_assessment,
                "investment_implications": implications,
                "summary": {
                    "overall_risk": risk_assessment.get("risk_level", "UNKNOWN"),
                    "primary_concerns": risk_assessment.get("risk_factors", [])[:3],
                    "recommended_actions": implications.get("portfolio_adjustments", [])[:2],
                    "market_outlook": risk_assessment.get("volatility_outlook", "unknown"),
                },
            }

        except Exception as e:
            logger.error(f"リスクメトリクス計算エラー: {e}")
            return {"error": str(e)}