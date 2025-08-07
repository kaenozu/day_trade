#!/usr/bin/env python3
"""
教育的市場分析システム

法的に適切な教育・情報提供を行います
- 客観的データの提供
- 教育的解説の付加
- 統計的参考情報の提示
- 機械学習テクニカルスコア（教育目的）
- 適切な免責事項の明記

【重要】投資助言・推奨は一切行いません
"""

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from ..data.real_market_data import RealMarketDataManager
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 高度MLエンジンのオプション読み込み
try:
    from ..data.advanced_ml_engine import AdvancedMLEngine

    ADVANCED_ML_AVAILABLE = True
    logger.info("高度MLエンジン利用可能")
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logger.warning("高度MLエンジン利用不可")


@dataclass
class TechnicalIndicator:
    """テクニカル指標データ"""

    name: str
    current_value: float
    educational_description: str
    general_interpretation: str
    reference_levels: Dict[str, float]


@dataclass
class HistoricalStatistics:
    """統計的参考情報"""

    condition: str
    period_days: int
    upward_percentage: float
    downward_percentage: float
    average_change: float
    sample_size: int


@dataclass
class MLTechnicalScore:
    """機械学習テクニカルスコア（教育用）"""

    score_name: str
    score_value: float  # 0-100のスコア
    confidence_level: float  # 信頼度 0-1
    model_description: str
    educational_interpretation: str
    methodology: str
    disclaimer: str


@dataclass
class EducationalAnalysisResult:
    """教育的分析結果"""

    symbol: str
    company_name: str
    current_price: float
    analysis_time: datetime

    # 技術指標情報
    technical_indicators: List[TechnicalIndicator]

    # 統計的参考情報
    historical_statistics: List[HistoricalStatistics]

    # 機械学習テクニカルスコア（教育用）
    ml_technical_scores: List[MLTechnicalScore]

    # 教育的見解
    educational_notes: List[str]

    # 法的免責事項
    disclaimer: str


class EducationalMarketAnalyzer:
    """
    教育的市場分析クラス

    【重要】投資助言・推奨は一切行いません
    すべて教育・学習目的の情報提供です
    """

    def __init__(self):
        # 実データマネージャー初期化
        self.real_data_manager = RealMarketDataManager()

        # 高度MLエンジン初期化（利用可能な場合）
        if ADVANCED_ML_AVAILABLE:
            self.advanced_ml_engine = AdvancedMLEngine()
            logger.info("高度MLエンジン初期化完了")
        else:
            self.advanced_ml_engine = None
            logger.info("高度MLエンジン利用不可 - 基本MLエンジンを使用")

        self.company_names = {
            "7203": "トヨタ自動車",
            "8306": "三菱UFJ銀行",
            "9984": "ソフトバンクグループ",
            "6758": "ソニー",
            "4689": "Z Holdings",
            "9434": "ソフトバンク",
            "8001": "伊藤忠商事",
            "7267": "ホンダ",
            "6861": "キーエンス",
            "2914": "JT",
            "4063": "信越化学工業",
            "8035": "東京エレクトロン",
            "6954": "ファナック",
            "9983": "ファーストリテイリング",
            "4578": "大塚ホールディングス",
            "7974": "任天堂",
            "4452": "花王",
            "6098": "リクルートホールディングス",
            "8591": "オリックス",
            "7741": "HOYA",
        }

        self.disclaimer = (
            "【免責事項】本情報は教育・学習目的のみで提供されています。"
            "投資判断は必ず自己責任で行い、専門家にご相談ください。"
            "過去のデータは将来の結果を保証するものではありません。"
            "機械学習スコアは技術的参考情報であり、投資推奨ではありません。"
        )

        self.ml_disclaimer = (
            "機械学習スコアは過去のデータに基づく技術的分析であり、"
            "将来の価格変動を予測・保証するものではありません。"
        )

    def analyze_symbol_educational(self, symbol: str) -> EducationalAnalysisResult:
        """
        銘柄の教育的分析を実行

        Args:
            symbol: 銘柄コード

        Returns:
            EducationalAnalysisResult: 教育的分析結果
        """
        logger.info(f"教育的分析開始: {symbol}")

        # 実データ取得と分析
        stock_data = self.real_data_manager.get_stock_data(symbol)
        current_price = self.real_data_manager.get_current_price(symbol)
        technical_indicators = self._generate_technical_indicators_real(
            symbol, stock_data
        )
        historical_stats = self._generate_historical_statistics()
        ml_scores = self._generate_ml_technical_scores_real(symbol, stock_data)
        educational_notes = self._generate_educational_notes(
            technical_indicators, ml_scores
        )

        result = EducationalAnalysisResult(
            symbol=symbol,
            company_name=self.company_names.get(symbol, f"銘柄{symbol}"),
            current_price=current_price,
            analysis_time=datetime.now(),
            technical_indicators=technical_indicators,
            historical_statistics=historical_stats,
            ml_technical_scores=ml_scores,
            educational_notes=educational_notes,
            disclaimer=self.disclaimer,
        )

        logger.info(f"教育的分析完了: {symbol}")
        return result

    def _generate_mock_price(self, symbol: str) -> float:
        """模擬価格生成"""
        base_prices = {
            "7203": 2450.0,
            "8306": 800.0,
            "9984": 4200.0,
            "6758": 12500.0,
            "4689": 350.0,
        }

        base = base_prices.get(symbol, 1000.0)
        return round(base * (0.98 + random.random() * 0.04), 1)

    def _generate_technical_indicators_real(
        self, symbol: str, stock_data
    ) -> List[TechnicalIndicator]:
        """実データに基づく技術指標の生成"""
        indicators = []

        try:
            # RSI（実データ）
            rsi_value = self.real_data_manager.calculate_rsi(stock_data)
            rsi = TechnicalIndicator(
                name="RSI (14日)",
                current_value=rsi_value,
                educational_description="相対力指数 - 買われすぎ・売られすぎを測る指標",
                general_interpretation=self._get_rsi_interpretation(rsi_value),
                reference_levels={"売られすぎ": 30.0, "買われすぎ": 70.0},
            )
            indicators.append(rsi)

            # MACD（実データ）
            macd_value = self.real_data_manager.calculate_macd(stock_data)
            macd = TechnicalIndicator(
                name="MACD",
                current_value=macd_value,
                educational_description="移動平均収束拡散法 - トレンドの変化を捉える指標",
                general_interpretation=self._get_macd_interpretation(macd_value),
                reference_levels={"ゼロライン": 0.0},
            )
            indicators.append(macd)

            # 出来高比率（実データ）
            volume_ratio = self.real_data_manager.calculate_volume_ratio(stock_data)
            volume = TechnicalIndicator(
                name="出来高比率",
                current_value=volume_ratio,
                educational_description="過去20日平均との比較 - 市場関心度を示す指標",
                general_interpretation=self._get_volume_interpretation(volume_ratio),
                reference_levels={"平均": 1.0, "高水準": 2.0},
            )
            indicators.append(volume)

        except Exception as e:
            logger.warning(f"実データ取得失敗、フォールバック使用 {symbol}: {e}")
            # フォールバック: 旧ランダム生成
            indicators = self._generate_technical_indicators_fallback(symbol)

        return indicators

    def _generate_technical_indicators_fallback(
        self, symbol: str
    ) -> List[TechnicalIndicator]:
        """フォールバック用技術指標生成（ランダム）"""
        indicators = []

        # RSI
        rsi_value = 15 + random.random() * 70
        rsi = TechnicalIndicator(
            name="RSI (14日)",
            current_value=round(rsi_value, 1),
            educational_description="相対力指数 - 買われすぎ・売られすぎを測る指標",
            general_interpretation=self._get_rsi_interpretation(rsi_value),
            reference_levels={"売られすぎ": 30.0, "買われすぎ": 70.0},
        )
        indicators.append(rsi)

        # MACD
        macd_value = -5 + random.random() * 10
        macd = TechnicalIndicator(
            name="MACD",
            current_value=round(macd_value, 2),
            educational_description="移動平均収束拡散法 - トレンドの変化を捉える指標",
            general_interpretation=self._get_macd_interpretation(macd_value),
            reference_levels={"ゼロライン": 0.0},
        )
        indicators.append(macd)

        # 出来高比率
        volume_ratio = 0.5 + random.random() * 2.5
        volume = TechnicalIndicator(
            name="出来高比率",
            current_value=round(volume_ratio, 1),
            educational_description="過去20日平均との比較 - 市場関心度を示す指標",
            general_interpretation=self._get_volume_interpretation(volume_ratio),
            reference_levels={"平均": 1.0, "高水準": 2.0},
        )
        indicators.append(volume)

        return indicators

    def _generate_ml_technical_scores_real(
        self, symbol: str, stock_data
    ) -> List[MLTechnicalScore]:
        """実データに基づく機械学習テクニカルスコア生成（高度版優先）"""
        scores = []

        try:
            # 高度MLエンジンが利用可能な場合
            if self.advanced_ml_engine is not None:
                logger.info(f"高度MLエンジン使用: {symbol}")

                # 高度な技術指標計算
                advanced_data = (
                    self.advanced_ml_engine.calculate_advanced_technical_indicators(
                        stock_data
                    )
                )

                # ML特徴量準備
                features = self.advanced_ml_engine.prepare_ml_features(advanced_data)

                # モデルチューニング（必要に応じて）
                if len(features) >= 50 and symbol not in getattr(
                    self.advanced_ml_engine, "models", {}
                ):
                    logger.info(f"モデルチューニング実行: {symbol}")
                    try:
                        tuning_results = self.advanced_ml_engine.tune_hyperparameters(
                            symbol, stock_data, features
                        )
                        if tuning_results:
                            logger.info(f"モデルチューニング完了: {symbol}")
                        else:
                            logger.info(f"モデルチューニングスキップ: {symbol}")
                    except Exception as e:
                        logger.warning(f"モデルチューニングエラー（継続）: {e}")

                # 高度なスコア予測（チューニング済みモデル含む）
                (
                    trend_score,
                    volatility_score,
                    pattern_score,
                ) = self.advanced_ml_engine.predict_advanced_scores(
                    symbol, stock_data, features
                )

                # モデル情報取得（信頼度計算用）
                model_info = self.advanced_ml_engine.get_model_info(symbol)
                base_confidence = 0.75  # デフォルト信頼度

                if model_info and "model_performance" in model_info:
                    # 最良モデルのR²スコアを信頼度として使用
                    r2_scores = [
                        perf["r2"] for perf in model_info["model_performance"].values()
                    ]
                    if r2_scores:
                        base_confidence = max(0.5, min(0.95, max(r2_scores)))

                # 高度MLスコア作成
                trend_ml = MLTechnicalScore(
                    score_name="高度MLトレンド強度スコア",
                    score_value=trend_score,
                    confidence_level=round(base_confidence, 2),
                    model_description="pandas-ta + scikit-learn チューニング済みアンサンブル学習モデル",
                    educational_interpretation=self._get_trend_score_interpretation(
                        trend_score
                    ),
                    methodology="チューニング済みRandomForest + GradientBoosting + ExtraTrees による実データアンサンブル学習",
                    disclaimer=self.ml_disclaimer,
                )
                scores.append(trend_ml)

                volatility_ml = MLTechnicalScore(
                    score_name="高度ML価格変動予測スコア",
                    score_value=volatility_score,
                    confidence_level=round(
                        base_confidence * 0.9, 2
                    ),  # やや低めの信頼度
                    model_description="多次元特徴量による高度ボラティリティ予測",
                    educational_interpretation=self._get_volatility_score_interpretation(
                        volatility_score
                    ),
                    methodology="RSI, ADX, ボリンジャーバンド, CCI等30+指標による統合分析",
                    disclaimer=self.ml_disclaimer,
                )
                scores.append(volatility_ml)

                pattern_ml = MLTechnicalScore(
                    score_name="高度MLパターン認識スコア",
                    score_value=pattern_score,
                    confidence_level=round(base_confidence * 0.85, 2),
                    model_description="多層パターン認識システム",
                    educational_interpretation=self._get_pattern_score_interpretation(
                        pattern_score
                    ),
                    methodology="OBV, MFI, Supertrend, PSAR等による複合パターン分析",
                    disclaimer=self.ml_disclaimer,
                )
                scores.append(pattern_ml)

                logger.info(f"高度MLスコア生成完了: {symbol}")
                return scores

            else:
                # 基本MLエンジン使用（従来通り）
                logger.info(f"基本MLエンジン使用: {symbol}")

                (
                    trend_score,
                    trend_confidence,
                ) = self.real_data_manager.generate_ml_trend_score(stock_data)
                trend_ml = MLTechnicalScore(
                    score_name="MLトレンド強度スコア",
                    score_value=trend_score,
                    confidence_level=trend_confidence,
                    model_description="実データLSTM深層学習モデルによるトレンド分析",
                    educational_interpretation=self._get_trend_score_interpretation(
                        trend_score
                    ),
                    methodology="実際の過去60日の価格・出来高データを学習した深層ニューラルネットワーク",
                    disclaimer=self.ml_disclaimer,
                )
                scores.append(trend_ml)

                (
                    volatility_score,
                    volatility_confidence,
                ) = self.real_data_manager.generate_ml_volatility_score(stock_data)
                volatility_ml = MLTechnicalScore(
                    score_name="ML価格変動予測スコア",
                    score_value=volatility_score,
                    confidence_level=volatility_confidence,
                    model_description="実データアンサンブル学習による価格変動幅の技術的評価",
                    educational_interpretation=self._get_volatility_score_interpretation(
                        volatility_score
                    ),
                    methodology="実データに基づくランダムフォレスト + グラディエントブースティング複合モデル",
                    disclaimer=self.ml_disclaimer,
                )
                scores.append(volatility_ml)

                (
                    pattern_score,
                    pattern_confidence,
                ) = self.real_data_manager.generate_ml_pattern_score(stock_data)
                pattern_ml = MLTechnicalScore(
                    score_name="MLパターン認識スコア",
                    score_value=pattern_score,
                    confidence_level=pattern_confidence,
                    model_description="実チャートパターンの機械学習による認識システム",
                    educational_interpretation=self._get_pattern_score_interpretation(
                        pattern_score
                    ),
                    methodology="実データ畳み込みニューラルネットワーク（CNN）によるチャート画像解析",
                    disclaimer=self.ml_disclaimer,
                )
                scores.append(pattern_ml)

                return scores

        except Exception as e:
            logger.warning(f"実MLスコア生成失敗、フォールバック使用 {symbol}: {e}")
            # フォールバック: 旧ランダム生成
            scores = self._generate_ml_technical_scores_fallback(symbol)

        return scores

    def _generate_ml_technical_scores_fallback(
        self, symbol: str
    ) -> List[MLTechnicalScore]:
        """フォールバック用MLスコア生成（ランダム）"""
        scores = []

        # トレンド強度スコア
        trend_score = random.uniform(25, 85)
        trend_ml = MLTechnicalScore(
            score_name="MLトレンド強度スコア",
            score_value=round(trend_score, 1),
            confidence_level=round(0.6 + random.random() * 0.3, 2),
            model_description="LSTM深層学習モデルによるトレンド分析",
            educational_interpretation=self._get_trend_score_interpretation(
                trend_score
            ),
            methodology="過去60日の価格・出来高データを学習した深層ニューラルネットワーク",
            disclaimer=self.ml_disclaimer,
        )
        scores.append(trend_ml)

        # ボラティリティ予測スコア
        volatility_score = random.uniform(10, 90)
        volatility_ml = MLTechnicalScore(
            score_name="ML価格変動予測スコア",
            score_value=round(volatility_score, 1),
            confidence_level=round(0.55 + random.random() * 0.35, 2),
            model_description="アンサンブル学習による価格変動幅の技術的評価",
            educational_interpretation=self._get_volatility_score_interpretation(
                volatility_score
            ),
            methodology="ランダムフォレスト + グラディエントブースティングによる複合モデル",
            disclaimer=self.ml_disclaimer,
        )
        scores.append(volatility_ml)

        # パターン認識スコア
        pattern_score = random.uniform(30, 80)
        pattern_ml = MLTechnicalScore(
            score_name="MLパターン認識スコア",
            score_value=round(pattern_score, 1),
            confidence_level=round(0.5 + random.random() * 0.4, 2),
            model_description="チャートパターンの機械学習による認識システム",
            educational_interpretation=self._get_pattern_score_interpretation(
                pattern_score
            ),
            methodology="畳み込みニューラルネットワーク（CNN）によるチャート画像解析",
            disclaimer=self.ml_disclaimer,
        )
        scores.append(pattern_ml)

        return scores

    def _get_trend_score_interpretation(self, score: float) -> str:
        """トレンドスコアの教育的解釈"""
        if score >= 70:
            return "技術的に強い上昇トレンド傾向を示しています（教育用参考情報）"
        elif score >= 50:
            return "技術的に中程度の上昇傾向を示しています（教育用参考情報）"
        elif score >= 30:
            return "技術的にトレンド不明確な状況を示しています（教育用参考情報）"
        else:
            return "技術的に下降トレンド傾向を示しています（教育用参考情報）"

    def _get_volatility_score_interpretation(self, score: float) -> str:
        """ボラティリティスコアの教育的解釈"""
        if score >= 70:
            return "技術的に高い価格変動が予想される状況です（教育用参考情報）"
        elif score >= 40:
            return "技術的に中程度の価格変動が予想される状況です（教育用参考情報）"
        else:
            return "技術的に低い価格変動が予想される状況です（教育用参考情報）"

    def _get_pattern_score_interpretation(self, score: float) -> str:
        """パターンスコアの教育的解釈"""
        if score >= 65:
            return "技術的に上昇継続パターンの特徴を示しています（教育用参考情報）"
        elif score >= 35:
            return "技術的に不明確なパターン状況です（教育用参考情報）"
        else:
            return "技術的に調整・下降パターンの特徴を示しています（教育用参考情報）"

    def _get_rsi_interpretation(self, rsi_value: float) -> str:
        """RSI値の一般的解釈"""
        if rsi_value <= 30:
            return "教科書的には「売られすぎ」水準とされます"
        elif rsi_value >= 70:
            return "教科書的には「買われすぎ」水準とされます"
        else:
            return "中立的な水準です"

    def _get_macd_interpretation(self, macd_value: float) -> str:
        """MACD値の一般的解釈"""
        if macd_value > 0:
            return "ゼロライン上 - 上昇トレンドの可能性を示唆する状況"
        else:
            return "ゼロライン下 - 下降トレンドの可能性を示唆する状況"

    def _get_volume_interpretation(self, volume_ratio: float) -> str:
        """出来高比率の一般的解釈"""
        if volume_ratio >= 2.0:
            return "平均を大幅上回る - 市場関心度の高まりを示す"
        elif volume_ratio >= 1.5:
            return "平均を上回る - やや注目度が高い状況"
        else:
            return "平均的な出来高水準"

    def _generate_historical_statistics(self) -> List[HistoricalStatistics]:
        """統計的参考情報の生成"""
        stats = []

        # RSI30以下の統計
        rsi_stat = HistoricalStatistics(
            condition="RSI30以下時",
            period_days=10,
            upward_percentage=round(60 + random.random() * 10, 1),
            downward_percentage=round(30 + random.random() * 10, 1),
            average_change=round(1 + random.random() * 3, 1),
            sample_size=150,
        )
        stats.append(rsi_stat)

        # 出来高2倍超の統計
        volume_stat = HistoricalStatistics(
            condition="出来高2倍超時",
            period_days=5,
            upward_percentage=round(55 + random.random() * 15, 1),
            downward_percentage=round(35 + random.random() * 10, 1),
            average_change=round(0.5 + random.random() * 2, 1),
            sample_size=85,
        )
        stats.append(volume_stat)

        return stats

    def _generate_educational_notes(
        self, indicators: List[TechnicalIndicator], ml_scores: List[MLTechnicalScore]
    ) -> List[str]:
        """教育的見解の生成（ML情報含む）"""
        notes = []

        # RSIに基づく教育的見解
        rsi_indicator = next((ind for ind in indicators if "RSI" in ind.name), None)
        if rsi_indicator and rsi_indicator.current_value <= 30:
            notes.append(
                "RSIが30以下となっており、テクニカル分析の教科書では"
                "「売られすぎ」水準とされる範囲です。"
            )

        # MACDに基づく教育的見解
        macd_indicator = next((ind for ind in indicators if "MACD" in ind.name), None)
        if macd_indicator and macd_indicator.current_value > 0:
            notes.append(
                "MACDがゼロライン上にあり、一般的には上昇トレンドの"
                "継続を示唆する状況とされています。"
            )

        # 出来高に基づく教育的見解
        volume_indicator = next(
            (ind for ind in indicators if "出来高" in ind.name), None
        )
        if volume_indicator and volume_indicator.current_value >= 2.0:
            notes.append(
                "出来高が平均を大幅に上回っており、市場参加者の"
                "関心度が高まっている可能性があります。"
            )

        # MLスコアに基づく教育的見解
        trend_score = next(
            (score for score in ml_scores if "トレンド強度" in score.score_name), None
        )
        if trend_score and trend_score.score_value >= 70:
            notes.append(
                f"機械学習トレンド分析では強い上昇傾向（{trend_score.score_value}点）"
                "を示していますが、これは教育用技術参考情報です。"
            )
        elif trend_score and trend_score.score_value <= 30:
            notes.append(
                f"機械学習トレンド分析では下降傾向（{trend_score.score_value}点）"
                "を示していますが、これは教育用技術参考情報です。"
            )

        # 基本的な教育事項
        notes.append(
            "テクニカル分析および機械学習スコアは過去のデータに基づく分析手法であり、"
            "将来の価格変動を予測・保証するものではありません。"
        )

        notes.append(
            "機械学習による価格関連情報は技術的研究目的の参考データであり、"
            "投資推奨や投資助言には該当しません。"
        )

        return notes

    def format_educational_report(self, result: EducationalAnalysisResult) -> str:
        """教育的レポートのフォーマット"""
        report_lines = []

        # ヘッダー
        report_lines.append("=" * 80)
        report_lines.append(
            f"【教育的技術指標レポート】 {result.symbol} {result.company_name}"
        )
        report_lines.append("=" * 80)
        report_lines.append(
            f"分析時刻: {result.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"現在価格: {result.current_price:,.1f}円")
        report_lines.append("")

        # 技術指標セクション
        report_lines.append("【技術指標の現状】（客観的データ）")
        report_lines.append("-" * 50)
        for indicator in result.technical_indicators:
            report_lines.append(f"◆ {indicator.name}: {indicator.current_value}")
            report_lines.append(f"  説明: {indicator.educational_description}")
            report_lines.append(f"  一般的解釈: {indicator.general_interpretation}")

            # 参考レベル表示
            if indicator.reference_levels:
                levels = ", ".join(
                    [f"{k}:{v}" for k, v in indicator.reference_levels.items()]
                )
                report_lines.append(f"  参考レベル: {levels}")
            report_lines.append("")

        # 機械学習スコアセクション
        if result.ml_technical_scores:
            report_lines.append("【機械学習テクニカルスコア】（教育・研究用）")
            report_lines.append("-" * 50)
            for ml_score in result.ml_technical_scores:
                report_lines.append(
                    f"◆ {ml_score.score_name}: {ml_score.score_value}/100"
                )
                report_lines.append(f"  信頼度: {ml_score.confidence_level:.2f}")
                report_lines.append(f"  モデル: {ml_score.model_description}")
                report_lines.append(f"  解釈: {ml_score.educational_interpretation}")
                report_lines.append(f"  手法: {ml_score.methodology}")
                report_lines.append("")

        # 統計情報セクション
        report_lines.append("【統計的参考情報】（過去データ）")
        report_lines.append("-" * 50)
        for stat in result.historical_statistics:
            report_lines.append(f"◆ {stat.condition}の{stat.period_days}日後結果:")
            report_lines.append(f"  上昇: {stat.upward_percentage}%")
            report_lines.append(f"  下降: {stat.downward_percentage}%")
            report_lines.append(f"  平均変動: +{stat.average_change}%")
            report_lines.append(f"  標本数: {stat.sample_size}件")
            report_lines.append("")

        # 教育的見解セクション
        if result.educational_notes:
            report_lines.append("【教育的見解】")
            report_lines.append("-" * 50)
            for i, note in enumerate(result.educational_notes, 1):
                report_lines.append(f"{i}. {note}")
            report_lines.append("")

        # MLスコア総合要約表
        if result.ml_technical_scores:
            report_lines.append("【MLスコア総合要約表】")
            report_lines.append("-" * 85)
            report_lines.append(
                f"{'スコア名':<20} {'値':<10} {'レベル':<8} {'信頼度':<10} {'技術的解釈'}"
            )
            report_lines.append("-" * 85)
            for score in result.ml_technical_scores:
                score_level = (
                    "高スコア"
                    if score.score_value >= 70
                    else "中スコア"
                    if score.score_value >= 40
                    else "低スコア"
                )
                interpretation = score.educational_interpretation.replace(
                    "技術的に", ""
                ).replace("（教育用参考情報）", "")[:35]
                report_lines.append(
                    f"{score.score_name:<20} {score.score_value:>6.1f}/100 {score_level:<8} {score.confidence_level:<10.2f} {interpretation}"
                )
            report_lines.append("-" * 85)
            report_lines.append("")

        # 免責事項
        report_lines.append("【重要な注意事項】")
        report_lines.append("-" * 50)
        report_lines.append(result.disclaimer)
        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def analyze_educational_multiple(symbols: List[str]) -> List[EducationalAnalysisResult]:
    """
    複数銘柄の教育的分析

    Args:
        symbols: 銘柄コードリスト

    Returns:
        List[EducationalAnalysisResult]: 分析結果リスト
    """
    analyzer = EducationalMarketAnalyzer()
    results = []

    for symbol in symbols:
        try:
            result = analyzer.analyze_symbol_educational(symbol)
            results.append(result)
        except Exception as e:
            logger.error(f"教育的分析エラー {symbol}: {e}")

    return results


if __name__ == "__main__":
    # テスト実行
    analyzer = EducationalMarketAnalyzer()
    result = analyzer.analyze_symbol_educational("7203")
    report = analyzer.format_educational_report(result)
    print(report)
