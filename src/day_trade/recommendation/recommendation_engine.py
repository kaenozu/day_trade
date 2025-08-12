#!/usr/bin/env python3
"""
推奨銘柄選定エンジン

Issue #455: 既存のテクニカル指標とML予測を組み合わせた
総合的な銘柄推奨システム
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

# 決定論的実行のためのシード設定
import random
random.seed(42)
np.random.seed(42)

from ..analysis.technical_indicators_unified import TechnicalIndicatorsManager
from ..data.advanced_ml_engine import AdvancedMLEngine
from ..data.batch_data_fetcher import AdvancedBatchDataFetcher, DataRequest
from ..utils.stock_name_helper import get_stock_helper, format_stock_display
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class RecommendationAction(Enum):
    """推奨アクション"""
    STRONG_BUY = "🔥 今すぐ買い"
    BUY = "📈 買い"
    HOLD = "⏸️ 様子見"
    SELL = "📉 売り"
    STRONG_SELL = "⚠️ 今すぐ売り"


@dataclass
class StockRecommendation:
    """株式推奨情報"""
    symbol: str
    name: str
    composite_score: float  # 総合スコア (0-100)
    technical_score: float  # テクニカル指標スコア (0-100)
    ml_score: float  # ML予測スコア (0-100)
    action: RecommendationAction
    confidence: float  # 信頼度 (0-100)
    reasons: List[str]  # 推奨理由リスト
    risk_level: str  # リスクレベル ("低", "中", "高")
    price_target: Optional[float] = None  # 目標価格
    stop_loss: Optional[float] = None  # ストップロス


class RecommendationEngine:
    """推奨銘柄選定エンジン"""

    def __init__(self):
        """初期化"""
        self.technical_manager = TechnicalIndicatorsManager()
        self.ml_engine = AdvancedMLEngine()
        self.data_fetcher = AdvancedBatchDataFetcher(max_workers=4)
        self.stock_helper = get_stock_helper()

        # スコア重み付け設定
        self.weights = {
            'technical': 0.6,  # テクニカル指標の重み
            'ml': 0.4,         # ML予測の重み
        }

        # リスク評価閾値
        self.risk_thresholds = {
            'high_volatility': 5.0,    # 高ボラティリティ閾値(%)
            'extreme_move': 10.0,      # 極端な価格変動閾値(%)
            'volume_spike': 3.0,       # 出来高急増閾値(倍)
        }

        logger.info("推奨銘柄選定エンジン初期化完了")

    async def analyze_all_stocks(self, symbols: Optional[List[str]] = None) -> List[StockRecommendation]:
        """
        全銘柄の分析と推奨銘柄選定

        Args:
            symbols: 分析対象銘柄リスト（Noneの場合は全銘柄）

        Returns:
            推奨銘柄リスト（スコア順）
        """
        start_time = time.time()

        if symbols is None:
            # settings.jsonから全銘柄を取得
            symbols = self._get_all_symbols()

        logger.info(f"推奨銘柄分析開始: {len(symbols)} 銘柄")

        try:
            # 1. バッチデータ取得
            stock_data = await self._fetch_batch_data(symbols)

            # 2. 各銘柄の分析
            recommendations = []
            for symbol in symbols:
                if symbol in stock_data and stock_data[symbol].success:
                    try:
                        recommendation = await self._analyze_single_stock(
                            symbol, stock_data[symbol].data
                        )
                        if recommendation:
                            recommendations.append(recommendation)

                    except Exception as e:
                        logger.warning(f"銘柄分析エラー {format_stock_display(symbol)}: {e}")
                else:
                    logger.warning(f"データ取得失敗: {format_stock_display(symbol)}")

            # 3. スコア順ソート
            recommendations.sort(key=lambda x: x.composite_score, reverse=True)

            elapsed_time = time.time() - start_time
            logger.info(f"推奨銘柄分析完了: {len(recommendations)} 銘柄 ({elapsed_time:.2f}秒)")

            return recommendations

        except Exception as e:
            logger.error(f"推奨銘柄分析エラー: {e}")
            return []

    async def _fetch_batch_data(self, symbols: List[str]) -> Dict[str, any]:
        """バッチデータ取得"""
        requests = [
            DataRequest(
                symbol=symbol,
                period="60d",
                preprocessing=True,
                priority=3
            )
            for symbol in symbols
        ]

        return self.data_fetcher.fetch_batch(requests, use_parallel=True)

    async def _analyze_single_stock(self, symbol: str, data: pd.DataFrame) -> Optional[StockRecommendation]:
        """単一銘柄の分析"""
        try:
            # 1. テクニカル指標分析
            technical_score, technical_reasons = await self._calculate_technical_score(symbol, data)

            # 2. ML予測分析
            ml_score, ml_reasons = await self._calculate_ml_score(symbol, data)

            # 3. 総合スコア計算
            composite_score = (
                technical_score * self.weights['technical'] +
                ml_score * self.weights['ml']
            )

            # 4. リスク評価
            risk_level = self._assess_risk_level(data)

            # 5. 推奨アクション決定
            action = self._determine_action(composite_score, risk_level)

            # 6. 信頼度計算
            confidence = self._calculate_confidence(technical_score, ml_score, data)

            # 7. 推奨理由統合
            all_reasons = technical_reasons + ml_reasons

            # 8. 価格目標・ストップロス設定
            current_price = data['終値'].iloc[-1] if '終値' in data.columns else data['Close'].iloc[-1]
            price_target, stop_loss = self._calculate_price_targets(current_price, composite_score, risk_level)

            return StockRecommendation(
                symbol=symbol,
                name=self.stock_helper.get_stock_name(symbol),
                composite_score=composite_score,
                technical_score=technical_score,
                ml_score=ml_score,
                action=action,
                confidence=confidence,
                reasons=all_reasons[:5],  # TOP5理由のみ
                risk_level=risk_level,
                price_target=price_target,
                stop_loss=stop_loss
            )

        except Exception as e:
            logger.error(f"単一銘柄分析エラー {format_stock_display(symbol)}: {e}")
            return None

    async def _calculate_technical_score(self, symbol: str, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """テクニカル指標スコア計算"""
        try:
            # テクニカル指標計算
            indicators = self.technical_manager.calculate_indicators(
                data=data,
                indicators=['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
                period=20
            )

            score = 50.0  # ベーススコア
            reasons = []

            if isinstance(indicators, dict):
                # SMA分析
                if 'sma' in indicators:
                    sma_score, sma_reason = self._analyze_sma_signal(data, indicators['sma'])
                    score += sma_score
                    if sma_reason:
                        reasons.append(sma_reason)

                # RSI分析
                if 'rsi' in indicators:
                    rsi_score, rsi_reason = self._analyze_rsi_signal(indicators['rsi'])
                    score += rsi_score
                    if rsi_reason:
                        reasons.append(rsi_reason)

                # MACD分析
                if 'macd' in indicators:
                    macd_score, macd_reason = self._analyze_macd_signal(indicators['macd'])
                    score += macd_score
                    if macd_reason:
                        reasons.append(macd_reason)

                # ボリンジャーバンド分析
                if 'bollinger_bands' in indicators:
                    bb_score, bb_reason = self._analyze_bollinger_signal(data, indicators['bollinger_bands'])
                    score += bb_score
                    if bb_reason:
                        reasons.append(bb_reason)

            # 出来高分析
            volume_score, volume_reason = self._analyze_volume_signal(data)
            score += volume_score
            if volume_reason:
                reasons.append(volume_reason)

            # スコア正規化 (0-100)
            score = max(0, min(100, score))

            return score, reasons

        except Exception as e:
            logger.warning(f"テクニカル指標計算エラー {format_stock_display(symbol)}: {e}")
            return 50.0, ["テクニカル分析データ不足"]

    async def _calculate_ml_score(self, symbol: str, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """ML予測スコア計算"""
        try:
            # ML指標計算
            ml_indicators = self.ml_engine.calculate_advanced_technical_indicators(data, symbol)

            reasons = []
            total_score = 0
            count = 0

            if ml_indicators:
                # トレンド強度
                if 'trend_strength' in ml_indicators:
                    trend_score = ml_indicators['trend_strength']
                    total_score += trend_score
                    count += 1

                    if trend_score > 70:
                        reasons.append("AI予測: 強い上昇トレンド")
                    elif trend_score < 30:
                        reasons.append("AI予測: 強い下降トレンド")

                # ボラティリティ予測
                if 'volatility_prediction' in ml_indicators:
                    vol_score = ml_indicators['volatility_prediction']
                    total_score += vol_score
                    count += 1

                    if vol_score > 70:
                        reasons.append("AI予測: 高ボラティリティ期待")

                # パターン認識
                if 'pattern_recognition' in ml_indicators:
                    pattern_score = ml_indicators['pattern_recognition']
                    total_score += pattern_score
                    count += 1

                    if pattern_score > 70:
                        reasons.append("AI予測: 有望パターン検出")

            # 平均スコア計算
            final_score = total_score / count if count > 0 else 50.0

            return final_score, reasons

        except Exception as e:
            logger.warning(f"ML予測計算エラー {format_stock_display(symbol)}: {e}")
            return 50.0, ["ML予測データ不足"]

    def _analyze_sma_signal(self, data: pd.DataFrame, sma_result) -> Tuple[float, Optional[str]]:
        """SMA信号分析"""
        try:
            if hasattr(sma_result, 'values') and 'sma' in sma_result.values:
                sma_values = sma_result.values['sma']
                if len(sma_values) > 1:
                    close_price = data['終値'].iloc[-1] if '終値' in data.columns else data['Close'].iloc[-1]
                    current_sma = sma_values[-1]

                    if close_price > current_sma * 1.02:  # 2%以上上抜け
                        return 15.0, "SMA上抜けシグナル"
                    elif close_price < current_sma * 0.98:  # 2%以上下抜け
                        return -15.0, "SMA下抜けシグナル"

            return 0.0, None

        except Exception:
            return 0.0, None

    def _analyze_rsi_signal(self, rsi_result) -> Tuple[float, Optional[str]]:
        """RSI信号分析"""
        try:
            if hasattr(rsi_result, 'values') and 'rsi' in rsi_result.values:
                rsi_values = rsi_result.values['rsi']
                if len(rsi_values) > 0:
                    current_rsi = rsi_values[-1]

                    if current_rsi < 30:  # 売られすぎ
                        return 10.0, "RSI売られすぎ（反発期待）"
                    elif current_rsi > 70:  # 買われすぎ
                        return -10.0, "RSI買われすぎ（調整懸念）"

            return 0.0, None

        except Exception:
            return 0.0, None

    def _analyze_macd_signal(self, macd_result) -> Tuple[float, Optional[str]]:
        """MACD信号分析"""
        try:
            if hasattr(macd_result, 'values'):
                macd_values = macd_result.values.get('macd', [])
                signal_values = macd_result.values.get('signal', [])

                if len(macd_values) > 1 and len(signal_values) > 1:
                    # ゴールデンクロス/デッドクロス判定
                    if macd_values[-1] > signal_values[-1] and macd_values[-2] <= signal_values[-2]:
                        return 12.0, "MACDゴールデンクロス"
                    elif macd_values[-1] < signal_values[-1] and macd_values[-2] >= signal_values[-2]:
                        return -12.0, "MACDデッドクロス"

            return 0.0, None

        except Exception:
            return 0.0, None

    def _analyze_bollinger_signal(self, data: pd.DataFrame, bb_result) -> Tuple[float, Optional[str]]:
        """ボリンジャーバンド信号分析"""
        try:
            if hasattr(bb_result, 'values'):
                upper = bb_result.values.get('upper', [])
                lower = bb_result.values.get('lower', [])

                if len(upper) > 0 and len(lower) > 0:
                    close_price = data['終値'].iloc[-1] if '終値' in data.columns else data['Close'].iloc[-1]

                    if close_price <= lower[-1]:  # 下限タッチ
                        return 8.0, "ボリンジャーバンド下限反発"
                    elif close_price >= upper[-1]:  # 上限タッチ
                        return -8.0, "ボリンジャーバンド上限到達"

            return 0.0, None

        except Exception:
            return 0.0, None

    def _analyze_volume_signal(self, data: pd.DataFrame) -> Tuple[float, Optional[str]]:
        """出来高信号分析"""
        try:
            volume_col = '出来高' if '出来高' in data.columns else 'Volume'
            if volume_col in data.columns and len(data) > 20:
                current_volume = data[volume_col].iloc[-1]
                avg_volume = data[volume_col].rolling(20).mean().iloc[-1]

                if current_volume > avg_volume * 2:  # 2倍以上
                    return 10.0, "出来高急増"
                elif current_volume > avg_volume * 1.5:  # 1.5倍以上
                    return 5.0, "出来高増加"

            return 0.0, None

        except Exception:
            return 0.0, None

    def _assess_risk_level(self, data: pd.DataFrame) -> str:
        """リスクレベル評価"""
        try:
            # ボラティリティ計算
            close_col = '終値' if '終値' in data.columns else 'Close'
            returns = data[close_col].pct_change().dropna()

            if len(returns) > 10:
                volatility = returns.std() * 100  # %変換

                if volatility > self.risk_thresholds['high_volatility']:
                    return "高"
                elif volatility > self.risk_thresholds['high_volatility'] / 2:
                    return "中"
                else:
                    return "低"

            return "中"

        except Exception:
            return "中"

    def _determine_action(self, score: float, risk_level: str) -> RecommendationAction:
        """推奨アクション決定"""
        # リスク調整
        risk_penalty = {"高": 10, "中": 5, "低": 0}
        adjusted_score = score - risk_penalty.get(risk_level, 5)

        if adjusted_score >= 80:
            return RecommendationAction.STRONG_BUY
        elif adjusted_score >= 65:
            return RecommendationAction.BUY
        elif adjusted_score >= 35:
            return RecommendationAction.HOLD
        elif adjusted_score >= 20:
            return RecommendationAction.SELL
        else:
            return RecommendationAction.STRONG_SELL

    def _calculate_confidence(self, technical_score: float, ml_score: float, data: pd.DataFrame) -> float:
        """信頼度計算"""
        try:
            # スコア一致度
            score_consistency = 100 - abs(technical_score - ml_score)

            # データ量
            data_adequacy = min(100, len(data) / 60 * 100)  # 60日分を100%とする

            # 総合信頼度
            confidence = (score_consistency * 0.6 + data_adequacy * 0.4)
            return max(0, min(100, confidence))

        except Exception:
            return 50.0

    def _calculate_price_targets(self, current_price: float, score: float, risk_level: str) -> Tuple[float, float]:
        """価格目標・ストップロス計算"""
        try:
            # リスクレベル別の目標・ロス率
            risk_params = {
                "低": {"target": 0.05, "stop": 0.03},
                "中": {"target": 0.08, "stop": 0.05},
                "高": {"target": 0.12, "stop": 0.08}
            }

            params = risk_params.get(risk_level, risk_params["中"])

            # スコア調整
            score_multiplier = score / 100
            target_rate = params["target"] * score_multiplier
            stop_rate = params["stop"]

            price_target = current_price * (1 + target_rate)
            stop_loss = current_price * (1 - stop_rate)

            return round(price_target, 0), round(stop_loss, 0)

        except Exception:
            return None, None

    def _get_all_symbols(self) -> List[str]:
        """全銘柄リスト取得"""
        try:
            # settings.jsonから銘柄リストを取得
            import json
            from pathlib import Path

            config_path = Path(__file__).parent.parent.parent.parent / "config" / "settings.json"

            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                symbols = []

                # stock_infoから銘柄を取得
                if 'stock_info' in settings:
                    for symbol, info in settings['stock_info'].items():
                        if isinstance(info, dict) and info.get('name'):
                            symbols.append(symbol)

                # watchlist.symbolsから取得
                if not symbols and 'watchlist' in settings and 'symbols' in settings['watchlist']:
                    for stock in settings['watchlist']['symbols']:
                        if isinstance(stock, dict) and 'code' in stock:
                            symbols.append(stock['code'])

                # symbolsリストからも取得（フォールバック）
                if not symbols and 'symbols' in settings:
                    symbols = settings['symbols']

                logger.info(f"設定ファイルから {len(symbols)} 銘柄を取得")
                if symbols:
                    return symbols

        except Exception as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}")

        # フォールバック: デフォルト銘柄
        default_symbols = ["7203", "8306", "9984", "6758", "4689"]
        logger.info(f"デフォルト銘柄を使用: {len(default_symbols)} 銘柄")
        return default_symbols

    def get_top_recommendations(self, recommendations: List[StockRecommendation], limit: int = 10) -> List[StockRecommendation]:
        """TOP推奨銘柄取得"""
        # 買い推奨のみフィルタ
        buy_recommendations = [
            r for r in recommendations
            if r.action in [RecommendationAction.STRONG_BUY, RecommendationAction.BUY]
        ]

        return buy_recommendations[:limit]

    def close(self):
        """リソース解放"""
        if hasattr(self.data_fetcher, 'close'):
            self.data_fetcher.close()
        logger.info("推奨銘柄選定エンジン終了")


# 便利関数
async def get_daily_recommendations(limit: int = 10) -> List[StockRecommendation]:
    """日次推奨銘柄取得"""
    engine = RecommendationEngine()
    try:
        all_recommendations = await engine.analyze_all_stocks()
        return engine.get_top_recommendations(all_recommendations, limit)
    finally:
        engine.close()


if __name__ == "__main__":
    # テスト実行
    async def test_recommendations():
        print("推奨銘柄選定エンジン テスト")

        recommendations = await get_daily_recommendations(5)

        print(f"\nTOP {len(recommendations)} 推奨銘柄:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.symbol} ({rec.name})")
            print(f"   スコア: {rec.composite_score:.1f}点")
            print(f"   アクション: {rec.action.value}")
            print(f"   理由: {', '.join(rec.reasons[:3])}")
            print(f"   リスク: {rec.risk_level}")
            if rec.price_target:
                print(f"   目標価格: {rec.price_target:.0f}円")
            if rec.stop_loss:
                print(f"   ストップロス: {rec.stop_loss:.0f}円")
            print()

    asyncio.run(test_recommendations())