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
from ..utils.stock_name_helper import get_stock_helper
from ..utils.logging_config import get_context_logger
# Issue #487対応: スマート銘柄選択統合
from ..automation.smart_symbol_selector import get_smart_selected_symbols
# Issue #464対応: アンサンブルシステム統合
from ..ml.ensemble_system import EnsembleSystem, EnsembleConfig

logger = get_context_logger(__name__)


class RecommendationAction(Enum):
    """推奨アクション"""
    STRONG_BUY = "[STRONG_BUY] 今すぐ買い"
    BUY = "[BUY] 買い"
    HOLD = "[HOLD] 様子見"
    SELL = "[SELL] 売り"
    STRONG_SELL = "[STRONG_SELL] 今すぐ売り"


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

        # Issue #464対応: アンサンブルシステム統合
        self.ensemble_system = None
        self._initialize_ensemble_system()

        # Issue #464対応: アンサンブル統合によるスコア重み付け最適化
        self.weights = {
            'technical': 0.4,   # テクニカル指標の重み（アンサンブル導入により調整）
            'ml': 0.3,          # 既存ML予測の重み（アンサンブル導入により調整）
            'ensemble': 0.3,    # アンサンブル予測の重み（新規追加）
        }

        # リスク評価閾値
        self.risk_thresholds = {
            'high_volatility': 5.0,    # 高ボラティリティ閾値(%)
            'extreme_move': 10.0,      # 極端な価格変動閾値(%)
            'volume_spike': 3.0,       # 出来高急増閾値(倍)
        }

        # Issue #587対応: 価格目標・ストップロス計算パラメータ
        self.price_target_params = {
            "低": {"target_factor": 0.05, "stop_loss_factor": 0.03},
            "中": {"target_factor": 0.08, "stop_loss_factor": 0.05},
            "高": {"target_factor": 0.12, "stop_loss_factor": 0.08}
        }

        # Issue #586対応: 信頼度計算の重み
        self.confidence_weights = {
            'score_consistency': 0.6,
            'data_adequacy': 0.4,
        }

        # Issue #587対応: 高度リスク管理戦略設定
        self.risk_management_config = {
            'use_atr_based_stops': True,    # ATRベースストップロス使用
            'atr_multiplier': 2.0,          # ATR乗数
            'max_position_risk': 0.02,      # 最大ポジションリスク(2%)
            'portfolio_correlation_limit': 0.7,  # ポートフォリオ相関限界
        }

        # Issue #585対応: アクション決定ロジック設定
        self.action_thresholds = {
            'strong_buy': 80,
            'buy': 65,
            'hold_upper': 35,
            'sell': 20,
            # strong_sell は sell 未満
        }

        self.risk_penalties = {
            '高': 10,
            '中': 5,
            '低': 0
        }

        # Issue #582対応: シグナル閾値設定
        self.signal_thresholds = {
            'sma_breakout': 0.02,       # SMA突破閾値(2%)
            'rsi_oversold': 30,         # RSI売られすぎ閾値
            'rsi_overbought': 70,       # RSI買われすぎ閾値
            'volume_spike': 2.0,        # 出来高急増閾値(倍率)
            'macd_signal_strength': 0.1, # MACDシグナル強度閾値
        }

        logger.info("推奨銘柄選定エンジン初期化完了")

    def _initialize_ensemble_system(self):
        """
        Issue #464対応: アンサンブルシステム初期化

        最高精度を目指すためのアンサンブル学習システムを構築
        """
        try:
            # アンサンブル設定（最適化された設定）
            ensemble_config = EnsembleConfig(
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                use_lstm_transformer=False,  # AdvancedMLEngineと重複回避
                enable_stacking=True,
                enable_dynamic_weighting=True,
                random_forest_params={
                    'n_estimators': 200,
                    'max_depth': 15,
                    'enable_hyperopt': True
                },
                gradient_boosting_params={
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'enable_hyperopt': True,
                    'early_stopping': True
                },
                svr_params={
                    'kernel': 'rbf',
                    'enable_hyperopt': True
                }
            )

            self.ensemble_system = EnsembleSystem(ensemble_config)
            self._ensemble_trained = False

            logger.info("アンサンブルシステム初期化完了（3モデル統合）")

        except Exception as e:
            logger.warning(f"アンサンブルシステム初期化失敗: {e}")
            self.ensemble_system = None
            # フォールバック：従来の重み付けに戻す
            self.weights = {
                'technical': 0.6,
                'ml': 0.4,
                'ensemble': 0.0,
            }

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
            requests = [
                DataRequest(
                    symbol=symbol,
                    period="60d",
                    preprocessing=True,
                    priority=3
                )
                for symbol in symbols
            ]
            stock_data = self.data_fetcher.fetch_batch(requests, use_parallel=True)

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
                        logger.warning(f"銘柄分析エラー {self.stock_helper.format_stock_display(symbol)}: {e}")
                else:
                    logger.warning(f"データ取得失敗: {self.stock_helper.format_stock_display(symbol)}")

            # 3. スコア順ソート
            recommendations.sort(key=lambda x: x.composite_score, reverse=True)

            elapsed_time = time.time() - start_time
            logger.info(f"推奨銘柄分析完了: {len(recommendations)} 銘柄 ({elapsed_time:.2f}秒)")

            return recommendations

        except Exception as e:
            logger.error(f"推奨銘柄分析エラー: {e}")
            return []


    async def _analyze_single_stock(self, symbol: str, data: pd.DataFrame) -> Optional[StockRecommendation]:
        """単一銘柄の分析"""
        try:
            # 1. テクニカル指標分析
            technical_score, technical_reasons = await self._calculate_technical_score(symbol, data)

            # 2. ML予測分析
            ml_score, ml_reasons = await self._calculate_ml_score(symbol, data)

            # Issue #464対応: 3. アンサンブル予測分析
            ensemble_score, ensemble_reasons = await self._calculate_ensemble_score(symbol, data)

            # 4. 総合スコア計算（アンサンブル統合版）
            composite_score = (
                technical_score * self.weights['technical'] +
                ml_score * self.weights['ml'] +
                ensemble_score * self.weights['ensemble']
            )

            # 4. リスク評価
            risk_level = self._assess_risk_level(data)

            # 5. 推奨アクション決定
            action = self._determine_action(composite_score, risk_level)

            # 6. 信頼度計算
            confidence = self._calculate_confidence(technical_score, ml_score, data)

            # 7. 推奨理由統合（アンサンブル統合版）
            all_reasons = technical_reasons + ml_reasons + ensemble_reasons

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
            logger.error(f"単一銘柄分析エラー {self.stock_helper.format_stock_display(symbol)}: {e}")
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
            # Issue #580対応: エラータイプ別の詳細ハンドリング
            error_info = self._analyze_technical_error(e, symbol, data)
            logger.warning(f"テクニカル指標計算エラー {self.stock_helper.format_stock_display(symbol)}: {error_info['message']}")
            logger.debug(f"テクニカル指標エラー詳細 {symbol}: {str(e)}", exc_info=True)
            return error_info['score'], error_info['reasons']

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
            # Issue #580対応: エラータイプ別の詳細ハンドリング
            error_info = self._analyze_ml_error(e, symbol, data)
            logger.warning(f"ML予測計算エラー {self.stock_helper.format_stock_display(symbol)}: {error_info['message']}")
            logger.debug(f"ML予測エラー詳細 {symbol}: {str(e)}", exc_info=True)
            return error_info['score'], error_info['reasons']

    async def _calculate_ensemble_score(self, symbol: str, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """
        Issue #464対応: アンサンブル予測スコア計算

        多様なMLモデルによる統合予測で最高精度を実現
        """
        try:
            if not self.ensemble_system:
                # アンサンブルシステムが利用できない場合はスキップ
                return 0.0, []

            reasons = []

            # データ形式の確認と変換
            if data.empty or len(data) < 30:
                logger.warning(f"アンサンブル予測: データ不足 {symbol}")
                return 0.0, ["データ不足"]

            # 特徴量データ準備
            try:
                # OHLCVデータの準備
                feature_columns = []
                for col_pattern in ['Open', '始値', 'High', '高値', 'Low', '安値', 'Close', '終値', 'Volume', '出来高']:
                    matching_cols = [col for col in data.columns if col_pattern in col]
                    if matching_cols:
                        feature_columns.extend(matching_cols[:1])  # 最初の一致のみ

                if len(feature_columns) < 4:  # 最低限OHLC必要
                    logger.warning(f"アンサンブル予測: 特徴量不足 {symbol}")
                    return 0.0, ["特徴量不足"]

                X = data[feature_columns].fillna(method='ffill').fillna(method='bfill').values
                if len(X) < 30:
                    return 0.0, ["履歴データ不足"]

                X = X[-30:]  # 最新30日分

                # ターゲット変数（価格変化率）
                price_col = next((col for col in data.columns if any(pattern in col for pattern in ['終値', 'Close'])), None)
                if not price_col:
                    return 0.0, ["価格データなし"]

                prices = data[price_col].fillna(method='ffill').values
                if len(prices) < 31:
                    return 0.0, ["価格履歴不足"]

                # 次期価格変化率をターゲットとして設定
                price_changes = np.diff(prices[-31:]) / prices[-31:-1]  # 30個の価格変化率
                y = price_changes

                # アンサンブル学習の実行（初回のみ）
                if not self._ensemble_trained:
                    logger.info(f"アンサンブルモデル学習開始: {symbol}")

                    # データ形状確認
                    if X.shape[0] != len(y):
                        min_len = min(X.shape[0], len(y))
                        X = X[-min_len:]
                        y = y[-min_len:]

                    X_reshaped = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X

                    training_results = self.ensemble_system.fit(
                        X_reshaped, y,
                        feature_names=[f"feature_{i}" for i in range(X_reshaped.shape[1])]
                    )

                    self._ensemble_trained = True
                    logger.info(f"アンサンブル学習完了: {len(training_results)}モデル")
                    reasons.append("アンサンブル学習完了")

                # 予測実行
                X_pred = X[-1:].reshape(1, -1) if len(X.shape) > 1 else X[-1:].reshape(1, -1)
                ensemble_prediction = self.ensemble_system.predict(X_pred)

                # 予測結果をスコアに変換（0-100スケール）
                if hasattr(ensemble_prediction, 'final_predictions') and len(ensemble_prediction.final_predictions) > 0:
                    price_change_pred = ensemble_prediction.final_predictions[0]
                    confidence = ensemble_prediction.ensemble_confidence[0] if hasattr(ensemble_prediction, 'ensemble_confidence') else 0.5

                    # 価格変化率を0-100スケールのスコアに変換
                    # 正の変化率：50+変化率*500（最大100）
                    # 負の変化率：50+変化率*500（最小0）
                    ensemble_score = max(0, min(100, 50 + price_change_pred * 500))

                    # 信頼度による調整
                    ensemble_score = ensemble_score * confidence + 50 * (1 - confidence)

                    # スコア別の理由追加
                    if ensemble_score > 70:
                        reasons.append("アンサンブル予測: 強い上昇期待")
                        if confidence > 0.8:
                            reasons.append("アンサンブル予測: 高信頼度")
                    elif ensemble_score > 60:
                        reasons.append("アンサンブル予測: 上昇期待")
                    elif ensemble_score < 40:
                        reasons.append("アンサンブル予測: 下降懸念")
                        if confidence > 0.8:
                            reasons.append("アンサンブル予測: 高確度下降")
                    elif ensemble_score < 30:
                        reasons.append("アンサンブル予測: 強い下降懸念")

                    # 使用モデル情報
                    if hasattr(ensemble_prediction, 'individual_predictions'):
                        model_count = len(ensemble_prediction.individual_predictions)
                        reasons.append(f"統合予測({model_count}モデル)")

                    return float(ensemble_score), reasons
                else:
                    return 0.0, ["予測失敗"]

            except Exception as e:
                logger.warning(f"アンサンブル予測処理エラー {symbol}: {e}")
                return 0.0, ["処理エラー"]

        except Exception as e:
            logger.warning(f"アンサンブル予測エラー {self.stock_helper.format_stock_display(symbol)}: {e}")
            return 0.0, ["システムエラー"]

    def _analyze_sma_signal(self, data: pd.DataFrame, sma_result) -> Tuple[float, Optional[str]]:
        """Issue #582対応: SMA信号分析（パラメータ化閾値）"""
        try:
            if hasattr(sma_result, 'values') and 'sma' in sma_result.values:
                sma_values = sma_result.values['sma']
                if len(sma_values) > 1:
                    close_price = data['終値'].iloc[-1] if '終値' in data.columns else data['Close'].iloc[-1]
                    current_sma = sma_values[-1]

                    # Issue #582対応: 設定可能な閾値を使用
                    breakout_threshold = self.signal_thresholds['sma_breakout']

                    if close_price > current_sma * (1 + breakout_threshold):  # 設定可能%以上上抜け
                        return 15.0, "SMA上抜けシグナル"
                    elif close_price < current_sma * (1 - breakout_threshold):  # 設定可能%以上下抜け
                        return -15.0, "SMA下抜けシグナル"

            return 0.0, None

        except Exception:
            return 0.0, None

    def _analyze_rsi_signal(self, rsi_result) -> Tuple[float, Optional[str]]:
        """Issue #582対応: RSI信号分析（パラメータ化閾値）"""
        try:
            if hasattr(rsi_result, 'values') and 'rsi' in rsi_result.values:
                rsi_values = rsi_result.values['rsi']
                if len(rsi_values) > 0:
                    current_rsi = rsi_values[-1]

                    # Issue #582対応: 設定可能な閾値を使用
                    oversold_threshold = self.signal_thresholds['rsi_oversold']
                    overbought_threshold = self.signal_thresholds['rsi_overbought']

                    if current_rsi < oversold_threshold:  # 設定可能な売られすぎ閾値
                        return 10.0, "RSI売られすぎ（反発期待）"
                    elif current_rsi > overbought_threshold:  # 設定可能な買われすぎ閾値
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
        """Issue #585対応: 推奨アクション決定（パラメータ化）"""
        # リスク調整（設定可能）
        adjusted_score = score - self.risk_penalties.get(risk_level, 5)

        # 閾値ベースのアクション決定（設定可能）
        if adjusted_score >= self.action_thresholds['strong_buy']:
            return RecommendationAction.STRONG_BUY
        elif adjusted_score >= self.action_thresholds['buy']:
            return RecommendationAction.BUY
        elif adjusted_score >= self.action_thresholds['hold_upper']:
            return RecommendationAction.HOLD
        elif adjusted_score >= self.action_thresholds['sell']:
            return RecommendationAction.SELL
        else:
            return RecommendationAction.STRONG_SELL

    def _calculate_confidence(self, technical_score: float, ml_score: float, data: pd.DataFrame) -> float:
        """Issue #586対応: 高度信頼度計算（ML信頼度・市場ボラティリティ考慮）"""
        try:
            # 1. スコア一致度（基本信頼度）
            score_consistency = 100 - abs(technical_score - ml_score)

            # 2. データ量・品質
            data_adequacy = min(100, len(data) / 60 * 100)  # 60日分を100%とする

            # 3. Issue #586対応: 市場ボラティリティ考慮
            close_col = '終値' if '終値' in data.columns else 'Close'
            if close_col in data.columns:
                # 価格変動率の標準偏差（ボラティリティ）
                price_volatility = data[close_col].pct_change().std() * 100
                # ボラティリティが低いほど信頼度が高い（逆相関）
                volatility_factor = max(20, 100 - (price_volatility * 10))
            else:
                volatility_factor = 60.0

            # 4. Issue #586対応: ML予測信頼度（簡易実装）
            # MLスコアの極端さ（0または100に近い）ほど信頼度が高い
            ml_confidence = min(abs(ml_score - 50) * 2, 100)

            # 5. Issue #586対応: 重み付き総合信頼度計算
            # 既存の重みに加えて新しい要素を統合
            base_confidence = (
                score_consistency * self.confidence_weights['score_consistency'] +
                data_adequacy * self.confidence_weights['data_adequacy']
            )

            # 高度要素の統合（20%の重み）
            advanced_factors = (volatility_factor * 0.4 + ml_confidence * 0.6)

            # 最終信頼度計算
            final_confidence = base_confidence * 0.8 + advanced_factors * 0.2

            return max(0, min(100, final_confidence))

        except Exception as e:
            logger.debug(f"信頼度計算エラー: {e}")
            return 50.0

    def _calculate_price_targets(self, current_price: float, score: float, risk_level: str) -> Tuple[float, float]:
        """Issue #587対応: 価格目標・ストップロス計算（パラメータ化）"""
        try:
            # リスクレベル別パラメータ取得
            params = self.price_target_params.get(risk_level, self.price_target_params["中"])

            # スコア調整による目標価格計算
            score_multiplier = score / 100
            target_rate = params["target_factor"] * score_multiplier
            stop_rate = params["stop_loss_factor"]

            # 価格目標・ストップロス計算
            price_target = current_price * (1 + target_rate)
            stop_loss = current_price * (1 - stop_rate)

            return round(price_target, 0), round(stop_loss, 0)

        except Exception:
            return None, None

    def _get_all_symbols(self) -> List[str]:
        """
        全銘柄リスト取得

        Issue #487対応: スマート銘柄自動選択の統合
        """
        try:
            # StockNameHelperのインスタンスから全銘柄情報を取得
            all_stock_info = self.stock_helper.get_all_symbols()
            symbols = list(all_stock_info.keys())

            logger.info(f"StockNameHelperから {len(symbols)} 銘柄を取得")
            if symbols:
                return symbols

        except Exception as e:
            logger.warning(f"StockNameHelperからの銘柄情報取得エラー: {e}")

        # フォールバック: デフォルト銘柄
        default_symbols = ["7203", "8306", "9984", "6758", "4689"]
        logger.info(f"デフォルト銘柄を使用: {len(default_symbols)} 銘柄")
        return default_symbols

    async def _get_smart_selected_symbols(self, target_count: int = 10) -> List[str]:
        """
        Issue #487対応: スマート銘柄自動選択

        市場流動性・出来高・ボラティリティに基づく最適銘柄選択

        Args:
            target_count: 目標銘柄数

        Returns:
            自動選択された最適銘柄リスト
        """
        try:
            logger.info("🤖 スマート銘柄自動選択を開始")
            smart_symbols = await get_smart_selected_symbols(target_count)

            if smart_symbols:
                logger.info(f"✅ スマート選択完了: {len(smart_symbols)}銘柄")
                return smart_symbols
            else:
                logger.warning("スマート選択が失敗、フォールバックシンボルを使用")

        except Exception as e:
            logger.error(f"スマート銘柄選択エラー: {e}")
            logger.info("フォールバックシンボルを使用")

        # フォールバック
        return self._get_all_symbols()[:target_count]

    def get_top_recommendations(self, recommendations: List[StockRecommendation], limit: int = 10) -> List[StockRecommendation]:
        """TOP推奨銘柄取得"""
        # 買い推奨のみフィルタ
        buy_recommendations = [
            r for r in recommendations
            if r.action in [RecommendationAction.STRONG_BUY, RecommendationAction.BUY]
        ]

        return buy_recommendations[:limit]

    def _analyze_technical_error(self, error: Exception, symbol: str, data: pd.DataFrame) -> Dict[str, any]:
        """
        テクニカル指標エラー分析 - Issue #580対応

        Args:
            error: 発生した例外
            symbol: 銘柄コード
            data: 価格データ

        Returns:
            エラー情報辞書（score, reasons, message）
        """
        error_type = type(error).__name__
        data_size = len(data) if data is not None else 0

        # エラータイプ別の分析
        if isinstance(error, KeyError):
            missing_column = str(error).replace("'", "")
            return {
                'score': 30.0,  # データ不整合時は低めのスコア
                'reasons': [f"データ構造エラー: {missing_column}列不足"],
                'message': f"データ列不足エラー ({missing_column})"
            }

        elif isinstance(error, ValueError):
            if "empty" in str(error).lower() or data_size < 10:
                return {
                    'score': 20.0,  # データ不足時は非常に低いスコア
                    'reasons': [f"データ不足: {data_size}日間のみ"],
                    'message': f"データ不足エラー (データ数: {data_size})"
                }
            else:
                return {
                    'score': 40.0,  # 計算エラー時は中低スコア
                    'reasons': ["計算値エラー: 数値変換失敗"],
                    'message': "数値計算エラー"
                }

        elif isinstance(error, IndexError):
            return {
                'score': 25.0,  # インデックスエラーは深刻
                'reasons': [f"データ配列エラー: インデックス範囲外"],
                'message': "データ配列アクセスエラー"
            }

        elif isinstance(error, AttributeError):
            missing_attr = str(error).split("'")[-2] if "'" in str(error) else "不明"
            return {
                'score': 35.0,  # 属性エラー
                'reasons': [f"システムエラー: {missing_attr}メソッド未実装"],
                'message': f"システム構造エラー ({missing_attr})"
            }

        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            return {
                'score': 45.0,  # ライブラリ不足は中立的スコア
                'reasons': ["システム制限: 必要ライブラリ未インストール"],
                'message': "ライブラリ依存エラー"
            }

        elif isinstance(error, TimeoutError):
            return {
                'score': 50.0,  # タイムアウトは中立
                'reasons': ["処理タイムアウト: 計算時間超過"],
                'message': "処理時間超過エラー"
            }

        else:
            # 未知のエラー
            return {
                'score': 50.0,  # デフォルトスコア
                'reasons': [f"テクニカル分析エラー: {error_type}"],
                'message': f"予期しないエラー ({error_type})"
            }

    def _analyze_ml_error(self, error: Exception, symbol: str, data: pd.DataFrame) -> Dict[str, any]:
        """
        ML予測エラー分析 - Issue #580対応

        Args:
            error: 発生した例外
            symbol: 銘柄コード
            data: 価格データ

        Returns:
            エラー情報辞書（score, reasons, message）
        """
        error_type = type(error).__name__
        data_size = len(data) if data is not None else 0

        # エラータイプ別の分析
        if isinstance(error, KeyError):
            missing_column = str(error).replace("'", "")
            return {
                'score': 35.0,  # MLではデータ不整合の影響は若干軽微
                'reasons': [f"ML入力エラー: {missing_column}特徴量不足"],
                'message': f"ML特徴量エラー ({missing_column})"
            }

        elif isinstance(error, ValueError):
            if "shape" in str(error).lower():
                return {
                    'score': 30.0,  # 形状エラーは深刻
                    'reasons': [f"MLモデルエラー: データ形状不一致"],
                    'message': "MLデータ形状エラー"
                }
            elif "empty" in str(error).lower() or data_size < 20:
                return {
                    'score': 25.0,  # MLには最低20日は必要
                    'reasons': [f"MLデータ不足: {data_size}日間では予測困難"],
                    'message': f"ML学習データ不足 (データ数: {data_size})"
                }
            else:
                return {
                    'score': 40.0,  # 計算エラー時は中低スコア
                    'reasons': ["ML計算エラー: 予測値生成失敗"],
                    'message': "ML予測計算エラー"
                }

        elif isinstance(error, RuntimeError):
            if "cuda" in str(error).lower() or "gpu" in str(error).lower():
                return {
                    'score': 55.0,  # GPU不足は機能制限のみ
                    'reasons': ["ML制限: GPU未使用でCPU予測実行"],
                    'message': "GPU利用不可（CPU予測に切替）"
                }
            else:
                return {
                    'score': 35.0,  # ランタイムエラーは深刻
                    'reasons': ["MLランタイムエラー: 予測処理失敗"],
                    'message': "ML予測処理エラー"
                }

        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            return {
                'score': 50.0,  # MLライブラリ不足時はニュートラル
                'reasons': ["ML制限: 高度予測ライブラリ未インストール"],
                'message': "ML依存ライブラリエラー"
            }

        elif isinstance(error, MemoryError):
            return {
                'score': 45.0,  # メモリ不足
                'reasons': ["MLメモリ不足: 簡易予測モードに切替"],
                'message': "MLメモリ不足エラー"
            }

        elif isinstance(error, TimeoutError):
            return {
                'score': 50.0,  # MLタイムアウトは中立
                'reasons': ["ML処理タイムアウト: 予測時間超過"],
                'message': "ML予測時間超過エラー"
            }

        elif isinstance(error, AttributeError):
            missing_attr = str(error).split("'")[-2] if "'" in str(error) else "不明"
            return {
                'score': 40.0,  # ML属性エラー
                'reasons': [f"MLシステムエラー: {missing_attr}機能未実装"],
                'message': f"ML機能エラー ({missing_attr})"
            }

        else:
            # 未知のエラー
            return {
                'score': 50.0,  # デフォルトスコア
                'reasons': [f"ML予測エラー: {error_type}"],
                'message': f"ML予期しないエラー ({error_type})"
            }

    def close(self):
        """リソース解放"""
        if hasattr(self.data_fetcher, 'close'):
            self.data_fetcher.close()
        if hasattr(self.technical_manager, 'close'):
            self.technical_manager.close()
        if hasattr(self.ml_engine, 'close'):
            self.ml_engine.close()
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


async def get_smart_daily_recommendations(limit: int = 10) -> List[StockRecommendation]:
    """
    Issue #487対応: スマート銘柄選択による日次推奨取得

    市場流動性・出来高・ボラティリティに基づく最適銘柄から推奨を生成

    Args:
        limit: 推奨銘柄数上限

    Returns:
        スマート選択されたベスト推奨銘柄リスト
    """
    engine = RecommendationEngine()
    try:
        logger.info("🚀 スマート銘柄選択による推奨分析を開始")

        # Step 1: スマート銘柄自動選択
        smart_symbols = await engine._get_smart_selected_symbols(target_count=limit * 2)

        # Step 2: 選択された銘柄の詳細分析
        smart_recommendations = await engine.analyze_all_stocks(smart_symbols)

        # Step 3: 最終推奨選定
        final_recommendations = engine.get_top_recommendations(smart_recommendations, limit)

        logger.info(f"✅ スマート推奨完了: {len(final_recommendations)}銘柄")
        return final_recommendations

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