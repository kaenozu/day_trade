#!/usr/bin/env python3
"""
Day Trade Personal - 1日単位デイトレード推奨エンジン（改善版）

デイトレードに特化した1日単位の売買タイミング推奨システム

Issue #849対応:
- print()文のlogging置換
- モックデータフォールバックの改善
- ハードコード銘柄マッピングの外部化
- デモコードの分離
- Windows環境対策統合
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime, time, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from src.day_trade.data.fetchers.yfinance_fetcher import YFinanceFetcher
# ML モデル関連インポート（オプション）
try:
    from integrated_model_loader import IntegratedModelLoader
    INTEGRATED_MODEL_AVAILABLE = True
except ImportError:
    INTEGRATED_MODEL_AVAILABLE = False

try:
    from model_performance_monitor import ModelPerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False

try:
    from ml_prediction_models import PredictionTask
    PREDICTION_TASK_AVAILABLE = True
except ImportError:
    PREDICTION_TASK_AVAILABLE = False

# Windows環境対策統合
try:
    from src.day_trade.utils.encoding_utils import setup_windows_encoding
    setup_windows_encoding()
except ImportError:
    pass

# ロギング設定
logger = logging.getLogger(__name__)

# 実データプロバイダー統合
try:
    from real_data_provider import RealDataProvider, RealDataAnalysisEngine
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

# 市場時間管理システム統合
try:
    from market_time_manager import MarketTimeManager, MarketSession
    MARKET_TIME_AVAILABLE = True
except ImportError:
    MARKET_TIME_AVAILABLE = False

class DayTradingSignal(Enum):
    """デイトレードシグナル"""
    STRONG_BUY = "強い買い"      # 即座に買い
    BUY = "買い"               # 押し目で買い
    HOLD = "ホールド"          # 既存ポジション維持
    SELL = "売り"              # 利確・損切り売り
    STRONG_SELL = "強い売り"    # 即座に売り
    WAIT = "待機"              # エントリーチャンス待ち

class TradingSession(Enum):
    """取引時間帯"""
    PRE_MARKET = "寄り前"      # 9:00前
    MORNING_SESSION = "前場"    # 9:00-11:30
    LUNCH_BREAK = "昼休み"     # 11:30-12:30
    AFTERNOON_SESSION = "後場"  # 12:30-15:00
    AFTER_MARKET = "大引け後"   # 15:00後

@dataclass
class DayTradingRecommendation:
    """デイトレード推奨"""
    symbol: str
    name: str
    signal: DayTradingSignal
    entry_timing: str          # エントリータイミング
    target_profit: float       # 目標利益率(%)
    stop_loss: float          # 損切りライン(%)
    holding_time: str         # 推奨保有時間
    confidence: float         # 信頼度
    risk_level: str          # リスクレベル
    volume_trend: str        # 出来高動向
    price_momentum: str      # 価格モメンタム
    intraday_volatility: float  # 日中ボラティリティ
    market_timing_score: float  # 市場タイミングスコア

class PersonalDayTradingEngine:
    """個人向けデイトレードエンジン"""

    def __init__(self, config_path: Optional[str] = None):
        # 設定読み込み（オプション）
        self.config_path = Path(config_path) if config_path else Path("config/day_trading_config.json")
        self.config = self._load_configuration() if config_path else {}

        # 動的銘柄取得システム
        self.daytrading_symbols = self._load_dynamic_symbols()

        # 市場時間管理システム初期化
        if MARKET_TIME_AVAILABLE:
            self.market_manager = MarketTimeManager()
            self.time_mode = "ACCURATE"
        else:
            self.market_manager = None
            self.time_mode = "SIMPLE"

        self.current_session = self._get_current_trading_session()

        # 実データプロバイダー初期化
        if REAL_DATA_AVAILABLE:
            self.real_data_engine = RealDataAnalysisEngine()
            self.data_mode = "REAL"
        else:
            self.real_data_engine = None
            self.data_mode = "DEMO"

        # データフェッチャー初期化
        try:
            from src.day_trade.data.stock_fetcher import StockFetcher
            self.data_fetcher = StockFetcher()
        except ImportError:
            # フォールバック用の簡易データフェッチャー
            self.data_fetcher = YFinanceFetcher()

        # MLモデル初期化
        try:
            from src.day_trade.ml.dynamic_weighting_system import DynamicWeightingSystem
            model_names = ['linear_regression', 'random_forest', 'gradient_boosting']
            self.ml_model = DynamicWeightingSystem(model_names=model_names)
        except ImportError:
            self.ml_model = None

        # MLモデルの初期化とロード（オプション）
        if INTEGRATED_MODEL_AVAILABLE:
            try:
                self.model_loader = IntegratedModelLoader()
                if PERFORMANCE_MONITOR_AVAILABLE and hasattr(self.model_loader, 'upgrade_db_path'):
                    self.model_performance_monitor = ModelPerformanceMonitor(
                        upgrade_db_path=self.model_loader.upgrade_db_path,
                        advanced_ml_db_path=getattr(self.model_loader, 'advanced_ml_predictions_db_path', None)
                    )
                else:
                    self.model_performance_monitor = None
            except Exception as e:
                logger.warning(f"MLモデル初期化に失敗: {e}")
                self.model_loader = None
                self.model_performance_monitor = None
        else:
            self.model_loader = None
            self.model_performance_monitor = None

        # データフェッチャー初期化（YFinanceFetcher使用）
        if not hasattr(self, 'data_fetcher') or self.data_fetcher is None:
            try:
                self.data_fetcher = YFinanceFetcher()
            except Exception as e:
                logger.error(f"データフェッチャー初期化失敗: {e}")
                self.data_fetcher = None

        # ログレベル設定
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        logger.setLevel(log_level)

        logger.info(f"デイトレードエンジン初期化完了 - データモード: {self.data_mode}, 時間モード: {self.time_mode}")

    def _load_dynamic_symbols(self) -> dict:
        """
        改善版動的銘柄取得（Issue #849対応）

        1. 設定ファイルからのフォールバック銘柄読み込み
        2. 動的銘柄選択システム
        3. 銘柄名辞書統合
        4. 段階的フォールバック機能
        """
        try:
            # 1. 設定ファイルからフォールバック銘柄を読み込み
            fallback_symbols = self._load_fallback_symbols_from_config()

            # 2. 動的銘柄選択を試行
            dynamic_symbols = self._try_dynamic_symbol_selection()

            # 3. 銘柄名解決を実行
            final_symbols = self._resolve_symbol_names(dynamic_symbols, fallback_symbols)

            logger.info(f"銘柄取得完了: {len(final_symbols)}銘柄（動的: {len(dynamic_symbols)}, フォールバック: {len(fallback_symbols)}）")
            return final_symbols

        except Exception as e:
            logger.error(f"銘柄取得プロセス失敗: {e}")
            # 最終フォールバック: 設定ファイルの銘柄のみ使用
            return self._load_fallback_symbols_from_config()

    def _load_fallback_symbols_from_config(self) -> dict:
        """設定ファイルからフォールバック銘柄を読み込み"""
        try:
            symbol_mapping = self.config.get("symbol_mapping", {})
            fallback_symbols = symbol_mapping.get("fallback_symbols", {})
            custom_symbols = symbol_mapping.get("custom_symbols", {})

            # フォールバックとカスタム銘柄を統合
            combined_symbols = {}
            combined_symbols.update(fallback_symbols)
            if isinstance(custom_symbols, dict):
                combined_symbols.update(custom_symbols)

            logger.info(f"設定ファイルから{len(combined_symbols)}銘柄を読み込み")
            return combined_symbols

        except Exception as e:
            logger.warning(f"設定ファイル銘柄読み込み失敗: {e}")
            # 最小限のデフォルト銘柄
            return {
                "7203": "トヨタ自動車",
                "9984": "ソフトバンクグループ",
                "8306": "三菱UFJフィナンシャル・グループ"
            }

    def _try_dynamic_symbol_selection(self) -> list:
        """動的銘柄選択を試行"""
        try:
            # 設定でdynamic selectionが有効な場合のみ実行
            symbol_selection_config = self.config.get("symbol_selection", {})
            if not symbol_selection_config.get("enable_dynamic_selection", True):
                logger.info("動的銘柄選択は設定で無効化されています")
                return []

            from src.day_trade.data.symbol_selector import DynamicSymbolSelector
            selector = DynamicSymbolSelector()

            # 設定に基づく銘柄数制限
            max_symbols = symbol_selection_config.get("max_symbols", 20)
            symbols = selector.get_liquid_symbols(limit=max_symbols)

            logger.info(f"動的銘柄選択成功: {len(symbols)}銘柄")
            return symbols

        except Exception as e:
            logger.warning(f"動的銘柄選択失敗: {e}")
            return []

    def _resolve_symbol_names(self, dynamic_symbols: list, fallback_symbols: dict) -> dict:
        """銘柄名解決（動的+フォールバック統合）"""
        symbol_dict = {}

        # 1. 動的取得銘柄の名前解決
        for symbol in dynamic_symbols:
            name = self._resolve_single_symbol_name(symbol)
            if name:
                symbol_dict[symbol] = name

        # 2. フォールバック銘柄を追加（重複は動的銘柄を優先）
        for symbol, name in fallback_symbols.items():
            if symbol not in symbol_dict:
                symbol_dict[symbol] = name

        return symbol_dict

    def _resolve_single_symbol_name(self, symbol: str) -> str:
        """単一銘柄の名前解決"""
        # 1. 銘柄名辞書から取得（最優先）
        try:
            from src.day_trade.data.symbol_names import get_symbol_name
            name = get_symbol_name(symbol)
            if name:
                logger.debug(f"Symbol name from dict: {symbol} -> {name}")
                return name
        except Exception as e:
            logger.debug(f"Symbol name dict lookup failed for {symbol}: {e}")

        # 2. 設定ファイルからの取得
        fallback_symbols = self.config.get("symbol_mapping", {}).get("fallback_symbols", {})
        if symbol in fallback_symbols:
            name = fallback_symbols[symbol]
            logger.debug(f"Symbol name from config: {symbol} -> {name}")
            return name

        # 3. フォールバック: 銘柄コードベースの名前
        fallback_name = f"銘柄{symbol}"
        logger.debug(f"Symbol name fallback: {symbol} -> {fallback_name}")
        return fallback_name

    def _get_current_trading_session(self) -> TradingSession:
        """現在の取引時間帯を取得（正確な市場時間管理）"""
        # 正確な市場時間管理システム使用
        if self.market_manager:
            from datetime import datetime
            market_session = self.market_manager.get_current_session(datetime.now())

            # MarketSessionをTradingSessionに変換
            session_map = {
                MarketSession.PRE_MARKET: TradingSession.PRE_MARKET,
                MarketSession.MORNING_SESSION: TradingSession.MORNING_SESSION,
                MarketSession.LUNCH_BREAK: TradingSession.LUNCH_BREAK,
                MarketSession.AFTERNOON_SESSION: TradingSession.AFTERNOON_SESSION,
                MarketSession.AFTER_MARKET: TradingSession.AFTER_MARKET,
                MarketSession.MARKET_CLOSED: TradingSession.AFTER_MARKET  # 休場日は大引け後として扱う
            }

            return session_map.get(market_session, TradingSession.AFTER_MARKET)

        # フォールバック: 従来の簡易判定
        now = datetime.now()
        current_time = now.time()

        if current_time < time(9, 0):
            return TradingSession.PRE_MARKET
        elif time(9, 0) <= current_time < time(11, 30):
            return TradingSession.MORNING_SESSION
        elif time(11, 30) <= current_time < time(12, 30):
            return TradingSession.LUNCH_BREAK
        elif time(12, 30) <= current_time < time(15, 0):
            return TradingSession.AFTERNOON_SESSION
        else:
            return TradingSession.AFTER_MARKET

    async def get_today_daytrading_recommendations(self, limit: int = 20) -> List[DayTradingRecommendation]:
        """今日のデイトレード推奨取得（実データ対応）"""
        if self.model_performance_monitor:
            await self.model_performance_monitor.check_and_trigger_retraining()

        current_session = self._get_current_trading_session()

        # 大引け後は翌日前場予想モードに切り替え
        if current_session == TradingSession.AFTER_MARKET:
            return await self.get_tomorrow_premarket_forecast(limit)

        # 実データエンジン使用可能な場合
        if self.real_data_engine:
            real_recommendations = await self.real_data_engine.analyze_daytrading_opportunities(limit)
            return [self._convert_to_daytrading_recommendation(rec, current_session) for rec in real_recommendations]

        # フォールバック: ダミーデータ使用
        recommendations = []
        symbols = list(self.daytrading_symbols.keys())[:limit]

        for symbol in symbols:
            rec = await self._analyze_daytrading_opportunity(symbol, current_session)
            recommendations.append(rec)

        # 市場タイミングスコア順でソート
        recommendations.sort(key=lambda x: x.market_timing_score, reverse=True)

        return recommendations

    def _convert_to_daytrading_recommendation(self, real_rec: Dict[str, any], session: TradingSession) -> DayTradingRecommendation:
        """実データ推奨をDayTradingRecommendationに変換"""

        # シグナル変換
        signal_map = {
            "強い買い": DayTradingSignal.STRONG_BUY,
            "●買い●": DayTradingSignal.BUY,
            "△やや買い△": DayTradingSignal.BUY,
            "…待機…": DayTradingSignal.WAIT,
            "▽売り▽": DayTradingSignal.SELL,
            "▼強い売り▼": DayTradingSignal.STRONG_SELL,
        }

        signal = signal_map.get(real_rec.get("signal", "…待機…"), DayTradingSignal.WAIT)

        # エントリータイミング生成
        entry_timing = self._generate_entry_timing(real_rec, session)

        # 保有時間推奨
        holding_time = self._generate_holding_time(real_rec, session)

        return DayTradingRecommendation(
            symbol=real_rec["symbol"],
            name=real_rec["name"],
            signal=signal,
            entry_timing=entry_timing,
            target_profit=real_rec.get("target_profit", 2.0),
            stop_loss=real_rec.get("stop_loss", 1.5),
            holding_time=holding_time,
            confidence=real_rec.get("confidence", 80),
            risk_level=self._calculate_risk_level(real_rec),
            volume_trend="実データベース",
            price_momentum=self._calculate_momentum(real_rec),
            intraday_volatility=real_rec.get("volatility", 3.0),
            market_timing_score=real_rec.get("trading_score", 70)
        )

    def _generate_entry_timing(self, real_rec: Dict[str, any], session: TradingSession) -> str:
        """エントリータイミング生成"""
        signal = real_rec.get("signal", "")
        change_pct = real_rec.get("change_percent", 0)

        if "強い買い" in signal:
            return "即座に成り行きで積極エントリー（実データ推奨）"
        elif "買い" in signal:
            if change_pct > 0:
                return "押し目5-10分待ってからエントリー（実ギャップ）"
            else:
                return "現在値近辺でエントリー（実下値）"
        elif "待機" in signal:
            return "待機期間中エントリーチャンス探し"
        else:
            return "利確・損切りタイミング判定（実データ）"

    def _generate_holding_time(self, real_rec: Dict[str, any], session: TradingSession) -> str:
        """保有時間推奨"""
        volatility = real_rec.get("volatility", 3.0)

        if volatility > 5.0:
            return "短時間1〜2時間程度"
        elif volatility > 3.0:
            return "30分〜1時間での決済"
        else:
            return "当日中決済完了推奨"

    def _calculate_risk_level(self, real_rec: Dict[str, any]) -> str:
        """リスクレベル計算"""
        volatility = real_rec.get("volatility", 3.0)
        trading_score = real_rec.get("trading_score", 70)

        if volatility > 6.0:
            return "[高リスク]"
        elif volatility > 3.0 and trading_score > 80:
            return "[中リスク]"
        else:
            return "[低リスク]"

    def _calculate_momentum(self, real_rec: Dict[str, any]) -> str:
        """価格モメンタム計算"""
        change_pct = real_rec.get("change_pct", 0)
        rsi = real_rec.get("rsi", 50)

        if change_pct > 2.0 and rsi < 70:
            return "強い上昇モメンタム"
        elif change_pct > 0 and rsi < 60:
            return "上昇基調"
        elif change_pct < -2.0 and rsi > 30:
            return "下落からの反転期待"
        else:
            return "モメンタム中立"

    async def get_tomorrow_premarket_forecast(self, limit: int = 20) -> List[DayTradingRecommendation]:
        """翌日前場予想取得（大引け後専用）"""
        recommendations = []
        symbols = list(self.daytrading_symbols.keys())[:limit]

        # 翌日の日付を生成
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y%m%d')

        for symbol in symbols:
            rec = await self._analyze_tomorrow_premarket_opportunity(symbol, tomorrow_str)
            recommendations.append(rec)

        # 市場タイミングスコア順でソート
        recommendations.sort(key=lambda x: x.market_timing_score, reverse=True)
        return recommendations

    async def _analyze_tomorrow_premarket_opportunity(self, symbol: str, tomorrow_str: str) -> DayTradingRecommendation:
        """翌日前場機会分析"""
        symbol_name = self.daytrading_symbols[symbol]

        # AIモデルによる予測に置き換え
        # 1. 予測に必要な市場データを取得
        market_data = self._fetch_mock_market_data(symbol, tomorrow_str)

        # 2. 取得したデータから特徴量を生成
        features = self._prepare_features_for_prediction(market_data)

        # 3. AIモデルで予測
        # predictions配列の各要素が、オーバーナイトギャップ、プレマーケットモメンタムなどの予測値に対応すると仮定
        # この部分のインデックスと意味は、実際のMLモデルの実装に依存
        # 例: predictions[0] = overnight_gap, predictions[1] = premarket_momentum など
        prediction_results, system_used = await self.model_loader.predict(symbol, features)

        # predictionsとconfidenceの形状を確認し、適切に割り当てる
        # ここでは単一の予測値を想定
        predicted_values = prediction_results.predictions.flatten()
        confidence_values = prediction_results.confidence.flatten() if prediction_results.confidence is not None else np.array([0.0])

        overnight_gap = predicted_values[0] if len(predicted_values) > 0 else 0.0
        premarket_momentum = predicted_values[1] if len(predicted_values) > 1 else 0.0
        volume_expectation = predicted_values[2] if len(predicted_values) > 2 else 0.0
        volatility_forecast = predicted_values[3] if len(predicted_values) > 3 else 0.0

        # 信頼度を考慮した上で、ランダム性を残すか、完全にAI予測に置き換えるかを検討する
        # ここではAI予測を優先
        confidence = 75.0 # AIモデルの信頼度をここに反映させる

        # 翌日前場シグナル判定
        signal = self._determine_tomorrow_premarket_signal(
            overnight_gap, premarket_momentum, volume_expectation, volatility_forecast
        )

        # 寄り付き戦略
        entry_timing = self._get_tomorrow_entry_strategy(signal, overnight_gap)

        # 翌日用利確・損切り設定
        target_profit, stop_loss = self._calculate_premarket_profit_stop_levels(
            signal, volatility_forecast
        )

        # 前場推奨保有時間
        holding_time = self._get_premarket_holding_time(signal)

        # 翌日予想信頼度 (AIモデルの信頼度を使用)
        # prediction_results.confidenceが存在し、かつ値が1つ以上ある場合、その平均値を使用
        # 存在しない場合、または値がない場合は、以前の計算ロジックをフォールバックとして使用するか、固定値を割り当てる
        if confidence_values.size > 0:
            confidence = np.mean(confidence_values) * 100 # %表示にするため
        else:
            # フォールバックとして以前のロジックを使用
            confidence = self._calculate_premarket_confidence(
                signal, volatility_forecast, volume_expectation
            )

        # リスク評価（オーバーナイト考慮）
        risk_level = self._assess_premarket_risk(volatility_forecast, overnight_gap)

        # 市場動向説明
        volume_trend = self._describe_volume_trend(volume_expectation)
        momentum_desc = self._describe_overnight_momentum(overnight_gap, premarket_momentum)

        # 翌日タイミングスコア
        market_timing_score = self._calculate_premarket_timing_score(
            signal, volatility_forecast, volume_expectation
        )

        return DayTradingRecommendation(
            symbol=symbol,
            name=symbol_name,
            signal=signal,
            entry_timing=entry_timing,
            target_profit=target_profit,
            stop_loss=stop_loss,
            holding_time=holding_time,
            confidence=confidence,
            risk_level=risk_level,
            volume_trend=volume_trend,
            price_momentum=momentum_desc,
            intraday_volatility=volatility_forecast,
            market_timing_score=market_timing_score
        )

    def _determine_tomorrow_premarket_signal(self, overnight_gap: float, momentum: float,
                                           volume: float, volatility: float) -> DayTradingSignal:
        """翌日前場シグナル判定"""
        # ポジティブギャップ + 強いモメンタム = 強い買い
        if overnight_gap >= 1.0 and momentum >= 1.0 and volume >= 1.5:
            return DayTradingSignal.STRONG_BUY

        # ネガティブギャップ + 弱いモメンタム = 強い売り
        elif overnight_gap <= -1.0 and momentum <= -1.0 and volume >= 1.5:
            return DayTradingSignal.STRONG_SELL

        # 中程度のポジティブ要因
        elif overnight_gap >= 0.5 or momentum >= 0.5:
            return DayTradingSignal.BUY

        # 中程度のネガティブ要因
        elif overnight_gap <= -0.5 or momentum <= -0.5:
            return DayTradingSignal.SELL

        # 方向感不明
        elif abs(overnight_gap) < 0.3 and abs(momentum) < 0.3:
            return DayTradingSignal.WAIT

        else:
            return DayTradingSignal.HOLD

    def _get_tomorrow_entry_strategy(self, signal: DayTradingSignal, overnight_gap: float) -> str:
        """翌日エントリー戦略"""
        gap_direction = "上ギャップ" if overnight_gap > 0 else "下ギャップ" if overnight_gap < -0.5 else "フラット"

        strategy_map = {
            DayTradingSignal.STRONG_BUY: f"寄り成行で積極エントリー（{gap_direction}想定）",
            DayTradingSignal.BUY: f"寄り後5-10分様子見してエントリー（{gap_direction}）",
            DayTradingSignal.STRONG_SELL: f"寄り成行で売りエントリー（{gap_direction}想定）",
            DayTradingSignal.SELL: f"戻り売りタイミング狙い（{gap_direction}）",
            DayTradingSignal.HOLD: f"寄り後の流れ確認してから判断（{gap_direction}）",
            DayTradingSignal.WAIT: f"前場中盤まで様子見推奨（{gap_direction}）"
        }

        return strategy_map.get(signal, "寄り付き後の値動き確認")

    def _calculate_premarket_profit_stop_levels(self, signal: DayTradingSignal,
                                              volatility: float) -> tuple[float, float]:
        """前場用利確・損切りライン"""
        # ボラティリティベース + オーバーナイトリスク考慮
        base_profit = min(volatility * 0.7, 3.5)  # 少し大きめの利確目標
        base_stop = min(volatility * 0.5, 2.5)    # オーバーナイトリスク考慮

        signal_adjustments = {
            DayTradingSignal.STRONG_BUY: (base_profit * 1.3, base_stop * 0.9),
            DayTradingSignal.BUY: (base_profit * 1.1, base_stop),
            DayTradingSignal.STRONG_SELL: (base_profit * 1.3, base_stop * 0.9),
            DayTradingSignal.SELL: (base_profit * 1.1, base_stop),
            DayTradingSignal.HOLD: (base_profit * 0.8, base_stop * 1.1),
            DayTradingSignal.WAIT: (base_profit * 0.6, base_stop * 0.8)
        }

        profit, stop = signal_adjustments.get(signal, (base_profit, base_stop))
        return round(profit, 1), round(stop, 1)

    def _get_premarket_holding_time(self, signal: DayTradingSignal) -> str:
        """前場推奨保有時間"""
        time_map = {
            DayTradingSignal.STRONG_BUY: "寄り〜前場終了まで（最大2時間30分）",
            DayTradingSignal.BUY: "寄り後1〜2時間程度",
            DayTradingSignal.STRONG_SELL: "即座決済〜30分以内",
            DayTradingSignal.SELL: "30分〜1時間での決済",
            DayTradingSignal.HOLD: "前場中の流れ次第",
            DayTradingSignal.WAIT: "エントリー機会待ち"
        }
        return time_map.get(signal, "前場中の判断")

    def _calculate_premarket_confidence(self, signal: DayTradingSignal,
                                      volatility: float, volume: float) -> float:
        """前場予想信頼度"""
        base_confidence = 65.0  # 予想なので基本信頼度を下げる

        signal_boost = {
            DayTradingSignal.STRONG_BUY: 20,
            DayTradingSignal.BUY: 15,
            DayTradingSignal.STRONG_SELL: 20,
            DayTradingSignal.SELL: 15,
            DayTradingSignal.HOLD: 5,
            DayTradingSignal.WAIT: 0
        }.get(signal, 0)

        volume_boost = min((volume - 1.0) * 8, 12)
        volatility_boost = 8 if 3.0 <= volatility <= 5.0 else 3

        final_confidence = base_confidence + signal_boost + volume_boost + volatility_boost
        return max(50, min(85, final_confidence))  # 予想なので最大85%

    def _assess_premarket_risk(self, volatility: float, overnight_gap: float) -> str:
        """前場リスク評価（オーバーナイト考慮）"""
        if volatility >= 5.0 or abs(overnight_gap) >= 1.5:
            return "高"
        elif volatility >= 3.5 or abs(overnight_gap) >= 0.8:
            return "中"
        else:
            return "低"

    def _describe_overnight_momentum(self, gap: float, momentum: float) -> str:
        """オーバーナイト・前場モメンタム説明"""
        gap_desc = "上ギャップ期待" if gap > 0.5 else "下ギャップ警戒" if gap < -0.5 else "フラット"
        momentum_desc = "強い上昇期待" if momentum > 1.0 else "上昇期待" if momentum > 0.5 else \
                       "強い下落警戒" if momentum < -1.0 else "下落警戒" if momentum < -0.5 else "横ばい"
        return f"{gap_desc}・{momentum_desc}"

    def _calculate_premarket_timing_score(self, signal: DayTradingSignal,
                                        volatility: float, volume: float) -> float:
        """翌日前場タイミングスコア"""
        base_score = {
            DayTradingSignal.STRONG_BUY: 85,
            DayTradingSignal.BUY: 75,
            DayTradingSignal.STRONG_SELL: 80,
            DayTradingSignal.SELL: 70,
            DayTradingSignal.HOLD: 50,
            DayTradingSignal.WAIT: 30
        }.get(signal, 40)

        # ボラティリティ・出来高調整
        vol_adj = 10 if 3.0 <= volatility <= 5.0 else 0
        volume_adj = min((volume - 1.0) * 15, 10)

        final_score = base_score + vol_adj + volume_adj
        return max(0, min(100, final_score))

    async def _analyze_daytrading_opportunity(self, symbol: str, session: TradingSession) -> DayTradingRecommendation:
        """デイトレード機会分析"""
        symbol_name = self.daytrading_symbols[symbol]

        # AIモデルによる予測に置き換え
        # 1. 予測に必要な市場データを取得
        market_data = self._fetch_mock_market_data(symbol, str(datetime.now().date()).replace('-', '')) # YYYYMMDD形式に変換

        # 2. 取得したデータから特徴量を生成
        features = self._prepare_features_for_prediction(market_data)

        # 3. AIモデルで予測
        # predictions配列の各要素が、日中ボラティリティ、出来高比率、価格モメンタムなどの予測値に対応すると仮定
        prediction_results, system_used = await self.model_loader.predict(symbol, features)

        predicted_values = prediction_results.predictions.flatten()
        confidence_values = prediction_results.confidence.flatten() if prediction_results.confidence is not None else np.array([0.0])

        intraday_volatility = predicted_values[0] if len(predicted_values) > 0 else 0.0
        volume_ratio = predicted_values[1] if len(predicted_values) > 1 else 0.0
        price_momentum = predicted_values[2] if len(predicted_values) > 2 else 0.0

        # 時間帯別補正
        session_multiplier = self._get_session_multiplier(session)
        adjusted_volatility = intraday_volatility * session_multiplier

        # シグナル判定
        signal = self._determine_daytrading_signal(
            volatility=adjusted_volatility,
            momentum=price_momentum,
            volume=volume_ratio,
            session=session
        )

        # エントリータイミング
        entry_timing = self._get_entry_timing(signal, session)

        # 利確・損切りライン設定
        target_profit, stop_loss = self._calculate_profit_stop_levels(
            signal, adjusted_volatility
        )

        # 推奨保有時間
        holding_time = self._get_recommended_holding_time(signal, session)

        # 信頼度計算（AIモデルの信頼度を使用）
        if confidence_values.size > 0:
            confidence = np.mean(confidence_values) * 100 # %表示にするため
        else:
            # フォールバックとして以前のロジックを使用
            confidence = self._calculate_daytrading_confidence(
                signal, adjusted_volatility, volume_ratio, session
            )

        # リスクレベル
        risk_level = self._assess_daytrading_risk(adjusted_volatility, signal)

        # 出来高・価格動向の文字化
        volume_trend = self._describe_volume_trend(volume_ratio)
        momentum_desc = self._describe_price_momentum(price_momentum)

        # 市場タイミングスコア（デイトレード適性度）
        market_timing_score = self._calculate_market_timing_score(
            signal, adjusted_volatility, volume_ratio, session
        )

        return DayTradingRecommendation(
            symbol=symbol,
            name=symbol_name,
            signal=signal,
            entry_timing=entry_timing,
            target_profit=target_profit,
            stop_loss=stop_loss,
            holding_time=holding_time,
            confidence=confidence,
            risk_level=risk_level,
            volume_trend=volume_trend,
            price_momentum=momentum_desc,
            intraday_volatility=adjusted_volatility,
            market_timing_score=market_timing_score
        )

    def _get_session_multiplier(self, session: TradingSession) -> float:
        """時間帯別の動きやすさ係数"""
        multipliers = {
            TradingSession.PRE_MARKET: 0.5,        # 寄り前は動き小さい
            TradingSession.MORNING_SESSION: 1.3,    # 前場は活発
            TradingSession.LUNCH_BREAK: 0.3,       # 昼休みは動かない
            TradingSession.AFTERNOON_SESSION: 1.0,  # 後場は標準
            TradingSession.AFTER_MARKET: 0.4       # 引け後は限定的
        }
        return multipliers.get(session, 1.0)

    def _fetch_mock_market_data(self, symbol: str, date_str: str) -> Dict[str, Any]:
        """
        改善版市場データ取得メソッド（Issue #849対応）

        1. 実データ取得の試行
        2. 段階的フォールバック機能
        3. 改善されたエラーハンドリング
        4. モックデータ通知システム
        """
        try:
            if self.data_fetcher is None:
                logger.warning(f"データフェッチャーが初期化されていません - モックデータ使用: {symbol}")
                return self._generate_intelligent_mock_data(symbol, date_str)

            # YFinanceFetcherで実データ取得を試行
            date_obj = datetime.strptime(date_str, '%Y%m%d')

            # データ取得の複数回試行（ネットワーク問題対応）
            for attempt in range(3):
                try:
                    historical_data = self.data_fetcher.get_historical_data(
                        code=symbol, period="5d", interval="1d"
                    )

                    if historical_data is not None and not historical_data.empty:
                        return self._process_real_market_data(historical_data, symbol)

                except Exception as e:
                    logger.debug(f"データ取得試行{attempt + 1}失敗 {symbol}: {e}")
                    if attempt < 2:  # 最後の試行でない場合は少し待つ
                        import time
                        time.sleep(0.5)

            # 実データ取得失敗 - モックデータフォールバック
            logger.info(f"実データ取得失敗のためモックデータ使用: {symbol}")
            return self._generate_intelligent_mock_data(symbol, date_str)

        except Exception as e:
            logger.error(f"市場データ取得中にエラー発生 {symbol}: {e}")
            return self._generate_intelligent_mock_data(symbol, date_str)

    def _process_real_market_data(self, historical_data, symbol: str) -> Dict[str, Any]:
        """実データの処理"""
        try:
            latest_data = historical_data.iloc[-1]
            prev_close = historical_data.iloc[-2]['Close'] if len(historical_data) >= 2 else latest_data['Close']

            return {
                "Open": float(latest_data.get("Open", 0)),
                "High": float(latest_data.get("High", 0)),
                "Low": float(latest_data.get("Low", 0)),
                "Close": float(latest_data.get("Close", 0)),
                "Volume": int(latest_data.get("Volume", 0)),
                "PrevClose": float(prev_close),
                "DateTime": latest_data.name,
                "DataSource": "REAL"
            }
        except Exception as e:
            logger.warning(f"実データ処理エラー {symbol}: {e}")
            return self._generate_intelligent_mock_data(symbol, "fallback")

    def _generate_intelligent_mock_data(self, symbol: str, date_str: str) -> Dict[str, Any]:
        """改善されたモックデータ生成"""
        # 設定に基づく通知制御
        mock_notification = self.config.get("data_fallback", {}).get("mock_data_notification", False)
        if mock_notification:
            logger.info(f"モックデータを生成中: {symbol}")

        # より現実的な価格レンジ（日本株価帯を考慮）
        seed_base = hash(symbol + date_str + "market_data") % 100000
        np.random.seed(seed_base)

        # 銘柄に応じた価格帯設定
        if len(symbol) == 4 and symbol.isdigit():
            # 日本株（4桁コード）
            price_base = int(symbol) * 0.5 + 1000  # 銘柄コードに基づく基準価格
        else:
            price_base = 2000  # その他の銘柄

        price_base = min(max(price_base, 100), 10000)  # 100円〜10,000円の範囲

        # より現実的な価格変動
        daily_volatility = np.random.uniform(0.01, 0.05)  # 1-5%の日中変動

        prev_close = price_base * (1 + np.random.uniform(-0.02, 0.02))
        gap = np.random.uniform(-0.01, 0.01)  # オーバーナイトギャップ
        open_price = prev_close * (1 + gap)

        high_price = open_price * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
        low_price = open_price * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
        close_price = open_price + (high_price - low_price) * np.random.uniform(-0.5, 0.5)

        # 出来高も現実的に
        volume_base = 1_000_000 if price_base > 1000 else 5_000_000
        volume = int(volume_base * np.random.uniform(0.5, 3.0))

        return {
            "Open": round(open_price, 0),
            "High": round(high_price, 0),
            "Low": round(low_price, 0),
            "Close": round(close_price, 0),
            "Volume": volume,
            "PrevClose": round(prev_close, 0),
            "DateTime": datetime.now(),
            "DataSource": "MOCK"
        }

    def _prepare_features_for_prediction(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        予測用特徴量準備メソッド
        RandomForestモデルが必要とする形式にデータを変換する。
        ここでは、簡易的なテクニカル指標と価格・出来高特徴量を生成。
        """
        # 必要なデータポイントが存在するか確認
        if not all(k in market_data for k in ["Open", "High", "Low", "Close", "Volume", "PrevClose"]):
            # データ不足の場合、ゼロ配列を返すか、エラーハンドリングを行う
            # ここでは暫定的に、num_featuresに合うようにゼロ埋めされた配列を返す
            num_features = 10  # モデルが期待する特徴量の数
            return np.zeros((1, num_features))

        open_p = market_data["Open"]
        high_p = market_data["High"]
        low_p = market_data["Low"]
        close_p = market_data["Close"]
        volume = market_data["Volume"]
        prev_close = market_data["PrevClose"]

        # 簡易的な特徴量エンジニアリング
        # 価格特徴量
        price_change = close_p - prev_close
        daily_range = high_p - low_p

        # 出来高特徴量 (ここでは単純な出来高を特徴量として使う)

        # テクニカル指標 (ここでは簡易的なRSIとMACDをモックとして生成)
        # 実際には src/day_trade/analysis/signals.py などからインポートして利用
        # RSIを模倣
        rsi_mock = 50 + (price_change / max(1, daily_range)) * 10
        rsi_mock = np.clip(rsi_mock, 0, 100) # RSIは0-100の範囲

        # MACDを模倣
        macd_mock = (close_p - prev_close) * 10
        macd_mock = np.clip(macd_mock, -100, 100) # 適当な範囲

        # モデルが期待する特徴量の順序に合わせて配列を作成
        # RandomForestModelは (n_samples, n_features) 形式を期待
        # 例: [price_change, daily_range, volume, rsi_mock, macd_mock, ...]
        # 現状は5つの特徴量
        features_array = np.array([
            price_change,
            daily_range,
            volume,
            rsi_mock,
            macd_mock,
            open_p, high_p, low_p, close_p, prev_close # 他の価格情報も特徴量として含める
        ]).reshape(1, -1) # 1サンプル、n_features

        # RandomForestModelの_hyperparameter_optimizationでnum_features=10としていたので、ここで合わせる
        # 実際には、config/ml.json の features に基づいて、適切な特徴量エンジニアリングパイプラインを構築する必要がある

        # もし特徴量の数が足りない場合、ゼロ埋めするなどの対応が必要
        num_expected_features = 10 # RandomForestModelが期待する特徴量の数
        if features_array.shape[1] < num_expected_features:
            padding = np.zeros((1, num_expected_features - features_array.shape[1]))
            features_array = np.hstack((features_array, padding))
        elif features_array.shape[1] > num_expected_features:
            features_array = features_array[:, :num_expected_features] # 多すぎる場合は切り詰める

        return features_array

    def _determine_daytrading_signal(self, volatility: float, momentum: float,
                                   volume: float, session: TradingSession) -> DayTradingSignal:
        """デイトレードシグナル判定"""

        # 昼休みや引け後は基本的に待機
        if session in [TradingSession.LUNCH_BREAK, TradingSession.AFTER_MARKET]:
            return DayTradingSignal.WAIT

        # 強いモメンタム + 高出来高 = 強いシグナル
        if momentum >= 2.0 and volume >= 1.8 and volatility >= 4.0:
            return DayTradingSignal.STRONG_BUY
        elif momentum <= -2.0 and volume >= 1.8 and volatility >= 4.0:
            return DayTradingSignal.STRONG_SELL

        # 中程度のシグナル
        elif momentum >= 1.0 and volume >= 1.2:
            return DayTradingSignal.BUY
        elif momentum <= -1.0 and volume >= 1.2:
            return DayTradingSignal.SELL

        # 横ばい・保留
        elif abs(momentum) < 0.5 or volatility < 2.0:
            return DayTradingSignal.WAIT

        # その他はホールド
        else:
            return DayTradingSignal.HOLD

    def _get_entry_timing(self, signal: DayTradingSignal, session: TradingSession) -> str:
        """エントリータイミング推奨"""

        timing_map = {
            DayTradingSignal.STRONG_BUY: {
                TradingSession.PRE_MARKET: "寄り成行で即エントリー",
                TradingSession.MORNING_SESSION: "今すぐ成行エントリー",
                TradingSession.AFTERNOON_SESSION: "押し目があれば即エントリー",
            },
            DayTradingSignal.BUY: {
                TradingSession.PRE_MARKET: "寄り後の値動き確認後",
                TradingSession.MORNING_SESSION: "5-10分値動き見て押し目エントリー",
                TradingSession.AFTERNOON_SESSION: "後場開始後の流れ確認",
            },
            DayTradingSignal.SELL: {
                TradingSession.MORNING_SESSION: "利確または損切り実行",
                TradingSession.AFTERNOON_SESSION: "大引け前に決済完了",
            },
            DayTradingSignal.STRONG_SELL: {
                TradingSession.MORNING_SESSION: "即座に決済",
                TradingSession.AFTERNOON_SESSION: "成行で即決済",
            },
            DayTradingSignal.WAIT: {
                TradingSession.PRE_MARKET: "寄り後の動向待ち",
                TradingSession.LUNCH_BREAK: "後場開始まで待機",
                TradingSession.AFTER_MARKET: "明日の寄り付き待ち",
            },
            DayTradingSignal.HOLD: {
                TradingSession.MORNING_SESSION: "ポジション維持・様子見",
                TradingSession.AFTERNOON_SESSION: "大引けまでホールド検討",
            }
        }

        return timing_map.get(signal, {}).get(session, "市況確認後に判断")

    def _calculate_profit_stop_levels(self, signal: DayTradingSignal,
                                    volatility: float) -> Tuple[float, float]:
        """利確・損切りライン計算"""

        # ボラティリティベースの利確・損切り設定
        base_profit = min(volatility * 0.6, 3.0)  # 最大3%利確
        base_stop = min(volatility * 0.4, 2.0)    # 最大2%損切り

        signal_adjustments = {
            DayTradingSignal.STRONG_BUY: (base_profit * 1.2, base_stop * 0.8),
            DayTradingSignal.BUY: (base_profit, base_stop),
            DayTradingSignal.STRONG_SELL: (base_profit * 1.2, base_stop * 0.8),
            DayTradingSignal.SELL: (base_profit, base_stop),
            DayTradingSignal.HOLD: (base_profit * 0.7, base_stop * 1.2),
            DayTradingSignal.WAIT: (0.5, 0.5)  # 待機時は小幅
        }

        profit, stop = signal_adjustments.get(signal, (base_profit, base_stop))
        return round(profit, 1), round(stop, 1)

    def _get_recommended_holding_time(self, signal: DayTradingSignal,
                                    session: TradingSession) -> str:
        """推奨保有時間"""

        if signal == DayTradingSignal.STRONG_BUY:
            return "30分〜2時間（短期集中）"
        elif signal == DayTradingSignal.BUY:
            return "1〜4時間（場中完結）"
        elif signal in [DayTradingSignal.SELL, DayTradingSignal.STRONG_SELL]:
            return "即座に決済"
        elif signal == DayTradingSignal.HOLD:
            if session == TradingSession.MORNING_SESSION:
                return "大引けまで保有検討"
            else:
                return "1〜2時間程度"
        else:  # WAIT
            return "エントリーチャンス待ち"

    def _calculate_daytrading_confidence(self, signal: DayTradingSignal,
                                       volatility: float, volume: float,
                                       session: TradingSession) -> float:
        """デイトレード信頼度計算"""

        base_confidence = 70.0

        # シグナル強度による調整
        signal_boost = {
            DayTradingSignal.STRONG_BUY: 15,
            DayTradingSignal.BUY: 10,
            DayTradingSignal.STRONG_SELL: 15,
            DayTradingSignal.SELL: 10,
            DayTradingSignal.HOLD: 5,
            DayTradingSignal.WAIT: -10
        }.get(signal, 0)

        # 出来高による調整
        volume_boost = min((volume - 1.0) * 10, 15)

        # ボラティリティによる調整（適度が最適）
        if 3.0 <= volatility <= 6.0:
            volatility_boost = 10  # 適度なボラティリティ
        elif volatility > 6.0:
            volatility_boost = 5   # 高すぎるボラティリティ
        else:
            volatility_boost = 0   # 低ボラティリティ

        # 時間帯による調整
        session_boost = {
            TradingSession.MORNING_SESSION: 10,
            TradingSession.AFTERNOON_SESSION: 5,
            TradingSession.PRE_MARKET: -5,
            TradingSession.LUNCH_BREAK: -15,
            TradingSession.AFTER_MARKET: -10
        }.get(session, 0)
        final_confidence = base_confidence + signal_boost + volume_boost + volatility_boost + session_boost
        return max(50, min(95, final_confidence))

    def _assess_daytrading_risk(self, volatility: float, signal: DayTradingSignal) -> str:
        """デイトレードリスク評価"""

        if volatility >= 6.0:
            return "高"
        elif volatility >= 4.0:
            return "中"
        else:
            return "低"

    def _describe_volume_trend(self, volume_ratio: float) -> str:
        """出来高動向の説明"""
        if volume_ratio >= 2.0:
            return "出来高急増"
        elif volume_ratio >= 1.5:
            return "出来高増加"
        elif volume_ratio >= 1.2:
            return "出来高やや多い"
        elif volume_ratio >= 0.8:
            return "出来高普通"
        else:
            return "出来高少ない"

    def _describe_price_momentum(self, momentum: float) -> str:
        """価格モメンタムの説明"""
        if momentum >= 2.0:
            return "強い上昇"
        elif momentum >= 1.0:
            return "上昇"
        elif momentum >= 0.5:
            return "やや上昇"
        elif momentum <= -2.0:
            return "強い下落"
        elif momentum <= -1.0:
            return "下落"
        elif momentum <= -0.5:
            return "やや下落"
        else:
            return "横ばい"

    def _calculate_market_timing_score(self, signal: DayTradingSignal,
                                     volatility: float, volume: float,
                                     session: TradingSession) -> float:
        """市場タイミングスコア算出"""

        # 基本スコア
        score = 50.0

        # シグナル強度
        signal_scores = {
            DayTradingSignal.STRONG_BUY: 90,
            DayTradingSignal.BUY: 75,
            DayTradingSignal.STRONG_SELL: 85,
            DayTradingSignal.SELL: 70,
            DayTradingSignal.HOLD: 60,
            DayTradingSignal.WAIT: 40
        }
        score = signal_scores.get(signal, 50)

        # ボラティリティ調整（デイトレードには適度が最適）
        if 3.0 <= volatility <= 6.0:
            score += 15
        elif volatility > 6.0:
            score += 5  # 高すぎは危険
        else:
            score -= 10  # 低すぎは面白くない

        # 出来高調整
        score += min((volume - 1.0) * 20, 15)

        # 時間帯調整
        session_adj = {
            TradingSession.MORNING_SESSION: 10,
            TradingSession.AFTERNOON_SESSION: 5,
            TradingSession.PRE_MARKET: -5,
            TradingSession.LUNCH_BREAK: -20,
            TradingSession.AFTER_MARKET: -15
        }.get(session, 0)
        score += session_adj

        return max(0, min(100, score))

    def get_session_advice(self) -> str:
        """現在の時間帯に応じたアドバイス（正確な市場時間管理）"""
        # 正確な市場時間管理システム使用
        if self.market_manager:
            return self.market_manager.get_session_advice()

        # フォールバック: 従来のアドバイス
        session = self._get_current_trading_session()
        current_time = datetime.now().strftime("%H:%M")
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%m/%d")

        advice_map = {
            TradingSession.PRE_MARKET: f"[{current_time}] 寄り前時間帯\n・寄り付き前の情報収集時間\n・前日の米国市況・為替確認\n・寄り成行注文の準備",

            TradingSession.MORNING_SESSION: f"[{current_time}] 前場時間帯\n・最も活発な取引時間\n・デイトレードのメイン時間帯\n・急な値動きに注意",

            TradingSession.LUNCH_BREAK: f"[{current_time}] 昼休み時間帯\n・取引量減少・値動き限定的\n・後場に向けた戦略見直し時間\n・新規エントリーは慎重に",

            TradingSession.AFTERNOON_SESSION: f"[{current_time}] 後場時間帯\n・決済中心の時間帯\n・大引けに向けポジション整理\n・15:00までに決済完了推奨",

            TradingSession.AFTER_MARKET: f"[{current_time}] 翌日前場予想モード\n・{tomorrow_str}（明日）の前場戦略立案時間\n・オーバーナイトギャップ・寄り付き戦略検討\n・翌日のエントリー銘柄選定"
        }

        return advice_map.get(session, "取引時間外")

    def _load_configuration(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            if self.config_path.exists():
                import json
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"設定ファイルを読み込みました: {self.config_path}")
                return config
            else:
                # デフォルト設定を作成
                default_config = {
                    "symbol_mapping": {},
                    "data_fallback": {
                        "enable_mock_data": True,
                        "mock_data_notification": False
                    },
                    "market_timing": {
                        "session_multipliers": {
                            "MORNING_SESSION": 1.3,
                            "AFTERNOON_SESSION": 1.1
                        }
                    },
                    "logging": {
                        "level": "INFO"
                    }
                }

                # 親ディレクトリを作成
                self.config_path.parent.mkdir(parents=True, exist_ok=True)

                # デフォルト設定を保存
                import json
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=2)

                logger.info(f"デフォルト設定ファイルを作成しました: {self.config_path}")
                return default_config

        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return {}

def create_day_trading_engine(config_path: Optional[str] = None) -> PersonalDayTradingEngine:
    """
    デイトレードエンジンファクトリー関数

    Args:
        config_path: 設定ファイルパス（Noneの場合はデフォルト）

    Returns:
        PersonalDayTradingEngineインスタンス
    """
    logger.info("デイトレードエンジンを作成中...")
    return PersonalDayTradingEngine(config_path=config_path)

# Issue #849対応: デモコードは examples/demo_day_trading_engine.py に分離済み
# デモ実行する場合は以下のコマンドを使用:
# python examples/demo_day_trading_engine.py

if __name__ == "__main__":
    print("Day Trading Engine Module - Issue #849対応完了")
    print("デモ実行: python examples/demo_day_trading_engine.py")
    print("統合テスト: python -m pytest tests/test_day_trading_engine_improved.py")
