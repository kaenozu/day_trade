#!/usr/bin/env python3
"""
デイトレードエンジンのコアクラス

PersonalDayTradingEngineとファクトリー関数を提供します。
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .enums import DayTradingSignal, TradingSession
from .recommendation import DayTradingRecommendation
from .data_handlers import SymbolManager, MarketDataHandler, FeaturePreparator
from .signal_analysis import TradingSessionAnalyzer, SignalAnalyzer, RealDataConverter
from .forecaster import TomorrowForecaster


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
    from market_time_manager import MarketTimeManager
    MARKET_TIME_AVAILABLE = True
except ImportError:
    MARKET_TIME_AVAILABLE = False

# ML モデル関連インポート（オプション）
try:
    from integrated_model_loader import IntegratedModelLoader
    INTEGRATED_MODEL_AVAILABLE = True
except ImportError:
    INTEGRATED_MODEL_AVAILABLE = False

try:
    from ..monitoring.model_performance_monitor import ModelPerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False


class PersonalDayTradingEngine:
    """個人向けデイトレードエンジン"""

    def __init__(self, config_path: Optional[str] = None):
        # 設定読み込み（オプション）
        self.config_path = Path(config_path) if config_path else Path("config/day_trading_config.json")
        self.config = self._load_configuration() if config_path else {}

        # 銘柄管理システム初期化
        self.symbol_manager = SymbolManager(self.config)

        # 市場時間管理システム初期化
        if MARKET_TIME_AVAILABLE:
            self.market_manager = MarketTimeManager()
            self.time_mode = "ACCURATE"
        else:
            self.market_manager = None
            self.time_mode = "SIMPLE"

        # セッション分析器初期化
        self.session_analyzer = TradingSessionAnalyzer(self.market_manager)
        self.current_session = self.session_analyzer.get_current_trading_session()

        # 実データプロバイダー初期化
        if REAL_DATA_AVAILABLE:
            self.real_data_engine = RealDataAnalysisEngine()
            self.data_mode = "REAL"
        else:
            self.real_data_engine = None
            self.data_mode = "DEMO"

        # データハンドラー初期化
        self.data_handler = MarketDataHandler(self.config)
        self.feature_preparator = FeaturePreparator()

        # シグナル分析器初期化
        self.signal_analyzer = SignalAnalyzer()
        self.real_data_converter = RealDataConverter()

        # 翌日予測器初期化
        self.forecaster = None

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
                # 翌日予測器にモデルを設定
                self.forecaster = TomorrowForecaster(self.model_loader)
                
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
                self.forecaster = TomorrowForecaster()
        else:
            self.model_loader = None
            self.model_performance_monitor = None
            self.forecaster = TomorrowForecaster()

        # ログレベル設定
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        logger.setLevel(log_level)

        logger.info(f"デイトレードエンジン初期化完了 - データモード: {self.data_mode}, 時間モード: {self.time_mode}")

    async def get_today_daytrading_recommendations(self, limit: int = 20) -> List[DayTradingRecommendation]:
        """今日のデイトレード推奨取得（実データ対応）"""
        if self.model_performance_monitor:
            await self.model_performance_monitor.check_and_trigger_retraining()

        current_session = self.session_analyzer.get_current_trading_session()

        # 大引け後は翌日前場予想モードに切り替え
        if current_session == TradingSession.AFTER_MARKET:
            return await self.forecaster.get_tomorrow_premarket_forecast(
                self.symbol_manager.get_symbols(), limit
            )

        # 実データエンジン使用可能な場合
        if self.real_data_engine:
            real_recommendations = await self.real_data_engine.analyze_daytrading_opportunities(limit)
            return [self.real_data_converter.convert_to_daytrading_recommendation(rec, current_session) 
                    for rec in real_recommendations]

        # フォールバック: ダミーデータ使用
        recommendations = []
        symbols = self.symbol_manager.get_symbols()
        symbol_list = list(symbols.keys())[:limit]

        for symbol in symbol_list:
            rec = await self._analyze_daytrading_opportunity(symbol, symbols[symbol], current_session)
            recommendations.append(rec)

        # 市場タイミングスコア順でソート
        recommendations.sort(key=lambda x: x.market_timing_score, reverse=True)

        return recommendations

    async def _analyze_daytrading_opportunity(self, symbol: str, symbol_name: str, session: TradingSession) -> DayTradingRecommendation:
        """デイトレード機会分析"""
        market_data = self.data_handler.fetch_market_data(symbol, str(datetime.now().date()).replace('-', ''))
        features = self.feature_preparator.prepare_features_for_prediction(market_data)

        if self.model_loader:
            try:
                prediction_results, _ = await self.model_loader.predict(symbol, features)
                return self._build_daytrading_recommendation_from_prediction(symbol, symbol_name, session, prediction_results)
            except Exception as e:
                logger.warning(f"AI予測失敗: {e}")

        # フォールバック予測
        return self._build_fallback_daytrading_recommendation(symbol, symbol_name, session, market_data)

    def _build_daytrading_recommendation_from_prediction(self, symbol: str, symbol_name: str, 
                                                         session: TradingSession, prediction_results: Any) -> DayTradingRecommendation:
        """AIの予測結果から当日の推奨を構築する"""
        import numpy as np
        
        predicted_values = prediction_results.predictions.flatten()
        confidence_values = prediction_results.confidence.flatten() if prediction_results.confidence is not None else np.array([0.0])

        intraday_volatility = predicted_values[0] if len(predicted_values) > 0 else 0.0
        volume_ratio = predicted_values[1] if len(predicted_values) > 1 else 0.0
        price_momentum = predicted_values[2] if len(predicted_values) > 2 else 0.0

        session_multiplier = self.session_analyzer.get_session_multiplier(session)
        adjusted_volatility = intraday_volatility * session_multiplier

        signal = self.signal_analyzer.determine_daytrading_signal(
            volatility=adjusted_volatility,
            momentum=price_momentum,
            volume=volume_ratio,
            session=session
        )
        entry_timing = self.signal_analyzer.get_entry_timing(signal, session)
        target_profit, stop_loss = self.signal_analyzer.calculate_profit_stop_levels(
            signal, adjusted_volatility
        )
        holding_time = self.signal_analyzer.get_recommended_holding_time(signal, session)

        if confidence_values.size > 0:
            confidence = np.mean(confidence_values) * 100
        else:
            confidence = self.signal_analyzer.calculate_daytrading_confidence(
                signal, adjusted_volatility, volume_ratio, session
            )

        risk_level = self.signal_analyzer.assess_daytrading_risk(adjusted_volatility, signal)
        volume_trend = self.signal_analyzer.describe_volume_trend(volume_ratio)
        momentum_desc = self.signal_analyzer.describe_price_momentum(price_momentum)
        market_timing_score = self.signal_analyzer.calculate_market_timing_score(
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

    def _build_fallback_daytrading_recommendation(self, symbol: str, symbol_name: str,
                                                  session: TradingSession, market_data: Dict[str, Any]) -> DayTradingRecommendation:
        """フォールバック当日推奨"""
        import numpy as np
        
        # 簡易的な分析ロジック
        volatility = abs(market_data["High"] - market_data["Low"]) / market_data["Close"] * 100 if market_data["Close"] > 0 else 3.0
        momentum = (market_data["Close"] - market_data["PrevClose"]) / market_data["PrevClose"] * 100 if market_data["PrevClose"] > 0 else 0
        volume_ratio = 1.0 + np.random.uniform(-0.3, 0.3)

        session_multiplier = self.session_analyzer.get_session_multiplier(session)
        adjusted_volatility = volatility * session_multiplier

        signal = self.signal_analyzer.determine_daytrading_signal(
            volatility=adjusted_volatility,
            momentum=momentum,
            volume=volume_ratio,
            session=session
        )

        return DayTradingRecommendation(
            symbol=symbol,
            name=symbol_name,
            signal=signal,
            entry_timing=self.signal_analyzer.get_entry_timing(signal, session),
            target_profit=self.signal_analyzer.calculate_profit_stop_levels(signal, adjusted_volatility)[0],
            stop_loss=self.signal_analyzer.calculate_profit_stop_levels(signal, adjusted_volatility)[1],
            holding_time=self.signal_analyzer.get_recommended_holding_time(signal, session),
            confidence=self.signal_analyzer.calculate_daytrading_confidence(signal, adjusted_volatility, volume_ratio, session),
            risk_level=self.signal_analyzer.assess_daytrading_risk(adjusted_volatility, signal),
            volume_trend=self.signal_analyzer.describe_volume_trend(volume_ratio),
            price_momentum=self.signal_analyzer.describe_price_momentum(momentum),
            intraday_volatility=adjusted_volatility,
            market_timing_score=self.signal_analyzer.calculate_market_timing_score(signal, adjusted_volatility, volume_ratio, session)
        )

    def get_session_advice(self) -> str:
        """現在の時間帯に応じたアドバイス（正確な市場時間管理）"""
        return self.session_analyzer.get_session_advice()

    def _load_configuration(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            if self.config_path.exists():
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