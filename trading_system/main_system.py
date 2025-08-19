import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from .backtest_engine import BacktestEngine
from .data_provider import MultiSourceDataProvider
from .datastructures import (
    NextMorningPrediction,
    PositionRecommendation,
    MarketSentiment,
)
from .enums import MarketDirection, PredictionConfidence, RiskLevel
from .risk_manager import RiskManager
from .sentiment_analyzer import AdvancedSentimentAnalyzer

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from prediction_accuracy_enhancement import PredictionAccuracyEnhancer
    ACCURACY_ENHANCEMENT_AVAILABLE = True
except ImportError:
    ACCURACY_ENHANCEMENT_AVAILABLE = False

try:
    from data_quality_monitor import DataQualityMonitor, DataSource
    DATA_QUALITY_AVAILABLE = True
except ImportError:
    DATA_QUALITY_AVAILABLE = False


class NextMorningTradingAdvanced:
    """翌朝場モード高度化システム"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        # 設定読み込み
        self.config_path = config_path or Path("config/next_morning_advanced_config.yaml")
        self.config = self._load_config()

        # コンポーネント初期化
        self.data_provider = MultiSourceDataProvider()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.risk_manager = RiskManager()
        self.backtest_engine = BacktestEngine()

        # 外部システム統合
        self.accuracy_enhancer = None
        self.data_quality_monitor = None

        if ACCURACY_ENHANCEMENT_AVAILABLE:
            try:
                self.accuracy_enhancer = PredictionAccuracyEnhancer()
                self.logger.info("Prediction accuracy enhancer integrated")
            except Exception as e:
                self.logger.warning(f"Failed to initialize accuracy enhancer: {e}")

        if DATA_QUALITY_AVAILABLE:
            try:
                self.data_quality_monitor = DataQualityMonitor()
                self.logger.info("Data quality monitor integrated")
            except Exception as e:
                self.logger.warning(f"Failed to initialize data quality monitor: {e}")

        # 学習済みモデル（セッション間で保持）
        self.trained_models = {}

        self.logger.info("Advanced Next Morning Trading System initialized")

    def _load_config(self) -> Dict[str, Any]:
        """設定読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Config loading failed: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'prediction': {
                'models': ['random_forest', 'xgboost', 'ensemble'],
                'lookback_days': 60,
                'feature_engineering': True,
                'confidence_threshold': 0.6
            },
            'risk_management': {
                'max_position_size': 0.1,
                'max_daily_loss': 0.02,
                'risk_free_rate': 0.02,
                'risk_reward_ratio': 2.0
            },
            'data_sources': {
                'primary_weight': 0.5,
                'secondary_weight': 0.3,
                'tertiary_weight': 0.2,
                'quality_threshold': 0.7
            },
            'backtest': {
                'default_period_months': 12,
                'min_trades': 30,
                'benchmark_symbol': '^N225'
            }
        }

    async def predict_next_morning(self, symbol: str, account_balance: float = 1000000,
                                  risk_tolerance: float = 0.05) -> NextMorningPrediction:
        """翌朝場予測実行"""
        start_time = time.time()
        self.logger.info(f"Starting next morning prediction for {symbol}")

        try:
            # 1. マルチソースデータ取得
            lookback_days = self.config['prediction']['lookback_days']
            period = f"{lookback_days}d"

            market_data = await self.data_provider.get_multi_source_data(symbol, period)

            if market_data.empty:
                raise ValueError(f"No market data available for {symbol}")

            # 2. データ品質検証
            if self.data_quality_monitor:
                try:
                    quality_result = await self.data_quality_monitor.validate_stock_data(
                        symbol, market_data, DataSource.YAHOO_FINANCE
                    )
                    if not quality_result.is_valid:
                        self.logger.warning(f"Data quality issues detected for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Data quality check failed: {e}")

            # 3. 高度センチメント分析
            market_sentiment = await self.sentiment_analyzer.analyze_market_sentiment(symbol, market_data)

            # 4. 機械学習予測
            ml_prediction = await self._generate_ml_prediction(symbol, market_data, market_sentiment)

            # 5. リスク指標計算
            risk_metrics = self.risk_manager.calculate_risk_metrics(market_data)

            # 6. ポジション推奨生成
            position_recommendation = self._generate_position_recommendation(
                symbol, ml_prediction, risk_metrics, account_balance, risk_tolerance
            )

            # 7. 結果統合
            prediction = NextMorningPrediction(
                symbol=symbol,
                prediction_date=datetime.now(),
                market_direction=ml_prediction['direction'],
                predicted_change_percent=ml_prediction['expected_change'],
                confidence=self._determine_confidence_level(ml_prediction['confidence']),
                confidence_score=ml_prediction['confidence'],
                market_sentiment=market_sentiment,
                risk_metrics=risk_metrics,
                position_recommendation=position_recommendation,
                supporting_data={
                    'data_sources': self.data_provider.data_sources,
                    'sentiment_factors': market_sentiment.key_factors,
                    'model_features': ml_prediction.get('features_used', []),
                    'prediction_timestamp': datetime.now().isoformat()
                },
                model_used=ml_prediction['model_name'],
                data_sources=list(self.data_provider.data_sources.keys())
            )

            processing_time = time.time() - start_time
            self.logger.info(f"Next morning prediction completed for {symbol} in {processing_time:.2f}s")
            self.logger.info(f"Prediction: {prediction.market_direction.value} ({prediction.predicted_change_percent:+.2f}%) - Confidence: {prediction.confidence.value}")

            return prediction

        except Exception as e:
            self.logger.error(f"Next morning prediction failed for {symbol}: {e}")
            raise

    async def _generate_ml_prediction(self, symbol: str, data: pd.DataFrame, sentiment: MarketSentiment) -> Dict[str, Any]:
        """機械学習予測生成"""
        try:
            # 特徴量エンジニアリング
            features = self._engineer_features(data, sentiment)

            if features.empty:
                raise ValueError("Feature engineering failed")

            # ターゲット変数作成（翌日リターン）
            returns = data['Close'].pct_change().shift(-1)  # 翌日リターン
            target_direction = (returns > 0).astype(int)  # 上昇=1, 下降=0

            # 有効なデータのみ使用
            valid_idx = ~(features.isnull().any(axis=1) | returns.isnull())
            X = features[valid_idx]
            y_direction = target_direction[valid_idx]
            y_returns = returns[valid_idx]

            if len(X) < 20:
                raise ValueError("Insufficient training data")

            # モデル学習・予測
            models_config = self.config['prediction']['models']
            predictions = {}

            for model_name in models_config:
                try:
                    pred = await self._train_and_predict_model(model_name, X, y_direction, y_returns)
                    predictions[model_name] = pred
                except Exception as e:
                    self.logger.warning(f"Model {model_name} failed: {e}")

            if not predictions:
                raise ValueError("All models failed")

            # アンサンブル予測
            ensemble_prediction = self._ensemble_predictions(predictions)

            return {
                'direction': self._convert_to_market_direction(ensemble_prediction['direction_prob'], ensemble_prediction['expected_return']),
                'expected_change': ensemble_prediction['expected_return'],
                'confidence': ensemble_prediction['confidence'],
                'model_name': 'Ensemble',
                'features_used': list(features.columns),
                'individual_predictions': predictions
            }

        except Exception as e:
            self.logger.error(f"ML prediction generation failed: {e}")
            # フォールバック：簡単な技術分析ベース予測
            return self._fallback_prediction(data)

    def _engineer_features(self, data: pd.DataFrame, sentiment: MarketSentiment) -> pd.DataFrame:
        """特徴量エンジニアリング"""
        features = pd.DataFrame(index=data.index)

        try:
            if 'Close' not in data.columns:
                return features

            # 価格関連特徴量
            features['price'] = data['Close']
            features['log_price'] = np.log(data['Close'])

            # リターン特徴量
            for period in [1, 3, 5, 10, 20]:
                features[f'return_{period}d'] = data['Close'].pct_change(period)
                features[f'log_return_{period}d'] = np.log(data['Close'] / data['Close'].shift(period))

            # 移動平均特徴量
            for period in [5, 10, 20, 50]:
                ma = data['Close'].rolling(period).mean()
                features[f'sma_{period}'] = ma
                features[f'price_vs_sma_{period}'] = data['Close'] / ma - 1

            # テクニカル指標
            features['rsi'] = self._calculate_rsi(data['Close'], 14)

            macd, macd_signal = self._calculate_macd(data['Close'])
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd - macd_signal

            # ボラティリティ特徴量
            features['volatility_20d'] = data['Close'].rolling(20).std()
            features['volatility_ratio'] = features['volatility_20d'] / features['volatility_20d'].rolling(60).mean()

            # ボリューム特徴量
            if 'Volume' in data.columns:
                features['volume'] = data['Volume']
                features['volume_sma_20'] = data['Volume'].rolling(20).mean()
                features['volume_ratio'] = data['Volume'] / features['volume_sma_20']

            # センチメント特徴量
            features['sentiment_score'] = sentiment.sentiment_score
            features['sentiment_confidence'] = sentiment.confidence
            features['technical_sentiment'] = sentiment.technical_sentiment
            features['fundamental_sentiment'] = sentiment.fundamental_sentiment
            features['news_sentiment'] = sentiment.news_sentiment

            # ラグ特徴量
            for lag in [1, 2, 3, 5]:
                features[f'close_lag_{lag}'] = data['Close'].shift(lag)
                features[f'return_lag_{lag}'] = data['Close'].pct_change().shift(lag)

            # 相互作用特徴量
            features['rsi_x_sentiment'] = features['rsi'] * features['sentiment_score']
            features['volatility_x_volume'] = features['volatility_20d'] * features.get('volume_ratio', 1)

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)

    async def _train_and_predict_model(self, model_name: str, X: pd.DataFrame,
                                     y_direction: pd.Series, y_returns: pd.Series) -> Dict[str, Any]:
        """個別モデル学習・予測"""
        try:
            # 最新のデータを予測用に分離
            X_train = X.iloc[:-1]
            X_current = X.iloc[-1:]
            y_direction_train = y_direction.iloc[:-1]
            y_returns_train = y_returns.iloc[:-1]

            if len(X_train) < 10:
                raise ValueError("Insufficient training data")

            # モデル作成
            if model_name == 'random_forest':
                direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
                returns_model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                direction_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                returns_model = xgb.XGBRegressor(random_state=42)
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                direction_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                returns_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
            else:
                # フォールバック
                direction_model = RandomForestClassifier(n_estimators=50, random_state=42)
                returns_model = RandomForestRegressor(n_estimators=50, random_state=42)

            # モデル学習
            direction_model.fit(X_train, y_direction_train)
            returns_model.fit(X_train, y_returns_train)

            # 予測実行
            direction_prob = direction_model.predict_proba(X_current)[0][1]  # 上昇確率
            expected_return = returns_model.predict(X_current)[0]

            # 信頼度計算
            confidence = self._calculate_model_confidence(direction_model, X_train, y_direction_train)

            return {
                'direction_prob': direction_prob,
                'expected_return': expected_return,
                'confidence': confidence,
                'model_name': model_name
            }

        except Exception as e:
            self.logger.error(f"Model training/prediction failed for {model_name}: {e}")
            raise

    def _calculate_model_confidence(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """モデル信頼度計算"""
        try:
            # 交差検証スコア
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)//10), n_jobs=-1)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()

            # 信頼度 = 精度 - 分散ペナルティ
            confidence = mean_score - std_score * 0.5
            return np.clip(confidence, 0.0, 1.0)

        except Exception:
            return 0.6  # デフォルト信頼度

    def _ensemble_predictions(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """アンサンブル予測"""
        if not predictions:
            raise ValueError("No predictions to ensemble")

        # 重み付け（信頼度ベース）
        total_confidence = sum(pred['confidence'] for pred in predictions.values())
        weights = {name: pred['confidence'] / total_confidence for name, pred in predictions.items()}

        # 加重平均
        ensemble_direction_prob = sum(pred['direction_prob'] * weights[name] for name, pred in predictions.items())
        ensemble_expected_return = sum(pred['expected_return'] * weights[name] for name, pred in predictions.items())
        ensemble_confidence = sum(pred['confidence'] * weights[name] for name, pred in predictions.items())

        return {
            'direction_prob': ensemble_direction_prob,
            'expected_return': ensemble_expected_return,
            'confidence': ensemble_confidence
        }

    def _convert_to_market_direction(self, direction_prob: float, expected_return: float) -> MarketDirection:
        """市場方向変換"""
        if direction_prob > 0.7 and expected_return > 0.02:
            return MarketDirection.STRONG_BULLISH
        elif direction_prob > 0.6 and expected_return > 0.005:
            return MarketDirection.BULLISH
        elif direction_prob < 0.3 and expected_return < -0.02:
            return MarketDirection.STRONG_BEARISH
        elif direction_prob < 0.4 and expected_return < -0.005:
            return MarketDirection.BEARISH
        else:
            return MarketDirection.NEUTRAL

    def _determine_confidence_level(self, confidence_score: float) -> PredictionConfidence:
        """信頼度レベル判定"""
        if confidence_score >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 0.8:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.6:
            return PredictionConfidence.MEDIUM
        elif confidence_score >= 0.4:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW

    def _fallback_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """フォールバック予測"""
        try:
            # 簡単な移動平均クロス戦略
            if len(data) < 20:
                return {
                    'direction': MarketDirection.NEUTRAL,
                    'expected_change': 0.0,
                    'confidence': 0.3,
                    'model_name': 'Fallback',
                    'features_used': []
                }

            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]

            if sma_5 > sma_20 and current_price > sma_5:
                direction = MarketDirection.BULLISH
                expected_change = 0.01
            elif sma_5 < sma_20 and current_price < sma_5:
                direction = MarketDirection.BEARISH
                expected_change = -0.01
            else:
                direction = MarketDirection.NEUTRAL
                expected_change = 0.0

            return {
                'direction': direction,
                'expected_change': expected_change,
                'confidence': 0.5,
                'model_name': 'Fallback_MA_Cross',
                'features_used': ['SMA_5', 'SMA_20']
            }

        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {e}")
            return {
                'direction': MarketDirection.NEUTRAL,
                'expected_change': 0.0,
                'confidence': 0.3,
                'model_name': 'Default',
                'features_used': []
            }

    def _generate_position_recommendation(self, symbol: str, ml_prediction: Dict,
                                        risk_metrics, account_balance: float,
                                        risk_tolerance: float) -> PositionRecommendation:
        """ポジション推奨生成"""
        try:
            direction = ml_prediction['direction']
            expected_return = ml_prediction['expected_change']
            confidence = ml_prediction['confidence']

            # 現在価格（模擬）
            entry_price = 1000.0  # 実装では実際の市場価格を取得

            # ポジションサイズ計算
            position_size = self.risk_manager.calculate_position_size(
                account_balance, risk_tolerance, risk_metrics.volatility
            )

            # リスクレベル判定
            risk_level = self._determine_risk_level(risk_metrics.volatility, abs(expected_return))

            # 損切り・利確価格計算
            stop_loss_price = self.risk_manager.calculate_stop_loss(
                entry_price, direction, risk_metrics.volatility
            )

            target_price = self.risk_manager.calculate_target_price(
                entry_price, direction, expected_return
            )

            # 保有期間推定
            holding_period = self._estimate_holding_period(direction, confidence)

            # 根拠生成
            rationale = self._generate_rationale(ml_prediction, risk_metrics)

            return PositionRecommendation(
                symbol=symbol,
                direction=direction,
                confidence=self._determine_confidence_level(confidence),
                entry_price=entry_price,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                position_size_percentage=position_size * 100,
                risk_level=risk_level,
                holding_period=holding_period,
                rationale=rationale
            )

        except Exception as e:
            self.logger.error(f"Position recommendation generation failed: {e}")
            # デフォルト推奨
            return PositionRecommendation(
                symbol=symbol,
                direction=MarketDirection.NEUTRAL,
                confidence=PredictionConfidence.LOW,
                entry_price=1000.0,
                target_price=1000.0,
                stop_loss_price=950.0,
                position_size_percentage=1.0,
                risk_level=RiskLevel.LOW,
                holding_period="1日",
                rationale="システムエラーのため保守的な推奨"
            )

    def _determine_risk_level(self, volatility: float, expected_return: float) -> RiskLevel:
        """リスクレベル判定"""
        risk_score = volatility + abs(expected_return)

        if risk_score < 0.05:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.08:
            return RiskLevel.LOW
        elif risk_score < 0.12:
            return RiskLevel.MEDIUM
        elif risk_score < 0.18:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _estimate_holding_period(self, direction: MarketDirection, confidence: float) -> str:
        """保有期間推定"""
        if direction == MarketDirection.NEUTRAL:
            return "様子見"
        elif confidence > 0.8:
            return "1-3日"
        elif confidence > 0.6:
            return "翌朝場のみ"
        else:
            return "日中監視"

    def _generate_rationale(self, ml_prediction: Dict, risk_metrics) -> str:
        """根拠生成"""
        direction = ml_prediction['direction']
        confidence = ml_prediction['confidence']
        model_name = ml_prediction['model_name']

        rationale_parts = []

        # 方向性
        if direction != MarketDirection.NEUTRAL:
            rationale_parts.append(f"{direction.value}トレンドを予測")

        # 信頼度
        confidence_desc = "高い" if confidence > 0.7 else "中程度" if confidence > 0.5 else "低い"
        rationale_parts.append(f"予測信頼度は{confidence_desc}({confidence:.1%})")

        # モデル
        rationale_parts.append(f"{model_name}モデルに基づく")

        # リスク
        volatility_desc = "高" if risk_metrics.volatility > 0.3 else "中" if risk_metrics.volatility > 0.2 else "低"
        rationale_parts.append(f"ボラティリティ{volatility_desc}")

        return "。".join(rationale_parts) + "。"

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD計算"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    async def run_strategy_backtest(self, symbol: str, months: int = 12):
        """戦略バックテスト実行"""
        try:
            # バックテスト期間設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30)

            # 履歴データ取得
            data = await self.data_provider.get_multi_source_data(symbol, f"{months*30}d")

            if data.empty:
                raise ValueError(f"No historical data for {symbol}")

            # バックテスト実行
            result = await self.backtest_engine.run_backtest(
                strategy_function=self.predict_next_morning,
                data=data,
                start_date=start_date,
                end_date=end_date
            )

            self.logger.info(f"Backtest completed for {symbol}: Win rate {result.win_rate:.1%}, Total return {result.total_return:+.1%}")

            return result

        except Exception as e:
            self.logger.error(f"Strategy backtest failed for {symbol}: {e}")
            raise
