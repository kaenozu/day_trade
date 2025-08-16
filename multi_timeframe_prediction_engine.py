#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタイムフレーム予測エンジン - Issue #882対応

デイトレード以外の取引に対応：1週間、1ヶ月、3ヶ月の予測機能を提供
既存のマルチタイムフレーム分析システムとML予測モデルを統合
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Windows環境対策
import sys
import os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 機械学習ライブラリ
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# 既存システムインポート
try:
    from src.day_trade.analysis.multi_timeframe_analysis import MultiTimeframeAnalyzer
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False

try:
    from overnight_prediction_model import OvernightPredictionModel
    OVERNIGHT_MODEL_AVAILABLE = True
except ImportError:
    OVERNIGHT_MODEL_AVAILABLE = False

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionTimeframe(Enum):
    """予測期間"""
    DAILY = "1日"
    WEEKLY = "1週間"
    MONTHLY = "1ヶ月"
    QUARTERLY = "3ヶ月"
    YEARLY = "1年"

class TradingStyle(Enum):
    """取引スタイル"""
    DAY_TRADING = "デイトレード"
    SWING_TRADING = "スイングトレード"
    POSITION_TRADING = "ポジショントレード"
    LONG_TERM_INVESTING = "長期投資"

@dataclass
class TimeframePredictionResult:
    """期間予測結果"""
    timeframe: PredictionTimeframe
    trading_style: TradingStyle
    symbol: str
    prediction_direction: str          # "UP", "DOWN", "SIDEWAYS"
    confidence: float                  # 0-100%
    expected_return: float             # 期待リターン(%)
    risk_level: str                   # "LOW", "MEDIUM", "HIGH"
    entry_price: float                # エントリー価格
    target_price: float               # 目標価格
    stop_loss_price: float            # 損切り価格
    holding_period_days: int          # 推奨保有日数
    volatility_forecast: float        # ボラティリティ予測
    feature_importance: Dict[str, float]  # 重要特徴量
    model_accuracy: float             # モデル精度
    created_at: datetime              # 予測作成時刻

@dataclass 
class MultiTimeframePrediction:
    """マルチタイムフレーム統合予測"""
    symbol: str
    predictions: Dict[PredictionTimeframe, TimeframePredictionResult]
    consensus_direction: str          # 統合方向性
    consensus_confidence: float       # 統合信頼度
    best_timeframe: PredictionTimeframe  # 最適期間
    recommended_strategy: str         # 推奨戦略
    risk_assessment: Dict[str, Any]   # リスク評価

class MultiTimeframePredictionEngine:
    """マルチタイムフレーム予測エンジン"""
    
    def __init__(self):
        """初期化"""
        self.models = {}  # 期間別モデル
        self.scalers = {}  # 期間別スケーラー
        self.feature_columns = {}  # 期間別特徴量
        
        # 既存システム統合
        if MULTI_TIMEFRAME_AVAILABLE:
            self.multi_analyzer = MultiTimeframeAnalyzer()
        else:
            self.multi_analyzer = None
            
        if OVERNIGHT_MODEL_AVAILABLE:
            self.overnight_model = OvernightPredictionModel()
        else:
            self.overnight_model = None
            
        # タイムフレーム設定
        self.timeframe_configs = {
            PredictionTimeframe.DAILY: {
                "feature_periods": [5, 10, 20],
                "prediction_horizon": 1,
                "min_data_points": 252,  # 1年
                "trading_style": TradingStyle.DAY_TRADING,
                "model_type": "classification"
            },
            PredictionTimeframe.WEEKLY: {
                "feature_periods": [4, 8, 13],  # 約1, 2, 3ヶ月
                "prediction_horizon": 5,  # 5営業日
                "min_data_points": 104,  # 2年の週足
                "trading_style": TradingStyle.SWING_TRADING,
                "model_type": "classification"
            },
            PredictionTimeframe.MONTHLY: {
                "feature_periods": [3, 6, 12],  # 3, 6, 12ヶ月
                "prediction_horizon": 22,  # 約1ヶ月
                "min_data_points": 36,   # 3年の月足
                "trading_style": TradingStyle.POSITION_TRADING,
                "model_type": "classification"
            },
            PredictionTimeframe.QUARTERLY: {
                "feature_periods": [2, 4, 8],   # 6ヶ月, 1年, 2年
                "prediction_horizon": 66,  # 約3ヶ月
                "min_data_points": 20,   # 5年の四半期
                "trading_style": TradingStyle.LONG_TERM_INVESTING,
                "model_type": "regression"
            }
        }
        
        logger.info("マルチタイムフレーム予測エンジン初期化完了")

    async def fetch_training_data(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """訓練データ取得"""
        try:
            import yfinance as yf
            
            # Yahoo Financeからデータ取得
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.error(f"データ取得失敗: {symbol}")
                return pd.DataFrame()
                
            # 基本的な前処理
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            logger.info(f"データ取得完了: {symbol}, {len(data)}期間")
            return data
            
        except Exception as e:
            logger.error(f"データ取得エラー ({symbol}): {e}")
            return pd.DataFrame()

    def create_features_for_timeframe(self, data: pd.DataFrame, timeframe: PredictionTimeframe) -> pd.DataFrame:
        """期間別特徴量作成"""
        try:
            config = self.timeframe_configs[timeframe]
            periods = config["feature_periods"]
            
            # データをタイムフレームに合わせてリサンプル
            if timeframe == PredictionTimeframe.WEEKLY:
                data = data.resample('W').agg({
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif timeframe == PredictionTimeframe.MONTHLY:
                data = data.resample('M').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min', 
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif timeframe == PredictionTimeframe.QUARTERLY:
                data = data.resample('Q').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last', 
                    'Volume': 'sum'
                }).dropna()
                
            features = pd.DataFrame(index=data.index)
            
            # 価格関連特徴量
            for period in periods:
                if len(data) > period:
                    # 移動平均
                    features[f'sma_{period}'] = data['Close'].rolling(period).mean()
                    features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
                    
                    # 価格モメンタム
                    features[f'momentum_{period}'] = data['Close'].pct_change(period)
                    features[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
                    
                    # ボラティリティ
                    features[f'volatility_{period}'] = data['Close'].pct_change().rolling(period).std()
                    
                    # 価格位置
                    features[f'price_position_{period}'] = (data['Close'] - data['Close'].rolling(period).min()) / (data['Close'].rolling(period).max() - data['Close'].rolling(period).min())
            
            # テクニカル指標
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # ボリンジャーバンド
            sma20 = data['Close'].rolling(20).mean()
            std20 = data['Close'].rolling(20).std()
            features['bb_upper'] = sma20 + (std20 * 2)
            features['bb_lower'] = sma20 - (std20 * 2)
            features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # 出来高指標
            features['volume_sma'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma']
            
            # 価格レンジ
            features['daily_range'] = (data['High'] - data['Low']) / data['Close']
            features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            
            # ターゲット作成
            horizon = config["prediction_horizon"]
            features['future_return'] = data['Close'].pct_change(horizon).shift(-horizon)
            
            if config["model_type"] == "classification":
                # 分類: 上昇/下落
                features['target'] = (features['future_return'] > 0).astype(int)
            else:
                # 回帰: リターン値
                features['target'] = features['future_return']
            
            # NaN値除去
            features = features.dropna()
            
            logger.info(f"{timeframe.value}特徴量作成完了: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"特徴量作成エラー ({timeframe.value}): {e}")
            return pd.DataFrame()

    async def train_timeframe_model(self, symbol: str, timeframe: PredictionTimeframe) -> bool:
        """期間別モデル訓練"""
        try:
            if not ML_AVAILABLE:
                logger.error("機械学習ライブラリが利用できません")
                return False
                
            logger.info(f"{timeframe.value}モデル訓練開始: {symbol}")
            
            # データ取得
            raw_data = await self.fetch_training_data(symbol)
            if raw_data.empty:
                return False
                
            # 特徴量作成
            features = self.create_features_for_timeframe(raw_data, timeframe)
            if features.empty:
                return False
                
            config = self.timeframe_configs[timeframe]
            
            # 最小データ点チェック
            if len(features) < config["min_data_points"]:
                logger.warning(f"データ不足: {len(features)} < {config['min_data_points']}")
                return False
                
            # 特徴量とターゲット分離
            feature_cols = [col for col in features.columns if col not in ['target', 'future_return']]
            X = features[feature_cols]
            y = features['target']
            
            # データ分割（時系列順序保持）
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル訓練
            if config["model_type"] == "classification":
                model = lgb.LGBMClassifier(
                    objective='binary',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train_scaled, y_train)
                
                # 評価
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"{timeframe.value}精度: {accuracy:.4f}")
                
            else:  # 回帰
                model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train_scaled, y_train)
                
                # 評価
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                logger.info(f"{timeframe.value}R²: {r2:.4f}, RMSE: {rmse:.4f}")
            
            # モデル保存
            self.models[timeframe] = model
            self.scalers[timeframe] = scaler
            self.feature_columns[timeframe] = feature_cols
            
            logger.info(f"{timeframe.value}モデル訓練完了")
            return True
            
        except Exception as e:
            logger.error(f"モデル訓練エラー ({timeframe.value}): {e}")
            return False

    async def predict_timeframe(self, symbol: str, timeframe: PredictionTimeframe, current_data: pd.DataFrame) -> Optional[TimeframePredictionResult]:
        """期間別予測実行"""
        try:
            if timeframe not in self.models:
                logger.warning(f"モデル未訓練: {timeframe.value}")
                return None
                
            model = self.models[timeframe]
            scaler = self.scalers[timeframe]
            feature_cols = self.feature_columns[timeframe]
            config = self.timeframe_configs[timeframe]
            
            # 特徴量作成
            features = self.create_features_for_timeframe(current_data, timeframe)
            if features.empty:
                return None
                
            # 最新データ取得
            latest_features = features[feature_cols].iloc[-1:].fillna(0)
            X_scaled = scaler.transform(latest_features)
            
            # 予測実行
            if config["model_type"] == "classification":
                prediction = model.predict(X_scaled)[0]
                prediction_proba = model.predict_proba(X_scaled)[0]
                confidence = max(prediction_proba) * 100
                direction = "UP" if prediction == 1 else "DOWN"
                expected_return = 5.0 if prediction == 1 else -3.0  # デフォルト値
                
            else:  # 回帰
                prediction = model.predict(X_scaled)[0]
                confidence = min(95, max(50, abs(prediction) * 100))
                direction = "UP" if prediction > 0.02 else "DOWN" if prediction < -0.02 else "SIDEWAYS"
                expected_return = prediction * 100
            
            # 現在価格とリスク計算
            current_price = current_data['Close'].iloc[-1]
            volatility = current_data['Close'].pct_change().rolling(20).std().iloc[-1]
            
            # 目標価格・損切り価格設定
            if direction == "UP":
                target_price = current_price * (1 + abs(expected_return) / 100)
                stop_loss_price = current_price * 0.95
            elif direction == "DOWN":
                target_price = current_price * (1 + expected_return / 100)
                stop_loss_price = current_price * 1.05
            else:
                target_price = current_price
                stop_loss_price = current_price * 0.98
            
            # リスクレベル判定
            if volatility > 0.03:
                risk_level = "HIGH"
            elif volatility > 0.015:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # 特徴量重要度
            try:
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_cols, importances))
                # 上位5つの重要特徴量
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                feature_importance = dict(sorted_importance)
            except:
                feature_importance = {}
            
            # 推奨保有日数
            holding_period_days = config["prediction_horizon"]
            
            result = TimeframePredictionResult(
                timeframe=timeframe,
                trading_style=config["trading_style"],
                symbol=symbol,
                prediction_direction=direction,
                confidence=confidence,
                expected_return=expected_return,
                risk_level=risk_level,
                entry_price=current_price,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                holding_period_days=holding_period_days,
                volatility_forecast=volatility * 100,
                feature_importance=feature_importance,
                model_accuracy=confidence / 100,  # 簡易的な精度
                created_at=datetime.now()
            )
            
            logger.info(f"{timeframe.value}予測完了: {direction} (信頼度: {confidence:.1f}%)")
            return result
            
        except Exception as e:
            logger.error(f"予測エラー ({timeframe.value}): {e}")
            return None

    async def generate_multi_timeframe_prediction(self, symbol: str) -> Optional[MultiTimeframePrediction]:
        """マルチタイムフレーム統合予測"""
        try:
            logger.info(f"マルチタイムフレーム予測開始: {symbol}")
            
            # データ取得
            current_data = await self.fetch_training_data(symbol, period="2y")
            if current_data.empty:
                return None
            
            # 各期間で予測実行
            predictions = {}
            for timeframe in [PredictionTimeframe.DAILY, PredictionTimeframe.WEEKLY, PredictionTimeframe.MONTHLY]:
                # モデル訓練（未訓練の場合）
                if timeframe not in self.models:
                    success = await self.train_timeframe_model(symbol, timeframe)
                    if not success:
                        continue
                
                # 予測実行
                prediction = await self.predict_timeframe(symbol, timeframe, current_data)
                if prediction:
                    predictions[timeframe] = prediction
            
            if not predictions:
                logger.error("全期間で予測失敗")
                return None
            
            # 統合分析
            directions = [pred.prediction_direction for pred in predictions.values()]
            confidences = [pred.confidence for pred in predictions.values()]
            
            # 合意方向
            up_count = directions.count("UP")
            down_count = directions.count("DOWN")
            sideways_count = directions.count("SIDEWAYS")
            
            if up_count > down_count and up_count > sideways_count:
                consensus_direction = "UP"
            elif down_count > up_count and down_count > sideways_count:
                consensus_direction = "DOWN"
            else:
                consensus_direction = "SIDEWAYS"
            
            # 統合信頼度（重み付き平均）
            weights = {
                PredictionTimeframe.DAILY: 0.3,
                PredictionTimeframe.WEEKLY: 0.4,
                PredictionTimeframe.MONTHLY: 0.3
            }
            
            consensus_confidence = sum(
                pred.confidence * weights.get(tf, 0.33) 
                for tf, pred in predictions.items()
            )
            
            # 最適期間（最高信頼度）
            best_timeframe = max(predictions.keys(), key=lambda tf: predictions[tf].confidence)
            
            # 推奨戦略
            if consensus_confidence > 75 and consensus_direction != "SIDEWAYS":
                recommended_strategy = f"積極的{consensus_direction}戦略"
            elif consensus_confidence > 60:
                recommended_strategy = f"慎重{consensus_direction}戦略"
            else:
                recommended_strategy = "様子見・分散投資"
            
            # リスク評価
            avg_volatility = np.mean([pred.volatility_forecast for pred in predictions.values()])
            risk_factors = []
            
            if avg_volatility > 3.0:
                risk_factors.append("高ボラティリティ")
            if len(set(directions)) == 3:
                risk_factors.append("期間間方向性不一致")
            if min(confidences) < 55:
                risk_factors.append("低信頼度予測含む")
            
            risk_assessment = {
                "overall_risk": "HIGH" if len(risk_factors) >= 2 else "MEDIUM" if risk_factors else "LOW",
                "risk_factors": risk_factors,
                "volatility_forecast": avg_volatility,
                "diversification_recommended": len(set(directions)) > 1
            }
            
            result = MultiTimeframePrediction(
                symbol=symbol,
                predictions=predictions,
                consensus_direction=consensus_direction,
                consensus_confidence=consensus_confidence,
                best_timeframe=best_timeframe,
                recommended_strategy=recommended_strategy,
                risk_assessment=risk_assessment
            )
            
            logger.info(f"マルチタイムフレーム予測完了: {consensus_direction} (信頼度: {consensus_confidence:.1f}%)")
            return result
            
        except Exception as e:
            logger.error(f"マルチタイムフレーム予測エラー: {e}")
            return None

    def print_prediction_summary(self, prediction: MultiTimeframePrediction):
        """予測結果サマリー表示"""
        try:
            print(f"\n{'='*60}")
            print(f"【マルチタイムフレーム予測サマリー】{prediction.symbol}")
            print(f"{'='*60}")
            
            print(f"\n【統合予測】")
            print(f"  方向性: {prediction.consensus_direction}")
            print(f"  信頼度: {prediction.consensus_confidence:.1f}%")
            print(f"  推奨戦略: {prediction.recommended_strategy}")
            print(f"  最適期間: {prediction.best_timeframe.value}")
            
            print(f"\n【期間別予測】")
            for timeframe, pred in prediction.predictions.items():
                print(f"  {timeframe.value}:")
                print(f"    方向性: {pred.prediction_direction}")
                print(f"    信頼度: {pred.confidence:.1f}%")
                print(f"    期待リターン: {pred.expected_return:.1f}%")
                print(f"    リスクレベル: {pred.risk_level}")
                print(f"    推奨保有期間: {pred.holding_period_days}日")
                print(f"    エントリー価格: {pred.entry_price:.2f}")
                print(f"    目標価格: {pred.target_price:.2f}")
                print(f"    損切り価格: {pred.stop_loss_price:.2f}")
            
            print(f"\n【リスク評価】")
            risk = prediction.risk_assessment
            print(f"  総合リスク: {risk['overall_risk']}")
            print(f"  ボラティリティ予測: {risk['volatility_forecast']:.2f}%")
            if risk['risk_factors']:
                print(f"  リスク要因:")
                for factor in risk['risk_factors']:
                    print(f"    - {factor}")
            print(f"  分散投資推奨: {'はい' if risk['diversification_recommended'] else 'いいえ'}")
            
            print(f"\n{'='*60}")
            
        except Exception as e:
            logger.error(f"サマリー表示エラー: {e}")

# 使用例・テスト
async def main():
    """メイン関数"""
    try:
        print("マルチタイムフレーム予測エンジンテスト開始")
        
        # エンジン初期化
        engine = MultiTimeframePredictionEngine()
        
        # テスト銘柄
        test_symbols = ["^N225", "7203.T", "6758.T"]  # 日経平均、トヨタ、ソニーG
        
        for symbol in test_symbols:
            print(f"\n{symbol}の予測実行中...")
            
            # マルチタイムフレーム予測
            prediction = await engine.generate_multi_timeframe_prediction(symbol)
            
            if prediction:
                engine.print_prediction_summary(prediction)
            else:
                print(f"❌ {symbol}の予測に失敗しました")
                
        print("\n✅ マルチタイムフレーム予測エンジンテスト完了")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())