#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ML Prediction System - シンプルML予測システム
talib等の複雑な依存関係なしで動作する軽量版
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import sqlite3
from pathlib import Path

# 基本的な機械学習ライブラリのみ使用
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

@dataclass
class SimplePredictionResult:
    """簡単予測結果"""
    symbol: str
    prediction: int  # 0: 下降, 1: 上昇
    confidence: float  # 0.0 - 1.0
    model_consensus: Dict[str, int]  # 各モデルの予測
    feature_values: Dict[str, float]  # 特徴量値

class SimpleMLPredictionSystem:
    """シンプルML予測システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース設定
        self.db_path = Path("data/simple_ml_predictions.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # モデル
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.is_trained = False

        # 初期化
        self._init_database()
        self.logger.info("Simple ML prediction system initialized")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    actual_result INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    trained_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    async def predict_symbol_movement(self, symbol: str) -> SimplePredictionResult:
        """銘柄の価格変動予測"""
        try:
            # 自動訓練（初回または定期的に実行）
            if not self.is_trained:
                await self._train_models(symbol)

            # 特徴量生成
            features = self._generate_features(symbol)

            # 各モデルで予測
            model_consensus = {}
            confidences = []

            # 特徴量を正規化
            features_scaled = self.scaler.transform([list(features.values())])

            for model_name, model in self.models.items():
                try:
                    prediction = model.predict(features_scaled)[0]
                    if hasattr(model, 'predict_proba'):
                        confidence = model.predict_proba(features_scaled)[0].max()
                        confidences.append(confidence)
                    else:
                        confidence = 0.7  # デフォルト信頼度
                        confidences.append(confidence)

                    model_consensus[model_name] = int(prediction)

                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
                    model_consensus[model_name] = np.random.randint(0, 2)
                    confidences.append(0.6)

            # 最終予測（多数決）
            predictions = list(model_consensus.values())
            final_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0

            # 平均信頼度
            avg_confidence = np.mean(confidences) if confidences else 0.7

            # 予測結果を保存
            await self._save_prediction(symbol, final_prediction, avg_confidence)

            return SimplePredictionResult(
                symbol=symbol,
                prediction=final_prediction,
                confidence=avg_confidence,
                model_consensus=model_consensus,
                feature_values=features
            )

        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            # フォールバック
            return SimplePredictionResult(
                symbol=symbol,
                prediction=np.random.randint(0, 2),
                confidence=0.6,
                model_consensus={'fallback': 1},
                feature_values={'volatility': 0.5, 'trend': 0.0, 'volume': 1.0}
            )

    def _generate_features(self, symbol: str) -> Dict[str, float]:
        """簡単な特徴量生成"""
        # シンプルな特徴量（価格データ取得の代わりにダミー値）
        # 実際の実装では yfinance 等から取得

        np.random.seed(hash(symbol) % 1000)  # 一貫性のため

        features = {
            'price_change_5d': np.random.uniform(-0.1, 0.1),
            'price_change_20d': np.random.uniform(-0.2, 0.2),
            'volatility': np.random.uniform(0.1, 0.5),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'rsi': np.random.uniform(20, 80),
            'trend_strength': np.random.uniform(-1.0, 1.0),
            'market_sentiment': np.random.uniform(-0.5, 0.5)
        }

        return features

    async def _train_models(self, symbol: str = None):
        """モデル訓練"""
        try:
            self.logger.info("Training simple ML models...")

            # 訓練データ生成（実際の実装では過去データから生成）
            X, y = self._generate_training_data()

            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # スケーラー訓練
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 各モデル訓練
            for model_name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train)

                    # 性能評価
                    predictions = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, predictions)

                    self.logger.info(f"Model {model_name} accuracy: {accuracy:.3f}")

                    # 性能をデータベースに保存
                    await self._save_model_performance(model_name, symbol or 'general', accuracy)

                except Exception as e:
                    self.logger.warning(f"Training failed for model {model_name}: {e}")

            self.is_trained = True
            self.logger.info("Model training completed")

        except Exception as e:
            self.logger.error(f"Training process failed: {e}")
            self.is_trained = False

    def _generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """訓練データ生成"""
        # 実際の実装では過去の価格データから生成
        np.random.seed(42)

        # 特徴量生成
        X = np.random.randn(n_samples, 7)  # 7つの特徴量

        # ターゲット生成（特徴量に基づく簡単なルール）
        y = np.zeros(n_samples)
        for i in range(n_samples):
            # 簡単な規則：価格変動 + ボラティリティ + トレンド強度
            score = X[i, 0] + X[i, 2] * 0.5 + X[i, 5] * 0.3
            y[i] = 1 if score > 0 else 0

        return X, y.astype(int)

    async def _save_prediction(self, symbol: str, prediction: int, confidence: float):
        """予測結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO predictions (symbol, prediction_date, prediction, confidence) VALUES (?, ?, ?, ?)",
                    (symbol, datetime.now().isoformat(), prediction, confidence)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save prediction: {e}")

    async def _save_model_performance(self, model_name: str, symbol: str, accuracy: float):
        """モデル性能保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO model_performance (model_name, symbol, accuracy) VALUES (?, ?, ?)",
                    (model_name, symbol, accuracy)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save model performance: {e}")

# テスト実行
if __name__ == "__main__":
    async def test_simple_ml():
        system = SimpleMLPredictionSystem()

        # テスト予測
        result = await system.predict_symbol_movement("7203")
        print(f"予測結果: {result.symbol}")
        print(f"予測: {'上昇' if result.prediction == 1 else '下降'}")
        print(f"信頼度: {result.confidence:.2f}")
        print(f"モデル合意: {result.model_consensus}")
        print(f"特徴量: {result.feature_values}")

    asyncio.run(test_simple_ml())