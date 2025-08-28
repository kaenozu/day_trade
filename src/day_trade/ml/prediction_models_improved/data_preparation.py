#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ準備パイプライン - ML Prediction Models Data Preparation

ML予測モデルのデータ準備を行います。
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.day_trade.ml.core_types import (
    DataPreparationError,
    DataProvider,
    DataQuality,
    PredictionTask,
)
from .data_types import TrainingConfig
from .feature_engineering import FeatureEngineer

# Real data provider availability check  
try:
    from src.day_trade.data.stock_fetcher import StockFetcher
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


class DataPreparationPipeline:
    """データ準備パイプライン（強化版）"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler() if self.config.enable_scaling else None
        self.feature_selector = None
        self.feature_engineer = FeatureEngineer()

    async def prepare_training_data(
        self, 
        symbol: str, 
        period: str = "1y",
        data_provider: Optional[DataProvider] = None
    ) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series], DataQuality]:
        """訓練データ準備（強化版）"""
        try:
            self.logger.info(f"データ準備開始: {symbol}")

            # データ取得
            data = await self._fetch_data(symbol, period, data_provider)

            # データ品質評価
            is_valid, quality, quality_message = self._assess_data_quality(data)
            if not is_valid or quality < self.config.min_data_quality:
                raise DataPreparationError(f"データ品質不足: {quality_message}")

            self.logger.info(f"データ品質評価: {quality.value} - {quality_message}")

            # 特徴量エンジニアリング
            features = await self.feature_engineer.engineer_features(symbol, data)

            # 特徴量後処理
            features = self._postprocess_features(features)

            # ターゲット変数作成
            targets = self.feature_engineer.create_target_variables(data)

            # データ整合性チェック
            features, targets = self._align_data(features, targets)

            self.logger.info(f"データ準備完了: features={features.shape}, quality={quality.value}")

            return features, targets, quality

        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            raise DataPreparationError(f"データ準備失敗: {e}") from e

    async def _fetch_data(
        self, 
        symbol: str, 
        period: str, 
        data_provider: Optional[DataProvider]
    ) -> pd.DataFrame:
        """データ取得（強化版）"""
        if data_provider and REAL_DATA_PROVIDER_AVAILABLE:
            # 実データプロバイダー使用
            try:
                data = await data_provider.get_stock_data(symbol, period)
                if not data.empty:
                    return data
                else:
                    self.logger.warning("実データプロバイダーが空データを返しました")
            except Exception as e:
                self.logger.error(f"実データ取得失敗: {e}")

        # フォールバック: 模擬データ生成（開発・テスト環境のみ）
        self.logger.warning("模擬データを生成します（本番環境では推奨されません）")
        return self._generate_mock_data(symbol, period)

    def _generate_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """模擬データ生成（改良版）"""
        # より現実的な模擬データを生成
        np.random.seed(hash(symbol) % (2**32))

        if period == "1y":
            days = 252
        elif period == "6mo":
            days = 126
        elif period == "3mo":
            days = 63
        else:
            days = 252

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # より現実的な価格変動シミュレーション
        initial_price = 1000 + np.random.randint(-500, 500)
        returns = np.random.normal(0.0005, 0.02, days)  # 日次リターン
        prices = [initial_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 10))  # 最小価格制限

        # OHLC価格生成
        high_multiplier = np.random.uniform(1.005, 1.03, days)
        low_multiplier = np.random.uniform(0.97, 0.995, days)

        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'High': [p * mult for p, mult in zip(prices, high_multiplier)],
            'Low': [p * mult for p, mult in zip(prices, low_multiplier)],
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.5, days).astype(int)
        }, index=dates)

        # 価格整合性修正
        for i in range(len(data)):
            data.loc[data.index[i], 'High'] = max(data.iloc[i]['High'],
                                                  data.iloc[i]['Open'],
                                                  data.iloc[i]['Close'])
            data.loc[data.index[i], 'Low'] = min(data.iloc[i]['Low'],
                                                 data.iloc[i]['Open'],
                                                 data.iloc[i]['Close'])

        return data

    def _assess_data_quality(self, data: pd.DataFrame) -> Tuple[bool, DataQuality, str]:
        """データ品質評価（詳細版）"""
        try:
            issues = []
            quality_score = 100.0

            # 基本チェック
            if data.empty:
                return False, DataQuality.INSUFFICIENT, "データが空"

            if len(data) < 30:
                return False, DataQuality.INSUFFICIENT, f"データ不足: {len(data)}行"

            # 必須カラムチェック
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                quality_score -= 50
                issues.append(f"必須カラム不足: {missing_columns}")

            # 欠損値チェック
            missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_rate > 0.1:
                quality_score -= 20
                issues.append(f"欠損値率高: {missing_rate:.1%}")
            elif missing_rate > 0.05:
                quality_score -= 10
                issues.append(f"欠損値率中: {missing_rate:.1%}")

            # 価格データ整合性
            if all(col in data.columns for col in required_columns):
                invalid_ohlc = ((data['High'] < data['Low']) |
                               (data['High'] < data['Open']) |
                               (data['High'] < data['Close']) |
                               (data['Low'] > data['Open']) |
                               (data['Low'] > data['Close'])).sum()

                if invalid_ohlc > 0:
                    invalid_rate = invalid_ohlc / len(data)
                    quality_score -= min(30, invalid_rate * 100)
                    issues.append(f"OHLC不整合: {invalid_ohlc}件")

            # ゼロまたは負の価格
            if 'Close' in data.columns:
                invalid_prices = (data['Close'] <= 0).sum()
                if invalid_prices > 0:
                    quality_score -= 25
                    issues.append(f"無効価格: {invalid_prices}件")

            # データの連続性（週末除く）
            date_gaps = self._detect_date_gaps(data.index)
            if date_gaps > len(data) * 0.1:
                quality_score -= 15
                issues.append(f"日付ギャップ多: {date_gaps}件")

            # 品質レベル決定
            if quality_score >= 90:
                quality = DataQuality.EXCELLENT
            elif quality_score >= 75:
                quality = DataQuality.GOOD
            elif quality_score >= 60:
                quality = DataQuality.FAIR
            elif quality_score >= 40:
                quality = DataQuality.POOR
            else:
                quality = DataQuality.INSUFFICIENT

            success = quality_score >= 40
            message = f"スコア: {quality_score:.1f}" + (f" - {'; '.join(issues)}" if issues else "")

            return success, quality, message

        except Exception as e:
            return False, DataQuality.INSUFFICIENT, f"評価エラー: {e}"

    def _detect_date_gaps(self, dates: pd.DatetimeIndex) -> int:
        """日付ギャップ検出"""
        try:
            # 営業日ベースでギャップを検出
            business_days = pd.bdate_range(start=dates.min(), end=dates.max())
            expected_count = len(business_days)
            actual_count = len(dates)
            return max(0, expected_count - actual_count)
        except Exception:
            return 0

    def _postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """特徴量後処理"""
        try:
            # 無限値、異常値の処理
            features = features.replace([np.inf, -np.inf], np.nan)

            # 数値列の特定
            numeric_columns = features.select_dtypes(include=[np.number]).columns

            # 外れ値処理（IQR方式）
            if self.config.outlier_detection:
                for col in numeric_columns:
                    Q1 = features[col].quantile(0.25)
                    Q3 = features[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    features[col] = features[col].clip(lower_bound, upper_bound)

            # 欠損値処理
            if self.config.handle_missing_values:
                features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # スケーリング
            if self.config.enable_scaling and self.scaler:
                features[numeric_columns] = self.scaler.fit_transform(features[numeric_columns])

            return features

        except Exception as e:
            self.logger.error(f"特徴量後処理エラー: {e}")
            return features

    def _align_data(
        self, 
        features: pd.DataFrame, 
        targets: Dict[PredictionTask, pd.Series]
    ) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
        """データ整合性確保"""
        try:
            # 共通のインデックスを取得
            common_index = features.index
            for target in targets.values():
                common_index = common_index.intersection(target.index)

            # 最後の行を除外（未来の値が不明）
            common_index = common_index[:-1]

            # データを共通インデックスに合わせる
            aligned_features = features.loc[common_index]
            aligned_targets = {}

            for task, target in targets.items():
                aligned_targets[task] = target.loc[common_index].dropna()

            self.logger.info(f"データ整合完了: 共通サンプル数={len(common_index)}")

            return aligned_features, aligned_targets

        except Exception as e:
            self.logger.error(f"データ整合エラー: {e}")
            return features, targets

    def prepare_prediction_data(
        self, 
        symbol: str, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """予測用データ準備"""
        try:
            self.logger.debug(f"予測用データ準備開始: {symbol}")

            # 特徴量エンジニアリング（同期版）
            features = self.feature_engineer.prepare_prediction_features(symbol, data)

            # 後処理
            features = self._postprocess_features(features)

            self.logger.debug(f"予測用データ準備完了: {features.shape}")
            return features

        except Exception as e:
            self.logger.error(f"予測用データ準備エラー: {e}")
            raise DataPreparationError(f"予測用データ準備失敗: {e}") from e

    def validate_features(self, features: pd.DataFrame) -> Tuple[bool, str]:
        """特徴量検証"""
        return self.feature_engineer.validate_features(features)