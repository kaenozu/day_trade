#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ準備・特徴量エンジニアリングパイプライン

機械学習モデル用のデータ準備、特徴量エンジニアリング、
データ品質評価、前処理を行うシステムです。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .base_types import (
    TrainingConfig,
    FeatureEngineringConfig,
    DataQualityReport,
    DATA_QUALITY_THRESHOLDS
)
from src.day_trade.ml.core_types import (
    DataPreparationError,
    PredictionTask,
    DataQuality,
    DataProvider
)

# 外部依存の確認
try:
    from src.day_trade.ml.feature_engineering_improved import enhanced_feature_engineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    # 実データプロバイダーの確認（もし存在する場合）
    REAL_DATA_PROVIDER_AVAILABLE = False
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


class DataPreparationPipeline:
    """データ準備パイプライン（強化版）"""

    def __init__(self, config: Optional[TrainingConfig] = None,
                 feature_config: Optional[FeatureEngineringConfig] = None):
        self.config = config or TrainingConfig()
        self.feature_config = feature_config or FeatureEngineringConfig()
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler() if self.config.enable_scaling else None
        self.feature_selector = None

    async def prepare_training_data(self, symbol: str, period: str = "1y",
                                  data_provider: Optional[DataProvider] = None) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series], DataQuality]:
        """訓練データ準備（強化版）"""
        try:
            self.logger.info(f"データ準備開始: {symbol}")

            # データ取得
            data = await self._fetch_data(symbol, period, data_provider)

            # データ品質評価
            quality_report = self._assess_data_quality_detailed(data, symbol)
            if quality_report.quality_level < self.config.min_data_quality:
                raise DataPreparationError(f"データ品質不足: {quality_report.quality_level.value}")

            self.logger.info(f"データ品質評価: {quality_report.quality_level.value} - スコア: {quality_report.quality_score:.2f}")

            # 特徴量エンジニアリング
            features = await self._engineer_features(symbol, data)

            # 特徴量後処理
            features = self._postprocess_features(features)

            # ターゲット変数作成
            targets = self._create_target_variables(data)

            # データ整合性チェック
            features, targets = self._align_data(features, targets)

            self.logger.info(f"データ準備完了: features={features.shape}, quality={quality_report.quality_level.value}")

            return features, targets, quality_report.quality_level

        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            raise DataPreparationError(f"データ準備失敗: {e}") from e

    async def _fetch_data(self, symbol: str, period: str, data_provider: Optional[DataProvider]) -> pd.DataFrame:
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

    def _assess_data_quality_detailed(self, data: pd.DataFrame, symbol: str) -> DataQualityReport:
        """詳細データ品質評価"""
        try:
            issues = []
            recommendations = []
            quality_score = 100.0
            timestamp = datetime.now()

            # 基本チェック
            if data.empty:
                return DataQualityReport(
                    symbol=symbol, timestamp=timestamp, total_samples=0,
                    missing_values_count=0, missing_values_rate=0.0,
                    duplicate_rows=0, date_gaps_count=0,
                    price_anomalies=0, volume_anomalies=0, ohlc_inconsistencies=0,
                    quality_score=0.0, quality_level=DataQuality.INSUFFICIENT,
                    issues=["データが空"], recommendations=["データ取得を確認してください"]
                )

            total_samples = len(data)
            if total_samples < DATA_QUALITY_THRESHOLDS['min_samples'][DataQuality.POOR]:
                quality_score -= 50
                issues.append(f"データ不足: {total_samples}行")
                recommendations.append("より多くのデータを取得してください")

            # 必須カラムチェック
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                quality_score -= 50
                issues.append(f"必須カラム不足: {missing_columns}")
                recommendations.append(f"必須カラムを追加してください: {missing_columns}")

            # 欠損値チェック
            missing_count = data.isnull().sum().sum()
            missing_rate = missing_count / (len(data) * len(data.columns))
            
            if missing_rate > DATA_QUALITY_THRESHOLDS['missing_rate'][DataQuality.POOR]:
                quality_score -= 30
                issues.append(f"欠損値率高: {missing_rate:.1%}")
                recommendations.append("欠損値処理を検討してください")
            elif missing_rate > DATA_QUALITY_THRESHOLDS['missing_rate'][DataQuality.FAIR]:
                quality_score -= 15
                issues.append(f"欠損値率中: {missing_rate:.1%}")

            # 重複行チェック
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > 0:
                quality_score -= min(20, duplicate_rows / len(data) * 100)
                issues.append(f"重複行: {duplicate_rows}件")
                recommendations.append("重複行の削除を検討してください")

            # 価格データ整合性
            ohlc_inconsistencies = 0
            price_anomalies = 0
            volume_anomalies = 0
            
            if all(col in data.columns for col in required_columns):
                # OHLC整合性
                ohlc_inconsistencies = ((data['High'] < data['Low']) |
                                       (data['High'] < data['Open']) |
                                       (data['High'] < data['Close']) |
                                       (data['Low'] > data['Open']) |
                                       (data['Low'] > data['Close'])).sum()

                if ohlc_inconsistencies > 0:
                    invalid_rate = ohlc_inconsistencies / len(data)
                    quality_score -= min(30, invalid_rate * 100)
                    issues.append(f"OHLC不整合: {ohlc_inconsistencies}件")
                    recommendations.append("OHLC価格の整合性を確認してください")

                # 価格異常値
                price_anomalies = (data['Close'] <= 0).sum()
                if price_anomalies > 0:
                    quality_score -= 25
                    issues.append(f"無効価格: {price_anomalies}件")
                    recommendations.append("価格データの妥当性を確認してください")

                # 極端な価格変動
                returns = data['Close'].pct_change()
                extreme_moves = (returns.abs() > 0.5).sum()  # 50%以上の変動
                if extreme_moves > 0:
                    quality_score -= min(15, extreme_moves / len(data) * 100)
                    issues.append(f"極端な価格変動: {extreme_moves}件")

            # 出来高異常
            if 'Volume' in data.columns:
                zero_volume = (data['Volume'] <= 0).sum()
                if zero_volume > 0:
                    volume_anomalies = zero_volume
                    quality_score -= min(10, zero_volume / len(data) * 100)
                    issues.append(f"ゼロ出来高: {zero_volume}件")

            # データの連続性（週末除く）
            date_gaps_count = self._detect_date_gaps(data.index)
            if date_gaps_count > len(data) * 0.1:
                quality_score -= 15
                issues.append(f"日付ギャップ多: {date_gaps_count}件")
                recommendations.append("データの連続性を改善してください")

            # 品質レベル決定
            if quality_score >= 90:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 75:
                quality_level = DataQuality.GOOD
            elif quality_score >= 60:
                quality_level = DataQuality.FAIR
            elif quality_score >= 40:
                quality_level = DataQuality.POOR
            else:
                quality_level = DataQuality.INSUFFICIENT

            return DataQualityReport(
                symbol=symbol,
                timestamp=timestamp,
                total_samples=total_samples,
                missing_values_count=missing_count,
                missing_values_rate=missing_rate,
                duplicate_rows=duplicate_rows,
                date_gaps_count=date_gaps_count,
                price_anomalies=price_anomalies,
                volume_anomalies=volume_anomalies,
                ohlc_inconsistencies=ohlc_inconsistencies,
                quality_score=quality_score,
                quality_level=quality_level,
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"データ品質評価エラー: {e}")
            return DataQualityReport(
                symbol=symbol, timestamp=datetime.now(), total_samples=0,
                missing_values_count=0, missing_values_rate=0.0,
                duplicate_rows=0, date_gaps_count=0,
                price_anomalies=0, volume_anomalies=0, ohlc_inconsistencies=0,
                quality_score=0.0, quality_level=DataQuality.INSUFFICIENT,
                issues=[f"評価エラー: {e}"], recommendations=["データ品質評価処理を確認してください"]
            )

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

    async def _engineer_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング（強化版）"""
        try:
            if FEATURE_ENGINEERING_AVAILABLE:
                # 既存の特徴量エンジニアリングシステムを使用
                feature_set = await enhanced_feature_engineer.extract_comprehensive_features(symbol, data)

                if hasattr(feature_set, 'to_dataframe'):
                    features = feature_set.to_dataframe()
                else:
                    features = self._convert_featureset_to_dataframe(feature_set, data)

                # 品質チェック
                if features.empty or len(features.columns) < 5:
                    self.logger.warning("高度特徴量エンジニアリング結果不十分、基本特徴量を使用")
                    features = self._extract_basic_features(data)
                else:
                    self.logger.info(f"高度特徴量エンジニアリング完了: {len(features.columns)}特徴量")

            else:
                # 基本特徴量のみ
                features = self._extract_basic_features(data)
                self.logger.info(f"基本特徴量エンジニアリング完了: {len(features.columns)}特徴量")

            return features

        except Exception as e:
            self.logger.error(f"特徴量エンジニアリングエラー: {e}")
            # フォールバック
            return self._extract_basic_features(data)

    def _extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量抽出（改良版）"""
        features = pd.DataFrame(index=data.index)

        try:
            if not self.feature_config.enable_price_features:
                self.logger.info("価格特徴量がスキップされました")
            else:
                # 価格系特徴量
                features['returns'] = data['Close'].pct_change()
                features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                features['price_range'] = (data['High'] - data['Low']) / data['Close']
                features['body_size'] = abs(data['Close'] - data['Open']) / data['Close']
                features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
                features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']

            # 移動平均とその比率
            if self.feature_config.enable_technical_indicators:
                for window in self.feature_config.sma_periods:
                    sma = data['Close'].rolling(window).mean()
                    features[f'sma_{window}'] = sma
                    features[f'sma_ratio_{window}'] = data['Close'] / sma
                    features[f'sma_slope_{window}'] = sma.diff() / sma.shift(1)

                # 指数移動平均
                for window in self.feature_config.ema_periods:
                    ema = data['Close'].ewm(span=window).mean()
                    features[f'ema_{window}'] = ema
                    features[f'ema_ratio_{window}'] = data['Close'] / ema

            # ボラティリティ
            for window in self.feature_config.volatility_periods:
                features[f'volatility_{window}'] = features['returns'].rolling(window).std()
                features[f'volatility_ratio_{window}'] = (features[f'volatility_{window}'] /
                                                         features[f'volatility_{window}'].rolling(window*2).mean())

            # 出来高特徴量
            if self.feature_config.enable_volume_features and 'Volume' in data.columns:
                features['volume_ma_20'] = data['Volume'].rolling(20).mean()
                features['volume_ratio'] = data['Volume'] / features['volume_ma_20']
                features['volume_price_trend'] = (data['Volume'] * features['returns']).rolling(5).mean()

            # テクニカル指標
            if self.feature_config.enable_technical_indicators:
                features = self._add_technical_indicators(features, data)

            # 欠損値処理
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"基本特徴量抽出エラー: {e}")
            # 最小限の特徴量
            min_features = pd.DataFrame(index=data.index)
            min_features['returns'] = data['Close'].pct_change().fillna(0)
            min_features['sma_20'] = data['Close'].rolling(20).mean().fillna(method='ffill').fillna(data['Close'])
            return min_features

    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標追加"""
        try:
            # RSI
            if self.feature_config.enable_rsi:
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(self.feature_config.rsi_period).mean()
                avg_loss = loss.rolling(self.feature_config.rsi_period).mean()
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            if self.feature_config.enable_macd:
                ema_12 = data['Close'].ewm(span=12).mean()
                ema_26 = data['Close'].ewm(span=26).mean()
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']

            # ボリンジャーバンド
            if self.feature_config.enable_bollinger_bands:
                sma_bb = data['Close'].rolling(self.feature_config.bb_period).mean()
                std_bb = data['Close'].rolling(self.feature_config.bb_period).std()
                features['bb_upper'] = sma_bb + (std_bb * self.feature_config.bb_std)
                features['bb_lower'] = sma_bb - (std_bb * self.feature_config.bb_std)
                features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            return features

        except Exception as e:
            self.logger.warning(f"テクニカル指標追加エラー: {e}")
            return features

    def _convert_featureset_to_dataframe(self, feature_set, data: pd.DataFrame) -> pd.DataFrame:
        """FeatureSetをDataFrameに変換（改良版）"""
        try:
            if isinstance(feature_set, list):
                # 時系列FeatureSetリストの場合
                feature_rows = []
                timestamps = []

                for fs in feature_set:
                    row_features = {}
                    # 各カテゴリの特徴量を統合
                    for category in ['price_features', 'technical_features', 'volume_features',
                                   'momentum_features', 'volatility_features', 'pattern_features',
                                   'market_features', 'statistical_features']:
                        if hasattr(fs, category):
                            features_dict = getattr(fs, category)
                            if isinstance(features_dict, dict):
                                # プレフィックスを追加して名前衝突を回避
                                prefixed_features = {f"{category}_{k}": v for k, v in features_dict.items()}
                                row_features.update(prefixed_features)

                    if row_features:
                        feature_rows.append(row_features)
                        timestamps.append(getattr(fs, 'timestamp', data.index[len(timestamps)]))

                if feature_rows:
                    features_df = pd.DataFrame(feature_rows, index=timestamps[:len(feature_rows)])
                else:
                    features_df = self._extract_basic_features(data)

            else:
                # 単一FeatureSetの場合
                all_features = {}
                for category in ['price_features', 'technical_features', 'volume_features',
                               'momentum_features', 'volatility_features', 'pattern_features',
                               'market_features', 'statistical_features']:
                    if hasattr(feature_set, category):
                        features_dict = getattr(feature_set, category)
                        if isinstance(features_dict, dict):
                            prefixed_features = {f"{category}_{k}": v for k, v in features_dict.items()}
                            all_features.update(prefixed_features)

                if all_features:
                    timestamp = getattr(feature_set, 'timestamp', data.index[-1])
                    features_df = pd.DataFrame([all_features], index=[timestamp])
                else:
                    features_df = self._extract_basic_features(data)

            # 欠損値処理
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return features_df

        except Exception as e:
            self.logger.error(f"FeatureSet変換エラー: {e}")
            return self._extract_basic_features(data)

    def _postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """特徴量後処理"""
        try:
            # 無限値、異常値の処理
            features = features.replace([np.inf, -np.inf], np.nan)

            # 数値列の特定
            numeric_columns = features.select_dtypes(include=[np.number]).columns

            # 外れ値処理
            if self.config.outlier_detection:
                for col in numeric_columns:
                    if self.feature_config.outlier_method == "IQR":
                        Q1 = features[col].quantile(0.25)
                        Q3 = features[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - self.feature_config.outlier_threshold * IQR
                        upper_bound = Q3 + self.feature_config.outlier_threshold * IQR
                        features[col] = features[col].clip(lower_bound, upper_bound)
                    elif self.feature_config.outlier_method == "zscore":
                        z_scores = np.abs((features[col] - features[col].mean()) / features[col].std())
                        features[col] = features[col].where(z_scores <= 3, features[col].median())

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

    def _create_target_variables(self, data: pd.DataFrame) -> Dict[PredictionTask, pd.Series]:
        """ターゲット変数作成（改良版）"""
        targets = {}

        try:
            # 価格方向予測（翌日の価格変動方向）
            returns = data['Close'].pct_change().shift(-1)  # 翌日のリターン

            # 閾値を動的に調整（ボラティリティベース）
            volatility = returns.rolling(20).std().fillna(returns.std())
            threshold = volatility * 0.5  # ボラティリティの半分を閾値とする

            direction = pd.Series(index=data.index, dtype='int')
            direction[returns > threshold] = 1   # 上昇
            direction[returns < -threshold] = -1  # 下落
            direction[(returns >= -threshold) & (returns <= threshold)] = 0  # 横ばい

            targets[PredictionTask.PRICE_DIRECTION] = direction

            # 価格回帰予測（翌日の終値）
            targets[PredictionTask.PRICE_REGRESSION] = data['Close'].shift(-1)

            # ボラティリティ予測（翌日の変動率）
            if 'High' in data.columns and 'Low' in data.columns:
                high_low_range = (data['High'] - data['Low']) / data['Close']
                targets[PredictionTask.VOLATILITY] = high_low_range.shift(-1)

            # トレンド強度予測
            trend_strength = abs(returns) / volatility
            targets[PredictionTask.TREND_STRENGTH] = trend_strength.shift(-1)

            return targets

        except Exception as e:
            self.logger.error(f"ターゲット変数作成エラー: {e}")
            # 最小限のターゲット
            simple_direction = pd.Series(0, index=data.index)
            simple_direction[data['Close'].pct_change().shift(-1) > 0] = 1
            simple_direction[data['Close'].pct_change().shift(-1) < 0] = -1
            return {PredictionTask.PRICE_DIRECTION: simple_direction}

    def _align_data(self, features: pd.DataFrame, targets: Dict[PredictionTask, pd.Series]) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
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