#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Anomaly Detector - 高度異常値検出システム

統計的手法＋機械学習による多層異常検知
Issue #795-2実装：異常値検出アルゴリズム強化
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
import statistics
from collections import deque

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

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class AnomalySeverity(Enum):
    """異常重要度"""
    CRITICAL = "致命的"     # システム停止レベル
    HIGH = "高"           # 即座対応必要
    MEDIUM = "中"         # 監視強化
    LOW = "低"           # 記録のみ

class AnomalyCategory(Enum):
    """異常カテゴリ"""
    PRICE_ANOMALY = "価格異常"
    VOLUME_ANOMALY = "出来高異常"
    TEMPORAL_ANOMALY = "時系列異常"
    STATISTICAL_OUTLIER = "統計的外れ値"
    PATTERN_ANOMALY = "パターン異常"
    DATA_QUALITY_ISSUE = "データ品質問題"

@dataclass
class AnomalyDetection:
    """異常検出結果"""
    detection_id: str
    symbol: str
    timestamp: datetime
    category: AnomalyCategory
    severity: AnomalySeverity
    anomaly_score: float        # 0-100の異常度スコア
    description: str
    affected_data: Dict[str, Any]
    detection_method: str
    confidence: float          # 検出信頼度
    recommended_action: str

    # メタデータ
    detector_version: str = "v1.0"
    false_positive_probability: float = 0.0

@dataclass
class DetectionStats:
    """検出統計"""
    total_detections: int
    by_severity: Dict[str, int]
    by_category: Dict[str, int]
    false_positive_rate: float
    detection_accuracy: float
    last_updated: datetime

class AdvancedAnomalyDetector:
    """高度異常値検出システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 検出アルゴリズム設定
        self.detection_config = {
            'statistical_threshold': 3.0,    # Z-scoreの閾値
            'isolation_contamination': 0.1,  # Isolation Forestの汚染率
            'price_change_threshold': 15.0,  # 異常な価格変動%
            'volume_spike_threshold': 5.0,   # 出来高スパイク倍率
            'temporal_gap_threshold': 2,     # 時系列ギャップ日数
            'pattern_deviation_threshold': 0.8,  # パターン偏差閾値
        }

        # 機械学習モデル
        self.isolation_forest = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

        # 検出履歴（学習データとして使用）
        self.detection_history: deque = deque(maxlen=1000)
        self.feedback_data: List[Tuple[Dict, bool]] = []  # (features, is_anomaly)

        # 統計情報
        self.detection_stats = DetectionStats(
            total_detections=0,
            by_severity={s.value: 0 for s in AnomalySeverity},
            by_category={c.value: 0 for c in AnomalyCategory},
            false_positive_rate=0.0,
            detection_accuracy=0.0,
            last_updated=datetime.now()
        )

        # モデル訓練
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()

        self.logger.info("Advanced anomaly detector initialized")

    def _initialize_ml_models(self):
        """機械学習モデルの初期化"""

        try:
            self.isolation_forest = IsolationForest(
                contamination=self.detection_config['isolation_contamination'],
                random_state=42,
                n_estimators=100
            )
            self.logger.info("Isolation Forest model initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")

    async def detect_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyDetection]:
        """包括的異常検出"""

        self.logger.debug(f"Running anomaly detection for {symbol}")

        if data.empty or len(data) < 10:
            return []

        all_anomalies = []

        # 1. 統計的異常検出
        statistical_anomalies = await self._detect_statistical_anomalies(symbol, data)
        all_anomalies.extend(statistical_anomalies)

        # 2. 価格異常検出
        price_anomalies = await self._detect_price_anomalies(symbol, data)
        all_anomalies.extend(price_anomalies)

        # 3. 出来高異常検出
        volume_anomalies = await self._detect_volume_anomalies(symbol, data)
        all_anomalies.extend(volume_anomalies)

        # 4. 時系列異常検出
        temporal_anomalies = await self._detect_temporal_anomalies(symbol, data)
        all_anomalies.extend(temporal_anomalies)

        # 5. 機械学習による検出
        if SKLEARN_AVAILABLE and len(data) >= 30:
            ml_anomalies = await self._detect_ml_anomalies(symbol, data)
            all_anomalies.extend(ml_anomalies)

        # 6. パターン異常検出
        pattern_anomalies = await self._detect_pattern_anomalies(symbol, data)
        all_anomalies.extend(pattern_anomalies)

        # 重複除去と優先度付け
        unique_anomalies = self._deduplicate_and_prioritize(all_anomalies)

        # 統計更新
        self._update_detection_stats(unique_anomalies)

        return unique_anomalies

    async def _detect_statistical_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyDetection]:
        """統計的異常検出"""

        anomalies = []

        if not SCIPY_AVAILABLE or len(data) < 10:
            return anomalies

        try:
            # 日次リターンのZ-score分析
            returns = data['Close'].pct_change().dropna()

            if len(returns) >= 5:
                z_scores = np.abs(stats.zscore(returns))
                threshold = self.detection_config['statistical_threshold']

                outlier_indices = np.where(z_scores > threshold)[0]

                for idx in outlier_indices:
                    date = returns.index[idx]
                    return_value = returns.iloc[idx]
                    z_score = z_scores[idx]

                    # 重要度判定
                    if z_score > 5.0:
                        severity = AnomalySeverity.CRITICAL
                    elif z_score > 4.0:
                        severity = AnomalySeverity.HIGH
                    elif z_score > 3.5:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW

                    anomaly = AnomalyDetection(
                        detection_id=f"STAT_{symbol}_{date.strftime('%Y%m%d')}",
                        symbol=symbol,
                        timestamp=date,
                        category=AnomalyCategory.STATISTICAL_OUTLIER,
                        severity=severity,
                        anomaly_score=min(100, z_score * 20),
                        description=f"統計的外れ値: {return_value*100:.2f}%変動 (Z-score: {z_score:.2f})",
                        affected_data={
                            'return_percent': return_value * 100,
                            'z_score': z_score,
                            'price': float(data.loc[date, 'Close'])
                        },
                        detection_method="Z-score Analysis",
                        confidence=min(95, z_score * 15),
                        recommended_action="データ検証・他ソース確認"
                    )

                    anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Statistical anomaly detection failed: {e}")

        return anomalies

    async def _detect_price_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyDetection]:
        """価格異常検出"""

        anomalies = []

        try:
            # 極端な価格変動
            if len(data) >= 2:
                price_changes = data['Close'].pct_change().dropna()
                threshold = self.detection_config['price_change_threshold'] / 100

                extreme_changes = price_changes[abs(price_changes) > threshold]

                for date, change in extreme_changes.items():
                    severity = AnomalySeverity.CRITICAL if abs(change) > 0.3 else AnomalySeverity.HIGH

                    anomaly = AnomalyDetection(
                        detection_id=f"PRICE_{symbol}_{date.strftime('%Y%m%d')}",
                        symbol=symbol,
                        timestamp=date,
                        category=AnomalyCategory.PRICE_ANOMALY,
                        severity=severity,
                        anomaly_score=min(100, abs(change) * 200),
                        description=f"極端価格変動: {change*100:+.2f}%",
                        affected_data={
                            'change_percent': change * 100,
                            'current_price': float(data.loc[date, 'Close']),
                            'previous_price': float(data['Close'].shift(1).loc[date])
                        },
                        detection_method="Price Change Analysis",
                        confidence=90.0,
                        recommended_action="緊急確認・取引一時停止検討"
                    )

                    anomalies.append(anomaly)

            # OHLC整合性チェック
            for date in data.index:
                try:
                    row = data.loc[date]
                    high, low, open_price, close = row['High'], row['Low'], row['Open'], row['Close']

                    # 整合性違反チェック
                    violations = []
                    if high < low:
                        violations.append("高値 < 安値")
                    if high < open_price:
                        violations.append("高値 < 始値")
                    if high < close:
                        violations.append("高値 < 終値")
                    if low > open_price:
                        violations.append("安値 > 始値")
                    if low > close:
                        violations.append("安値 > 終値")

                    if violations:
                        anomaly = AnomalyDetection(
                            detection_id=f"OHLC_{symbol}_{date.strftime('%Y%m%d')}",
                            symbol=symbol,
                            timestamp=date,
                            category=AnomalyCategory.DATA_QUALITY_ISSUE,
                            severity=AnomalySeverity.HIGH,
                            anomaly_score=80.0,
                            description=f"OHLC整合性エラー: {', '.join(violations)}",
                            affected_data={
                                'open': float(open_price),
                                'high': float(high),
                                'low': float(low),
                                'close': float(close),
                                'violations': violations
                            },
                            detection_method="OHLC Consistency Check",
                            confidence=95.0,
                            recommended_action="データソース確認・修正"
                        )

                        anomalies.append(anomaly)

                except Exception:
                    continue

        except Exception as e:
            self.logger.error(f"Price anomaly detection failed: {e}")

        return anomalies

    async def _detect_volume_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyDetection]:
        """出来高異常検出"""

        anomalies = []

        if 'Volume' not in data.columns:
            return anomalies

        try:
            volumes = data['Volume'].dropna()

            if len(volumes) >= 10:
                # 移動平均ベースの異常検出
                window = min(20, len(volumes))
                avg_volume = volumes.rolling(window).mean()

                # スパイク検出
                threshold = self.detection_config['volume_spike_threshold']

                for date in volumes.index:
                    current_volume = volumes[date]
                    avg_vol = avg_volume[date]

                    if pd.notna(avg_vol) and avg_vol > 0:
                        volume_ratio = current_volume / avg_vol

                        if volume_ratio > threshold:
                            if volume_ratio > 10:
                                severity = AnomalySeverity.HIGH
                            elif volume_ratio > 7:
                                severity = AnomalySeverity.MEDIUM
                            else:
                                severity = AnomalySeverity.LOW

                            anomaly = AnomalyDetection(
                                detection_id=f"VOL_{symbol}_{date.strftime('%Y%m%d')}",
                                symbol=symbol,
                                timestamp=date,
                                category=AnomalyCategory.VOLUME_ANOMALY,
                                severity=severity,
                                anomaly_score=min(100, volume_ratio * 10),
                                description=f"出来高急増: 平均の{volume_ratio:.1f}倍",
                                affected_data={
                                    'current_volume': int(current_volume),
                                    'average_volume': int(avg_vol),
                                    'volume_ratio': volume_ratio
                                },
                                detection_method="Volume Spike Detection",
                                confidence=80.0,
                                recommended_action="材料・ニュース確認"
                            )

                            anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Volume anomaly detection failed: {e}")

        return anomalies

    async def _detect_temporal_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyDetection]:
        """時系列異常検出"""

        anomalies = []

        try:
            # 日付間隔の異常
            if len(data) >= 2:
                dates = pd.to_datetime(data.index)
                intervals = dates[1:] - dates[:-1]

                # 異常に長い間隔
                threshold_days = self.detection_config['temporal_gap_threshold']

                for i, interval in enumerate(intervals):
                    gap_days = interval.days

                    if gap_days > threshold_days:
                        date = dates[i+1]

                        if gap_days > 7:
                            severity = AnomalySeverity.HIGH
                        elif gap_days > 5:
                            severity = AnomalySeverity.MEDIUM
                        else:
                            severity = AnomalySeverity.LOW

                        anomaly = AnomalyDetection(
                            detection_id=f"TEMP_{symbol}_{date.strftime('%Y%m%d')}",
                            symbol=symbol,
                            timestamp=date,
                            category=AnomalyCategory.TEMPORAL_ANOMALY,
                            severity=severity,
                            anomaly_score=min(100, gap_days * 10),
                            description=f"時系列ギャップ: {gap_days}日間のデータ欠損",
                            affected_data={
                                'gap_days': gap_days,
                                'previous_date': dates[i].isoformat(),
                                'current_date': date.isoformat()
                            },
                            detection_method="Temporal Gap Detection",
                            confidence=95.0,
                            recommended_action="データ補完・ソース確認"
                        )

                        anomalies.append(anomaly)

            # 未来日付チェック
            today = datetime.now().date()
            future_dates = [d for d in data.index if d.date() > today]

            if future_dates:
                anomaly = AnomalyDetection(
                    detection_id=f"FUTURE_{symbol}_{datetime.now().strftime('%Y%m%d')}",
                    symbol=symbol,
                    timestamp=datetime.now(),
                    category=AnomalyCategory.TEMPORAL_ANOMALY,
                    severity=AnomalySeverity.CRITICAL,
                    anomaly_score=100.0,
                    description=f"未来日付データ: {len(future_dates)}件",
                    affected_data={
                        'future_dates': [d.isoformat() for d in future_dates[:5]],
                        'count': len(future_dates)
                    },
                    detection_method="Future Date Detection",
                    confidence=100.0,
                    recommended_action="データソース緊急確認"
                )

                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Temporal anomaly detection failed: {e}")

        return anomalies

    async def _detect_ml_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyDetection]:
        """機械学習による異常検出"""

        anomalies = []

        if not SKLEARN_AVAILABLE or len(data) < 30:
            return anomalies

        try:
            # 特徴量作成
            features = self._extract_features(data)

            if len(features) == 0:
                return anomalies

            # 正規化
            features_scaled = self.scaler.fit_transform(features)

            # Isolation Forest予測
            if self.isolation_forest is None:
                self.isolation_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )

            outlier_predictions = self.isolation_forest.fit_predict(features_scaled)
            anomaly_scores = self.isolation_forest.decision_function(features_scaled)

            # 異常点の抽出
            for i, (is_outlier, score) in enumerate(zip(outlier_predictions, anomaly_scores)):
                if is_outlier == -1:  # 異常点
                    date = data.index[i]

                    # スコアを0-100に正規化
                    normalized_score = max(0, min(100, (0.5 - score) * 100))

                    if normalized_score > 80:
                        severity = AnomalySeverity.HIGH
                    elif normalized_score > 60:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW

                    anomaly = AnomalyDetection(
                        detection_id=f"ML_{symbol}_{date.strftime('%Y%m%d')}",
                        symbol=symbol,
                        timestamp=date,
                        category=AnomalyCategory.PATTERN_ANOMALY,
                        severity=severity,
                        anomaly_score=normalized_score,
                        description=f"機械学習検出異常: パターン偏差",
                        affected_data={
                            'ml_score': float(score),
                            'feature_vector': features[i].tolist()
                        },
                        detection_method="Isolation Forest",
                        confidence=70.0,
                        recommended_action="パターン分析・詳細確認"
                    )

                    anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"ML anomaly detection failed: {e}")

        return anomalies

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """機械学習用特徴量抽出"""

        try:
            features = []

            # 価格系特徴量
            data['returns'] = data['Close'].pct_change()
            data['volatility'] = data['returns'].rolling(5).std()
            data['price_range'] = (data['High'] - data['Low']) / data['Close']

            # 出来高特徴量
            if 'Volume' in data.columns:
                data['volume_ma'] = data['Volume'].rolling(10).mean()
                data['volume_ratio'] = data['Volume'] / data['volume_ma']
            else:
                data['volume_ratio'] = 1.0

            # 技術指標
            data['sma_5'] = data['Close'].rolling(5).mean()
            data['sma_20'] = data['Close'].rolling(20).mean()
            data['sma_ratio'] = data['sma_5'] / data['sma_20']

            # 特徴量配列作成
            feature_columns = ['returns', 'volatility', 'price_range', 'volume_ratio', 'sma_ratio']

            for col in feature_columns:
                if col in data.columns:
                    features.append(data[col].fillna(0).values)

            if features:
                return np.column_stack(features)
            else:
                return np.array([])

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.array([])

    async def _detect_pattern_anomalies(self, symbol: str, data: pd.DataFrame) -> List[AnomalyDetection]:
        """パターン異常検出"""

        anomalies = []

        try:
            # 価格パターンの異常
            if len(data) >= 5:
                # 連続上昇・下落パターン
                returns = data['Close'].pct_change().dropna()

                consecutive_up = 0
                consecutive_down = 0

                for i, ret in enumerate(returns):
                    if ret > 0:
                        consecutive_up += 1
                        consecutive_down = 0
                    elif ret < 0:
                        consecutive_down += 1
                        consecutive_up = 0
                    else:
                        consecutive_up = 0
                        consecutive_down = 0

                    # 異常な連続パターン検出
                    if consecutive_up >= 7 or consecutive_down >= 7:
                        date = returns.index[i]
                        direction = "上昇" if consecutive_up >= 7 else "下落"
                        count = consecutive_up if consecutive_up >= 7 else consecutive_down

                        anomaly = AnomalyDetection(
                            detection_id=f"PATTERN_{symbol}_{date.strftime('%Y%m%d')}",
                            symbol=symbol,
                            timestamp=date,
                            category=AnomalyCategory.PATTERN_ANOMALY,
                            severity=AnomalySeverity.MEDIUM,
                            anomaly_score=min(100, count * 10),
                            description=f"異常パターン: {count}日連続{direction}",
                            affected_data={
                                'pattern': f"{count}日連続{direction}",
                                'direction': direction,
                                'count': count
                            },
                            detection_method="Pattern Analysis",
                            confidence=75.0,
                            recommended_action="トレンド転換点監視"
                        )

                        anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Pattern anomaly detection failed: {e}")

        return anomalies

    def _deduplicate_and_prioritize(self, anomalies: List[AnomalyDetection]) -> List[AnomalyDetection]:
        """重複除去と優先度付け"""

        # 日付とカテゴリでグループ化
        grouped = {}
        for anomaly in anomalies:
            key = (anomaly.timestamp.date(), anomaly.category)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(anomaly)

        # 各グループから最高スコアのものを選択
        unique_anomalies = []
        for group in grouped.values():
            best_anomaly = max(group, key=lambda x: x.anomaly_score)
            unique_anomalies.append(best_anomaly)

        # 重要度順でソート
        severity_order = {
            AnomalySeverity.CRITICAL: 4,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 1
        }

        unique_anomalies.sort(
            key=lambda x: (severity_order[x.severity], x.anomaly_score),
            reverse=True
        )

        return unique_anomalies

    def _update_detection_stats(self, anomalies: List[AnomalyDetection]):
        """検出統計の更新"""

        self.detection_stats.total_detections += len(anomalies)

        for anomaly in anomalies:
            self.detection_stats.by_severity[anomaly.severity.value] += 1
            self.detection_stats.by_category[anomaly.category.value] += 1

        self.detection_stats.last_updated = datetime.now()

        # 検出履歴に追加
        self.detection_history.extend(anomalies)

    def get_detection_summary(self) -> Dict[str, Any]:
        """検出サマリー取得"""

        return {
            'stats': {
                'total_detections': self.detection_stats.total_detections,
                'by_severity': self.detection_stats.by_severity,
                'by_category': self.detection_stats.by_category,
                'last_updated': self.detection_stats.last_updated.isoformat()
            },
            'recent_anomalies': len([
                a for a in self.detection_history
                if a.timestamp > datetime.now() - timedelta(hours=24)
            ]),
            'critical_alerts': len([
                a for a in self.detection_history
                if a.severity == AnomalySeverity.CRITICAL and
                a.timestamp > datetime.now() - timedelta(hours=1)
            ])
        }

# グローバルインスタンス
advanced_anomaly_detector = AdvancedAnomalyDetector()

# テスト関数
async def test_advanced_anomaly_detector():
    """高度異常値検出システムのテスト"""

    print("=== 高度異常値検出システム テスト ===")

    detector = AdvancedAnomalyDetector()

    # テストデータ作成（異常値を含む）
    dates = pd.date_range(start='2025-07-01', end='2025-08-14', freq='D')
    np.random.seed(42)

    # 正常データ
    prices = [1000]
    for _ in range(len(dates)-1):
        change = np.random.normal(0, 0.02)  # 2%標準偏差
        prices.append(prices[-1] * (1 + change))

    # 異常値を意図的に挿入
    prices[10] *= 1.25  # 25%急騰
    prices[20] *= 0.8   # 20%急落
    prices[30] *= 1.15  # 15%上昇

    # DataFrameに変換
    test_data = pd.DataFrame({
        'Open': [p * 0.99 for p in prices],
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(10000, 100000, len(dates))
    }, index=dates)

    # 出来高スパイクも挿入
    test_data.loc[dates[15], 'Volume'] *= 8  # 8倍スパイク

    # OHLC整合性エラーも挿入
    test_data.loc[dates[25], 'High'] = test_data.loc[dates[25], 'Low'] * 0.9  # High < Low

    print(f"\n[ テストデータ概要 ]")
    print(f"期間: {len(test_data)}日")
    print(f"価格レンジ: ¥{test_data['Close'].min():.0f} - ¥{test_data['Close'].max():.0f}")
    print(f"意図的異常: 価格急変3件、出来高スパイク1件、OHLC整合性エラー1件")

    # 異常検出実行
    print(f"\n[ 異常検出実行 ]")
    anomalies = await detector.detect_anomalies("TEST", test_data)

    print(f"検出された異常: {len(anomalies)}件")

    # 重要度別表示
    for severity in [AnomalySeverity.CRITICAL, AnomalySeverity.HIGH, AnomalySeverity.MEDIUM, AnomalySeverity.LOW]:
        count = len([a for a in anomalies if a.severity == severity])
        if count > 0:
            print(f"  {severity.value}: {count}件")

    # 詳細表示（上位5件）
    print(f"\n[ 検出詳細（上位5件）]")
    for i, anomaly in enumerate(anomalies[:5], 1):
        print(f"{i}. {anomaly.severity.value} - {anomaly.category.value}")
        print(f"   日付: {anomaly.timestamp.strftime('%Y-%m-%d')}")
        print(f"   異常度: {anomaly.anomaly_score:.1f}")
        print(f"   内容: {anomaly.description}")
        print(f"   検出方法: {anomaly.detection_method}")
        print(f"   推奨対応: {anomaly.recommended_action}")
        print()

    # 統計情報
    summary = detector.get_detection_summary()
    print(f"[ 検出統計 ]")
    print(f"総検出数: {summary['stats']['total_detections']}")
    print(f"重要度別: {summary['stats']['by_severity']}")
    print(f"カテゴリ別: {summary['stats']['by_category']}")

    print(f"\n=== 高度異常値検出システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_advanced_anomaly_detector())