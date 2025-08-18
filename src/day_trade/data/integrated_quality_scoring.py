#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Quality Scoring - 統合品質スコアリングシステム

複数ソース検証 + 異常値検出 + 品質評価の統合システム
Issue #795-3実装：データ品質スコアリング機能
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import sqlite3
import statistics

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

# 既存システムのインポート
try:
    from enhanced_data_quality_system import MultiSourceDataValidator, enhanced_data_validator
    ENHANCED_VALIDATOR_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATOR_AVAILABLE = False

try:
    from advanced_anomaly_detector import AdvancedAnomalyDetector, advanced_anomaly_detector, AnomalySeverity
    ANOMALY_DETECTOR_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTOR_AVAILABLE = False

try:
    from real_data_provider_v2 import real_data_provider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False

class QualityGrade(Enum):
    """品質グレード"""
    AAA = "AAA"      # 95-100点: 投資判断に最適
    AA = "AA"        # 90-94点: 高品質、推奨レベル
    A = "A"          # 85-89点: 良好、使用可能
    BBB = "BBB"      # 80-84点: 普通、注意して使用
    BB = "BB"        # 70-79点: 低品質、リスク有り
    B = "B"          # 60-69点: 要注意、補強必要
    C = "C"          # 50-59点: 低信頼性、使用非推奨
    D = "D"          # 0-49点: 危険、使用禁止

@dataclass
class QualityComponent:
    """品質コンポーネント"""
    name: str
    weight: float              # 重み（合計1.0）
    score: float              # 0-100のスコア
    status: str              # OK, WARNING, ERROR
    details: Dict[str, Any]
    last_updated: datetime

@dataclass
class QualityAssessment:
    """品質評価"""
    symbol: str
    timestamp: datetime
    overall_score: float       # 総合スコア 0-100
    grade: QualityGrade       # 品質グレード

    # コンポーネント別スコア
    data_completeness: QualityComponent
    price_consistency: QualityComponent
    source_reliability: QualityComponent
    anomaly_assessment: QualityComponent
    temporal_quality: QualityComponent

    # メタデータ
    confidence_interval: Tuple[float, float]  # 信頼区間
    recommendation: str                       # 使用推奨度
    risk_level: str                          # リスクレベル
    next_evaluation: datetime                # 次回評価推奨時刻

@dataclass
class QualityAlert:
    """品質アラート"""
    alert_id: str
    symbol: str
    alert_type: str           # QUALITY_DEGRADATION, ANOMALY_DETECTED, etc.
    severity: str            # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    current_score: float
    threshold: float
    triggered_at: datetime
    acknowledged: bool = False

class IntegratedQualityScoring:
    """統合品質スコアリングシステム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # コンポーネント重み設定
        self.component_weights = {
            'data_completeness': 0.20,      # データ完整性
            'price_consistency': 0.25,      # 価格整合性
            'source_reliability': 0.20,     # ソース信頼性
            'anomaly_assessment': 0.25,     # 異常値評価
            'temporal_quality': 0.10        # 時系列品質
        }

        # 品質閾値設定
        self.quality_thresholds = {
            'excellent': 95.0,     # AAA
            'very_good': 90.0,     # AA
            'good': 85.0,          # A
            'fair': 80.0,          # BBB
            'poor': 70.0,          # BB
            'bad': 60.0,           # B
            'critical': 50.0       # C
        }

        # アラート設定
        self.alert_thresholds = {
            'quality_degradation': 75.0,    # 品質劣化警告
            'anomaly_critical': 80.0,       # 異常値重要警告
            'source_failure': 60.0          # ソース障害警告
        }

        # データベース初期化
        self.data_dir = Path("quality_scoring_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "quality_scores.db"
        self._init_database()

        # サブシステム初期化
        self.validator = enhanced_data_validator if ENHANCED_VALIDATOR_AVAILABLE else None
        self.anomaly_detector = advanced_anomaly_detector if ANOMALY_DETECTOR_AVAILABLE else None

        # キャッシュ
        self.score_cache = {}
        self.cache_expiry = 300  # 5分間キャッシュ

        self.logger.info("Integrated quality scoring system initialized")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 品質スコアテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    overall_score REAL,
                    grade TEXT,
                    completeness_score REAL,
                    consistency_score REAL,
                    reliability_score REAL,
                    anomaly_score REAL,
                    temporal_score REAL,
                    recommendation TEXT,
                    risk_level TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 品質アラートテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    alert_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    current_score REAL,
                    threshold REAL,
                    triggered_at TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            """)

            # 品質履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    daily_avg_score REAL,
                    min_score REAL,
                    max_score REAL,
                    score_volatility REAL,
                    alert_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def assess_data_quality(self, symbol: str, data: pd.DataFrame = None) -> QualityAssessment:
        """包括的データ品質評価"""

        self.logger.info(f"Starting comprehensive quality assessment for {symbol}")

        # キャッシュチェック
        cache_key = f"{symbol}_assessment"
        if cache_key in self.score_cache:
            cache_time, result = self.score_cache[cache_key]
            if datetime.now().timestamp() - cache_time < self.cache_expiry:
                return result

        # データ取得（提供されていない場合）
        if data is None and REAL_DATA_PROVIDER_AVAILABLE:
            data = await real_data_provider.get_stock_data(symbol, "1mo")

        if data is None or data.empty:
            return self._create_error_assessment(symbol, "データ取得失敗")

        # 各コンポーネントの評価
        completeness = await self._assess_data_completeness(symbol, data)
        consistency = await self._assess_price_consistency(symbol, data)
        reliability = await self._assess_source_reliability(symbol, data)
        anomaly = await self._assess_anomaly_status(symbol, data)
        temporal = await self._assess_temporal_quality(symbol, data)

        # 総合スコア計算
        overall_score = self._calculate_overall_score([
            completeness, consistency, reliability, anomaly, temporal
        ])

        # グレード決定
        grade = self._determine_grade(overall_score)

        # 信頼区間計算
        confidence_interval = self._calculate_confidence_interval(
            overall_score, [completeness, consistency, reliability, anomaly, temporal]
        )

        # 推奨事項生成
        recommendation = self._generate_recommendation(overall_score, grade, [
            completeness, consistency, reliability, anomaly, temporal
        ])

        # リスクレベル決定
        risk_level = self._determine_risk_level(overall_score, anomaly)

        # 次回評価時刻
        next_evaluation = self._calculate_next_evaluation(overall_score)

        # 評価結果作成
        assessment = QualityAssessment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            grade=grade,
            data_completeness=completeness,
            price_consistency=consistency,
            source_reliability=reliability,
            anomaly_assessment=anomaly,
            temporal_quality=temporal,
            confidence_interval=confidence_interval,
            recommendation=recommendation,
            risk_level=risk_level,
            next_evaluation=next_evaluation
        )

        # 結果保存
        await self._save_assessment(assessment)

        # アラートチェック
        await self._check_and_create_alerts(assessment)

        # キャッシュ保存
        self.score_cache[cache_key] = (datetime.now().timestamp(), assessment)

        return assessment

    async def _assess_data_completeness(self, symbol: str, data: pd.DataFrame) -> QualityComponent:
        """データ完整性評価"""

        try:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            total_cells = len(data) * len(required_columns)

            # 欠損値計算
            missing_count = 0
            for col in required_columns:
                if col in data.columns:
                    missing_count += data[col].isnull().sum()
                else:
                    missing_count += len(data)  # 列自体が欠損

            completeness_rate = (total_cells - missing_count) / total_cells
            score = completeness_rate * 100

            # ステータス判定
            if score >= 95:
                status = "OK"
            elif score >= 85:
                status = "WARNING"
            else:
                status = "ERROR"

            return QualityComponent(
                name="データ完整性",
                weight=self.component_weights['data_completeness'],
                score=score,
                status=status,
                details={
                    'total_cells': total_cells,
                    'missing_cells': missing_count,
                    'completeness_rate': completeness_rate,
                    'data_points': len(data)
                },
                last_updated=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Data completeness assessment failed: {e}")
            return self._create_error_component("データ完整性", str(e))

    async def _assess_price_consistency(self, symbol: str, data: pd.DataFrame) -> QualityComponent:
        """価格整合性評価"""

        try:
            consistency_errors = 0
            total_checks = 0

            for idx in data.index:
                try:
                    row = data.loc[idx]
                    high, low, open_price, close = row['High'], row['Low'], row['Open'], row['Close']

                    total_checks += 5

                    # OHLC関係チェック
                    if high < low:
                        consistency_errors += 1
                    if high < open_price:
                        consistency_errors += 1
                    if high < close:
                        consistency_errors += 1
                    if low > open_price:
                        consistency_errors += 1
                    if low > close:
                        consistency_errors += 1

                except Exception:
                    consistency_errors += 5  # 全チェック失敗
                    total_checks += 5

            if total_checks > 0:
                consistency_rate = (total_checks - consistency_errors) / total_checks
                score = consistency_rate * 100
            else:
                score = 0

            # ステータス判定
            if score >= 98:
                status = "OK"
            elif score >= 95:
                status = "WARNING"
            else:
                status = "ERROR"

            return QualityComponent(
                name="価格整合性",
                weight=self.component_weights['price_consistency'],
                score=score,
                status=status,
                details={
                    'total_checks': total_checks,
                    'consistency_errors': consistency_errors,
                    'consistency_rate': consistency_rate
                },
                last_updated=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Price consistency assessment failed: {e}")
            return self._create_error_component("価格整合性", str(e))

    async def _assess_source_reliability(self, symbol: str, data: pd.DataFrame) -> QualityComponent:
        """ソース信頼性評価"""

        try:
            # 複数ソース検証（可能な場合）
            if self.validator:
                comparison = await self.validator.validate_multiple_sources(symbol, "1mo")

                # 乖離率に基づくスコア
                max_divergence = comparison.max_divergence
                if max_divergence <= 1.0:
                    score = 95.0
                elif max_divergence <= 2.0:
                    score = 90.0
                elif max_divergence <= 3.0:
                    score = 85.0
                elif max_divergence <= 5.0:
                    score = 75.0
                else:
                    score = max(50, 100 - max_divergence * 5)

                status = "OK" if score >= 85 else "WARNING" if score >= 70 else "ERROR"

                return QualityComponent(
                    name="ソース信頼性",
                    weight=self.component_weights['source_reliability'],
                    score=score,
                    status=status,
                    details={
                        'max_divergence': max_divergence,
                        'source_count': len(comparison.reliability_scores),
                        'consensus_price': comparison.consensus_price
                    },
                    last_updated=datetime.now()
                )
            else:
                # シングルソースの場合はデフォルトスコア
                return QualityComponent(
                    name="ソース信頼性",
                    weight=self.component_weights['source_reliability'],
                    score=75.0,
                    status="WARNING",
                    details={'note': 'Single source - no comparison available'},
                    last_updated=datetime.now()
                )

        except Exception as e:
            self.logger.error(f"Source reliability assessment failed: {e}")
            return self._create_error_component("ソース信頼性", str(e))

    async def _assess_anomaly_status(self, symbol: str, data: pd.DataFrame) -> QualityComponent:
        """異常値評価"""

        try:
            if self.anomaly_detector:
                anomalies = await self.anomaly_detector.detect_anomalies(symbol, data)

                # 重要度別の重み付けスコア計算
                penalty = 0
                for anomaly in anomalies:
                    if anomaly.severity == AnomalySeverity.CRITICAL:
                        penalty += 20
                    elif anomaly.severity == AnomalySeverity.HIGH:
                        penalty += 10
                    elif anomaly.severity == AnomalySeverity.MEDIUM:
                        penalty += 5
                    elif anomaly.severity == AnomalySeverity.LOW:
                        penalty += 2

                score = max(0, 100 - penalty)

                # ステータス判定
                critical_count = len([a for a in anomalies if a.severity == AnomalySeverity.CRITICAL])
                high_count = len([a for a in anomalies if a.severity == AnomalySeverity.HIGH])

                if critical_count > 0:
                    status = "ERROR"
                elif high_count > 2:
                    status = "ERROR"
                elif high_count > 0 or len(anomalies) > 5:
                    status = "WARNING"
                else:
                    status = "OK"

                return QualityComponent(
                    name="異常値評価",
                    weight=self.component_weights['anomaly_assessment'],
                    score=score,
                    status=status,
                    details={
                        'total_anomalies': len(anomalies),
                        'critical_count': critical_count,
                        'high_count': high_count,
                        'penalty_score': penalty
                    },
                    last_updated=datetime.now()
                )
            else:
                # 異常検出器が利用できない場合
                return QualityComponent(
                    name="異常値評価",
                    weight=self.component_weights['anomaly_assessment'],
                    score=80.0,
                    status="WARNING",
                    details={'note': 'Anomaly detector not available'},
                    last_updated=datetime.now()
                )

        except Exception as e:
            self.logger.error(f"Anomaly assessment failed: {e}")
            return self._create_error_component("異常値評価", str(e))

    async def _assess_temporal_quality(self, symbol: str, data: pd.DataFrame) -> QualityComponent:
        """時系列品質評価"""

        try:
            # データ間隔の一貫性
            dates = pd.to_datetime(data.index)
            if len(dates) < 2:
                score = 50.0
                status = "WARNING"
                details = {'note': 'Insufficient data for temporal analysis'}
            else:
                intervals = dates[1:] - dates[:-1]
                interval_days = [i.days for i in intervals]

                # 間隔の一貫性評価
                if len(interval_days) > 0:
                    mean_interval = statistics.mean(interval_days)
                    if len(interval_days) > 1:
                        std_interval = statistics.stdev(interval_days)
                        cv = std_interval / mean_interval if mean_interval > 0 else 1
                    else:
                        cv = 0

                    # スコア計算（変動係数ベース）
                    score = max(0, 100 - cv * 50)

                    # 未来日付チェック
                    today = datetime.now().date()
                    future_count = len([d for d in dates if d.date() > today])
                    if future_count > 0:
                        score = max(0, score - future_count * 20)

                    # ステータス判定
                    if score >= 90 and future_count == 0:
                        status = "OK"
                    elif score >= 70:
                        status = "WARNING"
                    else:
                        status = "ERROR"

                    details = {
                        'mean_interval_days': mean_interval,
                        'interval_std': std_interval if len(interval_days) > 1 else 0,
                        'coefficient_variation': cv,
                        'future_dates_count': future_count,
                        'data_span_days': (dates.max() - dates.min()).days
                    }
                else:
                    score = 50.0
                    status = "WARNING"
                    details = {'note': 'Single data point'}

            return QualityComponent(
                name="時系列品質",
                weight=self.component_weights['temporal_quality'],
                score=score,
                status=status,
                details=details,
                last_updated=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Temporal quality assessment failed: {e}")
            return self._create_error_component("時系列品質", str(e))

    def _calculate_overall_score(self, components: List[QualityComponent]) -> float:
        """総合スコア計算"""

        weighted_sum = 0
        total_weight = 0

        for component in components:
            weighted_sum += component.score * component.weight
            total_weight += component.weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _determine_grade(self, score: float) -> QualityGrade:
        """品質グレード決定"""

        if score >= 95:
            return QualityGrade.AAA
        elif score >= 90:
            return QualityGrade.AA
        elif score >= 85:
            return QualityGrade.A
        elif score >= 80:
            return QualityGrade.BBB
        elif score >= 70:
            return QualityGrade.BB
        elif score >= 60:
            return QualityGrade.B
        elif score >= 50:
            return QualityGrade.C
        else:
            return QualityGrade.D

    def _calculate_confidence_interval(self, overall_score: float,
                                     components: List[QualityComponent]) -> Tuple[float, float]:
        """信頼区間計算"""

        scores = [c.score for c in components]
        if len(scores) > 1:
            std_dev = statistics.stdev(scores)
            margin = std_dev * 1.96 / len(scores) ** 0.5  # 95%信頼区間

            lower = max(0, overall_score - margin)
            upper = min(100, overall_score + margin)

            return (lower, upper)
        else:
            return (overall_score * 0.9, overall_score * 1.1)

    def _generate_recommendation(self, score: float, grade: QualityGrade,
                               components: List[QualityComponent]) -> str:
        """推奨事項生成"""

        if grade in [QualityGrade.AAA, QualityGrade.AA]:
            return "投資判断に最適な品質です。安心してご利用ください。"
        elif grade == QualityGrade.A:
            return "良好な品質です。推奨レベルで使用可能です。"
        elif grade == QualityGrade.BBB:
            return "普通の品質です。注意して使用してください。"
        elif grade in [QualityGrade.BB, QualityGrade.B]:
            return "品質が低い状態です。他のデータソースとの照合を推奨します。"
        elif grade == QualityGrade.C:
            return "低信頼性データです。投資判断には使用を控えてください。"
        else:
            return "危険な品質レベルです。このデータの使用は禁止します。"

    def _determine_risk_level(self, score: float, anomaly_component: QualityComponent) -> str:
        """リスクレベル決定"""

        if score >= 90 and anomaly_component.status == "OK":
            return "低リスク"
        elif score >= 80 and anomaly_component.status != "ERROR":
            return "中リスク"
        elif score >= 70:
            return "高リスク"
        else:
            return "極高リスク"

    def _calculate_next_evaluation(self, score: float) -> datetime:
        """次回評価時刻計算"""

        if score >= 90:
            hours = 6   # 高品質なら6時間後
        elif score >= 80:
            hours = 4   # 中品質なら4時間後
        elif score >= 70:
            hours = 2   # 低品質なら2時間後
        else:
            hours = 1   # 極低品質なら1時間後

        return datetime.now() + timedelta(hours=hours)

    def _create_error_component(self, name: str, error_msg: str) -> QualityComponent:
        """エラー時のコンポーネント作成"""

        return QualityComponent(
            name=name,
            weight=self.component_weights.get(name.lower().replace('ー', '_'), 0.2),
            score=0.0,
            status="ERROR",
            details={'error': error_msg},
            last_updated=datetime.now()
        )

    def _create_error_assessment(self, symbol: str, error_msg: str) -> QualityAssessment:
        """エラー時の評価作成"""

        error_component = QualityComponent(
            name="エラー",
            weight=1.0,
            score=0.0,
            status="ERROR",
            details={'error': error_msg},
            last_updated=datetime.now()
        )

        return QualityAssessment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=0.0,
            grade=QualityGrade.D,
            data_completeness=error_component,
            price_consistency=error_component,
            source_reliability=error_component,
            anomaly_assessment=error_component,
            temporal_quality=error_component,
            confidence_interval=(0.0, 0.0),
            recommendation="データ取得エラーのため評価不可",
            risk_level="評価不可",
            next_evaluation=datetime.now() + timedelta(hours=1)
        )

    async def _save_assessment(self, assessment: QualityAssessment):
        """評価結果の保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO quality_scores
                    (symbol, timestamp, overall_score, grade, completeness_score,
                     consistency_score, reliability_score, anomaly_score, temporal_score,
                     recommendation, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    assessment.symbol,
                    assessment.timestamp.isoformat(),
                    assessment.overall_score,
                    assessment.grade.value,
                    assessment.data_completeness.score,
                    assessment.price_consistency.score,
                    assessment.source_reliability.score,
                    assessment.anomaly_assessment.score,
                    assessment.temporal_quality.score,
                    assessment.recommendation,
                    assessment.risk_level
                ))

        except Exception as e:
            self.logger.error(f"Failed to save assessment: {e}")

    async def _check_and_create_alerts(self, assessment: QualityAssessment):
        """アラートチェックと生成"""

        alerts = []

        # 品質劣化アラート
        if assessment.overall_score < self.alert_thresholds['quality_degradation']:
            alert = QualityAlert(
                alert_id=f"QUALITY_DEGRADE_{assessment.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=assessment.symbol,
                alert_type="QUALITY_DEGRADATION",
                severity="HIGH" if assessment.overall_score < 60 else "MEDIUM",
                message=f"品質スコアが{assessment.overall_score:.1f}に低下",
                current_score=assessment.overall_score,
                threshold=self.alert_thresholds['quality_degradation'],
                triggered_at=datetime.now()
            )
            alerts.append(alert)

        # 異常値重要アラート
        if assessment.anomaly_assessment.status == "ERROR":
            alert = QualityAlert(
                alert_id=f"ANOMALY_CRITICAL_{assessment.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=assessment.symbol,
                alert_type="ANOMALY_DETECTED",
                severity="CRITICAL",
                message="重要な異常値が検出されました",
                current_score=assessment.anomaly_assessment.score,
                threshold=self.alert_thresholds['anomaly_critical'],
                triggered_at=datetime.now()
            )
            alerts.append(alert)

        # アラート保存
        for alert in alerts:
            await self._save_alert(alert)

    async def _save_alert(self, alert: QualityAlert):
        """アラート保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO quality_alerts
                    (alert_id, symbol, alert_type, severity, message,
                     current_score, threshold, triggered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.symbol,
                    alert.alert_type,
                    alert.severity,
                    alert.message,
                    alert.current_score,
                    alert.threshold,
                    alert.triggered_at.isoformat()
                ))

        except Exception as e:
            self.logger.error(f"Failed to save alert: {e}")

    async def get_quality_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """品質サマリー取得"""

        summaries = {}

        for symbol in symbols:
            try:
                assessment = await self.assess_data_quality(symbol)

                summaries[symbol] = {
                    'overall_score': assessment.overall_score,
                    'grade': assessment.grade.value,
                    'risk_level': assessment.risk_level,
                    'recommendation': assessment.recommendation,
                    'confidence_interval': assessment.confidence_interval,
                    'last_assessed': assessment.timestamp.isoformat()
                }

            except Exception as e:
                summaries[symbol] = {
                    'error': str(e),
                    'overall_score': 0,
                    'grade': 'D',
                    'risk_level': '評価不可'
                }

        return summaries

# グローバルインスタンス
integrated_quality_scorer = IntegratedQualityScoring()

# テスト関数
async def test_integrated_quality_scoring():
    """統合品質スコアリングシステムのテスト"""

    print("=== 統合品質スコアリングシステム テスト ===")

    scorer = IntegratedQualityScoring()

    # テスト銘柄
    test_symbols = ["7203", "8306", "4751"]

    print(f"\n[ {len(test_symbols)}銘柄の品質評価 ]")

    for symbol in test_symbols:
        print(f"\n--- {symbol} 品質評価 ---")

        try:
            assessment = await scorer.assess_data_quality(symbol)

            print(f"総合スコア: {assessment.overall_score:.1f}/100")
            print(f"品質グレード: {assessment.grade.value}")
            print(f"リスクレベル: {assessment.risk_level}")
            print(f"信頼区間: {assessment.confidence_interval[0]:.1f} - {assessment.confidence_interval[1]:.1f}")

            print(f"\nコンポーネント別スコア:")
            components = [
                assessment.data_completeness,
                assessment.price_consistency,
                assessment.source_reliability,
                assessment.anomaly_assessment,
                assessment.temporal_quality
            ]

            for comp in components:
                print(f"  {comp.name}: {comp.score:.1f} ({comp.status})")

            print(f"\n推奨事項: {assessment.recommendation}")
            print(f"次回評価: {assessment.next_evaluation.strftime('%H:%M')}")

        except Exception as e:
            print(f"❌ 評価エラー: {e}")

    # 品質サマリー
    print(f"\n[ 品質サマリー ]")
    summary = await scorer.get_quality_summary(test_symbols)

    for symbol, data in summary.items():
        if 'error' not in data:
            print(f"{symbol}: {data['grade']} ({data['overall_score']:.1f}) - {data['risk_level']}")
        else:
            print(f"{symbol}: エラー - {data['error']}")

    print(f"\n=== 統合品質スコアリングシステム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_integrated_quality_scoring())