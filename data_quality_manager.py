#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Manager - データ品質管理システム

Issue #810対応：データ品質管理の実装
データの完整性、一貫性、鮮度を包括的に管理
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
from pathlib import Path
import time

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

class DataQualityLevel(Enum):
    """データ品質レベル"""
    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"           # 85-94%
    ACCEPTABLE = "acceptable"  # 70-84%
    POOR = "poor"           # 50-69%
    CRITICAL = "critical"   # 0-49%

class QualityCheckType(Enum):
    """品質チェックタイプ"""
    COMPLETENESS = "completeness"     # 完整性
    CONSISTENCY = "consistency"       # 一貫性
    FRESHNESS = "freshness"          # 鮮度
    ACCURACY = "accuracy"            # 正確性
    VALIDITY = "validity"            # 有効性
    UNIQUENESS = "uniqueness"        # 一意性

@dataclass
class QualityCheckResult:
    """品質チェック結果"""
    check_type: QualityCheckType
    score: float  # 0-100
    issues: List[str]
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class DataQualityReport:
    """データ品質レポート"""
    symbol: str
    timestamp: datetime
    overall_score: float
    quality_level: DataQualityLevel
    check_results: List[QualityCheckResult]
    recommendations: List[str]
    data_points: int
    missing_data_ratio: float

class DataQualityChecker:
    """データ品質チェッカー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_completeness(self, data: pd.DataFrame) -> QualityCheckResult:
        """完整性チェック"""

        issues = []
        details = {}

        if data is None or len(data) == 0:
            return QualityCheckResult(
                check_type=QualityCheckType.COMPLETENESS,
                score=0.0,
                issues=["データが存在しません"],
                details={"total_records": 0},
                timestamp=datetime.now()
            )

        # 必須カラムの存在確認
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            issues.append(f"必須カラム欠損: {missing_columns}")

        # 欠損値チェック
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 1.0

        details = {
            "total_records": len(data),
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_ratio": missing_ratio,
            "missing_columns": missing_columns
        }

        # スコア計算
        column_score = (1 - len(missing_columns) / len(required_columns)) * 100
        data_score = (1 - missing_ratio) * 100
        overall_score = (column_score + data_score) / 2

        if missing_ratio > 0.1:
            issues.append(f"欠損値が多い: {missing_ratio:.1%}")

        return QualityCheckResult(
            check_type=QualityCheckType.COMPLETENESS,
            score=overall_score,
            issues=issues,
            details=details,
            timestamp=datetime.now()
        )

    def check_consistency(self, data: pd.DataFrame) -> QualityCheckResult:
        """一貫性チェック"""

        issues = []
        details = {}

        if data is None or len(data) == 0:
            return QualityCheckResult(
                check_type=QualityCheckType.CONSISTENCY,
                score=0.0,
                issues=["データが存在しません"],
                details={},
                timestamp=datetime.now()
            )

        consistency_violations = 0
        total_checks = 0

        # OHLC価格の論理的一貫性チェック
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High >= max(Open, Close)
            high_violations = (data['High'] < data[['Open', 'Close']].max(axis=1)).sum()

            # Low <= min(Open, Close)
            low_violations = (data['Low'] > data[['Open', 'Close']].min(axis=1)).sum()

            # High >= Low
            hl_violations = (data['High'] < data['Low']).sum()

            consistency_violations += high_violations + low_violations + hl_violations
            total_checks += len(data) * 3

            if high_violations > 0:
                issues.append(f"High価格の不整合: {high_violations}件")
            if low_violations > 0:
                issues.append(f"Low価格の不整合: {low_violations}件")
            if hl_violations > 0:
                issues.append(f"High-Low価格の不整合: {hl_violations}件")

        # 価格の妥当性（負の値チェック）
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    consistency_violations += negative_prices
                    total_checks += len(data)
                    issues.append(f"{col}に負の価格: {negative_prices}件")

        # 出来高の妥当性
        if 'Volume' in data.columns:
            negative_volume = (data['Volume'] < 0).sum()
            if negative_volume > 0:
                consistency_violations += negative_volume
                total_checks += len(data)
                issues.append(f"負の出来高: {negative_volume}件")

        # スコア計算
        consistency_ratio = 1 - (consistency_violations / max(total_checks, 1))
        score = consistency_ratio * 100

        details = {
            "total_checks": total_checks,
            "violations": consistency_violations,
            "consistency_ratio": consistency_ratio
        }

        return QualityCheckResult(
            check_type=QualityCheckType.CONSISTENCY,
            score=score,
            issues=issues,
            details=details,
            timestamp=datetime.now()
        )

    def check_freshness(self, data: pd.DataFrame) -> QualityCheckResult:
        """鮮度チェック"""

        issues = []
        details = {}

        if data is None or len(data) == 0:
            return QualityCheckResult(
                check_type=QualityCheckType.FRESHNESS,
                score=0.0,
                issues=["データが存在しません"],
                details={},
                timestamp=datetime.now()
            )

        try:
            # 最新データの日付確認
            if hasattr(data.index, 'to_pydatetime'):
                latest_date = pd.to_datetime(data.index[-1])
            else:
                latest_date = pd.to_datetime(data.index[-1])

            now = datetime.now()

            # タイムゾーン情報を統一
            if latest_date.tzinfo is not None:
                latest_date = latest_date.replace(tzinfo=None)

            time_diff = now - latest_date
            days_old = time_diff.days + time_diff.seconds / 86400

            # 鮮度スコア計算（平日考慮）
            # 1日以内: 100%, 2日以内: 80%, 3日以内: 60%, それ以上: 減算
            if days_old <= 1:
                score = 100
            elif days_old <= 2:
                score = 80
            elif days_old <= 3:
                score = 60
            elif days_old <= 7:
                score = max(0, 60 - (days_old - 3) * 10)
            else:
                score = max(0, 20 - (days_old - 7) * 2)

            details = {
                "latest_date": latest_date.isoformat(),
                "current_time": now.isoformat(),
                "days_old": days_old,
                "hours_old": time_diff.total_seconds() / 3600
            }

            if days_old > 1:
                issues.append(f"データが古い: {days_old:.1f}日前")
            if days_old > 7:
                issues.append("データが1週間以上古い")

        except Exception as e:
            score = 50  # エラー時はデフォルトスコア
            issues.append(f"鮮度チェックエラー: {str(e)}")
            details = {"error": str(e)}

        return QualityCheckResult(
            check_type=QualityCheckType.FRESHNESS,
            score=score,
            issues=issues,
            details=details,
            timestamp=datetime.now()
        )

    def check_accuracy(self, data: pd.DataFrame) -> QualityCheckResult:
        """正確性チェック"""

        issues = []
        details = {}

        if data is None or len(data) == 0:
            return QualityCheckResult(
                check_type=QualityCheckType.ACCURACY,
                score=0.0,
                issues=["データが存在しません"],
                details={},
                timestamp=datetime.now()
            )

        accuracy_issues = 0
        total_checks = 0

        # 価格変動の妥当性チェック（極端な値の検出）
        if 'Close' in data.columns and len(data) > 1:
            price_changes = data['Close'].pct_change().dropna()

            # 1日で50%以上の変動は異常とみなす
            extreme_changes = (abs(price_changes) > 0.5).sum()
            accuracy_issues += extreme_changes
            total_checks += len(price_changes)

            if extreme_changes > 0:
                issues.append(f"極端な価格変動: {extreme_changes}件")

            details.update({
                "max_price_change": price_changes.abs().max(),
                "extreme_changes": extreme_changes,
                "avg_volatility": price_changes.std()
            })

        # 出来高の異常値チェック
        if 'Volume' in data.columns and len(data) > 1:
            volume_median = data['Volume'].median()
            volume_q75 = data['Volume'].quantile(0.75)
            volume_q25 = data['Volume'].quantile(0.25)
            iqr = volume_q75 - volume_q25

            # IQRの3倍を超える値は異常とみなす
            upper_bound = volume_q75 + 3 * iqr
            lower_bound = max(0, volume_q25 - 3 * iqr)

            volume_outliers = ((data['Volume'] > upper_bound) |
                             (data['Volume'] < lower_bound)).sum()

            if volume_outliers > 0:
                accuracy_issues += volume_outliers
                issues.append(f"出来高異常値: {volume_outliers}件")

            total_checks += len(data)

            details.update({
                "volume_outliers": volume_outliers,
                "volume_median": volume_median,
                "volume_iqr": iqr
            })

        # 正確性スコア計算
        if total_checks > 0:
            accuracy_ratio = 1 - (accuracy_issues / total_checks)
            score = accuracy_ratio * 100
        else:
            score = 100  # チェック項目がない場合は満点

        details["total_accuracy_checks"] = total_checks
        details["accuracy_issues"] = accuracy_issues

        return QualityCheckResult(
            check_type=QualityCheckType.ACCURACY,
            score=score,
            issues=issues,
            details=details,
            timestamp=datetime.now()
        )

    def check_validity(self, data: pd.DataFrame) -> QualityCheckResult:
        """有効性チェック"""

        issues = []
        details = {}

        if data is None or len(data) == 0:
            return QualityCheckResult(
                check_type=QualityCheckType.VALIDITY,
                score=0.0,
                issues=["データが存在しません"],
                details={},
                timestamp=datetime.now()
            )

        validity_issues = 0
        total_checks = 0

        # データ型の妥当性
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                non_numeric = pd.to_numeric(data[col], errors='coerce').isnull().sum()
                if non_numeric > 0:
                    validity_issues += non_numeric
                    issues.append(f"{col}に非数値データ: {non_numeric}件")
                total_checks += len(data)

        # インデックスの有効性（日付）
        try:
            if hasattr(data.index, 'to_pydatetime'):
                invalid_dates = 0  # pandas DatetimeIndexは基本的に有効
            else:
                invalid_dates = pd.to_datetime(data.index, errors='coerce').isnull().sum()

            if invalid_dates > 0:
                validity_issues += invalid_dates
                issues.append(f"無効な日付: {invalid_dates}件")

            total_checks += len(data)

        except Exception as e:
            validity_issues += len(data)
            issues.append(f"日付インデックスエラー: {str(e)}")
            total_checks += len(data)

        # 有効性スコア計算
        if total_checks > 0:
            validity_ratio = 1 - (validity_issues / total_checks)
            score = validity_ratio * 100
        else:
            score = 100

        details = {
            "total_validity_checks": total_checks,
            "validity_issues": validity_issues,
            "validity_ratio": validity_ratio if total_checks > 0 else 1.0
        }

        return QualityCheckResult(
            check_type=QualityCheckType.VALIDITY,
            score=score,
            issues=issues,
            details=details,
            timestamp=datetime.now()
        )

class DataQualityManager:
    """データ品質管理システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checker = DataQualityChecker()

        # データベース設定
        self.db_path = Path("data_quality/quality_reports.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

        # 品質基準
        self.quality_thresholds = {
            DataQualityLevel.EXCELLENT: 95,
            DataQualityLevel.GOOD: 85,
            DataQualityLevel.ACCEPTABLE: 70,
            DataQualityLevel.POOR: 50,
            DataQualityLevel.CRITICAL: 0
        }

        self.logger.info("Data quality manager initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quality_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        quality_level TEXT NOT NULL,
                        data_points INTEGER NOT NULL,
                        missing_data_ratio REAL NOT NULL,
                        completeness_score REAL,
                        consistency_score REAL,
                        freshness_score REAL,
                        accuracy_score REAL,
                        validity_score REAL,
                        issues_summary TEXT,
                        recommendations TEXT
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def evaluate_data_quality(self, symbol: str) -> Dict[str, Any]:
        """データ品質評価"""

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "3mo")  # 3ヶ月分

            if data is None or len(data) == 0:
                return {
                    "symbol": symbol,
                    "overall_score": 0,
                    "quality_level": DataQualityLevel.CRITICAL.value,
                    "error": "データが取得できませんでした"
                }

            # 各品質チェック実行
            check_results = []

            check_results.append(self.checker.check_completeness(data))
            check_results.append(self.checker.check_consistency(data))
            check_results.append(self.checker.check_freshness(data))
            check_results.append(self.checker.check_accuracy(data))
            check_results.append(self.checker.check_validity(data))

            # 総合スコア計算（重み付き平均）
            weights = {
                QualityCheckType.COMPLETENESS: 0.25,
                QualityCheckType.CONSISTENCY: 0.25,
                QualityCheckType.FRESHNESS: 0.20,
                QualityCheckType.ACCURACY: 0.20,
                QualityCheckType.VALIDITY: 0.10
            }

            overall_score = sum(
                result.score * weights[result.check_type]
                for result in check_results
            )

            # 品質レベル判定
            quality_level = self._determine_quality_level(overall_score)

            # 推奨事項生成
            recommendations = self._generate_recommendations(check_results, quality_level)

            # 欠損データ比率計算
            missing_ratio = data.isnull().sum().sum() / data.size if data.size > 0 else 1.0

            # レポート作成
            report = DataQualityReport(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_score=overall_score,
                quality_level=quality_level,
                check_results=check_results,
                recommendations=recommendations,
                data_points=len(data),
                missing_data_ratio=missing_ratio
            )

            # データベース保存
            await self._save_report(report)

            # 結果辞書作成
            result_dict = {
                "symbol": symbol,
                "timestamp": report.timestamp.isoformat(),
                "overall_score": overall_score,
                "quality_level": quality_level.value,
                "data_points": len(data),
                "missing_data_ratio": missing_ratio,
                "check_scores": {
                    result.check_type.value: result.score
                    for result in check_results
                },
                "issues": [
                    issue for result in check_results for issue in result.issues
                ],
                "recommendations": recommendations
            }

            return result_dict

        except Exception as e:
            self.logger.error(f"データ品質評価エラー {symbol}: {e}")
            return {
                "symbol": symbol,
                "overall_score": 0,
                "quality_level": DataQualityLevel.CRITICAL.value,
                "error": str(e)
            }

    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """品質レベル判定"""

        if score >= 95:
            return DataQualityLevel.EXCELLENT
        elif score >= 85:
            return DataQualityLevel.GOOD
        elif score >= 70:
            return DataQualityLevel.ACCEPTABLE
        elif score >= 50:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL

    def _generate_recommendations(self, check_results: List[QualityCheckResult],
                                quality_level: DataQualityLevel) -> List[str]:
        """推奨事項生成"""

        recommendations = []

        # 品質レベル別の一般的な推奨事項
        if quality_level == DataQualityLevel.CRITICAL:
            recommendations.append("データ品質が極めて低いため、使用を避けることを推奨")
            recommendations.append("データソースの見直しが必要")
        elif quality_level == DataQualityLevel.POOR:
            recommendations.append("データ品質に重大な問題があります")
            recommendations.append("取引判断には注意が必要")
        elif quality_level == DataQualityLevel.ACCEPTABLE:
            recommendations.append("データ品質は基準を満たしていますが改善余地があります")
        elif quality_level == DataQualityLevel.GOOD:
            recommendations.append("データ品質は良好です")
        else:
            recommendations.append("データ品質は非常に優秀です")

        # 個別チェック結果に基づく推奨事項
        for result in check_results:
            if result.score < 70:
                if result.check_type == QualityCheckType.COMPLETENESS:
                    recommendations.append("データ欠損の対処が必要（補間・除外検討）")
                elif result.check_type == QualityCheckType.CONSISTENCY:
                    recommendations.append("データ整合性の確認と修正が必要")
                elif result.check_type == QualityCheckType.FRESHNESS:
                    recommendations.append("より新しいデータの取得を推奨")
                elif result.check_type == QualityCheckType.ACCURACY:
                    recommendations.append("異常値の除去・調整を検討")
                elif result.check_type == QualityCheckType.VALIDITY:
                    recommendations.append("データ形式の確認と修正が必要")

        return recommendations

    async def _save_report(self, report: DataQualityReport):
        """レポート保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # スコア辞書作成
                score_dict = {result.check_type.value: result.score for result in report.check_results}

                cursor.execute('''
                    INSERT INTO quality_reports
                    (symbol, timestamp, overall_score, quality_level, data_points,
                     missing_data_ratio, completeness_score, consistency_score,
                     freshness_score, accuracy_score, validity_score,
                     issues_summary, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report.symbol,
                    report.timestamp.isoformat(),
                    report.overall_score,
                    report.quality_level.value,
                    report.data_points,
                    report.missing_data_ratio,
                    score_dict.get('completeness', 0),
                    score_dict.get('consistency', 0),
                    score_dict.get('freshness', 0),
                    score_dict.get('accuracy', 0),
                    score_dict.get('validity', 0),
                    json.dumps([issue for result in report.check_results for issue in result.issues]),
                    json.dumps(report.recommendations)
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"レポート保存エラー: {e}")

# グローバルインスタンス
data_quality_manager = DataQualityManager()

# テスト実行
async def run_data_quality_test():
    """データ品質テスト実行"""

    print("=== 📊 データ品質管理システムテスト ===")

    test_symbols = ["7203", "8306", "4751"]

    for symbol in test_symbols:
        print(f"\n--- {symbol} データ品質評価 ---")

        result = await data_quality_manager.evaluate_data_quality(symbol)

        print(f"総合スコア: {result.get('overall_score', 0):.1f}/100")
        print(f"品質レベル: {result.get('quality_level', 'unknown')}")
        print(f"データ点数: {result.get('data_points', 0)}")
        print(f"欠損比率: {result.get('missing_data_ratio', 0):.1%}")

        if 'check_scores' in result:
            print("個別スコア:")
            for check_type, score in result['check_scores'].items():
                print(f"  {check_type}: {score:.1f}/100")

        if result.get('issues'):
            print("検出された問題:")
            for issue in result['issues'][:3]:  # 上位3件
                print(f"  • {issue}")

        if result.get('recommendations'):
            print("推奨事項:")
            for rec in result['recommendations'][:2]:  # 上位2件
                print(f"  • {rec}")

    print(f"\n✅ データ品質管理システム稼働確認完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_data_quality_test())