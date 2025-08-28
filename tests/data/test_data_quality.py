#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Management Tests
データ品質管理テスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# データ品質レベル定義
class QualityLevel(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    CRITICAL = "CRITICAL"

@dataclass
class QualityMetric:
    """品質メトリック"""
    name: str
    value: float
    threshold: float
    status: str
    description: str

@dataclass
class DataQualityReport:
    """データ品質レポート"""
    dataset_name: str
    timestamp: datetime
    overall_score: float
    quality_level: QualityLevel
    metrics: List[QualityMetric]
    issues: List[str]
    recommendations: List[str]

class MockDataQualityChecker:
    """データ品質チェッカーモック"""
    
    def __init__(self):
        self.thresholds = {
            'completeness': 0.95,      # 95%以上
            'accuracy': 0.98,          # 98%以上
            'consistency': 0.95,       # 95%以上
            'timeliness': 0.90,        # 90%以上
            'validity': 0.99,          # 99%以上
            'uniqueness': 0.99         # 99%以上
        }
    
    def check_completeness(self, data: pd.DataFrame) -> QualityMetric:
        """完全性チェック"""
        if data.empty:
            return QualityMetric(
                name='completeness',
                value=0.0,
                threshold=self.thresholds['completeness'],
                status='FAILED',
                description='データが空です'
            )
        
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        status = 'PASSED' if completeness >= self.thresholds['completeness'] else 'FAILED'
        
        return QualityMetric(
            name='completeness',
            value=completeness,
            threshold=self.thresholds['completeness'],
            status=status,
            description=f'欠損率: {missing_cells/total_cells*100:.2f}%'
        )
    
    def check_accuracy(self, data: pd.DataFrame) -> QualityMetric:
        """正確性チェック"""
        if data.empty:
            return QualityMetric(
                name='accuracy',
                value=0.0,
                threshold=self.thresholds['accuracy'],
                status='FAILED',
                description='データが空です'
            )
        
        # 価格データの妥当性チェック
        accuracy_issues = 0
        total_checks = 0
        
        # 数値列のチェック
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['open', 'high', 'low', 'close']:
                # 価格データの範囲チェック
                invalid_prices = ((data[col] < 0) | (data[col] > 1000000)).sum()
                accuracy_issues += invalid_prices
                total_checks += len(data)
            elif col == 'volume':
                # 出来高の負の値チェック
                invalid_volume = (data[col] < 0).sum()
                accuracy_issues += invalid_volume
                total_checks += len(data)
        
        # OHLC関係チェック
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # high >= max(open, close), low <= min(open, close)
            invalid_ohlc = (
                (data['high'] < data[['open', 'close']].max(axis=1)) |
                (data['low'] > data[['open', 'close']].min(axis=1))
            ).sum()
            accuracy_issues += invalid_ohlc
            total_checks += len(data)
        
        accuracy = 1 - (accuracy_issues / total_checks) if total_checks > 0 else 1.0
        status = 'PASSED' if accuracy >= self.thresholds['accuracy'] else 'FAILED'
        
        return QualityMetric(
            name='accuracy',
            value=accuracy,
            threshold=self.thresholds['accuracy'],
            status=status,
            description=f'不正値: {accuracy_issues}/{total_checks}'
        )
    
    def check_consistency(self, data: pd.DataFrame) -> QualityMetric:
        """一貫性チェック"""
        if data.empty:
            return QualityMetric(
                name='consistency',
                value=0.0,
                threshold=self.thresholds['consistency'],
                status='FAILED',
                description='データが空です'
            )
        
        consistency_issues = 0
        total_checks = 0
        
        # 日付の一貫性チェック
        if 'date' in data.columns:
            # 日付の順序チェック
            if len(data) > 1:
                date_inconsistencies = (data['date'].diff().dt.days < 0).sum()
                consistency_issues += date_inconsistencies
                total_checks += len(data) - 1
        
        # データ型の一貫性チェック
        for col in data.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                # 数値型であることをチェック
                non_numeric = pd.to_numeric(data[col], errors='coerce').isnull().sum() - data[col].isnull().sum()
                consistency_issues += non_numeric
                total_checks += len(data)
        
        consistency = 1 - (consistency_issues / total_checks) if total_checks > 0 else 1.0
        status = 'PASSED' if consistency >= self.thresholds['consistency'] else 'FAILED'
        
        return QualityMetric(
            name='consistency',
            value=consistency,
            threshold=self.thresholds['consistency'],
            status=status,
            description=f'不一致: {consistency_issues}/{total_checks}'
        )
    
    def check_timeliness(self, data: pd.DataFrame) -> QualityMetric:
        """適時性チェック"""
        if data.empty or 'date' not in data.columns:
            return QualityMetric(
                name='timeliness',
                value=0.0,
                threshold=self.thresholds['timeliness'],
                status='FAILED',
                description='日付データがありません'
            )
        
        # 最新データの鮮度チェック
        latest_date = pd.to_datetime(data['date']).max()
        current_date = datetime.now().date()
        
        if pd.isnull(latest_date):
            timeliness = 0.0
        else:
            days_old = (current_date - latest_date.date()).days
            # 1日以内は100%, 5日で50%, 10日で0%
            timeliness = max(0, 1 - (days_old / 10))
        
        status = 'PASSED' if timeliness >= self.thresholds['timeliness'] else 'FAILED'
        
        return QualityMetric(
            name='timeliness',
            value=timeliness,
            threshold=self.thresholds['timeliness'],
            status=status,
            description=f'最新データ: {days_old}日前' if not pd.isnull(latest_date) else '日付不明'
        )
    
    def check_validity(self, data: pd.DataFrame) -> QualityMetric:
        """妥当性チェック"""
        if data.empty:
            return QualityMetric(
                name='validity',
                value=0.0,
                threshold=self.thresholds['validity'],
                status='FAILED',
                description='データが空です'
            )
        
        validity_issues = 0
        total_checks = 0
        
        # 各カラムの妥当性チェック
        for col in data.columns:
            if col == 'volume':
                # 出来高の妥当性（正の整数）
                invalid_volume = ((data[col] < 0) | (data[col] % 1 != 0)).sum()
                validity_issues += invalid_volume
                total_checks += len(data)
                
            elif col in ['open', 'high', 'low', 'close']:
                # 価格の妥当性（正の数値、現実的な範囲）
                invalid_price = ((data[col] <= 0) | (data[col] > 100000)).sum()
                validity_issues += invalid_price
                total_checks += len(data)
        
        validity = 1 - (validity_issues / total_checks) if total_checks > 0 else 1.0
        status = 'PASSED' if validity >= self.thresholds['validity'] else 'FAILED'
        
        return QualityMetric(
            name='validity',
            value=validity,
            threshold=self.thresholds['validity'],
            status=status,
            description=f'不正値: {validity_issues}/{total_checks}'
        )
    
    def check_uniqueness(self, data: pd.DataFrame) -> QualityMetric:
        """一意性チェック"""
        if data.empty:
            return QualityMetric(
                name='uniqueness',
                value=0.0,
                threshold=self.thresholds['uniqueness'],
                status='FAILED',
                description='データが空です'
            )
        
        # 主キー（date+symbol）の重複チェック
        key_columns = []
        if 'date' in data.columns:
            key_columns.append('date')
        if 'symbol' in data.columns:
            key_columns.append('symbol')
        
        if key_columns:
            total_rows = len(data)
            unique_rows = len(data[key_columns].drop_duplicates())
            uniqueness = unique_rows / total_rows if total_rows > 0 else 1.0
        else:
            # キーがない場合は全行の重複チェック
            total_rows = len(data)
            unique_rows = len(data.drop_duplicates())
            uniqueness = unique_rows / total_rows if total_rows > 0 else 1.0
        
        status = 'PASSED' if uniqueness >= self.thresholds['uniqueness'] else 'FAILED'
        
        return QualityMetric(
            name='uniqueness',
            value=uniqueness,
            threshold=self.thresholds['uniqueness'],
            status=status,
            description=f'重複行: {total_rows - unique_rows}/{total_rows}'
        )
    
    def generate_comprehensive_report(self, data: pd.DataFrame, 
                                    dataset_name: str = "Unknown") -> DataQualityReport:
        """包括的品質レポート生成"""
        # 各品質チェック実行
        metrics = [
            self.check_completeness(data),
            self.check_accuracy(data),
            self.check_consistency(data),
            self.check_timeliness(data),
            self.check_validity(data),
            self.check_uniqueness(data)
        ]
        
        # 総合スコア計算（各メトリックの重み付き平均）
        weights = {
            'completeness': 0.2,
            'accuracy': 0.25,
            'consistency': 0.15,
            'timeliness': 0.1,
            'validity': 0.2,
            'uniqueness': 0.1
        }
        
        overall_score = sum(
            metric.value * weights.get(metric.name, 0.1) 
            for metric in metrics
        )
        
        # 品質レベル判定
        if overall_score >= 0.95:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.85:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 0.70:
            quality_level = QualityLevel.FAIR
        elif overall_score >= 0.50:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.CRITICAL
        
        # 問題点と推奨事項生成
        issues = []
        recommendations = []
        
        for metric in metrics:
            if metric.status == 'FAILED':
                issues.append(f'{metric.name}: {metric.description}')
                
                if metric.name == 'completeness':
                    recommendations.append('欠損データの補完または除去を検討してください')
                elif metric.name == 'accuracy':
                    recommendations.append('データ検証ルールの強化を推奨します')
                elif metric.name == 'consistency':
                    recommendations.append('データ形式の統一化が必要です')
                elif metric.name == 'timeliness':
                    recommendations.append('データ更新頻度の改善を検討してください')
                elif metric.name == 'validity':
                    recommendations.append('データ入力時のバリデーション強化を推奨します')
                elif metric.name == 'uniqueness':
                    recommendations.append('重複データの除去が必要です')
        
        return DataQualityReport(
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            overall_score=overall_score,
            quality_level=quality_level,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )


class MockDataCleaner:
    """データクリーニングモック"""
    
    def __init__(self):
        self.cleaning_stats = {
            'rows_processed': 0,
            'rows_removed': 0,
            'values_fixed': 0
        }
    
    def clean_missing_values(self, data: pd.DataFrame, 
                           strategy: str = 'drop') -> pd.DataFrame:
        """欠損値処理"""
        self.cleaning_stats['rows_processed'] = len(data)
        
        if strategy == 'drop':
            cleaned_data = data.dropna()
            self.cleaning_stats['rows_removed'] = len(data) - len(cleaned_data)
        elif strategy == 'fill_forward':
            cleaned_data = data.fillna(method='ffill')
            self.cleaning_stats['values_fixed'] = data.isnull().sum().sum()
        elif strategy == 'fill_mean':
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            cleaned_data = data.copy()
            for col in numeric_columns:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            self.cleaning_stats['values_fixed'] = data.isnull().sum().sum()
        else:
            cleaned_data = data.copy()
        
        return cleaned_data
    
    def clean_outliers(self, data: pd.DataFrame, 
                      method: str = 'iqr') -> pd.DataFrame:
        """外れ値処理"""
        cleaned_data = data.copy()
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound))
                cleaned_data = cleaned_data[~outliers]
                self.cleaning_stats['rows_removed'] += outliers.sum()
                
            elif method == 'z_score':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > 3
                cleaned_data = cleaned_data[~outliers]
                self.cleaning_stats['rows_removed'] += outliers.sum()
        
        return cleaned_data
    
    def clean_duplicates(self, data: pd.DataFrame, 
                        subset: Optional[List[str]] = None) -> pd.DataFrame:
        """重複データ除去"""
        initial_count = len(data)
        
        if subset:
            cleaned_data = data.drop_duplicates(subset=subset)
        else:
            cleaned_data = data.drop_duplicates()
        
        self.cleaning_stats['rows_removed'] += initial_count - len(cleaned_data)
        
        return cleaned_data
    
    def standardize_formats(self, data: pd.DataFrame) -> pd.DataFrame:
        """フォーマット標準化"""
        cleaned_data = data.copy()
        
        # 日付フォーマット標準化
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
            self.cleaning_stats['values_fixed'] += 1
        
        # 数値型変換
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        
        return cleaned_data
    
    def get_cleaning_stats(self) -> Dict[str, int]:
        """クリーニング統計取得"""
        return self.cleaning_stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.cleaning_stats = {
            'rows_processed': 0,
            'rows_removed': 0,
            'values_fixed': 0
        }


def create_test_data(quality: str = 'good') -> pd.DataFrame:
    """テストデータ作成"""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    base_price = 1000
    
    # 価格データ生成
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'date': dates,
        'symbol': ['TEST'] * 100,
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 100)
    })
    
    # 品質レベルに応じてデータを劣化
    if quality == 'poor':
        # 欠損値追加
        data.loc[np.random.choice(100, 10, replace=False), 'close'] = np.nan
        
        # 外れ値追加
        data.loc[95, 'high'] = 999999
        data.loc[96, 'low'] = -100
        
        # 重複追加
        data = pd.concat([data, data.iloc[:5]], ignore_index=True)
        
        # OHLC関係の不整合
        data.loc[90, 'high'] = data.loc[90, 'low'] - 100
        
    elif quality == 'critical':
        # より多くの問題を追加
        data.loc[np.random.choice(100, 30, replace=False), 'close'] = np.nan
        data.loc[np.random.choice(100, 20, replace=False), 'volume'] = -1000
        data = pd.concat([data, data.iloc[:20]], ignore_index=True)
        
        # 日付の問題
        data.loc[50:60, 'date'] = pd.NaT
    
    return data


class TestMockDataQualityChecker:
    """データ品質チェッカーテストクラス"""
    
    @pytest.fixture
    def quality_checker(self):
        """品質チェッカーフィクスチャ"""
        return MockDataQualityChecker()
    
    @pytest.fixture
    def good_data(self):
        """良好データフィクスチャ"""
        return create_test_data('good')
    
    @pytest.fixture
    def poor_data(self):
        """不良データフィクスチャ"""
        return create_test_data('poor')
    
    def test_completeness_check(self, quality_checker, good_data, poor_data):
        """完全性チェックテスト"""
        # 良好データ
        good_metric = quality_checker.check_completeness(good_data)
        assert good_metric.name == 'completeness'
        assert good_metric.value > 0.9
        assert good_metric.status == 'PASSED'
        
        # 不良データ
        poor_metric = quality_checker.check_completeness(poor_data)
        assert poor_metric.value < good_metric.value
    
    def test_accuracy_check(self, quality_checker, good_data, poor_data):
        """正確性チェックテスト"""
        # 良好データ
        good_metric = quality_checker.check_accuracy(good_data)
        assert good_metric.name == 'accuracy'
        assert good_metric.value > 0.9
        
        # 不良データ（外れ値や不正値含む）
        poor_metric = quality_checker.check_accuracy(poor_data)
        assert poor_metric.value < good_metric.value
    
    def test_consistency_check(self, quality_checker, good_data):
        """一貫性チェックテスト"""
        metric = quality_checker.check_consistency(good_data)
        
        assert metric.name == 'consistency'
        assert metric.value >= 0
        assert metric.status in ['PASSED', 'FAILED']
    
    def test_timeliness_check(self, quality_checker, good_data):
        """適時性チェックテスト"""
        metric = quality_checker.check_timeliness(good_data)
        
        assert metric.name == 'timeliness'
        assert 0 <= metric.value <= 1
        assert metric.status in ['PASSED', 'FAILED']
    
    def test_validity_check(self, quality_checker, good_data, poor_data):
        """妥当性チェックテスト"""
        # 良好データ
        good_metric = quality_checker.check_validity(good_data)
        assert good_metric.value > 0.9
        
        # 不良データ
        poor_metric = quality_checker.check_validity(poor_data)
        assert poor_metric.value < good_metric.value
    
    def test_uniqueness_check(self, quality_checker, good_data, poor_data):
        """一意性チェックテスト"""
        # 良好データ（重複なし）
        good_metric = quality_checker.check_uniqueness(good_data)
        assert good_metric.value == 1.0
        
        # 不良データ（重複あり）
        poor_metric = quality_checker.check_uniqueness(poor_data)
        assert poor_metric.value < 1.0
    
    def test_comprehensive_report(self, quality_checker, good_data, poor_data):
        """包括的レポートテスト"""
        # 良好データのレポート
        good_report = quality_checker.generate_comprehensive_report(good_data, "Good Data")
        
        assert good_report.dataset_name == "Good Data"
        assert good_report.overall_score > 0.8
        assert good_report.quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]
        assert len(good_report.metrics) == 6
        
        # 不良データのレポート
        poor_report = quality_checker.generate_comprehensive_report(poor_data, "Poor Data")
        
        assert poor_report.overall_score < good_report.overall_score
        assert poor_report.quality_level in [QualityLevel.POOR, QualityLevel.CRITICAL]
        assert len(poor_report.issues) > 0
        assert len(poor_report.recommendations) > 0
    
    def test_empty_data_handling(self, quality_checker):
        """空データ処理テスト"""
        empty_data = pd.DataFrame()
        
        report = quality_checker.generate_comprehensive_report(empty_data, "Empty")
        
        assert report.overall_score == 0.0
        assert report.quality_level == QualityLevel.CRITICAL
        assert all(metric.status == 'FAILED' for metric in report.metrics)


class TestMockDataCleaner:
    """データクリーナーテストクラス"""
    
    @pytest.fixture
    def data_cleaner(self):
        """データクリーナーフィクスチャ"""
        return MockDataCleaner()
    
    @pytest.fixture
    def dirty_data(self):
        """汚れたデータフィクスチャ"""
        data = create_test_data('good')
        
        # 意図的に問題を追加
        data.loc[10:15, 'close'] = np.nan  # 欠損値
        data.loc[90, 'high'] = 99999      # 外れ値
        data = pd.concat([data, data.iloc[:3]], ignore_index=True)  # 重複
        
        return data
    
    def test_missing_values_cleaning(self, data_cleaner, dirty_data):
        """欠損値クリーニングテスト"""
        # ドロップ戦略
        cleaned_drop = data_cleaner.clean_missing_values(dirty_data, 'drop')
        assert len(cleaned_drop) < len(dirty_data)
        assert cleaned_drop.isnull().sum().sum() == 0
        
        data_cleaner.reset_stats()
        
        # 前方埋め戦略
        cleaned_fill = data_cleaner.clean_missing_values(dirty_data, 'fill_forward')
        assert len(cleaned_fill) == len(dirty_data)
        assert cleaned_fill.isnull().sum().sum() < dirty_data.isnull().sum().sum()
    
    def test_outliers_cleaning(self, data_cleaner, dirty_data):
        """外れ値クリーニングテスト"""
        initial_count = len(dirty_data)
        
        # IQR方式
        cleaned_iqr = data_cleaner.clean_outliers(dirty_data, 'iqr')
        assert len(cleaned_iqr) <= initial_count
        
        data_cleaner.reset_stats()
        
        # Z-score方式
        cleaned_z = data_cleaner.clean_outliers(dirty_data, 'z_score')
        assert len(cleaned_z) <= initial_count
    
    def test_duplicates_cleaning(self, data_cleaner, dirty_data):
        """重複クリーニングテスト"""
        initial_count = len(dirty_data)
        
        cleaned = data_cleaner.clean_duplicates(dirty_data)
        
        assert len(cleaned) <= initial_count
        assert cleaned.duplicated().sum() == 0
    
    def test_format_standardization(self, data_cleaner):
        """フォーマット標準化テスト"""
        # 不正フォーマットのデータ
        messy_data = pd.DataFrame({
            'date': ['2023-01-01', '01/02/2023', '2023-01-03'],
            'open': ['1000', '1010', '1020'],
            'close': [1000.5, '1010.5', 1020.5]
        })
        
        standardized = data_cleaner.standardize_formats(messy_data)
        
        assert pd.api.types.is_datetime64_any_dtype(standardized['date'])
        assert pd.api.types.is_numeric_dtype(standardized['open'])
    
    def test_cleaning_stats(self, data_cleaner, dirty_data):
        """クリーニング統計テスト"""
        data_cleaner.reset_stats()
        
        cleaned = data_cleaner.clean_missing_values(dirty_data, 'drop')
        stats = data_cleaner.get_cleaning_stats()
        
        assert stats['rows_processed'] == len(dirty_data)
        assert stats['rows_removed'] >= 0
        assert isinstance(stats['values_fixed'], int)


class TestDataQualityIntegration:
    """データ品質統合テスト"""
    
    def test_quality_assessment_workflow(self):
        """品質評価ワークフローテスト"""
        # コンポーネント作成
        checker = MockDataQualityChecker()
        cleaner = MockDataCleaner()
        
        # テストデータ
        raw_data = create_test_data('poor')
        
        # 1. 初期品質評価
        initial_report = checker.generate_comprehensive_report(raw_data, "Raw Data")
        assert initial_report.quality_level in [QualityLevel.POOR, QualityLevel.CRITICAL]
        
        # 2. データクリーニング
        cleaned_data = cleaner.clean_missing_values(raw_data, 'drop')
        cleaned_data = cleaner.clean_outliers(cleaned_data, 'iqr')
        cleaned_data = cleaner.clean_duplicates(cleaned_data)
        
        # 3. クリーニング後の品質評価
        final_report = checker.generate_comprehensive_report(cleaned_data, "Cleaned Data")
        
        # 品質が改善されていることを確認
        assert final_report.overall_score > initial_report.overall_score
        assert len(final_report.issues) <= len(initial_report.issues)
    
    def test_quality_monitoring(self):
        """品質モニタリングテスト"""
        checker = MockDataQualityChecker()
        
        # 時系列での品質変化をシミュレート
        datasets = [
            create_test_data('good'),
            create_test_data('poor'),
            create_test_data('critical')
        ]
        
        reports = []
        for i, data in enumerate(datasets):
            report = checker.generate_comprehensive_report(data, f"Dataset_{i}")
            reports.append(report)
        
        # 品質劣化の検出
        assert reports[0].overall_score > reports[1].overall_score > reports[2].overall_score
        
        # アラートが必要なケースの検出
        critical_reports = [r for r in reports if r.quality_level == QualityLevel.CRITICAL]
        assert len(critical_reports) > 0
    
    def test_automated_cleaning_pipeline(self):
        """自動クリーニングパイプラインテスト"""
        def automated_cleaning_pipeline(data: pd.DataFrame) -> pd.DataFrame:
            """自動クリーニングパイプライン"""
            cleaner = MockDataCleaner()
            
            # ステップ1: 重複除去
            data = cleaner.clean_duplicates(data)
            
            # ステップ2: 外れ値除去
            data = cleaner.clean_outliers(data, 'iqr')
            
            # ステップ3: 欠損値処理
            data = cleaner.clean_missing_values(data, 'drop')
            
            # ステップ4: フォーマット標準化
            data = cleaner.standardize_formats(data)
            
            return data
        
        # パイプライン実行
        raw_data = create_test_data('critical')
        cleaned_data = automated_cleaning_pipeline(raw_data)
        
        # 品質チェック
        checker = MockDataQualityChecker()
        
        raw_report = checker.generate_comprehensive_report(raw_data, "Raw")
        cleaned_report = checker.generate_comprehensive_report(cleaned_data, "Cleaned")
        
        # 改善を確認
        assert cleaned_report.overall_score > raw_report.overall_score
        
    def test_quality_thresholds_customization(self):
        """品質しきい値カスタマイゼーションテスト"""
        checker = MockDataQualityChecker()
        
        # しきい値を厳しく設定
        checker.thresholds['completeness'] = 0.99
        checker.thresholds['accuracy'] = 0.995
        
        data = create_test_data('good')
        report = checker.generate_comprehensive_report(data, "Strict")
        
        # 厳しいしきい値により品質レベルが下がる可能性
        assert report.quality_level in [QualityLevel.GOOD, QualityLevel.FAIR, QualityLevel.POOR]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])