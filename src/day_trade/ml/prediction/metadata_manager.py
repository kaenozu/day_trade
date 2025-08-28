#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデルメタデータ管理システム

機械学習モデルのメタデータ管理、データベーススキーマ、
性能追跡、バージョン管理を行うシステムです。
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base_types import (
    DataQuality,
    ModelTrainingResult,
    DataQualityReport
)
from src.day_trade.ml.core_types import (
    ModelType,
    PredictionTask,
    ModelMetadata,
    ModelPerformance
)


class ModelMetadataManager:
    """モデルメタデータ管理システム（強化版）"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_metadata_database()

    def _init_metadata_database(self):
        """メタデータ用データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # モデルメタデータテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    feature_columns TEXT NOT NULL,
                    target_info TEXT NOT NULL,
                    training_samples INTEGER,
                    training_period TEXT,
                    data_quality TEXT,
                    hyperparameters TEXT,
                    preprocessing_config TEXT,
                    feature_selection_config TEXT,
                    performance_metrics TEXT,
                    cross_validation_scores TEXT,
                    feature_importance TEXT,
                    is_classifier BOOLEAN,
                    model_size_mb REAL,
                    training_time_seconds REAL,
                    python_version TEXT,
                    sklearn_version TEXT,
                    framework_versions TEXT,
                    validation_status TEXT DEFAULT 'pending',
                    deployment_status TEXT DEFAULT 'development',
                    performance_threshold_met BOOLEAN DEFAULT FALSE,
                    data_drift_detected BOOLEAN DEFAULT FALSE
                )
            """)

            # モデル性能履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    dataset_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    r2_score REAL,
                    mse REAL,
                    rmse REAL,
                    mae REAL,
                    cross_val_mean REAL,
                    cross_val_std REAL,
                    prediction_stability REAL,
                    confidence_calibration REAL,
                    feature_drift_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES model_metadata (model_id)
                )
            """)

            # アンサンブル予測履歴テーブル（強化版）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task TEXT NOT NULL,
                    final_prediction TEXT,
                    confidence REAL,
                    consensus_strength REAL,
                    disagreement_score REAL,
                    prediction_stability REAL,
                    diversity_score REAL,
                    model_count INTEGER,
                    avg_model_quality REAL,
                    confidence_variance REAL,
                    model_predictions TEXT,
                    model_weights TEXT,
                    model_quality_scores TEXT,
                    excluded_models TEXT,
                    ensemble_method TEXT,
                    quality_metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 予測精度追跡テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_accuracy_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    predicted_value TEXT,
                    actual_value TEXT,
                    prediction_date TEXT,
                    evaluation_date TEXT,
                    confidence_at_prediction REAL,
                    accuracy_score REAL,
                    error_magnitude REAL,
                    model_used TEXT,
                    task TEXT,
                    was_correct BOOLEAN,
                    error_category TEXT,
                    market_conditions TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # モデル重み履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    static_weight REAL,
                    dynamic_weight REAL,
                    performance_contribution REAL,
                    confidence_contribution REAL,
                    quality_contribution REAL,
                    weight_change_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # データ品質履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    total_samples INTEGER,
                    missing_values_count INTEGER,
                    missing_values_rate REAL,
                    duplicate_rows INTEGER,
                    date_gaps_count INTEGER,
                    price_anomalies INTEGER,
                    volume_anomalies INTEGER,
                    ohlc_inconsistencies INTEGER,
                    quality_score REAL,
                    quality_level TEXT,
                    issues TEXT,
                    recommendations TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 訓練結果履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_results_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    training_start TEXT NOT NULL,
                    training_end TEXT NOT NULL,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    feature_count INTEGER,
                    train_score REAL,
                    test_score REAL,
                    cross_val_mean REAL,
                    cross_val_std REAL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    r2_score REAL,
                    mse REAL,
                    rmse REAL,
                    mae REAL,
                    feature_importance TEXT,
                    hyperparameters TEXT,
                    model_size_mb REAL,
                    data_quality TEXT,
                    training_successful BOOLEAN,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_metadata(self, metadata: ModelMetadata) -> bool:
        """メタデータ保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_metadata VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.model_type.value,
                    metadata.task.value,
                    metadata.symbol,
                    metadata.version,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    json.dumps(metadata.feature_columns),
                    json.dumps(metadata.target_info),
                    metadata.training_samples,
                    metadata.training_period,
                    metadata.data_quality.value,
                    json.dumps(metadata.hyperparameters),
                    json.dumps(metadata.preprocessing_config),
                    json.dumps(metadata.feature_selection_config),
                    json.dumps(metadata.performance_metrics),
                    json.dumps(metadata.cross_validation_scores),
                    json.dumps(metadata.feature_importance),
                    metadata.is_classifier,
                    metadata.model_size_mb,
                    metadata.training_time_seconds,
                    metadata.python_version,
                    metadata.sklearn_version,
                    json.dumps(metadata.framework_versions),
                    metadata.validation_status,
                    metadata.deployment_status,
                    metadata.performance_threshold_met,
                    metadata.data_drift_detected
                ))
            return True
        except Exception as e:
            self.logger.error(f"メタデータ保存エラー: {e}")
            return False

    def load_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """メタデータ読み込み"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_metadata WHERE model_id = ?
                """, (model_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_metadata(row)
                return None

        except Exception as e:
            self.logger.error(f"メタデータ読み込みエラー: {e}")
            return None

    def _row_to_metadata(self, row) -> ModelMetadata:
        """データベース行をModelMetadataに変換"""
        return ModelMetadata(
            model_id=row[0],
            model_type=ModelType(row[1]),
            task=PredictionTask(row[2]),
            symbol=row[3],
            version=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            feature_columns=json.loads(row[7]),
            target_info=json.loads(row[8]),
            training_samples=row[9],
            training_period=row[10],
            data_quality=DataQuality(row[11]),
            hyperparameters=json.loads(row[12]),
            preprocessing_config=json.loads(row[13]),
            feature_selection_config=json.loads(row[14]),
            performance_metrics=json.loads(row[15]),
            cross_validation_scores=json.loads(row[16]),
            feature_importance=json.loads(row[17]),
            is_classifier=bool(row[18]),
            model_size_mb=row[19],
            training_time_seconds=row[20],
            python_version=row[21],
            sklearn_version=row[22],
            framework_versions=json.loads(row[23]),
            validation_status=row[24],
            deployment_status=row[25],
            performance_threshold_met=bool(row[26]),
            data_drift_detected=bool(row[27])
        )

    def get_model_versions(self, symbol: str, model_type: ModelType, 
                          task: PredictionTask) -> List[str]:
        """モデルバージョン一覧取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version FROM model_metadata
                    WHERE symbol = ? AND model_type = ? AND task = ?
                    ORDER BY created_at DESC
                """, (symbol, model_type.value, task.value))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"バージョン一覧取得エラー: {e}")
            return []

    def save_training_result(self, result: ModelTrainingResult) -> bool:
        """訓練結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO training_results_history
                    (model_id, symbol, model_type, task, training_start, training_end,
                     training_samples, test_samples, feature_count, train_score, test_score,
                     cross_val_mean, cross_val_std, accuracy, precision_score, recall_score,
                     f1_score, r2_score, mse, rmse, mae, feature_importance, hyperparameters,
                     model_size_mb, data_quality, training_successful, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.model_id, result.symbol, result.model_type.value, result.task.value,
                    result.training_start.isoformat(), result.training_end.isoformat(),
                    result.training_samples, result.test_samples, result.feature_count,
                    result.train_score, result.test_score, result.cross_val_mean, result.cross_val_std,
                    result.accuracy, result.precision, result.recall, result.f1_score,
                    result.r2_score, result.mse, result.rmse, result.mae,
                    json.dumps(result.feature_importance), json.dumps(result.hyperparameters),
                    result.model_size_mb, result.data_quality.value,
                    result.training_successful, result.error_message
                ))
            return True
        except Exception as e:
            self.logger.error(f"訓練結果保存エラー: {e}")
            return False

    def save_data_quality_report(self, report: DataQualityReport) -> bool:
        """データ品質レポート保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO data_quality_history
                    (symbol, evaluation_date, total_samples, missing_values_count,
                     missing_values_rate, duplicate_rows, date_gaps_count, price_anomalies,
                     volume_anomalies, ohlc_inconsistencies, quality_score, quality_level,
                     issues, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.symbol, report.timestamp.isoformat(), report.total_samples,
                    report.missing_values_count, report.missing_values_rate, report.duplicate_rows,
                    report.date_gaps_count, report.price_anomalies, report.volume_anomalies,
                    report.ohlc_inconsistencies, report.quality_score, report.quality_level.value,
                    json.dumps(report.issues), json.dumps(report.recommendations)
                ))
            return True
        except Exception as e:
            self.logger.error(f"データ品質レポート保存エラー: {e}")
            return False

    def get_latest_models(self, symbol: str, limit: int = 10) -> List[ModelMetadata]:
        """最新モデル一覧取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_metadata
                    WHERE symbol = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (symbol, limit))
                
                models = []
                for row in cursor.fetchall():
                    try:
                        models.append(self._row_to_metadata(row))
                    except Exception as e:
                        self.logger.warning(f"メタデータ変換エラー: {e}")
                
                return models
        except Exception as e:
            self.logger.error(f"最新モデル取得エラー: {e}")
            return []

    def get_performance_history(self, model_id: str, 
                              limit: int = 50) -> List[Dict[str, any]]:
        """性能履歴取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_performance_history
                    WHERE model_id = ?
                    ORDER BY evaluation_date DESC
                    LIMIT ?
                """, (model_id, limit))
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"性能履歴取得エラー: {e}")
            return []

    def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """古いレコードのクリーンアップ"""
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            deleted_count = 0
            
            with sqlite3.connect(self.db_path) as conn:
                # 古い性能履歴を削除
                cursor = conn.execute("""
                    DELETE FROM model_performance_history 
                    WHERE evaluation_date < ?
                """, (cutoff_str,))
                deleted_count += cursor.rowcount
                
                # 古いアンサンブル予測履歴を削除
                cursor = conn.execute("""
                    DELETE FROM ensemble_prediction_history 
                    WHERE timestamp < ?
                """, (cutoff_str,))
                deleted_count += cursor.rowcount
                
                # 古い予測精度追跡を削除
                cursor = conn.execute("""
                    DELETE FROM prediction_accuracy_tracking 
                    WHERE prediction_date < ?
                """, (cutoff_str,))
                deleted_count += cursor.rowcount
                
                # VACUUMでデータベースを最適化
                conn.execute("VACUUM")
            
            self.logger.info(f"古いレコードをクリーンアップしました: {deleted_count}件")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            return 0

    def get_summary_stats(self) -> Dict[str, any]:
        """サマリー統計取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # モデル数
                cursor = conn.execute("SELECT COUNT(*) FROM model_metadata")
                stats['total_models'] = cursor.fetchone()[0]
                
                # 銘柄数
                cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM model_metadata")
                stats['total_symbols'] = cursor.fetchone()[0]
                
                # 最新訓練日
                cursor = conn.execute("""
                    SELECT MAX(updated_at) FROM model_metadata
                """)
                latest_training = cursor.fetchone()[0]
                stats['latest_training'] = latest_training
                
                # モデルタイプ別統計
                cursor = conn.execute("""
                    SELECT model_type, COUNT(*) FROM model_metadata
                    GROUP BY model_type
                """)
                stats['models_by_type'] = dict(cursor.fetchall())
                
                # タスク別統計
                cursor = conn.execute("""
                    SELECT task, COUNT(*) FROM model_metadata
                    GROUP BY task
                """)
                stats['models_by_task'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            self.logger.error(f"サマリー統計取得エラー: {e}")
            return {}