#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデルメタデータ管理システム

Issue #850-3 & #850-5: モデルメタデータ管理の強化
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.day_trade.ml.core_types import (
    ModelType,
    PredictionTask,
    DataQuality,
)
from .data_structures import ModelMetadata


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

    def save_ensemble_prediction(self, symbol: str, timestamp: datetime,
                                task: PredictionTask, ensemble_data: Dict) -> bool:
        """アンサンブル予測結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO ensemble_prediction_history
                    (symbol, timestamp, task, final_prediction, confidence,
                     consensus_strength, disagreement_score, prediction_stability,
                     diversity_score, model_count, model_predictions, model_weights,
                     model_quality_scores, excluded_models, ensemble_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timestamp.isoformat(), task.value,
                    str(ensemble_data.get('final_prediction')),
                    ensemble_data.get('confidence', 0.0),
                    ensemble_data.get('consensus_strength', 0.0),
                    ensemble_data.get('disagreement_score', 0.0),
                    ensemble_data.get('prediction_stability', 0.0),
                    ensemble_data.get('diversity_score', 0.0),
                    ensemble_data.get('total_models_used', 0),
                    json.dumps(ensemble_data.get('model_predictions', {})),
                    json.dumps(ensemble_data.get('model_weights', {})),
                    json.dumps(ensemble_data.get('model_quality_scores', {})),
                    json.dumps(ensemble_data.get('excluded_models', [])),
                    ensemble_data.get('ensemble_method', 'unknown')
                ))
            return True
        except Exception as e:
            self.logger.error(f"アンサンブル予測保存エラー: {e}")
            return False

    def get_model_summary(self) -> Dict:
        """モデル概要情報取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 基本統計
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_models,
                           COUNT(DISTINCT symbol) as symbols_count,
                           COUNT(DISTINCT model_type) as model_types_count
                    FROM model_metadata
                """)
                basic_stats = cursor.fetchone()

                # 最近の活動
                cursor = conn.execute("""
                    SELECT model_id, symbol, model_type, created_at
                    FROM model_metadata
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                recent_models = cursor.fetchall()

                return {
                    'total_models': basic_stats[0] if basic_stats else 0,
                    'symbols_count': basic_stats[1] if basic_stats else 0,
                    'model_types_count': basic_stats[2] if basic_stats else 0,
                    'recent_models': [
                        {
                            'model_id': row[0],
                            'symbol': row[1],
                            'model_type': row[2],
                            'created_at': row[3]
                        }
                        for row in recent_models
                    ]
                }
        except Exception as e:
            self.logger.error(f"サマリー取得エラー: {e}")
            return {'error': str(e)}