#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メタデータ管理システム - ML Prediction Models Metadata Manager

ML予測モデルのメタデータ管理と永続化を行います。
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.day_trade.ml.core_types import (
    DataQuality,
    ModelMetadata,
    ModelType,
    PredictionTask,
)
from .database_schema import DatabaseSchemaManager


class ModelMetadataManager:
    """モデルメタデータ管理システム（強化版）"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self.schema_manager = DatabaseSchemaManager(db_path)
        self._init_metadata_database()

    def _init_metadata_database(self) -> None:
        """メタデータ用データベース初期化"""
        self.schema_manager.init_database()

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

    def get_model_versions(
        self, 
        symbol: str, 
        model_type: ModelType, 
        task: PredictionTask
    ) -> List[str]:
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

    def get_models_by_symbol(self, symbol: str) -> List[ModelMetadata]:
        """シンボル別モデル一覧取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_metadata
                    WHERE symbol = ?
                    ORDER BY updated_at DESC
                """, (symbol,))
                return [self._row_to_metadata(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"シンボル別モデル取得エラー: {e}")
            return []

    def get_model_performance_history(
        self, 
        model_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """モデル性能履歴取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_performance_history
                    WHERE model_id = ?
                    ORDER BY evaluation_date DESC
                    LIMIT ?
                """, (model_id, limit))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"性能履歴取得エラー: {e}")
            return []

    def save_ensemble_prediction(
        self,
        symbol: str,
        prediction_data: Dict,
        task: PredictionTask
    ) -> bool:
        """アンサンブル予測結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO ensemble_prediction_history 
                    (symbol, timestamp, task, final_prediction, confidence, 
                     consensus_strength, disagreement_score, prediction_stability,
                     diversity_score, model_count, avg_model_quality,
                     confidence_variance, model_predictions, model_weights,
                     model_quality_scores, excluded_models, ensemble_method,
                     quality_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now().isoformat(),
                    task.value,
                    str(prediction_data.get('final_prediction')),
                    prediction_data.get('confidence', 0.0),
                    prediction_data.get('consensus_strength', 0.0),
                    prediction_data.get('disagreement_score', 0.0),
                    prediction_data.get('prediction_stability', 0.0),
                    prediction_data.get('diversity_score', 0.0),
                    prediction_data.get('total_models_used', 0),
                    prediction_data.get('avg_model_quality', 0.0),
                    prediction_data.get('confidence_variance', 0.0),
                    json.dumps(prediction_data.get('model_predictions', {})),
                    json.dumps(prediction_data.get('model_weights', {})),
                    json.dumps(prediction_data.get('model_quality_scores', {})),
                    json.dumps(prediction_data.get('excluded_models', [])),
                    prediction_data.get('ensemble_method', 'weighted_average'),
                    json.dumps(prediction_data.get('quality_metrics', {}))
                ))
            return True

        except Exception as e:
            self.logger.error(f"アンサンブル予測保存エラー: {e}")
            return False

    def get_ensemble_prediction_history(
        self,
        symbol: str,
        task: Optional[PredictionTask] = None,
        limit: int = 100
    ) -> List[Dict]:
        """アンサンブル予測履歴取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if task:
                    cursor = conn.execute("""
                        SELECT * FROM ensemble_prediction_history
                        WHERE symbol = ? AND task = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (symbol, task.value, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM ensemble_prediction_history
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (symbol, limit))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"アンサンブル予測履歴取得エラー: {e}")
            return []

    def delete_model_metadata(self, model_id: str) -> bool:
        """モデルメタデータ削除"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 関連する履歴データも削除
                conn.execute(
                    "DELETE FROM model_performance_history WHERE model_id = ?",
                    (model_id,)
                )
                conn.execute(
                    "DELETE FROM model_metadata WHERE model_id = ?",
                    (model_id,)
                )
            
            self.logger.info(f"モデルメタデータ削除完了: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"モデルメタデータ削除エラー: {e}")
            return False

    def update_model_deployment_status(
        self, 
        model_id: str, 
        status: str
    ) -> bool:
        """モデルデプロイ状態更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE model_metadata 
                    SET deployment_status = ?, updated_at = ?
                    WHERE model_id = ?
                """, (status, datetime.now().isoformat(), model_id))
            return True

        except Exception as e:
            self.logger.error(f"デプロイ状態更新エラー: {e}")
            return False

    def get_metadata_summary(self) -> Dict:
        """メタデータサマリー取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 基本統計
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_models,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        COUNT(DISTINCT model_type) as unique_model_types,
                        COUNT(DISTINCT task) as unique_tasks,
                        AVG(model_size_mb) as avg_model_size,
                        AVG(training_time_seconds) as avg_training_time
                    FROM model_metadata
                """)
                basic_stats = dict(zip([col[0] for col in cursor.description], 
                                     cursor.fetchone()))

                # モデルタイプ別統計
                cursor = conn.execute("""
                    SELECT model_type, COUNT(*) as count
                    FROM model_metadata
                    GROUP BY model_type
                """)
                model_type_stats = dict(cursor.fetchall())

                # タスク別統計
                cursor = conn.execute("""
                    SELECT task, COUNT(*) as count
                    FROM model_metadata
                    GROUP BY task
                """)
                task_stats = dict(cursor.fetchall())

                # 最新の訓練日時
                cursor = conn.execute("""
                    SELECT MAX(created_at) as latest_training
                    FROM model_metadata
                """)
                latest_training = cursor.fetchone()[0]

                return {
                    'basic_stats': basic_stats,
                    'model_type_distribution': model_type_stats,
                    'task_distribution': task_stats,
                    'latest_training': latest_training
                }

        except Exception as e:
            self.logger.error(f"サマリー取得エラー: {e}")
            return {'error': str(e)}

    def cleanup_old_records(self, days_to_keep: int = 30) -> bool:
        """古い履歴レコードのクリーンアップ"""
        try:
            cutoff_date = (datetime.now() - 
                         timedelta(days=days_to_keep)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # 古い性能履歴削除
                cursor = conn.execute("""
                    DELETE FROM model_performance_history
                    WHERE evaluation_date < ?
                """, (cutoff_date,))
                perf_deleted = cursor.rowcount

                # 古いアンサンブル予測履歴削除
                cursor = conn.execute("""
                    DELETE FROM ensemble_prediction_history
                    WHERE timestamp < ?
                """, (cutoff_date,))
                ensemble_deleted = cursor.rowcount

                # 古い重み履歴削除
                cursor = conn.execute("""
                    DELETE FROM model_weight_history
                    WHERE created_at < ?
                """, (cutoff_date,))
                weight_deleted = cursor.rowcount

            self.logger.info(f"クリーンアップ完了: "
                           f"性能履歴{perf_deleted}件, "
                           f"アンサンブル履歴{ensemble_deleted}件, "
                           f"重み履歴{weight_deleted}件削除")
            return True

        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            return False