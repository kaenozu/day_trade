#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データベーススキーマ管理 - ML Prediction Models Database Schema

ML予測モデルのデータベーススキーマ定義と初期化を行います。
"""

import logging
import sqlite3
from pathlib import Path


class DatabaseSchemaManager:
    """データベーススキーマ管理クラス"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)

    def init_database(self) -> None:
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # モデルメタデータテーブル
                self._create_model_metadata_table(conn)
                
                # モデル性能履歴テーブル
                self._create_performance_history_table(conn)
                
                # アンサンブル予測履歴テーブル
                self._create_ensemble_prediction_table(conn)
                
                # 予測精度追跡テーブル
                self._create_accuracy_tracking_table(conn)
                
                # モデル重み履歴テーブル
                self._create_weight_history_table(conn)

                # インデックス作成
                self._create_database_indexes(conn)

            self.logger.info("データベース初期化完了")

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")
            raise

    def _create_model_metadata_table(self, conn: sqlite3.Connection) -> None:
        """モデルメタデータテーブル作成"""
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

    def _create_performance_history_table(self, conn: sqlite3.Connection) -> None:
        """モデル性能履歴テーブル作成"""
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

    def _create_ensemble_prediction_table(self, conn: sqlite3.Connection) -> None:
        """アンサンブル予測履歴テーブル作成"""
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

    def _create_accuracy_tracking_table(self, conn: sqlite3.Connection) -> None:
        """予測精度追跡テーブル作成"""
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

    def _create_weight_history_table(self, conn: sqlite3.Connection) -> None:
        """モデル重み履歴テーブル作成"""
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

    def _create_database_indexes(self, conn: sqlite3.Connection) -> None:
        """データベースインデックス作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_model_metadata_symbol ON model_metadata(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_model_metadata_type_task ON model_metadata(model_type, task)",
            "CREATE INDEX IF NOT EXISTS idx_performance_history_model_id ON model_performance_history(model_id)",
            "CREATE INDEX IF NOT EXISTS idx_ensemble_history_symbol ON ensemble_prediction_history(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_tracking_symbol ON prediction_accuracy_tracking(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_weight_history_symbol_task ON model_weight_history(symbol, task)"
        ]

        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"インデックス作成警告: {e}")

    def upgrade_schema(self, target_version: str) -> bool:
        """スキーマ更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # バージョンテーブルがない場合作成
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version TEXT PRIMARY KEY,
                        applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 現在のバージョン取得
                cursor = conn.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
                current_version = cursor.fetchone()
                current_version = current_version[0] if current_version else "0.0.0"

                if current_version == target_version:
                    self.logger.info(f"スキーマは最新版です: {target_version}")
                    return True

                # 更新処理
                success = self._apply_schema_migrations(conn, current_version, target_version)
                
                if success:
                    # バージョン記録
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (target_version,)
                    )
                    self.logger.info(f"スキーマ更新完了: {current_version} -> {target_version}")

                return success

        except Exception as e:
            self.logger.error(f"スキーマ更新エラー: {e}")
            return False

    def _apply_schema_migrations(self, conn: sqlite3.Connection, from_version: str, to_version: str) -> bool:
        """スキーママイグレーション実行"""
        try:
            # バージョン別マイグレーション処理
            migrations = {
                "1.0.0": self._migrate_to_v1_0_0,
                "2.0.0": self._migrate_to_v2_0_0
            }

            # 必要なマイグレーションを順次実行
            for version in ["1.0.0", "2.0.0"]:
                if self._version_compare(from_version, version) < 0 and self._version_compare(version, to_version) <= 0:
                    if version in migrations:
                        migrations[version](conn)
                        self.logger.info(f"マイグレーション適用: {version}")

            return True

        except Exception as e:
            self.logger.error(f"マイグレーション実行エラー: {e}")
            return False

    def _migrate_to_v1_0_0(self, conn: sqlite3.Connection) -> None:
        """バージョン1.0.0へのマイグレーション"""
        # 必要に応じてカラム追加などの処理
        pass

    def _migrate_to_v2_0_0(self, conn: sqlite3.Connection) -> None:
        """バージョン2.0.0へのマイグレーション"""
        # 新しいテーブルやカラムの追加
        try:
            conn.execute("""
                ALTER TABLE model_metadata 
                ADD COLUMN model_hash TEXT DEFAULT ''
            """)
        except sqlite3.OperationalError:
            # カラムが既に存在する場合は無視
            pass

    def _version_compare(self, v1: str, v2: str) -> int:
        """バージョン比較（-1: v1 < v2, 0: v1 == v2, 1: v1 > v2）"""
        try:
            v1_parts = [int(x) for x in v1.split('.')]
            v2_parts = [int(x) for x in v2.split('.')]
            
            # 長さを合わせる
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts += [0] * (max_len - len(v1_parts))
            v2_parts += [0] * (max_len - len(v2_parts))
            
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0

        except Exception:
            return 0

    def validate_schema(self) -> Dict[str, bool]:
        """スキーマ検証"""
        validation_result = {
            'model_metadata': False,
            'model_performance_history': False,
            'ensemble_prediction_history': False,
            'prediction_accuracy_tracking': False,
            'model_weight_history': False
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]

                for table_name in validation_result.keys():
                    validation_result[table_name] = table_name in existing_tables

        except Exception as e:
            self.logger.error(f"スキーマ検証エラー: {e}")

        return validation_result

    def get_table_info(self, table_name: str) -> List[Dict]:
        """テーブル情報取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                return [
                    {
                        'cid': col[0],
                        'name': col[1],
                        'type': col[2],
                        'notnull': bool(col[3]),
                        'default_value': col[4],
                        'pk': bool(col[5])
                    }
                    for col in columns
                ]

        except Exception as e:
            self.logger.error(f"テーブル情報取得エラー: {e}")
            return []

    def optimize_database(self) -> bool:
        """データベース最適化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # VACUUM実行
                conn.execute("VACUUM")
                
                # ANALYZE実行
                conn.execute("ANALYZE")
                
                # WALモード設定（パフォーマンス向上）
                conn.execute("PRAGMA journal_mode=WAL")
                
                # 同期設定（パフォーマンス重視）
                conn.execute("PRAGMA synchronous=NORMAL")

            self.logger.info("データベース最適化完了")
            return True

        except Exception as e:
            self.logger.error(f"データベース最適化エラー: {e}")
            return False