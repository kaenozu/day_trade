#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Manager - データベース管理

性能監視システムのデータベース操作を管理するモジュール
SQLiteを使用した性能履歴、アラート、再学習履歴の永続化
"""

import json
import logging
import sqlite3
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from .config import EnhancedPerformanceConfigManager
from .types import PerformanceMetrics, PerformanceAlert, RetrainingResult, RetrainingScope

logger = logging.getLogger(__name__)


class DatabaseManager:
    """データベース管理クラス
    
    性能監視システムのデータ永続化を管理します。
    SQLiteを使用して性能履歴、アラート、再学習履歴を保存・取得します。
    
    Attributes:
        config_manager: 設定管理インスタンス
        db_path: データベースファイルのパス
    """

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        """初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager
        
        # データベースパスの設定
        db_config = config_manager.config.get('database', {})
        db_path_str = db_config.get('connection', {}).get('path', 'data/model_performance.db')
        self.db_path = Path(db_path_str)
        
        # データベースディレクトリの作成
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # データベース初期化
        self._init_database()

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 性能履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall REAL,
                        f1_score REAL,
                        r2_score REAL,
                        mse REAL,
                        mae REAL,
                        prediction_time_ms REAL,
                        confidence_avg REAL,
                        sample_size INTEGER,
                        status TEXT,
                        prediction_accuracy REAL DEFAULT 0.0,
                        return_prediction REAL DEFAULT 0.0,
                        volatility_prediction REAL DEFAULT 0.0,
                        source TEXT DEFAULT 'validator',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # パフォーマンス履歴にインデックス作成
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_performance_symbol_timestamp 
                    ON model_performance_history(symbol, timestamp DESC)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_performance_model_timestamp 
                    ON model_performance_history(model_type, timestamp DESC)
                ''')

                # アラートテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        model_type TEXT,
                        alert_level TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        message TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at DATETIME,
                        recommended_action TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # アラートテーブルにインデックス作成
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_symbol_timestamp 
                    ON performance_alerts(symbol, timestamp DESC)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_resolved 
                    ON performance_alerts(resolved)
                ''')

                # 再学習履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS retraining_history (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        trigger_type TEXT NOT NULL,
                        affected_symbols TEXT NOT NULL,
                        affected_models TEXT,
                        reason TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        recommended_action TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        completed_at DATETIME,
                        scope TEXT,
                        improvement REAL DEFAULT 0.0,
                        duration REAL DEFAULT 0.0,
                        estimated_time REAL DEFAULT 0.0,
                        actual_benefit REAL DEFAULT 0.0,
                        error_message TEXT,
                        config_snapshot TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # 再学習履歴にインデックス作成
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_retraining_timestamp 
                    ON retraining_history(timestamp DESC)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_retraining_scope 
                    ON retraining_history(scope)
                ''')

                conn.commit()
                logger.info(f"データベース初期化完了: {self.db_path}")

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            raise

    async def save_performance_metrics(self, metrics: PerformanceMetrics):
        """性能指標をデータベースに保存
        
        Args:
            metrics: 保存する性能メトリクス
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_performance_history
                    (symbol, model_type, timestamp, accuracy, precision_score, recall, f1_score,
                     r2_score, mse, mae, prediction_time_ms, confidence_avg, sample_size, status,
                     prediction_accuracy, return_prediction, volatility_prediction, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.symbol, metrics.model_type, metrics.timestamp,
                    metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score,
                    metrics.r2_score, metrics.mse, metrics.mae, metrics.prediction_time_ms,
                    metrics.confidence_avg, metrics.sample_size, metrics.status.value,
                    metrics.prediction_accuracy, metrics.return_prediction,
                    metrics.volatility_prediction, metrics.source
                ))
                conn.commit()
                logger.debug(f"性能指標を保存: {metrics.symbol}_{metrics.model_type}")
                
        except Exception as e:
            logger.error(f"性能指標保存エラー: {e}")

    async def save_alert(self, alert: PerformanceAlert):
        """アラートをデータベースに保存
        
        Args:
            alert: 保存するアラート
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO performance_alerts
                    (id, timestamp, symbol, model_type, alert_level, metric_name,
                     current_value, threshold_value, message, resolved, resolved_at, recommended_action)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.timestamp, alert.symbol, alert.model_type,
                    alert.alert_level.value, alert.metric_name, alert.current_value,
                    alert.threshold_value, alert.message, alert.resolved, alert.resolved_at,
                    alert.recommended_action
                ))
                conn.commit()
                logger.debug(f"アラートを保存: {alert.id}")
                
        except Exception as e:
            logger.error(f"アラート保存エラー: {e}")

    async def save_retraining_result(self, result: RetrainingResult, trigger_info: Dict = None):
        """再学習結果をデータベースに記録
        
        Args:
            result: 再学習結果
            trigger_info: トリガー情報（オプション）
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 設定スナップショットを保存
                config_snapshot = yaml.dump(self.config_manager.config)
                
                trigger_id = trigger_info.get('id', f"retraining_{int(datetime.now().timestamp())}")

                cursor.execute('''
                    INSERT INTO retraining_history
                    (id, timestamp, trigger_type, affected_symbols, reason,
                     severity, recommended_action, scope, improvement,
                     duration, estimated_time, actual_benefit, status,
                     error_message, config_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trigger_id,
                    datetime.now().isoformat(),
                    result.scope.value,
                    ','.join(result.affected_symbols),
                    trigger_info.get('reason', f"性能低下により{result.scope.value}再学習を実行"),
                    'INFO' if result.triggered else 'WARNING',
                    f"{result.scope.value}再学習を実行",
                    result.scope.value,
                    result.improvement,
                    result.duration,
                    result.estimated_time,
                    result.actual_benefit,
                    'success' if result.error is None else 'error',
                    result.error,
                    config_snapshot
                ))
                conn.commit()
                logger.info(f"再学習結果を記録: {trigger_id}")
                
        except Exception as e:
            logger.error(f"再学習結果の記録に失敗: {e}")

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """性能監視の要約情報を取得
        
        Args:
            hours: 取得対象の時間範囲（時間）
            
        Returns:
            要約情報辞書
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 最新の性能情報（銘柄別）
                cursor.execute('''
                    SELECT symbol, accuracy, prediction_accuracy, return_prediction, 
                           volatility_prediction, timestamp, model_type
                    FROM model_performance_history
                    WHERE timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                    LIMIT 50
                '''.format(hours))
                recent_performance = cursor.fetchall()

                # 再学習履歴
                cursor.execute('''
                    SELECT scope, improvement, status, timestamp, duration,
                           estimated_time, actual_benefit
                    FROM retraining_history
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''')
                recent_retraining = cursor.fetchall()

                # アクティブなアラート
                cursor.execute('''
                    SELECT alert_level, symbol, metric_name, current_value, threshold_value, 
                           message, timestamp
                    FROM performance_alerts
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''')
                active_alerts = cursor.fetchall()

                return {
                    'timestamp': datetime.now().isoformat(),
                    'recent_performance': recent_performance,
                    'recent_retraining': recent_retraining,
                    'active_alerts': active_alerts,
                    'performance_count': len(recent_performance),
                    'alert_count': len(active_alerts),
                    'retraining_count': len(recent_retraining)
                }

        except Exception as e:
            logger.error(f"サマリー情報の取得に失敗: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_symbol_performance_history(
            self, symbol: str, model_type: str = None, days: int = 30
    ) -> List[PerformanceMetrics]:
        """特定銘柄の性能履歴を取得
        
        Args:
            symbol: 銘柄コード
            model_type: モデルタイプ（オプション）
            days: 取得対象の日数
            
        Returns:
            性能メトリクスのリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if model_type:
                    cursor.execute('''
                        SELECT symbol, model_type, timestamp, accuracy, precision_score, recall,
                               f1_score, r2_score, mse, mae, prediction_time_ms, confidence_avg,
                               sample_size, status, prediction_accuracy, return_prediction,
                               volatility_prediction, source
                        FROM model_performance_history
                        WHERE symbol = ? AND model_type = ? 
                        AND timestamp >= datetime('now', '-{} days')
                        ORDER BY timestamp DESC
                    '''.format(days), (symbol, model_type))
                else:
                    cursor.execute('''
                        SELECT symbol, model_type, timestamp, accuracy, precision_score, recall,
                               f1_score, r2_score, mse, mae, prediction_time_ms, confidence_avg,
                               sample_size, status, prediction_accuracy, return_prediction,
                               volatility_prediction, source
                        FROM model_performance_history
                        WHERE symbol = ?
                        AND timestamp >= datetime('now', '-{} days')
                        ORDER BY timestamp DESC
                    '''.format(days), (symbol,))

                results = cursor.fetchall()
                
                # PerformanceMetricsオブジェクトに変換
                metrics_list = []
                for row in results:
                    from .types import PerformanceStatus
                    
                    metrics = PerformanceMetrics(
                        symbol=row[0],
                        model_type=row[1],
                        timestamp=datetime.fromisoformat(row[2]) if isinstance(row[2], str) else row[2],
                        accuracy=row[3],
                        precision=row[4],
                        recall=row[5],
                        f1_score=row[6],
                        r2_score=row[7],
                        mse=row[8],
                        mae=row[9],
                        prediction_time_ms=row[10],
                        confidence_avg=row[11],
                        sample_size=row[12] or 0,
                        status=PerformanceStatus(row[13]) if row[13] else PerformanceStatus.UNKNOWN,
                        prediction_accuracy=row[14] or 0.0,
                        return_prediction=row[15] or 0.0,
                        volatility_prediction=row[16] or 0.0,
                        source=row[17] or 'unknown'
                    )
                    metrics_list.append(metrics)
                
                return metrics_list

        except Exception as e:
            logger.error(f"性能履歴取得エラー {symbol}: {e}")
            return []

    def cleanup_old_data(self, days: int = 90):
        """古いデータをクリーンアップ
        
        Args:
            days: 保持期間（日数）
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 古い性能履歴を削除
                cursor.execute('''
                    DELETE FROM model_performance_history
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))
                performance_deleted = cursor.rowcount
                
                # 解決済みの古いアラートを削除
                cursor.execute('''
                    DELETE FROM performance_alerts
                    WHERE resolved = TRUE 
                    AND timestamp < datetime('now', '-{} days')
                '''.format(days // 3))  # アラートはより短期間で削除
                alerts_deleted = cursor.rowcount
                
                # 古い再学習履歴を削除（保持期間を長めに設定）
                cursor.execute('''
                    DELETE FROM retraining_history
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days * 2))
                retraining_deleted = cursor.rowcount
                
                conn.commit()
                logger.info(
                    f"データクリーンアップ完了: "
                    f"性能履歴 {performance_deleted}件, "
                    f"アラート {alerts_deleted}件, "
                    f"再学習履歴 {retraining_deleted}件削除"
                )

        except Exception as e:
            logger.error(f"データクリーンアップエラー: {e}")

    def get_database_stats(self) -> Dict[str, Any]:
        """データベース統計情報を取得
        
        Returns:
            統計情報辞書
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # テーブルごとの件数
                tables = ['model_performance_history', 'performance_alerts', 'retraining_history']
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # データベースファイルサイズ
                stats['db_file_size'] = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                # 最古・最新のレコード
                cursor.execute('''
                    SELECT MIN(timestamp), MAX(timestamp) 
                    FROM model_performance_history
                ''')
                min_ts, max_ts = cursor.fetchone()
                stats['performance_date_range'] = {
                    'oldest': min_ts,
                    'newest': max_ts
                }
                
                return stats

        except Exception as e:
            logger.error(f"データベース統計情報取得エラー: {e}")
            return {'error': str(e)}