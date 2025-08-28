#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Database Manager
データベース管理クラス
"""

import sqlite3
import logging
import yaml
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .enums_and_models import (
    PerformanceMetrics,
    PerformanceAlert,
    RetrainingTrigger,
    RetrainingResult,
    AlertLevel
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """データベース管理クラス"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
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

                conn.commit()
                logger.info("データベース初期化完了")

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")

    async def save_performance_metrics(self, metrics: PerformanceMetrics):
        """性能指標をデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_performance_history
                    (symbol, model_type, timestamp, accuracy, precision_score, recall, f1_score,
                     prediction_time_ms, confidence_avg, sample_size, status,
                     prediction_accuracy, return_prediction, volatility_prediction, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.symbol, metrics.model_type, metrics.timestamp,
                    metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score,
                    metrics.prediction_time_ms, metrics.confidence_avg,
                    metrics.sample_size, metrics.status.value,
                    metrics.prediction_accuracy, metrics.return_prediction,
                    metrics.volatility_prediction, metrics.source
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"性能指標保存エラー: {e}")

    async def save_alert(self, alert: PerformanceAlert):
        """アラートをデータベースに保存"""
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

        except Exception as e:
            logger.error(f"アラート保存エラー: {e}")

    async def save_retraining_trigger(self, trigger: RetrainingTrigger):
        """再学習トリガーをデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO retraining_history
                    (id, timestamp, trigger_type, affected_symbols, affected_models,
                     reason, severity, recommended_action, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trigger.trigger_id, trigger.timestamp, trigger.trigger_type,
                    json.dumps(trigger.affected_symbols), json.dumps(trigger.affected_models),
                    trigger.reason, trigger.severity.value, trigger.recommended_action, 'triggered'
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"再学習トリガー保存エラー: {e}")

    async def record_enhanced_retraining_result(
            self, result: RetrainingResult, config: Dict[str, Any]
    ):
        """改善版再学習結果をデータベースに記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 設定スナップショットを保存
                config_snapshot = yaml.dump(config)

                trigger_id = f"retraining_{int(time.time())}"

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
                    f"性能低下により{result.scope.value}再学習を実行",
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

        except Exception as e:
            logger.error(f"改善版再学習結果の記録に失敗: {e}")

    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """改善版性能監視の要約情報を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 最新の性能情報（銘柄別）
                cursor.execute('''
                    SELECT symbol, accuracy, prediction_accuracy, return_prediction, volatility_prediction, timestamp
                    FROM model_performance_history
                    WHERE timestamp >= datetime('now', '-24 hours')
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''')
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
                    SELECT alert_level, symbol, metric_name, current_value, threshold_value, message
                    FROM performance_alerts
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''')
                active_alerts = cursor.fetchall()

                return {
                    'recent_performance': recent_performance,
                    'recent_retraining': recent_retraining,
                    'active_alerts': active_alerts,
                    'alert_count': len(active_alerts)
                }

        except Exception as e:
            logger.error(f"改善版サマリー情報の取得に失敗: {e}")
            return {'error': str(e)}