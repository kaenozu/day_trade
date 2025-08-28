#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Manager - データベース管理システム

技術分析結果の保存・読み出し機能
"""

import json
import logging
import sqlite3
from pathlib import Path

from .types import TechnicalAnalysisResult


class DatabaseManager:
    """データベース管理システム"""

    def __init__(self, db_path: Path):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 技術分析結果テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS technical_analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        signals TEXT,
                        patterns TEXT,
                        indicators TEXT,
                        overall_sentiment TEXT,
                        confidence_score REAL,
                        risk_level TEXT,
                        recommendations TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')

                # シグナル履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signal_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        indicator_name TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        strength REAL,
                        confidence REAL,
                        timeframe TEXT,
                        description TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def save_analysis_result(self, result: TechnicalAnalysisResult):
        """分析結果保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO technical_analysis_results
                    (symbol, analysis_type, signals, patterns, indicators,
                     overall_sentiment, confidence_score, risk_level, recommendations, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.symbol,
                    result.analysis_type.value,
                    json.dumps([{
                        'indicator_name': s.indicator_name,
                        'signal_type': s.signal_type,
                        'strength': s.strength,
                        'confidence': s.confidence,
                        'timeframe': s.timeframe,
                        'description': s.description
                    } for s in result.signals]),
                    json.dumps([{
                        'pattern_name': p.pattern_name,
                        'match_score': p.match_score,
                        'pattern_type': p.pattern_type,
                        'reliability': p.reliability
                    } for s in result.patterns]),
                    json.dumps(result.indicators),
                    result.overall_sentiment,
                    result.confidence_score,
                    result.risk_level,
                    json.dumps(result.recommendations),
                    result.timestamp.isoformat()
                ))

                # 個別シグナル保存
                for signal in result.signals:
                    cursor.execute('''
                        INSERT INTO signal_history
                        (symbol, indicator_name, signal_type, strength, confidence,
                         timeframe, description, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.symbol,
                        signal.indicator_name,
                        signal.signal_type,
                        signal.strength,
                        signal.confidence,
                        signal.timeframe,
                        signal.description,
                        signal.timestamp.isoformat()
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"分析結果保存エラー: {e}")