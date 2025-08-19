#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Persistence Module - データ永続化システム  
Issue #933 Phase 3対応: パフォーマンスデータ永続化
"""

import sqlite3
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import csv

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    performance_monitor = None
    HAS_PERFORMANCE_MONITOR = False

try:
    from version import get_version_info
    HAS_VERSION_INFO = True
except ImportError:
    HAS_VERSION_INFO = False


class DataPersistence:
    """データ永続化システム"""
    
    def __init__(self, db_path: str = "data/daytrade_performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # データベース初期化
        self._initialize_database()
        
        # 自動バックアップスケジューラ
        self._backup_thread = None
        self._stop_backup = False
        self.start_auto_backup()
    
    def _initialize_database(self):
        """データベースの初期化とテーブル作成"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    result_data TEXT,
                    confidence_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    status_code INTEGER NOT NULL,
                    request_size INTEGER DEFAULT 0,
                    response_size INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_percent REAL,
                    memory_rss_mb REAL,
                    memory_vms_mb REAL,
                    memory_percent REAL,
                    total_analyses INTEGER DEFAULT 0,
                    total_api_calls INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    stack_trace TEXT,
                    context_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recommendations_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    symbol_name TEXT,
                    recommendation TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    price REAL,
                    change_percent REAL,
                    category TEXT,
                    star_rating TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            ''')
            
            # インデックス作成
            conn.execute('CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON analysis_history(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_history(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_endpoint ON api_performance(endpoint)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_performance(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_error_timestamp ON error_logs(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp ON recommendations_history(timestamp)')
            
            conn.commit()
    
    def save_analysis_result(self, symbol: str, analysis_type: str, duration_ms: float,
                           result_data: Dict[str, Any], confidence_score: float = None,
                           session_id: str = None):
        """分析結果を保存"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO analysis_history 
                    (symbol, analysis_type, duration_ms, result_data, confidence_score, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, analysis_type, duration_ms, 
                    json.dumps(result_data, ensure_ascii=False),
                    confidence_score, session_id
                ))
                conn.commit()
    
    def save_api_performance(self, endpoint: str, response_time_ms: float, 
                           status_code: int, request_size: int = 0, 
                           response_size: int = 0, session_id: str = None):
        """API性能データを保存"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO api_performance 
                    (endpoint, response_time_ms, status_code, request_size, response_size, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (endpoint, response_time_ms, status_code, request_size, response_size, session_id))
                conn.commit()
    
    def save_system_metrics(self, cpu_percent: float = None, memory_rss_mb: float = None,
                          memory_vms_mb: float = None, memory_percent: float = None,
                          total_analyses: int = 0, total_api_calls: int = 0,
                          session_id: str = None):
        """システムメトリクスを保存"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO system_metrics 
                    (cpu_percent, memory_rss_mb, memory_vms_mb, memory_percent, 
                     total_analyses, total_api_calls, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (cpu_percent, memory_rss_mb, memory_vms_mb, memory_percent,
                      total_analyses, total_api_calls, session_id))
                conn.commit()
    
    def save_error_log(self, error_type: str, error_message: str, 
                      stack_trace: str = None, context_data: Dict = None,
                      session_id: str = None):
        """エラーログを保存"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO error_logs 
                    (error_type, error_message, stack_trace, context_data, session_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    error_type, error_message, stack_trace,
                    json.dumps(context_data, ensure_ascii=False) if context_data else None,
                    session_id
                ))
                conn.commit()
    
    def save_recommendations(self, recommendations: List[Dict[str, Any]], session_id: str = None):
        """推奨銘柄データを保存"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for rec in recommendations:
                    conn.execute('''
                        INSERT INTO recommendations_history
                        (symbol, symbol_name, recommendation, confidence_score, 
                         price, change_percent, category, star_rating, session_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rec.get('code', ''),
                        rec.get('name', ''),
                        rec.get('recommendation', ''),
                        rec.get('confidence', 0.0),
                        rec.get('price', 0.0),
                        rec.get('change_percent', 0.0),
                        rec.get('category', ''),
                        rec.get('star_rating', ''),
                        session_id
                    ))
                conn.commit()
    
    def get_analysis_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """分析統計を取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 基本統計
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_analyses,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(duration_ms) as min_duration_ms,
                    MAX(duration_ms) as max_duration_ms,
                    AVG(confidence_score) as avg_confidence
                FROM analysis_history 
                WHERE timestamp > ?
            ''', (cutoff_time,)).fetchone()
            
            # 分析タイプ別統計
            type_stats = conn.execute('''
                SELECT 
                    analysis_type,
                    COUNT(*) as count,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(confidence_score) as avg_confidence
                FROM analysis_history 
                WHERE timestamp > ?
                GROUP BY analysis_type
                ORDER BY count DESC
            ''', (cutoff_time,)).fetchall()
            
            # 銘柄別統計
            symbol_stats = conn.execute('''
                SELECT 
                    symbol,
                    COUNT(*) as count,
                    AVG(duration_ms) as avg_duration_ms,
                    MAX(timestamp) as last_analysis
                FROM analysis_history 
                WHERE timestamp > ?
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 10
            ''', (cutoff_time,)).fetchall()
            
            return {
                'period_hours': hours,
                'total_statistics': dict(stats) if stats else {},
                'by_analysis_type': [dict(row) for row in type_stats],
                'by_symbol': [dict(row) for row in symbol_stats]
            }
    
    def get_api_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """API統計を取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 基本統計
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time_ms) as avg_response_time_ms,
                    MIN(response_time_ms) as min_response_time_ms,
                    MAX(response_time_ms) as max_response_time_ms
                FROM api_performance 
                WHERE timestamp > ?
            ''', (cutoff_time,)).fetchone()
            
            # エンドポイント別統計
            endpoint_stats = conn.execute('''
                SELECT 
                    endpoint,
                    COUNT(*) as requests,
                    AVG(response_time_ms) as avg_response_time_ms,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
                FROM api_performance 
                WHERE timestamp > ?
                GROUP BY endpoint
                ORDER BY requests DESC
            ''', (cutoff_time,)).fetchall()
            
            return {
                'period_hours': hours,
                'total_statistics': dict(stats) if stats else {},
                'by_endpoint': [dict(row) for row in endpoint_stats]
            }
    
    def export_data(self, format: str = 'json', output_dir: str = "exports") -> str:
        """データをエクスポート"""
        export_path = Path(output_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'json':
            return self._export_json(export_path, timestamp)
        elif format.lower() == 'csv':
            return self._export_csv(export_path, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, export_path: Path, timestamp: str) -> str:
        """JSON形式でエクスポート"""
        filename = export_path / f"daytrade_data_{timestamp}.json"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': get_version_info() if HAS_VERSION_INFO else {'version': 'unknown'}
                },
                'analysis_history': [dict(row) for row in conn.execute('SELECT * FROM analysis_history ORDER BY timestamp DESC LIMIT 1000')],
                'api_performance': [dict(row) for row in conn.execute('SELECT * FROM api_performance ORDER BY timestamp DESC LIMIT 1000')],
                'system_metrics': [dict(row) for row in conn.execute('SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 100')],
                'error_logs': [dict(row) for row in conn.execute('SELECT * FROM error_logs ORDER BY timestamp DESC LIMIT 500')],
                'recommendations_history': [dict(row) for row in conn.execute('SELECT * FROM recommendations_history ORDER BY timestamp DESC LIMIT 100')]
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        return str(filename)
    
    def _export_csv(self, export_path: Path, timestamp: str) -> str:
        """CSV形式でエクスポート"""
        exported_files = []
        
        tables = [
            'analysis_history', 'api_performance', 'system_metrics', 
            'error_logs', 'recommendations_history'
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            for table in tables:
                filename = export_path / f"{table}_{timestamp}.csv"
                
                cursor = conn.execute(f'SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 1000')
                rows = cursor.fetchall()
                
                if rows:
                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        # ヘッダー
                        writer.writerow(rows[0].keys())
                        # データ
                        for row in rows:
                            writer.writerow(row)
                    
                    exported_files.append(str(filename))
        
        return f"Exported {len(exported_files)} files: {exported_files}"
    
    def start_auto_backup(self, interval_hours: int = 24):
        """自動バックアップを開始"""
        if self._backup_thread is None or not self._backup_thread.is_alive():
            self._stop_backup = False
            self._backup_thread = threading.Thread(target=self._backup_loop, args=(interval_hours,))
            self._backup_thread.daemon = True
            self._backup_thread.start()
    
    def stop_auto_backup(self):
        """自動バックアップを停止"""
        self._stop_backup = True
        if self._backup_thread:
            self._backup_thread.join(timeout=1)
    
    def _backup_loop(self, interval_hours: int):
        """バックアップループ"""
        import time
        
        while not self._stop_backup:
            try:
                # JSON形式でバックアップ
                backup_file = self.export_data(format='json', output_dir='backups')
                print(f"[バックアップ] データベースをバックアップしました: {backup_file}")
                
                # 古いバックアップの削除（30日以上古いもの）
                self._cleanup_old_backups()
                
            except Exception as e:
                print(f"[バックアップエラー] {e}")
            
            # 指定時間待機
            time.sleep(interval_hours * 3600)
    
    def _cleanup_old_backups(self, days: int = 30):
        """古いバックアップファイルを削除"""
        backup_dir = Path("backups")
        if not backup_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file in backup_dir.glob("daytrade_data_*.json"):
            if file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    file.unlink()
                    print(f"[クリーンアップ] 古いバックアップを削除: {file}")
                except Exception as e:
                    print(f"[クリーンアップエラー] {file}: {e}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """データベース情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # テーブル情報
            tables = conn.execute('''
                SELECT name, sql FROM sqlite_master WHERE type='table'
            ''').fetchall()
            
            # テーブルごとの行数
            table_counts = {}
            for table in tables:
                count = conn.execute(f'SELECT COUNT(*) FROM {table["name"]}').fetchone()[0]
                table_counts[table['name']] = count
            
            # データベースサイズ
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'database_path': str(self.db_path),
                'database_size_mb': db_size / 1024 / 1024,
                'table_info': [dict(row) for row in tables],
                'table_row_counts': table_counts,
                'total_rows': sum(table_counts.values())
            }


# グローバルインスタンス
data_persistence = DataPersistence()


if __name__ == "__main__":
    # テスト実行
    persistence = DataPersistence()
    
    # テストデータ作成
    persistence.save_analysis_result("7203", "test_analysis", 150.5, 
                                   {"recommendation": "BUY", "price": 1500}, 
                                   confidence_score=0.85)
    
    persistence.save_api_performance("/api/recommendations", 120.3, 200, 
                                   request_size=100, response_size=2500)
    
    persistence.save_system_metrics(cpu_percent=15.2, memory_rss_mb=128.5, 
                                  total_analyses=5, total_api_calls=12)
    
    # 統計表示
    print("=== 分析統計 ===")
    analysis_stats = persistence.get_analysis_statistics()
    print(json.dumps(analysis_stats, ensure_ascii=False, indent=2, default=str))
    
    print("\n=== API統計 ===")
    api_stats = persistence.get_api_statistics()
    print(json.dumps(api_stats, ensure_ascii=False, indent=2, default=str))
    
    print("\n=== データベース情報 ===")
    db_info = persistence.get_database_info()
    print(json.dumps(db_info, ensure_ascii=False, indent=2, default=str))
    
    # エクスポートテスト
    export_result = persistence.export_data('json')
    print(f"\nエクスポート完了: {export_result}")