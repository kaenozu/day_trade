#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Service - データベースサービス
SQLiteを使用した軽量データベース機能
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from contextlib import contextmanager

class DatabaseService:
    """データベースサービス"""
    
    def __init__(self, db_path: str = "data/daytrade.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # データベース初期化
        self._initialize_database()
    
    def _initialize_database(self):
        """データベーステーブル初期化"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 価格データテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS price_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        source TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                ''')
                
                # ポートフォリオテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        name TEXT,
                        quantity INTEGER NOT NULL,
                        average_price REAL NOT NULL,
                        purchase_date TEXT NOT NULL,
                        sector TEXT,
                        category TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 取引履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        name TEXT,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        total_amount REAL NOT NULL,
                        commission REAL DEFAULT 0,
                        transaction_date TEXT NOT NULL,
                        notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # アラートテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        condition_json TEXT NOT NULL,
                        message_template TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        status TEXT NOT NULL,
                        expires_at TEXT,
                        last_triggered TEXT,
                        trigger_count INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 通知テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS notifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        notification_id TEXT UNIQUE NOT NULL,
                        alert_id TEXT,
                        symbol TEXT NOT NULL,
                        message TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        data_json TEXT,
                        is_read BOOLEAN DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # バックテスト結果テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        initial_capital REAL NOT NULL,
                        final_capital REAL NOT NULL,
                        total_return_pct REAL NOT NULL,
                        sharpe_ratio REAL,
                        max_drawdown_pct REAL,
                        win_rate REAL,
                        total_trades INTEGER,
                        result_json TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # インデックス作成
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_timestamp ON price_data(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio_positions(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol)')
                
                conn.commit()
                self.logger.info("データベース初期化完了")
                
        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """データベース接続コンテキストマネージャー"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # 辞書形式でアクセス可能
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"データベースエラー: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    # ===== 価格データ関連 =====
    
    def save_price_data(self, symbol: str, timestamp: str, open_price: float,
                       high_price: float, low_price: float, close_price: float,
                       volume: int, source: str = "unknown") -> bool:
        """価格データ保存"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"価格データ保存エラー: {e}")
            return False
    
    def get_price_data(self, symbol: str, start_date: str = None, end_date: str = None,
                      limit: int = 1000) -> List[Dict[str, Any]]:
        """価格データ取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM price_data WHERE symbol = ?"
                params = [symbol]
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"価格データ取得エラー: {e}")
            return []
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """最新価格取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM price_data 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (symbol,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            self.logger.error(f"最新価格取得エラー: {e}")
            return None
    
    # ===== ポートフォリオ関連 =====
    
    def save_portfolio_position(self, symbol: str, name: str, quantity: int,
                               average_price: float, purchase_date: str,
                               sector: str = "", category: str = "") -> bool:
        """ポートフォリオポジション保存"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 既存ポジションチェック
                cursor.execute('SELECT * FROM portfolio_positions WHERE symbol = ?', (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    # 平均価格再計算
                    total_quantity = existing['quantity'] + quantity
                    total_cost = (existing['quantity'] * existing['average_price']) + (quantity * average_price)
                    new_average_price = total_cost / total_quantity if total_quantity > 0 else 0
                    
                    cursor.execute('''
                        UPDATE portfolio_positions 
                        SET quantity = ?, average_price = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    ''', (total_quantity, new_average_price, symbol))
                else:
                    cursor.execute('''
                        INSERT INTO portfolio_positions 
                        (symbol, name, quantity, average_price, purchase_date, sector, category)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, name, quantity, average_price, purchase_date, sector, category))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"ポートフォリオ保存エラー: {e}")
            return False
    
    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """ポートフォリオポジション取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM portfolio_positions ORDER BY created_at DESC')
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"ポートフォリオ取得エラー: {e}")
            return []
    
    def update_portfolio_position_quantity(self, symbol: str, new_quantity: int) -> bool:
        """ポートフォリオ数量更新"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if new_quantity <= 0:
                    # 数量が0以下の場合は削除
                    cursor.execute('DELETE FROM portfolio_positions WHERE symbol = ?', (symbol,))
                else:
                    cursor.execute('''
                        UPDATE portfolio_positions 
                        SET quantity = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    ''', (new_quantity, symbol))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"ポートフォリオ更新エラー: {e}")
            return False
    
    # ===== 取引履歴関連 =====
    
    def save_transaction(self, symbol: str, name: str, action: str, quantity: int,
                        price: float, total_amount: float, commission: float = 0,
                        transaction_date: str = None, notes: str = "") -> bool:
        """取引履歴保存"""
        try:
            if not transaction_date:
                transaction_date = datetime.now().isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO transactions 
                    (symbol, name, action, quantity, price, total_amount, commission, transaction_date, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, name, action, quantity, price, total_amount, commission, transaction_date, notes))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"取引履歴保存エラー: {e}")
            return False
    
    def get_transactions(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """取引履歴取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute('''
                        SELECT * FROM transactions 
                        WHERE symbol = ? 
                        ORDER BY transaction_date DESC 
                        LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM transactions 
                        ORDER BY transaction_date DESC 
                        LIMIT ?
                    ''', (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"取引履歴取得エラー: {e}")
            return []
    
    # ===== アラート関連 =====
    
    def save_alert(self, alert_id: str, user_id: str, symbol: str, alert_type: str,
                  condition: Dict[str, Any], message_template: str, priority: str,
                  status: str, expires_at: str = None) -> bool:
        """アラート保存"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, user_id, symbol, alert_type, condition_json, message_template, priority, status, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (alert_id, user_id, symbol, alert_type, json.dumps(condition), message_template, priority, status, expires_at))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"アラート保存エラー: {e}")
            return False
    
    def get_alerts(self, user_id: str = None, status: str = None) -> List[Dict[str, Any]]:
        """アラート取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM alerts WHERE 1=1"
                params = []
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # JSONデータを復元
                alerts = []
                for row in rows:
                    alert = dict(row)
                    alert['condition'] = json.loads(alert['condition_json'])
                    del alert['condition_json']
                    alerts.append(alert)
                
                return alerts
                
        except Exception as e:
            self.logger.error(f"アラート取得エラー: {e}")
            return []
    
    # ===== バックテスト結果関連 =====
    
    def save_backtest_result(self, strategy_name: str, symbol: str, parameters: Dict[str, Any],
                           start_date: str, end_date: str, initial_capital: float,
                           final_capital: float, total_return_pct: float,
                           result_data: Dict[str, Any]) -> bool:
        """バックテスト結果保存"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO backtest_results 
                    (strategy_name, symbol, parameters_json, start_date, end_date, 
                     initial_capital, final_capital, total_return_pct, sharpe_ratio, 
                     max_drawdown_pct, win_rate, total_trades, result_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy_name, symbol, json.dumps(parameters), start_date, end_date,
                    initial_capital, final_capital, total_return_pct,
                    result_data.get('sharpe_ratio'), result_data.get('max_drawdown_pct'),
                    result_data.get('win_rate'), result_data.get('total_trades'),
                    json.dumps(result_data)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"バックテスト結果保存エラー: {e}")
            return False
    
    def get_backtest_results(self, strategy_name: str = None, symbol: str = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """バックテスト結果取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM backtest_results WHERE 1=1"
                params = []
                
                if strategy_name:
                    query += " AND strategy_name = ?"
                    params.append(strategy_name)
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # JSONデータを復元
                results = []
                for row in rows:
                    result = dict(row)
                    result['parameters'] = json.loads(result['parameters_json'])
                    result['result_data'] = json.loads(result['result_json'])
                    del result['parameters_json']
                    del result['result_json']
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"バックテスト結果取得エラー: {e}")
            return []
    
    # ===== 統計・分析関連 =====
    
    def get_database_stats(self) -> Dict[str, Any]:
        """データベース統計情報"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # 各テーブルのレコード数
                tables = ['price_data', 'portfolio_positions', 'transactions', 'alerts', 'notifications', 'backtest_results']
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    stats[f'{table}_count'] = count
                
                # 最新データの日付
                cursor.execute('SELECT MAX(timestamp) FROM price_data')
                latest_price_date = cursor.fetchone()[0]
                stats['latest_price_date'] = latest_price_date
                
                cursor.execute('SELECT MAX(transaction_date) FROM transactions')
                latest_transaction_date = cursor.fetchone()[0]
                stats['latest_transaction_date'] = latest_transaction_date
                
                # データベースサイズ
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
                
                return stats
                
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """古いデータのクリーンアップ"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 古い価格データ削除
                cursor.execute('DELETE FROM price_data WHERE created_at < ?', (cutoff_date,))
                deleted_prices = cursor.rowcount
                
                # 古い通知削除
                cursor.execute('DELETE FROM notifications WHERE created_at < ?', (cutoff_date,))
                deleted_notifications = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"クリーンアップ完了: 価格データ{deleted_prices}件、通知{deleted_notifications}件を削除")
                return True
                
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            return False