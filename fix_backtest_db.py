#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Database Fix - バックテストデータベース修正
"""

import sqlite3
from pathlib import Path

def fix_backtest_database():
    """バックテストデータベースの修正"""

    db_path = Path("backtest_data/backtest_results.db")
    db_path.parent.mkdir(exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 既存テーブル削除して再作成
            cursor.execute("DROP TABLE IF EXISTS backtest_results")

            # 正しいスキーマで再作成
            cursor.execute('''
                CREATE TABLE backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return REAL NOT NULL,
                    annualized_return REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    avg_winning_trade REAL NOT NULL,
                    avg_losing_trade REAL NOT NULL,
                    largest_win REAL NOT NULL,
                    largest_loss REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')

            conn.commit()
            print("SUCCESS: backtest_results table recreated")
            return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    fix_backtest_database()