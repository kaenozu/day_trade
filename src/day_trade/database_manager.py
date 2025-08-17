#!/usr / bin / env python3
# -*- coding: utf - 8 -*-
"""
Database Manager - データベース操作を管理する
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Optional

from src.day_trade.data_models import Position, PositionStatus, RiskLevel

class DatabaseManager:
    """DatabaseManagerクラス"""
    """__init__関数"""
    def __init__(self, db_path: str = 'data / daytrade.db') -> None:
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """データベースに接続し、テーブルを作成する"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.logger.info(f"データベースに接続しました: {self.db_path}")
            self._create_tables()
        except sqlite3.Error as e:
            self.logger.error(f"データベース接続エラー: {e}")
            self.conn: Optional[Any] = None

    def _create_tables(self) -> None:
        """ポジション管理用のテーブルを作成する"""
        if not self.conn:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    current_price REAL,
                    status TEXT NOT NULL,
                    pnl REAL,
                    pnl_percent REAL,
                    risk_level TEXT,
                    max_holding_time INTEGER,
                    close_time TEXT
                )
            """)
            self.conn.commit()
            self.logger.info("'positions'テーブルの準備ができました")
        except sqlite3.Error as e:
            self.logger.error(f"テーブル作成エラー: {e}")

    def save_position(self, position: Position) -> None:
        """ポジションをデータベースに保存（新規作成または更新）"""
        if not self.conn:
            return

        query = """
            INSERT OR REPLACE INTO positions (
                symbol, name, entry_price, quantity, entry_time,
                stop_loss, take_profit, current_price, status, pnl,
                pnl_percent, risk_level, max_holding_time, close_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            position.symbol,
            position.name,
            position.entry_price,
            position.quantity,
            position.entry_time.isoformat(),
            position.stop_loss,
            position.take_profit,
            position.current_price,
            position.status.value,
            position.pnl,
            position.pnl_percent,
            position.risk_level.value,
            position.max_holding_time,
            datetime.now().isoformat() if position.status != PositionStatus.OPEN else None
        )

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            self.logger.info(f"ポジションを保存しました: {position.symbol}")
        except sqlite3.Error as e:
            self.logger.error(f"ポジション保存エラー: {e}")

    def load_open_positions(self) -> List[Position]:
        """オープンなポジションをデータベースから読み込む"""
        if not self.conn:
            return []

        query = "SELECT * FROM positions WHERE status = ?"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (PositionStatus.OPEN.value,))
            rows = cursor.fetchall()

            positions = []
            for row in rows:
                positions.append(self._row_to_position(row))

            self.logger.info(f"{len(positions)}件のオープンなポジションを読み込みました")
            return positions
        except sqlite3.Error as e:
            self.logger.error(f"オープンポジション読み込みエラー: {e}")
            return []

    def delete_all_positions(self) -> None:
        """すべてのポジションを削除する（テスト用）"""
        if not self.conn:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM positions")
            self.conn.commit()
            self.logger.info("すべてのポジションを削除しました")
        except sqlite3.Error as e:
            self.logger.error(f"ポジション削除エラー: {e}")

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        """データベースの行をPositionオブジェクトに変換"""
        return Position(
            symbol = row['symbol'],
            name = row['name'],
            entry_price = row['entry_price'],
            quantity = row['quantity'],
            entry_time = datetime.fromisoformat(row['entry_time']),
            stop_loss = row['stop_loss'],
            take_profit = row['take_profit'],
            current_price = row['current_price'],
            status = PositionStatus(row['status']),
            pnl = row['pnl'],
            pnl_percent = row['pnl_percent'],
            risk_level = RiskLevel(row['risk_level']),
            max_holding_time = row['max_holding_time']
        )

    def close(self) -> None:
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
            self.logger.info("データベース接続を閉じました")