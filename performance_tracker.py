#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Tracker - 包括的パフォーマンス追跡システム

実戦デイトレード向け総合パフォーマンス分析・追跡
ポートフォリオ管理・リスク分析・収益最適化統合
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import statistics
from collections import defaultdict, deque
import uuid

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select # For async operations

Base = declarative_base()

class DBTrade(Base):
    __tablename__ = 'trades'
    trade_id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    name = Column(String)
    trade_type = Column(String)
    entry_date = Column(DateTime, nullable=False)
    entry_price = Column(Float)
    quantity = Column(Integer)
    entry_amount = Column(Float)
    exit_date = Column(DateTime)
    exit_price = Column(Float)
    exit_amount = Column(Float)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    trade_result = Column(String)
    risk_level = Column(String)
    confidence_score = Column(Float)
    sector = Column(String)
    theme = Column(String)
    trading_session = Column(String)
    strategy_used = Column(String)
    notes = Column(String)
    created_at = Column(DateTime)

class DBPortfolio(Base):
    __tablename__ = 'portfolios'
    portfolio_id = Column(String, primary_key=True)
    portfolio_name = Column(String)
    initial_capital = Column(Float)
    current_capital = Column(Float)
    total_invested = Column(Float)
    cash_balance = Column(Float)
    total_return = Column(Float)
    daily_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    var_95 = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    last_updated = Column(DateTime)
    created_at = Column(DateTime)

class DBDailyPerformance(Base):
    __tablename__ = 'daily_performance'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    portfolio_value = Column(Float)
    daily_return = Column(Float)
    benchmark_return = Column(Float)
    trades_count = Column(Integer)
    profit_loss = Column(Float)
    created_at = Column(DateTime)



# Windows環境での文字化け対策
# Windows環境での文字化け対策 (今後、共通ユーティリティに集約予定)
# import sys
# import os
# os.environ['PYTHONIOENCODING'] = 'utf-8'

# if sys.platform == 'win32':
#     try:
#         sys.stdout.reconfigure(encoding='utf-8')
#         sys.stderr.reconfigure(encoding='utf-8')
#     except:
#         import codecs
#         sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
#         sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

try:
    from enhanced_symbol_manager import EnhancedSymbolManager
    ENHANCED_SYMBOLS_AVAILABLE = True
except ImportError:
    ENHANCED_SYMBOLS_AVAILABLE = False

try:
    from prediction_validator import PredictionValidator
    PREDICTION_VALIDATOR_AVAILABLE = True
except ImportError:
    PREDICTION_VALIDATOR_AVAILABLE = False

class TradeType(Enum):
    """取引タイプ"""
    BUY = "買い"
    SELL = "売り"
    HOLD = "保有継続"

class TradeResult(Enum):
    """取引結果"""
    PROFIT = "利益"
    LOSS = "損失"
    BREAKEVEN = "建値"
    OPEN = "未決済"

class RiskLevel(Enum):
    """リスクレベル"""
    VERY_LOW = "超低リスク"
    LOW = "低リスク"
    MEDIUM = "中リスク"
    HIGH = "高リスク"
    VERY_HIGH = "超高リスク"

@dataclass
class Trade:
    """取引記録"""
    trade_id: str
    symbol: str
    name: str
    trade_type: TradeType
    entry_date: datetime
    entry_price: float
    quantity: int
    entry_amount: float          # 投資金額

    # 決済情報
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_amount: Optional[float] = None

    # パフォーマンス
    profit_loss: Optional[float] = None      # 損益額
    profit_loss_pct: Optional[float] = None  # 損益率(%)
    trade_result: TradeResult = TradeResult.OPEN

    # 分析情報
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confidence_score: float = 75.0           # 信頼度スコア
    sector: Optional[str] = None             # セクター
    theme: Optional[str] = None              # テーマ

    # 追加メタデータ
    trading_session: Optional[str] = None    # 取引セッション
    strategy_used: Optional[str] = None      # 使用戦略
    notes: Optional[str] = None              # 備考

@dataclass
class Portfolio:
    """ポートフォリオ情報"""
    portfolio_id: str
    portfolio_name: str
    initial_capital: float       # 初期資本
    current_capital: float       # 現在資本
    total_invested: float        # 総投資額
    cash_balance: float          # 現金残高

    # パフォーマンス指標
    total_return: float          # 総リターン(%)
    daily_return: float          # 日次リターン(%)
    volatility: float            # ボラティリティ(%)
    sharpe_ratio: float          # シャープレシオ
    max_drawdown: float          # 最大ドローダウン(%)

    # リスク指標
    var_95: float               # バリューアットリスク(95%)
    beta: float                 # ベータ値
    alpha: float                # アルファ値

    # 取引統計
    total_trades: int           # 総取引数
    winning_trades: int         # 勝ち取引数
    losing_trades: int          # 負け取引数
    win_rate: float            # 勝率(%)

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    period_start: datetime
    period_end: datetime

    # 基本指標
    total_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # リスク指標
    max_drawdown: float
    max_drawdown_duration: int   # 日数
    var_95: float
    cvar_95: float              # 条件付きVaR

    # 取引指標
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # ベンチマーク比較
    benchmark_return: float
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float

class PerformanceTracker:
    """
    包括的パフォーマンス追跡システム
    実戦デイトレード向け総合パフォーマンス分析
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("performance_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "performance.db"

        # メモリキャッシュ
        self.active_trades: Dict[str, Trade] = {}
        self.portfolio_cache: Dict[str, Portfolio] = {}
        self.performance_history: deque = deque(maxlen=365)  # 1年間

        # デフォルトポートフォリオ作成
        self.default_portfolio_id = "MAIN_PORTFOLIO"

        # ベンチマーク設定（TOPIX想定）
        self.benchmark_return = 0.08  # 年率8%

        # 外部システム統合
        if ENHANCED_SYMBOLS_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()

        if PREDICTION_VALIDATOR_AVAILABLE:
            self.prediction_validator = PredictionValidator()

        self.logger.info("Performance tracker initialized for comprehensive analysis")

    async def ainit(self):
        # データベース初期化
        await self._init_database()
        # デフォルトポートフォリオ作成
        await self._ensure_default_portfolio()

    async def _init_database(self):
            """データベース初期化"""
            async with aiosqlite.connect(self.db_path) as conn:
                # SQLiteのPRAGMA foreign_keysを有効にする (ORM使用時に重要)
                await conn.execute("PRAGMA foreign_keys = ON;")

                # SQLAlchemyモデルを使用してテーブルを作成
                # aiosqliteは同期的なSQLAlchemyエンジンと直接統合できないため、
                # 少し異なるアプローチが必要。ここでは、aiosqliteの接続を直接使う。
                # Alembicなどのマイグレーションツールを導入すれば、この部分はよりクリーンになる。

                # ここでは、手動で各テーブルが存在するか確認し、なければ作成する。
                # これは暫定的な措置であり、イシューの「データベースマイグレーションツール」のタスクで改善予定。
                for table_name, create_sql in [
                    ("trades", DBTrade.__table__.create(engine=None, checkfirst=True) if not await self._table_exists(conn, "trades") else None),
                    ("portfolios", DBPortfolio.__table__.create(engine=None, checkfirst=True) if not await self._table_exists(conn, "portfolios") else None),
                    ("daily_performance", DBDailyPerformance.__table__.create(engine=None, checkfirst=True) if not await self._table_exists(conn, "daily_performance") else None),
                ]:
                    if create_sql:
                        # create_all は engine が必要なので、ここでは raw SQL で作成
                        # 理想的には、ここでは Base.metadata.create_all(engine) を使うべきだが、
                        # aiosqlite と同期 SQLAlchemy engine の直接統合が複雑なため、
                        # マイグレーションツール導入までraw SQLを使う。
                        # このコードはあくまで一時的な代替案
                        if table_name == "trades":
                            await conn.execute("""
                                CREATE TABLE IF NOT EXISTS trades (
                                    trade_id TEXT PRIMARY KEY,
                                    symbol TEXT NOT NULL,
                                    name TEXT,
                                    trade_type TEXT,
                                    entry_date TEXT NOT NULL,
                                    entry_price REAL,
                                    quantity INTEGER,
                                    entry_amount REAL,
                                    exit_date TEXT,
                                    exit_price REAL,
                                    exit_amount REAL,
                                    profit_loss REAL,
                                    profit_loss_pct REAL,
                                    trade_result TEXT,
                                    risk_level TEXT,
                                    confidence_score REAL,
                                    sector TEXT,
                                    theme TEXT,
                                    trading_session TEXT,
                                    strategy_used TEXT,
                                    notes TEXT,
                                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                                )
                            """)
                        elif table_name == "portfolios":
                            await conn.execute("""
                                CREATE TABLE IF NOT EXISTS portfolios (
                                    portfolio_id TEXT PRIMARY KEY,
                                    portfolio_name TEXT,
                                    initial_capital REAL,
                                    current_capital REAL,
                                    total_invested REAL,
                                    cash_balance REAL,
                                    total_return REAL,
                                    daily_return REAL,
                                    volatility REAL,
                                    sharpe_ratio REAL,
                                    max_drawdown REAL,
                                    var_95 REAL,
                                    beta REAL,
                                    alpha REAL,
                                    total_trades INTEGER,
                                    winning_trades INTEGER,
                                    losing_trades INTEGER,
                                    win_rate REAL,
                                    last_updated TEXT,
                                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                                )
                            """)
                        elif table_name == "daily_performance":
                            await conn.execute("""
                                CREATE TABLE IF NOT EXISTS daily_performance (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    date TEXT NOT NULL,
                                    portfolio_value REAL,
                                    daily_return REAL,
                                    benchmark_return REAL,
                                    trades_count INTEGER,
                                    profit_loss REAL,
                                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                                )
                            """)

                # インデックス作成
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(entry_date)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_performance(date)")

        async def _table_exists(self, conn, table_name):

    async def _ensure_default_portfolio(self):
        """デフォルトポートフォリオ確保"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("""
                    SELECT * FROM portfolios WHERE portfolio_id = ?
                """, (self.default_portfolio_id,))

                row = await cursor.fetchone()
                if not row:
                    # デフォルトポートフォリオ作成
                    default_portfolio = Portfolio(
                        portfolio_id=self.default_portfolio_id,
                        portfolio_name="メインポートフォリオ",
                        initial_capital=1000000.0,  # 100万円
                        current_capital=1000000.0,
                        total_invested=0.0,
                        cash_balance=1000000.0,
                        total_return=0.0,
                        daily_return=0.0,
                        volatility=0.0,
                        sharpe_ratio=0.0,
                        max_drawdown=0.0,
                        var_95=0.0,
                        beta=1.0,
                        alpha=0.0,
                        total_trades=0,
                        winning_trades=0,
                        losing_trades=0,
                        win_rate=0.0
                    )

                    # ポートフォリオ保存
                    await self._save_portfolio(default_portfolio)
                    self.logger.info("Created default portfolio with 1,000,000 yen")

        except Exception as e:
            self.logger.error(f"Failed to ensure default portfolio: {e}")

    async def record_trade(self, trade: Trade) -> bool:
        """
        取引記録

        Args:
            trade: 取引データ

        Returns:
            記録成功フラグ
        """
        try:
            # データベースに保存
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    INSERT OR REPLACE INTO trades (
                        trade_id, symbol, name, trade_type, entry_date,
                        entry_price, quantity, entry_amount, exit_date, exit_price,
                        exit_amount, profit_loss, profit_loss_pct, trade_result,
                        risk_level, confidence_score, sector, theme,
                        trading_session, strategy_used, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.trade_id, trade.symbol, trade.name, trade.trade_type.value,
                    trade.entry_date.isoformat(), trade.entry_price, trade.quantity,
                    trade.entry_amount, trade.exit_date.isoformat() if trade.exit_date else None,
                    trade.exit_price, trade.exit_amount, trade.profit_loss,
                    trade.profit_loss_pct, trade.trade_result.value, trade.risk_level.value,
                    trade.confidence_score, trade.sector, trade.theme,
                    trade.trading_session, trade.strategy_used, trade.notes
                ))

            # メモリキャッシュ更新
            if trade.trade_result == TradeResult.OPEN:
                self.active_trades[trade.trade_id] = trade
            elif trade.trade_id in self.active_trades:
                del self.active_trades[trade.trade_id]

            # ポートフォリオ更新
            await self._update_portfolio_from_trade(trade)

            self.logger.info(f"Recorded trade: {trade.symbol} {trade.trade_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to record trade: {e}")
            return False

    async def _update_portfolio_from_trade(self, trade: Trade):
        """取引からポートフォリオ更新"""
        try:
            portfolio = await self.get_portfolio(self.default_portfolio_id)
            if not portfolio:
                return

            if trade.trade_result != TradeResult.OPEN:
                # 決済済み取引の場合
                portfolio.total_trades += 1

                if trade.profit_loss and trade.profit_loss > 0:
                    portfolio.winning_trades += 1
                elif trade.profit_loss and trade.profit_loss < 0:
                    portfolio.losing_trades += 1

                # 勝率計算
                if portfolio.total_trades > 0:
                    portfolio.win_rate = portfolio.winning_trades / portfolio.total_trades * 100

                # 資本更新
                if trade.profit_loss:
                    portfolio.current_capital += trade.profit_loss
                    portfolio.total_return = (portfolio.current_capital - portfolio.initial_capital) / portfolio.initial_capital * 100

            # データベース更新
            await self._save_portfolio(portfolio)

        except Exception as e:
            self.logger.error(f"Failed to update portfolio from trade: {e}")

    async def get_portfolio(self, portfolio_id: str = None) -> Optional[Portfolio]:
        """ポートフォリオ取得"""
        if portfolio_id is None:
            portfolio_id = self.default_portfolio_id

        # キャッシュチェック
        if portfolio_id in self.portfolio_cache:
            return self.portfolio_cache[portfolio_id]

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("""
                    SELECT * FROM portfolios WHERE portfolio_id = ?
                """, (portfolio_id,))

                row = await cursor.fetchone()
                if row:
                    portfolio = Portfolio(
                        portfolio_id=row[0],
                        portfolio_name=row[1],
                        initial_capital=row[2],
                        current_capital=row[3],
                        total_invested=row[4],
                        cash_balance=row[5],
                        total_return=row[6],
                        daily_return=row[7],
                        volatility=row[8],
                        sharpe_ratio=row[9],
                        max_drawdown=row[10],
                        var_95=row[11],
                        beta=row[12],
                        alpha=row[13],
                        total_trades=row[14],
                        winning_trades=row[15],
                        losing_trades=row[16],
                        win_rate=row[17],
                        last_updated=datetime.fromisoformat(row[18]) if row[18] else datetime.now()
                    )

                    # キャッシュ更新
                    self.portfolio_cache[portfolio_id] = portfolio
                    return portfolio

        except Exception as e:
            self.logger.error(f"Failed to get portfolio: {e}")

        return None

    async def _save_portfolio(self, portfolio: Portfolio):
        """ポートフォリオ保存"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    INSERT OR REPLACE INTO portfolios (
                        portfolio_id, portfolio_name, initial_capital, current_capital,
                        total_invested, cash_balance, total_return, daily_return,
                        volatility, sharpe_ratio, max_drawdown, var_95, beta, alpha,
                        total_trades, winning_trades, losing_trades, win_rate, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    portfolio.portfolio_id, portfolio.portfolio_name,
                    portfolio.initial_capital, portfolio.current_capital,
                    portfolio.total_invested, portfolio.cash_balance,
                    portfolio.total_return, portfolio.daily_return,
                    portfolio.volatility, portfolio.sharpe_ratio,
                    portfolio.max_drawdown, portfolio.var_95,
                    portfolio.beta, portfolio.alpha,
                    portfolio.total_trades, portfolio.winning_trades,
                    portfolio.losing_trades, portfolio.win_rate,
                    portfolio.last_updated.isoformat()
                ))

            # キャッシュ更新
            self.portfolio_cache[portfolio.portfolio_id] = portfolio

        except Exception as e:
            self.logger.error(f"Failed to save portfolio: {e}")



    async def calculate_performance_metrics(self, days_back: int = 30) -> PerformanceMetrics:
        """
        パフォーマンス指標計算

        Args:
            days_back: 対象期間（日数）

        Returns:
            パフォーマンス指標
        """
        try:
            period_start = datetime.now() - timedelta(days=days_back)
            period_end = datetime.now()

            # 取引データ取得
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("""
                    SELECT * FROM trades
                    WHERE entry_date >= ? AND trade_result != 'OPEN'
                    ORDER BY entry_date
                """, (period_start.isoformat(),))

                trades_data = await cursor.fetchall()

            if not trades_data:
                # データがない場合のデフォルト
                return PerformanceMetrics(
                    period_start=period_start, period_end=period_end,
                    total_return_pct=0.0, annualized_return=0.0, volatility=0.0,
                    sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                    max_drawdown=0.0, max_drawdown_duration=0, var_95=0.0, cvar_95=0.0,
                    total_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                    profit_factor=0.0, benchmark_return=self.benchmark_return,
                    alpha=0.0, beta=1.0, tracking_error=0.0, information_ratio=0.0
                )

            # 基本統計計算
            total_trades = len(trades_data)
            returns = [row[11] for row in trades_data if row[11] is not None]  # profit_loss_pct

            if not returns:
                returns = [0.0]

            # リターン統計
            total_return_pct = sum(returns)
            annualized_return = total_return_pct * (365 / days_back) if days_back > 0 else 0
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0

            # シャープレシオ（リスクフリーレート0.5%想定）
            risk_free_rate = 0.5
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0

            # ソルティノレシオ
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = statistics.stdev(negative_returns) if len(negative_returns) > 1 else volatility
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0

            # 最大ドローダウン
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0

            # カルマーレシオ
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

            # VaR計算
            var_95 = np.percentile(returns, 5) if returns else 0.0
            cvar_95 = np.mean([r for r in returns if r <= var_95]) if returns else 0.0

            # 取引統計
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]

            win_rate = len(winning_trades) / len(returns) * 100 if returns else 0.0
            avg_win = statistics.mean(winning_trades) if winning_trades else 0.0
            avg_loss = statistics.mean(losing_trades) if losing_trades else 0.0

            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            # ベンチマーク比較
            benchmark_return = self.benchmark_return * (days_back / 365)
            alpha = annualized_return - self.benchmark_return
            beta = 1.0  # 簡易実装

            tracking_error = abs(annualized_return - self.benchmark_return)
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0

            return PerformanceMetrics(
                period_start=period_start,
                period_end=period_end,
                total_return_pct=total_return_pct,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=0,  # 簡易実装
                var_95=var_95,
                cvar_95=cvar_95,
                total_trades=total_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                benchmark_return=benchmark_return,
                alpha=alpha,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio
            )

        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return PerformanceMetrics(
                period_start=datetime.now() - timedelta(days=days_back),
                period_end=datetime.now(),
                total_return_pct=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                max_drawdown=0.0, max_drawdown_duration=0, var_95=0.0, cvar_95=0.0,
                total_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                profit_factor=0.0, benchmark_return=0.0,
                alpha=0.0, beta=1.0, tracking_error=0.0, information_ratio=0.0
            )

    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """包括的パフォーマンスレポート生成"""

        try:
            # 現在のポートフォリオ
            portfolio = await self.get_portfolio()

            # パフォーマンス指標（30日・90日）
            metrics_30d = await self.calculate_performance_metrics(30)
            metrics_90d = await self.calculate_performance_metrics(90)

            # セクター・テーマ分析
            sector_performance = await self._analyze_sector_performance()
            theme_performance = await self._analyze_theme_performance()

            # リスク分析
            risk_analysis = await self._analyze_risk_profile()

            return {
                "portfolio_summary": {
                    "portfolio_name": portfolio.portfolio_name if portfolio else "N/A",
                    "initial_capital": portfolio.initial_capital if portfolio else 0,
                    "current_capital": portfolio.current_capital if portfolio else 0,
                    "total_return": portfolio.total_return if portfolio else 0,
                    "cash_balance": portfolio.cash_balance if portfolio else 0,
                    "total_trades": portfolio.total_trades if portfolio else 0,
                    "win_rate": portfolio.win_rate if portfolio else 0
                },
                "performance_metrics": {
                    "30_days": {
                        "total_return": metrics_30d.total_return_pct,
                        "annualized_return": metrics_30d.annualized_return,
                        "volatility": metrics_30d.volatility,
                        "sharpe_ratio": metrics_30d.sharpe_ratio,
                        "max_drawdown": metrics_30d.max_drawdown,
                        "win_rate": metrics_30d.win_rate,
                        "profit_factor": metrics_30d.profit_factor
                    },
                    "90_days": {
                        "total_return": metrics_90d.total_return_pct,
                        "annualized_return": metrics_90d.annualized_return,
                        "volatility": metrics_90d.volatility,
                        "sharpe_ratio": metrics_90d.sharpe_ratio,
                        "max_drawdown": metrics_90d.max_drawdown,
                        "win_rate": metrics_90d.win_rate,
                        "profit_factor": metrics_90d.profit_factor
                    }
                },
                "benchmark_comparison": {
                    "alpha_30d": metrics_30d.alpha,
                    "alpha_90d": metrics_90d.alpha,
                    "tracking_error_30d": metrics_30d.tracking_error,
                    "information_ratio_30d": metrics_30d.information_ratio
                },
                "sector_performance": sector_performance,
                "theme_performance": theme_performance,
                "risk_analysis": risk_analysis,
                "system_status": {
                    "active_trades": len(self.active_trades),
                    "tracking_status": "正常稼働",
                    "last_updated": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return {"error": f"Report generation failed: {e}"}

    async def _analyze_sector_performance(self) -> Dict[str, Any]:
        """セクター別パフォーマンス分析"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("""
                    SELECT sector, COUNT(*) as trades,
                           AVG(profit_loss_pct) as avg_return,
                           SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
                    FROM trades
                    WHERE sector IS NOT NULL AND trade_result != 'OPEN'
                    GROUP BY sector
                    ORDER BY avg_return DESC
                """)

                sectors = await cursor.fetchall()

                sector_analysis = {}
                for sector, trades, avg_return, win_rate in sectors:
                    sector_analysis[sector] = {
                        "trades_count": trades,
                        "avg_return": avg_return or 0,
                        "win_rate": win_rate or 0
                    }

                return sector_analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze sector performance: {e}")
            return {}

    async def _analyze_theme_performance(self) -> Dict[str, Any]:
        """テーマ別パフォーマンス分析"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("""
                    SELECT theme, COUNT(*) as trades,
                           AVG(profit_loss_pct) as avg_return,
                           SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
                    FROM trades
                    WHERE theme IS NOT NULL AND trade_result != 'OPEN'
                    GROUP BY theme
                    ORDER BY avg_return DESC
                """)

                themes = await cursor.fetchall()

                theme_analysis = {}
                for theme, trades, avg_return, win_rate in themes:
                    theme_analysis[theme] = {
                        "trades_count": trades,
                        "avg_return": avg_return or 0,
                        "win_rate": win_rate or 0
                    }

                return theme_analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze theme performance: {e}")
            return {}

    async def _analyze_risk_profile(self) -> Dict[str, Any]:
        """リスクプロファイル分析"""
        try:
            portfolio = await self.get_portfolio()
            if not portfolio:
                return {}

            # 基本リスク指標
            risk_profile = {
                "portfolio_volatility": portfolio.volatility,
                "max_drawdown": portfolio.max_drawdown,
                "var_95": portfolio.var_95,
                "beta": portfolio.beta,
                "risk_level": self._classify_risk_level(portfolio)
            }

            # リスク分散度
            risk_profile["diversification_score"] = await self._calculate_diversification_score()

            # リスク提言
            risk_profile["risk_recommendations"] = self._generate_risk_recommendations(portfolio)

            return risk_profile

        except Exception as e:
            self.logger.error(f"Failed to analyze risk profile: {e}")
            return {}

    def _classify_risk_level(self, portfolio: Portfolio) -> str:
        """リスクレベル分類"""
        if portfolio.volatility < 5:
            return "低リスク"
        elif portfolio.volatility < 15:
            return "中リスク"
        else:
            return "高リスク"

    async def _calculate_diversification_score(self) -> float:
        """分散化スコア計算"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute("""
                    SELECT COUNT(DISTINCT sector) as sector_count,
                           COUNT(DISTINCT theme) as theme_count,
                           COUNT(*) as total_trades
                    FROM trades
                    WHERE trade_result != 'OPEN'
                """)

                row = await cursor.fetchone()
                if row:
                    sector_count, theme_count, total_trades = row
                    # 簡易分散化スコア（セクター数・テーマ数基準）
                    diversification_score = min(100, (sector_count * 10 + theme_count * 5))
                    return diversification_score

        except Exception as e:
            self.logger.error(f"Failed to calculate diversification score: {e}")

        return 0.0

    def _generate_risk_recommendations(self, portfolio: Portfolio) -> List[str]:
        """リスク管理提言生成"""
        recommendations = []

        if portfolio.max_drawdown > 20:
            recommendations.append("最大ドローダウンが20%を超えています。ポジションサイズの見直しを推奨")

        if portfolio.volatility > 20:
            recommendations.append("ボラティリティが高い状態です。よりディフェンシブな銘柄への配分を検討")

        if portfolio.win_rate < 50:
            recommendations.append("勝率が50%を下回っています。エントリー基準の厳格化を推奨")

        if not recommendations:
            recommendations.append("現在のリスク管理は良好です")

        return recommendations

# テスト関数
