#!/usr/bin/env python3
"""
プロダクション監視ダッシュボード コア機能

Issue #324: プロダクション運用監視ダッシュボード構築
リアルタイム運用状況監視システム
"""

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class PortfolioSnapshot:
    """ポートフォリオスナップショット"""

    timestamp: str
    total_value: float
    cash: float
    positions: Dict[str, Dict[str, Any]]
    daily_return: float
    cumulative_return: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class SystemMetrics:
    """システムメトリクス"""

    timestamp: str
    cpu_usage: float
    memory_usage_mb: float
    disk_usage_gb: float
    active_threads: int
    processing_time_ms: float
    error_count: int
    warning_count: int


@dataclass
class TradingMetrics:
    """取引メトリクス"""

    timestamp: str
    trades_today: int
    successful_trades: int
    failed_trades: int
    average_execution_time_ms: float
    total_commission: float
    total_slippage: float
    win_rate: float


@dataclass
class RiskMetrics:
    """リスクメトリクス"""

    timestamp: str
    current_drawdown: float
    max_drawdown: float
    portfolio_var_95: float
    portfolio_volatility: float
    concentration_risk: float
    leverage_ratio: float
    beta_to_market: float


class ProductionDashboard:
    """プロダクション監視ダッシュボード"""

    def __init__(self, data_retention_days: int = 30):
        """
        初期化

        Args:
            data_retention_days: データ保持日数
        """
        self.data_retention_days = data_retention_days
        self.running = False

        # データストレージ
        self.db_path = Path("production_dashboard.db")
        self._init_database()

        # 最新データキャッシュ
        self.latest_portfolio: Optional[PortfolioSnapshot] = None
        self.latest_system: Optional[SystemMetrics] = None
        self.latest_trading: Optional[TradingMetrics] = None
        self.latest_risk: Optional[RiskMetrics] = None

        # 更新間隔設定
        self.update_intervals = {
            "portfolio": 10,  # 10秒
            "system": 5,  # 5秒
            "trading": 30,  # 30秒
            "risk": 60,  # 60秒
        }

        # アラート設定
        self.alert_thresholds = {
            "max_drawdown": -0.10,  # 10%ドローダウン
            "cpu_usage": 80,  # CPU使用率80%
            "memory_usage": 2048,  # メモリ使用量2GB
            "error_rate": 0.05,  # エラー率5%
            "processing_time": 1000,  # 処理時間1秒
        }

        print("プロダクション監視ダッシュボード初期化完了")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # ポートフォリオデータテーブル
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT PRIMARY KEY,
                    total_value REAL,
                    cash REAL,
                    positions TEXT,
                    daily_return REAL,
                    cumulative_return REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL
                )
            """
            )

            # システムメトリクステーブル
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TEXT PRIMARY KEY,
                    cpu_usage REAL,
                    memory_usage_mb REAL,
                    disk_usage_gb REAL,
                    active_threads INTEGER,
                    processing_time_ms REAL,
                    error_count INTEGER,
                    warning_count INTEGER
                )
            """
            )

            # 取引メトリクステーブル
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    timestamp TEXT PRIMARY KEY,
                    trades_today INTEGER,
                    successful_trades INTEGER,
                    failed_trades INTEGER,
                    average_execution_time_ms REAL,
                    total_commission REAL,
                    total_slippage REAL,
                    win_rate REAL
                )
            """
            )

            # リスクメトリクステーブル
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    timestamp TEXT PRIMARY KEY,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    portfolio_var_95 REAL,
                    portfolio_volatility REAL,
                    concentration_risk REAL,
                    leverage_ratio REAL,
                    beta_to_market REAL
                )
            """
            )

            conn.commit()

    def start_monitoring(self):
        """監視開始"""
        print("プロダクション監視開始")
        self.running = True

        # 各メトリクス更新スレッド開始
        threads = [
            threading.Thread(target=self._portfolio_monitor_loop, daemon=True),
            threading.Thread(target=self._system_monitor_loop, daemon=True),
            threading.Thread(target=self._trading_monitor_loop, daemon=True),
            threading.Thread(target=self._risk_monitor_loop, daemon=True),
            threading.Thread(target=self._cleanup_loop, daemon=True),
        ]

        for thread in threads:
            thread.start()

        print("全監視スレッド開始完了")

    def stop_monitoring(self):
        """監視停止"""
        print("プロダクション監視停止")
        self.running = False

    def _portfolio_monitor_loop(self):
        """ポートフォリオ監視ループ"""
        while self.running:
            try:
                snapshot = self._collect_portfolio_snapshot()
                self._save_portfolio_snapshot(snapshot)
                self.latest_portfolio = snapshot

                # アラートチェック
                self._check_portfolio_alerts(snapshot)

            except Exception as e:
                print(f"ポートフォリオ監視エラー: {e}")

            time.sleep(self.update_intervals["portfolio"])

    def _system_monitor_loop(self):
        """システム監視ループ"""
        while self.running:
            try:
                metrics = self._collect_system_metrics()
                self._save_system_metrics(metrics)
                self.latest_system = metrics

                # アラートチェック
                self._check_system_alerts(metrics)

            except Exception as e:
                print(f"システム監視エラー: {e}")

            time.sleep(self.update_intervals["system"])

    def _trading_monitor_loop(self):
        """取引監視ループ"""
        while self.running:
            try:
                metrics = self._collect_trading_metrics()
                self._save_trading_metrics(metrics)
                self.latest_trading = metrics

                # アラートチェック
                self._check_trading_alerts(metrics)

            except Exception as e:
                print(f"取引監視エラー: {e}")

            time.sleep(self.update_intervals["trading"])

    def _risk_monitor_loop(self):
        """リスク監視ループ"""
        while self.running:
            try:
                metrics = self._collect_risk_metrics()
                self._save_risk_metrics(metrics)
                self.latest_risk = metrics

                # アラートチェック
                self._check_risk_alerts(metrics)

            except Exception as e:
                print(f"リスク監視エラー: {e}")

            time.sleep(self.update_intervals["risk"])

    def _cleanup_loop(self):
        """データクリーンアップループ"""
        while self.running:
            try:
                self._cleanup_old_data()
            except Exception as e:
                print(f"データクリーンアップエラー: {e}")

            # 1時間ごとに実行
            time.sleep(3600)

    def _collect_portfolio_snapshot(self) -> PortfolioSnapshot:
        """ポートフォリオスナップショット収集"""
        # 実際の実装では、ポートフォリオマネージャーからデータを取得
        # ここではデモ用のサンプルデータ

        import random

        current_time = datetime.now()
        base_value = 1000000  # 100万円ベース

        # サンプルデータ生成
        daily_change = random.uniform(-0.03, 0.03)  # ±3%の日次変動
        total_value = base_value * (1 + daily_change)

        positions = {
            "7203.T": {
                "quantity": 100,
                "price": 2500,
                "value": 250000,
                "pnl": random.uniform(-10000, 15000),
            },
            "8306.T": {
                "quantity": 1000,
                "price": 800,
                "value": 800000,
                "pnl": random.uniform(-20000, 30000),
            },
            "9984.T": {
                "quantity": 50,
                "price": 5000,
                "value": 250000,
                "pnl": random.uniform(-15000, 20000),
            },
        }

        cash = total_value - sum(pos["value"] for pos in positions.values())
        unrealized_pnl = sum(pos["pnl"] for pos in positions.values())

        return PortfolioSnapshot(
            timestamp=current_time.isoformat(),
            total_value=total_value,
            cash=cash,
            positions=positions,
            daily_return=daily_change,
            cumulative_return=random.uniform(-0.05, 0.15),  # ±5%-15%累積リターン
            unrealized_pnl=unrealized_pnl,
            realized_pnl=random.uniform(-5000, 10000),
        )

    def _collect_system_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""
        import random

        current_time = datetime.now()

        # システムリソース情報取得
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # プロセス情報
        process = psutil.Process()
        active_threads = process.num_threads()

        return SystemMetrics(
            timestamp=current_time.isoformat(),
            cpu_usage=cpu_usage,
            memory_usage_mb=memory.used / (1024 * 1024),
            disk_usage_gb=disk.used / (1024 * 1024 * 1024),
            active_threads=active_threads,
            processing_time_ms=random.uniform(50, 200),  # デモ用
            error_count=random.randint(0, 2),
            warning_count=random.randint(0, 5),
        )

    def _collect_trading_metrics(self) -> TradingMetrics:
        """取引メトリクス収集"""
        import random

        current_time = datetime.now()

        # デモ用取引データ
        trades_today = random.randint(5, 20)
        successful_trades = random.randint(int(trades_today * 0.7), trades_today)
        failed_trades = trades_today - successful_trades

        return TradingMetrics(
            timestamp=current_time.isoformat(),
            trades_today=trades_today,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            average_execution_time_ms=random.uniform(100, 500),
            total_commission=random.uniform(1000, 5000),
            total_slippage=random.uniform(500, 2000),
            win_rate=successful_trades / trades_today if trades_today > 0 else 0,
        )

    def _collect_risk_metrics(self) -> RiskMetrics:
        """リスクメトリクス収集"""
        import random

        current_time = datetime.now()

        return RiskMetrics(
            timestamp=current_time.isoformat(),
            current_drawdown=random.uniform(-0.08, 0),
            max_drawdown=random.uniform(-0.15, -0.05),
            portfolio_var_95=random.uniform(-0.05, -0.02),
            portfolio_volatility=random.uniform(0.12, 0.25),
            concentration_risk=random.uniform(0.3, 0.8),
            leverage_ratio=random.uniform(0.8, 1.2),
            beta_to_market=random.uniform(0.7, 1.3),
        )

    def _save_portfolio_snapshot(self, snapshot: PortfolioSnapshot):
        """ポートフォリオスナップショット保存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO portfolio_snapshots
                (timestamp, total_value, cash, positions, daily_return, cumulative_return, unrealized_pnl, realized_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    snapshot.timestamp,
                    snapshot.total_value,
                    snapshot.cash,
                    json.dumps(snapshot.positions),
                    snapshot.daily_return,
                    snapshot.cumulative_return,
                    snapshot.unrealized_pnl,
                    snapshot.realized_pnl,
                ),
            )
            conn.commit()

    def _save_system_metrics(self, metrics: SystemMetrics):
        """システムメトリクス保存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO system_metrics
                (timestamp, cpu_usage, memory_usage_mb, disk_usage_gb, active_threads, processing_time_ms, error_count, warning_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.timestamp,
                    metrics.cpu_usage,
                    metrics.memory_usage_mb,
                    metrics.disk_usage_gb,
                    metrics.active_threads,
                    metrics.processing_time_ms,
                    metrics.error_count,
                    metrics.warning_count,
                ),
            )
            conn.commit()

    def _save_trading_metrics(self, metrics: TradingMetrics):
        """取引メトリクス保存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO trading_metrics
                (timestamp, trades_today, successful_trades, failed_trades, average_execution_time_ms, total_commission, total_slippage, win_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.timestamp,
                    metrics.trades_today,
                    metrics.successful_trades,
                    metrics.failed_trades,
                    metrics.average_execution_time_ms,
                    metrics.total_commission,
                    metrics.total_slippage,
                    metrics.win_rate,
                ),
            )
            conn.commit()

    def _save_risk_metrics(self, metrics: RiskMetrics):
        """リスクメトリクス保存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO risk_metrics
                (timestamp, current_drawdown, max_drawdown, portfolio_var_95, portfolio_volatility, concentration_risk, leverage_ratio, beta_to_market)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.timestamp,
                    metrics.current_drawdown,
                    metrics.max_drawdown,
                    metrics.portfolio_var_95,
                    metrics.portfolio_volatility,
                    metrics.concentration_risk,
                    metrics.leverage_ratio,
                    metrics.beta_to_market,
                ),
            )
            conn.commit()

    def _check_portfolio_alerts(self, snapshot: PortfolioSnapshot):
        """ポートフォリオアラートチェック"""
        alerts = []

        if snapshot.daily_return < -0.05:  # 5%以上の下落
            alerts.append(f"大幅下落警告: 日次リターン {snapshot.daily_return:.2%}")

        if snapshot.cash < 50000:  # 現金残高5万円未満
            alerts.append(f"現金残高警告: {snapshot.cash:,.0f}円")

        if alerts:
            self._send_alert("Portfolio", alerts)

    def _check_system_alerts(self, metrics: SystemMetrics):
        """システムアラートチェック"""
        alerts = []

        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append(f"高CPU使用率: {metrics.cpu_usage:.1f}%")

        if metrics.memory_usage_mb > self.alert_thresholds["memory_usage"]:
            alerts.append(f"高メモリ使用量: {metrics.memory_usage_mb:.0f}MB")

        if metrics.processing_time_ms > self.alert_thresholds["processing_time"]:
            alerts.append(f"処理時間遅延: {metrics.processing_time_ms:.0f}ms")

        if alerts:
            self._send_alert("System", alerts)

    def _check_trading_alerts(self, metrics: TradingMetrics):
        """取引アラートチェック"""
        alerts = []

        if metrics.win_rate < 0.5 and metrics.trades_today > 10:
            alerts.append(f"低勝率警告: {metrics.win_rate:.1%}")

        if metrics.failed_trades > 5:
            alerts.append(f"取引失敗多発: {metrics.failed_trades}回")

        if alerts:
            self._send_alert("Trading", alerts)

    def _check_risk_alerts(self, metrics: RiskMetrics):
        """リスクアラートチェック"""
        alerts = []

        if metrics.current_drawdown < self.alert_thresholds["max_drawdown"]:
            alerts.append(f"大幅ドローダウン: {metrics.current_drawdown:.2%}")

        if metrics.concentration_risk > 0.8:
            alerts.append(f"高集中リスク: {metrics.concentration_risk:.1%}")

        if alerts:
            self._send_alert("Risk", alerts)

    def _send_alert(self, category: str, messages: List[str]):
        """アラート送信"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[ALERT {timestamp}] {category}:")
        for message in messages:
            print(f"  - {message}")

    def _cleanup_old_data(self):
        """古いデータクリーンアップ"""
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        cutoff_str = cutoff_date.isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            tables = [
                "portfolio_snapshots",
                "system_metrics",
                "trading_metrics",
                "risk_metrics",
            ]
            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_str,))

            conn.commit()

    def get_current_status(self) -> Dict[str, Any]:
        """現在のステータス取得"""
        return {
            "portfolio": asdict(self.latest_portfolio) if self.latest_portfolio else None,
            "system": asdict(self.latest_system) if self.latest_system else None,
            "trading": asdict(self.latest_trading) if self.latest_trading else None,
            "risk": asdict(self.latest_risk) if self.latest_risk else None,
            "monitoring_active": self.running,
            "last_updated": datetime.now().isoformat(),
        }

    def get_historical_data(self, metric_type: str, hours: int = 24) -> List[Dict[str, Any]]:
        """過去データ取得"""
        start_time = datetime.now() - timedelta(hours=hours)
        start_str = start_time.isoformat()

        table_map = {
            "portfolio": "portfolio_snapshots",
            "system": "system_metrics",
            "trading": "trading_metrics",
            "risk": "risk_metrics",
        }

        if metric_type not in table_map:
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # 辞書形式で結果取得
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT * FROM {table_map[metric_type]}
                WHERE timestamp >= ?
                ORDER BY timestamp
            """,
                (start_str,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def generate_status_report(self) -> str:
        """ステータスレポート生成"""
        status = self.get_current_status()

        report = []
        report.append("=" * 60)
        report.append("プロダクション運用ステータス")
        report.append("=" * 60)
        report.append(f"更新時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ポートフォリオ状況
        if status["portfolio"]:
            p = status["portfolio"]
            report.append("\n【ポートフォリオ】")
            report.append(f"  総資産: {p['total_value']:,.0f}円")
            report.append(f"  現金: {p['cash']:,.0f}円")
            report.append(f"  日次リターン: {p['daily_return']:.2%}")
            report.append(f"  累積リターン: {p['cumulative_return']:.2%}")
            report.append(f"  含み損益: {p['unrealized_pnl']:,.0f}円")

        # システム状況
        if status["system"]:
            s = status["system"]
            report.append("\n【システム】")
            report.append(f"  CPU使用率: {s['cpu_usage']:.1f}%")
            report.append(f"  メモリ使用量: {s['memory_usage_mb']:.0f}MB")
            report.append(f"  アクティブスレッド: {s['active_threads']}個")
            report.append(f"  処理時間: {s['processing_time_ms']:.0f}ms")

        # 取引状況
        if status["trading"]:
            t = status["trading"]
            report.append("\n【取引】")
            report.append(f"  本日取引数: {t['trades_today']}回")
            report.append(f"  成功: {t['successful_trades']}回")
            report.append(f"  失敗: {t['failed_trades']}回")
            report.append(f"  勝率: {t['win_rate']:.1%}")

        # リスク状況
        if status["risk"]:
            r = status["risk"]
            report.append("\n【リスク】")
            report.append(f"  現在ドローダウン: {r['current_drawdown']:.2%}")
            report.append(f"  最大ドローダウン: {r['max_drawdown']:.2%}")
            report.append(f"  VaR(95%): {r['portfolio_var_95']:.2%}")
            report.append(f"  ボラティリティ: {r['portfolio_volatility']:.2%}")

        report.append(f"\n監視状態: {'稼働中' if status['monitoring_active'] else '停止中'}")

        return "\n".join(report)


if __name__ == "__main__":
    # テスト実行
    print("プロダクション監視ダッシュボードテスト")
    print("=" * 50)

    dashboard = ProductionDashboard()

    try:
        # 監視開始
        dashboard.start_monitoring()

        print("監視開始 - 30秒間テスト実行")

        # 30秒間監視
        for i in range(6):
            time.sleep(5)
            print(f"\n--- {(i+1)*5}秒経過 ---")
            print(dashboard.generate_status_report())

        # 監視停止
        dashboard.stop_monitoring()
        print("\n監視停止")

        # 履歴データ確認
        portfolio_history = dashboard.get_historical_data("portfolio", hours=1)
        print(f"\nポートフォリオ履歴: {len(portfolio_history)}件")

        print("\nダッシュボードテスト完了")

    except KeyboardInterrupt:
        print("\nユーザー中断")
        dashboard.stop_monitoring()
    except Exception as e:
        print(f"テストエラー: {e}")
        dashboard.stop_monitoring()
