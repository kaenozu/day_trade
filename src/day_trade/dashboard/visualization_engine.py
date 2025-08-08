#!/usr/bin/env python3
"""
ダッシュボード可視化エンジン

Issue #324: プロダクション運用監視ダッシュボード構築
リアルタイムチャート・グラフ生成機能
"""

import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8")
# 日本語フォント設定（Windows環境対応）
try:
    import matplotlib.font_manager as fm

    # システムに日本語フォントが存在するかチェック
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    japanese_fonts = ["MS Gothic", "Yu Gothic", "Meiryo", "DejaVu Sans"]
    for font in japanese_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = [font]
            break
    else:
        # 日本語フォントが見つからない場合は英語表記に変更
        plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
        print("日本語フォントが見つからないため、英語表記を使用します")
except Exception as e:
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    print(f"フォント設定エラー: {e}")

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止


class DashboardVisualizationEngine:
    """ダッシュボード可視化エンジン"""

    def __init__(self, output_dir: str = "dashboard_charts"):
        """
        初期化

        Args:
            output_dir: チャート出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # カラーパレット設定
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#ff9800",
            "danger": "#f44336",
            "info": "#17a2b8",
            "profit": "#4caf50",
            "loss": "#f44336",
            "neutral": "#9e9e9e",
        }

        print("ダッシュボード可視化エンジン初期化完了")

    def create_portfolio_value_chart(self, data: List[Dict[str, Any]]) -> str:
        """ポートフォリオ価値推移チャート作成"""
        if not data:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        # データ準備
        timestamps = [datetime.fromisoformat(item["timestamp"]) for item in data]
        values = [item["total_value"] for item in data]

        # メインチャート
        ax.plot(
            timestamps,
            values,
            color=self.colors["primary"],
            linewidth=2,
            label="ポートフォリオ価値",
        )

        # トレンドライン
        if len(values) > 10:
            trend_values = np.polyval(
                np.polyfit(range(len(values)), values, 1), range(len(values))
            )
            ax.plot(
                timestamps,
                trend_values,
                "--",
                color=self.colors["secondary"],
                alpha=0.7,
                label="トレンド",
            )

        ax.set_title("Portfolio Value Trend", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value (JPY)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Y軸フォーマット
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}円"))

        # X軸フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # 保存
        filename = f"portfolio_value_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def create_system_metrics_chart(self, data: List[Dict[str, Any]]) -> str:
        """システムメトリクスチャート作成"""
        if not data:
            return ""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # データ準備
        timestamps = [datetime.fromisoformat(item["timestamp"]) for item in data]
        cpu_usage = [item["cpu_usage"] for item in data]
        memory_usage = [item["memory_usage_mb"] / 1024 for item in data]  # GB変換
        processing_time = [item["processing_time_ms"] for item in data]
        error_count = [item["error_count"] for item in data]

        # CPU使用率
        ax1.plot(timestamps, cpu_usage, color=self.colors["warning"], linewidth=2)
        ax1.axhline(
            y=80,
            color=self.colors["danger"],
            linestyle="--",
            alpha=0.7,
            label="警告レベル",
        )
        ax1.set_title("CPU使用率", fontweight="bold")
        ax1.set_ylabel("CPU使用率 (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # メモリ使用量
        ax2.plot(timestamps, memory_usage, color=self.colors["info"], linewidth=2)
        ax2.axhline(
            y=2,
            color=self.colors["danger"],
            linestyle="--",
            alpha=0.7,
            label="警告レベル",
        )
        ax2.set_title("メモリ使用量", fontweight="bold")
        ax2.set_ylabel("メモリ使用量 (GB)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 処理時間
        ax3.plot(
            timestamps, processing_time, color=self.colors["secondary"], linewidth=2
        )
        ax3.axhline(
            y=1000,
            color=self.colors["danger"],
            linestyle="--",
            alpha=0.7,
            label="警告レベル",
        )
        ax3.set_title("処理時間", fontweight="bold")
        ax3.set_ylabel("処理時間 (ms)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # エラー数
        ax4.bar(
            timestamps, error_count, color=self.colors["danger"], alpha=0.7, width=0.02
        )
        ax4.set_title("エラー数", fontweight="bold")
        ax4.set_ylabel("エラー数")
        ax4.grid(True, alpha=0.3)

        # X軸フォーマット統一
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle("システムメトリクス", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # 保存
        filename = f"system_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def create_trading_performance_chart(self, data: List[Dict[str, Any]]) -> str:
        """取引パフォーマンスチャート作成"""
        if not data:
            return ""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # データ準備
        timestamps = [datetime.fromisoformat(item["timestamp"]) for item in data]
        trades_today = [item["trades_today"] for item in data]
        win_rate = [item["win_rate"] * 100 for item in data]  # パーセント変換
        successful_trades = [item["successful_trades"] for item in data]
        failed_trades = [item["failed_trades"] for item in data]

        # 取引数推移
        ax1.plot(
            timestamps,
            trades_today,
            color=self.colors["primary"],
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax1.set_title("日次取引数", fontweight="bold")
        ax1.set_ylabel("取引数")
        ax1.grid(True, alpha=0.3)

        # 勝率推移
        ax2.plot(timestamps, win_rate, color=self.colors["success"], linewidth=2)
        ax2.axhline(
            y=50,
            color=self.colors["neutral"],
            linestyle="--",
            alpha=0.7,
            label="50%ライン",
        )
        ax2.set_title("勝率推移", fontweight="bold")
        ax2.set_ylabel("勝率 (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        # 成功・失敗取引数
        width = 0.01
        ax3.bar(
            [t - timedelta(minutes=30) for t in timestamps],
            successful_trades,
            width=width,
            color=self.colors["success"],
            alpha=0.7,
            label="成功",
        )
        ax3.bar(
            [t + timedelta(minutes=30) for t in timestamps],
            failed_trades,
            width=width,
            color=self.colors["danger"],
            alpha=0.7,
            label="失敗",
        )
        ax3.set_title("取引成功・失敗数", fontweight="bold")
        ax3.set_ylabel("取引数")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 累積取引数
        cumulative_trades = np.cumsum(trades_today)
        ax4.plot(timestamps, cumulative_trades, color=self.colors["info"], linewidth=2)
        ax4.fill_between(
            timestamps, cumulative_trades, alpha=0.3, color=self.colors["info"]
        )
        ax4.set_title("累積取引数", fontweight="bold")
        ax4.set_ylabel("累積取引数")
        ax4.grid(True, alpha=0.3)

        # X軸フォーマット統一
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle("取引パフォーマンス", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # 保存
        filename = f"trading_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def create_risk_metrics_heatmap(self, data: List[Dict[str, Any]]) -> str:
        """リスクメトリクスヒートマップ作成"""
        if not data:
            return ""

        # データをDataFrameに変換
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # リスク指標のみ抽出
        risk_columns = [
            "current_drawdown",
            "portfolio_var_95",
            "portfolio_volatility",
            "concentration_risk",
            "leverage_ratio",
        ]
        risk_data = df[risk_columns].copy()

        # パーセント変換
        for col in ["current_drawdown", "portfolio_var_95", "portfolio_volatility"]:
            risk_data[col] = risk_data[col] * 100

        # 列名を日本語に変更
        risk_data.columns = [
            "ドローダウン(%)",
            "VaR95%(%)",
            "ボラティリティ(%)",
            "集中リスク",
            "レバレッジ比率",
        ]

        # ヒートマップ作成
        fig, ax = plt.subplots(figsize=(12, 8))

        # 最新10データポイントのみ表示
        recent_data = risk_data.tail(10)

        sns.heatmap(
            recent_data.T,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            ax=ax,
            cbar_kws={"label": "リスク値"},
        )

        ax.set_title("リスクメトリクス ヒートマップ", fontsize=16, fontweight="bold")
        ax.set_xlabel("時間", fontsize=12)
        ax.set_ylabel("リスク指標", fontsize=12)

        # X軸の時間フォーマット
        x_labels = [t.strftime("%H:%M") for t in recent_data.index]
        ax.set_xticklabels(x_labels, rotation=45)

        plt.tight_layout()

        # 保存
        filename = f"risk_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def create_positions_pie_chart(
        self, positions_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """ポジション構成円グラフ作成"""
        if not positions_data:
            return ""

        # データ準備
        symbols = list(positions_data.keys())
        values = [pos["value"] for pos in positions_data.values()]
        total_value = sum(values)

        if total_value == 0:
            return ""

        # パーセンテージ計算
        [v / total_value * 100 for v in values]

        # 色の設定
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 円グラフ
        wedges, texts, autotexts = ax1.pie(
            values, labels=symbols, autopct="%1.1f%%", colors=colors, startangle=90
        )
        ax1.set_title("ポジション構成比", fontsize=14, fontweight="bold")

        # 詳細テーブル
        table_data = []
        for symbol, pos in positions_data.items():
            table_data.append(
                [
                    symbol,
                    f"{pos['quantity']:,}",
                    f"{pos['price']:,.0f}円",
                    f"{pos['value']:,.0f}円",
                    f"{pos['value']/total_value*100:.1f}%",
                ]
            )

        ax2.axis("tight")
        ax2.axis("off")
        table = ax2.table(
            cellText=table_data,
            colLabels=["銘柄", "数量", "価格", "評価額", "構成比"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # ヘッダー行のスタイル
        for column_idx in range(len(table_data[0])):
            table[(0, column_idx)].set_facecolor("#4CAF50")
            table[(0, column_idx)].set_text_props(weight="bold", color="white")

        ax2.set_title("ポジション詳細", fontsize=14, fontweight="bold")

        plt.suptitle(
            f"ポートフォリオ構成 (総額: {total_value:,.0f}円)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # 保存
        filename = (
            f"positions_composition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def create_comprehensive_dashboard(
        self,
        portfolio_data: List[Dict[str, Any]],
        system_data: List[Dict[str, Any]],
        trading_data: List[Dict[str, Any]],
        risk_data: List[Dict[str, Any]],
        positions_data: Dict[str, Dict[str, Any]],
    ) -> str:
        """統合ダッシュボード作成"""

        fig = plt.figure(figsize=(20, 12))

        # レイアウト設定
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. ポートフォリオ価値推移 (大きく表示)
        ax1 = fig.add_subplot(gs[0, :2])
        if portfolio_data:
            timestamps = [
                datetime.fromisoformat(item["timestamp"]) for item in portfolio_data
            ]
            values = [item["total_value"] for item in portfolio_data]
            ax1.plot(timestamps, values, color=self.colors["primary"], linewidth=3)
            ax1.set_title("ポートフォリオ価値推移", fontsize=14, fontweight="bold")
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
            ax1.grid(True, alpha=0.3)

        # 2. システム状態
        ax2 = fig.add_subplot(gs[0, 2])
        if system_data:
            latest_system = system_data[-1]
            metrics = ["CPU\n(%)", "Memory\n(GB)", "Errors"]
            values = [
                latest_system["cpu_usage"],
                latest_system["memory_usage_mb"] / 1024,
                latest_system["error_count"],
            ]
            colors_sys = [
                self.colors["warning"],
                self.colors["info"],
                self.colors["danger"],
            ]
            bars = ax2.bar(metrics, values, color=colors_sys, alpha=0.7)
            ax2.set_title("システム状態", fontweight="bold")
            ax2.set_ylim(0, max(100, max(values) * 1.2))

            # 値をバーに表示
            for bar, value in zip(bars, values):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # 3. 取引統計
        ax3 = fig.add_subplot(gs[0, 3])
        if trading_data:
            latest_trading = trading_data[-1]
            ax3.pie(
                [latest_trading["successful_trades"], latest_trading["failed_trades"]],
                labels=["成功", "失敗"],
                colors=[self.colors["success"], self.colors["danger"]],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax3.set_title(
                f"本日取引: {latest_trading['trades_today']}回", fontweight="bold"
            )

        # 4. リスク指標
        ax4 = fig.add_subplot(gs[1, :2])
        if risk_data:
            timestamps = [
                datetime.fromisoformat(item["timestamp"]) for item in risk_data
            ]
            drawdown = [item["current_drawdown"] * 100 for item in risk_data]
            volatility = [item["portfolio_volatility"] * 100 for item in risk_data]

            ax4_twin = ax4.twinx()

            line1 = ax4.plot(
                timestamps,
                drawdown,
                color=self.colors["danger"],
                linewidth=2,
                label="ドローダウン(%)",
            )
            line2 = ax4_twin.plot(
                timestamps,
                volatility,
                color=self.colors["warning"],
                linewidth=2,
                label="ボラティリティ(%)",
            )

            ax4.set_title("リスク指標推移", fontweight="bold")
            ax4.set_ylabel("ドローダウン (%)", color=self.colors["danger"])
            ax4_twin.set_ylabel("ボラティリティ (%)", color=self.colors["warning"])

            # 凡例統合
            lines = line1 + line2
            labels = [line.get_label() for line in lines]
            ax4.legend(lines, labels, loc="upper left")
            ax4.grid(True, alpha=0.3)

        # 5. ポジション構成
        ax5 = fig.add_subplot(gs[1, 2:])
        if positions_data:
            symbols = list(positions_data.keys())
            values = [pos["value"] for pos in positions_data.values()]
            colors_pos = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
            ax5.pie(values, labels=symbols, colors=colors_pos, autopct="%1.1f%%")
            ax5.set_title("ポジション構成", fontweight="bold")

        # 6. パフォーマンスサマリー
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")

        # サマリー情報作成
        summary_text = []
        if portfolio_data and trading_data and risk_data:
            latest_portfolio = portfolio_data[-1]
            latest_trading = trading_data[-1]
            latest_risk = risk_data[-1]

            # 安全なデータアクセス
            total_value = latest_portfolio.get("total_value", 0)
            daily_return = latest_portfolio.get("daily_return", 0)
            trades_today = latest_trading.get("trades_today", 0)
            win_rate = latest_trading.get("win_rate", 0)
            avg_exec_time = latest_trading.get("average_execution_time_ms", 0)
            current_drawdown = latest_risk.get("current_drawdown", 0)
            var_95 = latest_risk.get("portfolio_var_95", 0)
            volatility = latest_risk.get("portfolio_volatility", 0)

            summary_text.extend(
                [
                    f"Portfolio: Total {total_value:,.0f}JPY, "
                    + f"Daily Return {daily_return:.2%}",
                    f"Trading: Today {trades_today} trades, "
                    + f"Win Rate {win_rate:.1%}, "
                    + f"Avg Time {avg_exec_time:.0f}ms",
                    f"Risk: Drawdown {current_drawdown:.2%}, "
                    + f"VaR(95%) {var_95:.2%}, "
                    + f"Volatility {volatility:.2%}",
                    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )
        else:
            summary_text.append("Data loading...")

        # テキスト表示
        summary_str = "\n".join(summary_text)
        ax6.text(
            0.05,
            0.8,
            summary_str,
            transform=ax6.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        ax6.set_title(
            "パフォーマンスサマリー",
            fontsize=14,
            fontweight="bold",
            x=0.05,
            y=0.95,
            ha="left",
            transform=ax6.transAxes,
        )

        # 全体タイトル
        fig.suptitle(
            "プロダクション運用監視ダッシュボード", fontsize=20, fontweight="bold"
        )

        # 保存
        filename = (
            f"comprehensive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def chart_to_base64(self, filepath: str) -> str:
        """チャートをBase64エンコード"""
        try:
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Base64変換エラー: {e}")
            return ""

    def cleanup_old_charts(self, hours: int = 24):
        """古いチャートファイルクリーンアップ"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        cleaned_count = 0
        for chart_file in self.output_dir.glob("*.png"):
            try:
                file_time = datetime.fromtimestamp(chart_file.stat().st_mtime)
                if file_time < cutoff_time:
                    chart_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                print(f"ファイル削除エラー: {chart_file} - {e}")

        if cleaned_count > 0:
            print(f"古いチャート {cleaned_count}件を削除しました")


if __name__ == "__main__":
    # テスト実行
    print("ダッシュボード可視化エンジンテスト")
    print("=" * 50)

    engine = DashboardVisualizationEngine()

    # サンプルデータ生成
    import random

    current_time = datetime.now()
    sample_data = []

    for i in range(24):  # 24時間分
        timestamp = current_time - timedelta(hours=23 - i)
        sample_data.append(
            {
                "timestamp": timestamp.isoformat(),
                "total_value": 1000000 + random.uniform(-50000, 50000),
                "cpu_usage": random.uniform(20, 80),
                "memory_usage_mb": random.uniform(1024, 3072),
                "processing_time_ms": random.uniform(100, 500),
                "error_count": random.randint(0, 3),
                "trades_today": random.randint(5, 20),
                "successful_trades": random.randint(3, 18),
                "failed_trades": random.randint(0, 5),
                "win_rate": random.uniform(0.6, 0.9),
                "current_drawdown": random.uniform(-0.1, 0),
                "portfolio_var_95": random.uniform(-0.05, -0.01),
                "portfolio_volatility": random.uniform(0.1, 0.3),
                "concentration_risk": random.uniform(0.3, 0.8),
                "leverage_ratio": random.uniform(0.8, 1.2),
            }
        )

    # サンプルポジションデータ
    positions_data = {
        "7203.T": {"quantity": 100, "price": 2500, "value": 250000, "pnl": 10000},
        "8306.T": {"quantity": 1000, "price": 800, "value": 800000, "pnl": -5000},
        "9984.T": {"quantity": 50, "price": 5000, "value": 250000, "pnl": 15000},
    }

    try:
        # 各種チャート作成テスト
        print("ポートフォリオ価値チャート作成...")
        portfolio_chart = engine.create_portfolio_value_chart(sample_data)
        print(f"作成完了: {portfolio_chart}")

        print("システムメトリクスチャート作成...")
        system_chart = engine.create_system_metrics_chart(sample_data)
        print(f"作成完了: {system_chart}")

        print("取引パフォーマンスチャート作成...")
        trading_chart = engine.create_trading_performance_chart(sample_data)
        print(f"作成完了: {trading_chart}")

        print("リスクヒートマップ作成...")
        risk_heatmap = engine.create_risk_metrics_heatmap(sample_data)
        print(f"作成完了: {risk_heatmap}")

        print("ポジション円グラフ作成...")
        positions_chart = engine.create_positions_pie_chart(positions_data)
        print(f"作成完了: {positions_chart}")

        print("統合ダッシュボード作成...")
        comprehensive_chart = engine.create_comprehensive_dashboard(
            sample_data, sample_data, sample_data, sample_data, positions_data
        )
        print(f"作成完了: {comprehensive_chart}")

        print("\n可視化エンジンテスト成功")

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()
