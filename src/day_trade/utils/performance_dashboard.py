#!/usr/bin/env python3
"""
パフォーマンス監視ダッシュボード

Issue #311: リアルタイムパフォーマンス可視化システム
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from .logging_config import get_context_logger
from .performance_monitor import get_performance_monitor

logger = get_context_logger(__name__)


class PerformanceDashboard:
    """
    パフォーマンス監視ダッシュボード

    機能:
    - リアルタイム監視チャート
    - パフォーマンス指標の可視化
    - アラート状況の表示
    - ボトルネック分析結果
    """

    def __init__(self, output_dir: str = "dashboard_output", update_interval: int = 5):
        """初期化"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.update_interval = update_interval
        self.monitor = get_performance_monitor()

        # 日本語フォント設定
        plt.rcParams["font.family"] = [
            "DejaVu Sans",
            "Arial Unicode MS",
            "Hiragino Sans",
            "Yu Gothic",
            "Meiryo",
            "Takao",
            "IPAexGothic",
            "IPAPGothic",
            "VL PGothic",
            "Noto Sans CJK JP",
        ]

        logger.info("パフォーマンスダッシュボード初期化完了")
        logger.info(f"  - 出力ディレクトリ: {self.output_dir}")
        logger.info(f"  - 更新間隔: {self.update_interval}秒")

    def create_realtime_dashboard(self) -> Path:
        """リアルタイムダッシュボード作成"""
        try:
            # 最新のパフォーマンス要約取得
            summary = self.monitor.get_performance_summary(hours=24)
            if "error" in summary:
                logger.warning("パフォーマンスデータが不足しています")
                return self._create_empty_dashboard()

            # ボトルネック分析
            bottlenecks = self.monitor.get_bottleneck_analysis()

            # 4x2のレイアウトでダッシュボード作成
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(
                "📊 パフォーマンス監視ダッシュボード", fontsize=16, fontweight="bold"
            )

            # 1. 処理時間の推移
            self._plot_execution_time_trend(axes[0, 0])

            # 2. メモリ使用量の推移
            self._plot_memory_usage_trend(axes[0, 1])

            # 3. CPU使用率の推移
            self._plot_cpu_usage_trend(axes[0, 2])

            # 4. 成功率
            self._plot_success_rate(axes[0, 3], summary)

            # 5. 遅いプロセス Top 5
            self._plot_slow_processes(axes[1, 0], bottlenecks)

            # 6. メモリ消費プロセス Top 5
            self._plot_memory_heavy_processes(axes[1, 1], bottlenecks)

            # 7. 基準値比較
            self._plot_baseline_comparison(axes[1, 2], summary)

            # 8. システム概要
            self._plot_system_overview(axes[1, 3], summary)

            plt.tight_layout()

            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_path = self.output_dir / f"performance_dashboard_{timestamp}.png"

            plt.savefig(dashboard_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"ダッシュボード作成完了: {dashboard_path}")
            return dashboard_path

        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {e}")
            return self._create_empty_dashboard()

    def _plot_execution_time_trend(self, ax):
        """処理時間推移のプロット"""
        try:
            recent_metrics = [
                m
                for m in self.monitor.metrics_history
                if m.timestamp.timestamp()
                > (datetime.now() - timedelta(hours=6)).timestamp()
            ]

            if not recent_metrics:
                ax.text(
                    0.5,
                    0.5,
                    "データなし",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("⏱️ 処理時間推移 (6時間)")
                return

            timestamps = [m.timestamp for m in recent_metrics]
            execution_times = [m.execution_time for m in recent_metrics]

            ax.plot(timestamps, execution_times, "b-", linewidth=2, alpha=0.7)
            ax.scatter(timestamps, execution_times, c="blue", s=20, alpha=0.6)

            # 基準線
            baseline_avg = 3.6  # 85銘柄ML分析の基準時間
            ax.axhline(
                y=baseline_avg,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"基準値: {baseline_avg}s",
            )
            ax.axhline(
                y=baseline_avg * 1.5,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="警告閾値",
            )

            ax.set_title("⏱️ 処理時間推移 (6時間)")
            ax.set_ylabel("実行時間 (秒)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        except Exception as e:
            logger.debug(f"処理時間推移プロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _plot_memory_usage_trend(self, ax):
        """メモリ使用量推移のプロット"""
        try:
            recent_metrics = [
                m
                for m in self.monitor.metrics_history
                if m.timestamp.timestamp()
                > (datetime.now() - timedelta(hours=6)).timestamp()
            ]

            if not recent_metrics:
                ax.text(
                    0.5,
                    0.5,
                    "データなし",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("💾 メモリ使用量推移 (6時間)")
                return

            timestamps = [m.timestamp for m in recent_metrics]
            memory_usage = [m.memory_peak_mb for m in recent_metrics]

            ax.plot(timestamps, memory_usage, "r-", linewidth=2, alpha=0.7)
            ax.scatter(timestamps, memory_usage, c="red", s=20, alpha=0.6)

            # 警告線
            ax.axhline(
                y=1000, color="orange", linestyle="--", alpha=0.7, label="警告閾値: 1GB"
            )

            ax.set_title("💾 メモリ使用量推移 (6時間)")
            ax.set_ylabel("メモリ使用量 (MB)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        except Exception as e:
            logger.debug(f"メモリ使用量推移プロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _plot_cpu_usage_trend(self, ax):
        """CPU使用率推移のプロット"""
        try:
            # システム監視データ使用
            recent_system = [
                s
                for s in self.monitor.system_history
                if s.timestamp.timestamp()
                > (datetime.now() - timedelta(hours=6)).timestamp()
            ]

            if not recent_system:
                ax.text(
                    0.5,
                    0.5,
                    "データなし",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("🖥️ CPU使用率推移 (6時間)")
                return

            timestamps = [s.timestamp for s in recent_system]
            cpu_usage = [s.cpu_usage_percent for s in recent_system]

            ax.plot(timestamps, cpu_usage, "g-", linewidth=2, alpha=0.7)
            ax.scatter(timestamps, cpu_usage, c="green", s=20, alpha=0.6)

            # 警告線
            ax.axhline(
                y=80, color="orange", linestyle="--", alpha=0.7, label="警告閾値: 80%"
            )

            ax.set_title("🖥️ CPU使用率推移 (6時間)")
            ax.set_ylabel("CPU使用率 (%)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        except Exception as e:
            logger.debug(f"CPU使用率推移プロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _plot_success_rate(self, ax, summary: Dict):
        """成功率の表示"""
        try:
            success_rate = summary.get("success_rate", 0)

            # 円グラフで成功率を表示
            sizes = [success_rate, 1 - success_rate]
            labels = ["成功", "失敗"]
            colors = ["#2ecc71", "#e74c3c"]

            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )

            ax.set_title(f'✅ 成功率 ({summary.get("total_operations", 0)}回実行)')

        except Exception as e:
            logger.debug(f"成功率プロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _plot_slow_processes(self, ax, bottlenecks: Dict):
        """遅いプロセス Top 5"""
        try:
            if "error" in bottlenecks or not bottlenecks.get("slow_processes"):
                ax.text(
                    0.5,
                    0.5,
                    "データなし",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("🐌 遅いプロセス Top 5")
                return

            processes = bottlenecks["slow_processes"][:5]
            names = [
                p["process"][:15] + "..." if len(p["process"]) > 15 else p["process"]
                for p in processes
            ]
            times = [p["execution_time"] for p in processes]

            bars = ax.barh(names, times, color="orange", alpha=0.7)
            ax.set_title("🐌 遅いプロセス Top 5")
            ax.set_xlabel("実行時間 (秒)")

            # 値をバーに表示
            for bar, time in zip(bars, times):
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{time:.2f}s",
                    ha="left",
                    va="center",
                    fontsize=8,
                )

        except Exception as e:
            logger.debug(f"遅いプロセスプロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _plot_memory_heavy_processes(self, ax, bottlenecks: Dict):
        """メモリ消費プロセス Top 5"""
        try:
            if "error" in bottlenecks or not bottlenecks.get("memory_heavy_processes"):
                ax.text(
                    0.5,
                    0.5,
                    "データなし",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("🧠 メモリ消費 Top 5")
                return

            processes = bottlenecks["memory_heavy_processes"][:5]
            names = [
                p["process"][:15] + "..." if len(p["process"]) > 15 else p["process"]
                for p in processes
            ]
            memory = [p["memory_peak_mb"] for p in processes]

            bars = ax.barh(names, memory, color="red", alpha=0.7)
            ax.set_title("🧠 メモリ消費 Top 5")
            ax.set_xlabel("メモリ使用量 (MB)")

            # 値をバーに表示
            for bar, mem in zip(bars, memory):
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{mem:.0f}MB",
                    ha="left",
                    va="center",
                    fontsize=8,
                )

        except Exception as e:
            logger.debug(f"メモリ消費プロセスプロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _plot_baseline_comparison(self, ax, summary: Dict):
        """基準値比較"""
        try:
            comparison = summary.get("baseline_comparison", {})
            if not comparison:
                ax.text(
                    0.5,
                    0.5,
                    "データなし",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("📊 基準値比較")
                return

            processes = []
            ratios = []
            colors = []

            for process_name, data in comparison.items():
                processes.append(process_name.replace("_", " ").title()[:20])
                ratios.append(data["performance_ratio"])

                # 状態に応じた色分け
                status = data["status"]
                if status == "good":
                    colors.append("#2ecc71")  # 緑
                elif status == "warning":
                    colors.append("#f39c12")  # オレンジ
                else:
                    colors.append("#e74c3c")  # 赤

            ax.bar(processes, ratios, color=colors, alpha=0.7)
            ax.set_title("📊 基準値比較 (1.0=基準)")
            ax.set_ylabel("パフォーマンス比率")
            ax.axhline(y=1.0, color="green", linestyle="-", alpha=0.5, label="基準値")
            ax.axhline(y=1.2, color="orange", linestyle="--", alpha=0.5, label="警告")
            ax.axhline(y=1.5, color="red", linestyle="--", alpha=0.5, label="危険")

            # X軸ラベルを回転
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax.legend(fontsize=8)

        except Exception as e:
            logger.debug(f"基準値比較プロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _plot_system_overview(self, ax, summary: Dict):
        """システム概要"""
        try:
            # 最新のシステム指標取得
            if self.monitor.system_history:
                latest_system = self.monitor.system_history[-1]
            else:
                ax.text(
                    0.5,
                    0.5,
                    "データなし",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("💻 システム概要")
                return

            # システム情報テキスト
            overview_text = f"""
システム状況:
• 総操作回数: {summary.get('total_operations', 0)}
• 成功率: {summary.get('success_rate', 0):.1%}
• 平均実行時間: {summary.get('avg_execution_time', 0):.2f}s
• 最大実行時間: {summary.get('max_execution_time', 0):.2f}s

現在のシステム:
• メモリ使用率: {latest_system.memory_usage_percent:.1f}%
• CPU使用率: {latest_system.cpu_usage_percent:.1f}%
• 利用可能メモリ: {latest_system.available_memory_gb:.1f}GB
• アクティブプロセス: {latest_system.active_processes}

最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()

            ax.text(
                0.05,
                0.95,
                overview_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            )

            ax.set_title("💻 システム概要")
            ax.axis("off")

        except Exception as e:
            logger.debug(f"システム概要プロットエラー: {e}")
            ax.text(
                0.5, 0.5, "エラー発生", ha="center", va="center", transform=ax.transAxes
            )

    def _create_empty_dashboard(self) -> Path:
        """空のダッシュボード作成"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "📊 パフォーマンスダッシュボード\n\n"
            + "データ収集中...\n"
            + "監視システムが十分なデータを収集してから\n"
            + "詳細な分析結果を表示します。",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_title("パフォーマンス監視ダッシュボード - データ準備中", fontsize=16)
        ax.axis("off")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = self.output_dir / f"empty_dashboard_{timestamp}.png"

        plt.savefig(dashboard_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        return dashboard_path

    def generate_performance_report(self) -> Path:
        """詳細パフォーマンスレポート生成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"performance_report_{timestamp}.html"

            # データ取得
            summary = self.monitor.get_performance_summary(hours=24)
            bottlenecks = self.monitor.get_bottleneck_analysis()

            # HTMLレポート生成
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>パフォーマンス監視レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }}
        h1, h2 {{ color: #333; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #007bff; }}
        .alert {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
        .success {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
        .error {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 パフォーマンス監視レポート</h1>
        <p class="timestamp">生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>

        <h2>システム概要</h2>
        <div class="metric-card {'success' if summary.get('success_rate', 0) > 0.9 else 'alert' if summary.get('success_rate', 0) > 0.7 else 'error'}">
            <strong>成功率:</strong> {summary.get('success_rate', 0):.1%} ({summary.get('total_operations', 0)}回実行)
        </div>
        <div class="metric-card">
            <strong>平均実行時間:</strong> {summary.get('avg_execution_time', 0):.3f}秒
        </div>
        <div class="metric-card">
            <strong>最大実行時間:</strong> {summary.get('max_execution_time', 0):.3f}秒
        </div>
        <div class="metric-card">
            <strong>平均メモリ使用量:</strong> {summary.get('avg_memory_usage_mb', 0):.1f}MB
        </div>

        <h2>基準値比較</h2>
        <table>
            <tr>
                <th>プロセス</th>
                <th>基準時間</th>
                <th>現在の平均</th>
                <th>パフォーマンス比率</th>
                <th>状態</th>
            </tr>
            """

            for process_name, data in summary.get("baseline_comparison", {}).items():
                status_emoji = {"good": "✅", "warning": "⚠️", "critical": "❌"}.get(
                    data["status"], "❓"
                )
                html_content += f"""
            <tr>
                <td>{process_name}</td>
                <td>{data["baseline_seconds"]:.2f}s</td>
                <td>{data["current_avg_seconds"]:.2f}s</td>
                <td>{data["performance_ratio"]:.2f}x</td>
                <td>{status_emoji} {data["status"]}</td>
            </tr>
                """

            html_content += """
        </table>

        <h2>ボトルネック分析</h2>
        """

            if "error" not in bottlenecks and bottlenecks.get("slow_processes"):
                html_content += """
        <h3>🐌 処理時間が長いプロセス Top 10</h3>
        <table>
            <tr>
                <th>プロセス名</th>
                <th>実行時間</th>
                <th>実行日時</th>
            </tr>
                """
                for proc in bottlenecks["slow_processes"][:10]:
                    html_content += f"""
            <tr>
                <td>{proc["process"]}</td>
                <td>{proc["execution_time"]:.3f}秒</td>
                <td>{proc["timestamp"][:19].replace('T', ' ')}</td>
            </tr>
                    """
                html_content += "</table>"

            if "error" not in bottlenecks and bottlenecks.get("memory_heavy_processes"):
                html_content += """
        <h3>🧠 メモリ消費量が多いプロセス Top 10</h3>
        <table>
            <tr>
                <th>プロセス名</th>
                <th>メモリ使用量</th>
                <th>実行日時</th>
            </tr>
                """
                for proc in bottlenecks["memory_heavy_processes"][:10]:
                    html_content += f"""
            <tr>
                <td>{proc["process"]}</td>
                <td>{proc["memory_peak_mb"]:.1f}MB</td>
                <td>{proc["timestamp"][:19].replace('T', ' ')}</td>
            </tr>
                    """
                html_content += "</table>"

            html_content += """
        <h2>推奨アクション</h2>
        <div class="metric-card">
        """

            # 推奨アクション生成
            recommendations = []
            if summary.get("success_rate", 1) < 0.9:
                recommendations.append(
                    "❗ 成功率が90%を下回っています。エラー処理の見直しを検討してください。"
                )

            baseline_issues = [
                data
                for data in summary.get("baseline_comparison", {}).values()
                if data["performance_ratio"] > 1.5
            ]
            if baseline_issues:
                recommendations.append(
                    "⚠️ 基準値を大幅に超過するプロセスがあります。パフォーマンス最適化が必要です。"
                )

            if not recommendations:
                recommendations.append(
                    "✅ 現在のところ、大きな問題は検出されていません。"
                )

            for rec in recommendations:
                html_content += f"<p>{rec}</p>"

            html_content += """
        </div>
    </div>
</body>
</html>
            """

            # ファイル保存
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"パフォーマンスレポート生成完了: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"パフォーマンスレポート生成エラー: {e}")
            raise


def create_dashboard() -> Path:
    """簡単なダッシュボード作成関数"""
    dashboard = PerformanceDashboard()
    return dashboard.create_realtime_dashboard()


def create_detailed_report() -> Path:
    """詳細レポート作成関数"""
    dashboard = PerformanceDashboard()
    return dashboard.generate_performance_report()


if __name__ == "__main__":
    # テスト実行
    print("=== パフォーマンスダッシュボード テスト ===")

    try:
        dashboard = PerformanceDashboard()

        # ダッシュボード作成
        dashboard_path = dashboard.create_realtime_dashboard()
        print(f"ダッシュボード作成: {dashboard_path}")

        # HTMLレポート作成
        report_path = dashboard.generate_performance_report()
        print(f"HTMLレポート作成: {report_path}")

        print("ダッシュボードテスト完了")

    except Exception as e:
        print(f"ダッシュボードテストエラー: {e}")
