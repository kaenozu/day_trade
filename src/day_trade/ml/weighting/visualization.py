#!/usr/bin/env python3
"""
Visualization utilities for Dynamic Weighting System

このモジュールは重み変化の可視化機能を提供します。
matplotlib を使用してグラフやチャートを生成し、
動的重み調整の動作を視覚的に分析できます。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .core import MarketRegime
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class WeightVisualization:
    """
    重み可視化クラス

    重みの変化、市場状態の変遷、パフォーマンス指標等を
    グラフィカルに表示する機能を提供します。
    """

    def __init__(self, model_names: List[str]):
        """
        初期化

        Args:
            model_names: モデル名のリスト
        """
        self.model_names = model_names

    def plot_weight_evolution(
        self,
        weight_history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ):
        """
        重み変化の可視化

        Args:
            weight_history: 重み履歴データ
            save_path: 保存パス（指定時はファイル保存、未指定時は画面表示）
            figsize: 図のサイズ
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if not weight_history:
                logger.warning("重み履歴データが存在しません")
                return

            # データ準備
            timestamps = [
                datetime.fromtimestamp(h['timestamp']) 
                for h in weight_history
            ]

            plt.figure(figsize=figsize)

            # 各モデルの重み変化をプロット
            for model_name in self.model_names:
                weights = [
                    h['weights'].get(model_name, 0) 
                    for h in weight_history
                ]
                plt.plot(
                    timestamps, weights, 
                    label=model_name, marker='o', markersize=3
                )

            plt.xlabel('時間')
            plt.ylabel('重み')
            plt.title('動的重み調整の変化')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # 市場状態の背景色
            self._add_regime_background(plt, timestamps, weight_history)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"重み変化グラフ保存: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib未インストール")
        except Exception as e:
            logger.error(f"重み変化可視化エラー: {e}")

    def plot_performance_comparison(
        self,
        performance_data: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ):
        """
        モデル性能比較グラフ

        Args:
            performance_data: モデル別性能データ
            save_path: 保存パス
            figsize: 図のサイズ
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if not performance_data:
                logger.warning("性能データが存在しません")
                return

            metrics = list(next(iter(performance_data.values())).keys())
            model_names = list(performance_data.keys())
            
            # グラフ作成
            fig, axes = plt.subplots(
                1, len(metrics), figsize=figsize, 
                sharey=False
            )
            if len(metrics) == 1:
                axes = [axes]

            for i, metric in enumerate(metrics):
                values = [
                    performance_data[model][metric] 
                    for model in model_names
                ]
                
                axes[i].bar(model_names, values)
                axes[i].set_title(f'{metric.upper()}')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"性能比較グラフ保存: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib未インストール")
        except Exception as e:
            logger.error(f"性能比較可視化エラー: {e}")

    def plot_regime_timeline(
        self,
        regime_history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 4)
    ):
        """
        市場状態変遷のタイムライン表示

        Args:
            regime_history: 市場状態変更履歴
            save_path: 保存パス
            figsize: 図のサイズ
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if not regime_history:
                logger.warning("市場状態履歴データが存在しません")
                return

            fig, ax = plt.subplots(figsize=figsize)

            # 市場状態の色マップ
            regime_colors = self._get_regime_colors()
            
            # タイムライン作成
            for i, change in enumerate(regime_history):
                start_time = datetime.fromtimestamp(change['timestamp'])
                end_time = (
                    datetime.fromtimestamp(regime_history[i + 1]['timestamp'])
                    if i + 1 < len(regime_history)
                    else datetime.now()
                )
                
                regime = change['new_regime']
                color = regime_colors.get(regime, 'gray')
                
                ax.barh(
                    0, (end_time - start_time).total_seconds() / 3600,  # 時間単位
                    left=start_time, height=0.5,
                    color=color, alpha=0.7,
                    label=regime.value if i == 0 or change['new_regime'] != regime_history[i-1]['new_regime'] else ""
                )

            ax.set_xlabel('時間')
            ax.set_ylabel('市場状態')
            ax.set_title('市場状態変遷タイムライン')
            ax.legend()

            # 時間軸フォーマット
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.xticks(rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"市場状態タイムライン保存: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib未インストール")
        except Exception as e:
            logger.error(f"市場状態タイムライン可視化エラー: {e}")

    def create_dashboard(
        self,
        weight_history: List[Dict[str, Any]],
        performance_data: Dict[str, Dict[str, float]],
        regime_history: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """
        総合ダッシュボードの作成

        Args:
            weight_history: 重み履歴
            performance_data: 性能データ
            regime_history: 市場状態履歴
            save_path: 保存パス
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            if not any([weight_history, performance_data, regime_history]):
                logger.warning("表示するデータが存在しません")
                return

            # グリッドレイアウト設定
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 2, figure=fig)

            # 重み変化グラフ
            if weight_history:
                ax1 = fig.add_subplot(gs[0, :])
                self._plot_weights_on_axis(ax1, weight_history)

            # 性能比較グラフ
            if performance_data:
                ax2 = fig.add_subplot(gs[1, 0])
                self._plot_performance_on_axis(ax2, performance_data)

            # 市場状態分布
            if regime_history:
                ax3 = fig.add_subplot(gs[1, 1])
                self._plot_regime_distribution_on_axis(ax3, regime_history)

            # 統計サマリー
            ax4 = fig.add_subplot(gs[2, :])
            self._plot_statistics_table_on_axis(
                ax4, weight_history, performance_data, regime_history
            )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ダッシュボード保存: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib未インストール")
        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {e}")

    def _add_regime_background(
        self, 
        plt, 
        timestamps: List[datetime], 
        weight_history: List[Dict[str, Any]]
    ):
        """市場状態の背景色を追加"""
        regime_colors = self._get_regime_colors()

        for i, h in enumerate(weight_history[:-1]):
            if i + 1 < len(timestamps):
                plt.axvspan(
                    timestamps[i], timestamps[i + 1],
                    alpha=0.2, 
                    color=regime_colors.get(
                        h.get('regime', MarketRegime.SIDEWAYS), 'white'
                    )
                )

    def _get_regime_colors(self) -> Dict[MarketRegime, str]:
        """市場状態カラーマップを取得"""
        return {
            MarketRegime.BULL_MARKET: 'lightgreen',
            MarketRegime.BEAR_MARKET: 'lightcoral',
            MarketRegime.HIGH_VOLATILITY: 'lightyellow',
            MarketRegime.LOW_VOLATILITY: 'lightblue',
            MarketRegime.SIDEWAYS: 'lightgray'
        }

    def _plot_weights_on_axis(self, ax, weight_history: List[Dict[str, Any]]):
        """指定軸に重み変化をプロット"""
        timestamps = [
            datetime.fromtimestamp(h['timestamp']) 
            for h in weight_history
        ]

        for model_name in self.model_names:
            weights = [
                h['weights'].get(model_name, 0) 
                for h in weight_history
            ]
            ax.plot(timestamps, weights, label=model_name, marker='o', markersize=2)

        ax.set_xlabel('時間')
        ax.set_ylabel('重み')
        ax.set_title('動的重み調整の変化')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_on_axis(
        self, 
        ax, 
        performance_data: Dict[str, Dict[str, float]]
    ):
        """指定軸に性能データをプロット"""
        models = list(performance_data.keys())
        rmse_values = [performance_data[m].get('rmse', 0) for m in models]

        ax.bar(models, rmse_values)
        ax.set_title('RMSE比較')
        ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', rotation=45)

    def _plot_regime_distribution_on_axis(
        self, 
        ax, 
        regime_history: List[Dict[str, Any]]
    ):
        """指定軸に市場状態分布をプロット"""
        regime_counts = {}
        for change in regime_history:
            regime = change['new_regime']
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

        if regime_counts:
            ax.pie(
                regime_counts.values(), 
                labels=regime_counts.keys(), 
                autopct='%1.1f%%'
            )
            ax.set_title('市場状態分布')

    def _plot_statistics_table_on_axis(
        self, 
        ax, 
        weight_history: List[Dict[str, Any]], 
        performance_data: Dict[str, Dict[str, float]], 
        regime_history: List[Dict[str, Any]]
    ):
        """指定軸に統計テーブルをプロット"""
        # 統計情報の準備
        stats = [
            ['総重み更新回数', str(len(weight_history))],
            ['市場状態変更回数', str(len(regime_history))],
            ['監視モデル数', str(len(self.model_names))],
        ]

        if performance_data:
            avg_rmse = sum(
                data.get('rmse', 0) for data in performance_data.values()
            ) / len(performance_data)
            stats.append(['平均RMSE', f'{avg_rmse:.4f}'])

        # テーブル表示
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=stats,
            colLabels=['項目', '値'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.set_title('統計サマリー')