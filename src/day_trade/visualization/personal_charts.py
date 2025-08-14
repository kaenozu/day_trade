"""
個人投資家向けチャート生成機能

93%精度AIの分析結果を美しく分かりやすくグラフ表示
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import os


class PersonalChartGenerator:
    """個人投資家向けシンプルチャート生成"""

    def __init__(self):
        """初期化"""
        # 日本語フォント設定
        self._setup_japanese_font()

        # 個人投資家向けカラーパレット
        self.colors = {
            'buy': '#28a745',      # 緑（買い）
            'sell': '#dc3545',     # 赤（売り）
            'hold': '#ffc107',     # 黄（様子見）
            'background': '#f8f9fa',
            'text': '#343a40',
            'grid': '#dee2e6'
        }

        # グラフスタイル設定
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = self.colors['background']
        plt.rcParams['axes.edgecolor'] = self.colors['grid']
        plt.rcParams['grid.color'] = self.colors['grid']
        plt.rcParams['text.color'] = self.colors['text']

    def _setup_japanese_font(self):
        """日本語フォント設定"""
        try:
            # Windows環境での日本語フォント設定
            font_list = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'DejaVu Sans']
            for font_name in font_list:
                try:
                    plt.rcParams['font.family'] = font_name
                    break
                except:
                    continue
        except:
            # フォント設定に失敗した場合はデフォルトのまま
            pass

    def generate_analysis_chart(self, recommendations: List[Dict[str, Any]], save_path: str = None) -> str:
        """
        分析結果チャート生成

        Args:
            recommendations: 分析結果リスト
            save_path: 保存パス（指定なしの場合は自動生成）

        Returns:
            保存されたファイルパス
        """
        if not recommendations:
            return None

        # データ準備
        symbols = [rec['symbol'] for rec in recommendations[:3]]
        names = [rec['name'] for rec in recommendations[:3]]
        scores = [rec['score'] for rec in recommendations[:3]]
        confidences = [rec['confidence'] for rec in recommendations[:3]]
        actions = [rec['action'] for rec in recommendations[:3]]

        # フィギュア作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('個人投資家向け分析結果', fontsize=16, fontweight='bold')

        # スコア棒グラフ
        colors = [self._get_action_color(action) for action in actions]
        bars = ax1.bar(range(len(symbols)), scores, color=colors, alpha=0.8)

        ax1.set_title('銘柄スコア比較', fontsize=12, fontweight='bold')
        ax1.set_xlabel('銘柄')
        ax1.set_ylabel('スコア（点）')
        ax1.set_xticks(range(len(symbols)))
        ax1.set_xticklabels([f"{symbol}\n{name}" for symbol, name in zip(symbols, names)],
                           rotation=0, ha='center')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # 棒グラフの上にスコア表示
        for i, (bar, score, action) in enumerate(zip(bars, scores, actions)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}点\n[{action}]',
                    ha='center', va='bottom', fontweight='bold')

        # 信頼度円グラフ
        sizes = confidences
        labels = [f"{symbol}\n{confidence:.0f}%" for symbol, confidence in zip(symbols, confidences)]

        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='',
                                          startangle=90, textprops={'fontsize': 10})

        ax2.set_title('予測信頼度分布', fontsize=12, fontweight='bold')

        # レイアウト調整
        plt.tight_layout()

        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"analysis_chart_{timestamp}.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return save_path

    def generate_simple_summary(self, recommendations: List[Dict[str, Any]], save_path: str = None) -> str:
        """
        シンプルサマリーチャート生成

        Args:
            recommendations: 分析結果リスト
            save_path: 保存パス

        Returns:
            保存されたファイルパス
        """
        if not recommendations:
            return None

        # データ準備
        symbols = [rec['symbol'] for rec in recommendations[:3]]
        names = [rec['name'] for rec in recommendations[:3]]
        actions = [rec['action'] for rec in recommendations[:3]]
        scores = [rec['score'] for rec in recommendations[:3]]
        confidences = [rec['confidence'] for rec in recommendations[:3]]

        # シンプルな横棒グラフ
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = range(len(symbols))
        colors = [self._get_action_color(action) for action in actions]

        # 横棒グラフ
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, height=0.6)

        # 設定
        ax.set_xlabel('スコア（点）', fontsize=12)
        ax.set_title('今日の推奨銘柄 - AI分析結果', fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{symbol} ({name})" for symbol, name in zip(symbols, names)])
        ax.set_xlim(0, 100)

        # グリッド
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_axisbelow(True)

        # 棒の横にスコアと推奨を表示
        for i, (bar, score, action, confidence) in enumerate(zip(bars, scores, actions, confidences)):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f}点 [{action}] (信頼度{confidence:.0f}%)',
                   ha='left', va='center', fontweight='bold')

        # 注意書き
        ax.text(0.5, -0.15, '※ 投資は自己責任で行ってください',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=10, style='italic', color='gray')

        plt.tight_layout()

        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"summary_chart_{timestamp}.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return save_path

    def _get_action_color(self, action: str) -> str:
        """アクション別の色を取得"""
        action_colors = {
            '買い': self.colors['buy'],
            '売り': self.colors['sell'],
            '様子見': self.colors['hold'],
            '検討': self.colors['hold']
        }
        return action_colors.get(action, self.colors['hold'])

    def test_chart_generation(self):
        """テスト用チャート生成"""
        # テストデータ
        test_recommendations = [
            {
                'symbol': '7203',
                'name': 'トヨタ自動車',
                'action': '買い',
                'score': 85.2,
                'confidence': 87.5,
                'risk_level': '中'
            },
            {
                'symbol': '8306',
                'name': '三菱UFJ',
                'action': '様子見',
                'score': 72.8,
                'confidence': 73.2,
                'risk_level': '低'
            },
            {
                'symbol': '9984',
                'name': 'ソフトバンクG',
                'action': '買い',
                'score': 78.9,
                'confidence': 81.3,
                'risk_level': '中'
            }
        ]

        # チャート生成
        chart_path = self.generate_analysis_chart(test_recommendations, "test_analysis.png")
        summary_path = self.generate_simple_summary(test_recommendations, "test_summary.png")

        return chart_path, summary_path


if __name__ == "__main__":
    # テスト実行
    chart_gen = PersonalChartGenerator()
    chart_path, summary_path = chart_gen.test_chart_generation()
    print(f"テストチャート生成完了:")
    print(f"- 詳細分析: {chart_path}")
    print(f"- サマリー: {summary_path}")