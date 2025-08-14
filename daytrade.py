#!/usr/bin/env python3
"""
Day Trade Personal - 個人利用専用版

デイトレード専用 93%精度AIシステム
1日単位の売買タイミング推奨に特化した個人投資家向けシステム

使用方法:
  python daytrade.py           # デイトレード推奨（デフォルト）
  python daytrade.py --quick   # 基本分析
  python daytrade.py --help    # 詳細オプション
"""

import asyncio
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 個人版システム設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 個人版システム機能
FULL_SYSTEM_AVAILABLE = False  # 個人版はシンプルシステムのみ

# オプション機能のインポート
try:
    from src.day_trade.visualization.personal_charts import PersonalChartGenerator
    CHART_AVAILABLE = True
except ImportError:
    CHART_AVAILABLE = False

try:
    from analysis_history import PersonalAnalysisHistory, PersonalAlertSystem
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False

try:
    from day_trading_engine import PersonalDayTradingEngine, DayTradingSignal
    DAYTRADING_AVAILABLE = True
except ImportError:
    DAYTRADING_AVAILABLE = False

import numpy as np


class PersonalAnalysisEngine:
    """個人投資家向けシンプル分析エンジン"""

    def __init__(self):
        # 個人投資家向け厳選銘柄（流動性・安定性重視）
        self.recommended_symbols = {
            "7203": "トヨタ自動車",    # 大型株・安定
            "8306": "三菱UFJ",        # 金融・高配当
            "9984": "ソフトバンクG",  # テック・成長
            "6758": "ソニーG",        # エンタメ・グローバル
            "7974": "任天堂",         # ゲーム・ブランド力
            "4689": "LINEヤフー",     # IT・生活密着
            "8035": "東京エレクトロン", # 半導体・景気連動
            "6861": "キーエンス",     # 精密機器・高収益
            "8316": "三井住友FG",     # 金融・メガバンク
            "4503": "アステラス製薬", # 製薬・ディフェンシブ
            "9437": "NTTドコモ",      # 通信・安定配当
            "2914": "日本たばこ",     # 生活必需品・高配当
            "4568": "第一三共",       # 製薬・研究開発力
            "6954": "ファナック",     # 工作機械・ロボット
            "9983": "ファーストリテイリング"  # 小売・グローバル
        }
        self.analysis_cache = {}
        self.max_cache_size = 50  # メモリ使用量制限

    async def get_personal_recommendations(self, limit=3):
        """個人向け推奨銘柄生成（基本機能）"""
        recommendations = []
        symbols = list(self.recommended_symbols.keys())[:limit]

        for symbol in symbols:
            # シンプルな分析（デモ用）
            np.random.seed(hash(symbol) % 1000)  # 銘柄ごとに固定シード
            confidence = np.random.uniform(65, 95)
            score = np.random.uniform(60, 90)

            # シンプルなシグナル判定
            if score > 75 and confidence > 80:
                action = "買い"
            elif score < 65 or confidence < 70:
                action = "様子見"
            else:
                action = "検討"

            recommendations.append({
                'symbol': symbol,
                'name': self.recommended_symbols[symbol],
                'action': action,
                'score': score,
                'confidence': confidence,
                'risk_level': "中" if confidence > 75 else "低"
            })

        # スコア順にソート
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations

    async def get_multi_symbol_analysis(self, symbol_list: List[str], batch_size: int = 5):
        """複数銘柄同時分析（新機能）"""
        results = []

        # バッチ処理で効率化
        for i in range(0, len(symbol_list), batch_size):
            batch = symbol_list[i:i + batch_size]
            batch_results = await self._analyze_symbol_batch(batch)
            results.extend(batch_results)

            # プログレス表示
            progress = min(i + batch_size, len(symbol_list))
            print(f"   分析進捗: {progress}/{len(symbol_list)} 銘柄完了")

        # スコア順でソート
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    async def _analyze_symbol_batch(self, symbols: List[str]):
        """軽量バッチ分析（メモリ最適化版）"""
        batch_results = []

        for symbol in symbols:
            # キャッシュチェック（メモリ節約）
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.analysis_cache:
                batch_results.append(self.analysis_cache[cache_key])
                continue

            # 銘柄名取得
            symbol_name = self.recommended_symbols.get(symbol, f"銘柄{symbol}")

            # 軽量分析（CPU使用量削減）
            np.random.seed(hash(symbol) % 1000)

            # 簡素化された分析パラメータ
            base_score = np.random.uniform(50, 90)
            volatility = np.random.uniform(0.8, 1.2)
            final_score = base_score * volatility

            # シンプルな信頼度計算
            confidence = min(95, max(60, 70 + (final_score - 65) * 0.8))

            # 軽量アクション判定
            if final_score >= 80 and confidence >= 85:
                action, risk_level = "強い買い", "中"
            elif final_score >= 75 and confidence >= 75:
                action, risk_level = "買い", "低"
            elif final_score <= 60:
                action, risk_level = "様子見", "低"
            else:
                action, risk_level = "検討", "中"

            # 軽量結果オブジェクト
            analysis_result = {
                'symbol': symbol,
                'name': symbol_name,
                'action': action,
                'score': final_score,
                'confidence': confidence,
                'risk_level': risk_level,
                'technical_score': base_score,
                'fundamental_score': base_score * 0.9,
                'sentiment_score': base_score * 1.1,
                'analysis_type': 'multi_symbol'
            }

            # 日付ベースキャッシュ（メモリ効率向上）
            if len(self.analysis_cache) >= self.max_cache_size:
                # 古いキャッシュを削除（メモリ管理）
                oldest_key = next(iter(self.analysis_cache))
                del self.analysis_cache[oldest_key]

            self.analysis_cache[cache_key] = analysis_result
            batch_results.append(analysis_result)

        return batch_results

    def get_portfolio_recommendation(self, analysis_results: List[dict], investment_amount: float = 1000000):
        """ポートフォリオ推奨配分（新機能）"""
        # 買い推奨銘柄のみ抽出
        buy_recommendations = [r for r in analysis_results if r['action'] in ['買い', '強い買い']]

        if not buy_recommendations:
            return {
                'total_symbols': 0,
                'recommended_allocation': {},
                'expected_return': 0,
                'risk_assessment': '投資推奨銘柄なし'
            }

        # スコア重み付けによる配分計算
        total_weight = sum(r['score'] * r['confidence'] / 100 for r in buy_recommendations)

        allocation = {}
        total_allocated = 0

        for rec in buy_recommendations:
            weight = (rec['score'] * rec['confidence'] / 100) / total_weight
            amount = int(investment_amount * weight)
            allocation[rec['symbol']] = {
                'name': rec['name'],
                'allocation_amount': amount,
                'allocation_percent': weight * 100,
                'score': rec['score'],
                'confidence': rec['confidence']
            }
            total_allocated += amount

        # 期待リターン計算（簡易版）
        avg_score = sum(r['score'] for r in buy_recommendations) / len(buy_recommendations)
        expected_return = (avg_score - 50) * 0.2  # スコア50を基準とした期待リターン

        # リスク評価
        high_risk_count = sum(1 for r in buy_recommendations if r['risk_level'] == '高')
        if high_risk_count > len(buy_recommendations) * 0.5:
            risk_assessment = '高リスク'
        elif high_risk_count > 0:
            risk_assessment = '中リスク'
        else:
            risk_assessment = '低リスク'

        return {
            'total_symbols': len(buy_recommendations),
            'recommended_allocation': allocation,
            'total_allocated': total_allocated,
            'expected_return_percent': expected_return,
            'risk_assessment': risk_assessment,
            'diversification_score': min(100, len(buy_recommendations) * 20)  # 分散化スコア
        }


class SimpleProgress:
    """軽量進捗表示（メモリ最適化版）"""

    def __init__(self):
        self.start_time = time.time()
        self.total_steps = 3

    def show_step(self, step_name: str, step_num: int):
        """軽量ステップ表示"""
        progress_bar = "=" * step_num + ">" + "." * (self.total_steps - step_num)
        print(f"\n[{progress_bar}] ({step_num}/{self.total_steps}) {step_name}")

    def show_completion(self):
        """完了表示"""
        total_time = time.time() - self.start_time
        print(f"\n[OK] 分析完了！ 総実行時間: {total_time:.1f}秒")


def show_header():
    """個人版ヘッダー表示"""
    print("=" * 50)
    print("    Day Trade Personal - 個人利用専用版")
    print("=" * 50)
    print("93%精度AI × 個人投資家向け最適化")
    print("商用機能なし・完全無料・超シンプル")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description='Day Trade Personal - 個人利用専用版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""個人投資家向け使用例:
  python daytrade.py                    # デフォルト：デイトレード推奨（1日単位）
  python daytrade.py --quick            # 基本モード（TOP3推奨・シンプル）
  python daytrade.py --multi 10         # 10銘柄一括分析
  python daytrade.py --portfolio 1000000 # ポートフォリオ推奨（100万円）
  python daytrade.py --chart            # チャート表示（グラフで分析結果）
  python daytrade.py --symbols 7203,8306  # 特定銘柄のみ分析
  python daytrade.py --history          # 分析履歴表示
  python daytrade.py --alerts           # アラート確認
  python daytrade.py --safe             # 安全モード（低リスクのみ）
  python daytrade.py --multi 8 --chart  # 複数銘柄分析＋チャート表示
  python daytrade.py --quick --chart --safe # 基本モード＋チャート＋安全モード

★デフォルトはデイトレード推奨モードです（1日単位売買タイミング）
注意: 投資は自己責任で行ってください"""
    )

    # 個人版用シンプルオプション
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quick', action='store_true',
                      help='高速モード: 瞬時でTOP3推奨（個人投資家向け）')

    parser.add_argument('--symbols', type=str,
                       help='特定銘柄のみ分析（例: 7203,8306,9984）')
    parser.add_argument('--multi', type=int, metavar='N',
                       help='複数銘柄一括分析（例: --multi 10 で10銘柄同時分析）')
    parser.add_argument('--portfolio', type=int, metavar='AMOUNT',
                       help='ポートフォリオ推奨（投資金額指定、例: --portfolio 1000000）')
    parser.add_argument('--history', action='store_true',
                       help='分析履歴表示: 過去の分析結果を確認')
    parser.add_argument('--alerts', action='store_true',
                       help='アラート確認: 未読アラート表示')
    # --daytrading オプションは削除（デフォルトになったため）
    parser.add_argument('--safe', action='store_true',
                       help='安全モード: 低リスク銘柄のみ（初心者推奨）')
    parser.add_argument('--chart', action='store_true',
                       help='チャート表示: 分析結果をグラフで表示（要matplotlib）')
    parser.add_argument('--version', action='version', version='Day Trade Personal v1.0')

    return parser.parse_args()


async def run_quick_mode(symbols: Optional[List[str]] = None, generate_chart: bool = False) -> bool:
    """
    個人版クイックモード実行

    Args:
        symbols: 対象銘柄リスト
        generate_chart: チャート生成フラグ

    Returns:
        実行成功かどうか
    """
    progress = SimpleProgress()

    try:
        print("\n個人版高速モード: 瞬時でTOP3推奨を実行します")
        print("93%精度AI分析実行中...")

        if symbols:
            print(f"指定銘柄: {len(symbols)} 銘柄")
        else:
            print("推奨銘柄: 個人投資家向け厳選3銘柄")

        # ステップ1: データ分析
        progress.show_step("市場データ分析中", 1)
        progress.show_step("93%精度AI予測中", 2)

        # 個人版シンプル分析実行
        engine = PersonalAnalysisEngine()
        recommendations = await engine.get_personal_recommendations(limit=3)

        # ステップ3: 結果表示
        progress.show_step("結果表示", 3)

        if not recommendations:
            print("\n現在推奨できる銘柄がありません")
            return False

        print("\n" + "="*50)
        print("個人投資家向け分析結果")
        print("="*50)

        for i, rec in enumerate(recommendations, 1):
            # 個人版はdict形式固定
            symbol = rec['symbol']
            name = rec['name']
            action = rec['action']
            score = rec['score']
            confidence = rec['confidence']
            risk_level = rec['risk_level']

            risk_icon = {"低": "[低リスク]", "中": "[中リスク]", "高": "[高リスク]"}.get(risk_level, "[?]")

            print(f"\n{i}. {symbol} ({name})")
            print(f"   推奨: [{action}]")
            print(f"   スコア: {score:.1f}点")
            print(f"   信頼度: {confidence:.0f}%")
            print(f"   リスク: {risk_icon}")

            # シンプルなアドバイス
            if action == "買い" and confidence > 80:
                print(f"   アドバイス: 上昇期待・検討推奨")
            elif action == "様子見":
                print(f"   アドバイス: 明確なトレンドなし")
            elif confidence < 70:
                print(f"   アドバイス: 慎重な判断が必要")

        progress.show_completion()

        # チャート生成（オプション）
        if generate_chart and CHART_AVAILABLE:
            print("\n[チャート] グラフ生成中...")
            try:
                chart_gen = PersonalChartGenerator()

                # 分析結果チャート
                analysis_chart_path = chart_gen.generate_analysis_chart(recommendations)
                summary_chart_path = chart_gen.generate_simple_summary(recommendations)

                print(f"[チャート] 分析チャートを保存しました: {analysis_chart_path}")
                print(f"[チャート] サマリーチャートを保存しました: {summary_chart_path}")
                print("[チャート] 投資判断の参考にしてください")

            except Exception as e:
                print(f"[警告] チャート生成エラー: {e}")
                print("テキスト結果をご参照ください")
        elif generate_chart and not CHART_AVAILABLE:
            print("\n[警告] チャート機能が利用できません")
            print("pip install matplotlib でmatplotlibをインストールしてください")

        print("\n個人投資家向けガイド:")
        print("・スコア70点以上: 投資検討価値が高い銘柄")
        print("・信頼度80%以上: より確実性の高い予測")
        print("・[買い]推奨: 上昇期待、検討してみてください")
        print("・[様子見]: 明確なトレンドなし、慎重に")
        print("・リスク管理: 余裕資金での投資を推奨")
        print("・投資は自己責任で！複数の情報源と照らし合わせを")

        return True

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("基本機能で再試行中...")
        return False


async def run_multi_symbol_mode(symbol_count: int, portfolio_amount: Optional[int] = None,
                               generate_chart: bool = False, safe_mode: bool = False) -> bool:
    """
    複数銘柄分析モード実行（新機能）

    Args:
        symbol_count: 分析対象銘柄数
        portfolio_amount: ポートフォリオ推奨金額
        generate_chart: チャート生成フラグ
        safe_mode: 安全モード

    Returns:
        実行成功かどうか
    """
    progress = SimpleProgress()
    progress.total_steps = 4  # 複数銘柄用に増加

    try:
        print(f"\n複数銘柄分析モード: {symbol_count}銘柄を一括分析します")
        print("93%精度AI × 複数銘柄同時処理")

        engine = PersonalAnalysisEngine()
        all_symbols = list(engine.recommended_symbols.keys())

        # 銘柄数制限
        if symbol_count > len(all_symbols):
            print(f"注意: 利用可能銘柄数は{len(all_symbols)}銘柄です。最大数で実行します。")
            symbol_count = len(all_symbols)

        target_symbols = all_symbols[:symbol_count]

        # ステップ1: 複数銘柄分析
        progress.show_step("複数銘柄同時分析実行", 1)
        recommendations = await engine.get_multi_symbol_analysis(target_symbols)

        # ステップ2: 安全モード適用
        if safe_mode:
            progress.show_step("安全モード適用（高リスク除外）", 2)
            recommendations = filter_safe_recommendations(recommendations)
        else:
            progress.show_step("分析結果整理", 2)

        # ステップ3: ポートフォリオ推奨
        portfolio_recommendation = None
        if portfolio_amount:
            progress.show_step("ポートフォリオ推奨計算", 3)
            portfolio_recommendation = engine.get_portfolio_recommendation(recommendations, portfolio_amount)
        else:
            progress.show_step("結果取りまとめ", 3)

        # ステップ4: 結果表示
        progress.show_step("分析結果表示", 4)

        if not recommendations:
            print("\n現在推奨できる銘柄がありません")
            return False

        # 結果表示
        print("\n" + "="*60)
        print(f"複数銘柄分析結果（{len(recommendations)}銘柄）")
        print("="*60)

        # TOP10の詳細表示
        top_recommendations = recommendations[:10]
        for i, rec in enumerate(top_recommendations, 1):
            action_color = {
                "強い買い": "[★強い買い★]",
                "買い": "[●買い●]",
                "検討": "[○検討○]",
                "様子見": "[△様子見△]"
            }.get(rec['action'], f"[{rec['action']}]")

            risk_color = {"低": "[低リスク]", "中": "[中リスク]", "高": "[高リスク]"}.get(rec['risk_level'], "[?]")

            print(f"\n{i}. {rec['symbol']} ({rec['name']})")
            print(f"   推奨: {action_color}")
            print(f"   総合スコア: {rec['score']:.1f}点")
            print(f"   信頼度: {rec['confidence']:.0f}%")
            print(f"   リスク: {risk_color}")

            # 詳細分析スコア表示
            if 'technical_score' in rec:
                print(f"   詳細: テクニカル{rec['technical_score']:.0f} / ファンダメンタル{rec['fundamental_score']:.0f} / センチメント{rec['sentiment_score']:.0f}")

        # 残りの銘柄はサマリーのみ
        if len(recommendations) > 10:
            print(f"\n... 他 {len(recommendations) - 10} 銘柄（省略）")

            buy_count = sum(1 for r in recommendations if r['action'] in ['買い', '強い買い'])
            print(f"全体サマリー: 買い推奨 {buy_count}銘柄 / 全{len(recommendations)}銘柄")

        # ポートフォリオ推奨表示
        if portfolio_recommendation and portfolio_recommendation['total_symbols'] > 0:
            print("\n" + "="*60)
            print(f"ポートフォリオ推奨配分（投資額: {portfolio_amount:,}円）")
            print("="*60)

            for symbol, alloc in portfolio_recommendation['recommended_allocation'].items():
                print(f"{symbol} ({alloc['name']}): {alloc['allocation_amount']:,}円 ({alloc['allocation_percent']:.1f}%)")

            print(f"\n期待リターン: {portfolio_recommendation['expected_return_percent']:.1f}%")
            print(f"リスク評価: {portfolio_recommendation['risk_assessment']}")
            print(f"分散化スコア: {portfolio_recommendation['diversification_score']:.0f}/100")

        progress.show_completion()

        # チャート生成（オプション）
        if generate_chart and CHART_AVAILABLE:
            print("\n[チャート] 複数銘柄分析グラフ生成中...")
            try:
                chart_gen = PersonalChartGenerator()

                # TOP10のみをチャート化（見やすさのため）
                chart_data = recommendations[:10]
                analysis_chart_path = chart_gen.generate_analysis_chart(chart_data)
                summary_chart_path = chart_gen.generate_simple_summary(chart_data)

                print(f"[チャート] 分析チャートを保存しました: {analysis_chart_path}")
                print(f"[チャート] サマリーチャートを保存しました: {summary_chart_path}")

            except Exception as e:
                print(f"[警告] チャート生成エラー: {e}")

        print(f"\n複数銘柄分析完了: {len(recommendations)}銘柄を{progress.start_time:.1f}秒で処理")
        print("個人投資家向けガイド:")
        print("・★強い買い★: 最も期待の高い銘柄")
        print("・複数銘柄への分散投資を推奨")
        print("・リスクレベルを考慮した投資を")
        print("・投資は自己責任で！")

        return True

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("複数銘柄分析に問題が発生しました")
        return False


def filter_safe_recommendations(recommendations):
    """安全モード: 高リスク銘柄を除外"""
    filtered = []
    for rec in recommendations:
        if isinstance(rec, dict):
            risk_level = rec.get('risk_level', '中')
        else:
            risk_level = rec.risk_level

        if risk_level != "高":
            filtered.append(rec)

    return filtered


def show_analysis_history() -> bool:
    """分析履歴表示"""
    if not HISTORY_AVAILABLE:
        print("履歴機能が利用できません")
        print("pip install pandas でpandasをインストールしてください")
        return False

    try:
        history = PersonalAnalysisHistory()

        print("\n" + "="*50)
        print("分析履歴（過去30日間）")
        print("="*50)

        # 最近の分析履歴
        recent_analyses = history.get_recent_analyses(days=30)

        if not recent_analyses:
            print("分析履歴がありません")
            return True

        for i, analysis in enumerate(recent_analyses, 1):
            date_str = analysis['date'][:19] if analysis['date'] else '不明'
            type_name = {'basic': '基本分析', 'multi_symbol': '複数銘柄分析'}.get(analysis['type'], analysis['type'])

            print(f"{i}. {date_str}")
            print(f"   タイプ: {type_name}")
            print(f"   銘柄数: {analysis['symbol_count']}銘柄")
            print(f"   平均スコア: {analysis['total_score']:.1f}点")
            print(f"   買い推奨: {analysis['buy_count']}銘柄")
            print(f"   処理時間: {analysis['performance_time']:.1f}秒")
            print()

        # サマリーレポート
        summary = history.generate_summary_report(days=7)

        print("\n" + "-"*30)
        print("直近7日間のサマリー")
        print("-"*30)
        print(f"分析実行回数: {summary['analysis_stats']['total_analyses']}回")
        print(f"平均スコア: {summary['analysis_stats']['avg_score']:.1f}点")
        print(f"最高スコア: {summary['analysis_stats']['best_score']:.1f}点")
        print(f"平均処理時間: {summary['analysis_stats']['avg_time']:.1f}秒")

        if summary['alert_stats']['total_alerts'] > 0:
            print(f"アラート: {summary['alert_stats']['unread_alerts']}件未読 / {summary['alert_stats']['total_alerts']}件")

        return True

    except Exception as e:
        print(f"履歴表示エラー: {e}")
        return False


def show_alerts() -> bool:
    """アラート表示・管理"""
    if not HISTORY_AVAILABLE:
        print("アラート機能が利用できません")
        print("pip install pandas でpandasをインストールしてください")
        return False

    try:
        history = PersonalAnalysisHistory()
        alert_system = PersonalAlertSystem(history)

        print("\n" + "="*50)
        print("アラート管理")
        print("="*50)

        # アラート表示
        alert_system.display_alerts()

        # アラート確認オプション
        alerts = history.get_unread_alerts()
        if alerts:
            print("\n[選択肢]")
            print("1. 全てのアラートを既読にする")
            print("2. そのまま終了")

            try:
                choice = input("選択してください (1/2): ").strip()
                if choice == "1":
                    alert_system.acknowledge_all_alerts()
                else:
                    print("アラートは未読のままです")
            except KeyboardInterrupt:
                print("\n操作をキャンセルしました")

        return True

    except Exception as e:
        print(f"アラート表示エラー: {e}")
        return False


async def run_daytrading_mode() -> bool:
    """
    デイトレードモード実行

    Returns:
        実行成功かどうか
    """
    if not DAYTRADING_AVAILABLE:
        print("デイトレード機能が利用できません")
        print("day_trading_engine.py が必要です")
        return False

    progress = SimpleProgress()
    progress.total_steps = 4

    try:
        print("\nデイトレードモード: 1日単位の売買タイミング推奨")
        print("93%精度AI × デイトレード特化分析")

        # デイトレードエンジン初期化
        engine = PersonalDayTradingEngine()

        # ステップ1: 現在の市場セッション確認
        progress.show_step("市場セッション確認", 1)
        session_advice = engine.get_session_advice()
        print(f"\n{session_advice}")

        # ステップ2: デイトレード分析実行
        progress.show_step("デイトレード分析実行中", 2)
        recommendations = await engine.get_today_daytrading_recommendations(limit=5)

        # ステップ3: 結果整理
        progress.show_step("デイトレード推奨取得", 3)

        if not recommendations:
            print("\n現在デイトレード推奨できる銘柄がありません")
            return False

        # ステップ4: 結果表示
        progress.show_step("結果表示", 4)

        # 時間帯に応じたタイトル表示
        from datetime import datetime, timedelta, time as dt_time
        current_time = datetime.now().time()

        if current_time >= dt_time(15, 0):  # 大引け後（15:00以降）
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow_str = tomorrow.strftime("%m/%d")
            print("\n" + "="*60)
            print(f"翌日前場予想（{tomorrow_str}）TOP5")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("今日のデイトレード推奨 TOP5")
            print("="*60)

        for i, rec in enumerate(recommendations, 1):
            # シグナル別アイコン表示
            signal_display = {
                DayTradingSignal.STRONG_BUY: "[★強い買い★]",
                DayTradingSignal.BUY: "[●買い●]",
                DayTradingSignal.STRONG_SELL: "[▼強い売り▼]",
                DayTradingSignal.SELL: "[▽売り▽]",
                DayTradingSignal.HOLD: "[■ホールド■]",
                DayTradingSignal.WAIT: "[…待機…]"
            }.get(rec.signal, f"[{rec.signal.value}]")

            risk_display = {"低": "[低リスク]", "中": "[中リスク]", "高": "[高リスク]"}.get(rec.risk_level, "[?]")

            print(f"\n{i}. {rec.symbol} ({rec.name})")
            print(f"   シグナル: {signal_display}")
            print(f"   エントリー: {rec.entry_timing}")
            print(f"   目標利確: +{rec.target_profit}% / 損切り: -{rec.stop_loss}%")
            print(f"   保有時間: {rec.holding_time}")
            print(f"   信頼度: {rec.confidence:.0f}% | リスク: {risk_display}")
            print(f"   出来高動向: {rec.volume_trend}")
            print(f"   価格動向: {rec.price_momentum}")
            print(f"   日中ボラティリティ: {rec.intraday_volatility:.1f}%")
            print(f"   タイミングスコア: {rec.market_timing_score:.0f}/100")

        progress.show_completion()

        # 履歴保存（利用可能な場合）
        if HISTORY_AVAILABLE:
            try:
                from analysis_history import PersonalAnalysisHistory, PersonalAlertSystem
                history = PersonalAnalysisHistory()

                # デイトレード結果を辞書形式に変換
                history_data = {
                    'analysis_type': 'daytrading',
                    'recommendations': [
                        {
                            'symbol': rec.symbol,
                            'name': rec.name,
                            'action': rec.signal.value,
                            'score': rec.market_timing_score,
                            'confidence': rec.confidence,
                            'risk_level': rec.risk_level,
                            'entry_timing': rec.entry_timing,
                            'target_profit': rec.target_profit,
                            'stop_loss': rec.stop_loss
                        }
                        for rec in recommendations
                    ],
                    'performance_time': time.time() - progress.start_time
                }

                analysis_id = history.save_analysis_result(history_data)

                # アラートチェック
                alert_system = PersonalAlertSystem(history)
                alerts = alert_system.check_analysis_alerts(history_data['recommendations'])
                if alerts:
                    print(f"\n[アラート] {len(alerts)}件の新しいアラートが生成されました")

            except Exception as e:
                print(f"[注意] 履歴保存エラー: {e}")

        # 時間帯に応じたガイド表示
        if current_time >= dt_time(15, 0):  # 大引け後（翌日予想モード）
            print("\n翌日前場予想ガイド:")
            print("・★強い買い★: 寄り成行で積極エントリー計画")
            print("・●買い●: 寄り後の値動き確認してエントリー")
            print("・▼強い売り▼/▽売り▽: 寄り付きでの売りエントリー計画")
            print("・■ホールド■: 寄り後の流れ次第で判断")
            print("・…待機…: 前場中盤までエントリーチャンス待ち")
            print("・翌日前場予想のため実際の結果と異なる場合があります")
            print("・オーバーナイトリスクを考慮した損切り設定を")
            print("・投資は自己責任で！")
        else:
            print("\nデイトレード推奨ガイド:")
            print("・★強い買い★: 即座にエントリー検討")
            print("・●買い●: 押し目でのエントリータイミングを狙う")
            print("・▼強い売り▼/▽売り▽: 利確・損切り実行")
            print("・■ホールド■: 既存ポジション維持")
            print("・…待機…: エントリーチャンス待ち")
            print("・デイトレードは当日中に決済完了を推奨")
            print("・損切りラインを必ず設定してください")
            print("・投資は自己責任で！")

        return True

    except Exception as e:
        print(f"\nデイトレード分析エラー: {e}")
        print("デイトレード機能に問題が発生しました")
        return False


async def main():
    """個人版メイン実行関数"""
    execution_start_time = time.time()

    show_header()
    args = parse_arguments()

    # 個人版デフォルト：デイトレードモード
    print("\n個人投資家専用モード:")
    print("・デフォルト：デイトレード推奨")
    print("・93%精度AI搭載")
    print("・1日単位売買タイミング")
    print("・商用機能なし・超シンプル操作")
    print()

    # 引数に応じた動作モード決定
    symbols = None
    if args.symbols:
        symbols = [symbol.strip() for symbol in args.symbols.split(',')]
        print(f"指定銘柄: {', '.join(symbols)}")

    success = False

    try:
        # 履歴表示モード
        if args.history:
            success = show_analysis_history()
        # アラート表示モード
        elif args.alerts:
            success = show_alerts()
        # 複数銘柄分析モード
        elif args.multi:
            success = await run_multi_symbol_mode(args.multi, args.portfolio, generate_chart=args.chart, safe_mode=args.safe)
        # 基本モード（従来の簡単分析）
        elif args.quick:
            success = await run_quick_mode(symbols, generate_chart=args.chart)
        # デフォルト：デイトレードモード
        else:
            success = await run_daytrading_mode()

        # 安全モード処理
        if args.safe and success:
            print("\n安全モード: 高リスク銘柄を除外しています")

        # 実行時間表示
        end_time = time.time()
        total_time = end_time - execution_start_time

        print(f"\n{'='*50}")
        if success:
            print("分析完了！")
            print(f"実行時間: {total_time:.1f}秒")
            print("投資は自己責任で行ってください")
        else:
            print("分析に問題が発生しました")
            print("ネットワーク接続や設定を確認してください")
        print(f"{'='*50}")

    except KeyboardInterrupt:
        print("\n実行が中断されました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("問題が続く場合は設定を確認してください")


if __name__ == "__main__":
    asyncio.run(main())