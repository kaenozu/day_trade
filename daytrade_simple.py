#!/usr/bin/env python3
"""
DayTrade自動分析 - シンプル実行インターフェース

Issue #457: なにも考えずに実行できるシンプルなインターフェース
使用方法: python daytrade_simple.py

Features:
- ゼロコンフィグ実行
- リアルタイム進捗表示
- 明確な推奨銘柄出力
- 初心者向けのわかりやすい説明
"""

import asyncio
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# モジュールインポート
from src.day_trade.automation.auto_pipeline_manager import run_auto_pipeline
from src.day_trade.recommendation.recommendation_engine import get_daily_recommendations


class SimpleProgress:
    """シンプル進捗表示"""
    
    def __init__(self):
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 3
        
    def show_step(self, step_name: str, step_num: int):
        """ステップ表示"""
        self.current_step = step_num
        elapsed = time.time() - self.start_time
        
        progress_bar = "=" * step_num + ">" + "." * (self.total_steps - step_num)
        print(f"\n[{progress_bar}] ({step_num}/{self.total_steps}) {step_name}")
        print(f"経過時間: {elapsed:.1f}秒")
        
    def show_completion(self):
        """完了表示"""
        total_time = time.time() - self.start_time
        print(f"\n[===] 完了! 総実行時間: {total_time:.1f}秒")


def print_header():
    """ヘッダー表示"""
    print("=" * 60)
    print("          DayTrade 自動分析システム")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_recommendation_result(recommendation):
    """推奨結果の表示"""
    # アクション表示のマッピング（絵文字なし）
    action_display = {
        "🔥 今すぐ買い": "[強い買い]",
        "📈 買い": "[買い]",
        "⏸️ 様子見": "[様子見]",
        "📉 売り": "[売り]",
        "⚠️ 今すぐ売り": "[強い売り]"
    }
    
    action_text = action_display.get(recommendation.action.value, recommendation.action.value)
    
    print(f"   {recommendation.symbol} ({recommendation.name})")
    print(f"   推奨度: {recommendation.composite_score:.0f}点")
    print(f"   アクション: {action_text}")
    print(f"   信頼度: {recommendation.confidence:.0f}%")
    print(f"   リスク: {recommendation.risk_level}")
    
    if recommendation.reasons:
        print(f"   理由: {', '.join(recommendation.reasons[:2])}")
    
    if recommendation.price_target:
        print(f"   目標価格: {recommendation.price_target:.0f}円")
    if recommendation.stop_loss:
        print(f"   ストップロス: {recommendation.stop_loss:.0f}円")


async def run_quick_mode(symbols: Optional[List[str]] = None) -> bool:
    """
    高速モード実行
    
    Args:
        symbols: 対象銘柄（指定なしの場合は主要5銘柄）
        
    Returns:
        実行成功可否
    """
    progress = SimpleProgress()
    
    try:
        # デフォルト銘柄設定
        if symbols is None:
            symbols = ["7203", "8306", "9984", "6758", "4689"]  # 主要5銘柄
        
        print("高速モード実行中...")
        print(f"対象銘柄: {len(symbols)} 銘柄")
        
        # ステップ1: データ収集
        progress.show_step("最新データ収集中", 1)
        
        # ステップ2: AI分析
        progress.show_step("AI分析・推奨計算中", 2)
        
        # 推奨銘柄取得
        recommendations = await get_daily_recommendations(3)  # TOP3
        
        # ステップ3: 結果表示
        progress.show_step("結果表示", 3)
        progress.show_completion()
        
        # 結果表示
        print("\n" + "=" * 50)
        print("         今日の推奨銘柄 TOP3")
        print("=" * 50)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. ")
                print_recommendation_result(rec)
        else:
            print("\n推奨銘柄が見つかりませんでした。")
            print("市場状況により推奨できる銘柄がない可能性があります。")
        
        # 簡単なアドバイス
        print("\n" + "=" * 50)
        print("         取引アドバイス")
        print("=" * 50)
        print("・[強い買い]は積極的な投資候補です")
        print("・[買い]は慎重な投資候補です")
        print("・目標価格とストップロスを参考に取引してください")
        print("・リスク管理を忘れずに!")
        
        return True
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("システム管理者にお問い合わせください。")
        return False


async def run_full_mode(symbols: Optional[List[str]] = None) -> bool:
    """
    フルモード実行（自動パイプライン使用）
    
    Args:
        symbols: 対象銘柄
        
    Returns:
        実行成功可否
    """
    progress = SimpleProgress()
    
    try:
        print("フルモード実行中...")
        if symbols:
            print(f"対象銘柄: {len(symbols)} 銘柄")
        else:
            print("全銘柄対象")
        
        # ステップ1: データ収集・学習
        progress.show_step("データ収集・AI学習中", 1)
        
        # 自動パイプライン実行
        result = await run_auto_pipeline(symbols)
        
        if not result.success:
            print(f"\nパイプライン実行エラー: {result.error_message}")
            return False
        
        # ステップ2: 推奨分析
        progress.show_step("推奨銘柄分析中", 2)
        
        # ステップ3: 結果表示
        progress.show_step("結果表示", 3)
        progress.show_completion()
        
        # 詳細結果表示
        print("\n" + "=" * 60)
        print("           実行結果詳細")
        print("=" * 60)
        
        print(f"データ収集: {len(result.data_collection.collected_symbols)} 銘柄成功")
        print(f"AI学習: {len(result.model_update.models_updated)} モデル更新")
        print(f"データ品質: {result.quality_report.overall_score:.1f}点")
        print(f"推奨生成: {result.recommendations_generated} 件")
        
        # TOP5推奨表示
        recommendations = await get_daily_recommendations(5)  # TOP5
        
        print("\n" + "=" * 50)
        print("         今日の推奨銘柄 TOP5")
        print("=" * 50)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. ")
                print_recommendation_result(rec)
        else:
            print("\n推奨銘柄が見つかりませんでした。")
        
        return True
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("詳細なエラー情報については、ログファイルを確認してください。")
        return False


def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="DayTrade自動分析 - シンプル実行インターフェース",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python daytrade_simple.py                    # 高速モード（TOP3推奨）
  python daytrade_simple.py --full             # フルモード（全銘柄分析）
  python daytrade_simple.py --quick            # 高速モード明示
  python daytrade_simple.py --symbols 7203,8306  # 特定銘柄のみ
  python daytrade_simple.py --safe             # 安全モード（低リスク銘柄のみ）
        """
    )
    
    # 実行モード
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", 
        action="store_true",
        help="高速モード: 主要銘柄のみでTOP3推奨（デフォルト）"
    )
    mode_group.add_argument(
        "--full",
        action="store_true", 
        help="フルモード: 全銘柄分析でTOP5推奨（時間がかかります）"
    )
    
    # 銘柄指定
    parser.add_argument(
        "--symbols",
        type=str,
        help="分析対象銘柄（カンマ区切り）例: 7203,8306,9984"
    )
    
    # 安全モード
    parser.add_argument(
        "--safe",
        action="store_true",
        help="安全モード: 高リスク銘柄を除外"
    )
    
    # バージョン情報
    parser.add_argument(
        "--version",
        action="version",
        version="DayTrade Simple Interface v1.0"
    )
    
    return parser.parse_args()


async def main():
    """メイン実行関数"""
    print_header()
    
    # 引数解析
    args = parse_arguments()
    
    # 銘柄リスト処理
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        print(f"指定銘柄: {', '.join(symbols)}")
    
    # 安全モード処理
    if args.safe:
        print("安全モード: 高リスク銘柄を除外します")
        # 安全銘柄のみに制限
        if symbols is None:
            symbols = ["7203", "8306", "9983"]  # 大型安定株
    
    # 実行モード決定
    success = False
    
    try:
        if args.full:
            print("フルモード: 全銘柄分析を実行します")
            success = await run_full_mode(symbols)
        else:
            print("高速モード: 主要銘柄のTOP3推奨を実行します")
            success = await run_quick_mode(symbols)
        
        # 結果サマリー
        print("\n" + "=" * 60)
        if success:
            print("[OK] 実行完了しました!")
            print("\n投資は自己責任で行ってください。")
            print("このシステムの推奨は参考情報であり、")
            print("投資判断は必ずご自身で行ってください。")
        else:
            print("[ERROR] 実行に失敗しました。")
            print("設定やネットワーク接続を確認してください。")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n実行が中断されました。")
    except Exception as e:
        print(f"\n予期しないエラーが発生しました: {e}")
        success = False
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)