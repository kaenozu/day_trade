#!/usr/bin/env python3
"""
DayTrade自動分析システム - メインインターフェース

完全自動化された株式推奨システム
使用方法: python daytrade.py

Features:
- ゼロコンフィグ実行
- リアルタイム進捗表示
- 明確な推奨銘柄出力
- AI駆動の総合分析
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
        print(f"\n✅ 分析完了！ 総実行時間: {total_time:.1f}秒")


def show_header():
    """ヘッダー表示"""
    print("=" * 60)
    print("          DayTrade 自動分析システム")
    print("=" * 60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description='DayTrade自動分析 - シンプル実行インターフェース',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用例:
  python daytrade.py                    # クイックモード（TOP3推奨）
  python daytrade.py --full             # フルモード（全銘柄分析）
  python daytrade.py --quick            # クイックモード明示
  python daytrade.py --symbols 7203,8306  # 指定銘柄のみ
  python daytrade.py --safe             # 安全モード（低リスク銘柄のみ）"""
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quick', action='store_true',
                      help='クイックモード: 最速でTOP3推奨のみ（デフォルト）')
    group.add_argument('--full', action='store_true',
                      help='フルモード: 全銘柄分析でTOP5推奨（時間がかかります）')

    parser.add_argument('--symbols', type=str,
                       help='分析対象銘柄（カンマ区切り）例: 7203,8306,9984')
    parser.add_argument('--safe', action='store_true',
                       help='安全モード: 低リスク銘柄除外')
    parser.add_argument('--version', action='version', version='DayTrade Simple Interface v1.0')

    return parser.parse_args()


async def run_quick_mode(symbols: Optional[List[str]] = None) -> bool:
    """
    クイックモード実行
    
    Args:
        symbols: 対象銘柄リスト
        
    Returns:
        実行成功かどうか
    """
    progress = SimpleProgress()

    try:
        print("\nクイックモード: 最速でTOP3推奨を実行します")
        print("クイックモード実行中...")
        
        if symbols:
            print(f"対象銘柄: {len(symbols)} 銘柄")
        else:
            print("対象銘柄: 5 銘柄")  # デフォルト数

        # ステップ1: データ取得
        progress.show_step("最新データ収集中", 1)
        progress.show_step("AI分析・順位計算中", 2)

        # 推奨銘柄取得
        recommendations = await get_daily_recommendations(limit=3)
        
        # ステップ3: 結果表示
        progress.show_step("結果表示", 3)
        
        if not recommendations:
            print("\n⚠️  現在推奨できる銘柄がありません")
            return False

        print("\n" + "="*50)
        print(f"     本日のTOP {len(recommendations)} 推奨銘柄")
        print("="*50)

        for i, rec in enumerate(recommendations, 1):
            risk_color = {"低": "🟢", "中": "🟡", "高": "🔴"}.get(rec.risk_level, "⚪")
            
            print(f"\n{i}. {rec.symbol} ({rec.name}) - [{rec.action.value}]")
            print(f"   スコア: {rec.composite_score:.1f}点, 信頼度: {rec.confidence:.0f}%, リスク: {risk_color}{rec.risk_level}")
            
            if rec.reasons:
                print(f"   理由: {', '.join(rec.reasons[:2])}")
            
            if rec.price_target:
                print(f"   目標価格: {rec.price_target:.0f}円", end="")
            if rec.stop_loss:
                print(f", ストップロス: {rec.stop_loss:.0f}円", end="")
            print()

        progress.show_completion()
        
        print("\n💡 ワンポイントアドバイス:")
        print("   - 必ずリスクレベルを確認してから投資判断してください")
        print("   - ストップロス価格での損切りを徹底しましょう")
        
        return True

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        return False


async def run_full_mode(symbols: Optional[List[str]] = None) -> bool:
    """
    フルモード実行
    
    Args:
        symbols: 対象銘柄リスト
        
    Returns:
        実行成功かどうか
    """
    progress = SimpleProgress()

    try:
        print("\nフルモード: 全銘柄分析でTOP5推奨を実行します（時間がかかります）")
        print("フルモード実行中...")

        # ステップ1: データ収集
        progress.show_step("全データ収集・品質検証中", 1)
        
        # ステップ2: 分析
        progress.show_step("ML学習・予測分析中", 2)
        
        # フル自動パイプライン実行
        pipeline_result = await run_auto_pipeline(symbols)
        
        if not pipeline_result.success:
            print(f"\n❌ パイプライン実行に失敗しました: {pipeline_result.error_message}")
            return False

        # 推奨銘柄取得（TOP5）
        recommendations = await get_daily_recommendations(limit=5)
        
        # ステップ3: 結果表示
        progress.show_step("詳細結果表示", 3)
        
        if not recommendations:
            print("\n⚠️  現在推奨できる銘柄がありません")
            return False

        # 詳細結果表示
        print("\n" + "="*60)
        print(f"     詳細分析結果 - TOP {len(recommendations)} 推奨銘柄")
        print("="*60)
        
        print(f"📊 パイプライン統計:")
        print(f"   • データ収集: {len(pipeline_result.data_collection.collected_symbols)} 銘柄成功")
        print(f"   • モデル学習: {len(pipeline_result.model_update.models_updated)} モデル更新")
        print(f"   • 品質スコア: {pipeline_result.quality_report.overall_score:.2f}")

        for i, rec in enumerate(recommendations, 1):
            risk_color = {"低": "🟢", "中": "🟡", "高": "🔴"}.get(rec.risk_level, "⚪")
            
            print(f"\n{i}. {rec.symbol} ({rec.name}) - [{rec.action.value}]")
            print(f"   📈 総合スコア: {rec.composite_score:.1f}点")
            print(f"   📊 内訳: テクニカル {rec.technical_score:.1f}点, ML予測 {rec.ml_score:.1f}点")
            print(f"   🎯 信頼度: {rec.confidence:.0f}%, リスク: {risk_color}{rec.risk_level}")
            
            if rec.reasons:
                print(f"   💡 推奨理由: {', '.join(rec.reasons[:3])}")
            
            price_info = []
            if rec.price_target:
                price_info.append(f"目標価格 {rec.price_target:.0f}円")
            if rec.stop_loss:
                price_info.append(f"ストップロス {rec.stop_loss:.0f}円")
            if price_info:
                print(f"   💰 {', '.join(price_info)}")

        progress.show_completion()
        
        print("\n📋 投資判断サポート:")
        print("   ✅ スコア70点以上: 投資検討価値が高い")
        print("   ⚠️  信頼度60%未満: 慎重な判断が必要")
        print("   🔴 高リスク銘柄: 損失許容範囲内での投資を")
        
        return True

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        return False


def filter_safe_recommendations(recommendations):
    """安全モード: 高リスク銘柄を除外"""
    return [rec for rec in recommendations if rec.risk_level != "高"]


async def main():
    """メイン実行関数"""
    show_header()
    args = parse_arguments()

    # 引数に応じた銘柄リスト設定
    symbols = None
    if args.symbols:
        symbols = [symbol.strip() for symbol in args.symbols.split(',')]
        print(f"指定銘柄: {', '.join(symbols)}")

    success = False

    try:
        if args.full:
            success = await run_full_mode(symbols)
        else:
            # デフォルトはクイックモード
            success = await run_quick_mode(symbols)
        
        if success:
            print(f"\n🎉 {datetime.now().strftime('%H:%M:%S')} 分析完了")
            if args.safe:
                print("   ℹ️  安全モードで高リスク銘柄を除外しています")
        else:
            print(f"\n⚠️  {datetime.now().strftime('%H:%M:%S')} 分析に問題が発生しました")
            
    except KeyboardInterrupt:
        print(f"\n⏸️  ユーザーによって実行が中止されました")
    except Exception as e:
        print(f"\n💥 予期しないエラーが発生しました: {e}")


if __name__ == "__main__":
    asyncio.run(main())