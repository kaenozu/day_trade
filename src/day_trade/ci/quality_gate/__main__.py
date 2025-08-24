#!/usr/bin/env python3
"""
品質ゲートシステム - CLI インターフェース

コマンドライン実行用のエントリーポイント。
モジュール化されたシステムの統一的な実行インターフェースを提供する。
"""

import argparse
import asyncio
import json
import sys

from .system import AdvancedQualityGateSystem


async def main():
    """メイン実行関数
    
    コマンドライン引数を解析し、品質チェックを実行する。
    """
    parser = argparse.ArgumentParser(
        description="Advanced Quality Gate System - Modular Version",
        epilog="Example: python -m day_trade.ci.quality_gate --project-root . --format markdown"
    )
    
    parser.add_argument(
        "--project-root", 
        default=".", 
        help="Project root directory (default: current directory)"
    )
    
    parser.add_argument(
        "--output", 
        help="Output report file path (default: stdout)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "detailed"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    
    parser.add_argument(
        "--history",
        type=int,
        help="Show quality history for specified days"
    )
    
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Cleanup old reports older than specified days"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )

    args = parser.parse_args()

    try:
        # 品質システム初期化
        quality_system = AdvancedQualityGateSystem(args.project_root)
        
        # 統計情報表示
        if args.stats:
            stats = quality_system.database.get_statistics()
            print("📊 Database Statistics:")
            print(f"  Total reports: {stats.get('total_reports', 0)}")
            print(f"  Latest report: {stats.get('latest_report', 'N/A')}")
            print(f"  Oldest report: {stats.get('oldest_report', 'N/A')}")
            print(f"  Average score: {stats.get('average_score', 0):.1f}")
            return
            
        # 履歴表示
        if args.history:
            history = quality_system.get_quality_history(args.history)
            print(f"📈 Quality History (last {args.history} days):")
            for entry in history:
                print(f"  {entry['timestamp']}: {entry['overall_score']:.1f} ({entry['overall_level']})")
            return
            
        # 古いデータのクリーンアップ
        if args.cleanup:
            quality_system.cleanup_old_data(args.cleanup)
            print(f"🧹 Cleaned up data older than {args.cleanup} days")
            return

        # 品質チェック実行
        print("🔍 Starting comprehensive quality check...")
        report = await quality_system.run_comprehensive_quality_check()

        # レポート出力の生成
        if args.format == "json":
            output = json.dumps(report, default=str, indent=2, ensure_ascii=False)
        elif args.format == "detailed":
            output = quality_system.generate_detailed_report(report)
        else:  # markdown
            output = quality_system.generate_ci_report(report)

        # 出力先の決定と書き込み
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"📝 Report saved to: {args.output}")
        else:
            print("\n" + "="*50)
            print(output)

        # 終了コードの設定
        if report["overall_level"] in ["critical", "needs_improvement"]:
            print(f"\n❌ Quality check failed: {report['overall_level']}")
            sys.exit(1)
        else:
            print(f"\n✅ Quality check passed: {report['overall_level']}")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⏸️  Quality check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error during quality check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())