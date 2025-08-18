#!/usr/bin/env python3
"""
Day Trade Personal - 軽量版メインエントリーポイント

メモリ効率を重視した軽量バージョン
"""

import sys
from pathlib import Path

# システムパス追加
sys.path.append(str(Path(__file__).parent / "src"))

from daytrade_cli import DayTradeCLI
from daytrade_core import DayTradeCoreLight


def main():
    """軽量版メイン実行関数"""
    # CLIでの実行
    cli = DayTradeCLI()
    
    # 軽量版を使用するようにCLIを修正
    original_run_quick_mode = cli._run_quick_mode
    original_run_multi_mode = cli._run_multi_mode
    original_run_default_mode = cli._run_default_mode
    original_run_validate_mode = cli._run_validate_mode
    
    def lightweight_run_quick_mode(args):
        core = DayTradeCoreLight(debug=args.debug, use_cache=not args.no_cache)
        import asyncio
        return asyncio.run(core.run_quick_analysis(args.symbols))
    
    def lightweight_run_multi_mode(args):
        core = DayTradeCoreLight(debug=args.debug, use_cache=not args.no_cache)
        import asyncio
        return asyncio.run(core.run_multi_analysis(args.symbols))
    
    def lightweight_run_default_mode(args):
        core = DayTradeCoreLight(debug=args.debug, use_cache=not args.no_cache)
        import asyncio
        return asyncio.run(core.run_daytrading_analysis(args.symbols))
    
    def lightweight_run_validate_mode(args):
        core = DayTradeCoreLight(debug=args.debug, use_cache=not args.no_cache)
        import asyncio
        return asyncio.run(core.run_validation(args.symbols))
    
    # 軽量版メソッドに置き換え
    cli._run_quick_mode = lightweight_run_quick_mode
    cli._run_multi_mode = lightweight_run_multi_mode
    cli._run_default_mode = lightweight_run_default_mode
    cli._run_validate_mode = lightweight_run_validate_mode
    
    print("🪶 Day Trade Personal - 軽量版")
    print("📊 メモリ効率最適化版 v2.0")
    print("="*50)
    
    return cli.execute()


if __name__ == "__main__":
    sys.exit(main())