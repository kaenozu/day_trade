#!/usr/bin/env python3
"""
分析専用ダッシュボード起動スクリプト

【重要】完全セーフモード
- 自動取引: 完全無効
- 注文実行: 完全無効
- 分析・教育・研究専用

使用方法:
    python run_analysis_dashboard.py
"""

import sys
from pathlib import Path

import uvicorn

from src.day_trade.config.trading_mode_config import (
    is_safe_mode,
    log_current_configuration,
)

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """メイン実行関数"""
    print("=" * 80)
    print("[SECURE] Day Trade 分析専用ダッシュボード起動")
    print("=" * 80)

    # セーフモード確認
    if not is_safe_mode():
        print("[ERROR] セーフモードが無効です")
        print("自動取引機能が有効になっている可能性があります")
        return 1

    # 設定状況をログ出力
    log_current_configuration()

    print("\n" + "=" * 80)
    print("[OK] セーフモード確認完了")
    print("[OK] 自動取引機能は完全に無効化されています")
    print("[OK] 分析・教育・研究専用システムとして起動します")
    print("=" * 80)
    print("[WEB] ダッシュボードURL: http://localhost:8000")
    print("[API] ドキュメント: http://localhost:8000/docs")
    print("[STATUS] システム状態: http://localhost:8000/api/system/status")
    print("=" * 80)
    print("[WARN] 重要事項:")
    print("   - このシステムは実際の取引を行いません")
    print("   - 教育・学習・研究目的でのみ使用してください")
    print("   - 投資判断は必ず自己責任で行ってください")
    print("=" * 80)

    try:
        # FastAPIサーバー起動
        uvicorn.run(
            "src.day_trade.dashboard.analysis_dashboard_server:app",
            host="127.0.0.1",
            port=8000,
            reload=True,  # 開発用
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("🛑 分析ダッシュボードを停止しました")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
