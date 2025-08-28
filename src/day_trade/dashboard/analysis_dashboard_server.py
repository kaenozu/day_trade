"""
分析専用ダッシュボードサーバー（統合エントリーポイント）

【重要】自動取引機能は完全に無効化されています
分析・情報提供・教育支援のみを行うセーフモードサーバー

このファイルはanalysis_server/パッケージへの統合エントリーポイントです。
下位互換性を保ちながら、モジュラー構造への移行を実現します。
"""

# 新しいモジュラー構造からインポート
from .analysis_server import (
    app,
    create_app,
    get_analysis_engine,
    get_chart_manager,
    get_connection_manager,
    get_educational_system,
    get_report_manager,
    get_system_status,
    print_server_info,
)

# 下位互換性のための旧名称エクスポート
manager = get_connection_manager()
analysis_engine = None  # 実行時に初期化される
chart_manager = None    # 実行時に初期化される
report_manager = None   # 実行時に初期化される
educational_system = None  # 実行時に初期化される

# メイン実行ブロック
if __name__ == "__main__":
    import uvicorn
    
    print_server_info()
    uvicorn.run(app, host="127.0.0.1", port=8000)