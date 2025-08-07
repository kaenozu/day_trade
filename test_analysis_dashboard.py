"""
分析専用ダッシュボードのテスト

セーフモードとダッシュボード機能をテストします
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.dashboard.analysis_dashboard_server import app
from src.day_trade.config.trading_mode_config import is_safe_mode, get_current_trading_config
from fastapi.testclient import TestClient


def test_safe_mode_configuration():
    """セーフモード設定のテスト"""
    print("=" * 60)
    print("セーフモード設定テスト")
    print("=" * 60)

    assert is_safe_mode(), "セーフモードが無効です"

    config = get_current_trading_config()
    assert not config.enable_automatic_trading, "自動取引が有効になっています"
    assert not config.enable_order_execution, "注文実行が有効になっています"
    assert config.disable_order_api, "注文APIが有効になっています"

    print("✅ セーフモード設定: 正常")
    print("✅ 自動取引: 無効")
    print("✅ 注文実行: 無効")
    print("✅ 注文API: 無効")


def test_dashboard_endpoints():
    """ダッシュボードエンドポイントのテスト"""
    print("\n" + "=" * 60)
    print("ダッシュボードエンドポイントテスト")
    print("=" * 60)

    client = TestClient(app)

    # メインページテスト
    response = client.get("/")
    assert response.status_code == 200
    assert "分析専用ダッシュボード" in response.text
    assert "セーフモード有効" in response.text
    print("✅ メインページ: OK")

    # システム状態API テスト
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()

    assert data["safe_mode"] is True, "セーフモードがTrueではありません"
    assert data["trading_disabled"] is True, "取引無効化がTrueではありません"
    assert data["system_type"] == "analysis_only", "システムタイプが正しくありません"
    print("✅ システム状態API: OK")
    print(f"   - セーフモード: {data['safe_mode']}")
    print(f"   - 取引無効: {data['trading_disabled']}")
    print(f"   - システム種別: {data['system_type']}")

    # 監視銘柄API テスト
    response = client.get("/api/analysis/symbols")
    assert response.status_code == 200
    data = response.json()

    print("✅ 監視銘柄API: OK")
    print(f"   - 銘柄数: {data['count']}")

    # 分析レポートAPI テスト (初期化前なのでエラーが予想される)
    response = client.get("/api/analysis/report")
    if response.status_code == 500:
        print("✅ 分析レポートAPI: 初期化前エラー（正常な動作）")
    else:
        print("✅ 分析レポートAPI: OK")


def test_security_features():
    """セキュリティ機能のテスト"""
    print("\n" + "=" * 60)
    print("セキュリティ機能テスト")
    print("=" * 60)

    client = TestClient(app)

    # システム状態での安全性確認
    response = client.get("/api/system/status")
    data = response.json()

    # 自動取引が完全に無効化されていることを確認
    assert "warning" in data, "警告メッセージがありません"
    assert "自動取引機能は完全に無効化" in data["warning"], "無効化警告がありません"

    print("✅ 警告メッセージ: 表示されています")
    print("✅ 自動取引無効化: 確認済み")
    print("✅ セキュリティチェック: 合格")


async def main():
    """メインテスト実行"""
    print("分析専用ダッシュボード テスト開始")
    print("=" * 80)

    try:
        test_safe_mode_configuration()
        test_dashboard_endpoints()
        test_security_features()

        print("\n" + "=" * 80)
        print("🎉 全テスト合格！")
        print("✅ ダッシュボードは正常に動作しています")
        print("✅ セーフモードが正しく設定されています")
        print("✅ 自動取引機能は完全に無効化されています")
        print()
        print("📊 ダッシュボードの起動:")
        print("   python run_analysis_dashboard.py")
        print()
        print("🌐 アクセスURL:")
        print("   http://localhost:8000")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ テスト失敗: {e}")
        print("=" * 80)
        raise e


if __name__ == "__main__":
    try:
        # TestClient のために一時的にevent loopを作成
        asyncio.run(main())
    except ImportError as e:
        if "TestClient" in str(e):
            print("注意: TestClient が利用できないため、基本テストのみ実行します")
            test_safe_mode_configuration()
            print("\n✅ セーフモード設定テスト完了")
            print("📦 pip install httpx でテストクライアントをインストールできます")
        else:
            raise e
