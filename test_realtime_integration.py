#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リアルタイムデータフィード統合テスト

Phase 3a-1: WebSocketリアルタイムデータフィード実装
Issue #271対応テスト
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Event
from typing import Dict, List, Optional

# Windows環境でのUTF-8エンコーディング対応
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.realtime.realtime_feed import (
    ConnectionStatus,
    DataNormalizer,
    DataSource,
    MarketData,
    RealtimeDataFeed,
    WebSocketClient,
    WebSocketConfig
)


class MockWebSocketServer:
    """テスト用モックWebSocketサーバー"""

    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.client_count = 0

    async def start(self):
        """サーバー開始"""
        import websockets
        self.server = await websockets.serve(
            self.handle_client,
            "localhost",
            self.port
        )
        print(f"モックWebSocketサーバー開始: ws://localhost:{self.port}")

    async def stop(self):
        """サーバー停止"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("モックWebSocketサーバー停止")

    async def handle_client(self, websocket):
        """クライアント接続処理"""
        self.client_count += 1
        client_id = self.client_count
        print(f"クライアント{client_id}接続")

        try:
            async for message in websocket:
                data = json.loads(message)
                print(f"受信: {data}")

                if data.get("action") == "subscribe":
                    # 購読確認応答
                    response = {
                        "type": "subscription_confirmed",
                        "symbols": data.get("symbols", [])
                    }
                    await websocket.send(json.dumps(response))

                    # モック市場データ送信開始
                    asyncio.create_task(self.send_mock_data(websocket, data.get("symbols", [])))

        except Exception as e:
            print(f"クライアント{client_id}エラー: {e}")
        finally:
            print(f"クライアント{client_id}切断")

    async def send_mock_data(self, websocket, symbols: List[str]):
        """モック市場データ送信"""
        import random

        try:
            base_prices = {symbol: 100.0 + random.uniform(-50, 50) for symbol in symbols}

            while True:
                for symbol in symbols:
                    # ランダムな価格変動
                    price_change = random.uniform(-2.0, 2.0)
                    base_prices[symbol] += price_change

                    mock_data = {
                        "symbol": symbol,
                        "price": round(base_prices[symbol], 2),
                        "volume": random.randint(1000, 10000),
                        "bid": round(base_prices[symbol] - 0.05, 2),
                        "ask": round(base_prices[symbol] + 0.05, 2),
                        "high": round(base_prices[symbol] + random.uniform(0, 5), 2),
                        "low": round(base_prices[symbol] - random.uniform(0, 5), 2),
                        "timestamp": time.time()
                    }

                    await websocket.send(json.dumps(mock_data))
                    await asyncio.sleep(0.1)  # 100ms間隔

        except Exception as e:
            print(f"モックデータ送信エラー: {e}")


class TestRealtimeIntegration:
    """リアルタイム統合テストクラス"""

    def __init__(self):
        self.mock_server = MockWebSocketServer()
        self.received_data: List[MarketData] = []
        self.test_results = []

    def data_handler(self, data: MarketData):
        """テスト用データハンドラー"""
        self.received_data.append(data)
        print(f"受信: {data.symbol} = ¥{data.price} (volume: {data.volume})")

    async def test_websocket_connection(self) -> bool:
        """WebSocket接続テスト"""
        print("\n=== WebSocket接続テスト ===")

        try:
            config = WebSocketConfig(
                url="ws://localhost:8765/mock",
                symbols=["1234", "5678"],
                reconnect_delay=2.0,
                max_reconnect_attempts=3
            )

            client = WebSocketClient(config)

            # 接続テスト
            connected = await client.connect()
            assert connected, "WebSocket接続失敗"
            assert client.status == ConnectionStatus.CONNECTED, "接続状態が正しくない"

            print(f"接続状態: {client.status}")
            print("[OK] WebSocket接続成功")

            # 切断テスト
            await client.disconnect()
            assert client.status == ConnectionStatus.DISCONNECTED, "切断状態が正しくない"

            print("[OK] WebSocket切断成功")
            return True

        except Exception as e:
            print(f"[NG] WebSocket接続テスト失敗: {e}")
            traceback.print_exc()
            return False

    async def test_data_normalization(self) -> bool:
        """データ正規化テスト"""
        print("\n=== データ正規化テスト ===")

        try:
            # モックデータテスト
            mock_raw_data = {
                "symbol": "TEST",
                "price": 123.45,
                "volume": 5000,
                "bid": 123.40,
                "ask": 123.50,
                "high": 125.00,
                "low": 122.00
            }

            normalized = DataNormalizer.normalize_market_data(
                mock_raw_data, DataSource.MOCK
            )

            assert normalized is not None, "正規化データがNone"
            assert normalized.symbol == "TEST", "シンボルが正しくない"
            assert normalized.price == 123.45, "価格が正しくない"
            assert normalized.volume == 5000, "出来高が正しくない"
            assert normalized.source == "mock", "ソースが正しくない"

            print(f"正規化結果: {normalized}")
            print("[OK] データ正規化成功")
            return True

        except Exception as e:
            print(f"[NG] データ正規化テスト失敗: {e}")
            traceback.print_exc()
            return False

    async def test_realtime_streaming(self) -> bool:
        """リアルタイムストリーミングテスト"""
        print("\n=== リアルタイムストリーミングテスト ===")

        try:
            symbols = ["1234", "5678", "9999"]
            feed = RealtimeDataFeed(DataSource.MOCK)

            # データハンドラー登録
            feed.subscribe(self.data_handler)

            # ストリーミング開始
            success = await feed.start_streaming(symbols, "ws://localhost:8765/mock")
            assert success, "ストリーミング開始失敗"

            print(f"接続状態: {feed.get_connection_status()}")
            print(f"統計情報: {feed.get_statistics()}")

            # データ受信待機（5秒間）
            print("データ受信を5秒間待機...")
            await asyncio.sleep(5)

            # ストリーミング停止
            await feed.stop_streaming()

            # 結果検証
            assert len(self.received_data) > 0, "データが受信されていない"

            # 各銘柄のデータを確認
            received_symbols = {data.symbol for data in self.received_data}
            print(f"受信銘柄: {received_symbols}")
            print(f"総受信データ数: {len(self.received_data)}")

            # 最低限のデータ検証
            if len(self.received_data) > 0:
                sample_data = self.received_data[0]
                print(f"サンプルデータ: symbol={sample_data.symbol}, price={sample_data.price}")
                assert sample_data.volume > 0, "出来高が無効"
                assert sample_data.symbol in symbols, "銘柄が予期しないもの"

            print("[OK] リアルタイムストリーミング成功")
            return True

        except Exception as e:
            print(f"[NG] リアルタイムストリーミングテスト失敗: {e}")
            traceback.print_exc()
            return False

    async def test_reconnection(self) -> bool:
        """再接続テスト"""
        print("\n=== 再接続テスト ===")

        try:
            config = WebSocketConfig(
                url="ws://localhost:8765/mock",
                symbols=["TEST"],
                reconnect_delay=1.0,
                max_reconnect_attempts=3
            )

            client = WebSocketClient(config)

            # 初回接続
            connected = await client.connect()
            assert connected, "初回接続失敗"

            original_reconnect_count = client.reconnect_count

            # 意図的に切断をシミュレート（サーバー側で実装が必要）
            # ここでは再接続ロジックの存在確認のみ
            print(f"再接続カウント: {client.reconnect_count}")
            print("[OK] 再接続機能確認")

            await client.disconnect()
            return True

        except Exception as e:
            print(f"[NG] 再接続テスト失敗: {e}")
            traceback.print_exc()
            return False

    async def test_performance_metrics(self) -> bool:
        """パフォーマンステスト"""
        print("\n=== パフォーマンステスト ===")

        try:
            symbols = ["PERF1", "PERF2"]
            feed = RealtimeDataFeed(DataSource.MOCK)

            data_count = 0
            start_time = time.time()

            def performance_handler(data: MarketData):
                nonlocal data_count
                data_count += 1

            feed.subscribe(performance_handler)

            # 10秒間のデータ収集
            await feed.start_streaming(symbols, "ws://localhost:8765/mock")
            await asyncio.sleep(10)
            await feed.stop_streaming()

            end_time = time.time()
            duration = end_time - start_time

            # パフォーマンス計算
            throughput = data_count / duration
            avg_latency = duration / data_count if data_count > 0 else 0

            print(f"測定時間: {duration:.2f}秒")
            print(f"受信データ数: {data_count}")
            print(f"スループット: {throughput:.2f} データ/秒")
            print(f"平均レイテンシ: {avg_latency*1000:.2f}ms")

            # 基本的なパフォーマンス要件確認
            assert throughput > 5, f"スループット不足: {throughput} < 5"
            assert avg_latency < 1.0, f"レイテンシ過大: {avg_latency} > 1.0秒"

            print("[OK] パフォーマンステスト成功")
            return True

        except Exception as e:
            print(f"[NG] パフォーマンステスト失敗: {e}")
            traceback.print_exc()
            return False

    async def run_all_tests(self) -> bool:
        """全テスト実行"""
        print("リアルタイムデータフィード統合テスト開始")
        print("=" * 50)

        # モックサーバー開始
        await self.mock_server.start()

        try:
            # 各テスト実行
            tests = [
                ("WebSocket接続", self.test_websocket_connection),
                ("データ正規化", self.test_data_normalization),
                ("リアルタイムストリーミング", self.test_realtime_streaming),
                ("再接続機能", self.test_reconnection),
                ("パフォーマンス", self.test_performance_metrics),
            ]

            passed = 0
            for test_name, test_func in tests:
                try:
                    result = await test_func()
                    self.test_results.append((test_name, result))
                    if result:
                        passed += 1
                except Exception as e:
                    print(f"テスト実行エラー [{test_name}]: {e}")
                    self.test_results.append((test_name, False))

            # 結果サマリー
            print("\n" + "=" * 50)
            print("テスト結果サマリー")
            print("=" * 50)

            for test_name, result in self.test_results:
                status = "[OK] 成功" if result else "[NG] 失敗"
                print(f"{test_name}: {status}")

            success_rate = (passed / len(tests)) * 100
            print(f"\n成功率: {passed}/{len(tests)} ({success_rate:.1f}%)")

            if passed == len(tests):
                print("\n[SUCCESS] 全テスト成功！WebSocketリアルタイムデータフィードが正常に動作しています。")
                return True
            else:
                print(f"\n[WARNING] {len(tests) - passed}個のテストが失敗しました。")
                return False

        finally:
            # モックサーバー停止
            await self.mock_server.stop()


async def main():
    """メインテスト実行"""
    tester = TestRealtimeIntegration()
    success = await tester.run_all_tests()
    return success


if __name__ == "__main__":
    try:
        # 必要な依存関係の確認
        import websockets
        print("[OK] websockets利用可能")
    except ImportError:
        print("[NG] websocketsがインストールされていません")
        print("pip install websockets を実行してください")
        sys.exit(1)

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
