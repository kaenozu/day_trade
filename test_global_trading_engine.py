#!/usr/bin/env python3
"""
Global Trading Engine - 24時間連続取引システム統合デモ
Forex・Crypto・Stock市場の包括的分析・取引システム

世界3大市場での24時間ノンストップ取引を実現
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.day_trade.analysis.cross_market_correlation import create_correlation_engine
from src.day_trade.data.crypto_data_collector import create_crypto_collector
from src.day_trade.data.forex_data_collector import create_forex_collector
from src.day_trade.models.database import init_global_database
from src.day_trade.models.global_ai_models import (
    GlobalModelConfig,
    create_global_ai_models,
)

# プロジェクト内インポート
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

class GlobalTradingSession:
    """グローバル取引セッション管理"""

    def __init__(self):
        # 取引セッション定義
        self.sessions = {
            'tokyo': {
                'name': 'Tokyo Session',
                'timezone': 'Asia/Tokyo',
                'start_hour': 9,
                'end_hour': 17,
                'markets': ['forex', 'stock']
            },
            'london': {
                'name': 'London Session',
                'timezone': 'Europe/London',
                'start_hour': 8,
                'end_hour': 16,
                'markets': ['forex', 'stock']
            },
            'new_york': {
                'name': 'New York Session',
                'timezone': 'America/New_York',
                'start_hour': 9,
                'end_hour': 17,
                'markets': ['forex', 'stock']
            },
            'crypto': {
                'name': '24/7 Crypto Session',
                'timezone': 'UTC',
                'start_hour': 0,
                'end_hour': 24,
                'markets': ['crypto']
            }
        }

    def get_active_sessions(self) -> List[str]:
        """現在アクティブなセッション取得"""
        current_utc = datetime.now(timezone.utc)
        active = []

        # Cryptoは常時アクティブ
        active.append('crypto')

        # 各市場セッション時間をUTCで近似チェック
        utc_hour = current_utc.hour

        # Tokyo: UTC 0:00-8:00 (JST 9:00-17:00)
        if 0 <= utc_hour <= 8:
            active.append('tokyo')

        # London: UTC 8:00-16:00 (GMT 8:00-16:00)
        if 8 <= utc_hour <= 16:
            active.append('london')

        # New York: UTC 13:00-21:00 (EST 9:00-17:00)
        if 13 <= utc_hour <= 21:
            active.append('new_york')

        return active

    def get_primary_session(self) -> str:
        """主要セッション判定"""
        active_sessions = self.get_active_sessions()

        # 優先順位: New York > London > Tokyo > Crypto only
        if 'new_york' in active_sessions:
            return 'new_york'
        elif 'london' in active_sessions:
            return 'london'
        elif 'tokyo' in active_sessions:
            return 'tokyo'
        else:
            return 'crypto'

class GlobalTradingEngine:
    """グローバル取引エンジン統合システム"""

    def __init__(self):
        # コンポーネント初期化
        self.forex_collector = create_forex_collector()
        self.crypto_collector = create_crypto_collector()
        self.correlation_engine = create_correlation_engine(
            self.forex_collector, self.crypto_collector
        )

        # AI モデル
        model_config = GlobalModelConfig(
            sequence_length=60,
            forex_features=24,
            crypto_features=32,
            hidden_size=128,
            num_layers=2,
            prediction_horizons=[1, 5, 15, 60]  # 1分、5分、15分、1時間
        )
        self.ai_models = create_global_ai_models(model_config)

        # セッション管理
        self.session_manager = GlobalTradingSession()

        # システム状態
        self.is_running = False
        self.system_stats = {
            'start_time': None,
            'total_predictions': 0,
            'total_correlations': 0,
            'active_markets': [],
            'current_session': None,
            'uptime_hours': 0.0
        }

        # パフォーマンス統計
        self.performance_metrics = {
            'prediction_latency': [],
            'correlation_calc_time': [],
            'data_collection_rate': 0,
            'system_health_score': 1.0
        }

        logger.info("Global Trading Engine initialized")

    async def start_24h_operation(self, duration_hours: float = 1.0):
        """24時間連続運用開始"""

        print("=" * 70)
        print("🌍 GLOBAL TRADING ENGINE - 24時間連続取引システム")
        print("=" * 70)
        print(f"運用開始時刻: {datetime.now()}")
        print(f"テスト期間: {duration_hours} 時間")
        print("=" * 70)

        self.is_running = True
        self.system_stats['start_time'] = datetime.now()

        # データベース初期化
        try:
            init_global_database()
            print("✅ データベース初期化完了")
        except Exception as e:
            logger.warning(f"Database initialization warning: {e}")

        # 並行処理タスク
        tasks = [
            self._continuous_data_collection(),
            self._continuous_ai_analysis(),
            self._continuous_correlation_analysis(),
            self._continuous_performance_monitoring(),
            self._session_management_loop(),
            self._system_health_monitoring()
        ]

        try:
            # 指定時間後に停止
            duration_seconds = duration_hours * 3600

            print("🚀 24時間連続システム開始...")
            print(f"⏰ {duration_seconds:.0f} 秒間の運用テスト実行中...")

            # タスク実行とタイムアウト
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=duration_seconds
            )

        except asyncio.TimeoutError:
            print(f"\n✅ {duration_hours} 時間のテスト運用が正常完了しました")

        finally:
            await self._shutdown_system()

    async def _continuous_data_collection(self):
        """継続的データ収集"""
        logger.info("Starting continuous data collection...")

        while self.is_running:
            try:
                # セッション判定
                current_session = self.session_manager.get_primary_session()

                # アクティブ市場でのデータ収集
                if current_session in ['tokyo', 'london', 'new_york']:
                    # Forex データ収集
                    forex_data = self.forex_collector.get_all_latest_ticks()
                    if forex_data:
                        logger.debug(f"Forex data: {len(forex_data)} pairs")

                # 暗号通貨データ（24時間）
                crypto_data = self.crypto_collector.get_all_market_data()
                if crypto_data:
                    logger.debug(f"Crypto data: {len(crypto_data)} symbols")

                # 統計更新
                self.performance_metrics['data_collection_rate'] = len(forex_data) + len(crypto_data) if forex_data and crypto_data else 0

                await asyncio.sleep(5)  # 5秒間隔

            except Exception as e:
                logger.error(f"Data collection error: {e}")
                await asyncio.sleep(10)

    async def _continuous_ai_analysis(self):
        """継続的AI分析"""
        logger.info("Starting continuous AI analysis...")

        while self.is_running:
            try:
                # 模擬AI分析（実際のデータでテスト）
                start_time = time.time()

                # ダミーデータでAI予測実行
                batch_size = 2
                sequence_length = 60

                forex_data = np.random.randn(batch_size, sequence_length, 24).astype(np.float32)
                crypto_data = np.random.randn(batch_size, sequence_length, 32).astype(np.float32)

                # Tensor変換
                import torch
                forex_tensor = torch.tensor(forex_data)
                crypto_tensor = torch.tensor(crypto_data)
                forex_ids = torch.randint(0, 10, (batch_size,))
                crypto_ids = torch.randint(0, 20, (batch_size,))

                # AI予測実行
                with torch.no_grad():
                    predictions = self.ai_models(forex_tensor, crypto_tensor, forex_ids, crypto_ids)

                # レイテンシ記録
                prediction_time = (time.time() - start_time) * 1000
                self.performance_metrics['prediction_latency'].append(prediction_time)

                # 統計更新
                self.system_stats['total_predictions'] += len(predictions['forex_predictions']) + len(predictions['crypto_predictions'])

                logger.debug(f"AI analysis completed in {prediction_time:.2f}ms")

                await asyncio.sleep(15)  # 15秒間隔

            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                await asyncio.sleep(30)

    async def _continuous_correlation_analysis(self):
        """継続的相関分析"""
        logger.info("Starting continuous correlation analysis...")

        while self.is_running:
            try:
                start_time = time.time()

                # 相関分析実行（縮小版）
                asset_pairs = [
                    ("EUR/USD", "BTCUSDT"),
                    ("USD/JPY", "ETHUSDT"),
                    ("BTCUSDT", "ETHUSDT")
                ]

                correlation_count = 0
                for asset1, asset2 in asset_pairs:
                    # 模擬相関計算
                    correlation = np.random.uniform(-0.5, 0.5)
                    correlation_count += 1

                # 時間記録
                correlation_time = (time.time() - start_time) * 1000
                self.performance_metrics['correlation_calc_time'].append(correlation_time)

                # 統計更新
                self.system_stats['total_correlations'] += correlation_count

                logger.debug(f"Correlation analysis: {correlation_count} pairs in {correlation_time:.2f}ms")

                await asyncio.sleep(30)  # 30秒間隔

            except Exception as e:
                logger.error(f"Correlation analysis error: {e}")
                await asyncio.sleep(60)

    async def _session_management_loop(self):
        """セッション管理ループ"""
        logger.info("Starting session management...")

        while self.is_running:
            try:
                # 現在のセッション判定
                active_sessions = self.session_manager.get_active_sessions()
                primary_session = self.session_manager.get_primary_session()

                # システム状態更新
                self.system_stats['active_markets'] = active_sessions
                self.system_stats['current_session'] = primary_session

                # セッション変更時のログ
                current_time = datetime.now()
                logger.info(f"Active sessions: {active_sessions}, Primary: {primary_session}")

                await asyncio.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"Session management error: {e}")
                await asyncio.sleep(120)

    async def _continuous_performance_monitoring(self):
        """継続パフォーマンス監視"""
        logger.info("Starting performance monitoring...")

        while self.is_running:
            try:
                # パフォーマンス指標計算
                if self.performance_metrics['prediction_latency']:
                    avg_latency = np.mean(self.performance_metrics['prediction_latency'][-100:])  # 最新100件
                else:
                    avg_latency = 0

                if self.performance_metrics['correlation_calc_time']:
                    avg_correlation_time = np.mean(self.performance_metrics['correlation_calc_time'][-50:])  # 最新50件
                else:
                    avg_correlation_time = 0

                # システムヘルススコア計算
                health_score = 1.0
                if avg_latency > 1000:  # 1秒以上
                    health_score -= 0.2
                if avg_correlation_time > 2000:  # 2秒以上
                    health_score -= 0.1

                self.performance_metrics['system_health_score'] = max(0.0, health_score)

                logger.debug(f"Performance: Latency={avg_latency:.1f}ms, Health={health_score:.2f}")

                await asyncio.sleep(30)  # 30秒間隔

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _system_health_monitoring(self):
        """システムヘルス監視"""
        logger.info("Starting system health monitoring...")

        while self.is_running:
            try:
                # 稼働時間計算
                if self.system_stats['start_time']:
                    uptime = datetime.now() - self.system_stats['start_time']
                    self.system_stats['uptime_hours'] = uptime.total_seconds() / 3600

                # ヘルス状況表示（10分ごと）
                if int(self.system_stats['uptime_hours'] * 6) % 1 == 0:  # 10分ごと
                    self._display_system_status()

                await asyncio.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(120)

    def _display_system_status(self):
        """システム状況表示"""
        stats = self.system_stats
        metrics = self.performance_metrics

        print(f"\n📊 システム状況 [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"   稼働時間: {stats['uptime_hours']:.2f} 時間")
        print(f"   現在セッション: {stats['current_session']}")
        print(f"   アクティブ市場: {', '.join(stats['active_markets'])}")
        print(f"   総予測回数: {stats['total_predictions']}")
        print(f"   総相関分析: {stats['total_correlations']}")
        print(f"   システムヘルス: {metrics['system_health_score']:.1%}")

        if metrics['prediction_latency']:
            avg_latency = np.mean(metrics['prediction_latency'][-10:])
            print(f"   平均予測時間: {avg_latency:.1f}ms")

    async def _shutdown_system(self):
        """システム停止処理"""

        print("\n🔄 システム停止処理中...")
        self.is_running = False

        try:
            # データ収集停止
            if hasattr(self.forex_collector, 'cleanup'):
                await self.forex_collector.cleanup()
            if hasattr(self.crypto_collector, 'cleanup'):
                await self.crypto_collector.cleanup()

            print("✅ データ収集システム停止完了")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

        # 最終統計表示
        self._display_final_report()

    def _display_final_report(self):
        """最終レポート表示"""

        stats = self.system_stats
        metrics = self.performance_metrics

        print("\n" + "=" * 70)
        print("🌍 GLOBAL TRADING ENGINE - 最終運用レポート")
        print("=" * 70)

        print(f"運用期間: {stats['start_time']} ～ {datetime.now()}")
        print(f"総稼働時間: {stats['uptime_hours']:.2f} 時間")

        print("\n📈 処理統計:")
        print(f"  ✅ 総AI予測回数: {stats['total_predictions']:,}")
        print(f"  ✅ 総相関分析回数: {stats['total_correlations']:,}")
        print(f"  ✅ データ収集レート: {metrics['data_collection_rate']}/cycle")

        print("\n⚡ パフォーマンス指標:")
        if metrics['prediction_latency']:
            avg_latency = np.mean(metrics['prediction_latency'])
            min_latency = np.min(metrics['prediction_latency'])
            max_latency = np.max(metrics['prediction_latency'])
            print(f"  AI予測レイテンシ: 平均 {avg_latency:.1f}ms (範囲: {min_latency:.1f}-{max_latency:.1f}ms)")

        if metrics['correlation_calc_time']:
            avg_corr_time = np.mean(metrics['correlation_calc_time'])
            print(f"  相関分析時間: 平均 {avg_corr_time:.1f}ms")

        print(f"  最終ヘルススコア: {metrics['system_health_score']:.1%}")

        print("\n🏆 運用判定:")
        if stats['uptime_hours'] >= 0.95 and metrics['system_health_score'] >= 0.8:
            grade = "EXCELLENT - 24時間運用対応"
            emoji = "🌟"
        elif stats['uptime_hours'] >= 0.8 and metrics['system_health_score'] >= 0.7:
            grade = "GOOD - 実用レベル達成"
            emoji = "✅"
        elif stats['uptime_hours'] >= 0.5:
            grade = "ACCEPTABLE - 基本機能確認"
            emoji = "⚠️"
        else:
            grade = "NEEDS IMPROVEMENT - 要改善"
            emoji = "❌"

        print(f"  {emoji} 総合評価: {grade}")

        print("\n💡 推奨事項:")
        print("  • 本格運用にはリアルAPI接続が必要")
        print("  • 高頻度取引には専用インフラ推奨")
        print("  • 24時間監視・アラート体制構築")
        print("  • 各市場法規制への対応確認")

        print("\n" + "=" * 70)
        print("🚀 Global Trading Engine テスト完了！")
        print("=" * 70)

async def main():
    """メイン実行"""

    # ログレベル設定
    logging.getLogger().setLevel(logging.INFO)

    try:
        # Global Trading Engine 起動
        engine = GlobalTradingEngine()

        # 1時間（実際は短縮版で5分）の24時間システムデモ
        demo_duration = 5.0 / 60  # 5分 = 0.083時間

        await engine.start_24h_operation(duration_hours=demo_duration)

        return 0

    except KeyboardInterrupt:
        print("\n\nシステム運用が中断されました")
        return 2
    except Exception as e:
        print(f"\n❌ システムエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
