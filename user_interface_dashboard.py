#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Interface Dashboard - ユーザーインターフェース・ダッシュボード

Issue #814対応：使いやすいコマンドラインインターフェース
統合ダッシュボードと簡単な操作コマンド
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
from pathlib import Path

# Windows環境での文字化け対策
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class DayTradingDashboard:
    """デイトレーディングダッシュボード"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # デフォルト設定
        self.default_symbols = ["7203", "8306", "4751", "9984", "6501"]
        self.portfolio_value = 1000000  # 100万円

        # システムコンポーネント
        self.systems_loaded = False

        self.logger.info("Day Trading Dashboard initialized")

    async def load_systems(self):
        """システムコンポーネント読み込み"""
        if self.systems_loaded:
            return

        try:
            # 各システムをインポート
            global optimized_prediction_system, data_quality_manager
            global enhanced_risk_management, paper_trading_engine, backtest_engine

            from optimized_prediction_system import optimized_prediction_system
            from data_quality_manager import data_quality_manager
            from enhanced_risk_management_system import enhanced_risk_management
            from backtest_paper_trading_system import paper_trading_engine, backtest_engine

            self.systems_loaded = True
            print("[OK] システムコンポーネント読み込み完了")

        except ImportError as e:
            print(f"[ERROR] システム読み込みエラー: {e}")

    def print_header(self, title: str):
        """ヘッダー表示"""
        print(f"\n{'='*60}")
        print(f"  🚀 {title}")
        print(f"{'='*60}")

    def print_section(self, title: str):
        """セクション表示"""
        print(f"\n--- {title} ---")

    async def show_main_dashboard(self):
        """メインダッシュボード表示"""

        self.print_header("デイトレーディング システム ダッシュボード")

        print(f"📅 現在時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 ポートフォリオ価値: ¥{self.portfolio_value:,.0f}")
        print(f"📊 監視銘柄: {', '.join(self.default_symbols)}")

        # システムステータス
        self.print_section("システムステータス")

        status_items = [
            ("予測システム", "✅ 稼働中"),
            ("データ品質管理", "✅ 稼働中"),
            ("リスク管理", "✅ 稼働中"),
            ("ペーパートレード", "✅ 利用可能"),
            ("バックテスト", "✅ 利用可能")
        ]

        for system, status in status_items:
            print(f"  {system:<15} {status}")

        # 利用可能コマンド
        self.print_section("利用可能コマンド")

        commands = [
            ("1", "市場分析", "全銘柄の予測・リスク分析"),
            ("2", "個別銘柄分析", "特定銘柄の詳細分析"),
            ("3", "ポートフォリオ分析", "現在のポートフォリオ評価"),
            ("4", "バックテスト実行", "戦略のバックテスト"),
            ("5", "ペーパートレード", "仮想取引実行"),
            ("6", "データ品質チェック", "データ品質評価"),
            ("7", "システム設定", "設定変更"),
            ("q", "終了", "システム終了")
        ]

        for cmd, name, desc in commands:
            print(f"  [{cmd}] {name:<15} - {desc}")

    async def run_market_analysis(self):
        """市場分析実行"""

        self.print_header("📈 市場分析")

        await self.load_systems()

        print("銘柄分析中...")

        results = []

        for symbol in self.default_symbols:
            try:
                print(f"\n📊 {symbol} 分析中...")

                # 予測実行
                prediction = await optimized_prediction_system.predict_with_optimized_models(symbol)

                # リスクメトリクス取得
                position_value = self.portfolio_value * 0.1  # 10%ポジションと仮定
                risk_metrics = await enhanced_risk_management.calculate_risk_metrics(
                    symbol, position_value, self.portfolio_value
                )

                # データ品質評価
                quality_result = await data_quality_manager.evaluate_data_quality(symbol)

                results.append({
                    'symbol': symbol,
                    'prediction': '上昇' if prediction.prediction else '下降',
                    'confidence': prediction.confidence,
                    'risk_level': risk_metrics.risk_level.value,
                    'volatility': risk_metrics.volatility,
                    'quality_score': quality_result.get('overall_score', 0)
                })

                print(f"  予測: {results[-1]['prediction']} (信頼度: {results[-1]['confidence']:.1%})")
                print(f"  リスク: {results[-1]['risk_level']} (ボラティリティ: {results[-1]['volatility']:.1f}%)")
                print(f"  データ品質: {results[-1]['quality_score']:.1f}/100")

            except Exception as e:
                print(f"  ❌ {symbol} 分析エラー: {e}")

        # 分析結果サマリー
        if results:
            self.print_section("分析結果サマリー")

            print(f"{'銘柄':<8} {'予測':<8} {'信頼度':<8} {'リスク':<12} {'品質':<8}")
            print("-" * 50)

            for result in results:
                print(f"{result['symbol']:<8} "
                      f"{result['prediction']:<8} "
                      f"{result['confidence']:.1%:<8} "
                      f"{result['risk_level']:<12} "
                      f"{result['quality_score']:.0f}/<8")

            # 推奨銘柄
            high_confidence = [r for r in results if r['confidence'] > 0.6 and r['quality_score'] > 80]
            if high_confidence:
                print(f"\n🎯 推奨銘柄:")
                for rec in high_confidence[:3]:  # 上位3銘柄
                    print(f"  • {rec['symbol']}: {rec['prediction']} (信頼度{rec['confidence']:.1%})")

    async def run_individual_analysis(self, symbol: str = None):
        """個別銘柄分析"""

        if not symbol:
            symbol = input("分析する銘柄コードを入力してください: ").strip()

        if not symbol:
            print("❌ 銘柄コードが入力されていません")
            return

        self.print_header(f"🔍 {symbol} 詳細分析")

        await self.load_systems()

        try:
            # 現在価格取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "1mo")
            if data is not None and len(data) > 0:
                current_price = float(data['Close'].iloc[-1])
                price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                print(f"📈 現在価格: ¥{current_price:.2f} ({price_change:+.2f}%)")

            # 予測分析
            self.print_section("AI予測")
            prediction = await optimized_prediction_system.predict_with_optimized_models(symbol)

            print(f"予測方向: {'📈 上昇' if prediction.prediction else '📉 下降'}")
            print(f"信頼度: {prediction.confidence:.1%}")

            if prediction.model_consensus:
                print("モデル合意:")
                for model, pred in prediction.model_consensus.items():
                    print(f"  {model}: {'上昇' if pred else '下降'}")

            # リスク分析
            self.print_section("リスク分析")
            position_value = self.portfolio_value * 0.05  # 5%ポジション
            risk_metrics = await enhanced_risk_management.calculate_risk_metrics(
                symbol, position_value, self.portfolio_value
            )

            print(f"リスクレベル: {risk_metrics.risk_level.value}")
            print(f"ボラティリティ: {risk_metrics.volatility:.1f}%")
            print(f"1日VaR: ¥{risk_metrics.var_1day:,.0f}")
            print(f"最大ドローダウン: {risk_metrics.max_drawdown:.1f}%")
            print(f"シャープレシオ: {risk_metrics.sharpe_ratio:.3f}")

            # ポジションサイジング
            self.print_section("推奨ポジション")
            from enhanced_risk_management_system import PositionSizingMethod
            sizing = await enhanced_risk_management.calculate_position_sizing(
                symbol, current_price, self.portfolio_value,
                PositionSizingMethod.ATR_BASED
            )

            print(f"推奨株数: {sizing.recommended_quantity:,}株")
            print(f"投資額: ¥{sizing.recommended_value:,.0f} ({sizing.position_weight:.1f}%)")
            print(f"ストップロス: ¥{sizing.stop_loss_price:.2f}")
            print(f"利確目標: ¥{sizing.take_profit_price:.2f}")
            print(f"リスクリワード比: {sizing.risk_reward_ratio:.2f}:1")

            # データ品質
            self.print_section("データ品質")
            quality_result = await data_quality_manager.evaluate_data_quality(symbol)

            print(f"総合スコア: {quality_result.get('overall_score', 0):.1f}/100")
            print(f"品質レベル: {quality_result.get('quality_level', 'unknown')}")

            if quality_result.get('recommendations'):
                print("推奨事項:")
                for rec in quality_result['recommendations'][:3]:
                    print(f"  • {rec}")

        except Exception as e:
            print(f"❌ 分析エラー: {e}")

    async def run_backtest(self):
        """バックテスト実行"""

        self.print_header("📊 バックテスト実行")

        await self.load_systems()

        symbol = input("バックテストする銘柄 (デフォルト:7203): ").strip() or "7203"
        period = input("期間 (1y/6mo/3mo, デフォルト:6mo): ").strip() or "6mo"

        print(f"\n🔄 {symbol} バックテスト実行中 (期間: {period})...")

        try:
            result = await backtest_engine.run_simple_ma_crossover_backtest(
                symbol, period, initial_capital=self.portfolio_value
            )

            print(f"\n📊 バックテスト結果:")
            print(f"戦略: {result.strategy_name}")
            print(f"期間: {result.start_date.date()} ~ {result.end_date.date()}")
            print(f"初期資金: ¥{result.initial_capital:,.0f}")
            print(f"最終資金: ¥{result.final_capital:,.0f}")
            print(f"総リターン: {result.total_return:+.2f}%")
            print(f"年率リターン: {result.annualized_return:+.2f}%")
            print(f"最大ドローダウン: {result.max_drawdown:.2f}%")
            print(f"シャープレシオ: {result.sharpe_ratio:.3f}")
            print(f"勝率: {result.win_rate:.1f}%")
            print(f"総取引数: {result.total_trades}")

            # パフォーマンス評価
            if result.total_return > 0:
                print("\n✅ ポジティブリターン")
            else:
                print("\n❌ ネガティブリターン")

            if result.sharpe_ratio > 1.0:
                print("📈 優秀なリスク調整済みリターン")
            elif result.sharpe_ratio > 0.5:
                print("📊 まずまずのリスク調整済みリターン")
            else:
                print("📉 低いリスク調整済みリターン")

        except Exception as e:
            print(f"❌ バックテストエラー: {e}")

    async def run_paper_trading(self):
        """ペーパートレード実行"""

        self.print_header("💰 ペーパートレーディング")

        await self.load_systems()

        # ポートフォリオ状況表示
        await paper_trading_engine.update_portfolio()
        portfolio = paper_trading_engine.portfolio

        print(f"現金残高: ¥{portfolio.cash_balance:,.0f}")
        print(f"総資産価値: ¥{portfolio.total_value:,.0f}")
        print(f"総損益: ¥{portfolio.total_pnl:+,.0f} ({portfolio.total_return:+.2f}%)")
        print(f"ポジション数: {len(portfolio.positions)}")

        if portfolio.positions:
            print("\n現在のポジション:")
            for symbol, pos in portfolio.positions.items():
                print(f"  {symbol}: {pos.quantity}株 (平均¥{pos.average_price:.2f}, 時価¥{pos.current_price:.2f})")

        # 操作メニュー
        print("\n操作選択:")
        print("  [1] 買い注文")
        print("  [2] 売り注文")
        print("  [3] ポートフォリオ更新")
        print("  [4] 戻る")

        choice = input("\n選択してください: ").strip()

        if choice == "1":
            await self._execute_buy_order()
        elif choice == "2":
            await self._execute_sell_order()
        elif choice == "3":
            await paper_trading_engine.update_portfolio()
            print("✅ ポートフォリオ更新完了")

    async def _execute_buy_order(self):
        """買い注文実行"""

        symbol = input("銘柄コード: ").strip()
        if not symbol:
            print("❌ 銘柄コードが必要です")
            return

        try:
            quantity = int(input("株数: ").strip())
            if quantity <= 0:
                print("❌ 正の株数を入力してください")
                return
        except ValueError:
            print("❌ 数値を入力してください")
            return

        try:
            from backtest_paper_trading_system import OrderSide, OrderType

            order = await paper_trading_engine.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity
            )

            print(f"✅ 買い注文実行: {order.status.value}")
            print(f"注文ID: {order.order_id}")

            if order.status.value == "executed":
                print(f"約定価格: ¥{order.filled_price:.2f}")
                print(f"手数料: ¥{order.commission:.0f}")

        except Exception as e:
            print(f"❌ 注文エラー: {e}")

    async def _execute_sell_order(self):
        """売り注文実行"""

        # 保有ポジション表示
        portfolio = paper_trading_engine.portfolio
        if not portfolio.positions:
            print("❌ 売却可能なポジションがありません")
            return

        print("保有ポジション:")
        for symbol, pos in portfolio.positions.items():
            print(f"  {symbol}: {pos.quantity}株")

        symbol = input("売却銘柄コード: ").strip()
        if symbol not in portfolio.positions:
            print("❌ 指定された銘柄を保有していません")
            return

        max_quantity = portfolio.positions[symbol].quantity

        try:
            quantity = int(input(f"売却株数 (最大{max_quantity}株): ").strip())
            if quantity <= 0 or quantity > max_quantity:
                print(f"❌ 1-{max_quantity}株の範囲で入力してください")
                return
        except ValueError:
            print("❌ 数値を入力してください")
            return

        try:
            from backtest_paper_trading_system import OrderSide, OrderType

            order = await paper_trading_engine.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity
            )

            print(f"✅ 売り注文実行: {order.status.value}")
            print(f"注文ID: {order.order_id}")

            if order.status.value == "executed":
                print(f"約定価格: ¥{order.filled_price:.2f}")
                print(f"手数料: ¥{order.commission:.0f}")

        except Exception as e:
            print(f"❌ 注文エラー: {e}")

    async def run_interactive_interface(self):
        """インタラクティブインターフェース実行"""

        while True:
            await self.show_main_dashboard()

            choice = input("\n選択してください: ").strip().lower()

            if choice == 'q' or choice == 'quit':
                print("\n👋 システムを終了します")
                break
            elif choice == '1':
                await self.run_market_analysis()
            elif choice == '2':
                await self.run_individual_analysis()
            elif choice == '3':
                print("\n🚧 ポートフォリオ分析は開発中です")
            elif choice == '4':
                await self.run_backtest()
            elif choice == '5':
                await self.run_paper_trading()
            elif choice == '6':
                await self.run_data_quality_check()
            elif choice == '7':
                await self.run_system_settings()
            else:
                print("❌ 無効な選択です")

            input("\nEnterキーで続行...")

    async def run_data_quality_check(self):
        """データ品質チェック実行"""

        self.print_header("📊 データ品質チェック")

        await self.load_systems()

        print("データ品質評価中...")

        for symbol in self.default_symbols:
            try:
                result = await data_quality_manager.evaluate_data_quality(symbol)

                print(f"\n{symbol}:")
                print(f"  総合スコア: {result.get('overall_score', 0):.1f}/100")
                print(f"  品質レベル: {result.get('quality_level', 'unknown')}")
                print(f"  データ点数: {result.get('data_points', 0)}")

            except Exception as e:
                print(f"\n{symbol}: エラー - {e}")

    async def run_system_settings(self):
        """システム設定"""

        self.print_header("⚙️ システム設定")

        print("現在の設定:")
        print(f"  ポートフォリオ価値: ¥{self.portfolio_value:,.0f}")
        print(f"  監視銘柄: {', '.join(self.default_symbols)}")

        print("\n設定変更:")
        print("  [1] ポートフォリオ価値変更")
        print("  [2] 監視銘柄変更")
        print("  [3] 戻る")

        choice = input("\n選択してください: ").strip()

        if choice == "1":
            try:
                new_value = float(input("新しいポートフォリオ価値: ").strip())
                if new_value > 0:
                    self.portfolio_value = new_value
                    print(f"✅ ポートフォリオ価値を¥{new_value:,.0f}に変更しました")
                else:
                    print("❌ 正の値を入力してください")
            except ValueError:
                print("❌ 数値を入力してください")

        elif choice == "2":
            new_symbols = input("監視銘柄 (カンマ区切り): ").strip()
            if new_symbols:
                self.default_symbols = [s.strip() for s in new_symbols.split(',')]
                print(f"✅ 監視銘柄を変更しました: {', '.join(self.default_symbols)}")

# グローバルインスタンス
dashboard = DayTradingDashboard()

# メイン実行関数
async def main():
    """メイン実行"""

    print("🚀 デイトレーディングシステム起動中...")

    try:
        await dashboard.run_interactive_interface()
    except KeyboardInterrupt:
        print("\n\n👋 システムを終了します")
    except Exception as e:
        print(f"\n❌ システムエラー: {e}")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.WARNING)  # UIでは警告レベル以上のみ表示

    # メイン実行
    asyncio.run(main())