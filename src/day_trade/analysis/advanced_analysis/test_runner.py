#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analysis Test Runner - 高度技術分析テストランナー

システムテストとデモンストレーション機能
"""

import asyncio
import logging
import sys
import os

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

from .analyzer import AdvancedTechnicalAnalysis


class TestRunner:
    """テストランナー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = AdvancedTechnicalAnalysis()

    async def run_comprehensive_test(self):
        """包括的テスト実行"""

        print("=== 📈 高度技術分析システムテスト ===")

        test_symbols = ["7203", "8306"]

        for symbol in test_symbols:
            print(f"\n🔬 {symbol} 高度技術分析")

            try:
                # 包括的技術分析実行
                result = await self.analyzer.perform_comprehensive_analysis(symbol)

                print(f"  ✅ 分析完了:")
                print(f"    総合センチメント: {result.overall_sentiment}")
                print(f"    信頼度: {result.confidence_score:.1%}")
                print(f"    リスクレベル: {result.risk_level}")
                print(f"    検出シグナル数: {len(result.signals)}")
                print(f"    検出パターン数: {len(result.patterns)}")
                print(f"    計算指標数: {len(result.indicators)}")

                # 主要シグナル表示
                strong_signals = [s for s in result.signals if s.strength > 60]
                if strong_signals:
                    print(f"    主要シグナル:")
                    for signal in strong_signals[:5]:
                        print(f"      - {signal.indicator_name}: {signal.signal_type} ({signal.strength:.0f})")

                # パターン表示
                if result.patterns:
                    print(f"    検出パターン:")
                    for pattern in result.patterns[:3]:
                        print(f"      - {pattern.pattern_name} (信頼度: {pattern.reliability:.1%})")

                # 推奨事項表示
                if result.recommendations:
                    print(f"    推奨事項:")
                    for rec in result.recommendations[:3]:
                        print(f"      - {rec}")

            except Exception as e:
                print(f"  ❌ {symbol} エラー: {e}")

        print(f"\n✅ 高度技術分析システムテスト完了")


# テスト実行関数
async def run_advanced_technical_analysis_test():
    """高度技術分析テスト実行"""
    test_runner = TestRunner()
    await test_runner.run_comprehensive_test()


def main():
    """メイン関数"""
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # テスト実行
    asyncio.run(run_advanced_technical_analysis_test())


if __name__ == "__main__":
    main()