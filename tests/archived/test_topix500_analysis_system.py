#!/usr/bin/env python3
"""
TOPIX500分析システム テストスイート
Issue #314: TOPIX500全銘柄対応の包括的テスト

大規模処理・セクター分析・性能要件の検証
"""

import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.day_trade.topix.topix500_analysis_system import (
        PerformanceMetrics,
        SectorAnalysisResult,
        TOPIX500AnalysisSystem,
        TOPIX500Symbol,
    )
    from src.day_trade.utils.logging_config import get_context_logger
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なモジュールが見つかりません")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_context_logger(__name__)


def generate_mock_stock_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """モック株式データ生成"""
    dates = pd.date_range(start="2024-01-01", periods=days)

    # セクターごとに価格レンジを調整
    if symbol.startswith("79"):  # 銀行セクター
        base_price = 1000
    elif symbol.startswith("43"):  # 医薬品セクター
        base_price = 4000
    elif symbol.startswith("72"):  # 自動車セクター
        base_price = 2500
    elif symbol.startswith("99"):  # IT・通信セクター
        base_price = 5000
    else:
        base_price = 2000

    # 価格データ生成（リアルな変動パターン）
    price_changes = np.random.normal(0, 0.02, days)  # 日次2%標準偏差
    cumulative_changes = np.cumprod(1 + price_changes)

    close_prices = base_price * cumulative_changes

    return pd.DataFrame(
        {
            "Open": close_prices * np.random.uniform(0.99, 1.01, days),
            "High": close_prices * np.random.uniform(1.00, 1.05, days),
            "Low": close_prices * np.random.uniform(0.95, 1.00, days),
            "Close": close_prices,
            "Volume": np.random.randint(100000, 5000000, days),
            "Adj Close": close_prices,
        },
        index=dates,
    )


def generate_topix500_mock_data() -> Dict[str, pd.DataFrame]:
    """TOPIX500モックデータセット生成"""
    mock_symbols = []

    # 主要セクター代表銘柄
    sectors = {
        "banks": ["7182", "7186", "8316", "8354", "8411"],  # 銀行
        "pharmaceuticals": ["4502", "4503", "4568", "4578", "4581"],  # 医薬品
        "automotive": ["7203", "7201", "7267", "7269", "7270"],  # 自動車
        "technology": ["9984", "9983", "6758", "6861", "4755"],  # IT・通信
        "retail": ["8267", "9983", "3099", "7741", "9831"],  # 小売
        "manufacturing": ["6501", "6502", "6503", "6504", "6505"],  # 製造業
        "energy": ["5020", "5019", "1605", "1662", "1928"],  # エネルギー
        "real_estate": ["8802", "8804", "8830", "3289", "3290"],  # 不動産
        "construction": ["1812", "1813", "1801", "1802", "1803"],  # 建設
        "chemicals": ["4005", "4061", "4063", "4188", "4183"],  # 化学
    }

    # 各セクター5銘柄ずつ、計50銘柄の代表セット
    for sector_symbols in sectors.values():
        mock_symbols.extend(sector_symbols)

    # 残りの450銘柄を追加（簡易的に番号生成）
    base_codes = [
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "80",
        "81",
        "82",
    ]

    for i, base in enumerate(base_codes):
        if len(mock_symbols) >= 500:
            break
        for j in range(10):
            if len(mock_symbols) >= 500:
                break
            mock_symbols.append(f"{base}{j:02d}")

    # 500銘柄に調整
    mock_symbols = mock_symbols[:500]

    # データ生成
    stock_data = {}
    for symbol in mock_symbols:
        stock_data[symbol] = generate_mock_stock_data(symbol)

    logger.info(f"TOPIX500モックデータ生成完了: {len(stock_data)}銘柄")
    return stock_data


class TestTOPIX500AnalysisSystem:
    """TOPIX500分析システム テストクラス"""

    @pytest.fixture
    def mock_data_small(self):
        """小規模テスト用データ（10銘柄）"""
        test_symbols = [
            "7203",
            "8306",
            "9984",
            "4502",
            "7182",
            "8267",
            "6501",
            "5020",
            "8802",
            "1812",
        ]

        stock_data = {}
        for symbol in test_symbols:
            stock_data[symbol] = generate_mock_stock_data(symbol)

        return stock_data

    @pytest.fixture
    def mock_data_medium(self):
        """中規模テスト用データ（50銘柄）"""
        test_symbols = [f"{30+i:02d}{j:02d}" for i in range(10) for j in range(5)]

        stock_data = {}
        for symbol in test_symbols:
            stock_data[symbol] = generate_mock_stock_data(symbol)

        return stock_data

    @pytest.fixture
    def analysis_system(self):
        """TOPIX500分析システム インスタンス"""
        return TOPIX500AnalysisSystem(
            enable_cache=True,
            enable_parallel=True,
            max_concurrent_symbols=20,
            max_concurrent_sectors=5,
            memory_limit_gb=1.0,
            processing_timeout=15,
            batch_size=10,
        )

    @pytest.mark.asyncio
    async def test_system_initialization(self, analysis_system):
        """システム初期化テスト"""
        # 基本属性確認
        assert analysis_system.max_concurrent_symbols == 20
        assert analysis_system.max_concurrent_sectors == 5
        assert analysis_system.memory_limit_gb == 1.0
        assert analysis_system.processing_timeout == 15
        assert analysis_system.batch_size == 10

        # 統合コンポーネント確認
        assert hasattr(analysis_system, "cache_manager")
        assert hasattr(analysis_system, "parallel_engine")
        assert hasattr(analysis_system, "multi_timeframe_analyzer")
        assert hasattr(analysis_system, "ml_models")
        assert hasattr(analysis_system, "volatility_predictor")

        logger.info("✅ システム初期化テスト完了")

    @pytest.mark.asyncio
    async def test_small_batch_analysis(self, analysis_system, mock_data_small):
        """小規模バッチ分析テスト（10銘柄）"""
        start_time = time.time()

        try:
            result = await analysis_system.analyze_batch_comprehensive(
                stock_data=mock_data_small,
                enable_sector_analysis=True,
                enable_ml_prediction=True,
            )

            processing_time = time.time() - start_time

            # 結果検証
            assert "symbol_results" in result
            assert "sector_analysis" in result
            assert "performance_metrics" in result

            # 全銘柄処理確認
            assert len(result["symbol_results"]) == len(mock_data_small)

            # 処理時間確認（10銘柄は5秒以内）
            assert processing_time < 5.0

            # パフォーマンスメトリクス確認
            metrics = result["performance_metrics"]
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.total_symbols == len(mock_data_small)
            assert metrics.processing_time_seconds == pytest.approx(
                processing_time, rel=0.1
            )

            logger.info(
                f"✅ 小規模バッチテスト完了: {len(mock_data_small)}銘柄 {processing_time:.2f}秒"
            )

        except Exception as e:
            logger.error(f"小規模バッチテストエラー: {e}")
            raise

    @pytest.mark.asyncio
    async def test_medium_batch_analysis(self, analysis_system, mock_data_medium):
        """中規模バッチ分析テスト（50銘柄）"""
        start_time = time.time()

        try:
            result = await analysis_system.analyze_batch_comprehensive(
                stock_data=mock_data_medium,
                enable_sector_analysis=True,
                enable_ml_prediction=True,
            )

            processing_time = time.time() - start_time

            # 結果検証
            assert "symbol_results" in result
            assert "sector_analysis" in result

            # 全銘柄処理確認
            processed_count = len(
                [
                    r
                    for r in result["symbol_results"].values()
                    if r.get("success", False)
                ]
            )

            # 80%以上の成功率を期待
            success_rate = processed_count / len(mock_data_medium)
            assert success_rate >= 0.8

            # 処理時間確認（50銘柄は12秒以内）
            assert processing_time < 12.0

            logger.info(
                f"✅ 中規模バッチテスト完了: {processed_count}/{len(mock_data_medium)}銘柄 "
                f"{processing_time:.2f}秒 成功率{success_rate:.1%}"
            )

        except Exception as e:
            logger.error(f"中規模バッチテストエラー: {e}")
            raise

    @pytest.mark.asyncio
    async def test_sector_analysis_functionality(
        self, analysis_system, mock_data_medium
    ):
        """セクター分析機能テスト"""
        try:
            result = await analysis_system.analyze_batch_comprehensive(
                stock_data=mock_data_medium,
                enable_sector_analysis=True,
                enable_ml_prediction=False,  # ML無効でセクター分析に集中
            )

            sector_analysis = result["sector_analysis"]

            # セクター結果存在確認
            assert isinstance(sector_analysis, dict)
            assert len(sector_analysis) > 0

            # 各セクター結果の構造確認
            for sector_name, sector_result in sector_analysis.items():
                assert isinstance(sector_result, SectorAnalysisResult)
                assert hasattr(sector_result, "sector_name")
                assert hasattr(sector_result, "symbol_count")
                assert hasattr(sector_result, "avg_performance_score")
                assert hasattr(sector_result, "sector_trend")

                # セクター統計の妥当性確認
                assert sector_result.symbol_count > 0
                assert -1.0 <= sector_result.avg_performance_score <= 1.0
                assert sector_result.sector_trend in ["BULLISH", "BEARISH", "NEUTRAL"]

            logger.info(
                f"✅ セクター分析テスト完了: {len(sector_analysis)}セクター検出"
            )

        except Exception as e:
            logger.error(f"セクター分析テストエラー: {e}")
            raise

    @pytest.mark.asyncio
    async def test_memory_management(self, analysis_system, mock_data_medium):
        """メモリ管理テスト"""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # 複数回実行してメモリリーク確認
            for i in range(3):
                result = await analysis_system.analyze_batch_comprehensive(
                    stock_data=mock_data_medium,
                    enable_sector_analysis=True,
                    enable_ml_prediction=True,
                )

                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory

                # メモリ増加を制限内に抑制（制限の80%以内）
                memory_limit_mb = analysis_system.memory_limit_gb * 1024 * 0.8
                assert memory_increase < memory_limit_mb

                # ガベージコレクション実行
                gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024
            total_increase = final_memory - initial_memory

            logger.info(
                f"✅ メモリ管理テスト完了: 初期{initial_memory:.1f}MB → 最終{final_memory:.1f}MB "
                f"(+{total_increase:.1f}MB)"
            )

        except Exception as e:
            logger.error(f"メモリ管理テストエラー: {e}")
            raise

    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, analysis_system, mock_data_small):
        """キャッシュ効果テスト"""
        try:
            # 1回目実行
            start_time1 = time.time()
            result1 = await analysis_system.analyze_batch_comprehensive(
                stock_data=mock_data_small,
                enable_sector_analysis=True,
                enable_ml_prediction=True,
            )
            time1 = time.time() - start_time1

            # 2回目実行（キャッシュ効果期待）
            start_time2 = time.time()
            result2 = await analysis_system.analyze_batch_comprehensive(
                stock_data=mock_data_small,
                enable_sector_analysis=True,
                enable_ml_prediction=True,
            )
            time2 = time.time() - start_time2

            # キャッシュによる高速化確認
            if time1 > 0.5:  # 最初の処理が十分長い場合のみテスト
                speedup = time1 / time2
                assert speedup > 1.2  # 20%以上の高速化

                logger.info(
                    f"✅ キャッシュ効果テスト完了: 1回目{time1:.2f}秒 → 2回目{time2:.2f}秒 "
                    f"高速化{speedup:.1f}倍"
                )
            else:
                logger.info(
                    "✅ キャッシュテスト完了: 処理時間短すぎキャッシュ効果測定不可"
                )

            # 結果の一貫性確認
            assert len(result1["symbol_results"]) == len(result2["symbol_results"])

        except Exception as e:
            logger.error(f"キャッシュ効果テストエラー: {e}")
            raise

    @pytest.mark.asyncio
    async def test_error_handling(self, analysis_system):
        """エラーハンドリングテスト"""
        try:
            # 空データテスト
            empty_result = await analysis_system.analyze_batch_comprehensive(
                stock_data={}, enable_sector_analysis=True, enable_ml_prediction=True
            )

            assert "symbol_results" in empty_result
            assert len(empty_result["symbol_results"]) == 0

            # 不正データテスト
            invalid_data = {"INVALID": pd.DataFrame()}  # 空のDataFrame

            invalid_result = await analysis_system.analyze_batch_comprehensive(
                stock_data=invalid_data,
                enable_sector_analysis=True,
                enable_ml_prediction=True,
            )

            # エラーは発生させずに適切に処理
            assert "symbol_results" in invalid_result

            logger.info("✅ エラーハンドリングテスト完了")

        except Exception as e:
            logger.error(f"エラーハンドリングテストエラー: {e}")
            raise


class TestTOPIX500Performance:
    """TOPIX500性能要件テストクラス"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_scale_performance(self):
        """フルスケール性能テスト（500銘柄）"""
        logger.info("=== TOPIX500フルスケール性能テスト開始 ===")

        # フルスケールデータ生成
        start_data_gen = time.time()
        full_stock_data = generate_topix500_mock_data()
        data_gen_time = time.time() - start_data_gen

        logger.info(f"データ生成完了: {len(full_stock_data)}銘柄 {data_gen_time:.2f}秒")

        # 性能最適化設定でシステム初期化
        analysis_system = TOPIX500AnalysisSystem(
            enable_cache=True,
            enable_parallel=True,
            max_concurrent_symbols=50,  # 最大並列数
            max_concurrent_sectors=10,
            memory_limit_gb=1.0,
            processing_timeout=20,  # 目標時間
            batch_size=25,
        )

        try:
            # フルスケール分析実行
            start_time = time.time()

            result = await analysis_system.analyze_batch_comprehensive(
                stock_data=full_stock_data,
                enable_sector_analysis=True,
                enable_ml_prediction=True,
            )

            total_time = time.time() - start_time

            # 性能要件検証
            performance_metrics = result["performance_metrics"]

            # 処理時間要件（20秒以内）
            assert total_time <= 20.0, f"処理時間超過: {total_time:.2f}秒 > 20秒"

            # メモリ使用量要件（1GB以内）
            assert (
                performance_metrics.peak_memory_mb <= 1024
            ), f"メモリ使用量超過: {performance_metrics.peak_memory_mb:.1f}MB > 1024MB"

            # 成功率要件（85%以上）
            success_count = len(
                [
                    r
                    for r in result["symbol_results"].values()
                    if r.get("success", False)
                ]
            )
            success_rate = success_count / len(full_stock_data)
            assert success_rate >= 0.85, f"成功率不足: {success_rate:.1%} < 85%"

            # セクター分析完了確認
            sector_count = len(result["sector_analysis"])
            assert (
                sector_count >= 5
            ), f"セクター分析不足: {sector_count}セクター < 5セクター"

            logger.info("=== TOPIX500フルスケール性能テスト結果 ===")
            logger.info(
                f"総処理時間: {total_time:.2f}秒 ({'✅' if total_time <= 20 else '❌'})"
            )
            logger.info(
                f"処理成功: {success_count}/{len(full_stock_data)}銘柄 ({success_rate:.1%})"
            )
            logger.info(f"ピークメモリ: {performance_metrics.peak_memory_mb:.1f}MB")
            logger.info(f"セクター数: {sector_count}")
            logger.info(
                f"平均処理時間/銘柄: {(total_time/len(full_stock_data)*1000):.1f}ms"
            )

            # スループット計算
            throughput = len(full_stock_data) / total_time
            logger.info(f"スループット: {throughput:.1f}銘柄/秒")

            logger.info("✅ TOPIX500フルスケール性能テスト完了")

        except Exception as e:
            logger.error(f"フルスケール性能テストエラー: {e}")
            raise

        finally:
            # クリーンアップ
            if hasattr(analysis_system, "shutdown"):
                analysis_system.shutdown()


if __name__ == "__main__":
    """テスト実行"""
    print("=== TOPIX500分析システム テストスイート ===")

    # 基本テスト実行
    try:
        # 小規模テスト
        print("\n1. 小規模テスト（10銘柄）...")
        pytest.main(
            [
                __file__ + "::TestTOPIX500AnalysisSystem::test_small_batch_analysis",
                "-v",
                "-s",
            ]
        )

        # 中規模テスト
        print("\n2. 中規模テスト（50銘柄）...")
        pytest.main(
            [
                __file__ + "::TestTOPIX500AnalysisSystem::test_medium_batch_analysis",
                "-v",
                "-s",
            ]
        )

        # セクター分析テスト
        print("\n3. セクター分析テスト...")
        pytest.main(
            [
                __file__
                + "::TestTOPIX500AnalysisSystem::test_sector_analysis_functionality",
                "-v",
                "-s",
            ]
        )

        # キャッシュテスト
        print("\n4. キャッシュ効果テスト...")
        pytest.main(
            [
                __file__ + "::TestTOPIX500AnalysisSystem::test_cache_effectiveness",
                "-v",
                "-s",
            ]
        )

        print("\n✅ 基本テストスイート完了")

        # フルスケールテスト（オプション）
        run_full_test = input("\nフルスケールテスト（500銘柄）を実行しますか？ (y/N): ")
        if run_full_test.lower() == "y":
            print("\n5. フルスケール性能テスト...")
            pytest.main(
                [
                    __file__ + "::TestTOPIX500Performance::test_full_scale_performance",
                    "-v",
                    "-s",
                    "-m",
                    "slow",
                ]
            )

    except KeyboardInterrupt:
        print("\nテスト中断")
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()
