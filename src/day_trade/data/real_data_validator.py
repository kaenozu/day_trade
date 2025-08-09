#!/usr/bin/env python3
"""
実市場データ取得・検証システム

Issue #321: 実データでの最終動作確認テスト
実際の市場データの品質確認・整合性検証
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Tuple

import pandas as pd
import yfinance as yf

from ..utils.performance_monitor import get_performance_monitor
from ..utils.structured_logging import get_structured_logger

logger = get_structured_logger()
perf_monitor = get_performance_monitor()


class DataQuality(Enum):
    """データ品質レベル"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class DataQualityMetrics:
    """データ品質メトリクス"""

    symbol: str
    completeness: float  # データ完全性 (0-1)
    accuracy: float  # データ精度 (0-1)
    consistency: float  # データ整合性 (0-1)
    timeliness: float  # データ適時性 (0-1)
    overall_quality: DataQuality
    issues: List[str]
    record_count: int
    date_range: Tuple[str, str]


class RealDataValidator:
    """実市場データ検証システム"""

    def __init__(self):
        self.topix_core30 = [
            "7203.T",  # トヨタ自動車
            "8306.T",  # 三菱UFJフィナンシャル・グループ
            "9984.T",  # ソフトバンクグループ
            "6758.T",  # ソニーグループ
            "9432.T",  # 日本電信電話
            "8001.T",  # 伊藤忠商事
            "6861.T",  # キーエンス
            "8058.T",  # 三菱商事
            "4502.T",  # 武田薬品工業
            "7974.T",  # 任天堂
            "8411.T",  # みずほフィナンシャルグループ
            "8316.T",  # 三井住友フィナンシャルグループ
            "8031.T",  # 三井物産
            "8053.T",  # 住友商事
            "7751.T",  # キヤノン
            "6981.T",  # 村田製作所
            "9983.T",  # ファーストリテイリング
            "4568.T",  # 第一三共
            "6367.T",  # ダイキン工業
            "6954.T",  # ファナック
            "9434.T",  # ソフトバンク
            "4063.T",  # 信越化学工業
            "6098.T",  # リクルートホールディングス
            "8035.T",  # 東京エレクトロン
            "4523.T",  # エーザイ
            "7267.T",  # 本田技研工業
            "9101.T",  # 日本郵船
            "8766.T",  # 東京海上ホールディングス
            "8802.T",  # 三菱地所
            "5401.T",  # 日本製鉄
        ]

        # テスト用追加銘柄（合計85銘柄にするため）
        additional_symbols = [
            "4689.T",
            "2914.T",
            "4755.T",
            "3659.T",
            "9613.T",
            "2432.T",
            "4385.T",
            "9437.T",
            "4704.T",
            "4751.T",
            "3382.T",
            "2801.T",
            "2502.T",
            "9201.T",
            "9202.T",
            "5020.T",
            "9501.T",
            "9502.T",
            "8604.T",
            "7182.T",
            "4005.T",
            "4061.T",
            "8795.T",
            "4777.T",
            "3776.T",
            "4478.T",
            "4485.T",
            "4490.T",
            "3900.T",
            "3774.T",
            "4382.T",
            "4386.T",
            "4475.T",
            "4421.T",
            "3655.T",
            "3844.T",
            "4833.T",
            "4563.T",
            "4592.T",
            "4564.T",
            "4588.T",
            "4596.T",
            "4591.T",
            "4565.T",
            "7707.T",
            "3692.T",
            "3656.T",
            "3760.T",
            "9449.T",
            "4726.T",
            "7779.T",
            "6178.T",
            "4847.T",
            "4598.T",
            "4880.T",
        ]

        self.test_symbols = self.topix_core30 + additional_symbols
        self.validation_results = {}

        logger.info(
            f"実データ検証システム初期化完了 - 対象銘柄: {len(self.test_symbols)}銘柄"
        )

    def validate_market_data(self, days: int = 30) -> Dict[str, DataQualityMetrics]:
        """市場データ検証実行"""

        with logger.operation_context(
            "real_data_validation",
            data={"symbols": len(self.test_symbols), "days": days},
        ):
            logger.log_info("実市場データ検証開始")

            start_time = time.time()
            validation_results = {}

            # 日付範囲設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            logger.log_info(
                f"検証期間: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}"
            )

            # 銘柄別データ検証
            successful_validations = 0

            for i, symbol in enumerate(self.test_symbols):
                try:
                    with perf_monitor.monitor(f"validate_symbol_{symbol}"):
                        metrics = self._validate_single_symbol(
                            symbol, start_date, end_date
                        )
                        validation_results[symbol] = metrics

                        if metrics.overall_quality in [
                            DataQuality.EXCELLENT,
                            DataQuality.GOOD,
                        ]:
                            successful_validations += 1

                        # プログレス表示（10銘柄ごと）
                        if (i + 1) % 10 == 0:
                            progress = (i + 1) / len(self.test_symbols) * 100
                            logger.log_info(
                                f"検証進捗: {progress:.1f}% ({i + 1}/{len(self.test_symbols)})"
                            )

                except Exception as e:
                    logger.log_error(f"銘柄 {symbol} の検証でエラー", e)
                    validation_results[symbol] = self._create_error_metrics(
                        symbol, str(e)
                    )

            validation_time = time.time() - start_time

            # 検証結果サマリー
            summary = self._generate_validation_summary(validation_results)

            logger.log_info(
                "実市場データ検証完了",
                {
                    "total_symbols": len(self.test_symbols),
                    "successful_validations": successful_validations,
                    "validation_time_seconds": validation_time,
                    "success_rate": successful_validations / len(self.test_symbols),
                    "summary": summary,
                },
            )

            self.validation_results = validation_results
            return validation_results

    def _validate_single_symbol(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> DataQualityMetrics:
        """個別銘柄データ検証"""

        issues = []

        try:
            # Yahoo Financeからデータ取得
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                return DataQualityMetrics(
                    symbol=symbol,
                    completeness=0.0,
                    accuracy=0.0,
                    consistency=0.0,
                    timeliness=0.0,
                    overall_quality=DataQuality.UNUSABLE,
                    issues=["データ取得不可"],
                    record_count=0,
                    date_range=("", ""),
                )

            # データ品質メトリクス計算
            completeness = self._calculate_completeness(hist, start_date, end_date)
            accuracy = self._calculate_accuracy(hist)
            consistency = self._calculate_consistency(hist)
            timeliness = self._calculate_timeliness(hist, end_date)

            # 問題点検出
            if completeness < 0.9:
                issues.append(f"データ欠損率高: {(1-completeness)*100:.1f}%")

            if accuracy < 0.8:
                issues.append(f"データ精度低: {accuracy:.2f}")

            if consistency < 0.8:
                issues.append(f"データ整合性低: {consistency:.2f}")

            if timeliness < 0.8:
                issues.append(f"データ適時性低: {timeliness:.2f}")

            # 全体品質評価
            overall_score = (completeness + accuracy + consistency + timeliness) / 4

            if overall_score >= 0.9:
                overall_quality = DataQuality.EXCELLENT
            elif overall_score >= 0.8:
                overall_quality = DataQuality.GOOD
            elif overall_score >= 0.7:
                overall_quality = DataQuality.FAIR
            elif overall_score >= 0.5:
                overall_quality = DataQuality.POOR
            else:
                overall_quality = DataQuality.UNUSABLE

            return DataQualityMetrics(
                symbol=symbol,
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                timeliness=timeliness,
                overall_quality=overall_quality,
                issues=issues,
                record_count=len(hist),
                date_range=(
                    hist.index.min().strftime("%Y-%m-%d"),
                    hist.index.max().strftime("%Y-%m-%d"),
                ),
            )

        except Exception as e:
            return self._create_error_metrics(symbol, str(e))

    def _calculate_completeness(
        self, data: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> float:
        """データ完全性計算"""

        # 営業日数計算（土日を除く）
        business_days = pd.bdate_range(start_date, end_date)
        expected_records = len(business_days)
        actual_records = len(data)

        if expected_records == 0:
            return 1.0

        # 完全性スコア（営業日に対する実際のレコード率）
        completeness = min(1.0, actual_records / expected_records)

        return completeness

    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """データ精度計算"""

        accuracy_score = 1.0

        # 価格データの妥当性チェック
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                # 負の価格チェック
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    accuracy_score -= 0.2

                # 異常な価格変動チェック（前日比500%以上）
                if len(data) > 1:
                    price_changes = data[col].pct_change().abs()
                    extreme_changes = (price_changes > 5.0).sum()
                    if extreme_changes > 0:
                        accuracy_score -= 0.1

        # 高値 >= 安値の関係チェック
        if "High" in data.columns and "Low" in data.columns:
            invalid_hl = (data["High"] < data["Low"]).sum()
            if invalid_hl > 0:
                accuracy_score -= 0.3

        # 始値・終値が高値・安値の範囲内かチェック
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            invalid_ohlc = (
                (data["Open"] > data["High"])
                | (data["Open"] < data["Low"])
                | (data["Close"] > data["High"])
                | (data["Close"] < data["Low"])
            ).sum()
            if invalid_ohlc > 0:
                accuracy_score -= 0.2

        return max(0.0, accuracy_score)

    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """データ整合性計算"""

        consistency_score = 1.0

        # 出来高の整合性チェック
        if "Volume" in data.columns:
            # 出来高0のチェック
            zero_volume = (data["Volume"] == 0).sum()
            if zero_volume > len(data) * 0.1:  # 10%以上が0出来高
                consistency_score -= 0.2

            # 異常な出来高（平均の10倍以上）
            mean_volume = data["Volume"].mean()
            if mean_volume > 0:
                extreme_volume = (data["Volume"] > mean_volume * 10).sum()
                if extreme_volume > len(data) * 0.05:  # 5%以上が異常出来高
                    consistency_score -= 0.1

        # 価格データの一貫性（欠損値チェック）
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                missing_values = data[col].isna().sum()
                if missing_values > 0:
                    consistency_score -= 0.1

        # 時系列順序の整合性
        if not data.index.is_monotonic_increasing:
            consistency_score -= 0.3

        return max(0.0, consistency_score)

    def _calculate_timeliness(self, data: pd.DataFrame, end_date: datetime) -> float:
        """データ適時性計算"""

        if data.empty:
            return 0.0

        # 最新データの日付
        latest_date = data.index.max()

        # 現在時刻からの遅延日数
        delay_days = (end_date.date() - latest_date.date()).days

        # 適時性スコア（遅延日数に基づく）
        if delay_days <= 1:
            timeliness = 1.0
        elif delay_days <= 2:
            timeliness = 0.8
        elif delay_days <= 7:
            timeliness = 0.6
        elif delay_days <= 30:
            timeliness = 0.4
        else:
            timeliness = 0.2

        return timeliness

    def _create_error_metrics(self, symbol: str, error_msg: str) -> DataQualityMetrics:
        """エラー時のメトリクス作成"""
        return DataQualityMetrics(
            symbol=symbol,
            completeness=0.0,
            accuracy=0.0,
            consistency=0.0,
            timeliness=0.0,
            overall_quality=DataQuality.UNUSABLE,
            issues=[f"データ取得エラー: {error_msg}"],
            record_count=0,
            date_range=("", ""),
        )

    def _generate_validation_summary(
        self, results: Dict[str, DataQualityMetrics]
    ) -> Dict[str, Any]:
        """検証結果サマリー生成"""

        quality_counts = {quality: 0 for quality in DataQuality}

        total_records = 0
        total_issues = 0
        avg_completeness = 0.0
        avg_accuracy = 0.0
        avg_consistency = 0.0
        avg_timeliness = 0.0

        valid_results = []

        for metrics in results.values():
            quality_counts[metrics.overall_quality] += 1
            total_records += metrics.record_count
            total_issues += len(metrics.issues)

            if metrics.overall_quality != DataQuality.UNUSABLE:
                valid_results.append(metrics)

        if valid_results:
            avg_completeness = sum(m.completeness for m in valid_results) / len(
                valid_results
            )
            avg_accuracy = sum(m.accuracy for m in valid_results) / len(valid_results)
            avg_consistency = sum(m.consistency for m in valid_results) / len(
                valid_results
            )
            avg_timeliness = sum(m.timeliness for m in valid_results) / len(
                valid_results
            )

        return {
            "total_symbols": len(results),
            "quality_distribution": {
                quality.value: count for quality, count in quality_counts.items()
            },
            "usable_symbols": quality_counts[DataQuality.EXCELLENT]
            + quality_counts[DataQuality.GOOD]
            + quality_counts[DataQuality.FAIR],
            "total_records": total_records,
            "total_issues": total_issues,
            "average_metrics": {
                "completeness": avg_completeness,
                "accuracy": avg_accuracy,
                "consistency": avg_consistency,
                "timeliness": avg_timeliness,
            },
        }

    def get_high_quality_symbols(
        self, min_quality: DataQuality = DataQuality.GOOD
    ) -> List[str]:
        """高品質データ銘柄取得"""

        quality_levels = {
            DataQuality.EXCELLENT: 5,
            DataQuality.GOOD: 4,
            DataQuality.FAIR: 3,
            DataQuality.POOR: 2,
            DataQuality.UNUSABLE: 1,
        }

        min_level = quality_levels[min_quality]

        high_quality = []
        for symbol, metrics in self.validation_results.items():
            if quality_levels[metrics.overall_quality] >= min_level:
                high_quality.append(symbol)

        return high_quality

    def generate_validation_report(self) -> str:
        """検証レポート生成"""

        if not self.validation_results:
            return (
                "検証結果がありません。先に validate_market_data() を実行してください。"
            )

        summary = self._generate_validation_summary(self.validation_results)

        report = []
        report.append("実市場データ検証レポート")
        report.append("=" * 50)
        report.append(f"検証日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"対象銘柄数: {summary['total_symbols']}")
        report.append(f"総レコード数: {summary['total_records']:,}")
        report.append("")

        report.append("品質分布:")
        for quality, count in summary["quality_distribution"].items():
            percentage = count / summary["total_symbols"] * 100
            report.append(f"  {quality.upper()}: {count}銘柄 ({percentage:.1f}%)")

        report.append("")
        report.append(
            f"利用可能銘柄数: {summary['usable_symbols']} / {summary['total_symbols']}"
        )
        report.append(
            f"利用可能率: {summary['usable_symbols'] / summary['total_symbols'] * 100:.1f}%"
        )

        report.append("")
        report.append("平均品質メトリクス:")
        avg_metrics = summary["average_metrics"]
        report.append(f"  データ完全性: {avg_metrics['completeness']:.3f}")
        report.append(f"  データ精度: {avg_metrics['accuracy']:.3f}")
        report.append(f"  データ整合性: {avg_metrics['consistency']:.3f}")
        report.append(f"  データ適時性: {avg_metrics['timeliness']:.3f}")

        # 問題のある銘柄
        problem_symbols = []
        for symbol, metrics in self.validation_results.items():
            if metrics.overall_quality in [DataQuality.POOR, DataQuality.UNUSABLE]:
                problem_symbols.append(
                    (symbol, metrics.overall_quality.value, metrics.issues)
                )

        if problem_symbols:
            report.append("")
            report.append("問題のある銘柄:")
            for symbol, quality, issues in problem_symbols[:10]:  # 最大10銘柄
                report.append(f"  {symbol} ({quality}): {', '.join(issues[:3])}")

        # 高品質銘柄（ML分析推奨）
        excellent_symbols = self.get_high_quality_symbols(DataQuality.EXCELLENT)
        if excellent_symbols:
            report.append("")
            report.append(f"優良品質銘柄 ({len(excellent_symbols)}銘柄): ML分析推奨")
            report.append(f"  {', '.join(excellent_symbols[:20])}")  # 最大20銘柄表示
            if len(excellent_symbols) > 20:
                report.append(f"  ... 他{len(excellent_symbols) - 20}銘柄")

        return "\n".join(report)


if __name__ == "__main__":
    # テスト実行
    print("Real Market Data Validation Test")
    print("=" * 50)

    try:
        validator = RealDataValidator()

        # 実市場データ検証実行
        print("実市場データ検証開始...")
        validation_results = validator.validate_market_data(days=30)

        # 検証レポート生成・表示
        report = validator.generate_validation_report()
        print("\n" + report)

        # 高品質銘柄取得
        excellent_symbols = validator.get_high_quality_symbols(DataQuality.EXCELLENT)
        good_symbols = validator.get_high_quality_symbols(DataQuality.GOOD)

        print(f"\n優良品質銘柄: {len(excellent_symbols)}銘柄")
        print(f"良品質銘柄: {len(good_symbols)}銘柄")

        if len(good_symbols) >= 85:
            print(f"\n✅ ML分析対象85銘柄の確保成功: {len(good_symbols)}銘柄利用可能")
        else:
            print(f"\n⚠️ ML分析対象85銘柄不足: {len(good_symbols)}銘柄のみ利用可能")

        print("\n実市場データ検証完了")

    except Exception as e:
        print(f"実市場データ検証エラー: {e}")
        import traceback

        traceback.print_exc()
