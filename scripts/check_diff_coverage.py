#!/usr/bin/env python3
"""
差分カバレッジ検証スクリプト

新規追加・変更されたコードに対して高いカバレッジ閾値を適用し、
コード品質の劣化を防止する。Issue #129 対応。
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent


class DiffCoverageChecker:
    """差分カバレッジ検証クラス"""

    def __init__(
        self,
        base_ref: str = "origin/main",
        min_coverage: float = 80.0,
        critical_files_min_coverage: float = 90.0,
    ):
        """
        Args:
            base_ref: 比較ベースとなるGitリファレンス
            min_coverage: 新規コードの最小カバレッジ閾値（%）
            critical_files_min_coverage: 重要ファイルの最小カバレッジ閾値（%）
        """
        self.base_ref = base_ref
        self.min_coverage = min_coverage
        self.critical_files_min_coverage = critical_files_min_coverage
        self.errors = []
        self.warnings = []

        # 重要ファイルパターン（高い品質が求められる）
        self.critical_patterns = [
            "*/models/*",
            "*/core/*",
            "*/data/stock_fetcher.py",
            "*/utils/exceptions.py",
            "*/utils/enhanced_error_handler.py",
        ]

    def check_diff_coverage(self) -> bool:
        """差分カバレッジをチェック"""

        logger.info(f"🔍 差分カバレッジ検証開始 (base: {self.base_ref})")

        try:
            # 1. 変更されたファイル一覧を取得
            changed_files = self._get_changed_files()

            if not changed_files:
                logger.info("✅ 変更されたPythonファイルがないため、チェックをスキップ")
                return True

            logger.info(f"📝 変更されたファイル数: {len(changed_files)}")

            # 2. 現在のカバレッジデータを取得
            coverage_data = self._get_coverage_data()

            if not coverage_data:
                logger.warning(
                    "⚠️ カバレッジデータが取得できないため、基本チェックのみ実行"
                )
                return self._basic_validation(changed_files)

            # 3. 各ファイルのカバレッジをチェック
            self._check_file_coverage(changed_files, coverage_data)

            # 4. 結果レポート
            self._generate_report()

            return len(self.errors) == 0

        except Exception as e:
            logger.error(f"❌ 差分カバレッジ検証中にエラー: {e}")
            return False

    def _get_changed_files(self) -> List[str]:
        """変更されたPythonファイル一覧を取得"""

        try:
            # Git diffでPythonファイルの変更を取得
            cmd = [
                "git",
                "diff",
                "--name-only",
                f"{self.base_ref}...HEAD",
                "--",
                "*.py",
            ]

            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(f"Git diff実行失敗: {result.stderr}")
                return []

            changed_files = [
                f.strip()
                for f in result.stdout.split("\n")
                if f.strip() and f.startswith("src/")
            ]

            return changed_files

        except Exception as e:
            logger.error(f"変更ファイル取得エラー: {e}")
            return []

    def _get_coverage_data(self) -> Optional[Dict]:
        """カバレッジデータを取得"""

        coverage_file = PROJECT_ROOT / "coverage.json"

        if not coverage_file.exists():
            logger.warning("coverage.json が見つかりません")
            return None

        try:
            with open(coverage_file, encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"カバレッジデータ読み込みエラー: {e}")
            return None

    def _check_file_coverage(
        self, changed_files: List[str], coverage_data: Dict
    ) -> None:
        """各ファイルのカバレッジをチェック"""

        files_data = coverage_data.get("files", {})

        for file_path in changed_files:
            # srcから始まるパスを正規化
            normalized_path = str(Path(file_path))

            # カバレッジデータから該当ファイルを検索
            coverage_info = None
            for cov_file_path, info in files_data.items():
                if normalized_path in cov_file_path or file_path in cov_file_path:
                    coverage_info = info
                    break

            if not coverage_info:
                self.warnings.append(f"カバレッジ情報が見つかりません: {file_path}")
                continue

            # カバレッジ率を計算
            total_lines = coverage_info.get("summary", {}).get("num_statements", 0)
            covered_lines = coverage_info.get("summary", {}).get("covered_lines", 0)

            if total_lines == 0:
                continue  # 実行可能行がない場合はスキップ

            coverage_percent = (covered_lines / total_lines) * 100

            # 閾値判定
            required_coverage = self._get_required_coverage(file_path)

            logger.info(
                f"📊 {file_path}: {coverage_percent:.1f}% "
                f"(required: {required_coverage:.1f}%)"
            )

            if coverage_percent < required_coverage:
                self.errors.append(
                    f"{file_path}: カバレッジが閾値を下回っています "
                    f"({coverage_percent:.1f}% < {required_coverage:.1f}%)"
                )
            else:
                logger.info(f"✅ {file_path}: カバレッジ要件を満たしています")

    def _get_required_coverage(self, file_path: str) -> float:
        """ファイルに応じた必要カバレッジ率を取得"""

        # 重要ファイルの判定
        for pattern in self.critical_patterns:
            if self._matches_pattern(file_path, pattern):
                return self.critical_files_min_coverage

        return self.min_coverage

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """ファイルパスがパターンにマッチするか判定"""

        import fnmatch

        normalized_path = file_path.replace("\\", "/")
        normalized_pattern = pattern.replace("\\", "/")

        return fnmatch.fnmatch(normalized_path, normalized_pattern)

    def _basic_validation(self, changed_files: List[str]) -> bool:
        """基本的な検証（カバレッジデータが取得できない場合）"""

        logger.info("📋 基本検証モードで実行")

        for file_path in changed_files:
            # ファイルの存在確認
            full_path = PROJECT_ROOT / file_path

            if not full_path.exists():
                self.errors.append(f"ファイルが存在しません: {file_path}")
                continue

            # 基本的な構文チェック
            try:
                with open(full_path, encoding="utf-8") as f:
                    compile(f.read(), str(full_path), "exec")
                logger.info(f"✅ {file_path}: 構文チェック完了")

            except SyntaxError as e:
                self.errors.append(f"{file_path}: 構文エラー - {e}")

            except Exception as e:
                self.warnings.append(f"{file_path}: 検証中にエラー - {e}")

        return len(self.errors) == 0

    def _generate_report(self) -> None:
        """検証結果レポートを生成"""

        logger.info("📊 差分カバレッジ検証結果")

        if len(self.errors) == 0:
            logger.info("🎉 全ての変更ファイルがカバレッジ要件を満たしています")
        else:
            logger.error(
                f"❌ {len(self.errors)}個のファイルがカバレッジ要件を満たしていません"
            )

        if len(self.warnings) > 0:
            logger.warning(f"⚠️ {len(self.warnings)}件の警告があります")

        # エラー詳細
        if self.errors:
            logger.error("🚨 エラー詳細:")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")

        # 警告詳細
        if self.warnings:
            logger.warning("⚠️ 警告詳細:")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")

        # サマリー
        logger.info("📋 検証サマリー:")
        logger.info(f"  - 最小カバレッジ閾値: {self.min_coverage}%")
        logger.info(f"  - 重要ファイル閾値: {self.critical_files_min_coverage}%")
        logger.info(f"  - エラー: {len(self.errors)}件")
        logger.info(f"  - 警告: {len(self.warnings)}件")


def main():
    """メイン関数"""

    import argparse

    parser = argparse.ArgumentParser(description="差分カバレッジ検証")
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="比較ベースとなるGitリファレンス",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=80.0,
        help="新規コードの最小カバレッジ閾値（%）",
    )
    parser.add_argument(
        "--critical-coverage",
        type=float,
        default=90.0,
        help="重要ファイルの最小カバレッジ閾値（%）",
    )

    args = parser.parse_args()

    checker = DiffCoverageChecker(
        base_ref=args.base_ref,
        min_coverage=args.min_coverage,
        critical_files_min_coverage=args.critical_coverage,
    )

    success = checker.check_diff_coverage()

    if success:
        logger.info("✅ 差分カバレッジ検証成功")
        sys.exit(0)
    else:
        logger.error("❌ 差分カバレッジ検証失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
