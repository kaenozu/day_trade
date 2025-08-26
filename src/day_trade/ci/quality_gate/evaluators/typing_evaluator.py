#!/usr/bin/env python3
"""
品質ゲートシステム - 型チェック評価器

型アノテーションとMyPy型チェックの評価を行うモジュール。
型安全性の品質判定を実施する。
"""

import subprocess
from typing import List

from ..types import QualityGate, QualityResult
from .base_evaluator import BaseEvaluator


class TypingEvaluator(BaseEvaluator):
    """型チェック評価器
    
    型アノテーションの網羅性とMyPy型チェック結果を評価し、
    品質ゲートのパス/フェイル判定を行う。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        super().__init__(project_root)

    async def evaluate(self, gate: QualityGate) -> QualityResult:
        """型チェックゲートを評価
        
        MyPyを使用して型チェックを実行し、
        型安全性スコアによる品質判定を実行する。
        
        Args:
            gate: 評価対象の型チェックゲート
            
        Returns:
            型チェック評価結果
        """
        try:
            # MyPy型チェックを実行
            typing_results = self._run_mypy_check()
            typing_score = typing_results["typing_score"]

            # 品質レベル判定
            level = self._determine_quality_level(typing_score, gate)
            passed = typing_score >= gate.threshold_acceptable

            # 推奨事項生成
            recommendations = self._generate_typing_recommendations(
                typing_results, gate
            )

            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=typing_score,
                level=level,
                passed=passed,
                message=f"型チェックスコア: {typing_score:.1f}%",
                details=typing_results,
                recommendations=recommendations,
            )

        except Exception as e:
            return self._create_error_result(gate, e)

    def _run_mypy_check(self) -> dict:
        """MyPy型チェックを実行
        
        Returns:
            型チェック結果
        """
        try:
            result = subprocess.run(
                [
                    "mypy", 
                    "src/", 
                    "--ignore-missing-imports", 
                    "--follow-imports=silent",
                    "--show-error-codes",
                    "--no-strict-optional",  # より寛容な設定
                ],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                # 型エラーなし
                return {
                    "typing_score": 100.0,
                    "error_count": 0,
                    "warning_count": 0,
                    "note_count": 0,
                    "mypy_output": "No type errors found",
                    "error_details": [],
                }
            else:
                # 型エラーを解析
                return self._parse_mypy_output(result.stdout, result.stderr)

        except subprocess.TimeoutExpired:
            return {
                "typing_score": 0.0,
                "error_count": 0,
                "error": "MyPy型チェックがタイムアウトしました",
            }
        except FileNotFoundError:
            return {
                "typing_score": 50.0,
                "error_count": 0,
                "error": "MyPyが見つかりません。pip install mypy でインストールしてください。",
            }

    def _parse_mypy_output(self, stdout: str, stderr: str) -> dict:
        """MyPy出力を解析
        
        Args:
            stdout: 標準出力
            stderr: 標準エラー出力
            
        Returns:
            解析結果
        """
        output_lines = stdout.split('\n') if stdout else []
        
        error_count = 0
        warning_count = 0
        note_count = 0
        error_details = []
        
        # 各行を解析
        for line in output_lines:
            if not line.strip():
                continue
                
            if ": error:" in line:
                error_count += 1
                error_details.append(self._parse_error_line(line, "error"))
            elif ": warning:" in line:
                warning_count += 1
                error_details.append(self._parse_error_line(line, "warning"))
            elif ": note:" in line:
                note_count += 1
                error_details.append(self._parse_error_line(line, "note"))

        # スコア計算（エラーが多いほど低下）
        typing_score = self._calculate_typing_score(error_count, warning_count, note_count)

        return {
            "typing_score": typing_score,
            "error_count": error_count,
            "warning_count": warning_count,
            "note_count": note_count,
            "mypy_output": stdout[:2000],  # 最初の2000文字のみ
            "error_details": error_details[:20],  # 最初の20個のエラーのみ
        }

    def _parse_error_line(self, line: str, error_type: str) -> dict:
        """エラー行を解析
        
        Args:
            line: MyPy出力の1行
            error_type: エラータイプ
            
        Returns:
            エラー詳細
        """
        parts = line.split(f": {error_type}:")
        if len(parts) >= 2:
            location_part = parts[0].strip()
            message_part = parts[1].strip()
            
            # ファイル名と行番号を抽出
            location_parts = location_part.split(":")
            if len(location_parts) >= 2:
                file_path = location_parts[0]
                try:
                    line_number = int(location_parts[1])
                except ValueError:
                    line_number = 0
            else:
                file_path = location_part
                line_number = 0
            
            return {
                "type": error_type,
                "file": file_path,
                "line": line_number,
                "message": message_part,
            }
        else:
            return {
                "type": error_type,
                "file": "unknown",
                "line": 0,
                "message": line.strip(),
            }

    def _calculate_typing_score(self, errors: int, warnings: int, notes: int) -> float:
        """型チェックスコアを計算
        
        Args:
            errors: エラー数
            warnings: 警告数
            notes: ノート数
            
        Returns:
            0-100の型チェックスコア
        """
        # エラー・警告による減点
        score_reduction = (errors * 5) + (warnings * 2) + (notes * 0.5)
        
        typing_score = max(0, 100 - score_reduction)
        return round(typing_score, 1)

    def _generate_typing_recommendations(
        self, typing_results: dict, gate: QualityGate
    ) -> List[str]:
        """型チェック改善推奨事項を生成
        
        Args:
            typing_results: 型チェック結果
            gate: 品質ゲート定義
            
        Returns:
            推奨事項のリスト
        """
        recommendations = []
        typing_score = typing_results["typing_score"]
        error_count = typing_results.get("error_count", 0)
        warning_count = typing_results.get("warning_count", 0)

        # エラーがある場合
        if "error" in typing_results:
            recommendations.append(f"型チェックエラー: {typing_results['error']}")
            recommendations.append("MyPyの設定を確認してください。")
            return recommendations

        # 型チェックスコアが低い場合
        if typing_score < gate.threshold_acceptable:
            recommendations.extend([
                f"型チェックスコア {typing_score:.1f}% が目標値 {gate.threshold_acceptable}% を下回っています。",
                "型アノテーションを追加して型安全性を向上させてください。",
            ])

        # エラー・警告の詳細
        if error_count > 0:
            recommendations.extend([
                f"型エラー数: {error_count}個",
                "型エラーを段階的に修正してください。",
            ])

        if warning_count > 0:
            recommendations.append(f"型警告数: {warning_count}個")

        # よくあるエラーパターンの対処法
        error_details = typing_results.get("error_details", [])
        if error_details:
            common_issues = self._analyze_common_type_issues(error_details)
            
            for issue_type, count in common_issues.items():
                if issue_type == "missing_return_type":
                    recommendations.append(f"戻り値の型アノテーション不足: {count}個")
                elif issue_type == "missing_param_type":
                    recommendations.append(f"引数の型アノテーション不足: {count}個")
                elif issue_type == "incompatible_types":
                    recommendations.append(f"型の不適合: {count}個")
                elif issue_type == "untyped_function":
                    recommendations.append(f"型なし関数の呼び出し: {count}個")

        # 具体的な改善策
        if typing_score < gate.threshold_good:
            recommendations.extend([
                "型安全性向上のための推奨事項:",
                "・関数の戻り値と引数に型アノテーションを追加",
                "・Union型の使用でNone値を明示",
                "・typing モジュールの活用（List, Dict, Optional等）",
                "・段階的なmypy導入（--ignore-errorsオプション使用）",
            ])

        # MyPy設定の改善提案
        if error_count > 20:
            recommendations.extend([
                "多数のエラーがある場合の対処法:",
                "・mypy.iniファイルで段階的に厳密さを向上",
                "・per-module-optionsで段階的導入",
                "・型スタブファイル(.pyi)の作成検討",
            ])

        # ファイル別の問題集約
        if error_details:
            file_errors = {}
            for detail in error_details[:10]:  # 最初の10件を分析
                file_path = detail.get("file", "unknown")
                if file_path not in file_errors:
                    file_errors[file_path] = 0
                file_errors[file_path] += 1
            
            worst_files = sorted(file_errors.items(), key=lambda x: x[1], reverse=True)[:3]
            if worst_files:
                recommendations.append("型エラーが多いファイル:")
                for file_path, count in worst_files:
                    recommendations.append(f"  {file_path}: {count}個")

        return recommendations[:12]  # 最大12個の推奨事項に制限

    def _analyze_common_type_issues(self, error_details: List[dict]) -> dict:
        """よくある型エラーパターンを分析
        
        Args:
            error_details: エラー詳細リスト
            
        Returns:
            エラーパターンの集計
        """
        issue_patterns = {
            "missing_return_type": 0,
            "missing_param_type": 0,
            "incompatible_types": 0,
            "untyped_function": 0,
            "optional_issues": 0,
            "other": 0,
        }

        for detail in error_details:
            message = detail.get("message", "").lower()
            
            if "missing return type annotation" in message:
                issue_patterns["missing_return_type"] += 1
            elif "missing type annotation" in message and "parameter" in message:
                issue_patterns["missing_param_type"] += 1
            elif "incompatible" in message and "type" in message:
                issue_patterns["incompatible_types"] += 1
            elif "call to untyped function" in message:
                issue_patterns["untyped_function"] += 1
            elif "optional" in message or "none" in message:
                issue_patterns["optional_issues"] += 1
            else:
                issue_patterns["other"] += 1

        return issue_patterns