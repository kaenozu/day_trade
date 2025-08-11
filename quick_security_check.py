#!/usr/bin/env python3
"""
クイックセキュリティチェック
Issue #419対応
"""

import re
from pathlib import Path


def quick_security_scan():
    """高速セキュリティスキャン"""
    project_root = Path(".")
    issues = []

    # 主要なsrcディレクトリのみスキャン
    python_files = list(project_root.glob("src/**/*.py"))

    sensitive_patterns = {
        "password": r'password\s*=\s*["\'][^"\']{3,}["\']',
        "api_key": r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
        "secret": r'secret\s*=\s*["\'][^"\']{10,}["\']',
        "token": r'token\s*=\s*["\'][^"\']{20,}["\']',
    }

    dangerous_patterns = {
        "eval": r"\beval\s*\(",
        "exec": r"\bexec\s*\(",
        "shell_injection": r"subprocess\.[^(]*\([^)]*shell\s*=\s*True",
    }

    print("=== クイックセキュリティスキャン開始 ===")
    print(f"スキャン対象: {len(python_files)}ファイル")

    for file_path in python_files[:50]:  # 最初の50ファイルのみ
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # 機密情報チェック
            for pattern_name, pattern in sensitive_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    issues.append(f"[高] {file_path}: 機密情報({pattern_name})検出")

            # 危険な関数チェック
            for pattern_name, pattern in dangerous_patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    issues.append(f"[高] {file_path}: 危険な関数({pattern_name})検出")

        except (UnicodeDecodeError, PermissionError):
            continue

    print("\n=== スキャン結果 ===")
    if issues:
        print(f"検出された問題: {len(issues)}件")
        for issue in issues[:10]:  # 最初の10件のみ表示
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... 他 {len(issues) - 10} 件")
    else:
        print("セキュリティ問題は検出されませんでした。")

    return len(issues)


if __name__ == "__main__":
    issue_count = quick_security_scan()
    print(f"\n総合評価: {'要改善' if issue_count > 0 else '良好'}")
