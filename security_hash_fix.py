#!/usr/bin/env python3
"""
MD5ハッシュセキュリティ修正スクリプト
Issue #419対応
"""

import re
from pathlib import Path
from typing import List, Tuple


def fix_md5_hash_in_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    ファイル内のMD5ハッシュを修正

    Returns:
        (修正実行したか, 修正内容リスト)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        fixes = []

        # パターン1: import hashlib + hashlib.md5(...).hexdigest()
        pattern1 = r"import hashlib\s*\n\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*hashlib\.md5\(([^)]+)\)\.hexdigest\(\)"

        def replace1(match):
            var_name = match.group(1)
            hash_input = match.group(2)
            fixes.append(f"MD5 -> SHA256: {var_name}")
            return f"from ..security.secure_hash_utils import replace_md5_hash\n        {var_name} = replace_md5_hash({hash_input})"

        content = re.sub(pattern1, replace1, content)

        # パターン2: hashlib.md5(...).hexdigest()[:length]
        pattern2 = r"hashlib\.md5\(([^)]+)\)\.hexdigest\(\)\[:(\d+)\]"

        def replace2(match):
            hash_input = match.group(1)
            length = match.group(2)
            fixes.append(f"MD5 -> SHA256 (length={length})")
            return f"replace_md5_hash({hash_input}, length={length})"

        content = re.sub(pattern2, replace2, content)

        # パターン3: hashlib.md5(...).hexdigest()
        pattern3 = r"hashlib\.md5\(([^)]+)\)\.hexdigest\(\)"

        def replace3(match):
            hash_input = match.group(1)
            fixes.append("MD5 -> SHA256")
            return f"replace_md5_hash({hash_input})"

        content = re.sub(pattern3, replace3, content)

        # パターン4: hashlib.sha1(...) (SHA1も弱い)
        pattern4 = r"hashlib\.sha1\(([^)]+)\)"

        def replace4(match):
            hash_input = match.group(1)
            fixes.append("SHA1 -> SHA256")
            # SHA1の場合はhash_objなので、直接置換
            return f"hashlib.sha256({hash_input}, usedforsecurity=False)"

        content = re.sub(pattern4, replace4, content)

        # import文の重複を削除し、secure_hash_utilsのimportを追加
        if fixes and "from ..security.secure_hash_utils import replace_md5_hash" not in content:
            # 既存のimport hashlib行を探して置換または追加
            if "import hashlib" in content:
                content = re.sub(
                    r"(\s+)import hashlib",
                    r"\1from ..security.secure_hash_utils import replace_md5_hash",
                    content,
                    count=1,
                )
            else:
                # 適切な位置にimportを追加
                lines = content.split("\n")
                import_added = False
                for i, line in enumerate(lines):
                    if line.strip().startswith("def ") and not import_added:
                        lines.insert(
                            i,
                            "        from ..security.secure_hash_utils import replace_md5_hash",
                        )
                        import_added = True
                        break
                content = "\n".join(lines)

        # 変更があった場合のみファイル保存
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, fixes
        else:
            return False, []

    except Exception as e:
        print(f"エラー: {file_path} - {e}")
        return False, [f"エラー: {e}"]


def main():
    """主要なファイルのMD5ハッシュ修正"""
    # Banditで検出された高重要度問題のファイルリスト
    target_files = [
        "src/day_trade/api/enhanced_external_api_client.py",
        "src/day_trade/api/external_api_client.py",
        "src/day_trade/cache/key_generator.py",
        "src/day_trade/cache/unified_cache_manager.py",
        "src/day_trade/models/model_validator.py",
        "src/day_trade/optimization/hyperparameter_optimizer.py",
        "src/day_trade/risk_management/factories/risk_analyzer_factory.py",
        "src/day_trade/risk_management/utils/decorators.py",
        "src/day_trade/trading/validation/id_generator.py",
        "src/day_trade/utils/unified_cache_manager.py",
    ]

    base_path = Path(".")
    total_fixes = 0

    print("=== MD5ハッシュセキュリティ修正開始 ===")

    for file_path_str in target_files:
        file_path = base_path / file_path_str
        if file_path.exists():
            print(f"\n修正中: {file_path}")
            modified, fixes = fix_md5_hash_in_file(file_path)
            if modified:
                print(f"  [完了] 修正完了: {len(fixes)}箇所")
                for fix in fixes:
                    print(f"    - {fix}")
                total_fixes += len(fixes)
            else:
                print("  [情報] 修正不要")
        else:
            print(f"  [警告] ファイルなし: {file_path}")

    print(f"\n=== 修正完了: 合計{total_fixes}箇所 ===")


if __name__ == "__main__":
    main()
