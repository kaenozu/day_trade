#!/usr/bin/env python3
"""設定検証ツール"""

import json
from pathlib import Path
from typing import Dict, List


def validate_config(config_file: Path) -> Dict:
    result = {"valid": False, "errors": []}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 必須項目チェック
        required_keys = ["application", "logging", "database"]
        for key in required_keys:
            if key not in config:
                result["errors"].append(f"必須項目が不足: {key}")

        if not result["errors"]:
            result["valid"] = True

    except Exception as e:
        result["errors"].append(f"ファイルエラー: {e}")

    return result


def main():
    config_dir = Path(__file__).parent.parent / "config"
    config_files = list(config_dir.glob("*.json"))

    print("設定ファイル検証結果")
    print("=" * 30)

    for config_file in config_files:
        result = validate_config(config_file)
        status = "✅" if result["valid"] else "❌"
        print(f"{status} {config_file.name}")

        for error in result["errors"]:
            print(f"  エラー: {error}")


if __name__ == "__main__":
    main()
