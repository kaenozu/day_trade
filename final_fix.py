import os
import re

# 修正が必要なパターン
pattern = r"extra=\{\{([^}]+)\}\}"

files_to_fix = [
    "src/day_trade/automation/optimized_orchestrator.py",
    "src/day_trade/core/trade_operations.py",
    "src/day_trade/core/trade_manager.py",
]

fixed_count = 0

for file_path in files_to_fix:
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # 二重括弧修正: extra={{...}} -> extra={...}
            content = re.sub(r"extra=\{\{([^}]+)\}\}", r"extra={\1}", content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed: {file_path}")
                fixed_count += 1
            else:
                print(f"No issues: {file_path}")

        except Exception as e:
            print(f"Error: {file_path}: {e}")

print(f"\nFixed {fixed_count} files")
