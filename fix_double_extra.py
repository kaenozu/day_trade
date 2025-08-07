import os
import re

# 二重extra修正パターン: extra={"extra": {...}} -> extra={...}
pattern = r'extra=\{"extra": ([^}]+)\}'

files_to_fix = [
    "src/day_trade/automation/optimized_orchestrator.py",
    "src/day_trade/core/trade_operations.py",
    "src/day_trade/core/alerts.py",
    "src/day_trade/core/trade_manager.py",
    "src/day_trade/core/unified_error_handler.py",
]

fixed_count = 0

for file_path in files_to_fix:
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # 二重extra修正: extra={"extra": {...}} -> extra={...}
            content = re.sub(r'extra=\{"extra": ([^}]+)\}', r"extra={\1}", content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed double extra: {file_path}")
                fixed_count += 1
            else:
                print(f"No double extra found: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

print(f"\nFixed double extra in {fixed_count} files")
