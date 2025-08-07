import os
import re

# 修正パターン
fixes = [
    # extra={{...}} -> extra={...}
    (r"extra=\{\{([^}]+)\}\}", r"extra={\1}"),
    # 行末の}} -> }
    (r"\}\}(\s*[),])", r"}\1"),
    # {{key -> {"key
    (r"\{\{(\w+)", r'{"\1"'),
    # missing quote fixes
    (r"extra=\{(\w+): ([^}]+)\}", r'extra={"\1": \2}'),
]

files_to_fix = [
    "src/day_trade/automation/optimized_orchestrator.py",
    "src/day_trade/core/trade_operations.py",
    "src/day_trade/core/alerts.py",
    "src/day_trade/core/trade_manager.py",
    "src/day_trade/core/unified_error_handler.py",
    "src/day_trade/models/database.py",
]

fixed_count = 0

for file_path in files_to_fix:
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply all fixes
            for pattern, replacement in fixes:
                content = re.sub(pattern, replacement, content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed syntax: {file_path}")
                fixed_count += 1
            else:
                print(f"No syntax issues: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

print(f"\nFixed syntax in {fixed_count} files")
