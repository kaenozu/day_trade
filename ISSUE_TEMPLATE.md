# Issue: daytrade_core.py の改善と CLI 統合

## 問題点

`daytrade_core.py` はアプリケーションのエントリーポイントとして機能していますが、以下の改善点があります。

### 1. CLIとの一貫性
- `daytrade_cli.py` と `daytrade_core.py` の機能が重複しており、混乱を招く可能性があります。
- CLIのオプションも両方のファイルで定義されており、メンテナンス性が低下しています。

### 2. エラーハンドリング
- `main()` 関数で例外が発生した場合の処理が不十分です。
- より詳細なエラーハンドリングを追加する必要があります。

### 3. ドキュメント
- ファイルヘッダーのコメントが日本語のみで、英語話者向けのドキュメントがありません。

## 改善提案

### 1. CLI統合
- `daytrade_cli.py` と `daytrade_core.py` の機能を統合し、単一のエントリーポイントにするべきです。

### 2. エラーハンドリングの改善
- `main()` 関数に詳細なエラーハンドリングを追加します。

### 3. ドキュメントの改善
- ファイルヘッダーに英語のコメントを追加します。

## 修正案

```python
#!/usr/bin/env python3
"""
Day Trade Personal - Core Entry Point

Refactored lightweight main file
"""

import sys
from pathlib import Path

# Add system path
sys.path.append(str(Path(__file__).parent / "src"))

from src.day_trade.core.application import DayTradeApplication
from src.day_trade.core.lightweight_application import LightweightDayTradeApplication

# Alias definitions
DayTradeCore = DayTradeApplication
DayTradeCoreLight = LightweightDayTradeApplication


def main():
    """Main execution function"""
    try:
        app = DayTradeApplication()
        return app.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## その他の提案

1. **設定管理**: 設定ファイルのパスを環境変数で指定できるようにすると、柔軟性が向上します。
2. **ロギング**: アプリケーション全体で一貫したロギング戦略を採用することを検討してください。