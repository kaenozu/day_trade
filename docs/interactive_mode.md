# 拡張インタラクティブモード

Day Trade CLIの拡張インタラクティブモードは、オートコンプリート、コマンド履歴、色分け表示などの高度な機能を提供します。

## 機能概要

### 🚀 主要機能

1. **オートコンプリート機能**
   - コマンド名、引数、銘柄コードの自動補完
   - Tabキーでの候補表示
   - あいまい検索対応

2. **コマンド履歴**
   - 過去のコマンドを保存・呼び出し
   - 上下矢印キーでの履歴移動
   - セッション間での履歴永続化

3. **色分けとハイライト**
   - コマンドの構文強調
   - エラー・成功メッセージの色分け
   - 見やすいプロンプト表示

4. **カスタムキーバインディング**
   - `Ctrl+C`: 終了確認
   - `Ctrl+L`: 画面クリア
   - `F1`: ヘルプ表示

5. **コンテキスト対応プロンプト**
   - 現在のモード表示
   - 時刻表示
   - 動的なツールバー

## 使用方法

### 起動方法

#### 1. メインコマンドから起動
```bash
python daytrade.py --interactive
# または
python daytrade.py -i
```

#### 2. 直接モジュールから起動
```bash
python -m src.day_trade.cli.enhanced_interactive
```

#### 3. CLIコマンドから起動
```bash
python -m src.day_trade.cli.interactive enhanced
```

### 必要な依存関係

拡張インタラクティブモードを使用するには、以下の依存関係が必要です：

```bash
pip install prompt_toolkit>=3.0.0
```

依存関係は既に `requirements.txt` と `setup.py` に追加されているため、通常のインストールで自動的に含まれます。

## 利用可能コマンド

### 基本コマンド

| コマンド | 説明 | 例 |
|---------|-----|---|
| `help` | ヘルプ表示 | `help` |
| `stock <コード>` | 個別銘柄情報 | `stock 7203` |
| `history <コード>` | ヒストリカルデータ | `history 7203` |
| `watch <コード...>` | 複数銘柄監視 | `watch 7203 9984` |
| `exit` / `quit` | 終了 | `exit` |

### ウォッチリスト管理

| コマンド | 説明 | 例 |
|---------|-----|---|
| `watchlist add <コード>` | 追加 | `watchlist add 7203` |
| `watchlist remove <コード>` | 削除 | `watchlist remove 7203` |
| `watchlist list` | 一覧表示 | `watchlist list` |
| `watchlist clear` | 全削除 | `watchlist clear` |

### 設定管理

| コマンド | 説明 | 例 |
|---------|-----|---|
| `config show` | 設定表示 | `config show` |
| `config set <キー> <値>` | 設定変更 | `config set api.timeout 30` |
| `config reset` | リセット | `config reset` |

### その他

| コマンド | 説明 | 例 |
|---------|-----|---|
| `validate <コード...>` | 銘柄コード検証 | `validate 7203 9984` |
| `backtest` | バックテスト | `backtest` |

## キーバインディング

| キー | 機能 |
|-----|-----|
| `Tab` | オートコンプリート |
| `↑` / `↓` | コマンド履歴 |
| `Ctrl+C` | 終了確認 |
| `Ctrl+L` | 画面クリア |
| `F1` | ヘルプ表示 |
| `Ctrl+R` | 履歴検索 |

## オートコンプリート対応項目

### 銘柄コード
- 主要日本株の銘柄コード
- ウォッチリストの銘柄コード
- 大文字小文字を区別しない検索

### コマンド階層
```
stock <銘柄コード>
history <銘柄コード>
watch <銘柄コード...>
watchlist
  ├── add <銘柄コード>
  ├── remove <銘柄コード>
  ├── list
  ├── clear
  ├── memo <銘柄コード>
  └── move <銘柄コード>
config
  ├── show
  ├── set
  │   ├── api.timeout
  │   ├── trading.commission
  │   └── display.theme [dark|light]
  └── reset
validate <銘柄コード...>
backtest
```

## 設定ファイル

### 履歴ファイル
コマンド履歴は以下の場所に保存されます：
- Windows: `C:\Users\{username}\.daytrade_history`
- Linux/Mac: `~/.daytrade_history`

### 設定のカスタマイズ
`EnhancedInteractiveCLI` クラスで以下をカスタマイズできます：

```python
# スタイルのカスタマイズ
self.style = Style.from_dict({
    'completion-menu.completion': 'bg:#008888 #ffffff',
    'completion-menu.completion.current': 'bg:#00aaaa #000000',
    'prompt': '#884444 bold',
    'input': '#ffffff',
})

# 銘柄コードリストの拡張
self.stock_codes.extend(['新しい銘柄コード'])
```

## トラブルシューティング

### prompt_toolkitが見つからない
```bash
❌ エラー: 拡張インタラクティブモードは利用できません。
prompt_toolkit>=3.0.0 をインストールしてください。
```

**解決方法:**
```bash
pip install prompt_toolkit>=3.0.0
```

### 履歴ファイルの権限エラー
履歴ファイルの書き込み権限を確認してください：
```bash
# Linux/Mac
chmod 644 ~/.daytrade_history

# Windows
# ファイルのプロパティから権限を確認
```

### オートコンプリートが動作しない
- Tabキーを押してください
- コマンドの階層構造を確認してください
- 大文字小文字は区別されません

## 開発者向け情報

### 新しいコマンドの追加
`enhanced_interactive.py` の `_create_command_completer()` メソッドでコマンド階層を定義：

```python
return NestedCompleter.from_nested_dict({
    'new_command': {
        'subcommand': WordCompleter(['option1', 'option2']),
    }
})
```

### 新しい銘柄コードの追加
`_load_stock_codes()` メソッドで銘柄コードリストを管理：

```python
def _load_stock_codes(self) -> List[str]:
    common_codes = ["7203", "9984", ...]  # 新しいコードを追加
    return common_codes
```

### カスタムキーバインディングの追加
`_setup_key_bindings()` メソッドで新しいキーバインディングを定義：

```python
@self.bindings.add('f2')
def _(event):
    """F2 で新しい機能"""
    self._new_function()
```

## 今後の拡張予定

- [ ] 銘柄コードの動的取得
- [ ] カスタムテーマ機能
- [ ] マクロ機能
- [ ] 多言語対応
- [ ] プラグインシステム
