# DB操作のトランザクション管理

## 概要

このドキュメントでは、アプリケーションでのデータベーストランザクション管理の強化について説明します。複数のDB操作を安全にアトミックに実行し、データ整合性を保証するための機能を提供します。

## 実装された機能

### 1. 明示的なトランザクション管理

`DatabaseManager.transaction_scope()` メソッドにより、より厳密なトランザクション制御を提供：

```python
from src.day_trade.models.database import db_manager

# 明示的なトランザクション制御
with db_manager.transaction_scope() as session:
    # 複数のDB操作
    session.add(obj1)
    session.add(obj2)
    session.flush()  # 中間状態を確認
    # さらなる処理...
    # コミットは自動実行
```

### 2. デッドロック対応と再試行機能

デッドロックやロック競合に対する自動再試行機能：

```python
# 再試行回数と待機時間を指定
with db_manager.transaction_scope(retry_count=5, retry_delay=0.2) as session:
    # デッドロックが発生しやすい処理
    pass
```

### 3. アトミック操作

複数の関連する操作をまとめて実行：

```python
def operation1(session):
    session.add(Stock(code="001", name="Stock 1"))

def operation2(session):
    session.add(Stock(code="002", name="Stock 2"))

# 両方の操作が成功するか、すべて失敗する
db_manager.atomic_operation([operation1, operation2])
```

### 4. 取引操作での実践例

`TradeOperations`クラスでの実装例：

```python
from src.day_trade.core.trade_operations import TradeOperations

trade_ops = TradeOperations()

# 買い注文 - 銘柄マスタ確認・追加、取引記録、ポートフォリオ更新を一括実行
result = trade_ops.buy_stock("7203", 100, price=2500.0)

# 売り注文 - 保有確認、取引記録、ポートフォリオ更新を一括実行
result = trade_ops.sell_stock("7203", 50, price=2600.0)
```

## 機能詳細

### transaction_scope の特徴

1. **自動コミット・ロールバック**: エラー発生時は自動的にロールバック
2. **再試行機能**: デッドロックなど再試行可能エラーに対する自動再試行
3. **指数バックオフ**: 再試行間隔の調整
4. **ログ出力**: エラーと再試行の詳細ログ

### 再試行対象エラー

以下のエラーパターンで自動再試行：
- deadlock
- lock timeout
- database is locked
- connection was dropped
- connection pool

### エラーハンドリング

すべてのDB操作エラーは適切な例外型に変換され、詳細なログと共に出力されます。

## 使用上の注意点

### 1. 適切なトランザクションスコープの選択

- **単一操作**: `session_scope()` を使用
- **複数関連操作**: `transaction_scope()` を使用
- **大量データ**: `bulk_insert()`, `bulk_update()` を使用

### 2. デッドロック回避のベストプラクティス

- 一貫したテーブルアクセス順序
- 長時間実行されるトランザクションの分割
- 必要最小限のロック範囲

### 3. パフォーマンス考慮事項

- `flush()` の適切な使用（中間確認が必要な場合のみ）
- バッチサイズの調整
- インデックスの活用

## テスト

トランザクション管理機能は `tests/test_transaction_management.py` で包括的にテスト：

```bash
# テスト実行
python -m pytest tests/test_transaction_management.py -v
```

## 移行ガイド

### 既存コードの更新

既存のコードで複数のDB操作を行っている箇所を特定し、`transaction_scope()` に移行：

**Before:**
```python
with db_manager.session_scope() as session:
    session.add(obj1)

with db_manager.session_scope() as session:
    session.add(obj2)
```

**After:**
```python
with db_manager.transaction_scope() as session:
    session.add(obj1)
    session.add(obj2)
    # 両方が成功するか、両方ともロールバック
```

### 推奨する移行手順

1. 関連するDB操作を特定
2. トランザクション境界を設計
3. `transaction_scope()` に移行
4. テストで動作確認
5. 段階的にデプロイ

## 監視とデバッグ

### ログ出力

トランザクション管理では以下の情報をログ出力：

- トランザクション開始・終了
- 再試行の詳細（回数、理由、待機時間）
- エラーの詳細とスタックトレース

### メトリクス

以下のメトリクスを監視推奨：

- トランザクション成功率
- 再試行発生頻度
- 平均実行時間
- デッドロック発生率

## TradeManagerの改善（Issue #187対応）

### データベース永続化対応

TradeManagerクラスが完全なデータベース永続化に対応し、メモリ内操作とDB操作の一貫性を保証：

```python
from src.day_trade.core.trade_manager import TradeManager, TradeType
from decimal import Decimal

# データベース永続化対応のTradeManager
trade_manager = TradeManager(
    commission_rate=Decimal("0.001"),
    tax_rate=Decimal("0.2"),
    load_from_db=True  # 既存取引を読み込み
)

# 取引を追加（データベースに永続化）
trade_id = trade_manager.add_trade(
    symbol="7203",
    trade_type=TradeType.BUY,
    quantity=100,
    price=Decimal("2500"),
    notes="テスト取引",
    persist_to_db=True  # データベースに永続化
)
```

### アトミック操作の実装

TradeManager内での複数操作が完全にアトミック：

1. **銘柄マスタの存在確認・作成**
2. **データベース取引記録の作成**
3. **メモリ内データ構造の更新**

すべてが成功するか、エラー時は全体がロールバック。

### 新機能

1. **データベースからの読み込み**
   ```python
   trade_manager = TradeManager(load_from_db=True)
   ```

2. **データベース同期**
   ```python
   trade_manager.sync_with_db()
   ```

3. **構造化ログ統合**
   - コンテキスト情報付きログ
   - ビジネスイベントログ
   - パフォーマンスメトリクス

### テストカバレッジ

新しいテストスイート `test_trade_manager_transactions.py`:

- データベース永続化テスト
- トランザクションロールバックテスト
- アトミック操作テスト
- 並行処理テスト
- データベース同期テスト

```bash
# TradeManagerトランザクションテストを実行
python -m pytest tests/test_trade_manager_transactions.py -v
```

## 今後の拡張予定

1. **分散トランザクション**: 複数データベース間での整合性保証
2. **読み取り専用トランザクション**: パフォーマンス最適化
3. **トランザクション監視**: リアルタイムでの状態監視
4. **自動チューニング**: ワークロードに応じた設定最適化
5. **バッチ処理機能**: TradeManagerでの大量取引一括処理
