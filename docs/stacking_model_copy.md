# StackingEnsemble Model Copy Mechanism

Issue #483対応: `_copy_model`の堅牢性改善

## 概要

StackingEnsembleでは、交差検証時にベースモデルを独立した状態でコピーする必要があります。
従来の単純な実装から、学習済み状態や内部状態を適切に扱う堅牢なコピーメカニズムに改善しました。

## 実装された機能

### 1. BaseModelInterface.copy()メソッド

すべてのベースモデルに標準的な`copy()`メソッドを提供：

```python
def copy(self) -> 'BaseModelInterface':
    """モデルの安全なコピー作成"""
    # 未学習状態の新しいインスタンス作成
    # 設定の深いコピー
    # 特徴量名などの基本属性保持
```

**特徴**：
- 未学習状態でコピーを作成（デフォルト動作）
- 設定の深いコピーで独立性確保
- 継承クラスでのオーバーライド可能

### 2. StackingEnsemble._copy_model()メソッド強化

3段階のフォールバック戦略を実装：

#### 段階1: カスタムcopyメソッド使用
```python
if hasattr(model, 'copy') and callable(model.copy):
    return model.copy()
```

#### 段階2: scikit-learn cloneメソッド使用
```python
from sklearn.base import clone
cloned_sklearn_model = clone(model.model)
```

#### 段階3: 基本コピー（フォールバック）
```python
model_class = model.__class__
new_model = model_class(model.config)
```

## 堅牢性の向上

### エラーハンドリング
- 各段階でのエラー捕捉
- 適切なフォールバック処理
- 詳細なログ出力

### 状態管理
- 学習済み状態の適切な初期化
- 内部メトリクスのクリア
- 特徴量名の保持

### 設定の独立性
- 深いコピーによる設定の独立
- ネストした辞書構造の適切なコピー

## 使用例

### カスタムモデルでのcopyメソッド実装

```python
class CustomModel(BaseModelInterface):
    def copy(self):
        # モデル固有のコピーロジック
        new_model = CustomModel(self.model_name, self.config)
        new_model.special_attribute = self.special_attribute.copy()
        return new_model
```

### StackingEnsembleでの自動使用

```python
base_models = {
    "model1": CustomModel("model1"),
    "model2": ScikitLearnModel("model2")
}

ensemble = StackingEnsemble(base_models, config)
# CV学習時に自動的に適切なコピー方法が選択される
ensemble.fit(X, y)
```

## テストカバレッジ

包括的なテストスイート（`test_copy_model.py`）を実装：

- カスタムcopyメソッドテスト
- scikit-learn cloneテスト
- フォールバック基本コピーテスト
- エラーハンドリングテスト
- 深いコピーテスト
- 統合テスト

## パフォーマンス考慮

- 軽量なコピー作成
- メモリ効率的な実装
- 不要な状態のコピー回避

## 互換性

- 既存のベースモデルとの後方互換性維持
- scikit-learn互換性の自動検出
- 段階的フォールバック

この実装により、StackingEnsembleのモデルコピー機能が大幅に改善され、
より堅牢で信頼性の高いアンサンブル学習が可能になりました。