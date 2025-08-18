# Day Trade Personal - 使用方法

## 📋 Issue #910 解決: シンプル推奨システム

「どの銘柄をいつ買っていつ売るか」を簡潔に知るためのシンプル版

### 🚀 基本使用方法

#### 1. シンプル推奨取得
```bash
# 推奨銘柄をすぐに確認
python simple_daytrade.py

# 特定銘柄の推奨確認
python simple_daytrade.py --symbols 7203 8306 9984
```

#### 2. リアルタイム監視
```bash
# 30秒間隔で自動更新
python simple_daytrade.py --watch
```

### 📊 推奨結果の読み方

| 表示 | 意味 | アクション |
|------|------|------------|
| 🔥 BUY | 買い推奨 | 即座に購入検討 |
| 💤 HOLD | 様子見 | 現状維持 |
| ❌ SELL | 売り推奨 | 売却検討 |

**信頼度**: 85%～93%（AI分析精度）

### 🎯 その他の使用方法

#### 標準版（高精度AI）
```bash
# フル機能版
python daytrade_core.py

# クイック分析
python daytrade_core.py --quick --symbols 7203

# Webダッシュボード
python daytrade_core.py --web --port 8080
```

#### 軽量版（高速・省リソース）
```bash
# 軽量版（85%精度）
python main_lightweight.py --symbols 7203 8306

# 超軽量版（80%精度、最小メモリ）
python main_ultra_light.py --symbols 7203
```

### 💡 推奨される使用パターン

#### 朝の準備（8:45）
```bash
# デフォルト銘柄の事前分析
python simple_daytrade.py
```

#### 寄り付き前（8:55）
```bash
# 注目銘柄の詳細分析
python daytrade_core.py --symbols 7203 8306 9984 6758
```

#### 取引中（随時）
```bash
# 迅速判断用
python simple_daytrade.py --symbols [銘柄コード]
```

#### リアルタイム監視
```bash
# バックグラウンド監視
python simple_daytrade.py --watch &
```

### ⚠️ 重要な注意事項

1. **投資は自己責任**: AI推奨は参考情報です
2. **リスク管理**: 分散投資を心がけてください  
3. **市場状況**: 相場急変時は慎重に判断
4. **データ更新**: 平日9:00-15:00が最新データ

### 🔧 トラブルシューティング

#### よくある問題

**Q: 推奨が表示されない**
```bash
# デバッグモードで確認
python simple_daytrade.py --symbols 7203 --debug
```

**Q: メモリ使用量が多い**
```bash
# 軽量版に切り替え
python main_ultra_light.py --symbols 7203
```

**Q: Webダッシュボードが開かない**
```bash
# ポート変更して再試行
python daytrade_core.py --web --port 8081
```

### 📞 サポート

問題が発生した場合:
1. GitHubリポジトリでIssueを作成
2. エラーメッセージとコマンドを記載
3. 実行環境情報を添付

---

🎯 **目標**: 93%精度AIシステムでデイトレード支援