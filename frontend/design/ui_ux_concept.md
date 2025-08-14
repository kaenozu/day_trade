# UI/UX Design Concept
# Day Trade ML System - Issue #803

## 🎨 デザインフィロソフィー

### コアコンセプト
**"データドリブン・インテリジェント・トレーディング"**

1. **可視性優先**: 重要な情報を即座に把握可能
2. **アクション指向**: 意思決定から実行まで最短経路
3. **レスポンシブ**: デスクトップ・タブレット・モバイル完全対応
4. **リアルタイム**: 市場変動・システム状態の即時反映

### ターゲットユーザー

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   トレーダー    │    │  システム管理者  │    │   経営陣        │
│                 │    │                 │    │                 │
│ • 取引実行      │    │ • 監視・運用    │    │ • KPI監視       │
│ • 市場分析      │    │ • 設定管理      │    │ • ROI分析       │
│ • ポートフォリオ│    │ • トラブル対応  │    │ • 戦略決定      │
│ • リスク管理    │    │ • パフォーマンス│    │ • レポート      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
            ┌───────────────────────────────────────┐
            │          統合ダッシュボード           │
            │                                       │
            │ • 役割別カスタマイズ                  │
            │ • 権限ベースアクセス制御              │
            │ • パーソナライゼーション              │
            └───────────────────────────────────────┘
```

## 🎯 UI構成設計

### メインダッシュボードレイアウト

```
┌──────────────────────────────────────────────────────────────────┐
│ ヘッダー: ロゴ | ナビゲーション | ユーザーメニュー | アラート通知   │
├──────────────────────────────────────────────────────────────────┤
│ ┌────────────┐ ┌─────────────────────────────────────────────────┐ │
│ │サイドバー  │ │             メインコンテンツエリア              │ │
│ │            │ │                                                 │ │
│ │• ダッシュ  │ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │
│ │  ボード    │ │ │ KPI概要     │ │ 市場状況    │ │ ML精度      │ │ │
│ │• 取引      │ │ │ カード      │ │ チャート    │ │ ゲージ      │ │ │
│ │• 分析      │ │ └─────────────┘ └─────────────┘ └─────────────┘ │ │
│ │• 監視      │ │                                                 │ │
│ │• 設定      │ │ ┌─────────────────────────────────────────────────┐ │ │
│ │• レポート  │ │ │           リアルタイム取引フィード              │ │ │
│ │            │ │ │ 時刻 | 銘柄 | アクション | 価格 | 利益 | 状態  │ │ │
│ └────────────┘ │ └─────────────────────────────────────────────────┘ │ │
├──────────────────────────────────────────────────────────────────┤
│ フッター: ステータス | 接続状態 | システム情報 | バージョン        │
└──────────────────────────────────────────────────────────────────┘
```

### カラースキーム

```css
/* プライマリカラー */
--primary-blue: #2563eb      /* アクション・リンク */
--primary-dark: #1e40af      /* ヘッダー・重要要素 */
--primary-light: #dbeafe     /* 背景・ハイライト */

/* セカンダリカラー */
--success-green: #10b981     /* 利益・成功状態 */
--warning-amber: #f59e0b     /* 警告・注意 */
--danger-red: #ef4444        /* 損失・エラー */
--info-cyan: #06b6d4         /* 情報・通知 */

/* ニュートラル */
--gray-50: #f9fafb          /* 背景 */
--gray-100: #f3f4f6         /* カード背景 */
--gray-900: #111827         /* テキスト */
--white: #ffffff            /* メイン背景 */

/* グラデーション */
--gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%)
--gradient-success: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)
--gradient-warning: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)
```

### タイポグラフィ

```css
/* フォントファミリー */
--font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif
--font-mono: 'JetBrains Mono', 'Monaco', 'Courier New', monospace
--font-jp: 'Noto Sans JP', sans-serif

/* フォントサイズ */
--text-xs: 0.75rem     /* 12px - ラベル・キャプション */
--text-sm: 0.875rem    /* 14px - 補助テキスト */
--text-base: 1rem      /* 16px - 本文 */
--text-lg: 1.125rem    /* 18px - 小見出し */
--text-xl: 1.25rem     /* 20px - 見出し */
--text-2xl: 1.5rem     /* 24px - 大見出し */
--text-3xl: 1.875rem   /* 30px - タイトル */
--text-4xl: 2.25rem    /* 36px - 数値表示 */
```

## 📱 レスポンシブデザイン

### ブレイクポイント

```css
/* モバイル（デフォルト） */
@media (min-width: 320px) {
  /* スマートフォン縦向き */
  .container { max-width: 100%; }
  .sidebar { transform: translateX(-100%); } /* 隠す */
  .main-content { padding: 1rem; }
}

/* タブレット */
@media (min-width: 768px) {
  /* タブレット・スマートフォン横向き */
  .container { max-width: 768px; }
  .sidebar { position: relative; transform: none; }
  .grid-cols-responsive { grid-template-columns: repeat(2, 1fr); }
}

/* デスクトップ */
@media (min-width: 1024px) {
  /* ノートPC・デスクトップ */
  .container { max-width: 1200px; }
  .sidebar { width: 240px; }
  .grid-cols-responsive { grid-template-columns: repeat(3, 1fr); }
}

/* 大画面 */
@media (min-width: 1440px) {
  /* 大型モニター */
  .container { max-width: 1400px; }
  .sidebar { width: 280px; }
  .grid-cols-responsive { grid-template-columns: repeat(4, 1fr); }
}
```

### モバイル最適化

```
┌─────────────────────┐
│ ☰ Day Trade ML     │ ← ハンバーガーメニュー
├─────────────────────┤
│ ┌─────────────────┐ │
│ │ 現在のROI       │ │
│ │    +5.2%        │ │ ← 重要KPIを上部配置
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ 取引状況        │ │
│ │ 実行中: 3件     │ │
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ 📊 詳細分析     │ │ ← タップで展開
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ 🔔 アラート     │ │
│ └─────────────────┘ │
├─────────────────────┤
│ [取引] [分析] [設定] │ ← タブナビゲーション
└─────────────────────┘
```

## 🔄 インタラクションデザイン

### アニメーション原則

1. **目的性**: 意味のあるアニメーション
2. **快適性**: 60fps滑らか動作
3. **効率性**: 300ms以内完了
4. **アクセシビリティ**: 減速オプション対応

```css
/* トランジション設定 */
.transition-smooth {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.transition-bounce {
  transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

/* ローディング状態 */
.loading-skeleton {
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
}

@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### ユーザーフィードバック

```typescript
// 状態フィードバック
interface FeedbackState {
  loading: boolean;
  success: boolean;
  error: string | null;
  progress?: number;
}

// 成功フィードバック
const showSuccess = (message: string) => {
  toast.success(message, {
    icon: '✅',
    duration: 3000,
    position: 'top-right'
  });
};

// エラーフィードバック
const showError = (error: string) => {
  toast.error(error, {
    icon: '❌',
    duration: 5000,
    position: 'top-right',
    action: {
      label: '再試行',
      onClick: () => retryAction()
    }
  });
};
```

## 📊 コンポーネント設計

### 原子コンポーネント（Atoms）

```typescript
// Button Component
interface ButtonProps {
  variant: 'primary' | 'secondary' | 'success' | 'danger';
  size: 'xs' | 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  icon?: ReactNode;
  children: ReactNode;
  onClick?: () => void;
}

// Card Component
interface CardProps {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}

// Chart Component
interface ChartProps {
  type: 'line' | 'bar' | 'pie' | 'gauge';
  data: ChartData;
  options?: ChartOptions;
  height?: number;
  realtime?: boolean;
}
```

### 分子コンポーネント（Molecules）

```typescript
// KPI Card
interface KPICardProps {
  title: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'flat';
  change?: number;
  status: 'good' | 'warning' | 'critical';
}

// Trade Entry
interface TradeEntryProps {
  symbol: string;
  action: 'buy' | 'sell';
  price: number;
  quantity: number;
  timestamp: Date;
  status: 'pending' | 'executed' | 'failed';
  profit?: number;
}

// Alert Banner
interface AlertBannerProps {
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  dismissible?: boolean;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}
```

### 生体コンポーネント（Organisms）

```typescript
// Trading Dashboard
interface TradingDashboardProps {
  kpis: KPIData[];
  trades: TradeData[];
  charts: ChartConfig[];
  alerts: AlertData[];
  user: UserData;
}

// Market Analysis Panel
interface MarketAnalysPanelProps {
  symbols: SymbolData[];
  technicalIndicators: IndicatorData[];
  predictions: PredictionData[];
  recommendations: RecommendationData[];
}

// System Monitoring Panel
interface SystemMonitoringPanelProps {
  services: ServiceStatus[];
  metrics: SystemMetrics;
  alerts: SystemAlert[];
  logs: LogEntry[];
}
```

## 🎛️ ダッシュボード詳細設計

### メインダッシュボード

```
┌─────────────────────────────────────────────────────────────┐
│ 📊 概要セクション                                           │
├─────────────────────────────────────────────────────────────┤
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│ │ 今日のROI  │ │ 取引実行数 │ │ ML予測精度 │ │ ポートフォリオ│ │
│ │   +5.2%    │ │    47件    │ │   93.4%    │ │  $125,430  │ │
│ │ ↗️ +1.2%   │ │ ↗️ +8件    │ │ ↗️ +0.4%   │ │ ↗️ +$3,240 │ │
│ └────────────┘ └────────────┘ └────────────┘ └────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ 📈 市場分析セクション                                       │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐ ┌───────────────────────────────┐ │
│ │ リアルタイム価格チャート │ │ AI予測 vs 実績比較            │ │
│ │                         │ │                               │ │
│ │ ▲ AAPL  $175.23        │ │ 精度: 93.4% (目標: 93%)       │ │
│ │ ▼ TSLA  $234.56        │ │ 信頼度: 89.2%                │ │
│ │ ▲ MSFT  $342.78        │ │ 改善提案: 1件                │ │
│ └─────────────────────────┘ └───────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ 🔄 取引実行セクション                                       │
├─────────────────────────────────────────────────────────────┤
│ 時刻     | 銘柄 | アクション | 価格    | 数量 | 利益    | 状態 │
│ 14:32:15 | AAPL | 売却       | $175.20 | 100  | +$240  | 完了 │
│ 14:28:03 | TSLA | 購入       | $234.50 | 50   | -      | 実行 │
│ 14:25:47 | MSFT | 売却       | $342.80 | 75   | +$180  | 完了 │
└─────────────────────────────────────────────────────────────┘
```

### トレーディングビュー

```
┌─────────────────────────────────────────────────────────────┐
│ 🎯 取引実行パネル                                           │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │ 推奨取引        │ │ リスク分析      │ │ 実行オプション  │ │
│ │                 │ │                 │ │                 │ │
│ │ 🔥 AAPL 売却    │ │ リスクスコア:   │ │ □ 自動実行      │ │
│ │ 信頼度: 92%     │ │ 3.2/10 (低)     │ │ □ 確認後実行    │ │
│ │ 予想利益: $320  │ │                 │ │ [実行] [保留]   │ │
│ │                 │ │ ストップロス:   │ │                 │ │
│ │ ⚡ TSLA 購入    │ │ $230.00         │ │                 │ │
│ │ 信頼度: 89%     │ │                 │ │                 │ │
│ │ 予想利益: $180  │ │ 最大損失:       │ │                 │ │
│ └─────────────────┘ │ $450            │ └─────────────────┘ │
│                     └─────────────────┘                     │
├─────────────────────────────────────────────────────────────┤
│ 📊 ポートフォリオ分析                                       │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐ ┌───────────────────────────────┐ │
│ │ 資産配分                │ │ パフォーマンス履歴            │ │
│ │ 🔵 テクノロジー 45%     │ │                               │ │
│ │ 🟢 ヘルスケア   25%     │ │ 📈 1ヶ月: +12.3%              │ │
│ │ 🟡 金融        20%     │ │ 📈 3ヶ月: +28.7%              │ │
│ │ 🔴 エネルギー   10%     │ │ 📈 1年:   +156.2%             │ │
│ └─────────────────────────┘ └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔔 通知・アラートシステム

### 通知レベル設計

```typescript
enum NotificationLevel {
  INFO = 'info',        // 情報：新しい推奨など
  WARNING = 'warning',  // 警告：リスク上昇など
  ERROR = 'error',      // エラー：取引失敗など
  CRITICAL = 'critical' // 重要：システム障害など
}

interface Notification {
  id: string;
  level: NotificationLevel;
  title: string;
  message: string;
  timestamp: Date;
  actions?: NotificationAction[];
  autoClose?: number;
  persistent?: boolean;
}
```

### 通知表示方式

1. **Toast通知**: 右上角・自動消去
2. **バナー通知**: 画面上部・重要情報
3. **モーダル通知**: 中央表示・確認必須
4. **プッシュ通知**: ブラウザ・モバイル

## 📱 モバイルアプリ設計

### ネイティブ機能

```
┌─────────────────────┐
│ 📱 モバイル機能     │
├─────────────────────┤
│ 🔔 プッシュ通知     │
│ • 重要アラート      │
│ • 取引完了通知      │
│ • 市場変動警告      │
├─────────────────────┤
│ 📍 位置情報         │
│ • 取引所営業時間    │
│ • 地域別税制対応    │
├─────────────────────┤
│ 🔐 生体認証         │
│ • 指紋認証          │
│ • 顔認証            │
│ • Touch ID/Face ID  │
├─────────────────────┤
│ 📶 オフライン機能   │
│ • 最新データ保存    │
│ • 読み取り専用表示  │
│ • 同期待ちキュー    │
└─────────────────────┘
```

### PWA対応

```typescript
// Service Worker
self.addEventListener('push', (event) => {
  const options = {
    body: event.data.text(),
    icon: '/icons/notification-icon.png',
    badge: '/icons/badge-icon.png',
    vibrate: [200, 100, 200],
    data: {
      url: '/dashboard'
    },
    actions: [
      {
        action: 'view',
        title: '確認',
        icon: '/icons/view-icon.png'
      },
      {
        action: 'dismiss',
        title: '閉じる',
        icon: '/icons/close-icon.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('Day Trade ML', options)
  );
});
```

## ♿ アクセシビリティ

### WCAG 2.1 AA準拠

```css
/* 高コントラスト対応 */
@media (prefers-contrast: high) {
  :root {
    --text-color: #000000;
    --bg-color: #ffffff;
    --border-color: #000000;
  }
}

/* 動き削減対応 */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* フォーカス表示 */
.focus-visible {
  outline: 2px solid var(--primary-blue);
  outline-offset: 2px;
}
```

### スクリーンリーダー対応

```typescript
// ARIA属性適用
const TradingCard = ({ trade }: { trade: TradeData }) => (
  <div
    role="article"
    aria-labelledby={`trade-${trade.id}-title`}
    aria-describedby={`trade-${trade.id}-details`}
  >
    <h3 id={`trade-${trade.id}-title`}>
      {trade.symbol} {trade.action}
    </h3>
    <div id={`trade-${trade.id}-details`}>
      価格: {trade.price}円
      数量: {trade.quantity}株
      <span aria-label={`利益${trade.profit > 0 ? '獲得' : '損失'}`}>
        利益: {trade.profit}円
      </span>
    </div>
  </div>
);
```

---

このUI/UX設計コンセプトを基に、次にReactフロントエンド構築を開始します。