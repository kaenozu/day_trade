#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customizable Dashboard - カスタマイズ可能ダッシュボード
Issue #949対応: 柔軟なレイアウト + ウィジェット管理 + 個人設定
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging

# 統合モジュール
try:
    from advanced_ai_engine import advanced_ai_engine
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False

try:
    from version import get_version_info
    VERSION_INFO = get_version_info()
except ImportError:
    VERSION_INFO = {"version": "2.1.0", "release_name": "Extended"}


class WidgetType(Enum):
    """ウィジェットタイプ"""
    STOCK_CHART = "STOCK_CHART"
    AI_RECOMMENDATIONS = "AI_RECOMMENDATIONS"
    MARKET_NEWS = "MARKET_NEWS"
    PERFORMANCE_METRICS = "PERFORMANCE_METRICS"
    QUICK_STATS = "QUICK_STATS"
    WATCHLIST = "WATCHLIST"
    ALERTS = "ALERTS"
    CALENDAR = "CALENDAR"
    CUSTOM_TEXT = "CUSTOM_TEXT"
    SYSTEM_STATUS = "SYSTEM_STATUS"


class WidgetSize(Enum):
    """ウィジェットサイズ"""
    SMALL = "1x1"      # 1列1行
    MEDIUM = "2x1"     # 2列1行
    LARGE = "2x2"      # 2列2行
    WIDE = "3x1"       # 3列1行
    TALL = "1x2"       # 1列2行
    EXTRA_LARGE = "3x2"  # 3列2行


@dataclass
class WidgetPosition:
    """ウィジェット位置"""
    x: int  # 列位置（0から開始）
    y: int  # 行位置（0から開始）
    w: int  # 幅（列数）
    h: int  # 高さ（行数）


@dataclass
class WidgetConfig:
    """ウィジェット設定"""
    widget_id: str
    widget_type: WidgetType
    title: str
    position: WidgetPosition
    visible: bool = True
    refresh_interval: int = 30  # 秒
    settings: Dict[str, Any] = None


@dataclass
class DashboardLayout:
    """ダッシュボードレイアウト"""
    layout_id: str
    name: str
    description: str
    grid_columns: int = 12  # グリッド列数
    widgets: List[WidgetConfig] = None
    theme: str = "default"
    created_at: datetime = None
    modified_at: datetime = None


class BaseWidget(ABC):
    """基底ウィジェットクラス"""

    def __init__(self, config: WidgetConfig):
        self.config = config
        self.last_update = None
        self.cache_duration = timedelta(seconds=config.refresh_interval)
        self._cached_data = None

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """ウィジェットデータ取得"""
        pass

    @abstractmethod
    def get_html_template(self) -> str:
        """HTMLテンプレート取得"""
        pass

    def should_refresh(self) -> bool:
        """リフレッシュが必要かチェック"""
        if self.last_update is None:
            return True
        return datetime.now() - self.last_update > self.cache_duration

    def get_cached_data(self) -> Dict[str, Any]:
        """キャッシュデータ取得"""
        if self.should_refresh():
            self._cached_data = self.get_data()
            self.last_update = datetime.now()
        return self._cached_data


class StockChartWidget(BaseWidget):
    """株価チャートウィジェット"""

    def get_data(self) -> Dict[str, Any]:
        """株価データ取得"""
        symbol = self.config.settings.get('symbol', '7203')
        period = self.config.settings.get('period', '1d')

        # 模擬データ生成
        import random
        times = []
        prices = []
        base_price = 1500 + hash(symbol) % 500

        for i in range(24):  # 24時間分
            times.append(f"{i:02d}:00")
            price = base_price + random.uniform(-50, 50)
            prices.append(round(price, 2))

        return {
            'symbol': symbol,
            'prices': prices,
            'times': times,
            'current_price': prices[-1] if prices else 0,
            'change': random.uniform(-3.0, 3.0),
            'volume': random.randint(1000000, 5000000)
        }

    def get_html_template(self) -> str:
        """HTMLテンプレート"""
        return f"""
        <div class="widget stock-chart h-full" id="widget-{self.config.widget_id}">
            <div class="widget-header">
                <h3 class="text-lg font-semibold">{self.config.title}</h3>
                <div class="widget-controls">
                    <button class="refresh-btn" onclick="refreshWidget('{self.config.widget_id}')">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                </div>
            </div>
            <div class="widget-content">
                <div class="stock-info mb-4">
                    <div class="symbol text-xl font-bold" x-text="data.symbol"></div>
                    <div class="price-info">
                        <span class="price text-2xl" x-text="'¥' + data.current_price.toLocaleString()"></span>
                        <span class="change ml-2 px-2 py-1 rounded text-sm"
                              :class="data.change >= 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'"
                              x-text="(data.change >= 0 ? '+' : '') + data.change.toFixed(2) + '%'"></span>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-{self.config.widget_id}" width="100%" height="200"></canvas>
                </div>
            </div>
        </div>
        """


class AIRecommendationsWidget(BaseWidget):
    """AI推奨銘柄ウィジェット"""

    def get_data(self) -> Dict[str, Any]:
        """AI推奨データ取得"""
        if HAS_AI_ENGINE:
            try:
                # 実際のAI分析データ取得
                recommendations = []
                symbols = self.config.settings.get('symbols', ['7203', '8306', '9984'])

                for symbol in symbols[:5]:  # 最大5件
                    signal = advanced_ai_engine.analyze_symbol(symbol)
                    recommendations.append({
                        'symbol': symbol,
                        'signal_type': signal.signal_type,
                        'confidence': signal.confidence,
                        'strength': signal.strength,
                        'risk_level': signal.risk_level
                    })

                return {'recommendations': recommendations}
            except Exception as e:
                logging.error(f"AI recommendations error: {e}")

        # フォールバック：模擬データ
        import random
        recommendations = []
        symbols = ['7203', '8306', '9984', '6758', '4689']

        for symbol in symbols[:3]:
            recommendations.append({
                'symbol': symbol,
                'signal_type': random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': random.uniform(0.6, 0.95),
                'strength': random.uniform(0.4, 0.9),
                'risk_level': random.choice(['LOW', 'MEDIUM', 'HIGH'])
            })

        return {'recommendations': recommendations}

    def get_html_template(self) -> str:
        """HTMLテンプレート"""
        return f"""
        <div class="widget ai-recommendations h-full" id="widget-{self.config.widget_id}">
            <div class="widget-header">
                <h3 class="text-lg font-semibold flex items-center">
                    <i class="fas fa-robot text-purple-600 mr-2"></i>
                    {self.config.title}
                </h3>
            </div>
            <div class="widget-content overflow-y-auto">
                <template x-for="rec in data.recommendations" :key="rec.symbol">
                    <div class="recommendation-item p-3 mb-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                        <div class="flex items-center justify-between mb-2">
                            <span class="font-bold text-lg" x-text="rec.symbol"></span>
                            <span class="px-2 py-1 rounded text-sm font-semibold"
                                  :class="getSignalClass(rec.signal_type)"
                                  x-text="rec.signal_type"></span>
                        </div>
                        <div class="flex items-center justify-between text-sm">
                            <div>
                                <span class="text-gray-600">信頼度: </span>
                                <span class="font-medium" x-text="(rec.confidence * 100).toFixed(1) + '%'"></span>
                            </div>
                            <div>
                                <span class="text-gray-600">リスク: </span>
                                <span class="px-2 py-1 rounded text-xs"
                                      :class="getRiskClass(rec.risk_level)"
                                      x-text="rec.risk_level"></span>
                            </div>
                        </div>
                    </div>
                </template>
            </div>
        </div>
        """


class PerformanceMetricsWidget(BaseWidget):
    """パフォーマンスメトリクスウィジェット"""

    def get_data(self) -> Dict[str, Any]:
        """パフォーマンスデータ取得"""
        if HAS_PERFORMANCE_MONITOR:
            try:
                return performance_monitor.get_performance_summary()
            except Exception as e:
                logging.error(f"Performance metrics error: {e}")

        # フォールバック：模擬データ
        import random
        return {
            'cpu_usage': random.uniform(20, 80),
            'memory_usage': random.uniform(30, 70),
            'disk_usage': random.uniform(40, 60),
            'response_time_ms': random.uniform(100, 500),
            'active_connections': random.randint(10, 100),
            'requests_per_minute': random.randint(50, 200)
        }

    def get_html_template(self) -> str:
        """HTMLテンプレート"""
        return f"""
        <div class="widget performance-metrics h-full" id="widget-{self.config.widget_id}">
            <div class="widget-header">
                <h3 class="text-lg font-semibold flex items-center">
                    <i class="fas fa-tachometer-alt text-blue-600 mr-2"></i>
                    {self.config.title}
                </h3>
            </div>
            <div class="widget-content">
                <div class="metrics-grid grid grid-cols-2 gap-4">
                    <div class="metric">
                        <div class="metric-label text-sm text-gray-600">CPU使用率</div>
                        <div class="metric-value text-2xl font-bold" x-text="data.cpu_usage.toFixed(1) + '%'"></div>
                        <div class="metric-bar bg-gray-200 rounded-full h-2 mt-2">
                            <div class="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                 :style="`width: ${{data.cpu_usage}}%`"></div>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label text-sm text-gray-600">メモリ使用率</div>
                        <div class="metric-value text-2xl font-bold" x-text="data.memory_usage.toFixed(1) + '%'"></div>
                        <div class="metric-bar bg-gray-200 rounded-full h-2 mt-2">
                            <div class="bg-green-500 h-2 rounded-full transition-all duration-300"
                                 :style="`width: ${{data.memory_usage}}%`"></div>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label text-sm text-gray-600">応答時間</div>
                        <div class="metric-value text-lg font-bold" x-text="data.response_time_ms.toFixed(0) + 'ms'"></div>
                    </div>

                    <div class="metric">
                        <div class="metric-label text-sm text-gray-600">アクティブ接続</div>
                        <div class="metric-value text-lg font-bold" x-text="data.active_connections"></div>
                    </div>
                </div>
            </div>
        </div>
        """


class CustomizableDashboard:
    """カスタマイズ可能ダッシュボード"""

    def __init__(self):
        self.layouts: Dict[str, DashboardLayout] = {}
        self.widgets: Dict[str, BaseWidget] = {}
        self.widget_classes = {
            WidgetType.STOCK_CHART: StockChartWidget,
            WidgetType.AI_RECOMMENDATIONS: AIRecommendationsWidget,
            WidgetType.PERFORMANCE_METRICS: PerformanceMetricsWidget,
        }

        # 設定ディレクトリ
        self.config_dir = 'data/dashboard_configs'
        os.makedirs(self.config_dir, exist_ok=True)

        # デフォルトレイアウト作成
        self._create_default_layouts()

    def _create_default_layouts(self):
        """デフォルトレイアウト作成"""
        # メイン分析ダッシュボード
        main_layout = DashboardLayout(
            layout_id="main_analysis",
            name="メイン分析ダッシュボード",
            description="株価分析とAI推奨銘柄の統合ビュー",
            grid_columns=12,
            widgets=[
                WidgetConfig(
                    widget_id=str(uuid.uuid4()),
                    widget_type=WidgetType.STOCK_CHART,
                    title="株価チャート",
                    position=WidgetPosition(x=0, y=0, w=8, h=4),
                    settings={'symbol': '7203', 'period': '1d'}
                ),
                WidgetConfig(
                    widget_id=str(uuid.uuid4()),
                    widget_type=WidgetType.AI_RECOMMENDATIONS,
                    title="AI推奨銘柄",
                    position=WidgetPosition(x=8, y=0, w=4, h=4),
                    settings={'symbols': ['7203', '8306', '9984', '6758', '4689']}
                ),
                WidgetConfig(
                    widget_id=str(uuid.uuid4()),
                    widget_type=WidgetType.PERFORMANCE_METRICS,
                    title="システム性能",
                    position=WidgetPosition(x=0, y=4, w=12, h=3),
                    settings={}
                )
            ],
            theme="default",
            created_at=datetime.now(),
            modified_at=datetime.now()
        )

        self.layouts["main_analysis"] = main_layout
        self.save_layout(main_layout)

        # コンパクトダッシュボード
        compact_layout = DashboardLayout(
            layout_id="compact_overview",
            name="コンパクト概要",
            description="重要指標を簡潔に表示",
            grid_columns=6,
            widgets=[
                WidgetConfig(
                    widget_id=str(uuid.uuid4()),
                    widget_type=WidgetType.AI_RECOMMENDATIONS,
                    title="Top推奨",
                    position=WidgetPosition(x=0, y=0, w=3, h=3),
                    settings={'symbols': ['7203', '8306', '9984']}
                ),
                WidgetConfig(
                    widget_id=str(uuid.uuid4()),
                    widget_type=WidgetType.PERFORMANCE_METRICS,
                    title="システム状況",
                    position=WidgetPosition(x=3, y=0, w=3, h=3),
                    settings={}
                )
            ],
            theme="compact",
            created_at=datetime.now(),
            modified_at=datetime.now()
        )

        self.layouts["compact_overview"] = compact_layout
        self.save_layout(compact_layout)

    def create_widget(self, config: WidgetConfig) -> BaseWidget:
        """ウィジェット作成"""
        widget_class = self.widget_classes.get(config.widget_type)
        if not widget_class:
            raise ValueError(f"Unknown widget type: {config.widget_type}")

        widget = widget_class(config)
        self.widgets[config.widget_id] = widget
        return widget

    def get_layout(self, layout_id: str) -> Optional[DashboardLayout]:
        """レイアウト取得"""
        return self.layouts.get(layout_id)

    def save_layout(self, layout: DashboardLayout):
        """レイアウト保存"""
        layout.modified_at = datetime.now()
        self.layouts[layout.layout_id] = layout

        # ファイルに保存
        file_path = os.path.join(self.config_dir, f"{layout.layout_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(layout), f, default=str, ensure_ascii=False, indent=2)

    def load_layout(self, layout_id: str) -> Optional[DashboardLayout]:
        """レイアウト読み込み"""
        file_path = os.path.join(self.config_dir, f"{layout_id}.json")

        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # データ復元
            layout = DashboardLayout(
                layout_id=data['layout_id'],
                name=data['name'],
                description=data['description'],
                grid_columns=data['grid_columns'],
                theme=data.get('theme', 'default'),
                created_at=datetime.fromisoformat(data['created_at']),
                modified_at=datetime.fromisoformat(data['modified_at']),
                widgets=[]
            )

            # ウィジェット復元
            for widget_data in data.get('widgets', []):
                config = WidgetConfig(
                    widget_id=widget_data['widget_id'],
                    widget_type=WidgetType(widget_data['widget_type']),
                    title=widget_data['title'],
                    position=WidgetPosition(**widget_data['position']),
                    visible=widget_data.get('visible', True),
                    refresh_interval=widget_data.get('refresh_interval', 30),
                    settings=widget_data.get('settings', {})
                )
                layout.widgets.append(config)

            self.layouts[layout_id] = layout
            return layout

        except Exception as e:
            logging.error(f"Failed to load layout {layout_id}: {e}")
            return None

    def get_dashboard_html(self, layout_id: str) -> str:
        """ダッシュボードHTML生成"""
        layout = self.get_layout(layout_id)
        if not layout:
            return "<div>Layout not found</div>"

        # ウィジェット作成・HTML生成
        widget_htmls = []
        widget_data = {}

        for widget_config in layout.widgets:
            if not widget_config.visible:
                continue

            try:
                widget = self.create_widget(widget_config)
                widget_html = widget.get_html_template()
                widget_htmls.append(widget_html)

                # ウィジェットデータ取得
                widget_data[widget_config.widget_id] = widget.get_cached_data()

            except Exception as e:
                logging.error(f"Failed to create widget {widget_config.widget_id}: {e}")

        # メインダッシュボードテンプレート
        dashboard_html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Day Trade Personal - {layout.name}</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
            <script src="https://unpkg.com/gridstack/dist/gridstack-all.js"></script>
            <link href="https://unpkg.com/gridstack/dist/gridstack.min.css" rel="stylesheet"/>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">

            <style>
                .widget {{
                    border: 1px solid #e5e7eb;
                    border-radius: 0.5rem;
                    background: white;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                }}

                .widget-header {{
                    padding: 1rem;
                    border-bottom: 1px solid #e5e7eb;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: #f9fafb;
                }}

                .widget-content {{
                    padding: 1rem;
                    flex: 1;
                    overflow: auto;
                }}

                .grid-stack-item {{
                    border-radius: 0.5rem;
                }}

                .refresh-btn {{
                    padding: 0.25rem 0.5rem;
                    border-radius: 0.25rem;
                    border: 1px solid #d1d5db;
                    background: white;
                    cursor: pointer;
                    transition: all 0.2s;
                }}

                .refresh-btn:hover {{
                    background: #f3f4f6;
                }}

                .metric-bar {{
                    transition: all 0.3s ease;
                }}
            </style>
        </head>

        <body class="bg-gray-100" x-data="dashboardApp()">
            <header class="bg-white shadow-md">
                <div class="container mx-auto px-6 py-4">
                    <div class="flex items-center justify-between">
                        <div>
                            <h1 class="text-2xl font-bold text-gray-800">{layout.name}</h1>
                            <p class="text-gray-600">{layout.description}</p>
                        </div>
                        <div class="flex items-center space-x-4">
                            <select id="layoutSelector" class="border rounded px-3 py-1">
                                <option value="{layout.layout_id}">現在のレイアウト</option>
                            </select>
                            <button @click="refreshAllWidgets()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                                <i class="fas fa-sync-alt mr-2"></i>全更新
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <main class="container mx-auto px-6 py-8">
                <div class="grid-stack">
                    {"".join(widget_htmls)}
                </div>
            </main>

            <script>
                function dashboardApp() {{
                    return {{
                        data: {json.dumps(widget_data, default=str, ensure_ascii=False)},

                        init() {{
                            this.initGridStack();
                            this.startAutoRefresh();
                        }},

                        initGridStack() {{
                            const grid = GridStack.init({{
                                cellHeight: 80,
                                margin: 10,
                                disableResize: false,
                                disableDrag: false
                            }});
                        }},

                        startAutoRefresh() {{
                            setInterval(() => {{
                                this.refreshAllWidgets();
                            }}, 30000); // 30秒間隔
                        }},

                        async refreshAllWidgets() {{
                            try {{
                                const response = await fetch('/api/dashboard/data/{layout_id}');
                                const newData = await response.json();
                                this.data = newData;
                            }} catch (error) {{
                                console.error('Failed to refresh widgets:', error);
                            }}
                        }},

                        getSignalClass(signal) {{
                            const classes = {{
                                'BUY': 'bg-green-100 text-green-800',
                                'SELL': 'bg-red-100 text-red-800',
                                'HOLD': 'bg-yellow-100 text-yellow-800'
                            }};
                            return classes[signal] || 'bg-gray-100 text-gray-800';
                        }},

                        getRiskClass(risk) {{
                            const classes = {{
                                'LOW': 'bg-green-100 text-green-800',
                                'MEDIUM': 'bg-yellow-100 text-yellow-800',
                                'HIGH': 'bg-red-100 text-red-800'
                            }};
                            return classes[risk] || 'bg-gray-100 text-gray-800';
                        }}
                    }}
                }}

                function refreshWidget(widgetId) {{
                    console.log('Refreshing widget:', widgetId);
                }}
            </script>
        </body>
        </html>
        """

        return dashboard_html

    def get_available_layouts(self) -> List[Dict[str, Any]]:
        """利用可能レイアウト一覧"""
        layouts = []
        for layout_id, layout in self.layouts.items():
            layouts.append({
                'layout_id': layout_id,
                'name': layout.name,
                'description': layout.description,
                'widget_count': len(layout.widgets),
                'theme': layout.theme,
                'modified_at': layout.modified_at.isoformat()
            })
        return layouts

    def get_widget_data(self, layout_id: str) -> Dict[str, Any]:
        """ウィジェットデータ取得（API用）"""
        layout = self.get_layout(layout_id)
        if not layout:
            return {}

        widget_data = {}
        for widget_config in layout.widgets:
            if not widget_config.visible:
                continue

            try:
                if widget_config.widget_id not in self.widgets:
                    self.create_widget(widget_config)

                widget = self.widgets[widget_config.widget_id]
                widget_data[widget_config.widget_id] = widget.get_cached_data()

            except Exception as e:
                logging.error(f"Failed to get widget data {widget_config.widget_id}: {e}")
                widget_data[widget_config.widget_id] = {'error': str(e)}

        return widget_data


# グローバルインスタンス
customizable_dashboard = CustomizableDashboard()


def get_dashboard_html(layout_id: str = "main_analysis") -> str:
    """ダッシュボードHTML取得"""
    return customizable_dashboard.get_dashboard_html(layout_id)


def get_widget_data(layout_id: str) -> Dict[str, Any]:
    """ウィジェットデータ取得"""
    return customizable_dashboard.get_widget_data(layout_id)


def get_available_layouts() -> List[Dict[str, Any]]:
    """利用可能レイアウト一覧"""
    return customizable_dashboard.get_available_layouts()


if __name__ == "__main__":
    print("=== Customizable Dashboard Test ===")

    # レイアウト一覧
    layouts = get_available_layouts()
    print(f"Available layouts: {len(layouts)}")
    for layout in layouts:
        print(f"  - {layout['name']}: {layout['widget_count']} widgets")

    # メインダッシュボードHTML生成
    print("\nGenerating main dashboard HTML...")
    html = get_dashboard_html("main_analysis")
    print(f"Generated HTML length: {len(html)} characters")

    # ウィジェットデータ取得テスト
    print("\nTesting widget data retrieval...")
    data = get_widget_data("main_analysis")
    print(f"Widget data keys: {list(data.keys())}")

    print("Customizable dashboard test completed!")