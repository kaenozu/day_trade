#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Web UI - スイングトレード用Web管理UI
Issue #941 対応: スケジュール管理UI、売りタイミング監視システムのWeb統合
"""

import json
import traceback
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
from dataclasses import asdict

# カスタムモジュール
try:
    from swing_trading_scheduler import (
        swing_trading_scheduler,
        PurchaseStrategy, HoldingStatus, SellSignalStrength
    )
    HAS_SWING_SCHEDULER = True
except ImportError:
    HAS_SWING_SCHEDULER = False

try:
    from version import get_version_info, __version_full__
    HAS_VERSION_INFO = True
except ImportError:
    HAS_VERSION_INFO = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False

try:
    from audit_logger import audit_logger
    HAS_AUDIT_LOGGER = True
except ImportError:
    HAS_AUDIT_LOGGER = False


class SwingTradingWebUI:
    """スイングトレードWeb UI"""

    def __init__(self, host="127.0.0.1", port=5001):
        self.app = Flask(__name__)
        self.app.secret_key = 'swing_trading_ui_secret_key'
        self.host = host
        self.port = port

        # ルート設定
        self._setup_routes()

        print(f"Swing Trading Web UI initialized on {host}:{port}")

    def _setup_routes(self):
        """ルート設定"""

        @self.app.route('/')
        def dashboard():
            """メインダッシュボード"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return "Swing Trading Scheduler not available", 500

                # ポートフォリオサマリー
                summary = swing_trading_scheduler.get_portfolio_summary()

                # 監視対象一覧
                monitoring_list = swing_trading_scheduler.get_monitoring_list()

                # 最新アラート
                alerts = swing_trading_scheduler.get_alerts(limit=10, unread_only=True)

                # バージョン情報
                version_info = get_version_info() if HAS_VERSION_INFO else {"version": "Unknown"}

                return render_template_string(DASHBOARD_TEMPLATE,
                                            summary=summary,
                                            monitoring_list=monitoring_list,
                                            alerts=alerts,
                                            version_info=version_info)

            except Exception as e:
                error_msg = f"Dashboard error: {str(e)}"
                print(error_msg)
                if HAS_AUDIT_LOGGER:
                    audit_logger.log_error_with_context(e, {"context": "swing_trading_dashboard"})
                return f"Error: {error_msg}", 500

        @self.app.route('/purchase_form')
        def purchase_form():
            """購入記録フォーム"""
            try:
                strategies = [strategy.value for strategy in PurchaseStrategy]
                return render_template_string(PURCHASE_FORM_TEMPLATE, strategies=strategies)
            except Exception as e:
                return f"Error: {str(e)}", 500

        @self.app.route('/record_purchase', methods=['POST'])
        def record_purchase():
            """購入記録処理"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return jsonify({"error": "Swing Trading Scheduler not available"}), 500

                # フォームデータ取得
                symbol = request.form['symbol']
                symbol_name = request.form['symbol_name']
                purchase_price = float(request.form['purchase_price'])
                shares = int(request.form['shares'])
                strategy = PurchaseStrategy(request.form['strategy'])
                purchase_reason = request.form['purchase_reason']
                target_profit_percent = float(request.form.get('target_profit_percent', 20.0))
                stop_loss_percent = float(request.form.get('stop_loss_percent', -10.0))
                expected_hold_days = int(request.form.get('expected_hold_days', 30))

                # 購入記録
                purchase_id = swing_trading_scheduler.record_purchase(
                    symbol=symbol,
                    symbol_name=symbol_name,
                    purchase_price=purchase_price,
                    shares=shares,
                    strategy=strategy,
                    purchase_reason=purchase_reason,
                    target_profit_percent=target_profit_percent,
                    stop_loss_percent=stop_loss_percent,
                    expected_hold_days=expected_hold_days
                )

                return jsonify({
                    "success": True,
                    "purchase_id": purchase_id,
                    "message": f"{symbol_name}の購入を記録しました"
                })

            except Exception as e:
                error_msg = str(e)
                print(f"Purchase record error: {error_msg}")
                if HAS_AUDIT_LOGGER:
                    audit_logger.log_error_with_context(e, {"context": "record_purchase"})
                return jsonify({"success": False, "error": error_msg}), 400

        @self.app.route('/monitoring')
        def monitoring():
            """監視画面"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return "Swing Trading Scheduler not available", 500

                # フィルター取得
                status_filter = request.args.get('status')
                status_enum = HoldingStatus(status_filter) if status_filter else None

                # 監視対象一覧
                monitoring_list = swing_trading_scheduler.get_monitoring_list(status_enum)

                # ステータス選択肢
                status_options = [status.value for status in HoldingStatus]

                return render_template_string(MONITORING_TEMPLATE,
                                            monitoring_list=monitoring_list,
                                            status_options=status_options,
                                            current_filter=status_filter)

            except Exception as e:
                return f"Error: {str(e)}", 500

        @self.app.route('/evaluate_sell/<purchase_id>')
        def evaluate_sell(purchase_id):
            """売りタイミング評価"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return jsonify({"error": "Swing Trading Scheduler not available"}), 500

                # 売りタイミング評価
                monitoring_schedule = swing_trading_scheduler.evaluate_sell_timing(purchase_id)

                if monitoring_schedule:
                    return jsonify({
                        "success": True,
                        "evaluation": {
                            "symbol": monitoring_schedule.symbol,
                            "current_price": monitoring_schedule.current_price,
                            "change_percent": monitoring_schedule.current_change_percent,
                            "signal_strength": monitoring_schedule.sell_signal_strength.value,
                            "confidence_score": monitoring_schedule.confidence_score,
                            "status": monitoring_schedule.status.value,
                            "alert_level": monitoring_schedule.alert_level,
                            "signals": monitoring_schedule.sell_signal_reasons,
                            "updated_at": monitoring_schedule.updated_at.isoformat()
                        }
                    })
                else:
                    return jsonify({"success": False, "error": "Evaluation failed"}), 400

            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/partial_sell_form/<purchase_id>')
        def partial_sell_form(purchase_id):
            """部分売却フォーム"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return "Swing Trading Scheduler not available", 500

                # 購入記録取得
                purchase_record = swing_trading_scheduler.get_purchase_record(purchase_id)
                if not purchase_record:
                    return "Purchase record not found", 404

                # 現在の保有株数
                current_shares = swing_trading_scheduler._get_current_shares(purchase_id)

                return render_template_string(PARTIAL_SELL_FORM_TEMPLATE,
                                            purchase_record=purchase_record,
                                            current_shares=current_shares)

            except Exception as e:
                return f"Error: {str(e)}", 500

        @self.app.route('/record_partial_sell', methods=['POST'])
        def record_partial_sell():
            """部分売却記録処理"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return jsonify({"error": "Swing Trading Scheduler not available"}), 500

                # フォームデータ取得
                purchase_id = request.form['purchase_id']
                sell_price = float(request.form['sell_price'])
                shares_sold = int(request.form['shares_sold'])
                sell_reason = request.form['sell_reason']

                # 部分売却記録
                sell_id = swing_trading_scheduler.record_partial_sell(
                    purchase_id=purchase_id,
                    sell_price=sell_price,
                    shares_sold=shares_sold,
                    sell_reason=sell_reason
                )

                return jsonify({
                    "success": True,
                    "sell_id": sell_id,
                    "message": f"{shares_sold}株の部分売却を記録しました"
                })

            except Exception as e:
                error_msg = str(e)
                print(f"Partial sell error: {error_msg}")
                return jsonify({"success": False, "error": error_msg}), 400

        @self.app.route('/alerts')
        def alerts():
            """アラート一覧"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return "Swing Trading Scheduler not available", 500

                # フィルター取得
                unread_only = request.args.get('unread_only', 'false').lower() == 'true'
                limit = int(request.args.get('limit', 50))

                # アラート一覧
                alerts_list = swing_trading_scheduler.get_alerts(limit=limit, unread_only=unread_only)

                return render_template_string(ALERTS_TEMPLATE,
                                            alerts=alerts_list,
                                            unread_only=unread_only)

            except Exception as e:
                return f"Error: {str(e)}", 500

        @self.app.route('/mark_alert_read/<alert_id>')
        def mark_alert_read(alert_id):
            """アラート既読処理"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return jsonify({"error": "Swing Trading Scheduler not available"}), 500

                success = swing_trading_scheduler.mark_alert_as_read(alert_id)

                return jsonify({
                    "success": success,
                    "message": "アラートを既読にしました" if success else "アラートが見つかりません"
                })

            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/portfolio_summary')
        def api_portfolio_summary():
            """ポートフォリオサマリーAPI"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return jsonify({"error": "Swing Trading Scheduler not available"}), 500

                summary = swing_trading_scheduler.get_portfolio_summary()
                return jsonify(summary)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/monitoring_list')
        def api_monitoring_list():
            """監視対象一覧API"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return jsonify({"error": "Swing Trading Scheduler not available"}), 500

                status_filter = request.args.get('status')
                status_enum = HoldingStatus(status_filter) if status_filter else None

                monitoring_list = swing_trading_scheduler.get_monitoring_list(status_enum)
                return jsonify(monitoring_list)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/alerts')
        def api_alerts():
            """アラートAPI"""
            try:
                if not HAS_SWING_SCHEDULER:
                    return jsonify({"error": "Swing Trading Scheduler not available"}), 500

                unread_only = request.args.get('unread_only', 'false').lower() == 'true'
                limit = int(request.args.get('limit', 50))

                alerts_list = swing_trading_scheduler.get_alerts(limit=limit, unread_only=unread_only)
                return jsonify(alerts_list)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def run(self, debug=False):
        """Webサーバー実行"""
        print(f"Starting Swing Trading Web UI on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


# HTMLテンプレート

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>スイングトレード管理ダッシュボード</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        .alert-priority-1 { border-left: 5px solid #dc3545; }
        .alert-priority-2 { border-left: 5px solid #fd7e14; }
        .alert-priority-3 { border-left: 5px solid #ffc107; }
        .alert-priority-4 { border-left: 5px solid #0dcaf0; }
        .alert-priority-5 { border-left: 5px solid #6c757d; }
        .status-monitoring { color: #0d6efd; }
        .status-sell-consider { color: #fd7e14; }
        .status-attention { color: #dc3545; }
        .status-partial-sold { color: #198754; }
        .status-sold { color: #6c757d; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="bi bi-graph-up-arrow"></i>
                スイングトレード管理
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text">{{ version_info.version if version_info else 'v1.0' }}</span>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- ポートフォリオサマリー -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">総ポジション数</h5>
                        <h3 class="text-primary">{{ summary.portfolio_summary.total_positions }}</h3>
                        <small class="text-muted">アクティブ: {{ summary.portfolio_summary.active_positions }}</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">投資総額</h5>
                        <h3 class="text-info">¥{{ "{:,.0f}".format(summary.portfolio_summary.total_invested) }}</h3>
                        <small class="text-muted">現在価値: ¥{{ "{:,.0f}".format(summary.portfolio_summary.current_value) }}</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">未実現損益</h5>
                        <h3 class="{{ 'text-success' if summary.portfolio_summary.unrealized_profit_loss >= 0 else 'text-danger' }}">
                            ¥{{ "{:+,.0f}".format(summary.portfolio_summary.unrealized_profit_loss) }}
                        </h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">未読アラート</h5>
                        <h3 class="text-warning">{{ summary.alerts.unread_alerts }}</h3>
                        <small class="text-muted">高優先度: {{ summary.alerts.high_priority_alerts }}</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- ナビゲーションタブ -->
        <ul class="nav nav-tabs mb-4" id="mainTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="monitoring-tab" data-bs-toggle="tab" data-bs-target="#monitoring" type="button">
                    <i class="bi bi-eye"></i> 監視対象
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="alerts-tab" data-bs-toggle="tab" data-bs-target="#alerts" type="button">
                    <i class="bi bi-bell"></i> アラート
                    {% if summary.alerts.unread_alerts > 0 %}
                        <span class="badge bg-danger">{{ summary.alerts.unread_alerts }}</span>
                    {% endif %}
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <a class="nav-link" href="/purchase_form">
                    <i class="bi bi-plus-circle"></i> 新規購入
                </a>
            </li>
        </ul>

        <div class="tab-content" id="mainTabsContent">
            <!-- 監視対象タブ -->
            <div class="tab-pane fade show active" id="monitoring" role="tabpanel">
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5 class="mb-0">監視対象銘柄</h5>
                                <button class="btn btn-sm btn-outline-primary" onclick="refreshMonitoring()">
                                    <i class="bi bi-arrow-clockwise"></i> 更新
                                </button>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-hover">
                                        <thead>
                                            <tr>
                                                <th>銘柄</th>
                                                <th>購入日</th>
                                                <th>購入価格</th>
                                                <th>現在価格</th>
                                                <th>変動率</th>
                                                <th>保有株数</th>
                                                <th>未実現損益</th>
                                                <th>ステータス</th>
                                                <th>アクション</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for item in monitoring_list %}
                                            <tr>
                                                <td>
                                                    <strong>{{ item.symbol }}</strong><br>
                                                    <small class="text-muted">{{ item.symbol_name }}</small>
                                                </td>
                                                <td>{{ item.purchase_date }}</td>
                                                <td>¥{{ "{:,.0f}".format(item.purchase_price) }}</td>
                                                <td>
                                                    {% if item.current_price %}
                                                        ¥{{ "{:,.0f}".format(item.current_price) }}
                                                    {% else %}
                                                        <span class="text-muted">-</span>
                                                    {% endif %}
                                                </td>
                                                <td>
                                                    {% if item.change_percent is not none %}
                                                        <span class="{{ 'text-success' if item.change_percent >= 0 else 'text-danger' }}">
                                                            {{ "{:+.1f}".format(item.change_percent) }}%
                                                        </span>
                                                    {% else %}
                                                        <span class="text-muted">-</span>
                                                    {% endif %}
                                                </td>
                                                <td>{{ item.current_shares }}株</td>
                                                <td class="{{ 'text-success' if item.unrealized_profit_loss >= 0 else 'text-danger' }}">
                                                    ¥{{ "{:+,.0f}".format(item.unrealized_profit_loss) }}
                                                </td>
                                                <td>
                                                    <span class="badge bg-{{
                                                        'primary' if item.status == 'monitoring' else
                                                        'warning' if item.status == 'sell_consider' else
                                                        'danger' if item.status == 'attention' else
                                                        'success' if item.status == 'partial_sold' else
                                                        'secondary'
                                                    }}">
                                                        {{ item.status }}
                                                    </span>
                                                    {% if item.alert_level > 0 %}
                                                        <br><small class="text-danger">Alert: {{ item.alert_level }}</small>
                                                    {% endif %}
                                                </td>
                                                <td>
                                                    <div class="btn-group-vertical btn-group-sm">
                                                        <button class="btn btn-outline-info btn-sm" onclick="evaluateSell('{{ item.purchase_id }}')">
                                                            <i class="bi bi-search"></i> 評価
                                                        </button>
                                                        {% if item.status not in ['sold'] %}
                                                            <a href="/partial_sell_form/{{ item.purchase_id }}" class="btn btn-outline-success btn-sm">
                                                                <i class="bi bi-currency-dollar"></i> 売却
                                                            </a>
                                                        {% endif %}
                                                    </div>
                                                </td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- アラートタブ -->
            <div class="tab-pane fade" id="alerts" role="tabpanel">
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">最新アラート</h5>
                            </div>
                            <div class="card-body">
                                {% for alert in alerts %}
                                <div class="alert alert-light alert-priority-{{ alert.priority }} {% if not alert.is_read %}border-primary{% endif %}" role="alert">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <h6 class="alert-heading">{{ alert.symbol }} - {{ alert.symbol_name or alert.symbol }}</h6>
                                            <p class="mb-1">{{ alert.message }}</p>
                                            <small class="text-muted">{{ alert.created_at }}</small>
                                        </div>
                                        <div>
                                            {% if not alert.is_read %}
                                                <button class="btn btn-sm btn-outline-primary" onclick="markAlertRead('{{ alert.id }}')">
                                                    既読
                                                </button>
                                            {% endif %}
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                                {% if alerts|length == 0 %}
                                <div class="text-center text-muted">
                                    <i class="bi bi-bell-slash"></i>
                                    <p>未読アラートはありません</p>
                                </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- JavaScript -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 監視データ更新
        function refreshMonitoring() {
            location.reload();
        }

        // 売りタイミング評価
        function evaluateSell(purchaseId) {
            fetch('/evaluate_sell/' + purchaseId)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('評価完了:\\n' +
                              'シグナル強度: ' + data.evaluation.signal_strength + '\\n' +
                              '信頼度: ' + data.evaluation.confidence_score.toFixed(2));
                        refreshMonitoring();
                    } else {
                        alert('評価エラー: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('評価エラー: ' + error);
                });
        }

        // アラート既読
        function markAlertRead(alertId) {
            fetch('/mark_alert_read/' + alertId)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    } else {
                        alert('エラー: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('エラー: ' + error);
                });
        }

        // 自動リフレッシュ（5分間隔）
        setInterval(function() {
            if (document.visibilityState === 'visible') {
                refreshMonitoring();
            }
        }, 300000);
    </script>
</body>
</html>
"""

PURCHASE_FORM_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>新規購入記録 - スイングトレード管理</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="bi bi-graph-up-arrow"></i>
                スイングトレード管理
            </a>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-8 mx-auto">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">新規購入記録</h5>
                    </div>
                    <div class="card-body">
                        <form id="purchaseForm">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="symbol" class="form-label">銘柄コード *</label>
                                        <input type="text" class="form-control" id="symbol" name="symbol" required
                                               placeholder="例: 7203">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="symbol_name" class="form-label">銘柄名 *</label>
                                        <input type="text" class="form-control" id="symbol_name" name="symbol_name" required
                                               placeholder="例: トヨタ自動車">
                                    </div>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="purchase_price" class="form-label">購入価格 *</label>
                                        <div class="input-group">
                                            <span class="input-group-text">¥</span>
                                            <input type="number" class="form-control" id="purchase_price" name="purchase_price"
                                                   required min="1" step="1" placeholder="2500">
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="shares" class="form-label">購入株数 *</label>
                                        <div class="input-group">
                                            <input type="number" class="form-control" id="shares" name="shares"
                                                   required min="1" step="1" placeholder="100">
                                            <span class="input-group-text">株</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="strategy" class="form-label">投資戦略 *</label>
                                        <select class="form-select" id="strategy" name="strategy" required>
                                            {% for strategy in strategies %}
                                            <option value="{{ strategy }}">{{ strategy.title() }}</option>
                                            {% endfor %}
                                        </select>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="expected_hold_days" class="form-label">予定保有期間</label>
                                        <div class="input-group">
                                            <input type="number" class="form-control" id="expected_hold_days" name="expected_hold_days"
                                                   value="30" min="1" step="1">
                                            <span class="input-group-text">日</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="target_profit_percent" class="form-label">目標利益率</label>
                                        <div class="input-group">
                                            <input type="number" class="form-control" id="target_profit_percent" name="target_profit_percent"
                                                   value="20" min="-100" step="0.1">
                                            <span class="input-group-text">%</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="stop_loss_percent" class="form-label">ストップロス</label>
                                        <div class="input-group">
                                            <input type="number" class="form-control" id="stop_loss_percent" name="stop_loss_percent"
                                                   value="-10" max="0" step="0.1">
                                            <span class="input-group-text">%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label for="purchase_reason" class="form-label">購入理由 *</label>
                                <textarea class="form-control" id="purchase_reason" name="purchase_reason" rows="3" required
                                          placeholder="例: PER低位、配当利回り良好、業績安定"></textarea>
                            </div>

                            <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                                <a href="/" class="btn btn-secondary">戻る</a>
                                <button type="submit" class="btn btn-primary">
                                    <i class="bi bi-save"></i> 購入記録
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('purchaseForm').addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(this);

            fetch('/record_purchase', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);
                    window.location.href = '/';
                } else {
                    alert('エラー: ' + data.error);
                }
            })
            .catch(error => {
                alert('エラー: ' + error);
            });
        });
    </script>
</body>
</html>
"""

PARTIAL_SELL_FORM_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>部分売却 - スイングトレード管理</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="bi bi-graph-up-arrow"></i>
                スイングトレード管理
            </a>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-8 mx-auto">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">部分売却</h5>
                    </div>
                    <div class="card-body">
                        <!-- 銘柄情報 -->
                        <div class="alert alert-info">
                            <h6><i class="bi bi-info-circle"></i> 銘柄情報</h6>
                            <p class="mb-1"><strong>{{ purchase_record.symbol }} - {{ purchase_record.symbol_name }}</strong></p>
                            <p class="mb-1">購入日: {{ purchase_record.purchase_date }} | 購入価格: ¥{{ "{:,.0f}".format(purchase_record.purchase_price) }}</p>
                            <p class="mb-0">現在保有株数: <strong>{{ current_shares }}株</strong></p>
                        </div>

                        <form id="partialSellForm">
                            <input type="hidden" name="purchase_id" value="{{ purchase_record.id }}">

                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="sell_price" class="form-label">売却価格 *</label>
                                        <div class="input-group">
                                            <span class="input-group-text">¥</span>
                                            <input type="number" class="form-control" id="sell_price" name="sell_price"
                                                   required min="1" step="1" placeholder="2750">
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="shares_sold" class="form-label">売却株数 *</label>
                                        <div class="input-group">
                                            <input type="number" class="form-control" id="shares_sold" name="shares_sold"
                                                   required min="1" max="{{ current_shares }}" step="1" placeholder="30">
                                            <span class="input-group-text">株</span>
                                        </div>
                                        <div class="form-text">最大: {{ current_shares }}株</div>
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label for="sell_reason" class="form-label">売却理由 *</label>
                                <textarea class="form-control" id="sell_reason" name="sell_reason" rows="3" required
                                          placeholder="例: 利益確定（一部）、リスク軽減、資金需要"></textarea>
                            </div>

                            <!-- 予想損益表示 -->
                            <div class="alert alert-light" id="profitLossPreview" style="display: none;">
                                <h6><i class="bi bi-calculator"></i> 予想損益</h6>
                                <p class="mb-0" id="profitLossText"></p>
                            </div>

                            <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                                <a href="/" class="btn btn-secondary">戻る</a>
                                <button type="submit" class="btn btn-success">
                                    <i class="bi bi-currency-dollar"></i> 売却実行
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const purchasePrice = {{ purchase_record.purchase_price }};

        // 損益プレビュー更新
        function updateProfitLossPreview() {
            const sellPrice = parseFloat(document.getElementById('sell_price').value);
            const sharesSold = parseInt(document.getElementById('shares_sold').value);

            if (sellPrice && sharesSold) {
                const profitLoss = (sellPrice - purchasePrice) * sharesSold;
                const profitLossPercent = ((sellPrice - purchasePrice) / purchasePrice) * 100;

                const preview = document.getElementById('profitLossPreview');
                const text = document.getElementById('profitLossText');

                const colorClass = profitLoss >= 0 ? 'text-success' : 'text-danger';
                const sign = profitLoss >= 0 ? '+' : '';

                text.innerHTML = `<span class="${colorClass}">` +
                               `${sign}¥${profitLoss.toLocaleString()} (${sign}${profitLossPercent.toFixed(1)}%)` +
                               '</span>';
                preview.style.display = 'block';
            } else {
                document.getElementById('profitLossPreview').style.display = 'none';
            }
        }

        // イベントリスナー
        document.getElementById('sell_price').addEventListener('input', updateProfitLossPreview);
        document.getElementById('shares_sold').addEventListener('input', updateProfitLossPreview);

        // フォーム送信
        document.getElementById('partialSellForm').addEventListener('submit', function(e) {
            e.preventDefault();

            if (!confirm('売却を実行しますか？')) {
                return;
            }

            const formData = new FormData(this);

            fetch('/record_partial_sell', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);
                    window.location.href = '/';
                } else {
                    alert('エラー: ' + data.error);
                }
            })
            .catch(error => {
                alert('エラー: ' + error);
            });
        });
    </script>
</body>
</html>
"""

MONITORING_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>監視管理 - スイングトレード管理</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="bi bi-graph-up-arrow"></i>
                スイングトレード管理
            </a>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">監視対象銘柄 ({{ monitoring_list|length }}件)</h5>
                        <div>
                            <select class="form-select form-select-sm d-inline-block w-auto" onchange="filterByStatus(this.value)">
                                <option value="">すべて</option>
                                {% for status in status_options %}
                                <option value="{{ status }}" {{ 'selected' if current_filter == status else '' }}>{{ status }}</option>
                                {% endfor %}
                            </select>
                            <button class="btn btn-sm btn-outline-primary ms-2" onclick="location.reload()">
                                <i class="bi bi-arrow-clockwise"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>銘柄</th>
                                        <th>購入日</th>
                                        <th>購入価格</th>
                                        <th>現在価格</th>
                                        <th>変動率</th>
                                        <th>保有株数</th>
                                        <th>未実現損益</th>
                                        <th>ステータス</th>
                                        <th>最終評価</th>
                                        <th>アクション</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for item in monitoring_list %}
                                    <tr>
                                        <td>
                                            <strong>{{ item.symbol }}</strong><br>
                                            <small class="text-muted">{{ item.symbol_name }}</small>
                                        </td>
                                        <td>{{ item.purchase_date }}</td>
                                        <td>¥{{ "{:,.0f}".format(item.purchase_price) }}</td>
                                        <td>
                                            {% if item.current_price %}
                                                ¥{{ "{:,.0f}".format(item.current_price) }}
                                            {% else %}
                                                <span class="text-muted">-</span>
                                            {% endif %}
                                        </td>
                                        <td>
                                            {% if item.change_percent is not none %}
                                                <span class="{{ 'text-success' if item.change_percent >= 0 else 'text-danger' }}">
                                                    {{ "{:+.1f}".format(item.change_percent) }}%
                                                </span>
                                            {% else %}
                                                <span class="text-muted">-</span>
                                            {% endif %}
                                        </td>
                                        <td>{{ item.current_shares }}株</td>
                                        <td class="{{ 'text-success' if item.unrealized_profit_loss >= 0 else 'text-danger' }}">
                                            ¥{{ "{:+,.0f}".format(item.unrealized_profit_loss) }}
                                        </td>
                                        <td>
                                            <span class="badge bg-{{
                                                'primary' if item.status == 'monitoring' else
                                                'warning' if item.status == 'sell_consider' else
                                                'danger' if item.status == 'attention' else
                                                'success' if item.status == 'partial_sold' else
                                                'secondary'
                                            }}">
                                                {{ item.status }}
                                            </span>
                                            {% if item.alert_level > 0 %}
                                                <br><small class="text-danger">Alert: {{ item.alert_level }}</small>
                                            {% endif %}
                                        </td>
                                        <td>
                                            <small class="text-muted">{{ item.last_evaluation or '-' }}</small>
                                        </td>
                                        <td>
                                            <div class="btn-group-vertical btn-group-sm">
                                                <button class="btn btn-outline-info btn-sm" onclick="evaluateSell('{{ item.purchase_id }}')">
                                                    <i class="bi bi-search"></i>
                                                </button>
                                                {% if item.status not in ['sold'] %}
                                                    <a href="/partial_sell_form/{{ item.purchase_id }}" class="btn btn-outline-success btn-sm">
                                                        <i class="bi bi-currency-dollar"></i>
                                                    </a>
                                                {% endif %}
                                            </div>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function filterByStatus(status) {
            const url = new URL(window.location);
            if (status) {
                url.searchParams.set('status', status);
            } else {
                url.searchParams.delete('status');
            }
            window.location.href = url.toString();
        }

        function evaluateSell(purchaseId) {
            fetch('/evaluate_sell/' + purchaseId)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('評価完了');
                        location.reload();
                    } else {
                        alert('評価エラー: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('評価エラー: ' + error);
                });
        }
    </script>
</body>
</html>
"""

ALERTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>アラート管理 - スイングトレード管理</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="bi bi-graph-up-arrow"></i>
                スイングトレード管理
            </a>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">アラート ({{ alerts|length }}件)</h5>
                        <div>
                            <a href="/alerts?unread_only=true" class="btn btn-sm btn-outline-warning {{ 'active' if unread_only else '' }}">
                                未読のみ
                            </a>
                            <a href="/alerts" class="btn btn-sm btn-outline-primary {{ '' if unread_only else 'active' }}">
                                すべて
                            </a>
                        </div>
                    </div>
                    <div class="card-body">
                        {% for alert in alerts %}
                        <div class="alert alert-light {% if not alert.is_read %}border-primary{% endif %}" role="alert">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h6 class="alert-heading">
                                        <i class="bi bi-{{
                                            'exclamation-triangle-fill text-danger' if alert.priority <= 2 else
                                            'exclamation-circle text-warning' if alert.priority == 3 else
                                            'info-circle text-info'
                                        }}"></i>
                                        {{ alert.symbol }}
                                        {% if alert.symbol_name %}
                                            - {{ alert.symbol_name }}
                                        {% endif %}
                                    </h6>
                                    <p class="mb-1">{{ alert.message }}</p>
                                    <small class="text-muted">
                                        {{ alert.created_at }} |
                                        優先度: {{ alert.priority }} |
                                        種類: {{ alert.alert_type }}
                                    </small>
                                </div>
                                <div>
                                    {% if not alert.is_read %}
                                        <button class="btn btn-sm btn-outline-primary" onclick="markAlertRead('{{ alert.id }}')">
                                            <i class="bi bi-check"></i> 既読
                                        </button>
                                    {% else %}
                                        <span class="badge bg-secondary">既読</span>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                        {% endfor %}

                        {% if alerts|length == 0 %}
                        <div class="text-center text-muted py-5">
                            <i class="bi bi-bell-slash display-4"></i>
                            <p class="mt-3">アラートはありません</p>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function markAlertRead(alertId) {
            fetch('/mark_alert_read/' + alertId)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    } else {
                        alert('エラー: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('エラー: ' + error);
                });
        }
    </script>
</body>
</html>
"""


# グローバルインスタンス
swing_trading_web_ui = SwingTradingWebUI()


if __name__ == "__main__":
    # テスト実行
    print("Swing Trading Web UI テスト開始")

    # Web UI起動
    ui = SwingTradingWebUI(host="127.0.0.1", port=5001)

    print("Web UI starting...")
    print("ブラウザで http://127.0.0.1:5001 にアクセスしてください")

    # デバッグモードで起動
    ui.run(debug=True)