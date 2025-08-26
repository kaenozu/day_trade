#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Dashboard Template
スイングトレード用ダッシュボードテンプレート
"""

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