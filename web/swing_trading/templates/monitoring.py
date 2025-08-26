#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Monitoring Template
スイングトレード用監視テンプレート
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