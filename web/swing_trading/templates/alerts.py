#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Alerts Template
スイングトレード用アラートテンプレート
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