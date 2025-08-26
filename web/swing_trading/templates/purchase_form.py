#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Purchase Form Template
スイングトレード用購入フォームテンプレート
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