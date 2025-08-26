#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Partial Sell Form Template
スイングトレード用部分売却フォームテンプレート
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