/**
 * Day Trade Web Client - Enhanced JavaScript
 * リアルタイム取引判断システム用フロントエンド
 */

// エラーハンドリング強化
class ErrorHandler {
    static show(message, type = 'error') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px;
            border-radius: 8px;
            color: white;
            background-color: ${type === 'error' ? '#dc3545' : type === 'success' ? '#28a745' : '#ffc107'};
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            animation: slideIn 0.3s ease-out;
        `;
        alertDiv.textContent = message;
        document.body.appendChild(alertDiv);

        setTimeout(() => {
            alertDiv.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => document.body.removeChild(alertDiv), 300);
        }, 5000);
    }
}

// リアルタイム価格更新機能
class RealTimePriceUpdater {
    constructor() {
        this.symbols = ['7203', '8306', '9984', '6758', '4689']; // 主要銘柄
        this.updateInterval = 30000; // 30秒間隔
        this.isRunning = false;
    }

    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.update();
        this.intervalId = setInterval(() => this.update(), this.updateInterval);
        console.log('リアルタイム価格更新開始');
    }

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isRunning = false;
        console.log('リアルタイム価格更新停止');
    }

    async update() {
        try {
            const symbolsParam = this.symbols.join(',');
            const response = await fetch(`/api/realtime/batch?symbols=${symbolsParam}`, {
                timeout: 10000
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.updatePriceDisplay(data.results);

        } catch (error) {
            console.warn('価格更新エラー:', error.message);
            ErrorHandler.show(`価格更新に失敗しました: ${error.message}`, 'warning');
        }
    }

    updatePriceDisplay(results) {
        results.forEach(result => {
            if (result.status === 'success') {
                const elements = document.querySelectorAll(`[data-symbol="${result.symbol}"]`);
                elements.forEach(element => {
                    const priceElement = element.querySelector('.current-price');
                    const changeElement = element.querySelector('.price-change');

                    if (priceElement) {
                        priceElement.textContent = `¥${result.current_price.toLocaleString()}`;
                    }

                    if (changeElement) {
                        const changeText = `${result.price_change >= 0 ? '+' : ''}${result.price_change_pct.toFixed(2)}%`;
                        changeElement.textContent = changeText;
                        changeElement.className = `price-change ${result.price_change >= 0 ? 'positive' : 'negative'}`;
                    }
                });
            }
        });
    }
}

// 推奨取得機能強化
async function loadRecommendations() {
    const resultDiv = document.getElementById('recommendationsResult');
    const button = event.target;

    // ローディング状態
    button.disabled = true;
    button.textContent = '分析中...';
    resultDiv.innerHTML = '<div class="loading">📊 AI分析実行中...</div>';

    try {
        const response = await fetch('/api/recommendations', {
            timeout: 30000
        });

        if (!response.ok) {
            throw new Error(`サーバーエラー: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayRecommendations(data);
        ErrorHandler.show('分析が完了しました', 'success');

    } catch (error) {
        console.error('推奨取得エラー:', error);
        resultDiv.innerHTML = `
            <div class="error-message">
                <h4>❌ 分析エラー</h4>
                <p>${error.message}</p>
                <button onclick="loadRecommendations()" class="retry-btn">再試行</button>
            </div>
        `;
        ErrorHandler.show(`分析に失敗しました: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        button.textContent = '売買推奨を取得';
    }
}

// 推奨表示機能
function displayRecommendations(data) {
    const resultDiv = document.getElementById('recommendationsResult');
    const containerDiv = document.getElementById('recommendationsContainer');
    const listDiv = document.getElementById('recommendationsList');
    const summaryDiv = document.getElementById('summaryStats');

    // サマリー統計表示
    summaryDiv.innerHTML = `
        <div class="stat-item">
            <span class="stat-number">${data.total_count}</span>
            <span class="stat-label">総銘柄数</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">${data.buy_count}</span>
            <span class="stat-label">買い推奨</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">${data.sell_count}</span>
            <span class="stat-label">売り推奨</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">${data.hold_count}</span>
            <span class="stat-label">様子見</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">${data.high_confidence_count}</span>
            <span class="stat-label">高信頼度</span>
        </div>
    `;

    // 個別銘柄表示
    listDiv.innerHTML = data.recommendations.map(rec => createRecommendationCard(rec)).join('');

    // 結果表示
    resultDiv.innerHTML = `
        <div class="analysis-summary">
            <h4>📈 AI分析結果</h4>
            <p>35銘柄の包括的な分析が完了しました。</p>
            <div class="analysis-timestamp">
                分析時刻: ${new Date(data.timestamp).toLocaleString('ja-JP')}
            </div>
        </div>
    `;

    // コンテナ表示
    containerDiv.style.display = 'block';

    // リアルタイム更新開始
    if (window.priceUpdater) {
        window.priceUpdater.start();
    }
}

// 推奨カード作成
function createRecommendationCard(rec) {
    const confidenceClass = rec.confidence > 0.8 ? 'high' : rec.confidence > 0.6 ? 'medium' : 'low';
    const recommendationClass = rec.recommendation === 'BUY' ? 'buy' : rec.recommendation === 'SELL' ? 'sell' : 'hold';

    return `
        <div class="recommendation-card ${recommendationClass}" data-symbol="${rec.symbol}">
            <div class="card-header">
                <h3>${rec.symbol} - ${rec.name}</h3>
                <div class="recommendation-badge ${recommendationClass}">
                    ${rec.recommendation_friendly || rec.recommendation}
                </div>
            </div>

            <div class="card-content">
                <div class="price-section">
                    <div class="current-price">¥${rec.price ? rec.price.toLocaleString() : 'N/A'}</div>
                    <div class="price-change ${rec.change >= 0 ? 'positive' : 'negative'}">
                        ${rec.change >= 0 ? '+' : ''}${rec.change?.toFixed(2) || '0.00'}%
                    </div>
                </div>

                <div class="confidence-section">
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}"
                             style="width: ${(rec.confidence * 100).toFixed(0)}%"></div>
                    </div>
                    <div class="confidence-text">
                        信頼度: ${(rec.confidence * 100).toFixed(0)}% (${rec.confidence_friendly})
                    </div>
                    <div class="star-rating">${rec.star_rating || '★★★☆☆'}</div>
                </div>

                <div class="details-section">
                    <div class="sector-info">
                        <span class="sector">${rec.sector}</span> -
                        <span class="category">${rec.category}</span>
                    </div>
                    <div class="risk-info">
                        リスク: <span class="risk-level">${rec.risk_level}</span>
                    </div>

                    ${rec.action ? `
                        <div class="action-advice">
                            <strong>アクション:</strong> ${rec.action}
                        </div>
                    ` : ''}

                    ${rec.timing ? `
                        <div class="timing-advice">
                            <strong>タイミング:</strong> ${rec.timing}
                        </div>
                    ` : ''}

                    ${rec.amount_suggestion ? `
                        <div class="amount-advice">
                            <strong>投資額:</strong> ${rec.amount_suggestion}
                        </div>
                    ` : ''}

                    ${rec.target_price ? `
                        <div class="target-info">
                            目標価格: ¥${rec.target_price.toLocaleString()} |
                            損切り: ¥${rec.stop_loss?.toLocaleString() || 'N/A'}
                        </div>
                    ` : ''}
                </div>

                <div class="reason-section">
                    <div class="analysis-reason">
                        ${rec.friendly_reason || rec.reason}
                    </div>
                    ${rec.real_data ? `
                        <div class="data-source">
                            📊 ${rec.data_source || 'リアルデータ'}
                        </div>
                    ` : ''}
                </div>

                ${rec.technical_indicators && !rec.technical_indicators.error ? `
                    <div class="technical-section">
                        <details>
                            <summary>テクニカル指標詳細</summary>
                            <div class="technical-grid">
                                <div class="technical-item">
                                    <span class="label">RSI:</span>
                                    <span class="value">${rec.technical_indicators.rsi?.value} (${rec.technical_indicators.rsi?.status})</span>
                                </div>
                                <div class="technical-item">
                                    <span class="label">MACD:</span>
                                    <span class="value">${rec.technical_indicators.macd?.status}</span>
                                </div>
                                <div class="technical-item">
                                    <span class="label">トレンド:</span>
                                    <span class="value">${rec.technical_indicators.moving_averages?.trend}</span>
                                </div>
                                <div class="technical-item">
                                    <span class="label">ボラティリティ:</span>
                                    <span class="value">${rec.technical_indicators.volatility?.level}</span>
                                </div>
                            </div>
                        </details>
                    </div>
                ` : ''}
            </div>

            <div class="card-footer">
                <button class="detail-btn" onclick="showDetailedAnalysis('${rec.symbol}')">
                    詳細分析
                </button>
                <div class="suitable-investor">
                    ${rec.who_suitable}
                </div>
            </div>
        </div>
    `;
}

// 詳細分析表示
async function showDetailedAnalysis(symbol) {
    try {
        const response = await fetch(`/api/analysis/${symbol}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // モーダル表示ロジック（実装予定）
        console.log('詳細分析:', data);
        ErrorHandler.show(`${symbol}の詳細分析を取得しました`, 'success');

    } catch (error) {
        ErrorHandler.show(`詳細分析の取得に失敗: ${error.message}`, 'error');
    }
}

// ページ読み込み時の初期化
document.addEventListener('DOMContentLoaded', function() {
    console.log('Day Trade Web Client初期化完了');

    // リアルタイム価格更新器初期化
    window.priceUpdater = new RealTimePriceUpdater();

    // CSS アニメーション追加
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #007bff;
            font-weight: bold;
        }
        .error-message {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .retry-btn {
            background: #dc3545;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
    `;
    document.head.appendChild(style);

    // エラーハンドリング設定
    window.addEventListener('error', function(event) {
        console.error('JavaScript Error:', event.error);
        ErrorHandler.show('予期しないエラーが発生しました', 'error');
    });

    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled Promise Rejection:', event.reason);
        ErrorHandler.show('非同期処理でエラーが発生しました', 'error');
    });
});

// ページ離脱時のクリーンアップ
window.addEventListener('beforeunload', function() {
    if (window.priceUpdater) {
        window.priceUpdater.stop();
    }
});