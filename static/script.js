// 時刻表示機能
function updateTime() {
    const timeElement = document.getElementById('currentTime');
    if (timeElement) {
        const now = new Date();
        const timeString = now.toLocaleTimeString('ja-JP', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        timeElement.textContent = timeString;
    }
}

// システムステータス更新
async function updateSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.textContent = `システム稼働中 (v${data.version})`;
            statusElement.className = 'status-live';
        }
    } catch (error) {
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.textContent = 'システム状態不明';
            statusElement.className = 'status-error';
        }
    }
}

// 推奨タイプのテキスト変換
function getRecommendationText(recommendation) {
    const textMap = {
        'BUY': '買い推奨',
        'STRONG_BUY': '強買い推奨', 
        'SELL': '売り推奨',
        'STRONG_SELL': '強売り推奨',
        'HOLD': '様子見'
    };
    return textMap[recommendation] || recommendation;
}

// 信頼度によるCSSクラス
function getConfidenceClass(confidence) {
    if (confidence >= 0.9) return 'confidence-very-high';
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.7) return 'confidence-medium';
    return 'confidence-low';
}

// 推奨タイプによるバッジクラス
function getBadgeClass(recommendation) {
    const classMap = {
        'BUY': 'badge-buy',
        'STRONG_BUY': 'badge-strong-buy',
        'SELL': 'badge-sell',
        'STRONG_SELL': 'badge-strong-sell',
        'HOLD': 'badge-hold'
    };
    return classMap[recommendation] || 'badge-default';
}

// ステータス表示の更新
function updateStatusDisplay(message, isSuccess = true) {
    const statusText = document.getElementById('statusText');
    const lastUpdate = document.getElementById('lastUpdate');
    
    if (statusText) {
        statusText.textContent = message;
        statusText.style.color = isSuccess ? '#48bb78' : '#f56565';
    }
    
    if (lastUpdate && isSuccess) {
        const now = new Date();
        lastUpdate.textContent = ` (${now.toLocaleTimeString('ja-JP')})`;
    }
}

// 手動リフレッシュ
function manualRefresh() {
    loadRecommendations();
}

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    updateTime();
    updateSystemStatus();
    // loadRecommendations(); // 自動読み込みを削除 - ユーザーがボタンを押したときのみ
    
    // 時刻を1秒ごとに更新
    setInterval(updateTime, 1000);
    
    // システムステータスを30秒ごとに更新
    setInterval(updateSystemStatus, 30000);
});

async function runAnalysis() {
    const resultDiv = document.getElementById('analysisResult');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '分析中...';

    try {
        const response = await fetch('/api/analysis/7203');
        const data = await response.json();

        resultDiv.innerHTML = `
            <strong>トヨタ自動車 (${data.symbol})</strong><br>
            推奨: ${data.recommendation}<br>
            信頼度: ${(data.confidence * 100).toFixed(1)}%<br>
            価格: ¥${data.price}<br>
            変動: ${data.change > 0 ? '+' : ''}${data.change}%
        `;
    } catch (error) {
        resultDiv.innerHTML = 'エラーが発生しました: ' + error.message;
    }
}

async function loadRecommendations() {
    const container = document.getElementById('recommendationsContainer');
    const summaryDiv = document.getElementById('summaryStats');
    const listDiv = document.getElementById('recommendationsList');

    container.style.display = 'block';
    listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">推奨銘柄を読み込み中...</div>';

    // 更新状態の表示
    updateStatusDisplay('更新中...', false);
    try {
        const response = await fetch('/api/recommendations');
        const data = await response.json();

        summaryDiv.innerHTML = `
            <div class="stat-item">
                <div class="stat-number">${data.total_count}</div>
                <div class="stat-label">総銘柄数</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.high_confidence_count}</div>
                <div class="stat-label">高信頼度</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.buy_count}</div>
                <div class="stat-label">買い推奨</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.sell_count}</div>
                <div class="stat-label">売り推奨</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.hold_count}</div>
                <div class="stat-label">様子見</div>
            </div>
        `;

        let recommendationsHtml = '';
        data.recommendations.forEach(stock => {
            const recClass = `rec-${stock.recommendation.toLowerCase()}`;
            const confidenceClass = getConfidenceClass(stock.confidence);
            const badgeClass = getBadgeClass(stock.recommendation);
            const changeColor = stock.change >= 0 ? '#48bb78' : '#f56565';
            const changePrefix = stock.change >= 0 ? '+' : '';

            // 推奨度の表示テキスト
            let confidenceText = '';
            if (stock.confidence >= 0.9) confidenceText = '超おすすめ！';
            else if (stock.confidence >= 0.8) confidenceText = 'かなりおすすめ';
            else if (stock.confidence >= 0.7) confidenceText = 'まあまあ';
            else confidenceText = '要検討';

            // 星評価の生成
            const starRating = '★'.repeat(Math.floor(stock.confidence * 5)) + '☆'.repeat(5 - Math.floor(stock.confidence * 5));

            // リスクレベルの表示
            let riskText = '';
            if (stock.confidence >= 0.85) riskText = '低リスク';
            else if (stock.confidence >= 0.75) riskText = '中リスク';
            else riskText = '高リスク';

            recommendationsHtml += `
                <div class="recommendation-card ${recClass} ${confidenceClass}">
                    <div class="stock-header">
                        <div>
                            <div class="stock-name">
                                ${stock.name}
                                <span class="stock-category" style="color: #666; font-size: 0.8rem;">(${stock.sector})</span>
                            </div>
                            <div class="stock-symbol">${stock.symbol}</div>
                            <div class="star-rating">${starRating}</div>
                        </div>
                        <div class="rec-badge ${badgeClass}">${getRecommendationText(stock.recommendation)}</div>
                    </div>
                    <div class="price-info">
                        <div>
                            <strong>¥${stock.price.toLocaleString()}</strong>
                            <span style="color: ${changeColor}; margin-left: 8px;">
                                ${changePrefix}${stock.change.toFixed(1)}%
                            </span>
                        </div>
                        <div>
                            <span style="font-size: 0.9rem; color: #4a5568; font-weight: bold;">
                                ${confidenceText} (信頼度: ${(stock.confidence * 100).toFixed(1)}%)
                            </span>
                        </div>
                    </div>
                    <div class="stock-details">
                        <div class="detail-row">
                            <span class="detail-label">リスク:</span>
                            <span class="detail-value">${riskText}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">理由:</span>
                            <span class="detail-value">${stock.reason}</span>
                        </div>
                    </div>
                    <div class="reason-friendly">
                        ${stock.friendly_reason || stock.reason}
                    </div>
                </div>
            `;
        });

        listDiv.innerHTML = recommendationsHtml;

        // 更新完了状態の表示
        updateStatusDisplay('更新完了', true);
    } catch (error) {
        console.error('推奨銘柄読み込みエラー:', error);
        listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #f56565;">エラーが発生しました: ' + error.message + '</div>';
        updateStatusDisplay('エラー', false, true);
    }
}

function getConfidenceClass(confidence) {
    if (confidence > 0.85) return 'confidence-high';
    if (confidence > 0.70) return 'confidence-medium';
    return 'confidence-low';
}

function getBadgeClass(recommendation) {
    switch (recommendation) {
        case 'BUY': return 'buy-badge';
        case 'SELL': return 'sell-badge';
        case 'HOLD': return 'hold-badge';
        default: return 'hold-badge';
    }
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        console.log('システム状態:', data.status);
    } catch (error) {
        console.error('状態更新エラー:', error);
    }
}

// 画面読み込み完了時の初回自動更新
document.addEventListener('DOMContentLoaded', function() {
    console.log('画面読み込み完了 - 初回データ更新を開始');

    // 自動読み込み機能を完全無効化
    // loadRecommendations(); // 削除
    // updateStatus(); // 削除
    // setInterval(updateStatus, 10000); // 削除

    console.log('手動操作専用モード - 自動読み込み無効');
});

// 手動更新ボタン用の関数（既存の機能を維持）
function manualRefresh() {
    console.log('手動更新が要求されました');
    loadRecommendations();
}

// リアルタイム価格更新機能
async function updateRealTimeData() {
    try {
        const response = await fetch('/api/realtime/snapshot');
        const data = await response.json();

        if (data.prices && Object.keys(data.prices).length > 0) {
            console.log('リアルタイム価格データを更新:', Object.keys(data.prices).length + '銘柄');
            updatePriceDisplay(data.prices);
        }
    } catch (error) {
        console.error('リアルタイムデータ更新エラー:', error);
    }
}

// 価格表示の更新
function updatePriceDisplay(prices) {
    // 推奨銘柄カードの価格を更新
    document.querySelectorAll('.recommendation-card').forEach(card => {
        const symbolElement = card.querySelector('.stock-symbol');
        if (symbolElement) {
            const symbol = symbolElement.textContent.split(' ')[0];
            if (prices[symbol]) {
                const priceInfo = card.querySelector('.price-info strong');
                if (priceInfo) {
                    priceInfo.textContent = `¥${prices[symbol].current_price.toLocaleString()}`;

                    // 価格変動の色を更新
                    const changeElement = priceInfo.nextElementSibling;
                    if (changeElement) {
                        const change = prices[symbol].price_change_pct;
                        const changeColor = change >= 0 ? '#48bb78' : '#f56565';
                        const changePrefix = change >= 0 ? '+' : '';
                        changeElement.style.color = changeColor;
                        changeElement.textContent = `${changePrefix}${change}%`;
                    }
                }
            }
        }
    });
}

// 拡張された自動更新システム
function startEnhancedAutoUpdate() {
    // 推奨銘柄の定期更新（5分間隔）
    // 定期更新機能を完全無効化
    // setInterval(() => {
    //     console.log('定期的な推奨銘柄更新を実行');
    //     loadRecommendations();
    // }, 300000); // 削除

    // setInterval(() => {
    //     updateRealTimeData();
    // }, 30000); // 削除

    console.log('自動更新システム無効化完了');
}

// 拡張自動更新の開始を無効化
// setTimeout(startEnhancedAutoUpdate, 2000); // 削除

// 状態表示の更新
function updateStatusDisplay(message, isSuccess = true, isError = false) {
    const statusText = document.getElementById('statusText');
    const lastUpdate = document.getElementById('lastUpdate');

    if (statusText) {
        statusText.textContent = message;
        statusText.style.color = isError ? '#f56565' : isSuccess ? '#48bb78' : '#ed8936';
    }

    if (lastUpdate && isSuccess) {
        const now = new Date();
        const timeString = now.toLocaleTimeString('ja-JP', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        lastUpdate.textContent = `(最終更新: ${timeString})`;
    }
}

// ページの可視性変更時の処理（自動読み込み無効化）
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        console.log('ページが再表示されました - 自動読み込みは無効');
        // loadRecommendations(); // 自動読み込みを無効化
    }
});

// ネットワーク接続状態の監視（自動読み込み無効化）
window.addEventListener('online', function() {
    console.log('インターネット接続が復帰しました - 自動読み込みは無効');
    updateStatusDisplay('接続復帰しました', true);
    // loadRecommendations(); // 自動読み込みを無効化
});

window.addEventListener('offline', function() {
    console.log('インターネット接続が切断されました');
    updateStatusDisplay('オフライン', false, true);
});
