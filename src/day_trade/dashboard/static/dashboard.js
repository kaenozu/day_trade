// WebSocket接続
const socket = io();

// 接続状態管理
let isConnected = false;

// セーフモード確認
const SAFE_MODE_ENABLED = true;

// 自動取引機能完全無効化
function blockTradingFunctions() {
    // 取引関連の関数を無効化
    window.executeTrade = function() {
        alert('⚠️ エラー: 自動取引機能は無効化されています。このシステムは分析専用です。');
        return false;
    };

    window.placeOrder = function() {
        alert('⚠️ エラー: 注文実行機能は無効化されています。このシステムは分析専用です。');
        return false;
    };

    // フォームの提出を防ぐ
    document.addEventListener('submit', function(e) {
        if (e.target.action && (e.target.action.includes('/trading/') || e.target.action.includes('/order/'))) {
            e.preventDefault();
            alert('⚠️ エラー: 取引関連の操作は無効化されています。');
            return false;
        }
    });

    console.log('🔒 取引機能を無効化しました（分析専用モード）');
}

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    blockTradingFunctions(); // 取引機能の無効化
});

function initializeDashboard() {
    // 分析専用システム確認メッセージ
    console.log('📊 分析専用システム: データ分析・監視機能のみ有効です');

    // WebSocketイベントリスナー
    socket.on('connect', function() {
        isConnected = true;
        updateConnectionStatus();
        console.log('WebSocket接続確立');
    });

    socket.on('disconnect', function() {
        isConnected = false;
        updateConnectionStatus();
        console.log('WebSocket接続切断');
    });

    socket.on('data_update', function(data) {
        updateDashboardData(data);
    });

    socket.on('analysis_complete', function(data) {
        showAlert('分析完了: ' + data.symbol, 'success');
        updateChart('comprehensive');
    });

    socket.on('system_alert', function(data) {
        showAlert(data.message, data.type || 'warning');
    });

    // 定期更新を開始
    setInterval(updateAllCharts, 30000); // 30秒ごと

    // 初回データ読み込み
    loadInitialData();

    // リフレッシュボタンイベント
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            updateAllCharts();
        });
    }
}

function updateConnectionStatus() {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        if (isConnected) {
            statusElement.className = 'badge bg-success me-3';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> 接続中';
        } else {
            statusElement.className = 'badge bg-danger me-3';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> 切断';
        }
    }
}

function loadInitialData() {
    // サマリー情報の初期読み込み
    fetch('/api/dashboard/summary')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert('データ読み込みエラー: ' + data.error, 'danger');
                return;
            }
            updateSummaryCards(data);
        })
        .catch(error => {
            console.error('サマリーデータ読み込みエラー:', error);
            showAlert('サマリーデータの読み込みに失敗しました', 'warning');
        });

    // 初期チャート読み込み
    updateAllCharts();

    // ステータスレポート読み込み
    loadStatusReport();
}

function updateSummaryCards(data) {
    // ポートフォリオ価値
    const portfolioElement = document.getElementById('portfolio-value');
    if (portfolioElement && data.portfolio_value) {
        portfolioElement.textContent = formatCurrency(data.portfolio_value);
    }

    // 日次リターン
    const returnElement = document.getElementById('daily-return');
    if (returnElement && data.daily_return !== undefined) {
        returnElement.textContent = formatPercentage(data.daily_return);
    }

    // 本日の分析数
    const analysisElement = document.getElementById('analysis-today');
    if (analysisElement && data.analysis_count !== undefined) {
        analysisElement.textContent = data.analysis_count.toString();
    }

    // システム状態
    const statusElement = document.getElementById('system-status');
    if (statusElement && data.system_status) {
        statusElement.textContent = data.system_status;
    }
}

function updateChart(chartType) {
    const chartContainer = document.getElementById(chartType + '-chart');
    if (!chartContainer) return;

    // ローディング表示
    chartContainer.innerHTML = '<div class="text-center p-3"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">読み込み中...</span></div></div>';

    fetch('/api/charts/' + chartType)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                chartContainer.innerHTML = '<div class="text-center p-3 text-danger">エラー: ' + data.error + '</div>';
                return;
            }

            if (data.image_data) {
                // Base64画像データを表示
                chartContainer.innerHTML = '<img src="data:image/png;base64,' + data.image_data + '" class="img-fluid" alt="' + chartType + 'チャート">';
            } else if (data.html) {
                // HTMLチャートを表示
                chartContainer.innerHTML = data.html;
            } else {
                chartContainer.innerHTML = '<div class="text-center p-3 text-muted">チャートデータがありません</div>';
            }
        })
        .catch(error => {
            console.error(chartType + 'チャート読み込みエラー:', error);
            chartContainer.innerHTML = '<div class="text-center p-3 text-danger">チャートの読み込みに失敗しました</div>';
        });
}

function loadStatusReport() {
    fetch('/api/dashboard/status')
        .then(response => response.json())
        .then(data => {
            const reportElement = document.getElementById('status-report');
            if (reportElement) {
                if (data.error) {
                    reportElement.textContent = 'エラー: ' + data.error;
                } else {
                    reportElement.textContent = data.status_text || 'ステータス情報を読み込み中...';
                }
            }
        })
        .catch(error => {
            console.error('ステータスレポート読み込みエラー:', error);
            const reportElement = document.getElementById('status-report');
            if (reportElement) {
                reportElement.textContent = 'ステータス情報の読み込みに失敗しました';
            }
        });
}

function updateDashboardData(data) {
    if (data.summary) {
        updateSummaryCards(data.summary);
    }

    if (data.charts) {
        Object.keys(data.charts).forEach(chartType => {
            updateChart(chartType);
        });
    }

    if (data.status) {
        const reportElement = document.getElementById('status-report');
        if (reportElement) {
            reportElement.textContent = data.status;
        }
    }
}

function updateAllCharts() {
    const chartTypes = ['comprehensive', 'portfolio', 'system', 'analysis', 'risk'];
    chartTypes.forEach(chartType => {
        updateChart(chartType);
    });
}

function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    const alertId = 'alert-' + Date.now();

    const alertHTML = '<div id="' + alertId + '" class="alert alert-' + type + ' alert-custom alert-dismissible fade show" role="alert">' +
        '<strong><i class="fas fa-exclamation-triangle me-2"></i></strong>' +
        message +
        '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' +
        '</div>';

    alertsContainer.insertAdjacentHTML('beforeend', alertHTML);

    // 5秒後に自動削除
    setTimeout(function() {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

// ユーティリティ関数
function formatCurrency(value) {
    if (typeof value !== 'number') return '--';
    return '¥' + value.toLocaleString('ja-JP');
}

function formatPercentage(value) {
    if (typeof value !== 'number') return '--';
    return value.toFixed(2) + '%';
}

// エラーハンドリング
window.addEventListener('error', function(e) {
    console.error('JavaScript エラー:', e.error);
});

// 分析専用システム確認
console.log('🔒 分析専用ダッシュボード初期化完了 - 取引機能は無効化されています');