
// WebSocket接続
const socket = io();

// 接続状態管理
let isConnected = false;

// セーフモード確認
const SAFE_MODE_ENABLED = true;


// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    // 分析専用システム確認メッセージ
    console.log('📊 分析専用システム: データ分析・監視機能のみ有効です');

    // WebSocketイベントリスナー
    socket.on('connect', function() {
        isConnected = true;
        updateConnectionStatus(true);
        console.log('WebSocketに接続されました');

        // 初期データ要求
        requestUpdate();
    });

    socket.on('disconnect', function() {
        isConnected = false;
        updateConnectionStatus(false);
        console.log('WebSocketから切断されました');
    });

    socket.on('dashboard_update', function(data) {
        updateDashboardData(data);
    });

    socket.on('error', function(data) {
        showAlert('エラー: ' + data.message, 'danger');
    });

    // 手動更新ボタン
    document.getElementById('refresh-btn').addEventListener('click', function() {
        requestUpdate();
        updateAllCharts();
    });

    // 定期更新設定 (30秒間隔)
    setInterval(function() {
        if (isConnected) {
            requestUpdate();
        }
    }, 30000);

    // 初回チャート読み込み
    setTimeout(updateAllCharts, 1000);
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.className = 'badge bg-success me-3';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> 接続中';
    } else {
        statusElement.className = 'badge bg-danger me-3';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> 切断';
    }
}

function requestUpdate() {
    if (isConnected) {
        socket.emit('request_update');
    }

    // ステータスレポート更新
    fetch('/api/report')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('status-report').textContent = data.report;
            }
        })
        .catch(error => console.error('レポート取得エラー:', error));
}

function updateDashboardData(data) {
    try {
        // ポートフォリオデータ
        if (data.portfolio) {
            const portfolioValue = data.portfolio.total_value;
            const dailyReturn = data.portfolio.daily_return;

            document.getElementById('portfolio-value').textContent =
                new Intl.NumberFormat('ja-JP', {
                    style: 'currency',
                    currency: 'JPY',
                    maximumFractionDigits: 0
                }).format(portfolioValue);

            const returnElement = document.getElementById('daily-return');
            const returnPercent = (dailyReturn * 100).toFixed(2) + '%';
            returnElement.textContent = returnPercent;

            // リターンの色分け
            if (dailyReturn > 0) {
                returnElement.parentElement.parentElement.className = 'card bg-success text-white';
            } else if (dailyReturn < 0) {
                returnElement.parentElement.parentElement.className = 'card bg-danger text-white';
            } else {
                returnElement.parentElement.parentElement.className = 'card bg-secondary text-white';
            }
        }

        // 分析データ
        if (data.analysis) {
            document.getElementById('analysis-today').textContent = data.analysis.completed_today + '件';
        }

        // システムデータ
        if (data.system) {
            const cpuUsage = data.system.cpu_usage;
            let systemStatus = '正常';
            let statusClass = 'bg-success';

            if (cpuUsage > 80) {
                systemStatus = '高負荷';
                statusClass = 'bg-danger';
            } else if (cpuUsage > 60) {
                systemStatus = '注意';
                statusClass = 'bg-warning text-dark';
            }

            document.getElementById('system-status').textContent = systemStatus;
            document.getElementById('system-status').parentElement.parentElement.className =
                'card ' + statusClass + ' text-white';
        }

    } catch (error) {
        console.error('データ更新エラー:', error);
    }
}

function updateChart(chartType) {
    const chartContainer = document.getElementById(chartType + '-chart');

    // ローディング表示
    chartContainer.innerHTML = `
        <div class="text-center p-3">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">読み込み中...</span>
            </div>
            <p class="mt-2">チャートを更新中...</p>
        </div>
    `;

    fetch('/api/chart/' + chartType)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                chartContainer.innerHTML = '<img src="data:image/png;base64,' + data.chart_data + '" class="img-fluid" alt="' + chartType + ' chart">';
            } else {
                chartContainer.innerHTML = '<div class="alert alert-danger">Chart Error: ' + data.error + '</div>';
            }
        })
        .catch(error => {
            console.error(chartType + ' chart error:', error);
            chartContainer.innerHTML = '<div class="alert alert-danger">チャートの読み込みに失敗しました</div>';
        });
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
