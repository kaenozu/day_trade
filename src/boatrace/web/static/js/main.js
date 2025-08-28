/**
 * Boatrace予想システム メインJavaScript
 */

// グローバル変数
let systemStatus = {
    api: 'unknown',
    database: 'unknown',
    prediction: 'unknown'
};

// DOM読み込み完了時の初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeSystem();
    setupEventListeners();
    checkSystemStatus();
});

/**
 * システム初期化
 */
function initializeSystem() {
    console.log('🚤 Boatrace予想システム 初期化中...');
    
    // ローディング表示の設定
    setupLoadingIndicators();
    
    // ツールチップの初期化
    initializeTooltips();
    
    // 自動更新の設定
    setupAutoRefresh();
    
    console.log('✅ システム初期化完了');
}

/**
 * イベントリスナーの設定
 */
function setupEventListeners() {
    // グローバルキーボードショートカット
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // フォームの改善
    enhanceForms();
    
    // テーブルの改善
    enhanceTables();
    
    // モーダルの改善
    enhanceModals();
}

/**
 * キーボードショートカット処理
 */
function handleKeyboardShortcuts(event) {
    // Ctrl + Alt + 組み合わせ
    if (event.ctrlKey && event.altKey) {
        switch(event.key) {
            case 'd':
                event.preventDefault();
                window.location.href = '/';
                break;
            case 'r':
                event.preventDefault();
                window.location.href = '/races';
                break;
            case 'p':
                event.preventDefault();
                window.location.href = '/portfolio';
                break;
            case 'b':
                event.preventDefault();
                window.location.href = '/betting';
                break;
            case 's':
                event.preventDefault();
                window.location.href = '/stadiums';
                break;
        }
    }
}

/**
 * ローディングインジケーターの設定
 */
function setupLoadingIndicators() {
    // すべてのフォーム送信にローディング表示
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            showGlobalLoading();
        });
    });
    
    // AJAX リクエストにローディング表示
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        showLoading();
        return originalFetch.apply(this, args).finally(() => {
            hideLoading();
        });
    };
}

/**
 * グローバルローディング表示
 */
function showGlobalLoading() {
    if (document.getElementById('globalLoading')) return;
    
    const loading = document.createElement('div');
    loading.id = 'globalLoading';
    loading.className = 'loading-overlay';
    loading.innerHTML = `
        <div class="text-center">
            <div class="loading-spinner"></div>
            <p class="mt-3">処理中...</p>
        </div>
    `;
    
    document.body.appendChild(loading);
    document.body.style.overflow = 'hidden';
}

/**
 * グローバルローディング非表示
 */
function hideGlobalLoading() {
    const loading = document.getElementById('globalLoading');
    if (loading) {
        document.body.removeChild(loading);
        document.body.style.overflow = 'auto';
    }
}

/**
 * 部分ローディング表示
 */
function showLoading(target = null) {
    const element = target || document.querySelector('.card-body') || document.body;
    
    const loading = document.createElement('div');
    loading.className = 'text-center py-4 loading-indicator';
    loading.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2 text-muted">読み込み中...</p>
    `;
    
    element.style.position = 'relative';
    element.appendChild(loading);
}

/**
 * 部分ローディング非表示
 */
function hideLoading(target = null) {
    const element = target || document;
    const loading = element.querySelector('.loading-indicator');
    if (loading) {
        loading.remove();
    }
}

/**
 * ツールチップ初期化
 */
function initializeTooltips() {
    // Bootstrap tooltip の初期化
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

/**
 * 自動更新の設定
 */
function setupAutoRefresh() {
    // 5分毎にシステム状態をチェック
    setInterval(checkSystemStatus, 300000);
    
    // 10分毎にページデータを更新（ダッシュボードのみ）
    if (window.location.pathname === '/') {
        setInterval(refreshDashboardData, 600000);
    }
}

/**
 * システム状態チェック
 */
async function checkSystemStatus() {
    try {
        // API状態チェック
        const response = await fetch('/api/system/status', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            const status = await response.json();
            updateSystemStatus(status);
        } else {
            updateSystemStatus({ api: 'error', database: 'error', prediction: 'error' });
        }
    } catch (error) {
        console.warn('システム状態チェック失敗:', error);
        updateSystemStatus({ api: 'error', database: 'error', prediction: 'error' });
    }
}

/**
 * システム状態表示更新
 */
function updateSystemStatus(status) {
    systemStatus = { ...systemStatus, ...status };
    
    // ナビゲーションバーの状態表示更新
    const statusIndicator = document.querySelector('.navbar-text');
    if (statusIndicator) {
        const icon = statusIndicator.querySelector('i');
        if (icon) {
            icon.className = systemStatus.api === 'OK' ? 
                'fas fa-circle text-success' : 'fas fa-circle text-danger';
        }
    }
}

/**
 * ダッシュボードデータ更新
 */
async function refreshDashboardData() {
    try {
        const response = await fetch('/api/dashboard/refresh', {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            updateDashboardElements(data);
        }
    } catch (error) {
        console.warn('ダッシュボードデータ更新失敗:', error);
    }
}

/**
 * ダッシュボード要素更新
 */
function updateDashboardElements(data) {
    // 統計カードの更新
    if (data.stats) {
        Object.keys(data.stats).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                element.textContent = data.stats[key];
            }
        });
    }
    
    // 最新レース情報の更新
    if (data.races) {
        updateRacesTable(data.races);
    }
}

/**
 * フォームの機能強化
 */
function enhanceForms() {
    document.querySelectorAll('form').forEach(form => {
        // 送信時の二重送信防止
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 送信中...';
                
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }, 3000);
            }
        });
        
        // リアルタイム入力検証
        const inputs = form.querySelectorAll('input[required], select[required]');
        inputs.forEach(input => {
            input.addEventListener('blur', validateInput);
            input.addEventListener('input', clearValidationError);
        });
    });
}

/**
 * 入力検証
 */
function validateInput(event) {
    const input = event.target;
    const value = input.value.trim();
    
    // 空値チェック
    if (input.hasAttribute('required') && !value) {
        showInputError(input, 'この項目は必須です');
        return false;
    }
    
    // 数値チェック
    if (input.type === 'number') {
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        const numValue = parseFloat(value);
        
        if (isNaN(numValue)) {
            showInputError(input, '数値を入力してください');
            return false;
        }
        
        if (min !== undefined && numValue < min) {
            showInputError(input, `${min}以上の値を入力してください`);
            return false;
        }
        
        if (max !== undefined && numValue > max) {
            showInputError(input, `${max}以下の値を入力してください`);
            return false;
        }
    }
    
    clearValidationError(event);
    return true;
}

/**
 * 入力エラー表示
 */
function showInputError(input, message) {
    clearValidationError({ target: input });
    
    input.classList.add('is-invalid');
    
    const feedback = document.createElement('div');
    feedback.className = 'invalid-feedback';
    feedback.textContent = message;
    
    input.parentNode.appendChild(feedback);
}

/**
 * 入力エラークリア
 */
function clearValidationError(event) {
    const input = event.target;
    input.classList.remove('is-invalid');
    
    const feedback = input.parentNode.querySelector('.invalid-feedback');
    if (feedback) {
        feedback.remove();
    }
}

/**
 * テーブルの機能強化
 */
function enhanceTables() {
    document.querySelectorAll('table').forEach(table => {
        // ソート機能の追加
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            if (header.textContent.trim()) {
                header.style.cursor = 'pointer';
                header.addEventListener('click', () => sortTable(table, index));
            }
        });
        
        // 行クリックハンドラーの改善
        const rows = table.querySelectorAll('tbody tr[data-href]');
        rows.forEach(row => {
            row.style.cursor = 'pointer';
            row.addEventListener('click', function(e) {
                // ボタンやリンククリック時は除外
                if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A' || 
                    e.target.closest('button') || e.target.closest('a')) {
                    return;
                }
                
                const href = this.dataset.href;
                if (href) {
                    window.location.href = href;
                }
            });
        });
    });
}

/**
 * テーブルソート
 */
function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const isNumeric = rows.some(row => {
        const cellText = row.cells[columnIndex]?.textContent.trim();
        return cellText && !isNaN(parseFloat(cellText.replace(/[^0-9.-]/g, '')));
    });
    
    rows.sort((a, b) => {
        const aText = a.cells[columnIndex]?.textContent.trim() || '';
        const bText = b.cells[columnIndex]?.textContent.trim() || '';
        
        if (isNumeric) {
            const aNum = parseFloat(aText.replace(/[^0-9.-]/g, '')) || 0;
            const bNum = parseFloat(bText.replace(/[^0-9.-]/g, '')) || 0;
            return aNum - bNum;
        } else {
            return aText.localeCompare(bText, 'ja');
        }
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

/**
 * モーダルの機能強化
 */
function enhanceModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('show.bs.modal', function() {
            // フォーカス管理
            const firstInput = modal.querySelector('input, select, textarea');
            if (firstInput) {
                setTimeout(() => firstInput.focus(), 500);
            }
        });
        
        modal.addEventListener('hidden.bs.modal', function() {
            // フォームリセット
            const form = modal.querySelector('form');
            if (form) {
                form.reset();
                form.querySelectorAll('.is-invalid').forEach(input => {
                    input.classList.remove('is-invalid');
                });
                form.querySelectorAll('.invalid-feedback').forEach(feedback => {
                    feedback.remove();
                });
            }
        });
    });
}

/**
 * 成功メッセージ表示
 */
function showSuccessMessage(message, duration = 3000) {
    showMessage(message, 'success', duration);
}

/**
 * エラーメッセージ表示
 */
function showErrorMessage(message, duration = 5000) {
    showMessage(message, 'danger', duration);
}

/**
 * 情報メッセージ表示
 */
function showInfoMessage(message, duration = 3000) {
    showMessage(message, 'info', duration);
}

/**
 * メッセージ表示
 */
function showMessage(message, type = 'info', duration = 3000) {
    const alertContainer = document.getElementById('alertContainer') || createAlertContainer();
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertContainer.appendChild(alert);
    
    // 自動削除
    if (duration > 0) {
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, duration);
    }
}

/**
 * アラートコンテナ作成
 */
function createAlertContainer() {
    const container = document.createElement('div');
    container.id = 'alertContainer';
    container.style.position = 'fixed';
    container.style.top = '20px';
    container.style.right = '20px';
    container.style.zIndex = '9999';
    container.style.width = '400px';
    
    document.body.appendChild(container);
    return container;
}

/**
 * データフォーマット用ユーティリティ
 */
const Utils = {
    // 数値を日本円形式でフォーマット
    formatCurrency: (amount) => {
        return new Intl.NumberFormat('ja-JP', {
            style: 'currency',
            currency: 'JPY',
            minimumFractionDigits: 0
        }).format(amount);
    },
    
    // パーセンテージフォーマット
    formatPercent: (value, decimals = 2) => {
        return (value * 100).toFixed(decimals) + '%';
    },
    
    // 日付フォーマット
    formatDate: (date) => {
        return new Intl.DateTimeFormat('ja-JP', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        }).format(new Date(date));
    },
    
    // 時間フォーマット
    formatTime: (date) => {
        return new Intl.DateTimeFormat('ja-JP', {
            hour: '2-digit',
            minute: '2-digit'
        }).format(new Date(date));
    },
    
    // デバウンス関数
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // スロットル関数
    throttle: (func, limit) => {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
};

// グローバルに公開
window.BoatraceUtils = Utils;
window.showSuccessMessage = showSuccessMessage;
window.showErrorMessage = showErrorMessage;
window.showInfoMessage = showInfoMessage;

console.log('🚤 Boatrace予想システム JavaScript 読み込み完了');