import { useEffect, useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Helmet } from 'react-helmet-async';
import {
  ArrowPathIcon,
  SignalIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';

// Components
import { RealTimeKPIGrid } from './RealTimeKPIGrid';
import { LiveTradingFeed } from './LiveTradingFeed';
import { RealTimeCharts } from './RealTimeCharts';
import { MarketStreamPanel } from './MarketStreamPanel';
import { MLPerformanceLive } from './MLPerformanceLive';
import { AlertsStream } from './AlertsStream';
import { SystemHealthIndicator } from './SystemHealthIndicator';
import { ConnectionStatus } from '../ui/ConnectionStatus';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { ErrorMessage } from '../ui/ErrorMessage';

// Hooks
import { useWebSocket } from '@/hooks/useWebSocket';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { useInterval } from '@/hooks/useInterval';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';

// Services
import { dashboardService } from '@/services/dashboardService';

// Types
import type {
  RealTimeDashboardData,
  ConnectionState,
  DashboardLayout,
  AlertData
} from '@/types/dashboard';

// Utils
import { cn } from '@/utils/cn';
import { formatCurrency, formatPercentage } from '@/utils/format';

interface RealTimeDashboardProps {
  className?: string;
  layout?: DashboardLayout;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const RealTimeDashboard: React.FC<RealTimeDashboardProps> = ({
  className,
  layout = 'grid',
  autoRefresh = true,
  refreshInterval = 30000, // 30秒
}) => {
  // State
  const [dashboardData, setDashboardData] = useState<RealTimeDashboardData | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useLocalStorage('dashboard-auto-refresh', autoRefresh);
  const [selectedTimeRange, setSelectedTimeRange] = useLocalStorage('dashboard-time-range', '24h');
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');

  // WebSocket接続
  const {
    data: realTimeData,
    isConnected,
    isConnecting,
    error: wsError,
    reconnectCount,
    send: sendWsMessage,
    reconnect: reconnectWs
  } = useWebSocket('/ws/dashboard', {
    autoReconnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 10,
  });

  // リアルタイムデータ処理
  useEffect(() => {
    if (realTimeData) {
      setDashboardData(prev => ({
        ...prev,
        ...realTimeData,
        lastUpdated: new Date(),
      }));
      setLastUpdated(new Date());
      setError(null);
    }
  }, [realTimeData]);

  // 接続状態管理
  useEffect(() => {
    if (isConnecting) {
      setConnectionState('connecting');
    } else if (isConnected) {
      setConnectionState('connected');
    } else if (wsError) {
      setConnectionState('error');
    } else {
      setConnectionState('disconnected');
    }
  }, [isConnected, isConnecting, wsError]);

  // 初期データ取得
  const fetchInitialData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const data = await dashboardService.getDashboardData({
        timeRange: selectedTimeRange,
        realTime: true,
      });

      setDashboardData(data);
      setLastUpdated(new Date());

    } catch (err) {
      const message = err instanceof Error ? err.message : 'データの取得に失敗しました';
      setError(message);
      console.error('Dashboard data fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedTimeRange]);

  // 初期化
  useEffect(() => {
    fetchInitialData();
  }, [fetchInitialData]);

  // 自動リフレッシュ
  useInterval(
    fetchInitialData,
    autoRefreshEnabled && !isConnected ? refreshInterval : null
  );

  // キーボードショートカット
  useKeyboardShortcuts({
    'r': () => fetchInitialData(),
    'cmd+r': () => fetchInitialData(),
    'c': () => reconnectWs(),
    'cmd+shift+c': () => reconnectWs(),
    'a': () => setAutoRefreshEnabled(!autoRefreshEnabled),
  });

  // WebSocketメッセージ送信
  const handleSubscribe = useCallback((stream: string) => {
    sendWsMessage({
      type: 'subscribe',
      stream,
    });
  }, [sendWsMessage]);

  const handleUnsubscribe = useCallback((stream: string) => {
    sendWsMessage({
      type: 'unsubscribe',
      stream,
    });
  }, [sendWsMessage]);

  // アラート処理
  const handleAlertAction = useCallback(async (alertId: string, action: string) => {
    try {
      await dashboardService.handleAlert(alertId, action);
      // リアルタイムでアラート状態を更新
      sendWsMessage({
        type: 'alert_action',
        alertId,
        action,
      });
    } catch (err) {
      console.error('Alert action error:', err);
    }
  }, [sendWsMessage]);

  // メモ化されたコンポーネント
  const connectionStatus = useMemo(() => (
    <ConnectionStatus
      state={connectionState}
      lastUpdated={lastUpdated}
      reconnectCount={reconnectCount}
      onReconnect={reconnectWs}
      className="mb-4"
    />
  ), [connectionState, lastUpdated, reconnectCount, reconnectWs]);

  const systemHealth = useMemo(() => (
    <SystemHealthIndicator
      data={dashboardData?.systemStatus}
      realTime={isConnected}
      className="mb-4"
    />
  ), [dashboardData?.systemStatus, isConnected]);

  // レンダリング
  if (isLoading && !dashboardData) {
    return (
      <div className="flex items-center justify-center h-full min-h-96">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600 dark:text-gray-400">
            ダッシュボードを読み込み中...
          </p>
        </div>
      </div>
    );
  }

  if (error && !dashboardData) {
    return (
      <div className="flex items-center justify-center h-full min-h-96">
        <ErrorMessage
          message={error}
          onRetry={fetchInitialData}
          className="max-w-md"
        />
      </div>
    );
  }

  return (
    <>
      <Helmet>
        <title>リアルタイムダッシュボード - Day Trade ML</title>
        <meta name="description" content="リアルタイム取引監視ダッシュボード" />
      </Helmet>

      <div className={cn('space-y-6', className)}>
        {/* ヘッダー & 接続状態 */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center">
              <SignalIcon className="h-7 w-7 mr-2 text-blue-500" />
              リアルタイムダッシュボード
            </h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              ライブ市場データと取引状況を監視
            </p>
          </div>

          <div className="flex items-center space-x-4">
            {/* 自動リフレッシュ切り替え */}
            <button
              onClick={() => setAutoRefreshEnabled(!autoRefreshEnabled)}
              className={cn(
                'flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                autoRefreshEnabled
                  ? 'bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200'
                  : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
              )}
            >
              <ArrowPathIcon className={cn('h-4 w-4 mr-2', autoRefreshEnabled && 'animate-spin')} />
              自動更新
            </button>

            {/* 手動リフレッシュ */}
            <button
              onClick={fetchInitialData}
              disabled={isLoading}
              className="flex items-center px-3 py-2 bg-blue-100 text-blue-700 rounded-lg text-sm font-medium hover:bg-blue-200 transition-colors dark:bg-blue-800 dark:text-blue-200"
            >
              <ArrowPathIcon className={cn('h-4 w-4 mr-2', isLoading && 'animate-spin')} />
              更新
            </button>
          </div>
        </div>

        {/* 接続状態 */}
        {connectionStatus}

        {/* システム健全性 */}
        {systemHealth}

        {/* メインコンテンツ */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* KPIグリッド */}
          <div className="lg:col-span-4">
            <RealTimeKPIGrid
              data={dashboardData?.kpis}
              realTimeUpdates={realTimeData?.kpis}
              isConnected={isConnected}
            />
          </div>

          {/* ライブチャート */}
          <div className="lg:col-span-2">
            <RealTimeCharts
              data={dashboardData?.charts}
              realTimeData={realTimeData?.charts}
              timeRange={selectedTimeRange}
              onTimeRangeChange={setSelectedTimeRange}
            />
          </div>

          {/* 市場ストリーム */}
          <div className="lg:col-span-2">
            <MarketStreamPanel
              data={dashboardData?.market}
              realTimeData={realTimeData?.market}
              onSubscribe={handleSubscribe}
              onUnsubscribe={handleUnsubscribe}
            />
          </div>

          {/* ライブ取引フィード */}
          <div className="lg:col-span-2">
            <LiveTradingFeed
              trades={dashboardData?.trades}
              realTimeTrade={realTimeData?.latestTrade}
              maxItems={50}
            />
          </div>

          {/* MLパフォーマンス */}
          <div className="lg:col-span-2">
            <MLPerformanceLive
              data={dashboardData?.mlPerformance}
              realTimeData={realTimeData?.ml}
              targetAccuracy={93.0}
            />
          </div>

          {/* アラートストリーム */}
          <div className="lg:col-span-4">
            <AlertsStream
              alerts={dashboardData?.alerts}
              realTimeAlert={realTimeData?.alerts}
              onAlertAction={handleAlertAction}
              maxItems={20}
            />
          </div>
        </div>

        {/* フッター情報 */}
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-4">
            <span>最終更新: {lastUpdated.toLocaleTimeString('ja-JP')}</span>
            {isConnected && (
              <span className="flex items-center">
                <CheckCircleIcon className="h-4 w-4 text-green-500 mr-1" />
                リアルタイム接続中
              </span>
            )}
            {reconnectCount > 0 && (
              <span className="flex items-center">
                <ExclamationTriangleIcon className="h-4 w-4 text-amber-500 mr-1" />
                再接続回数: {reconnectCount}
              </span>
            )}
          </div>

          <div className="flex items-center space-x-4">
            <span>キーボードショートカット: R (更新), C (再接続), A (自動更新)</span>
          </div>
        </div>
      </div>
    </>
  );
};