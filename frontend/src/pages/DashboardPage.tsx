import { useEffect, useState } from 'react';
import { Helmet } from 'react-helmet-async';
import { motion } from 'framer-motion';
import { useQuery } from 'react-query';

// Components
import { KPICard } from '@/components/cards/KPICard';
import { ChartCard } from '@/components/cards/ChartCard';
import { AnalysisFeed } from '@/components/analysis/AnalysisFeed';
import { MarketOverview } from '@/components/market/MarketOverview';
import { MLPerformance } from '@/components/ml/MLPerformance';
import { QuickStats } from '@/components/dashboard/QuickStats';
import { AlertsSummary } from '@/components/alerts/AlertsSummary';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import { RefreshButton } from '@/components/ui/RefreshButton';

// Hooks
import { useWebSocket } from '@/hooks/useWebSocket';
import { useLocalStorage } from '@/hooks/useLocalStorage';

// Services
import { dashboardService } from '@/services/dashboardService';

// Types
import type { DashboardData, KPIMetric, ChartData } from '@/types/dashboard';

// Utils
import { formatCurrency, formatPercentage } from '@/utils/format';

const REFRESH_INTERVAL = 30000; // 30秒

export const DashboardPage: React.FC = () => {
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [dashboardLayout, setDashboardLayout] = useLocalStorage('dashboard-layout', 'default');

  // WebSocket接続でリアルタイムデータを取得
  const { data: realTimeData, isConnected } = useWebSocket('/ws/dashboard');

  // ダッシュボードデータ取得
  const {
    data: dashboardData,
    isLoading,
    error,
    refetch,
    isRefetching
  } = useQuery<DashboardData>(
    'dashboard-data',
    dashboardService.getDashboardData,
    {
      refetchInterval: REFRESH_INTERVAL,
      onSuccess: () => setLastUpdated(new Date()),
    }
  );

  // リアルタイムデータで更新
  useEffect(() => {
    if (realTimeData) {
      setLastUpdated(new Date());
    }
  }, [realTimeData]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <ErrorMessage
          message="ダッシュボードデータの取得に失敗しました"
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  if (!dashboardData) {
    return null;
  }

  // KPIデータの準備
  const kpiMetrics: KPIMetric[] = [
    {
      id: 'roi',
      title: '今日のROI',
      value: dashboardData.roi.daily,
      unit: '%',
      change: dashboardData.roi.change,
      trend: dashboardData.roi.change >= 0 ? 'up' : 'down',
      status: dashboardData.roi.daily >= 5 ? 'good' : dashboardData.roi.daily >= 3 ? 'warning' : 'critical',
      target: 5.0,
    },
    {
      id: 'trades',
      title: '分析実行数',
      value: dashboardData.trades.executed,
      unit: '件',
      change: dashboardData.trades.change,
      trend: dashboardData.trades.change >= 0 ? 'up' : 'down',
      status: dashboardData.trades.executed >= 100 ? 'good' : 'warning',
      target: 100,
    },
    {
      id: 'accuracy',
      title: 'ML予測精度',
      value: dashboardData.ml.accuracy,
      unit: '%',
      change: dashboardData.ml.accuracyChange,
      trend: dashboardData.ml.accuracyChange >= 0 ? 'up' : 'down',
      status: dashboardData.ml.accuracy >= 93 ? 'good' : dashboardData.ml.accuracy >= 92 ? 'warning' : 'critical',
      target: 93.0,
    },
    {
      id: 'portfolio',
      title: 'ポートフォリオ',
      value: dashboardData.portfolio.totalValue,
      unit: '$',
      change: dashboardData.portfolio.dailyChange,
      trend: dashboardData.portfolio.dailyChange >= 0 ? 'up' : 'down',
      status: dashboardData.portfolio.dailyChange >= 0 ? 'good' : 'critical',
      format: 'currency',
    },
  ];

  return (
    <>
      <Helmet>
        <title>ダッシュボード - Day Trade ML</title>
        <meta name="description" content="リアルタイム取引ダッシュボード" />
      </Helmet>

      <div className="space-y-6">
        {/* ヘッダー */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              ダッシュボード
            </h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              最終更新: {lastUpdated.toLocaleTimeString('ja-JP')}
              {isConnected && (
                <span className="ml-2 inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></span>
                  リアルタイム
                </span>
              )}
            </p>
          </div>

          <RefreshButton
            onRefresh={() => refetch()}
            isRefreshing={isRefetching}
            className="ml-4"
          />
        </div>

        {/* KPI概要 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
        >
          {kpiMetrics.map((metric, index) => (
            <motion.div
              key={metric.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
            >
              <KPICard
                metric={metric}
                realTimeValue={realTimeData?.[metric.id]}
              />
            </motion.div>
          ))}
        </motion.div>

        {/* チャートセクション */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* ROIトレンド */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <ChartCard
              title="ROIトレンド"
              subtitle="過去24時間"
              data={dashboardData.charts.roiTrend}
              type="line"
              height={300}
              options={{
                responsive: true,
                plugins: {
                  legend: {
                    display: true,
                    position: 'top' as const,
                  },
                  tooltip: {
                    mode: 'index' as const,
                    intersect: false,
                  },
                },
                scales: {
                  y: {
                    beginAtZero: false,
                    ticks: {
                      callback: (value: any) => formatPercentage(value),
                    },
                  },
                },
              }}
            />
          </motion.div>

          {/* 取引量 */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <ChartCard
              title="取引量"
              subtitle="時間別実行数"
              data={dashboardData.charts.tradingVolume}
              type="bar"
              height={300}
              options={{
                responsive: true,
                plugins: {
                  legend: {
                    display: false,
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    ticks: {
                      precision: 0,
                    },
                  },
                },
              }}
            />
          </motion.div>
        </div>

        {/* ML パフォーマンス */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <MLPerformance
            data={dashboardData.ml}
            realTimeData={realTimeData?.ml}
          />
        </motion.div>

        {/* 市場概要とリアルタイム取引 */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* 市場概要 */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="lg:col-span-1"
          >
            <MarketOverview
              data={dashboardData.market}
              realTimeData={realTimeData?.market}
            />
          </motion.div>

          {/* リアルタイム分析フィード */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="lg:col-span-2"
          >
            <AnalysisFeed
              analyses={dashboardData.trades.recent}
              realTimeAnalysis={realTimeData?.latestAnalysis}
            />
          </motion.div>
        </div>

        {/* アラート概要 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.7 }}
        >
          <AlertsSummary
            alerts={dashboardData.alerts}
            onViewAll={() => {/* Navigate to alerts page */}}
          />
        </motion.div>

        {/* クイック統計 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <QuickStats
            data={dashboardData.quickStats}
            className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-6"
          />
        </motion.div>
      </div>
    </>
  );
};