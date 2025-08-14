import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ArrowUpIcon,
  ArrowDownIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';

// Types
import type { KPIMetric } from '@/types/dashboard';

// Utils
import { cn } from '@/utils/cn';
import { formatCurrency, formatPercentage, formatNumber } from '@/utils/format';

interface KPICardProps {
  metric: KPIMetric;
  realTimeValue?: number;
  className?: string;
  showTrend?: boolean;
  showTarget?: boolean;
}

export const KPICard: React.FC<KPICardProps> = ({
  metric,
  realTimeValue,
  className,
  showTrend = true,
  showTarget = true,
}) => {
  // 表示する値（リアルタイムが利用可能な場合はそれを使用）
  const displayValue = realTimeValue ?? metric.value;

  // フォーマット済みの値
  const formattedValue = useMemo(() => {
    switch (metric.format || 'number') {
      case 'currency':
        return formatCurrency(displayValue);
      case 'percentage':
        return formatPercentage(displayValue);
      default:
        return formatNumber(displayValue);
    }
  }, [displayValue, metric.format]);

  // ステータスに基づくスタイル
  const statusStyles = useMemo(() => {
    const base = "transition-all duration-200";

    switch (metric.status) {
      case 'good':
        return {
          card: `${base} border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20`,
          value: 'text-green-700 dark:text-green-300',
          trend: 'text-green-600 dark:text-green-400',
          icon: 'text-green-500',
        };
      case 'warning':
        return {
          card: `${base} border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20`,
          value: 'text-amber-700 dark:text-amber-300',
          trend: 'text-amber-600 dark:text-amber-400',
          icon: 'text-amber-500',
        };
      case 'critical':
        return {
          card: `${base} border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20`,
          value: 'text-red-700 dark:text-red-300',
          trend: 'text-red-600 dark:text-red-400',
          icon: 'text-red-500',
        };
      default:
        return {
          card: `${base} border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800`,
          value: 'text-gray-900 dark:text-white',
          trend: 'text-gray-600 dark:text-gray-400',
          icon: 'text-gray-500',
        };
    }
  }, [metric.status]);

  // トレンドアイコンと色
  const trendIcon = useMemo(() => {
    if (!metric.change || metric.change === 0) return null;

    const Icon = metric.trend === 'up' ? ArrowUpIcon : ArrowDownIcon;
    const colorClass = metric.trend === 'up' ? 'text-green-500' : 'text-red-500';

    return <Icon className={cn('h-4 w-4', colorClass)} />;
  }, [metric.change, metric.trend]);

  // 目標達成率
  const achievementRate = useMemo(() => {
    if (!metric.target || metric.target === 0) return null;
    return (displayValue / metric.target) * 100;
  }, [displayValue, metric.target]);

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        'relative rounded-lg border p-6 shadow-sm',
        statusStyles.card,
        className
      )}
    >
      {/* リアルタイム更新インジケーター */}
      {realTimeValue !== undefined && (
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="absolute top-2 right-2 w-2 h-2 bg-blue-400 rounded-full animate-pulse"
        />
      )}

      <div className="flex items-center justify-between">
        {/* タイトルとアイコン */}
        <div className="flex items-center space-x-2">
          <ChartBarIcon className={cn('h-5 w-5', statusStyles.icon)} />
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {metric.title}
          </h3>
        </div>

        {/* トレンドインジケーター */}
        {showTrend && trendIcon && (
          <div className="flex items-center space-x-1">
            {trendIcon}
            {metric.change !== undefined && (
              <span className={cn('text-sm font-medium', statusStyles.trend)}>
                {Math.abs(metric.change).toFixed(1)}
                {metric.unit === '%' ? 'pp' : metric.unit}
              </span>
            )}
          </div>
        )}
      </div>

      {/* メイン値 */}
      <div className="mt-4">
        <motion.div
          key={displayValue}
          initial={{ scale: 1.1 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.2 }}
          className={cn('text-3xl font-bold', statusStyles.value)}
        >
          {formattedValue}
          {metric.unit && metric.format !== 'currency' && metric.format !== 'percentage' && (
            <span className="text-xl font-normal ml-1">{metric.unit}</span>
          )}
        </motion.div>
      </div>

      {/* 目標達成率バー */}
      {showTarget && metric.target && achievementRate !== null && (
        <div className="mt-4">
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
            <span>目標: {formatNumber(metric.target)}{metric.unit}</span>
            <span>{Math.round(achievementRate)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(achievementRate, 100)}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
              className={cn(
                'h-2 rounded-full transition-colors duration-300',
                achievementRate >= 100 ? 'bg-green-500' :
                achievementRate >= 80 ? 'bg-blue-500' :
                achievementRate >= 60 ? 'bg-amber-500' : 'bg-red-500'
              )}
            />
          </div>
        </div>
      )}

      {/* 詳細情報（ホバー時） */}
      <div className="absolute inset-0 bg-gray-900 bg-opacity-90 rounded-lg opacity-0 hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
        <div className="text-center text-white p-4">
          <p className="text-sm font-medium">{metric.title}</p>
          <p className="text-2xl font-bold mt-1">{formattedValue}</p>
          {metric.change !== undefined && (
            <p className="text-sm mt-1">
              前回比: {metric.change > 0 ? '+' : ''}{metric.change.toFixed(2)}
              {metric.unit === '%' ? 'pp' : metric.unit}
            </p>
          )}
          {metric.target && (
            <p className="text-sm mt-1">
              目標: {formatNumber(metric.target)}{metric.unit}
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
};