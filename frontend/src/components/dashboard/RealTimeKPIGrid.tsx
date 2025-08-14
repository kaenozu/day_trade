import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUpIcon,
  TrendingDownIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  CpuChipIcon,
  BanknotesIcon,
} from '@heroicons/react/24/outline';

// Types
import type { KPIData, RealTimeKPIUpdate } from '@/types/dashboard';

// Utils
import { cn } from '@/utils/cn';
import { formatCurrency, formatPercentage, formatNumber } from '@/utils/format';

interface RealTimeKPIGridProps {
  data?: KPIData;
  realTimeUpdates?: RealTimeKPIUpdate;
  isConnected: boolean;
  className?: string;
}

interface KPICardProps {
  title: string;
  value: number;
  previousValue?: number;
  unit: string;
  format: 'currency' | 'percentage' | 'number';
  status: 'good' | 'warning' | 'critical';
  target?: number;
  icon: React.ComponentType<{ className?: string }>;
  realTimeValue?: number;
  isLive: boolean;
}

const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  previousValue,
  unit,
  format,
  status,
  target,
  icon: Icon,
  realTimeValue,
  isLive,
}) => {
  const [displayValue, setDisplayValue] = useState(value);
  const [isUpdating, setIsUpdating] = useState(false);

  // リアルタイム値の更新
  useEffect(() => {
    if (realTimeValue !== undefined && realTimeValue !== displayValue) {
      setIsUpdating(true);

      // カウントアップアニメーション
      const startValue = displayValue;
      const endValue = realTimeValue;
      const duration = 1000; // 1秒
      const startTime = Date.now();

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const currentValue = startValue + (endValue - startValue) * easeOutQuart;

        setDisplayValue(currentValue);

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          setIsUpdating(false);
        }
      };

      requestAnimationFrame(animate);
    }
  }, [realTimeValue, displayValue]);

  // 前回値との変化
  const change = useMemo(() => {
    if (previousValue === undefined) return null;
    return displayValue - previousValue;
  }, [displayValue, previousValue]);

  const changePercent = useMemo(() => {
    if (!change || !previousValue) return null;
    return (change / previousValue) * 100;
  }, [change, previousValue]);

  // フォーマット済み値
  const formattedValue = useMemo(() => {
    switch (format) {
      case 'currency':
        return formatCurrency(displayValue);
      case 'percentage':
        return formatPercentage(displayValue);
      default:
        return formatNumber(displayValue);
    }
  }, [displayValue, format]);

  // ステータス別スタイル
  const statusStyles = useMemo(() => {
    const base = "transition-all duration-300";

    switch (status) {
      case 'good':
        return {
          card: `${base} border-green-200 bg-gradient-to-br from-green-50 to-green-100 dark:border-green-800 dark:from-green-900/20 dark:to-green-800/20`,
          value: 'text-green-700 dark:text-green-300',
          icon: 'text-green-500',
          pulse: 'bg-green-400',
        };
      case 'warning':
        return {
          card: `${base} border-amber-200 bg-gradient-to-br from-amber-50 to-amber-100 dark:border-amber-800 dark:from-amber-900/20 dark:to-amber-800/20`,
          value: 'text-amber-700 dark:text-amber-300',
          icon: 'text-amber-500',
          pulse: 'bg-amber-400',
        };
      case 'critical':
        return {
          card: `${base} border-red-200 bg-gradient-to-br from-red-50 to-red-100 dark:border-red-800 dark:from-red-900/20 dark:to-red-800/20`,
          value: 'text-red-700 dark:text-red-300',
          icon: 'text-red-500',
          pulse: 'bg-red-400',
        };
      default:
        return {
          card: `${base} border-gray-200 bg-gradient-to-br from-white to-gray-50 dark:border-gray-700 dark:from-gray-800 dark:to-gray-700`,
          value: 'text-gray-900 dark:text-white',
          icon: 'text-gray-500',
          pulse: 'bg-blue-400',
        };
    }
  }, [status]);

  // 目標達成率
  const achievementRate = useMemo(() => {
    if (!target) return null;
    return Math.min((displayValue / target) * 100, 100);
  }, [displayValue, target]);

  return (
    <motion.div
      layout
      whileHover={{ scale: 1.02 }}
      className={cn(
        'relative rounded-xl border p-6 shadow-lg backdrop-blur-sm',
        statusStyles.card,
        isUpdating && 'ring-2 ring-blue-400 ring-opacity-50'
      )}
    >
      {/* リアルタイムインジケーター */}
      {isLive && (
        <div className="absolute top-3 right-3 flex items-center space-x-2">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className={cn('w-2 h-2 rounded-full', statusStyles.pulse)}
          />
          <span className="text-xs text-gray-500 dark:text-gray-400">LIVE</span>
        </div>
      )}

      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Icon className={cn('h-6 w-6', statusStyles.icon)} />
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {title}
          </h3>
        </div>
      </div>

      {/* メイン値 */}
      <div className="mb-4">
        <motion.div
          key={Math.floor(displayValue * 100)} // 小数点以下2桁での変化検出
          initial={{ scale: 1.1, opacity: 0.8 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.3 }}
          className={cn('text-3xl font-bold', statusStyles.value)}
        >
          {formattedValue}
          {unit && format === 'number' && (
            <span className="text-lg font-normal ml-1">{unit}</span>
          )}
        </motion.div>

        {/* 変化インジケーター */}
        {change !== null && changePercent !== null && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center mt-2 text-sm"
          >
            {change > 0 ? (
              <TrendingUpIcon className="h-4 w-4 text-green-500 mr-1" />
            ) : change < 0 ? (
              <TrendingDownIcon className="h-4 w-4 text-red-500 mr-1" />
            ) : null}

            <span className={cn(
              'font-medium',
              change > 0 ? 'text-green-600 dark:text-green-400' :
              change < 0 ? 'text-red-600 dark:text-red-400' :
              'text-gray-600 dark:text-gray-400'
            )}>
              {change > 0 ? '+' : ''}{formatNumber(change)}
              {unit && format === 'number' && unit}
              {format === 'percentage' && 'pp'}

              <span className="ml-1 text-gray-500">
                ({changePercent > 0 ? '+' : ''}{changePercent.toFixed(1)}%)
              </span>
            </span>
          </motion.div>
        )}
      </div>

      {/* 目標達成バー */}
      {target && achievementRate !== null && (
        <div className="mb-2">
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
            <span>目標: {formatNumber(target)}{unit}</span>
            <span>{Math.round(achievementRate)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${achievementRate}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
              className={cn(
                'h-2 rounded-full transition-all duration-500',
                achievementRate >= 100 ? 'bg-green-500' :
                achievementRate >= 80 ? 'bg-blue-500' :
                achievementRate >= 60 ? 'bg-amber-500' : 'bg-red-500'
              )}
            />
          </div>
        </div>
      )}

      {/* 更新アニメーション */}
      <AnimatePresence>
        {isUpdating && (
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-90 dark:bg-gray-800 dark:bg-opacity-90 rounded-xl"
          >
            <div className="flex items-center space-x-2 text-blue-600 dark:text-blue-400">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="w-5 h-5 border-2 border-current border-t-transparent rounded-full"
              />
              <span className="text-sm font-medium">更新中...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export const RealTimeKPIGrid: React.FC<RealTimeKPIGridProps> = ({
  data,
  realTimeUpdates,
  isConnected,
  className,
}) => {
  if (!data) {
    return (
      <div className={cn('grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6', className)}>
        {Array.from({ length: 4 }, (_, i) => (
          <div
            key={i}
            className="h-40 rounded-xl bg-gray-200 dark:bg-gray-700 animate-pulse"
          />
        ))}
      </div>
    );
  }

  const kpiCards = [
    {
      title: 'Today\'s ROI',
      value: data.roi.current,
      previousValue: data.roi.previous,
      unit: '%',
      format: 'percentage' as const,
      status: data.roi.current >= 5 ? 'good' as const :
              data.roi.current >= 3 ? 'warning' as const : 'critical' as const,
      target: 5.0,
      icon: TrendingUpIcon,
      realTimeValue: realTimeUpdates?.roi,
    },
    {
      title: 'Executed Trades',
      value: data.trades.executed,
      previousValue: data.trades.previous,
      unit: '件',
      format: 'number' as const,
      status: data.trades.executed >= 100 ? 'good' as const :
              data.trades.executed >= 50 ? 'warning' as const : 'critical' as const,
      target: 100,
      icon: ChartBarIcon,
      realTimeValue: realTimeUpdates?.trades,
    },
    {
      title: 'ML Accuracy',
      value: data.accuracy.current,
      previousValue: data.accuracy.previous,
      unit: '%',
      format: 'percentage' as const,
      status: data.accuracy.current >= 93 ? 'good' as const :
              data.accuracy.current >= 92 ? 'warning' as const : 'critical' as const,
      target: 93.0,
      icon: CpuChipIcon,
      realTimeValue: realTimeUpdates?.accuracy,
    },
    {
      title: 'Portfolio Value',
      value: data.portfolio.totalValue,
      previousValue: data.portfolio.previousValue,
      unit: '',
      format: 'currency' as const,
      status: data.portfolio.dailyChange >= 0 ? 'good' as const : 'critical' as const,
      icon: BanknotesIcon,
      realTimeValue: realTimeUpdates?.portfolioValue,
    },
  ];

  return (
    <div className={cn('grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6', className)}>
      {kpiCards.map((kpi, index) => (
        <motion.div
          key={kpi.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
        >
          <KPICard
            {...kpi}
            isLive={isConnected}
          />
        </motion.div>
      ))}
    </div>
  );
};