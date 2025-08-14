// Dashboard Types
// Day Trade ML System - Frontend Types

export interface KPIMetric {
  id: string;
  title: string;
  value: number;
  unit: string;
  change?: number;
  trend?: 'up' | 'down' | 'flat';
  status: 'good' | 'warning' | 'critical';
  target?: number;
  format?: 'number' | 'currency' | 'percentage';
  description?: string;
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string;
  borderWidth?: number;
  fill?: boolean;
  tension?: number;
}

export interface ChartOptions {
  responsive?: boolean;
  maintainAspectRatio?: boolean;
  plugins?: {
    legend?: {
      display?: boolean;
      position?: 'top' | 'bottom' | 'left' | 'right';
    };
    tooltip?: {
      mode?: 'index' | 'nearest' | 'point';
      intersect?: boolean;
    };
  };
  scales?: {
    x?: {
      beginAtZero?: boolean;
      ticks?: {
        callback?: (value: any) => string;
      };
    };
    y?: {
      beginAtZero?: boolean;
      ticks?: {
        callback?: (value: any) => string;
        precision?: number;
      };
    };
  };
}

export interface TradeData {
  id: string;
  timestamp: Date;
  symbol: string;
  action: 'buy' | 'sell';
  price: number;
  quantity: number;
  status: 'pending' | 'executed' | 'failed' | 'cancelled';
  profit?: number;
  confidence?: number;
  strategy?: string;
  executionTime?: number; // ms
}

export interface MLData {
  accuracy: number;
  accuracyChange: number;
  confidence: number;
  predictions: {
    total: number;
    correct: number;
    wrong: number;
  };
  models: {
    name: string;
    accuracy: number;
    lastUpdated: Date;
    status: 'active' | 'training' | 'disabled';
  }[];
  performance: {
    timestamp: Date;
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  }[];
}

export interface MarketData {
  indices: {
    name: string;
    value: number;
    change: number;
    changePercent: number;
  }[];
  topGainers: {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
  }[];
  topLosers: {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
  }[];
  volatility: {
    vix: number;
    change: number;
  };
  volume: {
    total: number;
    change: number;
  };
}

export interface PortfolioData {
  totalValue: number;
  dailyChange: number;
  dailyChangePercent: number;
  cash: number;
  positions: {
    symbol: string;
    quantity: number;
    avgPrice: number;
    currentPrice: number;
    unrealizedPnL: number;
    unrealizedPnLPercent: number;
    marketValue: number;
  }[];
  allocation: {
    sector: string;
    percentage: number;
    value: number;
  }[];
}

export interface AlertData {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: Date;
  source: string;
  acknowledged: boolean;
  actions?: {
    label: string;
    action: string;
  }[];
}

export interface SystemStatus {
  services: {
    name: string;
    status: 'healthy' | 'warning' | 'critical' | 'unknown';
    uptime: number;
    lastCheck: Date;
    responseTime?: number;
  }[];
  resources: {
    cpu: {
      usage: number;
      cores: number;
    };
    memory: {
      used: number;
      total: number;
      usage: number;
    };
    disk: {
      used: number;
      total: number;
      usage: number;
    };
  };
  network: {
    latency: number;
    throughput: number;
    errors: number;
  };
}

export interface QuickStatsData {
  label: string;
  value: number | string;
  icon?: string;
  color?: string;
  trend?: 'up' | 'down' | 'flat';
  change?: number;
}

export interface DashboardData {
  roi: {
    daily: number;
    weekly: number;
    monthly: number;
    yearly: number;
    change: number;
  };
  trades: {
    executed: number;
    pending: number;
    failed: number;
    change: number;
    recent: TradeData[];
  };
  ml: MLData;
  portfolio: PortfolioData;
  market: MarketData;
  alerts: AlertData[];
  systemStatus: SystemStatus;
  charts: {
    roiTrend: ChartData;
    tradingVolume: ChartData;
    mlAccuracy: ChartData;
    portfolioValue: ChartData;
  };
  quickStats: QuickStatsData[];
  lastUpdated: Date;
}

// Real-time WebSocket data types
export interface RealTimeData {
  [key: string]: any;
  latestTrade?: TradeData;
  ml?: Partial<MLData>;
  market?: Partial<MarketData>;
  portfolio?: Partial<PortfolioData>;
  alerts?: AlertData[];
  roi?: number;
  trades?: number;
  accuracy?: number;
}

// API Response types
export interface APIResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: Date;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// Filter and search types
export interface DashboardFilters {
  timeRange: '1h' | '24h' | '7d' | '30d' | '90d' | 'ytd' | '1y';
  symbols?: string[];
  strategies?: string[];
  status?: ('executed' | 'pending' | 'failed')[];
}

export interface SearchParams {
  query?: string;
  filters?: DashboardFilters;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  page?: number;
  pageSize?: number;
}

// Widget configuration types
export interface DashboardWidget {
  id: string;
  type: 'kpi' | 'chart' | 'table' | 'alert' | 'custom';
  title: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  config: {
    [key: string]: any;
  };
  permissions?: string[];
}

export interface DashboardLayout {
  id: string;
  name: string;
  description?: string;
  widgets: DashboardWidget[];
  isDefault: boolean;
  isPublic: boolean;
  createdBy: string;
  createdAt: Date;
  updatedAt: Date;
}

// User preferences
export interface DashboardPreferences {
  defaultLayout: string;
  theme: 'light' | 'dark' | 'auto';
  refreshInterval: number; // seconds
  notifications: {
    enabled: boolean;
    sounds: boolean;
    email: boolean;
    push: boolean;
    severityThreshold: 'low' | 'medium' | 'high' | 'critical';
  };
  charts: {
    defaultType: 'line' | 'bar' | 'area' | 'candle';
    showGrid: boolean;
    showTooltips: boolean;
    colorScheme: 'default' | 'colorBlind' | 'highContrast';
  };
}

// Error types
export interface DashboardError {
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
  recoverable: boolean;
}

export interface ServiceError {
  service: string;
  error: DashboardError;
  impact: 'low' | 'medium' | 'high' | 'critical';
  estimated_recovery?: Date;
}