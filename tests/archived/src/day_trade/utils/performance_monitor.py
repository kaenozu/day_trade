#!/usr/bin/env python3
"""
Performance Monitoring System (Simple Version)
"""

import time
from contextlib import contextmanager
from datetime import datetime
from typing import List
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance Metrics"""
    timestamp: datetime
    process_name: str
    execution_time: float
    success: bool

class PerformanceMonitor:
    """Performance Monitoring System"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.baseline_metrics = {
            "ml_analysis_85_stocks": 3.6,
            "data_fetch_85_stocks": 2.0,
            "portfolio_optimization": 1.0,
        }

    @contextmanager
    def monitor(self, process_name: str):
        """Monitoring Context"""
        start_time = time.time()
        success = True

        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            execution_time = time.time() - start_time

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                process_name=process_name,
                execution_time=execution_time,
                success=success
            )

            self.metrics_history.append(metrics)
            status = "SUCCESS" if success else "FAILED"
            print(f"Monitoring Complete: {process_name} - {execution_time:.3f}s ({status})")

# Global instance
global_monitor = PerformanceMonitor()

def get_performance_monitor():
    return global_monitor
