"""
Performance monitor module tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import psutil
import asyncio


class MockPerformanceMonitor:
    """Mock performance monitor for testing"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'response_time': 1.0,
            'error_rate': 0.05
        }
        self.monitoring_active = False
        self.start_time = None
    
    def start_monitoring(self) -> bool:
        """Start performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.start_time = datetime.now()
            return True
        return False
    
    def stop_monitoring(self) -> bool:
        """Stop performance monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            return True
        return False
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if not self.monitoring_active:
            return 0.0
        # Mock CPU usage between 10-90%
        import random
        return random.uniform(10.0, 90.0)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if not self.monitoring_active:
            return {'used': 0.0, 'available': 0.0, 'percent': 0.0}
        
        # Mock memory data
        total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        used_memory = total_memory * 0.6  # 60% used
        available_memory = total_memory - used_memory
        
        return {
            'used': used_memory,
            'available': available_memory,
            'percent': (used_memory / total_memory) * 100,
            'total': total_memory
        }
    
    def record_response_time(self, operation: str, duration: float) -> None:
        """Record operation response time"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                'response_times': [],
                'total_calls': 0,
                'error_count': 0
            }
        
        self.metrics[operation]['response_times'].append(duration)
        self.metrics[operation]['total_calls'] += 1
        
        # Check threshold
        if duration > self.thresholds['response_time']:
            self.alerts.append({
                'type': 'response_time',
                'operation': operation,
                'value': duration,
                'threshold': self.thresholds['response_time'],
                'timestamp': datetime.now()
            })
    
    def record_error(self, operation: str) -> None:
        """Record operation error"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                'response_times': [],
                'total_calls': 0,
                'error_count': 0
            }
        
        self.metrics[operation]['error_count'] += 1
        
        # Calculate error rate
        error_rate = (self.metrics[operation]['error_count'] / 
                     max(self.metrics[operation]['total_calls'], 1))
        
        if error_rate > self.thresholds['error_rate']:
            self.alerts.append({
                'type': 'error_rate',
                'operation': operation,
                'value': error_rate,
                'threshold': self.thresholds['error_rate'],
                'timestamp': datetime.now()
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'monitoring_active': self.monitoring_active,
            'uptime': self._get_uptime(),
            'system': {
                'cpu_usage': self.get_cpu_usage(),
                'memory': self.get_memory_usage()
            },
            'operations': {},
            'alerts': len(self.alerts)
        }
        
        for operation, metrics in self.metrics.items():
            if metrics['response_times']:
                avg_response = sum(metrics['response_times']) / len(metrics['response_times'])
                max_response = max(metrics['response_times'])
                min_response = min(metrics['response_times'])
            else:
                avg_response = max_response = min_response = 0.0
            
            error_rate = (metrics['error_count'] / max(metrics['total_calls'], 1))
            
            summary['operations'][operation] = {
                'total_calls': metrics['total_calls'],
                'error_count': metrics['error_count'],
                'error_rate': error_rate,
                'avg_response_time': avg_response,
                'max_response_time': max_response,
                'min_response_time': min_response
            }
        
        return summary
    
    def _get_uptime(self) -> float:
        """Calculate monitoring uptime in seconds"""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()


class MockSystemMetrics:
    """Mock system metrics collector"""
    
    def __init__(self):
        self.baseline_metrics = None
        self.current_metrics = {}
    
    def capture_baseline(self) -> Dict[str, Any]:
        """Capture baseline system metrics"""
        self.baseline_metrics = {
            'cpu_count': 8,
            'memory_total': 16 * 1024 * 1024 * 1024,
            'disk_total': 512 * 1024 * 1024 * 1024,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'disk_io': {'read_bytes': 0, 'write_bytes': 0},
            'timestamp': datetime.now()
        }
        return self.baseline_metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self.baseline_metrics:
            self.capture_baseline()
        
        # Mock current metrics with some variance
        import random
        
        self.current_metrics = {
            'cpu_percent': random.uniform(20.0, 80.0),
            'memory_percent': random.uniform(40.0, 85.0),
            'disk_percent': random.uniform(30.0, 70.0),
            'network_io': {
                'bytes_sent': self.baseline_metrics['network_io']['bytes_sent'] + random.randint(1000, 10000),
                'bytes_recv': self.baseline_metrics['network_io']['bytes_recv'] + random.randint(1000, 10000)
            },
            'disk_io': {
                'read_bytes': self.baseline_metrics['disk_io']['read_bytes'] + random.randint(100000, 1000000),
                'write_bytes': self.baseline_metrics['disk_io']['write_bytes'] + random.randint(100000, 1000000)
            },
            'timestamp': datetime.now()
        }
        
        return self.current_metrics
    
    def get_metrics_delta(self) -> Dict[str, Any]:
        """Calculate metrics delta from baseline"""
        current = self.get_current_metrics()
        
        if not self.baseline_metrics:
            return current
        
        return {
            'cpu_change': current['cpu_percent'],
            'memory_change': current['memory_percent'],
            'disk_change': current['disk_percent'],
            'network_delta': {
                'bytes_sent': current['network_io']['bytes_sent'] - self.baseline_metrics['network_io']['bytes_sent'],
                'bytes_recv': current['network_io']['bytes_recv'] - self.baseline_metrics['network_io']['bytes_recv']
            },
            'disk_delta': {
                'read_bytes': current['disk_io']['read_bytes'] - self.baseline_metrics['disk_io']['read_bytes'],
                'write_bytes': current['disk_io']['write_bytes'] - self.baseline_metrics['disk_io']['write_bytes']
            },
            'time_elapsed': (current['timestamp'] - self.baseline_metrics['timestamp']).total_seconds()
        }


class MockPerformanceBenchmark:
    """Mock performance benchmark for testing"""
    
    def __init__(self):
        self.benchmark_results = {}
        self.test_scenarios = {}
    
    def run_cpu_benchmark(self, duration: int = 5) -> Dict[str, float]:
        """Run CPU performance benchmark"""
        start_time = time.time()
        
        # Simulate CPU-intensive work
        operations = 0
        target_time = start_time + duration
        
        while time.time() < target_time:
            # Mock CPU work
            _ = sum(i * i for i in range(1000))
            operations += 1000
        
        actual_duration = time.time() - start_time
        ops_per_second = operations / actual_duration
        
        result = {
            'duration': actual_duration,
            'operations': operations,
            'ops_per_second': ops_per_second,
            'cpu_score': min(ops_per_second / 100000, 100.0)  # Normalized score
        }
        
        self.benchmark_results['cpu'] = result
        return result
    
    def run_memory_benchmark(self, data_size_mb: int = 100) -> Dict[str, float]:
        """Run memory performance benchmark"""
        start_time = time.time()
        
        # Simulate memory operations
        data_size = data_size_mb * 1024 * 1024
        test_data = bytearray(data_size)
        
        # Memory write test
        write_start = time.time()
        for i in range(0, data_size, 1024):
            test_data[i:i+1024] = b'x' * 1024
        write_time = time.time() - write_start
        
        # Memory read test
        read_start = time.time()
        total_bytes = 0
        for i in range(0, data_size, 1024):
            total_bytes += len(test_data[i:i+1024])
        read_time = time.time() - read_start
        
        total_time = time.time() - start_time
        
        result = {
            'data_size_mb': data_size_mb,
            'write_time': write_time,
            'read_time': read_time,
            'total_time': total_time,
            'write_speed_mb_s': data_size_mb / write_time if write_time > 0 else 0,
            'read_speed_mb_s': data_size_mb / read_time if read_time > 0 else 0,
            'memory_score': min((data_size_mb / total_time) / 10, 100.0)
        }
        
        self.benchmark_results['memory'] = result
        return result
    
    def run_disk_benchmark(self, file_size_mb: int = 50) -> Dict[str, float]:
        """Run disk I/O performance benchmark"""
        import tempfile
        import os
        
        start_time = time.time()
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Write test
            write_start = time.time()
            test_data = b'x' * (1024 * 1024)  # 1MB chunks
            for _ in range(file_size_mb):
                temp_file.write(test_data)
            temp_file.flush()
            write_time = time.time() - write_start
            
            # Read test
            read_start = time.time()
            temp_file.seek(0)
            total_read = 0
            while True:
                chunk = temp_file.read(1024 * 1024)
                if not chunk:
                    break
                total_read += len(chunk)
            read_time = time.time() - read_start
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass
        
        total_time = time.time() - start_time
        
        result = {
            'file_size_mb': file_size_mb,
            'write_time': write_time,
            'read_time': read_time,
            'total_time': total_time,
            'write_speed_mb_s': file_size_mb / write_time if write_time > 0 else 0,
            'read_speed_mb_s': file_size_mb / read_time if read_time > 0 else 0,
            'disk_score': min((file_size_mb * 2 / total_time) / 5, 100.0)
        }
        
        self.benchmark_results['disk'] = result
        return result
    
    def get_overall_score(self) -> Dict[str, float]:
        """Calculate overall performance score"""
        if not self.benchmark_results:
            return {'overall_score': 0.0, 'components': {}}
        
        component_scores = {}
        total_score = 0.0
        component_count = 0
        
        for component, result in self.benchmark_results.items():
            score_key = f"{component}_score"
            if score_key in result:
                component_scores[component] = result[score_key]
                total_score += result[score_key]
                component_count += 1
        
        overall_score = total_score / component_count if component_count > 0 else 0.0
        
        return {
            'overall_score': overall_score,
            'components': component_scores,
            'benchmark_count': component_count
        }


class TestPerformanceMonitor:
    """Test performance monitor functionality"""
    
    def test_monitor_lifecycle(self):
        monitor = MockPerformanceMonitor()
        
        # Initially not monitoring
        assert monitor.monitoring_active == False
        
        # Start monitoring
        assert monitor.start_monitoring() == True
        assert monitor.monitoring_active == True
        
        # Cannot start again
        assert monitor.start_monitoring() == False
        
        # Stop monitoring
        assert monitor.stop_monitoring() == True
        assert monitor.monitoring_active == False
        
        # Cannot stop again
        assert monitor.stop_monitoring() == False
    
    def test_cpu_monitoring(self):
        monitor = MockPerformanceMonitor()
        monitor.start_monitoring()
        
        cpu_usage = monitor.get_cpu_usage()
        assert isinstance(cpu_usage, float)
        assert 0.0 <= cpu_usage <= 100.0
        
        # Should return 0 when not monitoring
        monitor.stop_monitoring()
        assert monitor.get_cpu_usage() == 0.0
    
    def test_memory_monitoring(self):
        monitor = MockPerformanceMonitor()
        monitor.start_monitoring()
        
        memory_info = monitor.get_memory_usage()
        
        assert 'used' in memory_info
        assert 'available' in memory_info
        assert 'percent' in memory_info
        assert 'total' in memory_info
        
        assert memory_info['used'] + memory_info['available'] == memory_info['total']
        assert 0.0 <= memory_info['percent'] <= 100.0
    
    def test_response_time_recording(self):
        monitor = MockPerformanceMonitor()
        
        # Record some response times
        monitor.record_response_time("trade_execution", 0.5)
        monitor.record_response_time("trade_execution", 0.8)
        monitor.record_response_time("data_fetch", 1.2)
        
        assert "trade_execution" in monitor.metrics
        assert "data_fetch" in monitor.metrics
        
        trade_metrics = monitor.metrics["trade_execution"]
        assert trade_metrics['total_calls'] == 2
        assert len(trade_metrics['response_times']) == 2
        
        # Check if slow response generated alert
        data_metrics = monitor.metrics["data_fetch"]
        assert any(alert['type'] == 'response_time' and alert['operation'] == 'data_fetch' 
                  for alert in monitor.alerts)
    
    def test_error_recording(self):
        monitor = MockPerformanceMonitor()
        
        # Record some operations and errors
        monitor.record_response_time("api_call", 0.3)
        monitor.record_response_time("api_call", 0.4)
        monitor.record_error("api_call")
        monitor.record_error("api_call")
        
        api_metrics = monitor.metrics["api_call"]
        assert api_metrics['total_calls'] == 2
        assert api_metrics['error_count'] == 2
        
        # Error rate should be 100% (2 errors / 2 calls)
        error_rate = api_metrics['error_count'] / api_metrics['total_calls']
        assert error_rate > monitor.thresholds['error_rate']
        
        # Should have generated error rate alert
        assert any(alert['type'] == 'error_rate' and alert['operation'] == 'api_call'
                  for alert in monitor.alerts)
    
    def test_performance_summary(self):
        monitor = MockPerformanceMonitor()
        monitor.start_monitoring()
        
        # Add some metrics
        monitor.record_response_time("operation1", 0.3)
        monitor.record_response_time("operation1", 0.7)
        monitor.record_error("operation1")
        
        summary = monitor.get_performance_summary()
        
        assert summary['monitoring_active'] == True
        assert 'uptime' in summary
        assert 'system' in summary
        assert 'operations' in summary
        assert 'alerts' in summary
        
        # Check operation metrics
        op1_metrics = summary['operations']['operation1']
        assert op1_metrics['total_calls'] == 2
        assert op1_metrics['error_count'] == 1
        assert op1_metrics['error_rate'] == 0.5
        assert op1_metrics['avg_response_time'] == 0.5


class TestSystemMetrics:
    """Test system metrics collection"""
    
    def test_baseline_capture(self):
        metrics = MockSystemMetrics()
        
        baseline = metrics.capture_baseline()
        
        assert 'cpu_count' in baseline
        assert 'memory_total' in baseline
        assert 'disk_total' in baseline
        assert 'network_io' in baseline
        assert 'disk_io' in baseline
        assert 'timestamp' in baseline
        
        assert metrics.baseline_metrics is not None
    
    def test_current_metrics(self):
        metrics = MockSystemMetrics()
        
        current = metrics.get_current_metrics()
        
        assert 'cpu_percent' in current
        assert 'memory_percent' in current
        assert 'disk_percent' in current
        assert 'network_io' in current
        assert 'disk_io' in current
        assert 'timestamp' in current
        
        # Values should be within reasonable ranges
        assert 0 <= current['cpu_percent'] <= 100
        assert 0 <= current['memory_percent'] <= 100
        assert 0 <= current['disk_percent'] <= 100
    
    def test_metrics_delta(self):
        metrics = MockSystemMetrics()
        
        # Capture baseline first
        metrics.capture_baseline()
        time.sleep(0.1)  # Small delay to ensure timestamp difference
        
        delta = metrics.get_metrics_delta()
        
        assert 'cpu_change' in delta
        assert 'memory_change' in delta
        assert 'disk_change' in delta
        assert 'network_delta' in delta
        assert 'disk_delta' in delta
        assert 'time_elapsed' in delta
        
        assert delta['time_elapsed'] > 0


class TestPerformanceBenchmark:
    """Test performance benchmark functionality"""
    
    def test_cpu_benchmark(self):
        benchmark = MockPerformanceBenchmark()
        
        result = benchmark.run_cpu_benchmark(duration=1)
        
        assert 'duration' in result
        assert 'operations' in result
        assert 'ops_per_second' in result
        assert 'cpu_score' in result
        
        assert result['duration'] > 0
        assert result['operations'] > 0
        assert result['ops_per_second'] > 0
        assert 0 <= result['cpu_score'] <= 100
        
        # Should be stored in benchmark results
        assert 'cpu' in benchmark.benchmark_results
    
    def test_memory_benchmark(self):
        benchmark = MockPerformanceBenchmark()
        
        result = benchmark.run_memory_benchmark(data_size_mb=10)
        
        assert 'data_size_mb' in result
        assert 'write_time' in result
        assert 'read_time' in result
        assert 'total_time' in result
        assert 'write_speed_mb_s' in result
        assert 'read_speed_mb_s' in result
        assert 'memory_score' in result
        
        assert result['data_size_mb'] == 10
        assert result['write_time'] > 0
        assert result['read_time'] > 0
        assert result['write_speed_mb_s'] > 0
        assert result['read_speed_mb_s'] > 0
    
    def test_disk_benchmark(self):
        benchmark = MockPerformanceBenchmark()
        
        result = benchmark.run_disk_benchmark(file_size_mb=5)
        
        assert 'file_size_mb' in result
        assert 'write_time' in result
        assert 'read_time' in result
        assert 'total_time' in result
        assert 'write_speed_mb_s' in result
        assert 'read_speed_mb_s' in result
        assert 'disk_score' in result
        
        assert result['file_size_mb'] == 5
        assert result['write_time'] > 0
        assert result['read_time'] > 0
    
    def test_overall_score(self):
        benchmark = MockPerformanceBenchmark()
        
        # No benchmarks run yet
        score = benchmark.get_overall_score()
        assert score['overall_score'] == 0.0
        assert score['benchmark_count'] == 0
        
        # Run some benchmarks
        benchmark.run_cpu_benchmark(duration=1)
        benchmark.run_memory_benchmark(data_size_mb=10)
        
        score = benchmark.get_overall_score()
        assert score['overall_score'] > 0
        assert score['benchmark_count'] == 2
        assert 'cpu' in score['components']
        assert 'memory' in score['components']


class TestPerformanceIntegration:
    """Test performance monitoring integration scenarios"""
    
    def test_monitoring_with_benchmarks(self):
        """Test integration of monitoring and benchmarking"""
        monitor = MockPerformanceMonitor()
        benchmark = MockPerformanceBenchmark()
        
        monitor.start_monitoring()
        
        # Run benchmark while monitoring
        start_time = time.time()
        cpu_result = benchmark.run_cpu_benchmark(duration=1)
        end_time = time.time()
        
        # Record benchmark as operation
        monitor.record_response_time("cpu_benchmark", end_time - start_time)
        
        summary = monitor.get_performance_summary()
        
        assert "cpu_benchmark" in summary['operations']
        assert summary['operations']['cpu_benchmark']['total_calls'] == 1
        assert cpu_result['cpu_score'] > 0
    
    def test_performance_alerting(self):
        """Test performance alert generation"""
        monitor = MockPerformanceMonitor()
        
        # Set very low thresholds for testing
        monitor.thresholds['response_time'] = 0.1
        monitor.thresholds['error_rate'] = 0.1
        
        # Generate slow responses
        monitor.record_response_time("slow_operation", 2.0)
        monitor.record_response_time("slow_operation", 1.5)
        
        # Generate errors
        monitor.record_error("error_operation")
        monitor.record_error("error_operation")
        monitor.record_response_time("error_operation", 0.05)  # One success
        
        # Check alerts were generated
        assert len(monitor.alerts) > 0
        
        response_alerts = [a for a in monitor.alerts if a['type'] == 'response_time']
        error_alerts = [a for a in monitor.alerts if a['type'] == 'error_rate']
        
        assert len(response_alerts) > 0
        assert len(error_alerts) > 0
    
    def test_concurrent_monitoring(self):
        """Test concurrent performance monitoring"""
        monitor = MockPerformanceMonitor()
        monitor.start_monitoring()
        
        def worker_task(task_id: int):
            """Worker function for concurrent testing"""
            for i in range(5):
                start_time = time.time()
                time.sleep(0.01)  # Simulate work
                end_time = time.time()
                
                monitor.record_response_time(f"worker_{task_id}", end_time - start_time)
                
                if i == 2:  # Generate one error per worker
                    monitor.record_error(f"worker_{task_id}")
        
        # Run multiple workers concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        summary = monitor.get_performance_summary()
        
        # Should have metrics for all workers
        for i in range(3):
            worker_name = f"worker_{i}"
            assert worker_name in summary['operations']
            assert summary['operations'][worker_name]['total_calls'] == 5
            assert summary['operations'][worker_name]['error_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])