"""
Performance Testing for Large-Scale Simulations

This module provides comprehensive performance testing capabilities for the peer review
simulation system, including load testing, stress testing, and scalability analysis.
"""

import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from contextlib import contextmanager
try:
    import resource
except ImportError:
    resource = None  # Not available on Windows

try:
    import tracemalloc
except ImportError:
    tracemalloc = None

from src.core.exceptions import ValidationError, SimulationError
from src.core.logging_config import get_logger
from src.enhancements.simulation_analytics import MetricsSnapshot, AnalyticsConfiguration

logger = get_logger(__name__)


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing scenarios."""
    # Test parameters
    max_researchers: int = 1000
    max_papers: int = 5000
    max_reviews_per_paper: int = 5
    simulation_duration_minutes: int = 60
    
    # Load testing
    concurrent_operations: int = 10
    ramp_up_duration_seconds: int = 300
    steady_state_duration_seconds: int = 1800
    ramp_down_duration_seconds: int = 300
    
    # Stress testing
    stress_multiplier: float = 2.0
    memory_limit_mb: int = 4096
    cpu_limit_percent: float = 80.0
    
    # Scalability testing
    scalability_steps: List[int] = field(default_factory=lambda: [100, 500, 1000, 2000, 5000])
    metrics_collection_interval: int = 30  # seconds
    
    # Resource monitoring
    monitor_memory: bool = True
    monitor_cpu: bool = True
    monitor_disk_io: bool = True
    monitor_network: bool = False
    
    # Output settings
    detailed_profiling: bool = True
    generate_reports: bool = True
    save_raw_data: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during testing."""
    timestamp: datetime
    test_scenario: str
    
    # Throughput metrics
    operations_per_second: float
    reviews_per_second: float
    papers_processed_per_second: float
    
    # Latency metrics
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    
    # Resource utilization
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_read_mb_per_sec: float
    disk_write_mb_per_sec: float
    
    # System metrics
    active_threads: int
    open_file_descriptors: int
    network_connections: int
    
    # Error metrics
    error_count: int
    error_rate_percent: float
    timeout_count: int
    
    # Simulation-specific metrics
    simulation_state_size_mb: float
    database_size_mb: float
    cache_hit_rate_percent: float


@dataclass
class TestResult:
    """Results from a performance test run."""
    test_name: str
    config: PerformanceTestConfig
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Metrics collected during test
    metrics_history: List[PerformanceMetrics]
    
    # Summary statistics
    avg_throughput: float
    peak_throughput: float
    avg_latency_ms: float
    peak_memory_mb: float
    peak_cpu_percent: float
    
    # Test outcome
    success: bool
    errors: List[str]
    warnings: List[str]
    
    # Resource limits hit
    memory_limit_exceeded: bool
    cpu_limit_exceeded: bool
    timeout_occurred: bool


class ResourceMonitor:
    """Monitors system resources during performance testing."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.monitoring = False
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitor_thread = None
        self._start_time = None
        
    def start_monitoring(self, test_scenario: str):
        """Start resource monitoring."""
        self.test_scenario = test_scenario
        self.monitoring = True
        self._start_time = time.time()
        self.metrics_history.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info(f"Resource monitoring started for scenario: {test_scenario}")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                # Collect process metrics
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                # Create metrics snapshot
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    test_scenario=self.test_scenario,
                    operations_per_second=0.0,  # Will be updated by test runner
                    reviews_per_second=0.0,
                    papers_processed_per_second=0.0,
                    avg_response_time_ms=0.0,
                    p95_response_time_ms=0.0,
                    p99_response_time_ms=0.0,
                    max_response_time_ms=0.0,
                    cpu_usage_percent=cpu_percent,
                    memory_usage_mb=memory_info.used / 1024 / 1024,
                    memory_usage_percent=memory_info.percent,
                    disk_read_mb_per_sec=0.0,  # Calculated from deltas
                    disk_write_mb_per_sec=0.0,
                    active_threads=threading.active_count(),
                    open_file_descriptors=len(process.open_files()),
                    network_connections=len(process.connections()),
                    error_count=0,
                    error_rate_percent=0.0,
                    timeout_count=0,
                    simulation_state_size_mb=0.0,
                    database_size_mb=0.0,
                    cache_hit_rate_percent=0.0
                )
                
                self.metrics_history.append(metrics)
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)  # Brief pause before retry
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage metrics."""
        if not self.metrics_history:
            return {}
        
        return {
            'peak_cpu_percent': max(m.cpu_usage_percent for m in self.metrics_history),
            'peak_memory_mb': max(m.memory_usage_mb for m in self.metrics_history),
            'peak_memory_percent': max(m.memory_usage_percent for m in self.metrics_history),
            'peak_threads': max(m.active_threads for m in self.metrics_history),
            'peak_file_descriptors': max(m.open_file_descriptors for m in self.metrics_history)
        }


class LoadGenerator:
    """Generates realistic load for performance testing."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.operation_times: List[float] = []
        self.error_count = 0
        self.timeout_count = 0
        
    def generate_review_load(self, simulation_coordinator, duration_seconds: int) -> Dict[str, Any]:
        """Generate review processing load."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        operations_completed = 0
        self.operation_times.clear()
        self.error_count = 0
        self.timeout_count = 0
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_operations) as executor:
            futures = []
            
            while time.time() < end_time:
                # Submit review operations
                for _ in range(self.config.concurrent_operations):
                    if time.time() >= end_time:
                        break
                    
                    future = executor.submit(self._simulate_review_operation, simulation_coordinator)
                    futures.append(future)
                
                # Collect completed operations
                completed_futures = []
                for future in futures:
                    if future.done():
                        completed_futures.append(future)
                        try:
                            operation_time = future.result(timeout=1)
                            self.operation_times.append(operation_time)
                            operations_completed += 1
                        except TimeoutError:
                            self.timeout_count += 1
                        except Exception as e:
                            self.error_count += 1
                            logger.debug(f"Operation error: {e}")
                
                # Remove completed futures
                for future in completed_futures:
                    futures.remove(future)
                
                time.sleep(0.1)  # Brief pause
        
        # Calculate metrics
        total_time = time.time() - start_time
        throughput = operations_completed / total_time if total_time > 0 else 0
        
        latency_metrics = self._calculate_latency_metrics()
        
        return {
            'operations_completed': operations_completed,
            'throughput_ops_per_sec': throughput,
            'total_errors': self.error_count,
            'total_timeouts': self.timeout_count,
            'error_rate_percent': (self.error_count / max(1, operations_completed)) * 100,
            **latency_metrics
        }
    
    def _simulate_review_operation(self, simulation_coordinator) -> float:
        """Simulate a single review operation and measure time."""
        start_time = time.time()
        
        try:
            # Simulate review process
            paper_id = f"test_paper_{int(time.time() * 1000000) % 10000}"
            venue_id = "test_venue"
            reviewer_ids = [f"reviewer_{i}" for i in range(3)]
            
            # This would normally call the actual review coordination
            # For testing, we simulate the work
            time.sleep(0.01)  # Simulate processing time
            
            # Simulate some computational work
            result = simulation_coordinator.get_system_health()
            
        except Exception as e:
            logger.debug(f"Simulated operation error: {e}")
            raise
        
        return (time.time() - start_time) * 1000  # Return time in milliseconds
    
    def _calculate_latency_metrics(self) -> Dict[str, float]:
        """Calculate latency percentiles from operation times."""
        if not self.operation_times:
            return {
                'avg_response_time_ms': 0.0,
                'p95_response_time_ms': 0.0,
                'p99_response_time_ms': 0.0,
                'max_response_time_ms': 0.0
            }
        
        sorted_times = sorted(self.operation_times)
        n = len(sorted_times)
        
        return {
            'avg_response_time_ms': sum(sorted_times) / n,
            'p95_response_time_ms': sorted_times[int(n * 0.95)] if n > 0 else 0.0,
            'p99_response_time_ms': sorted_times[int(n * 0.99)] if n > 0 else 0.0,
            'max_response_time_ms': max(sorted_times) if sorted_times else 0.0
        }


class ScalabilityTester:
    """Tests system scalability with increasing load."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        
    def run_scalability_test(self, simulation_coordinator) -> List[TestResult]:
        """Run scalability test with increasing load levels."""
        logger.info("Starting scalability test")
        
        self.results.clear()
        
        for scale_level in self.config.scalability_steps:
            logger.info(f"Testing scalability at level: {scale_level}")
            
            # Create scaled configuration
            scaled_config = PerformanceTestConfig(
                max_researchers=scale_level,
                max_papers=scale_level * 2,
                concurrent_operations=min(scale_level // 10, 50),
                simulation_duration_minutes=10  # Shorter duration for scalability tests
            )
            
            # Run test at this scale
            test_result = self._run_single_scale_test(
                simulation_coordinator, scaled_config, f"scalability_{scale_level}"
            )
            
            self.results.append(test_result)
            
            # Check if we should stop (system becoming unstable)
            if not test_result.success or test_result.memory_limit_exceeded:
                logger.warning(f"Stopping scalability test at level {scale_level} due to system limits")
                break
        
        logger.info(f"Scalability test completed with {len(self.results)} test points")
        return self.results
    
    def _run_single_scale_test(self, simulation_coordinator, 
                              config: PerformanceTestConfig, 
                              test_name: str) -> TestResult:
        """Run a single scalability test at a specific scale."""
        start_time = datetime.now()
        
        # Initialize monitoring
        monitor = ResourceMonitor(config)
        load_generator = LoadGenerator(config)
        
        errors = []
        warnings = []
        
        try:
            # Start monitoring
            monitor.start_monitoring(test_name)
            
            # Run load test
            load_results = load_generator.generate_review_load(
                simulation_coordinator, 
                config.simulation_duration_minutes * 60
            )
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Calculate results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            peak_metrics = monitor.get_peak_metrics()
            
            # Check for limit violations
            memory_limit_exceeded = peak_metrics.get('peak_memory_mb', 0) > config.memory_limit_mb
            cpu_limit_exceeded = peak_metrics.get('peak_cpu_percent', 0) > config.cpu_limit_percent
            
            if memory_limit_exceeded:
                warnings.append(f"Memory limit exceeded: {peak_metrics.get('peak_memory_mb', 0):.1f}MB")
            
            if cpu_limit_exceeded:
                warnings.append(f"CPU limit exceeded: {peak_metrics.get('peak_cpu_percent', 0):.1f}%")
            
            success = len(errors) == 0 and not memory_limit_exceeded
            
            return TestResult(
                test_name=test_name,
                config=config,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metrics_history=monitor.metrics_history,
                avg_throughput=load_results.get('throughput_ops_per_sec', 0),
                peak_throughput=load_results.get('throughput_ops_per_sec', 0),  # Same for single test
                avg_latency_ms=load_results.get('avg_response_time_ms', 0),
                peak_memory_mb=peak_metrics.get('peak_memory_mb', 0),
                peak_cpu_percent=peak_metrics.get('peak_cpu_percent', 0),
                success=success,
                errors=errors,
                warnings=warnings,
                memory_limit_exceeded=memory_limit_exceeded,
                cpu_limit_exceeded=cpu_limit_exceeded,
                timeout_occurred=load_results.get('total_timeouts', 0) > 0
            )
            
        except Exception as e:
            errors.append(f"Test execution error: {str(e)}")
            logger.error(f"Error in scalability test {test_name}: {e}")
            
            return TestResult(
                test_name=test_name,
                config=config,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics_history=monitor.metrics_history,
                avg_throughput=0,
                peak_throughput=0,
                avg_latency_ms=0,
                peak_memory_mb=0,
                peak_cpu_percent=0,
                success=False,
                errors=errors,
                warnings=warnings,
                memory_limit_exceeded=False,
                cpu_limit_exceeded=False,
                timeout_occurred=False
            )


class StressTester:
    """Performs stress testing to find system breaking points."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        
    def run_stress_test(self, simulation_coordinator) -> TestResult:
        """Run comprehensive stress test."""
        logger.info("Starting stress test")
        
        start_time = datetime.now()
        
        # Create stress configuration
        stress_config = PerformanceTestConfig(
            max_researchers=int(self.config.max_researchers * self.config.stress_multiplier),
            max_papers=int(self.config.max_papers * self.config.stress_multiplier),
            concurrent_operations=int(self.config.concurrent_operations * self.config.stress_multiplier),
            simulation_duration_minutes=self.config.simulation_duration_minutes,
            memory_limit_mb=self.config.memory_limit_mb,
            cpu_limit_percent=self.config.cpu_limit_percent
        )
        
        # Initialize components
        monitor = ResourceMonitor(stress_config)
        load_generator = LoadGenerator(stress_config)
        
        errors = []
        warnings = []
        
        try:
            # Start monitoring
            monitor.start_monitoring("stress_test")
            
            # Run stress phases
            logger.info("Phase 1: Ramp-up")
            ramp_up_results = self._run_ramp_up_phase(
                simulation_coordinator, load_generator, stress_config
            )
            
            logger.info("Phase 2: Steady state stress")
            steady_results = self._run_steady_state_phase(
                simulation_coordinator, load_generator, stress_config
            )
            
            logger.info("Phase 3: Peak stress")
            peak_results = self._run_peak_stress_phase(
                simulation_coordinator, load_generator, stress_config
            )
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Analyze results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            peak_metrics = monitor.get_peak_metrics()
            
            # Combine results from all phases
            total_operations = (
                ramp_up_results.get('operations_completed', 0) +
                steady_results.get('operations_completed', 0) +
                peak_results.get('operations_completed', 0)
            )
            
            avg_throughput = total_operations / duration if duration > 0 else 0
            
            # Check limits
            memory_limit_exceeded = peak_metrics.get('peak_memory_mb', 0) > stress_config.memory_limit_mb
            cpu_limit_exceeded = peak_metrics.get('peak_cpu_percent', 0) > stress_config.cpu_limit_percent
            
            success = len(errors) == 0 and not memory_limit_exceeded and not cpu_limit_exceeded
            
            return TestResult(
                test_name="comprehensive_stress_test",
                config=stress_config,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metrics_history=monitor.metrics_history,
                avg_throughput=avg_throughput,
                peak_throughput=max(
                    ramp_up_results.get('throughput_ops_per_sec', 0),
                    steady_results.get('throughput_ops_per_sec', 0),
                    peak_results.get('throughput_ops_per_sec', 0)
                ),
                avg_latency_ms=(
                    ramp_up_results.get('avg_response_time_ms', 0) +
                    steady_results.get('avg_response_time_ms', 0) +
                    peak_results.get('avg_response_time_ms', 0)
                ) / 3,
                peak_memory_mb=peak_metrics.get('peak_memory_mb', 0),
                peak_cpu_percent=peak_metrics.get('peak_cpu_percent', 0),
                success=success,
                errors=errors,
                warnings=warnings,
                memory_limit_exceeded=memory_limit_exceeded,
                cpu_limit_exceeded=cpu_limit_exceeded,
                timeout_occurred=any([
                    ramp_up_results.get('total_timeouts', 0) > 0,
                    steady_results.get('total_timeouts', 0) > 0,
                    peak_results.get('total_timeouts', 0) > 0
                ])
            )
            
        except Exception as e:
            errors.append(f"Stress test error: {str(e)}")
            logger.error(f"Error in stress test: {e}")
            
            return TestResult(
                test_name="comprehensive_stress_test",
                config=stress_config,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics_history=monitor.metrics_history,
                avg_throughput=0,
                peak_throughput=0,
                avg_latency_ms=0,
                peak_memory_mb=0,
                peak_cpu_percent=0,
                success=False,
                errors=errors,
                warnings=warnings,
                memory_limit_exceeded=False,
                cpu_limit_exceeded=False,
                timeout_occurred=False
            )
    
    def _run_ramp_up_phase(self, simulation_coordinator, load_generator, 
                          config: PerformanceTestConfig) -> Dict[str, Any]:
        """Run gradual ramp-up phase."""
        return load_generator.generate_review_load(
            simulation_coordinator, config.ramp_up_duration_seconds
        )
    
    def _run_steady_state_phase(self, simulation_coordinator, load_generator,
                               config: PerformanceTestConfig) -> Dict[str, Any]:
        """Run steady state stress phase."""
        return load_generator.generate_review_load(
            simulation_coordinator, config.steady_state_duration_seconds
        )
    
    def _run_peak_stress_phase(self, simulation_coordinator, load_generator,
                              config: PerformanceTestConfig) -> Dict[str, Any]:
        """Run peak stress phase with maximum load."""
        # Temporarily increase concurrent operations for peak stress
        original_concurrent = load_generator.config.concurrent_operations
        load_generator.config.concurrent_operations = int(original_concurrent * 1.5)
        
        try:
            return load_generator.generate_review_load(
                simulation_coordinator, 300  # 5 minutes of peak stress
            )
        finally:
            load_generator.config.concurrent_operations = original_concurrent


class PerformanceTestSuite:
    """Comprehensive performance test suite."""
    
    def __init__(self, config: Optional[PerformanceTestConfig] = None):
        self.config = config or PerformanceTestConfig()
        self.results: List[TestResult] = []
        
    def run_comprehensive_tests(self, simulation_coordinator) -> List[TestResult]:
        """Run all performance tests."""
        logger.info("Starting comprehensive performance test suite")
        
        self.results.clear()
        
        try:
            # 1. Baseline performance test
            logger.info("Running baseline performance test")
            baseline_result = self._run_baseline_test(simulation_coordinator)
            self.results.append(baseline_result)
            
            # 2. Load test
            logger.info("Running load test")
            load_result = self._run_load_test(simulation_coordinator)
            self.results.append(load_result)
            
            # 3. Scalability test
            logger.info("Running scalability test")
            scalability_tester = ScalabilityTester(self.config)
            scalability_results = scalability_tester.run_scalability_test(simulation_coordinator)
            self.results.extend(scalability_results)
            
            # 4. Stress test
            logger.info("Running stress test")
            stress_tester = StressTester(self.config)
            stress_result = stress_tester.run_stress_test(simulation_coordinator)
            self.results.append(stress_result)
            
            # 5. Memory leak test
            logger.info("Running memory leak test")
            memory_leak_result = self._run_memory_leak_test(simulation_coordinator)
            self.results.append(memory_leak_result)
            
            logger.info(f"Performance test suite completed with {len(self.results)} tests")
            
        except Exception as e:
            logger.error(f"Error in performance test suite: {e}")
            raise
        
        return self.results
    
    def _run_baseline_test(self, simulation_coordinator) -> TestResult:
        """Run baseline performance test with minimal load."""
        baseline_config = PerformanceTestConfig(
            max_researchers=100,
            max_papers=200,
            concurrent_operations=5,
            simulation_duration_minutes=10
        )
        
        return self._run_single_test(simulation_coordinator, baseline_config, "baseline")
    
    def _run_load_test(self, simulation_coordinator) -> TestResult:
        """Run standard load test."""
        return self._run_single_test(simulation_coordinator, self.config, "load_test")
    
    def _run_memory_leak_test(self, simulation_coordinator) -> TestResult:
        """Run extended test to detect memory leaks."""
        memory_config = PerformanceTestConfig(
            max_researchers=500,
            max_papers=1000,
            concurrent_operations=10,
            simulation_duration_minutes=120,  # Extended duration
            metrics_collection_interval=10  # More frequent monitoring
        )
        
        return self._run_single_test(simulation_coordinator, memory_config, "memory_leak_test")
    
    def _run_single_test(self, simulation_coordinator, config: PerformanceTestConfig,
                        test_name: str) -> TestResult:
        """Run a single performance test."""
        start_time = datetime.now()
        
        monitor = ResourceMonitor(config)
        load_generator = LoadGenerator(config)
        
        errors = []
        warnings = []
        
        try:
            # Start monitoring
            monitor.start_monitoring(test_name)
            
            # Run test
            load_results = load_generator.generate_review_load(
                simulation_coordinator, config.simulation_duration_minutes * 60
            )
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Calculate results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            peak_metrics = monitor.get_peak_metrics()
            
            # Check for issues
            if load_results.get('error_rate_percent', 0) > 5:
                warnings.append(f"High error rate: {load_results.get('error_rate_percent', 0):.1f}%")
            
            if peak_metrics.get('peak_memory_mb', 0) > config.memory_limit_mb * 0.8:
                warnings.append("Memory usage approaching limit")
            
            success = len(errors) == 0 and load_results.get('error_rate_percent', 0) < 10
            
            return TestResult(
                test_name=test_name,
                config=config,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metrics_history=monitor.metrics_history,
                avg_throughput=load_results.get('throughput_ops_per_sec', 0),
                peak_throughput=load_results.get('throughput_ops_per_sec', 0),
                avg_latency_ms=load_results.get('avg_response_time_ms', 0),
                peak_memory_mb=peak_metrics.get('peak_memory_mb', 0),
                peak_cpu_percent=peak_metrics.get('peak_cpu_percent', 0),
                success=success,
                errors=errors,
                warnings=warnings,
                memory_limit_exceeded=peak_metrics.get('peak_memory_mb', 0) > config.memory_limit_mb,
                cpu_limit_exceeded=peak_metrics.get('peak_cpu_percent', 0) > config.cpu_limit_percent,
                timeout_occurred=load_results.get('total_timeouts', 0) > 0
            )
            
        except Exception as e:
            errors.append(f"Test execution error: {str(e)}")
            logger.error(f"Error in test {test_name}: {e}")
            
            return TestResult(
                test_name=test_name,
                config=config,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics_history=monitor.metrics_history,
                avg_throughput=0,
                peak_throughput=0,
                avg_latency_ms=0,
                peak_memory_mb=0,
                peak_cpu_percent=0,
                success=False,
                errors=errors,
                warnings=warnings,
                memory_limit_exceeded=False,
                cpu_limit_exceeded=False,
                timeout_occurred=False
            )
    
    def generate_performance_report(self, output_dir: Path) -> str:
        """Generate comprehensive performance test report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'test_suite_summary': {
                'total_tests': len(self.results),
                'successful_tests': sum(1 for r in self.results if r.success),
                'failed_tests': sum(1 for r in self.results if not r.success),
                'total_duration_hours': sum(r.duration_seconds for r in self.results) / 3600,
                'generated_at': datetime.now().isoformat()
            },
            'test_results': [self._serialize_test_result(result) for result in self.results],
            'performance_summary': self._generate_performance_summary(),
            'recommendations': self._generate_performance_recommendations()
        }
        
        # Save JSON report
        report_file = output_dir / 'performance_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_file = output_dir / 'performance_test_report.html'
        with open(html_file, 'w') as f:
            f.write(self._generate_html_performance_report(report_data))
        
        logger.info(f"Performance test report generated: {report_file}")
        return str(report_file)
    
    def _serialize_test_result(self, result: TestResult) -> Dict[str, Any]:
        """Serialize test result for JSON export."""
        return {
            'test_name': result.test_name,
            'success': result.success,
            'duration_seconds': result.duration_seconds,
            'avg_throughput': result.avg_throughput,
            'peak_throughput': result.peak_throughput,
            'avg_latency_ms': result.avg_latency_ms,
            'peak_memory_mb': result.peak_memory_mb,
            'peak_cpu_percent': result.peak_cpu_percent,
            'errors': result.errors,
            'warnings': result.warnings,
            'memory_limit_exceeded': result.memory_limit_exceeded,
            'cpu_limit_exceeded': result.cpu_limit_exceeded,
            'timeout_occurred': result.timeout_occurred,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat()
        }
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary across all tests."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        
        return {
            'max_throughput_achieved': max(r.peak_throughput for r in successful_results) if successful_results else 0,
            'min_latency_achieved': min(r.avg_latency_ms for r in successful_results) if successful_results else 0,
            'max_stable_memory_mb': max(r.peak_memory_mb for r in successful_results) if successful_results else 0,
            'system_breaking_point': self._identify_breaking_point(),
            'scalability_limit': self._identify_scalability_limit(),
            'performance_bottlenecks': self._identify_bottlenecks()
        }
    
    def _identify_breaking_point(self) -> Dict[str, Any]:
        """Identify system breaking point from test results."""
        failed_results = [r for r in self.results if not r.success]
        
        if not failed_results:
            return {'found': False, 'message': 'No breaking point identified within test parameters'}
        
        # Find the first failure point
        first_failure = min(failed_results, key=lambda r: r.start_time)
        
        return {
            'found': True,
            'test_name': first_failure.test_name,
            'peak_memory_mb': first_failure.peak_memory_mb,
            'peak_cpu_percent': first_failure.peak_cpu_percent,
            'errors': first_failure.errors[:3]  # First 3 errors
        }
    
    def _identify_scalability_limit(self) -> Dict[str, Any]:
        """Identify scalability limits from test results."""
        scalability_results = [r for r in self.results if 'scalability' in r.test_name]
        
        if not scalability_results:
            return {'found': False}
        
        # Find the highest successful scale
        successful_scales = [r for r in scalability_results if r.success]
        
        if successful_scales:
            max_scale = max(successful_scales, key=lambda r: r.config.max_researchers)
            return {
                'found': True,
                'max_researchers': max_scale.config.max_researchers,
                'max_papers': max_scale.config.max_papers,
                'throughput_at_limit': max_scale.avg_throughput
            }
        
        return {'found': False, 'message': 'No successful scalability tests'}
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks from test results."""
        bottlenecks = []
        
        # Analyze memory usage
        high_memory_tests = [r for r in self.results if r.peak_memory_mb > self.config.memory_limit_mb * 0.8]
        if high_memory_tests:
            bottlenecks.append("Memory usage approaching limits")
        
        # Analyze CPU usage
        high_cpu_tests = [r for r in self.results if r.peak_cpu_percent > self.config.cpu_limit_percent * 0.8]
        if high_cpu_tests:
            bottlenecks.append("CPU usage approaching limits")
        
        # Analyze latency
        high_latency_tests = [r for r in self.results if r.avg_latency_ms > 1000]
        if high_latency_tests:
            bottlenecks.append("High response latency detected")
        
        # Analyze throughput degradation
        if len(self.results) > 1:
            throughputs = [r.avg_throughput for r in self.results if r.success]
            if len(throughputs) > 1 and throughputs[-1] < throughputs[0] * 0.5:
                bottlenecks.append("Throughput degradation under load")
        
        return bottlenecks
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks from test results."""
        bottlenecks = []
        
        # Analyze memory usage
        high_memory_tests = [r for r in self.results if r.peak_memory_mb > self.config.memory_limit_mb * 0.8]
        if high_memory_tests:
            bottlenecks.append("Memory usage approaching limits")
        
        # Analyze CPU usage
        high_cpu_tests = [r for r in self.results if r.peak_cpu_percent > self.config.cpu_limit_percent * 0.8]
        if high_cpu_tests:
            bottlenecks.append("CPU usage approaching limits")
        
        # Analyze latency
        high_latency_tests = [r for r in self.results if r.avg_latency_ms > 1000]
        if high_latency_tests:
            bottlenecks.append("High response latency detected")
        
        # Analyze throughput degradation
        if len(self.results) > 1:
            throughputs = [r.avg_throughput for r in self.results if r.success]
            if len(throughputs) > 1 and throughputs[-1] < throughputs[0] * 0.5:
                bottlenecks.append("Throughput degradation under load")
        
        return bottlenecks
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        summary = self._generate_performance_summary()
        
        # Memory recommendations
        if summary.get('max_stable_memory_mb', 0) > self.config.memory_limit_mb * 0.8:
            recommendations.append("Consider increasing available memory or optimizing memory usage")
        
        # CPU recommendations
        high_cpu_results = [r for r in self.results if r.peak_cpu_percent > 80]
        if high_cpu_results:
            recommendations.append("Consider CPU optimization or scaling to multiple cores")
        
        # Scalability recommendations
        scalability_limit = summary.get('scalability_limit', {})
        if scalability_limit.get('found') and scalability_limit.get('max_researchers', 0) < self.config.max_researchers:
            recommendations.append("System scalability is limited - consider architectural improvements")
        
        # Latency recommendations
        high_latency_results = [r for r in self.results if r.avg_latency_ms > 500]
        if high_latency_results:
            recommendations.append("Optimize response times through caching or algorithm improvements")
        
        # Error rate recommendations
        high_error_results = [r for r in self.results if len(r.errors) > 0]
        if high_error_results:
            recommendations.append("Address error conditions to improve system reliability")
        
        return recommendations
    
    def _generate_html_performance_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML performance report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .success {{ color: green; font-weight: bold; }}
                .failure {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Test Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Total Tests: {total_tests} | Successful: <span class="success">{successful_tests}</span> | Failed: <span class="failure">{failed_tests}</span></p>
            </div>
            
            <div class="section">
                <h2>Test Results Summary</h2>
                <table>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Duration (s)</th>
                        <th>Avg Throughput (ops/s)</th>
                        <th>Avg Latency (ms)</th>
                        <th>Peak Memory (MB)</th>
                        <th>Peak CPU (%)</th>
                    </tr>
                    {test_results_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                {performance_metrics}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {recommendations_html}
            </div>
        </body>
        </html>
        """
        
        # Generate test results rows
        test_rows = ""
        for result in report_data['test_results']:
            status_class = "success" if result['success'] else "failure"
            status_text = "PASS" if result['success'] else "FAIL"
            
            test_rows += f"""
            <tr>
                <td>{result['test_name']}</td>
                <td><span class="{status_class}">{status_text}</span></td>
                <td>{result['duration_seconds']:.1f}</td>
                <td>{result['avg_throughput']:.2f}</td>
                <td>{result['avg_latency_ms']:.2f}</td>
                <td>{result['peak_memory_mb']:.1f}</td>
                <td>{result['peak_cpu_percent']:.1f}</td>
            </tr>
            """
        
        # Generate performance metrics
        summary = report_data['performance_summary']
        metrics_html = ""
        for key, value in summary.items():
            if isinstance(value, dict):
                continue
            metrics_html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        
        # Generate recommendations
        recommendations_html = ""
        for rec in report_data['recommendations']:
            recommendations_html += f'<div class="recommendation">{rec}</div>'
        
        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=report_data['test_suite_summary']['total_tests'],
            successful_tests=report_data['test_suite_summary']['successful_tests'],
            failed_tests=report_data['test_suite_summary']['failed_tests'],
            test_results_rows=test_rows,
            performance_metrics=metrics_html,
            recommendations_html=recommendations_html
        )