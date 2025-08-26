"""
Tests for Performance Testing System

This module contains comprehensive tests for the performance testing system,
including load testing, stress testing, and scalability analysis.
"""

import pytest
import json
import tempfile
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.enhancements.performance_testing import (
    PerformanceTestSuite, PerformanceTestConfig, ResourceMonitor,
    LoadGenerator, ScalabilityTester, StressTester, PerformanceMetrics,
    TestResult
)
from src.core.exceptions import SimulationError


class TestPerformanceTestConfig:
    """Test PerformanceTestConfig data class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PerformanceTestConfig()
        
        assert config.max_researchers == 1000
        assert config.max_papers == 5000
        assert config.max_reviews_per_paper == 5
        assert config.simulation_duration_minutes == 60
        assert config.concurrent_operations == 10
        assert config.stress_multiplier == 2.0
        assert config.memory_limit_mb == 4096
        assert config.scalability_steps == [100, 500, 1000, 2000, 5000]
        assert config.detailed_profiling is True
        assert config.generate_reports is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PerformanceTestConfig(
            max_researchers=500,
            concurrent_operations=20,
            stress_multiplier=3.0,
            memory_limit_mb=2048,
            scalability_steps=[50, 100, 200]
        )
        
        assert config.max_researchers == 500
        assert config.concurrent_operations == 20
        assert config.stress_multiplier == 3.0
        assert config.memory_limit_mb == 2048
        assert config.scalability_steps == [50, 100, 200]


class TestPerformanceMetrics:
    """Test PerformanceMetrics data class."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            test_scenario="load_test",
            operations_per_second=100.5,
            reviews_per_second=50.2,
            papers_processed_per_second=10.1,
            avg_response_time_ms=150.0,
            p95_response_time_ms=300.0,
            p99_response_time_ms=500.0,
            max_response_time_ms=800.0,
            cpu_usage_percent=75.5,
            memory_usage_mb=1024.0,
            memory_usage_percent=50.0,
            disk_read_mb_per_sec=10.0,
            disk_write_mb_per_sec=5.0,
            active_threads=20,
            open_file_descriptors=100,
            network_connections=50,
            error_count=2,
            error_rate_percent=1.5,
            timeout_count=1,
            simulation_state_size_mb=256.0,
            database_size_mb=512.0,
            cache_hit_rate_percent=85.0
        )
        
        assert metrics.test_scenario == "load_test"
        assert metrics.operations_per_second == 100.5
        assert metrics.cpu_usage_percent == 75.5
        assert metrics.error_count == 2


class TestTestResult:
    """Test TestResult data class."""
    
    def test_result_creation(self):
        """Test creating test result."""
        config = PerformanceTestConfig()
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)
        
        result = TestResult(
            test_name="baseline_test",
            config=config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=1800.0,
            metrics_history=[],
            avg_throughput=50.0,
            peak_throughput=75.0,
            avg_latency_ms=200.0,
            peak_memory_mb=800.0,
            peak_cpu_percent=60.0,
            success=True,
            errors=[],
            warnings=["Minor performance degradation"],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        
        assert result.test_name == "baseline_test"
        assert result.success is True
        assert result.duration_seconds == 1800.0
        assert result.avg_throughput == 50.0
        assert len(result.warnings) == 1


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    @pytest.fixture
    def config(self):
        return PerformanceTestConfig(metrics_collection_interval=1)
    
    @pytest.fixture
    def monitor(self, config):
        return ResourceMonitor(config)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.config is not None
        assert monitor.monitoring is False
        assert monitor.metrics_history == []
        assert monitor._monitor_thread is None
    
    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        monitor.start_monitoring("test_scenario")
        assert monitor.monitoring is True
        assert monitor.test_scenario == "test_scenario"
        assert monitor._monitor_thread is not None
        
        # Give it a moment to start
        time.sleep(0.1)
        
        monitor.stop_monitoring()
        assert monitor.monitoring is False
    
    def test_get_current_metrics(self, monitor):
        """Test getting current metrics."""
        # Initially no metrics
        current = monitor.get_current_metrics()
        assert current is None
        
        # Add a metric manually
        test_metric = PerformanceMetrics(
            timestamp=datetime.now(),
            test_scenario="test",
            operations_per_second=0,
            reviews_per_second=0,
            papers_processed_per_second=0,
            avg_response_time_ms=0,
            p95_response_time_ms=0,
            p99_response_time_ms=0,
            max_response_time_ms=0,
            cpu_usage_percent=50,
            memory_usage_mb=500,
            memory_usage_percent=25,
            disk_read_mb_per_sec=0,
            disk_write_mb_per_sec=0,
            active_threads=10,
            open_file_descriptors=50,
            network_connections=5,
            error_count=0,
            error_rate_percent=0,
            timeout_count=0,
            simulation_state_size_mb=0,
            database_size_mb=0,
            cache_hit_rate_percent=0
        )
        
        monitor.metrics_history.append(test_metric)
        
        current = monitor.get_current_metrics()
        assert current is not None
        assert current.cpu_usage_percent == 50
    
    def test_get_peak_metrics(self, monitor):
        """Test getting peak metrics."""
        # Add some test metrics
        for i in range(3):
            metric = PerformanceMetrics(
                timestamp=datetime.now(),
                test_scenario="test",
                operations_per_second=0,
                reviews_per_second=0,
                papers_processed_per_second=0,
                avg_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                max_response_time_ms=0,
                cpu_usage_percent=50 + i * 10,
                memory_usage_mb=500 + i * 100,
                memory_usage_percent=25 + i * 5,
                disk_read_mb_per_sec=0,
                disk_write_mb_per_sec=0,
                active_threads=10 + i,
                open_file_descriptors=50 + i * 10,
                network_connections=5,
                error_count=0,
                error_rate_percent=0,
                timeout_count=0,
                simulation_state_size_mb=0,
                database_size_mb=0,
                cache_hit_rate_percent=0
            )
            monitor.metrics_history.append(metric)
        
        peaks = monitor.get_peak_metrics()
        
        assert peaks['peak_cpu_percent'] == 70  # 50 + 2*10
        assert peaks['peak_memory_mb'] == 700   # 500 + 2*100
        assert peaks['peak_threads'] == 12      # 10 + 2


class TestLoadGenerator:
    """Test LoadGenerator functionality."""
    
    @pytest.fixture
    def config(self):
        return PerformanceTestConfig(concurrent_operations=2, simulation_duration_minutes=1)
    
    @pytest.fixture
    def load_generator(self, config):
        return LoadGenerator(config)
    
    @pytest.fixture
    def mock_coordinator(self):
        coordinator = Mock()
        coordinator.get_system_health.return_value = {'status': 'healthy'}
        return coordinator
    
    def test_load_generator_initialization(self, load_generator):
        """Test load generator initialization."""
        assert load_generator.config is not None
        assert load_generator.operation_times == []
        assert load_generator.error_count == 0
        assert load_generator.timeout_count == 0
    
    def test_generate_review_load(self, load_generator, mock_coordinator):
        """Test generating review load."""
        # Run for a very short duration for testing
        results = load_generator.generate_review_load(mock_coordinator, 2)
        
        assert 'operations_completed' in results
        assert 'throughput_ops_per_sec' in results
        assert 'total_errors' in results
        assert 'total_timeouts' in results
        assert 'error_rate_percent' in results
        assert 'avg_response_time_ms' in results
        
        # Should have completed some operations
        assert results['operations_completed'] >= 0
        assert results['throughput_ops_per_sec'] >= 0
    
    def test_simulate_review_operation(self, load_generator, mock_coordinator):
        """Test simulating a single review operation."""
        operation_time = load_generator._simulate_review_operation(mock_coordinator)
        
        assert isinstance(operation_time, float)
        assert operation_time > 0  # Should take some time
        
        # Verify the coordinator was called
        mock_coordinator.get_system_health.assert_called()
    
    def test_calculate_latency_metrics_empty(self, load_generator):
        """Test calculating latency metrics with no data."""
        metrics = load_generator._calculate_latency_metrics()
        
        assert metrics['avg_response_time_ms'] == 0.0
        assert metrics['p95_response_time_ms'] == 0.0
        assert metrics['p99_response_time_ms'] == 0.0
        assert metrics['max_response_time_ms'] == 0.0
    
    def test_calculate_latency_metrics_with_data(self, load_generator):
        """Test calculating latency metrics with sample data."""
        # Add some sample operation times
        load_generator.operation_times = [100, 150, 200, 250, 300, 400, 500]
        
        metrics = load_generator._calculate_latency_metrics()
        
        assert metrics['avg_response_time_ms'] > 0
        assert metrics['p95_response_time_ms'] > metrics['avg_response_time_ms']
        assert metrics['p99_response_time_ms'] >= metrics['p95_response_time_ms']
        assert metrics['max_response_time_ms'] == 500


class TestScalabilityTester:
    """Test ScalabilityTester functionality."""
    
    @pytest.fixture
    def config(self):
        return PerformanceTestConfig(
            scalability_steps=[10, 20],  # Small steps for testing
            simulation_duration_minutes=1  # Short duration
        )
    
    @pytest.fixture
    def scalability_tester(self, config):
        return ScalabilityTester(config)
    
    @pytest.fixture
    def mock_coordinator(self):
        coordinator = Mock()
        coordinator.get_system_health.return_value = {'status': 'healthy'}
        return coordinator
    
    def test_scalability_tester_initialization(self, scalability_tester):
        """Test scalability tester initialization."""
        assert scalability_tester.config is not None
        assert scalability_tester.results == []
    
    def test_run_scalability_test(self, scalability_tester, mock_coordinator):
        """Test running scalability test."""
        results = scalability_tester.run_scalability_test(mock_coordinator)
        
        assert isinstance(results, list)
        assert len(results) <= len(scalability_tester.config.scalability_steps)
        
        # Check that results are TestResult objects
        for result in results:
            assert isinstance(result, TestResult)
            assert result.test_name.startswith('scalability_')
    
    @patch('src.enhancements.performance_testing.ResourceMonitor')
    @patch('src.enhancements.performance_testing.LoadGenerator')
    def test_run_single_scale_test(self, mock_load_gen_class, mock_monitor_class, 
                                  scalability_tester, mock_coordinator):
        """Test running a single scale test."""
        # Mock the monitor
        mock_monitor = Mock()
        mock_monitor.metrics_history = []
        mock_monitor.get_peak_metrics.return_value = {
            'peak_memory_mb': 500,
            'peak_cpu_percent': 60
        }
        mock_monitor_class.return_value = mock_monitor
        
        # Mock the load generator
        mock_load_gen = Mock()
        mock_load_gen.generate_review_load.return_value = {
            'operations_completed': 100,
            'throughput_ops_per_sec': 10.0,
            'total_errors': 0,
            'total_timeouts': 0,
            'error_rate_percent': 0.0,
            'avg_response_time_ms': 100.0,
            'p95_response_time_ms': 200.0,
            'p99_response_time_ms': 300.0,
            'max_response_time_ms': 400.0
        }
        mock_load_gen_class.return_value = mock_load_gen
        
        config = PerformanceTestConfig(max_researchers=100)
        result = scalability_tester._run_single_scale_test(
            mock_coordinator, config, "test_scale"
        )
        
        assert isinstance(result, TestResult)
        assert result.test_name == "test_scale"
        assert result.success is True
        assert result.avg_throughput == 10.0


class TestStressTester:
    """Test StressTester functionality."""
    
    @pytest.fixture
    def config(self):
        return PerformanceTestConfig(
            stress_multiplier=1.5,  # Smaller multiplier for testing
            simulation_duration_minutes=1,  # Short duration
            ramp_up_duration_seconds=5,
            steady_state_duration_seconds=10,
            ramp_down_duration_seconds=5
        )
    
    @pytest.fixture
    def stress_tester(self, config):
        return StressTester(config)
    
    @pytest.fixture
    def mock_coordinator(self):
        coordinator = Mock()
        coordinator.get_system_health.return_value = {'status': 'healthy'}
        return coordinator
    
    def test_stress_tester_initialization(self, stress_tester):
        """Test stress tester initialization."""
        assert stress_tester.config is not None
    
    @patch('src.enhancements.performance_testing.ResourceMonitor')
    @patch('src.enhancements.performance_testing.LoadGenerator')
    def test_run_stress_test(self, mock_load_gen_class, mock_monitor_class, 
                           stress_tester, mock_coordinator):
        """Test running stress test."""
        # Mock the monitor
        mock_monitor = Mock()
        mock_monitor.metrics_history = []
        mock_monitor.get_peak_metrics.return_value = {
            'peak_memory_mb': 800,
            'peak_cpu_percent': 85
        }
        mock_monitor_class.return_value = mock_monitor
        
        # Mock the load generator
        mock_load_gen = Mock()
        mock_load_results = {
            'operations_completed': 50,
            'throughput_ops_per_sec': 5.0,
            'total_errors': 1,
            'total_timeouts': 0,
            'error_rate_percent': 2.0,
            'avg_response_time_ms': 200.0
        }
        mock_load_gen.generate_review_load.return_value = mock_load_results
        mock_load_gen_class.return_value = mock_load_gen
        
        result = stress_tester.run_stress_test(mock_coordinator)
        
        assert isinstance(result, TestResult)
        assert result.test_name == "comprehensive_stress_test"
        # Should have called generate_review_load multiple times (for different phases)
        assert mock_load_gen.generate_review_load.call_count >= 2


class TestPerformanceTestSuite:
    """Test PerformanceTestSuite functionality."""
    
    @pytest.fixture
    def config(self):
        return PerformanceTestConfig(
            max_researchers=50,  # Small for testing
            simulation_duration_minutes=1,  # Short duration
            scalability_steps=[10, 20]  # Small steps
        )
    
    @pytest.fixture
    def test_suite(self, config):
        return PerformanceTestSuite(config)
    
    @pytest.fixture
    def mock_coordinator(self):
        coordinator = Mock()
        coordinator.get_system_health.return_value = {'status': 'healthy'}
        return coordinator
    
    def test_test_suite_initialization(self, test_suite):
        """Test test suite initialization."""
        assert test_suite.config is not None
        assert test_suite.results == []
    
    def test_default_config_initialization(self):
        """Test test suite with default config."""
        suite = PerformanceTestSuite()
        assert suite.config is not None
        assert isinstance(suite.config, PerformanceTestConfig)
    
    @patch('src.enhancements.performance_testing.ResourceMonitor')
    @patch('src.enhancements.performance_testing.LoadGenerator')
    @patch('src.enhancements.performance_testing.ScalabilityTester')
    @patch('src.enhancements.performance_testing.StressTester')
    def test_run_comprehensive_tests(self, mock_stress_tester_class, mock_scalability_tester_class,
                                   mock_load_gen_class, mock_monitor_class, 
                                   test_suite, mock_coordinator):
        """Test running comprehensive tests."""
        # Mock all the components
        mock_monitor = Mock()
        mock_monitor.metrics_history = []
        mock_monitor.get_peak_metrics.return_value = {
            'peak_memory_mb': 400,
            'peak_cpu_percent': 50
        }
        mock_monitor_class.return_value = mock_monitor
        
        mock_load_gen = Mock()
        mock_load_gen.generate_review_load.return_value = {
            'operations_completed': 25,
            'throughput_ops_per_sec': 2.5,
            'total_errors': 0,
            'total_timeouts': 0,
            'error_rate_percent': 0.0,
            'avg_response_time_ms': 150.0
        }
        mock_load_gen_class.return_value = mock_load_gen
        
        # Mock scalability tester
        mock_scalability_tester = Mock()
        mock_scalability_result = TestResult(
            test_name="scalability_10",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=60.0,
            metrics_history=[],
            avg_throughput=5.0,
            peak_throughput=5.0,
            avg_latency_ms=100.0,
            peak_memory_mb=300.0,
            peak_cpu_percent=40.0,
            success=True,
            errors=[],
            warnings=[],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        mock_scalability_tester.run_scalability_test.return_value = [mock_scalability_result]
        mock_scalability_tester_class.return_value = mock_scalability_tester
        
        # Mock stress tester
        mock_stress_tester = Mock()
        mock_stress_result = TestResult(
            test_name="comprehensive_stress_test",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=120.0,
            metrics_history=[],
            avg_throughput=3.0,
            peak_throughput=4.0,
            avg_latency_ms=200.0,
            peak_memory_mb=600.0,
            peak_cpu_percent=70.0,
            success=True,
            errors=[],
            warnings=[],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        mock_stress_tester.run_stress_test.return_value = mock_stress_result
        mock_stress_tester_class.return_value = mock_stress_tester
        
        results = test_suite.run_comprehensive_tests(mock_coordinator)
        
        assert isinstance(results, list)
        assert len(results) >= 4  # baseline, load, scalability, stress, memory leak
        
        # Check that all test types are represented
        test_names = [result.test_name for result in results]
        assert any('baseline' in name for name in test_names)
        assert any('load_test' in name for name in test_names)
        assert any('scalability' in name for name in test_names)
        assert any('stress' in name for name in test_names)
    
    def test_generate_performance_report(self, test_suite):
        """Test generating performance report."""
        # Add some mock results
        mock_result = TestResult(
            test_name="test_baseline",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=60.0,
            metrics_history=[],
            avg_throughput=10.0,
            peak_throughput=15.0,
            avg_latency_ms=100.0,
            peak_memory_mb=400.0,
            peak_cpu_percent=50.0,
            success=True,
            errors=[],
            warnings=["Test warning"],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        
        test_suite.results = [mock_result]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            report_file = test_suite.generate_performance_report(output_dir)
            
            assert Path(report_file).exists()
            
            # Check JSON content
            with open(report_file) as f:
                report_data = json.load(f)
            
            assert 'test_suite_summary' in report_data
            assert 'test_results' in report_data
            assert 'performance_summary' in report_data
            assert 'recommendations' in report_data
            
            assert report_data['test_suite_summary']['total_tests'] == 1
            assert report_data['test_suite_summary']['successful_tests'] == 1
            
            # Check HTML report was also generated
            html_file = output_dir / 'performance_test_report.html'
            assert html_file.exists()
    
    def test_serialize_test_result(self, test_suite):
        """Test serializing test result."""
        result = TestResult(
            test_name="test_serialize",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=30.0,
            metrics_history=[],
            avg_throughput=5.0,
            peak_throughput=8.0,
            avg_latency_ms=150.0,
            peak_memory_mb=300.0,
            peak_cpu_percent=40.0,
            success=True,
            errors=["Test error"],
            warnings=["Test warning"],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=True
        )
        
        serialized = test_suite._serialize_test_result(result)
        
        assert serialized['test_name'] == "test_serialize"
        assert serialized['success'] is True
        assert serialized['duration_seconds'] == 30.0
        assert serialized['avg_throughput'] == 5.0
        assert serialized['errors'] == ["Test error"]
        assert serialized['warnings'] == ["Test warning"]
        assert serialized['timeout_occurred'] is True
    
    def test_generate_performance_summary(self, test_suite):
        """Test generating performance summary."""
        # Add some mock results
        successful_result = TestResult(
            test_name="successful_test",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=60.0,
            metrics_history=[],
            avg_throughput=10.0,
            peak_throughput=15.0,
            avg_latency_ms=100.0,
            peak_memory_mb=400.0,
            peak_cpu_percent=50.0,
            success=True,
            errors=[],
            warnings=[],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        
        failed_result = TestResult(
            test_name="failed_test",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=30.0,
            metrics_history=[],
            avg_throughput=5.0,
            peak_throughput=7.0,
            avg_latency_ms=200.0,
            peak_memory_mb=800.0,
            peak_cpu_percent=90.0,
            success=False,
            errors=["System overload"],
            warnings=[],
            memory_limit_exceeded=True,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        
        test_suite.results = [successful_result, failed_result]
        
        summary = test_suite._generate_performance_summary()
        
        assert 'max_throughput_achieved' in summary
        assert 'min_latency_achieved' in summary
        assert 'max_stable_memory_mb' in summary
        assert 'system_breaking_point' in summary
        assert 'scalability_limit' in summary
        assert 'performance_bottlenecks' in summary
        
        # Should use successful results for max values
        assert summary['max_throughput_achieved'] == 15.0
        assert summary['min_latency_achieved'] == 100.0
        assert summary['max_stable_memory_mb'] == 400.0
    
    def test_identify_breaking_point(self, test_suite):
        """Test identifying system breaking point."""
        # Add a failed result
        failed_result = TestResult(
            test_name="breaking_point_test",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=30.0,
            metrics_history=[],
            avg_throughput=0.0,
            peak_throughput=0.0,
            avg_latency_ms=1000.0,
            peak_memory_mb=5000.0,
            peak_cpu_percent=95.0,
            success=False,
            errors=["Memory exhausted", "CPU overload"],
            warnings=[],
            memory_limit_exceeded=True,
            cpu_limit_exceeded=True,
            timeout_occurred=False
        )
        
        test_suite.results = [failed_result]
        
        breaking_point = test_suite._identify_breaking_point()
        
        assert breaking_point['found'] is True
        assert breaking_point['test_name'] == "breaking_point_test"
        assert breaking_point['peak_memory_mb'] == 5000.0
        assert breaking_point['peak_cpu_percent'] == 95.0
        assert len(breaking_point['errors']) <= 3
    
    def test_identify_bottlenecks(self, test_suite):
        """Test identifying performance bottlenecks."""
        # Set the test suite config to have a lower memory limit for testing
        test_suite.config.memory_limit_mb = 1000
        
        # Add results with various bottlenecks
        high_memory_result = TestResult(
            test_name="high_memory_test",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=60.0,
            metrics_history=[],
            avg_throughput=10.0,
            peak_throughput=15.0,
            avg_latency_ms=100.0,
            peak_memory_mb=900.0,  # 90% of limit
            peak_cpu_percent=50.0,
            success=True,
            errors=[],
            warnings=[],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        
        high_latency_result = TestResult(
            test_name="high_latency_test",
            config=test_suite.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=60.0,
            metrics_history=[],
            avg_throughput=5.0,
            peak_throughput=8.0,
            avg_latency_ms=1500.0,  # High latency
            peak_memory_mb=400.0,
            peak_cpu_percent=60.0,
            success=True,
            errors=[],
            warnings=[],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        
        test_suite.results = [high_memory_result, high_latency_result]
        
        bottlenecks = test_suite._identify_bottlenecks()
        
        assert isinstance(bottlenecks, list)
        assert any("memory" in bottleneck.lower() for bottleneck in bottlenecks)
        assert any("latency" in bottleneck.lower() for bottleneck in bottlenecks)
    
    def test_generate_performance_recommendations(self, test_suite):
        """Test generating performance recommendations."""
        # Add results that should trigger recommendations
        problematic_result = TestResult(
            test_name="problematic_test",
            config=PerformanceTestConfig(memory_limit_mb=1000),
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=60.0,
            metrics_history=[],
            avg_throughput=2.0,
            peak_throughput=3.0,
            avg_latency_ms=800.0,  # High latency
            peak_memory_mb=900.0,  # High memory usage
            peak_cpu_percent=85.0,  # High CPU usage
            success=False,
            errors=["Timeout error"],
            warnings=[],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        
        test_suite.results = [problematic_result]
        
        recommendations = test_suite._generate_performance_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations for various issues
        rec_text = " ".join(recommendations).lower()
        assert any(keyword in rec_text for keyword in ["memory", "cpu", "latency", "error"])


if __name__ == "__main__":
    pytest.main([__file__])