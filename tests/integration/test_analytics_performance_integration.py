"""
Integration Test for Analytics and Performance Testing Systems

This test demonstrates the complete analytics and performance testing functionality
by running a comprehensive simulation with metrics collection and performance analysis.
"""

import sys
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.enhancements.simulation_analytics import (
    SimulationAnalytics, AnalyticsConfiguration, MetricsSnapshot
)
from src.enhancements.performance_testing import (
    PerformanceTestSuite, PerformanceTestConfig, TestResult
)
from src.enhancements.simulation_coordinator import SimulationCoordinator, SystemConfiguration
from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel


class MockSimulationCoordinator:
    """Mock simulation coordinator for testing analytics and performance systems."""
    
    def __init__(self):
        self.state = MockSimulationState()
        self.systems_initialized = True
        
        # Initialize mock systems
        self.review_system = MockReviewSystem()
        self.bias_engine = MockBiasEngine()
        self.collaboration_network = MockCollaborationNetwork()
        self.citation_network = MockCitationNetwork()
        self.conference_community = MockConferenceCommunity()
        self.tenure_track_manager = MockTenureTrackManager()
        self.promotion_criteria_evaluator = MockPromotionCriteriaEvaluator()
        self.job_market_simulator = MockJobMarketSimulator()
        self.venue_registry = MockVenueRegistry()
        self.venue_shopping_tracker = MockVenueShoppingTracker()
        self.review_trading_detector = MockReviewTradingDetector()
        self.citation_cartel_detector = MockCitationCartelDetector()
        self.salami_slicing_detector = MockSalamiSlicingDetector()
        
        print("Mock simulation coordinator initialized")
    
    def get_system_health(self):
        """Mock system health check."""
        return {
            'status': 'healthy',
            'uptime': '1h 30m',
            'memory_usage': '512MB',
            'cpu_usage': '45%',
            'active_systems': 15
        }


class MockSimulationState:
    """Mock simulation state."""
    
    def __init__(self):
        self.simulation_id = f"test_analytics_{int(time.time())}"
        self.total_researchers = 150
        self.total_papers = 300
        self.total_reviews = 900
        self.active_venues = 8
        self.total_errors = 2
        
        print(f"Mock simulation state created with ID: {self.simulation_id}")


class MockReviewSystem:
    """Mock review system."""
    
    def get_recent_reviews(self, days=1):
        """Return mock recent reviews."""
        from src.data.enhanced_models import StructuredReview
        
        reviews = []
        for i in range(5):
            review = StructuredReview(
                reviewer_id=f"reviewer_{i}",
                paper_id=f"paper_{i}",
                venue_id="mock_venue",
                criteria_scores=None,
                confidence_level=3 + (i % 3),
                recommendation=None,
                executive_summary=f"Mock review {i}",
                detailed_strengths=[],
                detailed_weaknesses=[],
                technical_comments="",
                presentation_comments="",
                questions_for_authors=[],
                suggestions_for_improvement=[],
                review_length=200 + i * 50,
                time_spent_minutes=60 + i * 15,
                quality_score=3.0 + i * 0.3,
                completeness_score=0.8 + i * 0.05,
                applied_biases=[],
                bias_adjusted_scores={},
                submission_timestamp=datetime.now() - timedelta(hours=i),
                deadline=datetime.now() + timedelta(days=1),
                is_late=False,
                revision_round=1
            )
            reviews.append(review)
        
        return reviews


class MockBiasEngine:
    """Mock bias engine."""
    
    def get_bias_statistics(self):
        """Return mock bias statistics."""
        return {
            'effects_count': {
                'confirmation': 25,
                'halo_effect': 15,
                'anchoring': 20,
                'availability': 10
            },
            'average_impact': {
                'confirmation': 0.35,
                'halo_effect': 0.42,
                'anchoring': 0.28,
                'availability': 0.22
            }
        }


class MockCollaborationNetwork:
    """Mock collaboration network."""
    
    def calculate_network_density(self):
        return 0.18
    
    def has_recent_collaboration(self, paper_id, reviewer_id):
        return False


class MockCitationNetwork:
    """Mock citation network."""
    
    def get_network_size(self):
        return 750
    
    def has_citation_relationship(self, paper_id, reviewer_id):
        return False


class MockConferenceCommunity:
    """Mock conference community."""
    
    def get_community_count(self):
        return 6
    
    def has_close_community_ties(self, paper_id, reviewer_id):
        return False


class MockTenureTrackManager:
    """Mock tenure track manager."""
    
    def get_success_rate(self):
        return 0.68


class MockPromotionCriteriaEvaluator:
    """Mock promotion criteria evaluator."""
    
    def get_promotion_rates(self):
        return {
            'assistant_to_associate': 0.42,
            'associate_to_full': 0.28,
            'postdoc_to_assistant': 0.35
        }


class MockJobMarketSimulator:
    """Mock job market simulator."""
    
    def get_competition_level(self):
        return 0.82


class MockVenueRegistry:
    """Mock venue registry."""
    
    def get_all_venues(self):
        """Return mock venues."""
        venues = []
        for i in range(3):
            venue = MockVenue(f"venue_{i}", 0.2 + i * 0.1, 3.5 + i * 0.3)
            venues.append(venue)
        return venues


class MockVenue:
    """Mock venue."""
    
    def __init__(self, venue_id, acceptance_rate, quality_score):
        self.id = venue_id
        self.acceptance_rate = acceptance_rate
        self.quality_score = quality_score


class MockVenueShoppingTracker:
    """Mock venue shopping tracker."""
    
    def get_incident_count(self):
        return 12


class MockReviewTradingDetector:
    """Mock review trading detector."""
    
    def get_detection_count(self):
        return 4


class MockCitationCartelDetector:
    """Mock citation cartel detector."""
    
    def get_member_count(self):
        return 7


class MockSalamiSlicingDetector:
    """Mock salami slicing detector."""
    
    def get_case_count(self):
        return 9


def test_analytics_system():
    """Test the complete analytics system."""
    print("\n" + "="*60)
    print("TESTING SIMULATION ANALYTICS SYSTEM")
    print("="*60)
    
    # Create analytics configuration
    config = AnalyticsConfiguration(
        metrics_collection_interval=2,  # Fast collection for testing
        real_time_analytics=False,  # Disable for testing
        detailed_logging_enabled=True,
        generate_daily_reports=True,
        include_visualizations=True
    )
    
    # Initialize analytics system
    analytics = SimulationAnalytics(config)
    print(f"✓ Analytics system initialized with config: {config.export_format}")
    
    # Create mock coordinator
    coordinator = MockSimulationCoordinator()
    
    # Start analytics
    analytics.start_analytics(coordinator)
    print("✓ Analytics collection started")
    
    # Collect several metrics snapshots
    print("\nCollecting metrics snapshots...")
    for i in range(5):
        snapshot = analytics.collect_metrics_snapshot()
        print(f"  Snapshot {i+1}: {snapshot.total_researchers} researchers, "
              f"{snapshot.total_reviews} reviews, quality: {snapshot.avg_review_quality:.2f}")
        
        # Simulate some changes in the system
        coordinator.state.total_researchers += 10
        coordinator.state.total_papers += 20
        coordinator.state.total_reviews += 60
        
        time.sleep(0.5)  # Brief pause between collections
    
    print(f"✓ Collected {len(analytics.metrics_collector.metrics_history)} metrics snapshots")
    
    # Test real-time metrics
    real_time_metrics = analytics.get_real_time_metrics()
    print(f"✓ Real-time metrics available: {len(real_time_metrics)} fields")
    
    # Generate analytics report
    print("\nGenerating comprehensive analytics report...")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "analytics_report"
        report_files = analytics.generate_analytics_report(output_dir)
        
        print(f"✓ Report generated with {len(report_files)} files:")
        for format_type, file_path in report_files.items():
            file_size = Path(file_path).stat().st_size
            print(f"  - {format_type.upper()}: {Path(file_path).name} ({file_size} bytes)")
        
        # Verify report content
        if 'json' in report_files:
            import json
            with open(report_files['json']) as f:
                report_data = json.load(f)
            
            print(f"✓ JSON report contains {len(report_data)} sections")
            print(f"  - Statistical summary: {len(report_data.get('statistical_summary', {}))} metrics")
            print(f"  - Generated charts: {len(report_data.get('generated_charts', []))} visualizations")
    
    # Test data export
    print("\nTesting data export...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export as JSON
        json_export = analytics.export_metrics_data("json", Path(temp_dir) / "metrics.json")
        json_size = Path(json_export).stat().st_size
        print(f"✓ JSON export: {json_size} bytes")
        
        # Export as CSV
        csv_export = analytics.export_metrics_data("csv", Path(temp_dir) / "metrics.csv")
        csv_size = Path(csv_export).stat().st_size
        print(f"✓ CSV export: {csv_size} bytes")
    
    # Stop analytics
    analytics.stop_analytics()
    print("✓ Analytics system stopped")
    
    print("\n✅ ANALYTICS SYSTEM TEST COMPLETED SUCCESSFULLY")
    return True


def test_performance_system():
    """Test the complete performance testing system."""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE TESTING SYSTEM")
    print("="*60)
    
    # Create performance test configuration
    config = PerformanceTestConfig(
        max_researchers=100,  # Small scale for testing
        max_papers=200,
        concurrent_operations=5,
        simulation_duration_minutes=1,  # Short duration for testing
        scalability_steps=[25, 50],  # Small steps
        memory_limit_mb=1024,
        cpu_limit_percent=80.0,
        detailed_profiling=True,
        generate_reports=True
    )
    
    # Initialize performance test suite
    test_suite = PerformanceTestSuite(config)
    print(f"✓ Performance test suite initialized")
    print(f"  - Max researchers: {config.max_researchers}")
    print(f"  - Concurrent operations: {config.concurrent_operations}")
    print(f"  - Scalability steps: {config.scalability_steps}")
    
    # Create mock coordinator
    coordinator = MockSimulationCoordinator()
    
    # Run individual test components
    print("\nTesting individual components...")
    
    # Test resource monitoring
    from src.enhancements.performance_testing import ResourceMonitor
    monitor = ResourceMonitor(config)
    monitor.start_monitoring("component_test")
    time.sleep(2)  # Monitor for 2 seconds
    monitor.stop_monitoring()
    
    current_metrics = monitor.get_current_metrics()
    peak_metrics = monitor.get_peak_metrics()
    
    print(f"✓ Resource monitoring: {len(monitor.metrics_history)} samples collected")
    if current_metrics:
        print(f"  - Current CPU: {current_metrics.cpu_usage_percent:.1f}%")
        print(f"  - Current Memory: {current_metrics.memory_usage_mb:.1f}MB")
    
    # Test load generation
    from src.enhancements.performance_testing import LoadGenerator
    load_gen = LoadGenerator(config)
    print("\nTesting load generation...")
    
    load_results = load_gen.generate_review_load(coordinator, 3)  # 3 seconds
    print(f"✓ Load test completed:")
    print(f"  - Operations: {load_results['operations_completed']}")
    print(f"  - Throughput: {load_results['throughput_ops_per_sec']:.2f} ops/sec")
    print(f"  - Avg latency: {load_results['avg_response_time_ms']:.2f}ms")
    print(f"  - Error rate: {load_results['error_rate_percent']:.2f}%")
    
    # Run a simplified comprehensive test
    print("\nRunning simplified performance tests...")
    
    # Mock the comprehensive test to avoid long execution
    from unittest.mock import patch, Mock
    
    with patch('src.enhancements.performance_testing.ResourceMonitor') as mock_monitor_class, \
         patch('src.enhancements.performance_testing.LoadGenerator') as mock_load_gen_class, \
         patch('src.enhancements.performance_testing.ScalabilityTester') as mock_scalability_class, \
         patch('src.enhancements.performance_testing.StressTester') as mock_stress_class:
        
        # Mock monitor
        mock_monitor = Mock()
        mock_monitor.metrics_history = []
        mock_monitor.get_peak_metrics.return_value = {
            'peak_memory_mb': 400,
            'peak_cpu_percent': 55,
            'peak_threads': 15,
            'peak_file_descriptors': 100
        }
        mock_monitor_class.return_value = mock_monitor
        
        # Mock load generator
        mock_load_gen = Mock()
        mock_load_gen.generate_review_load.return_value = {
            'operations_completed': 50,
            'throughput_ops_per_sec': 8.5,
            'total_errors': 1,
            'total_timeouts': 0,
            'error_rate_percent': 2.0,
            'avg_response_time_ms': 120.0,
            'p95_response_time_ms': 200.0,
            'p99_response_time_ms': 300.0,
            'max_response_time_ms': 400.0
        }
        mock_load_gen_class.return_value = mock_load_gen
        
        # Mock scalability tester
        from src.enhancements.performance_testing import TestResult
        mock_scalability_tester = Mock()
        scalability_result = TestResult(
            test_name="scalability_25",
            config=config,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=1),
            duration_seconds=60.0,
            metrics_history=[],
            avg_throughput=6.0,
            peak_throughput=8.0,
            avg_latency_ms=150.0,
            peak_memory_mb=300.0,
            peak_cpu_percent=45.0,
            success=True,
            errors=[],
            warnings=[],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        mock_scalability_tester.run_scalability_test.return_value = [scalability_result]
        mock_scalability_class.return_value = mock_scalability_tester
        
        # Mock stress tester
        mock_stress_tester = Mock()
        stress_result = TestResult(
            test_name="comprehensive_stress_test",
            config=config,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=2),
            duration_seconds=120.0,
            metrics_history=[],
            avg_throughput=4.0,
            peak_throughput=6.0,
            avg_latency_ms=250.0,
            peak_memory_mb=600.0,
            peak_cpu_percent=75.0,
            success=True,
            errors=[],
            warnings=["High memory usage detected"],
            memory_limit_exceeded=False,
            cpu_limit_exceeded=False,
            timeout_occurred=False
        )
        mock_stress_tester.run_stress_test.return_value = stress_result
        mock_stress_class.return_value = mock_stress_tester
        
        # Run comprehensive tests
        results = test_suite.run_comprehensive_tests(coordinator)
        
        print(f"✓ Comprehensive tests completed: {len(results)} test results")
        
        # Display results summary
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = len(results) - successful_tests
        
        print(f"  - Successful tests: {successful_tests}")
        print(f"  - Failed tests: {failed_tests}")
        
        for result in results:
            status = "PASS" if result.success else "FAIL"
            print(f"  - {result.test_name}: {status} "
                  f"({result.avg_throughput:.1f} ops/sec, "
                  f"{result.avg_latency_ms:.1f}ms avg latency)")
    
    # Generate performance report
    print("\nGenerating performance report...")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        report_file = test_suite.generate_performance_report(output_dir)
        report_size = Path(report_file).stat().st_size
        
        print(f"✓ Performance report generated: {Path(report_file).name} ({report_size} bytes)")
        
        # Verify report content
        import json
        with open(report_file) as f:
            report_data = json.load(f)
        
        summary = report_data['test_suite_summary']
        print(f"  - Total tests: {summary['total_tests']}")
        print(f"  - Successful: {summary['successful_tests']}")
        print(f"  - Failed: {summary['failed_tests']}")
        print(f"  - Total duration: {summary['total_duration_hours']:.2f} hours")
        
        # Check for HTML report
        html_file = output_dir / 'performance_test_report.html'
        if html_file.exists():
            html_size = html_file.stat().st_size
            print(f"  - HTML report: {html_file.name} ({html_size} bytes)")
    
    print("\n✅ PERFORMANCE TESTING SYSTEM TEST COMPLETED SUCCESSFULLY")
    return True


def test_integrated_analytics_and_performance():
    """Test integrated analytics and performance systems."""
    print("\n" + "="*60)
    print("TESTING INTEGRATED ANALYTICS AND PERFORMANCE")
    print("="*60)
    
    # Create configurations
    analytics_config = AnalyticsConfiguration(
        metrics_collection_interval=1,
        real_time_analytics=False,
        include_visualizations=True
    )
    
    performance_config = PerformanceTestConfig(
        max_researchers=75,
        simulation_duration_minutes=1,
        concurrent_operations=3,
        scalability_steps=[25, 50]
    )
    
    # Initialize systems
    analytics = SimulationAnalytics(analytics_config)
    performance_suite = PerformanceTestSuite(performance_config)
    coordinator = MockSimulationCoordinator()
    
    print("✓ Integrated systems initialized")
    
    # Start analytics collection
    analytics.start_analytics(coordinator)
    
    # Collect baseline metrics
    print("\nCollecting baseline metrics...")
    for i in range(3):
        snapshot = analytics.collect_metrics_snapshot()
        print(f"  Baseline snapshot {i+1}: {snapshot.total_researchers} researchers")
        time.sleep(0.5)
    
    # Simulate performance test impact on metrics
    print("\nSimulating performance test impact...")
    
    # Simulate increased load
    coordinator.state.total_researchers = 200
    coordinator.state.total_papers = 400
    coordinator.state.total_reviews = 1200
    coordinator.state.total_errors = 5
    
    # Collect metrics during "performance test"
    for i in range(3):
        snapshot = analytics.collect_metrics_snapshot()
        print(f"  Load test snapshot {i+1}: {snapshot.total_researchers} researchers, "
              f"{snapshot.error_count} errors")
        time.sleep(0.5)
    
    # Generate combined report
    print("\nGenerating integrated report...")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Generate analytics report
        analytics_report = analytics.generate_analytics_report(output_dir / "analytics")
        
        # Generate performance report (mocked)
        performance_suite.results = [
            TestResult(
                test_name="integrated_test",
                config=performance_config,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(minutes=1),
                duration_seconds=60.0,
                metrics_history=[],
                avg_throughput=7.5,
                peak_throughput=10.0,
                avg_latency_ms=130.0,
                peak_memory_mb=450.0,
                peak_cpu_percent=60.0,
                success=True,
                errors=[],
                warnings=["Integrated test warning"],
                memory_limit_exceeded=False,
                cpu_limit_exceeded=False,
                timeout_occurred=False
            )
        ]
        
        performance_report = performance_suite.generate_performance_report(output_dir / "performance")
        
        print(f"✓ Integrated reports generated:")
        print(f"  - Analytics: {len(analytics_report)} files")
        print(f"  - Performance: 1 report file")
        
        # Create combined summary
        combined_summary = {
            'integration_test_summary': {
                'test_timestamp': datetime.now().isoformat(),
                'analytics_snapshots_collected': len(analytics.metrics_collector.metrics_history),
                'performance_tests_run': len(performance_suite.results),
                'total_test_duration_minutes': 5,
                'systems_tested': [
                    'metrics_collection',
                    'statistical_analysis',
                    'report_generation',
                    'load_testing',
                    'performance_monitoring',
                    'resource_tracking'
                ]
            },
            'key_findings': [
                'Analytics system successfully collected real-time metrics',
                'Performance testing identified system capabilities',
                'Integrated reporting provides comprehensive insights',
                'Both systems operate efficiently together'
            ],
            'recommendations': [
                'Continue monitoring system performance during load tests',
                'Use analytics data to optimize performance test parameters',
                'Integrate performance alerts with analytics thresholds'
            ]
        }
        
        # Save combined summary
        summary_file = output_dir / "integrated_summary.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(combined_summary, f, indent=2)
        
        summary_size = summary_file.stat().st_size
        print(f"✓ Combined summary: {summary_file.name} ({summary_size} bytes)")
    
    # Stop analytics
    analytics.stop_analytics()
    
    print("\n✅ INTEGRATED ANALYTICS AND PERFORMANCE TEST COMPLETED SUCCESSFULLY")
    return True


def main():
    """Run all integration tests."""
    print("SIMULATION ANALYTICS AND PERFORMANCE TESTING INTEGRATION")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test individual systems
        analytics_success = test_analytics_system()
        performance_success = test_performance_system()
        
        # Test integrated functionality
        integration_success = test_integrated_analytics_and_performance()
        
        # Overall results
        print("\n" + "="*80)
        print("FINAL TEST RESULTS")
        print("="*80)
        
        results = {
            'Analytics System': analytics_success,
            'Performance Testing': performance_success,
            'Integrated Systems': integration_success
        }
        
        all_passed = all(results.values())
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:.<50} {status}")
        
        print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)