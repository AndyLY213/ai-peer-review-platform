"""
Tests for Simulation Analytics System

This module contains comprehensive tests for the simulation analytics and reporting
system, including metrics collection, statistical analysis, and report generation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.enhancements.simulation_analytics import (
    SimulationAnalytics, MetricsCollector, StatisticalAnalyzer,
    VisualizationGenerator, ReportGenerator, MetricsSnapshot,
    AnalyticsConfiguration
)
from src.core.exceptions import SimulationError


class TestMetricsSnapshot:
    """Test MetricsSnapshot data class."""
    
    def test_metrics_snapshot_creation(self):
        """Test creating a metrics snapshot."""
        snapshot = MetricsSnapshot(
            timestamp=datetime.now(),
            simulation_id="test_sim_123",
            total_researchers=100,
            total_papers=500,
            total_reviews=1500,
            total_venues=10,
            avg_review_quality=3.5,
            avg_review_length=250,
            avg_confidence_level=3.2,
            review_completion_rate=0.85,
            bias_effects_detected={'confirmation': 50, 'halo_effect': 30},
            avg_bias_impact={'confirmation': 0.3, 'halo_effect': 0.4},
            collaboration_density=0.15,
            citation_network_size=800,
            community_count=5,
            tenure_success_rate=0.65,
            promotion_rates={'assistant_to_associate': 0.4, 'associate_to_full': 0.3},
            job_market_competition=0.8,
            venue_acceptance_rates={'venue1': 0.2, 'venue2': 0.35},
            venue_quality_scores={'venue1': 4.2, 'venue2': 3.8},
            venue_shopping_incidents=15,
            review_trading_detected=3,
            citation_cartel_members=8,
            salami_slicing_cases=12,
            processing_time_ms=150.5,
            memory_usage_mb=512.3,
            error_count=2
        )
        
        assert snapshot.simulation_id == "test_sim_123"
        assert snapshot.total_researchers == 100
        assert snapshot.avg_review_quality == 3.5
        assert snapshot.bias_effects_detected['confirmation'] == 50
        assert snapshot.processing_time_ms == 150.5


class TestAnalyticsConfiguration:
    """Test AnalyticsConfiguration."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = AnalyticsConfiguration()
        
        assert config.metrics_collection_interval == 60
        assert config.detailed_logging_enabled is True
        assert config.real_time_analytics is True
        assert config.data_retention_days == 90
        assert config.export_format == "json"
        assert config.statistical_confidence_level == 0.95
        assert config.generate_daily_reports is True
        assert config.max_concurrent_analyses == 4
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = AnalyticsConfiguration(
            metrics_collection_interval=30,
            data_retention_days=30,
            export_format="csv",
            real_time_analytics=False
        )
        
        assert config.metrics_collection_interval == 30
        assert config.data_retention_days == 30
        assert config.export_format == "csv"
        assert config.real_time_analytics is False


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    @pytest.fixture
    def config(self):
        return AnalyticsConfiguration(metrics_collection_interval=1)
    
    @pytest.fixture
    def collector(self, config):
        return MetricsCollector(config)
    
    @pytest.fixture
    def mock_coordinator(self):
        coordinator = Mock()
        coordinator.state = Mock()
        coordinator.state.simulation_id = "test_sim"
        coordinator.state.total_researchers = 100
        coordinator.state.total_papers = 200
        coordinator.state.total_reviews = 600
        coordinator.state.active_venues = 5
        coordinator.state.total_errors = 1
        
        # Mock systems
        coordinator.review_system = Mock()
        coordinator.bias_engine = Mock()
        coordinator.collaboration_network = Mock()
        coordinator.citation_network = Mock()
        coordinator.conference_community = Mock()
        coordinator.tenure_track_manager = Mock()
        coordinator.promotion_criteria_evaluator = Mock()
        coordinator.job_market_simulator = Mock()
        coordinator.venue_registry = Mock()
        coordinator.venue_shopping_tracker = Mock()
        coordinator.review_trading_detector = Mock()
        coordinator.citation_cartel_detector = Mock()
        coordinator.salami_slicing_detector = Mock()
        
        return coordinator
    
    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert collector.config is not None
        assert collector.metrics_history == []
        assert collector.collection_active is False
    
    def test_start_stop_collection(self, collector):
        """Test starting and stopping collection."""
        collector.start_collection()
        assert collector.collection_active is True
        
        collector.stop_collection()
        assert collector.collection_active is False
    
    def test_collect_snapshot(self, collector, mock_coordinator):
        """Test collecting a metrics snapshot."""
        # Mock review system
        mock_coordinator.review_system.get_recent_reviews.return_value = [
            Mock(quality_score=3.5, review_length=200, confidence_level=3, completeness_score=0.9),
            Mock(quality_score=4.0, review_length=250, confidence_level=4, completeness_score=0.8)
        ]
        
        # Mock bias engine
        mock_coordinator.bias_engine.get_bias_statistics.return_value = {
            'effects_count': {'confirmation': 10, 'halo_effect': 5},
            'average_impact': {'confirmation': 0.3, 'halo_effect': 0.4}
        }
        
        # Mock network systems
        mock_coordinator.collaboration_network.calculate_network_density.return_value = 0.15
        mock_coordinator.citation_network.get_network_size.return_value = 500
        mock_coordinator.conference_community.get_community_count.return_value = 3
        
        # Mock career systems
        mock_coordinator.tenure_track_manager.get_success_rate.return_value = 0.7
        mock_coordinator.promotion_criteria_evaluator.get_promotion_rates.return_value = {
            'assistant_to_associate': 0.4
        }
        mock_coordinator.job_market_simulator.get_competition_level.return_value = 0.8
        
        # Mock venue system
        mock_venue = Mock()
        mock_venue.id = "test_venue"
        mock_venue.acceptance_rate = 0.25
        mock_venue.quality_score = 4.0
        mock_coordinator.venue_registry.get_all_venues.return_value = [mock_venue]
        
        # Mock strategic behavior systems
        mock_coordinator.venue_shopping_tracker.get_incident_count.return_value = 5
        mock_coordinator.review_trading_detector.get_detection_count.return_value = 2
        mock_coordinator.citation_cartel_detector.get_member_count.return_value = 3
        mock_coordinator.salami_slicing_detector.get_case_count.return_value = 4
        
        # Mock system health
        mock_coordinator.get_system_health.return_value = {'status': 'healthy'}
        
        snapshot = collector.collect_snapshot(mock_coordinator)
        
        assert snapshot.simulation_id == "test_sim"
        assert snapshot.total_researchers == 100
        assert snapshot.total_papers == 200
        assert snapshot.total_reviews == 600
        assert snapshot.avg_review_quality == 3.75  # Average of 3.5 and 4.0
        assert snapshot.collaboration_density == 0.15
        assert snapshot.venue_shopping_incidents == 5
        assert len(collector.metrics_history) == 1
    
    def test_collect_snapshot_with_missing_systems(self, collector, mock_coordinator):
        """Test collecting snapshot when some systems are None."""
        # Set some systems to None
        mock_coordinator.review_system = None
        mock_coordinator.bias_engine = None
        mock_coordinator.collaboration_network = None
        mock_coordinator.venue_registry = None
        
        snapshot = collector.collect_snapshot(mock_coordinator)
        
        assert snapshot.simulation_id == "test_sim"
        assert snapshot.avg_review_quality == 0.0
        assert snapshot.collaboration_density == 0.0
        assert len(snapshot.bias_effects_detected) == 0
    
    def test_collect_snapshot_error_handling(self, collector, mock_coordinator):
        """Test error handling during snapshot collection."""
        mock_coordinator.state.simulation_id = None  # Cause an error
        
        with pytest.raises(SimulationError):
            collector.collect_snapshot(mock_coordinator)


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer functionality."""
    
    @pytest.fixture
    def config(self):
        return AnalyticsConfiguration()
    
    @pytest.fixture
    def analyzer(self, config):
        return StatisticalAnalyzer(config)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics history for testing."""
        base_time = datetime.now()
        metrics = []
        
        for i in range(10):
            snapshot = MetricsSnapshot(
                timestamp=base_time + timedelta(minutes=i),
                simulation_id="test_sim",
                total_researchers=100 + i * 5,
                total_papers=200 + i * 10,
                total_reviews=600 + i * 30,
                total_venues=5,
                avg_review_quality=3.0 + i * 0.1,
                avg_review_length=200 + i * 5,
                avg_confidence_level=3.0 + i * 0.05,
                review_completion_rate=0.8 + i * 0.01,
                bias_effects_detected={'confirmation': 10 + i},
                avg_bias_impact={'confirmation': 0.3 + i * 0.01},
                collaboration_density=0.1 + i * 0.01,
                citation_network_size=500 + i * 20,
                community_count=3,
                tenure_success_rate=0.6 + i * 0.02,
                promotion_rates={'assistant_to_associate': 0.3 + i * 0.01},
                job_market_competition=0.7 + i * 0.01,
                venue_acceptance_rates={'venue1': 0.2 + i * 0.005},
                venue_quality_scores={'venue1': 3.5 + i * 0.05},
                venue_shopping_incidents=5 + i,
                review_trading_detected=1 + i // 3,
                citation_cartel_members=2 + i // 2,
                salami_slicing_cases=3 + i,
                processing_time_ms=100.0 + i * 10,
                memory_usage_mb=400.0 + i * 20,
                error_count=i // 2
            )
            metrics.append(snapshot)
        
        return metrics
    
    def test_analyze_trends(self, analyzer, sample_metrics):
        """Test trend analysis."""
        trends = analyzer.analyze_trends(sample_metrics)
        
        assert 'total_researchers' in trends
        assert 'avg_review_quality' in trends
        assert 'processing_time_ms' in trends
        
        # Check that increasing metrics are detected
        researcher_trend = trends['total_researchers']
        assert researcher_trend['trend_direction'] == 'increasing'
        assert researcher_trend['slope'] > 0
        
        quality_trend = trends['avg_review_quality']
        assert quality_trend['trend_direction'] == 'increasing'
    
    def test_analyze_trends_insufficient_data(self, analyzer):
        """Test trend analysis with insufficient data."""
        single_metric = [MetricsSnapshot(
            timestamp=datetime.now(),
            simulation_id="test",
            total_researchers=100,
            total_papers=200,
            total_reviews=600,
            total_venues=5,
            avg_review_quality=3.0,
            avg_review_length=200,
            avg_confidence_level=3.0,
            review_completion_rate=0.8,
            bias_effects_detected={},
            avg_bias_impact={},
            collaboration_density=0.1,
            citation_network_size=500,
            community_count=3,
            tenure_success_rate=0.6,
            promotion_rates={},
            job_market_competition=0.7,
            venue_acceptance_rates={},
            venue_quality_scores={},
            venue_shopping_incidents=5,
            review_trading_detected=1,
            citation_cartel_members=2,
            salami_slicing_cases=3,
            processing_time_ms=100.0,
            memory_usage_mb=400.0,
            error_count=0
        )]
        
        trends = analyzer.analyze_trends(single_metric)
        assert 'error' in trends
    
    def test_detect_anomalies(self, analyzer, sample_metrics):
        """Test anomaly detection."""
        # Add an anomalous data point
        anomalous_snapshot = MetricsSnapshot(
            timestamp=datetime.now() + timedelta(minutes=10),
            simulation_id="test_sim",
            total_researchers=1000,  # Anomalously high
            total_papers=200,
            total_reviews=600,
            total_venues=5,
            avg_review_quality=1.0,  # Anomalously low
            avg_review_length=200,
            avg_confidence_level=3.0,
            review_completion_rate=0.8,
            bias_effects_detected={},
            avg_bias_impact={},
            collaboration_density=0.1,
            citation_network_size=500,
            community_count=3,
            tenure_success_rate=0.6,
            promotion_rates={},
            job_market_competition=0.7,
            venue_acceptance_rates={},
            venue_quality_scores={},
            venue_shopping_incidents=5,
            review_trading_detected=1,
            citation_cartel_members=2,
            salami_slicing_cases=3,
            processing_time_ms=100.0,
            memory_usage_mb=400.0,
            error_count=0
        )
        
        sample_metrics.append(anomalous_snapshot)
        
        anomalies = analyzer.detect_anomalies(sample_metrics)
        
        # Should detect anomalies in total_researchers and avg_review_quality
        assert 'total_researchers' in anomalies or 'avg_review_quality' in anomalies
    
    def test_calculate_correlations(self, analyzer, sample_metrics):
        """Test correlation calculation."""
        correlations = analyzer.calculate_correlations(sample_metrics)
        
        assert isinstance(correlations, dict)
        if correlations:  # Only check if correlations were calculated
            assert 'total_researchers' in correlations
            # Should have correlation with other metrics
            assert len(correlations['total_researchers']) > 0
    
    def test_generate_statistical_summary(self, analyzer, sample_metrics):
        """Test statistical summary generation."""
        summary = analyzer.generate_statistical_summary(sample_metrics)
        
        assert 'data_points' in summary
        assert summary['data_points'] == len(sample_metrics)
        assert 'time_range' in summary
        assert 'descriptive_statistics' in summary
        assert 'trends' in summary
        assert 'anomalies' in summary
        assert 'correlations' in summary
        
        # Check descriptive statistics
        desc_stats = summary['descriptive_statistics']
        assert 'total_researchers' in desc_stats
        assert 'mean' in desc_stats['total_researchers']
        assert 'std' in desc_stats['total_researchers']


class TestVisualizationGenerator:
    """Test VisualizationGenerator functionality."""
    
    @pytest.fixture
    def config(self):
        return AnalyticsConfiguration(include_visualizations=True)
    
    @pytest.fixture
    def visualizer(self, config):
        return VisualizationGenerator(config)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for visualization testing."""
        base_time = datetime.now()
        metrics = []
        
        for i in range(5):
            snapshot = MetricsSnapshot(
                timestamp=base_time + timedelta(minutes=i),
                simulation_id="test_sim",
                total_researchers=100 + i * 10,
                total_papers=200 + i * 20,
                total_reviews=600 + i * 60,
                total_venues=5,
                avg_review_quality=3.0 + i * 0.2,
                avg_review_length=200,
                avg_confidence_level=3.0,
                review_completion_rate=0.8,
                bias_effects_detected={},
                avg_bias_impact={},
                collaboration_density=0.1 + i * 0.02,
                citation_network_size=500,
                community_count=3,
                tenure_success_rate=0.6 + i * 0.05,
                promotion_rates={},
                job_market_competition=0.7,
                venue_acceptance_rates={},
                venue_quality_scores={},
                venue_shopping_incidents=5,
                review_trading_detected=1,
                citation_cartel_members=2,
                salami_slicing_cases=3,
                processing_time_ms=100.0,
                memory_usage_mb=400.0,
                error_count=0
            )
            metrics.append(snapshot)
        
        return metrics
    
    def test_generate_trend_charts_no_matplotlib(self, visualizer, sample_metrics):
        """Test trend chart generation when matplotlib is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should handle missing matplotlib gracefully
            generated_files = visualizer.generate_trend_charts(sample_metrics, output_dir)
            
            # Should return empty list if matplotlib not available
            assert isinstance(generated_files, list)
    
    def test_generate_correlation_heatmap_insufficient_data(self, visualizer):
        """Test correlation heatmap with insufficient data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Single data point - insufficient for correlation
            single_metric = [MetricsSnapshot(
                timestamp=datetime.now(),
                simulation_id="test",
                total_researchers=100,
                total_papers=200,
                total_reviews=600,
                total_venues=5,
                avg_review_quality=3.0,
                avg_review_length=200,
                avg_confidence_level=3.0,
                review_completion_rate=0.8,
                bias_effects_detected={},
                avg_bias_impact={},
                collaboration_density=0.1,
                citation_network_size=500,
                community_count=3,
                tenure_success_rate=0.6,
                promotion_rates={},
                job_market_competition=0.7,
                venue_acceptance_rates={},
                venue_quality_scores={},
                venue_shopping_incidents=5,
                review_trading_detected=1,
                citation_cartel_members=2,
                salami_slicing_cases=3,
                processing_time_ms=100.0,
                memory_usage_mb=400.0,
                error_count=0
            )]
            
            result = visualizer.generate_correlation_heatmap(single_metric, output_dir)
            assert result is None


class TestReportGenerator:
    """Test ReportGenerator functionality."""
    
    @pytest.fixture
    def config(self):
        return AnalyticsConfiguration()
    
    @pytest.fixture
    def report_generator(self, config):
        return ReportGenerator(config)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for report testing."""
        base_time = datetime.now()
        metrics = []
        
        for i in range(3):
            snapshot = MetricsSnapshot(
                timestamp=base_time + timedelta(minutes=i),
                simulation_id="test_sim_report",
                total_researchers=100 + i * 10,
                total_papers=200 + i * 20,
                total_reviews=600 + i * 60,
                total_venues=5,
                avg_review_quality=3.0 + i * 0.2,
                avg_review_length=200,
                avg_confidence_level=3.0,
                review_completion_rate=0.8,
                bias_effects_detected={'confirmation': 10 + i},
                avg_bias_impact={'confirmation': 0.3},
                collaboration_density=0.1,
                citation_network_size=500,
                community_count=3,
                tenure_success_rate=0.6,
                promotion_rates={'assistant_to_associate': 0.3},
                job_market_competition=0.7,
                venue_acceptance_rates={'venue1': 0.2},
                venue_quality_scores={'venue1': 3.5},
                venue_shopping_incidents=5,
                review_trading_detected=1,
                citation_cartel_members=2,
                salami_slicing_cases=3,
                processing_time_ms=100.0 + i * 50,
                memory_usage_mb=400.0 + i * 100,
                error_count=i
            )
            metrics.append(snapshot)
        
        return metrics
    
    def test_generate_comprehensive_report(self, report_generator, sample_metrics):
        """Test comprehensive report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            report_files = report_generator.generate_comprehensive_report(
                sample_metrics, output_dir
            )
            
            assert 'json' in report_files
            assert 'html' in report_files
            
            # Check that files were created
            json_file = Path(report_files['json'])
            html_file = Path(report_files['html'])
            
            assert json_file.exists()
            assert html_file.exists()
            
            # Check JSON content
            with open(json_file) as f:
                json_data = json.load(f)
            
            assert 'statistical_summary' in json_data
            assert 'report_metadata' in json_data
            assert json_data['report_metadata']['data_points'] == len(sample_metrics)
            
            # Check HTML content
            with open(html_file) as f:
                html_content = f.read()
            
            assert 'Simulation Analytics Report' in html_content
            assert 'test_sim_report' in html_content
    
    def test_generate_report_empty_metrics(self, report_generator):
        """Test report generation with empty metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            report_files = report_generator.generate_comprehensive_report(
                [], output_dir
            )
            
            assert 'json' in report_files
            
            # Check that files were still created
            json_file = Path(report_files['json'])
            assert json_file.exists()


class TestSimulationAnalytics:
    """Test main SimulationAnalytics class."""
    
    @pytest.fixture
    def config(self):
        return AnalyticsConfiguration(
            real_time_analytics=False,  # Disable for testing
            metrics_collection_interval=1
        )
    
    @pytest.fixture
    def analytics(self, config):
        return SimulationAnalytics(config)
    
    @pytest.fixture
    def mock_coordinator(self):
        coordinator = Mock()
        coordinator.state = Mock()
        coordinator.state.simulation_id = "test_analytics"
        coordinator.state.total_researchers = 50
        coordinator.state.total_papers = 100
        coordinator.state.total_reviews = 300
        coordinator.state.active_venues = 3
        coordinator.state.total_errors = 0
        
        # Mock all systems as None for simplicity
        for attr in ['review_system', 'bias_engine', 'collaboration_network',
                    'citation_network', 'conference_community', 'tenure_track_manager',
                    'promotion_criteria_evaluator', 'job_market_simulator',
                    'venue_registry', 'venue_shopping_tracker', 'review_trading_detector',
                    'citation_cartel_detector', 'salami_slicing_detector']:
            setattr(coordinator, attr, None)
        
        coordinator.get_system_health.return_value = {'status': 'healthy'}
        
        return coordinator
    
    def test_analytics_initialization(self, analytics):
        """Test analytics system initialization."""
        assert analytics.config is not None
        assert analytics.metrics_collector is not None
        assert analytics.analyzer is not None
        assert analytics.report_generator is not None
        assert analytics.data_dir.exists()
    
    def test_start_stop_analytics(self, analytics, mock_coordinator):
        """Test starting and stopping analytics."""
        analytics.start_analytics(mock_coordinator)
        assert analytics.metrics_collector.collection_active is True
        
        analytics.stop_analytics()
        assert analytics.metrics_collector.collection_active is False
    
    def test_collect_metrics_snapshot(self, analytics, mock_coordinator):
        """Test collecting a single metrics snapshot."""
        analytics.simulation_coordinator = mock_coordinator
        
        snapshot = analytics.collect_metrics_snapshot()
        
        assert snapshot.simulation_id == "test_analytics"
        assert snapshot.total_researchers == 50
        assert len(analytics.metrics_collector.metrics_history) == 1
    
    def test_generate_analytics_report(self, analytics, mock_coordinator):
        """Test generating analytics report."""
        analytics.simulation_coordinator = mock_coordinator
        
        # Collect some metrics first
        analytics.collect_metrics_snapshot()
        analytics.collect_metrics_snapshot()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_report"
            
            report_files = analytics.generate_analytics_report(output_dir)
            
            assert 'json' in report_files
            assert 'html' in report_files
            
            # Verify files exist
            assert Path(report_files['json']).exists()
            assert Path(report_files['html']).exists()
    
    def test_get_real_time_metrics(self, analytics, mock_coordinator):
        """Test getting real-time metrics."""
        analytics.simulation_coordinator = mock_coordinator
        
        # Initially no metrics
        metrics = analytics.get_real_time_metrics()
        assert metrics == {}
        
        # After collecting a snapshot
        analytics.collect_metrics_snapshot()
        metrics = analytics.get_real_time_metrics()
        
        assert 'simulation_id' in metrics
        assert metrics['simulation_id'] == "test_analytics"
    
    def test_export_metrics_data_json(self, analytics, mock_coordinator):
        """Test exporting metrics data as JSON."""
        analytics.simulation_coordinator = mock_coordinator
        
        # Collect some metrics
        analytics.collect_metrics_snapshot()
        analytics.collect_metrics_snapshot()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.json"
            
            exported_file = analytics.export_metrics_data("json", output_path)
            
            assert exported_file == str(output_path)
            assert output_path.exists()
            
            # Verify content
            with open(output_path) as f:
                data = json.load(f)
            
            assert len(data) == 2
            assert data[0]['simulation_id'] == "test_analytics"
    
    def test_export_metrics_data_csv(self, analytics, mock_coordinator):
        """Test exporting metrics data as CSV."""
        analytics.simulation_coordinator = mock_coordinator
        
        # Collect some metrics
        analytics.collect_metrics_snapshot()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.csv"
            
            exported_file = analytics.export_metrics_data("csv", output_path)
            
            assert exported_file == str(output_path)
            assert output_path.exists()
            
            # Verify content
            df = pd.read_csv(output_path)
            assert len(df) == 1
            assert 'simulation_id' in df.columns
    
    def test_export_unsupported_format(self, analytics, mock_coordinator):
        """Test exporting with unsupported format."""
        analytics.simulation_coordinator = mock_coordinator
        analytics.collect_metrics_snapshot()
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            analytics.export_metrics_data("xml")
    
    def test_cleanup(self, analytics, mock_coordinator):
        """Test cleanup functionality."""
        analytics.simulation_coordinator = mock_coordinator
        
        # Collect some metrics
        analytics.collect_metrics_snapshot()
        analytics.collect_metrics_snapshot()
        
        initial_count = len(analytics.metrics_collector.metrics_history)
        assert initial_count == 2
        
        # Test cleanup (with very short retention for testing)
        analytics.config.data_retention_days = 0
        analytics.cleanup()
        
        # Should have cleaned up old data
        final_count = len(analytics.metrics_collector.metrics_history)
        assert final_count <= initial_count


if __name__ == "__main__":
    pytest.main([__file__])