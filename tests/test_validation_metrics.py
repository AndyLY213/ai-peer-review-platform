"""
Unit tests for ValidationMetrics class and real data validation framework.
"""

import pytest
import numpy as np
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.data.validation_metrics import (
    ValidationMetrics, StatisticalComparison, RealismIndicator, 
    ValidationAlert, BaselineStatistics
)
from src.data.peerread_loader import PeerReadLoader, VenueCharacteristics, PeerReadReview, PeerReadPaper
from src.data.enhanced_models import (
    StructuredReview, EnhancedResearcher, EnhancedVenue, 
    EnhancedReviewCriteria, ResearcherLevel, VenueType
)
from src.core.exceptions import ValidationError


class TestStatisticalComparison:
    """Test StatisticalComparison class."""
    
    def test_statistical_comparison_initialization(self):
        """Test StatisticalComparison initialization and calculations."""
        sim_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        real_values = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        comparison = StatisticalComparison("test_metric", sim_values, real_values)
        
        assert comparison.metric_name == "test_metric"
        assert comparison.simulation_values == sim_values
        assert comparison.real_values == real_values
        assert comparison.simulation_mean == 3.0
        assert abs(comparison.real_mean - 3.02) < 0.01
        assert comparison.wasserstein_distance is not None
        assert comparison.pearson_correlation is not None
        assert comparison.spearman_correlation is not None
        assert comparison.ks_statistic is not None
        assert comparison.similarity_score > 0
        assert comparison.deviation_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def test_statistical_comparison_empty_data(self):
        """Test StatisticalComparison with empty data."""
        comparison = StatisticalComparison("test_metric", [], [])
        
        assert comparison.simulation_mean == 0.0
        assert comparison.real_mean == 0.0
        assert comparison.wasserstein_distance is None
        assert comparison.similarity_score == 0.0
    
    def test_statistical_comparison_identical_data(self):
        """Test StatisticalComparison with identical data."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        comparison = StatisticalComparison("test_metric", values, values.copy())
        
        assert comparison.is_similar is True
        assert comparison.deviation_level == "LOW"
        assert comparison.similarity_score > 0.8
    
    def test_statistical_comparison_very_different_data(self):
        """Test StatisticalComparison with very different data."""
        sim_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        real_values = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        comparison = StatisticalComparison("test_metric", sim_values, real_values)
        
        assert comparison.is_similar is False
        assert comparison.deviation_level in ["HIGH", "CRITICAL"]
        assert comparison.similarity_score < 0.6


class TestRealismIndicator:
    """Test RealismIndicator class."""
    
    def test_realism_indicator_realistic(self):
        """Test RealismIndicator with realistic values."""
        indicator = RealismIndicator(
            aspect_name="review_quality",
            indicator_type="average_length",
            current_value=500.0,
            expected_value=480.0,
            tolerance=0.1
        )
        
        assert indicator.is_realistic is True
        assert indicator.alert_level == "OK"
        assert indicator.deviation_percentage < 0.1
    
    def test_realism_indicator_warning(self):
        """Test RealismIndicator with warning level deviation."""
        indicator = RealismIndicator(
            aspect_name="review_quality",
            indicator_type="average_length",
            current_value=600.0,
            expected_value=500.0,
            tolerance=0.1
        )
        
        assert indicator.is_realistic is False
        assert indicator.alert_level == "WARNING"
        assert 0.1 < indicator.deviation_percentage <= 0.2
    
    def test_realism_indicator_critical(self):
        """Test RealismIndicator with critical level deviation."""
        indicator = RealismIndicator(
            aspect_name="review_quality",
            indicator_type="average_length",
            current_value=800.0,
            expected_value=500.0,
            tolerance=0.1
        )
        
        assert indicator.is_realistic is False
        assert indicator.alert_level == "CRITICAL"
        assert indicator.deviation_percentage > 0.2
    
    def test_realism_indicator_zero_expected(self):
        """Test RealismIndicator with zero expected value."""
        indicator = RealismIndicator(
            aspect_name="test",
            indicator_type="test",
            current_value=5.0,
            expected_value=0.0,
            tolerance=0.1
        )
        
        assert indicator.deviation_percentage == 5.0


class TestValidationAlert:
    """Test ValidationAlert class."""
    
    def test_validation_alert_creation(self):
        """Test ValidationAlert creation and serialization."""
        alert = ValidationAlert(
            timestamp=datetime.now(),
            alert_level="WARNING",
            aspect="test_venue",
            metric="test_metric",
            message="Test alert message",
            current_value=10.0,
            expected_value=8.0,
            deviation_percentage=0.25
        )
        
        assert alert.alert_level == "WARNING"
        assert alert.aspect == "test_venue"
        assert alert.metric == "test_metric"
        
        # Test serialization
        alert_dict = alert.to_dict()
        assert isinstance(alert_dict, dict)
        assert alert_dict['alert_level'] == "WARNING"
        assert 'timestamp' in alert_dict


class TestBaselineStatistics:
    """Test BaselineStatistics class."""
    
    def test_baseline_statistics_serialization(self):
        """Test BaselineStatistics serialization and deserialization."""
        baseline = BaselineStatistics(
            venue_name="ACL",
            score_means={'impact': 3.2, 'substance': 3.4},
            score_stds={'impact': 0.8, 'substance': 0.9},
            review_length_mean=450.0,
            acceptance_rate=0.25
        )
        
        # Test serialization
        data = baseline.to_dict()
        assert isinstance(data, dict)
        assert data['venue_name'] == "ACL"
        assert data['score_means']['impact'] == 3.2
        
        # Test deserialization
        restored = BaselineStatistics.from_dict(data)
        assert restored.venue_name == "ACL"
        assert restored.score_means['impact'] == 3.2
        assert restored.acceptance_rate == 0.25


class TestValidationMetrics:
    """Test ValidationMetrics class."""
    
    @pytest.fixture
    def mock_peerread_loader(self):
        """Create mock PeerReadLoader."""
        loader = Mock(spec=PeerReadLoader)
        
        # Mock venue characteristics
        venue_chars = VenueCharacteristics(
            name="ACL",
            venue_type="conference",
            field="NLP",
            total_papers=100,
            accepted_papers=25,
            acceptance_rate=0.25,
            avg_reviews_per_paper=3.0,
            impact_scores=[3, 3, 4, 3, 4, 2, 3, 4, 3, 3],
            substance_scores=[3, 4, 3, 4, 3, 3, 4, 3, 4, 3],
            impact_mean=3.2,
            substance_mean=3.4
        )
        
        loader.load_all_venues.return_value = {"ACL": venue_chars}
        loader.get_venue_statistics.return_value = venue_chars
        
        # Mock papers with reviews
        review1 = PeerReadReview("paper1", "reviewer1")
        review1.impact = 3
        review1.substance = 4
        review1.comments = "This is a good paper with solid contributions."
        review1.reviewer_confidence = 4
        review1.recommendation = 1
        
        review2 = PeerReadReview("paper1", "reviewer2")
        review2.impact = 4
        review2.substance = 3
        review2.comments = "The paper has some interesting ideas but needs improvement."
        review2.reviewer_confidence = 3
        review2.recommendation = 2
        
        paper = PeerReadPaper("paper1", "Test Paper", "Test abstract")
        paper.venue = "ACL"
        paper.reviews = [review1, review2]
        
        loader.get_papers_by_venue.return_value = [paper]
        
        return loader
    
    @pytest.fixture
    def validation_metrics(self, mock_peerread_loader):
        """Create ValidationMetrics instance with mock loader."""
        return ValidationMetrics(mock_peerread_loader)
    
    def test_validation_metrics_initialization(self, validation_metrics):
        """Test ValidationMetrics initialization."""
        assert validation_metrics.monitoring_enabled is True
        assert len(validation_metrics.baseline_statistics) > 0
        assert "ACL" in validation_metrics.baseline_statistics
        assert isinstance(validation_metrics.validation_alerts, list)
    
    def test_compare_simulation_to_real(self, validation_metrics):
        """Test comparing simulation data to real data."""
        simulation_data = {
            'impact': [3.1, 3.3, 3.0, 3.4, 3.2, 3.1, 3.5, 3.0, 3.2, 3.3],
            'substance': [3.3, 3.5, 3.2, 3.6, 3.4, 3.3, 3.7, 3.2, 3.4, 3.5]
        }
        
        comparisons = validation_metrics.compare_simulation_to_real(simulation_data, "ACL")
        
        assert isinstance(comparisons, dict)
        assert 'impact' in comparisons
        assert 'substance' in comparisons
        
        impact_comparison = comparisons['impact']
        assert isinstance(impact_comparison, StatisticalComparison)
        assert impact_comparison.metric_name == 'impact'
        assert impact_comparison.wasserstein_distance is not None
    
    def test_compare_simulation_invalid_venue(self, validation_metrics):
        """Test comparing simulation data with invalid venue."""
        simulation_data = {'impact': [3.0, 3.1, 3.2]}
        
        with pytest.raises(ValidationError):
            validation_metrics.compare_simulation_to_real(simulation_data, "INVALID_VENUE")
    
    def test_calculate_kl_divergence(self, validation_metrics):
        """Test KL divergence calculation."""
        sim_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        real_values = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        kl_div = validation_metrics.calculate_kl_divergence(sim_values, real_values)
        assert kl_div is not None
        assert isinstance(kl_div, float)
        assert kl_div >= 0
    
    def test_calculate_wasserstein_distance(self, validation_metrics):
        """Test Wasserstein distance calculation."""
        sim_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        real_values = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        distance = validation_metrics.calculate_wasserstein_distance(sim_values, real_values)
        assert distance is not None
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_calculate_correlation_analysis(self, validation_metrics):
        """Test correlation analysis."""
        sim_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        real_values = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        correlations = validation_metrics.calculate_correlation_analysis(sim_values, real_values)
        
        assert isinstance(correlations, dict)
        assert 'pearson' in correlations
        assert 'spearman' in correlations
        
        pearson_corr, pearson_p = correlations['pearson']
        assert isinstance(pearson_corr, float)
        assert isinstance(pearson_p, float)
        assert -1 <= pearson_corr <= 1
    
    def test_calculate_correlation_different_lengths(self, validation_metrics):
        """Test correlation analysis with different length arrays."""
        sim_values = [1.0, 2.0, 3.0]
        real_values = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        correlations = validation_metrics.calculate_correlation_analysis(sim_values, real_values)
        
        # Should return empty dict or handle gracefully
        assert isinstance(correlations, dict)
    
    def test_calculate_baseline_statistics_from_peerread(self, validation_metrics):
        """Test calculating baseline statistics from PeerRead."""
        baseline = validation_metrics.calculate_baseline_statistics_from_peerread("ACL")
        
        assert isinstance(baseline, BaselineStatistics)
        assert baseline.venue_name == "ACL"
        assert baseline.acceptance_rate == 0.25
        assert 'impact' in baseline.score_means
        assert 'substance' in baseline.score_means
    
    def test_calculate_baseline_invalid_venue(self, validation_metrics):
        """Test calculating baseline for invalid venue."""
        with pytest.raises(ValidationError):
            validation_metrics.calculate_baseline_statistics_from_peerread("INVALID_VENUE")
    
    def test_enable_continuous_monitoring(self, validation_metrics):
        """Test enabling/disabling continuous monitoring."""
        validation_metrics.enable_continuous_monitoring(False)
        assert validation_metrics.monitoring_enabled is False
        
        validation_metrics.enable_continuous_monitoring(True)
        assert validation_metrics.monitoring_enabled is True
    
    def test_create_realism_indicators(self, validation_metrics):
        """Test creating realism indicators."""
        # Create mock simulation data
        review = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1"
        )
        review.review_length = 500
        review.criteria_scores = EnhancedReviewCriteria(
            novelty=6.0, technical_quality=7.0, clarity=6.5,
            significance=6.8, reproducibility=6.2, related_work=5.8
        )
        
        researcher = EnhancedResearcher(
            id="researcher1",
            name="Test Researcher",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF,
            h_index=12,
            years_active=8
        )
        
        venue = EnhancedVenue(
            id="venue1",
            name="ACL",
            venue_type=VenueType.MID_CONFERENCE,
            field="NLP",
            acceptance_rate=0.25
        )
        
        indicators = validation_metrics.create_realism_indicators([review], [researcher], [venue])
        
        assert isinstance(indicators, dict)
        assert 'review_quality' in indicators
        assert 'reviewer_behavior' in indicators
        assert 'venue_characteristics' in indicators
        
        # Check that indicators were created
        assert len(indicators['review_quality']) > 0
        assert len(indicators['reviewer_behavior']) > 0
        assert len(indicators['venue_characteristics']) > 0
        
        # Check indicator types
        for indicator in indicators['review_quality']:
            assert isinstance(indicator, RealismIndicator)
            assert indicator.aspect_name == "review_quality"
    
    def test_get_recent_alerts(self, validation_metrics):
        """Test getting recent alerts."""
        # Add some test alerts
        old_alert = ValidationAlert(
            timestamp=datetime.now() - timedelta(hours=48),
            alert_level="WARNING",
            aspect="test",
            metric="test",
            message="Old alert",
            current_value=1.0,
            expected_value=2.0,
            deviation_percentage=0.5
        )
        
        recent_alert = ValidationAlert(
            timestamp=datetime.now() - timedelta(hours=1),
            alert_level="CRITICAL",
            aspect="test",
            metric="test",
            message="Recent alert",
            current_value=1.0,
            expected_value=2.0,
            deviation_percentage=0.5
        )
        
        validation_metrics.validation_alerts = [old_alert, recent_alert]
        
        recent_alerts = validation_metrics.get_recent_alerts(24)
        assert len(recent_alerts) == 1
        assert recent_alerts[0].message == "Recent alert"
    
    def test_clear_alerts(self, validation_metrics):
        """Test clearing alerts."""
        # Add test alert
        alert = ValidationAlert(
            timestamp=datetime.now(),
            alert_level="WARNING",
            aspect="test",
            metric="test",
            message="Test alert",
            current_value=1.0,
            expected_value=2.0,
            deviation_percentage=0.5
        )
        validation_metrics.validation_alerts = [alert]
        
        assert len(validation_metrics.validation_alerts) == 1
        
        validation_metrics.clear_alerts()
        assert len(validation_metrics.validation_alerts) == 0
    
    def test_export_validation_report(self, validation_metrics):
        """Test exporting validation report."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            validation_metrics.export_validation_report(temp_path)
            
            # Verify file was created and contains expected data
            with open(temp_path, 'r') as f:
                report = json.load(f)
            
            assert 'timestamp' in report
            assert 'baseline_statistics' in report
            assert 'recent_alerts' in report
            assert 'alert_thresholds' in report
            assert 'monitoring_enabled' in report
            
        finally:
            Path(temp_path).unlink()
    
    def test_export_import_baseline_statistics(self, validation_metrics):
        """Test exporting and importing baseline statistics."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export baseline statistics
            validation_metrics.export_baseline_statistics(temp_path)
            
            # Clear current statistics
            original_stats = validation_metrics.baseline_statistics.copy()
            validation_metrics.baseline_statistics.clear()
            
            # Import statistics
            validation_metrics.import_baseline_statistics(temp_path)
            
            # Verify statistics were restored
            assert len(validation_metrics.baseline_statistics) > 0
            assert "ACL" in validation_metrics.baseline_statistics
            
        finally:
            Path(temp_path).unlink()
    
    def test_import_baseline_invalid_file(self, validation_metrics):
        """Test importing baseline from invalid file."""
        with pytest.raises(ValidationError):
            validation_metrics.import_baseline_statistics("nonexistent_file.json")
    
    @patch('src.data.validation_metrics.logger')
    def test_alert_generation(self, mock_logger, validation_metrics):
        """Test automatic alert generation during monitoring."""
        # Enable monitoring
        validation_metrics.enable_continuous_monitoring(True)
        
        # Create simulation data with significant deviation
        simulation_data = {
            'impact': [5.0, 5.1, 5.2, 5.3, 5.4] * 10  # Much higher than baseline ~3.2
        }
        
        # This should trigger alerts
        comparisons = validation_metrics.compare_simulation_to_real(simulation_data, "ACL")
        
        # Check that alerts were generated
        assert len(validation_metrics.validation_alerts) > 0
        
        # Check that warning was logged
        mock_logger.warning.assert_called()
    
    def test_field_mapping_to_peerread(self, validation_metrics):
        """Test field mapping to PeerRead format."""
        assert validation_metrics._map_field_to_peerread('novelty') == 'originality'
        assert validation_metrics._map_field_to_peerread('technical_quality') == 'substance'
        assert validation_metrics._map_field_to_peerread('significance') == 'impact'
        assert validation_metrics._map_field_to_peerread('unknown_field') == 'unknown_field'


class TestIntegration:
    """Integration tests for validation framework."""
    
    @pytest.fixture
    def real_peerread_loader(self):
        """Create real PeerReadLoader for integration tests."""
        # This would use actual PeerRead data if available
        # For testing, we'll mock it but with more realistic data
        loader = Mock(spec=PeerReadLoader)
        
        # Create more realistic venue characteristics
        acl_chars = VenueCharacteristics(
            name="ACL",
            venue_type="conference",
            field="NLP",
            total_papers=500,
            accepted_papers=125,
            acceptance_rate=0.25,
            avg_reviews_per_paper=3.2,
            impact_scores=list(np.random.normal(3.2, 0.8, 100).astype(int).clip(1, 5)),
            substance_scores=list(np.random.normal(3.4, 0.9, 100).astype(int).clip(1, 5)),
            impact_mean=3.2,
            substance_mean=3.4
        )
        
        loader.load_all_venues.return_value = {"ACL": acl_chars}
        loader.get_venue_statistics.return_value = acl_chars
        
        # Create realistic papers and reviews
        papers = []
        for i in range(10):
            paper = PeerReadPaper(f"paper_{i}", f"Paper {i}", f"Abstract {i}")
            paper.venue = "ACL"
            
            # Add reviews with realistic variation
            for j in range(3):
                review = PeerReadReview(f"paper_{i}", f"reviewer_{j}")
                review.impact = np.random.choice([2, 3, 3, 4, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
                review.substance = np.random.choice([2, 3, 3, 4, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
                review.comments = f"Review {j} for paper {i}. " * np.random.randint(10, 50)
                review.reviewer_confidence = np.random.randint(2, 5)
                review.recommendation = np.random.randint(1, 5)
                paper.reviews.append(review)
            
            papers.append(paper)
        
        loader.get_papers_by_venue.return_value = papers
        return loader
    
    def test_full_validation_workflow(self, real_peerread_loader):
        """Test complete validation workflow."""
        # Initialize validation metrics
        validation_metrics = ValidationMetrics(real_peerread_loader)
        
        # Create realistic simulation data
        simulation_data = {
            'impact': list(np.random.normal(3.3, 0.7, 50)),  # Slightly different from baseline
            'substance': list(np.random.normal(3.5, 0.8, 50))
        }
        
        # Compare simulation to real data
        comparisons = validation_metrics.compare_simulation_to_real(simulation_data, "ACL")
        
        # Verify comparisons
        assert len(comparisons) == 2
        for metric_name, comparison in comparisons.items():
            assert isinstance(comparison, StatisticalComparison)
            assert comparison.similarity_score > 0
            assert comparison.deviation_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        # Create realism indicators
        review = StructuredReview("reviewer1", "paper1", "venue1")
        review.review_length = 450
        
        researcher = EnhancedResearcher("researcher1", "Test", "AI")
        venue = EnhancedVenue("venue1", "ACL", VenueType.MID_CONFERENCE, "NLP")
        
        indicators = validation_metrics.create_realism_indicators([review], [researcher], [venue])
        
        # Verify indicators
        assert len(indicators['review_quality']) > 0
        assert len(indicators['reviewer_behavior']) > 0
        assert len(indicators['venue_characteristics']) > 0
        
        # Test export/import cycle
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            validation_metrics.export_validation_report(temp_path)
            
            with open(temp_path, 'r') as f:
                report = json.load(f)
            
            assert 'baseline_statistics' in report
            assert 'ACL' in report['baseline_statistics']
            
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])