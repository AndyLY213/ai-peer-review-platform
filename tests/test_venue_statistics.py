"""
Unit tests for the venue statistics system.

Tests venue statistics tracking, historical trend analysis, dynamic acceptance
rate calculation, continuous calibration, and realism validation metrics.
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import statistics

from src.enhancements.venue_statistics import (
    VenueStats, SubmissionRecord, AcceptanceRecord, TrendData, RealismMetrics
)
from src.data.enhanced_models import EnhancedVenue, VenueType
from src.data.peerread_loader import VenueCharacteristics
from src.core.exceptions import ValidationError, DatasetError


class TestSubmissionRecord:
    """Test SubmissionRecord dataclass."""
    
    def test_submission_record_creation(self):
        """Test creating a submission record."""
        record = SubmissionRecord(
            paper_id="paper-123",
            venue_id="venue-456",
            submission_date=datetime(2024, 1, 15)
        )
        
        assert record.paper_id == "paper-123"
        assert record.venue_id == "venue-456"
        assert record.submission_date == datetime(2024, 1, 15)
        assert record.decision is None
        assert record.average_score is None
    
    def test_submission_record_with_scores(self):
        """Test submission record with review scores."""
        scores = [7.5, 8.0, 6.5, 7.0]
        record = SubmissionRecord(
            paper_id="paper-123",
            venue_id="venue-456",
            submission_date=datetime(2024, 1, 15),
            review_scores=scores
        )
        
        assert record.review_scores == scores
        assert record.average_score == 7.25
    
    def test_submission_record_decision_update(self):
        """Test updating submission record with decision."""
        record = SubmissionRecord(
            paper_id="paper-123",
            venue_id="venue-456",
            submission_date=datetime(2024, 1, 15)
        )
        
        record.decision = "accept"
        record.decision_date = datetime(2024, 2, 15)
        record.review_scores = [8.0, 7.5, 8.5]
        record.reviewer_count = 3
        
        assert record.decision == "accept"
        assert record.decision_date == datetime(2024, 2, 15)
        assert len(record.review_scores) == 3
        assert record.reviewer_count == 3


class TestAcceptanceRecord:
    """Test AcceptanceRecord dataclass."""
    
    def test_acceptance_record_creation(self):
        """Test creating an acceptance record."""
        record = AcceptanceRecord(
            venue_id="venue-123",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_submissions=100,
            total_acceptances=25,
            acceptance_rate=0.25,
            average_review_score=7.2
        )
        
        assert record.venue_id == "venue-123"
        assert record.total_submissions == 100
        assert record.total_acceptances == 25
        assert record.acceptance_rate == 0.25
        assert record.average_review_score == 7.2
    
    def test_acceptance_record_validation_warning(self, caplog):
        """Test acceptance record validation warning for mismatched rates."""
        record = AcceptanceRecord(
            venue_id="venue-123",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_submissions=100,
            total_acceptances=25,
            acceptance_rate=0.30,  # Mismatched rate (should be 0.25)
            average_review_score=7.2
        )
        
        # Check that warning was logged
        assert "Acceptance rate mismatch" in caplog.text


class TestTrendData:
    """Test TrendData class."""
    
    def test_trend_data_creation(self):
        """Test creating trend data."""
        trend = TrendData(
            venue_id="venue-123",
            metric_name="acceptance_rate"
        )
        
        assert trend.venue_id == "venue-123"
        assert trend.metric_name == "acceptance_rate"
        assert len(trend.time_series) == 0
        assert trend.trend_direction is None
    
    def test_add_data_point(self):
        """Test adding data points to trend."""
        trend = TrendData(
            venue_id="venue-123",
            metric_name="acceptance_rate"
        )
        
        # Add data points
        trend.add_data_point(datetime(2024, 1, 1), 0.25)
        trend.add_data_point(datetime(2024, 2, 1), 0.30)
        trend.add_data_point(datetime(2024, 3, 1), 0.35)
        
        assert len(trend.time_series) == 3
        assert trend.time_series[0] == (datetime(2024, 1, 1), 0.25)
        assert trend.trend_direction == "increasing"
        assert trend.trend_strength > 0.8  # Strong positive correlation
    
    def test_trend_calculation_decreasing(self):
        """Test trend calculation for decreasing values."""
        trend = TrendData(
            venue_id="venue-123",
            metric_name="acceptance_rate"
        )
        
        # Add decreasing data points
        trend.add_data_point(datetime(2024, 1, 1), 0.35)
        trend.add_data_point(datetime(2024, 2, 1), 0.30)
        trend.add_data_point(datetime(2024, 3, 1), 0.25)
        
        assert trend.trend_direction == "decreasing"
        assert trend.trend_strength < -0.8  # Strong negative correlation
    
    def test_trend_calculation_stable(self):
        """Test trend calculation for stable values."""
        trend = TrendData(
            venue_id="venue-123",
            metric_name="acceptance_rate"
        )
        
        # Add stable data points
        trend.add_data_point(datetime(2024, 1, 1), 0.30)
        trend.add_data_point(datetime(2024, 2, 1), 0.30)
        trend.add_data_point(datetime(2024, 3, 1), 0.30)
        
        assert trend.trend_direction == "stable"
        assert abs(trend.trend_strength) < 0.1  # Very weak correlation


class TestRealismMetrics:
    """Test RealismMetrics class."""
    
    def test_realism_metrics_creation(self):
        """Test creating realism metrics."""
        baseline = {
            "acceptance_rate": 0.25,
            "avg_review_score": 7.0,
            "review_count": 3.0
        }
        current = {
            "acceptance_rate": 0.24,
            "avg_review_score": 7.1,
            "review_count": 3.2
        }
        
        metrics = RealismMetrics(
            venue_id="venue-123",
            peerread_baseline=baseline,
            current_metrics=current
        )
        
        assert metrics.venue_id == "venue-123"
        assert len(metrics.accuracy_scores) == 3
        assert metrics.overall_realism_score > 0.9  # Should be high accuracy
    
    def test_realism_metrics_poor_accuracy(self):
        """Test realism metrics with poor accuracy."""
        baseline = {
            "acceptance_rate": 0.25,
            "avg_review_score": 7.0
        }
        current = {
            "acceptance_rate": 0.50,  # 100% error
            "avg_review_score": 5.0   # ~28% error
        }
        
        metrics = RealismMetrics(
            venue_id="venue-123",
            peerread_baseline=baseline,
            current_metrics=current
        )
        
        assert metrics.accuracy_scores["acceptance_rate"] == 0.0  # 100% error = 0% accuracy
        assert metrics.accuracy_scores["avg_review_score"] < 0.8  # Poor accuracy
        assert metrics.overall_realism_score < 0.5  # Poor overall score
    
    def test_realism_metrics_zero_baseline(self):
        """Test realism metrics with zero baseline values."""
        baseline = {
            "acceptance_rate": 0.0,
            "avg_review_score": 7.0
        }
        current = {
            "acceptance_rate": 0.0,  # Matches zero baseline
            "avg_review_score": 7.0
        }
        
        metrics = RealismMetrics(
            venue_id="venue-123",
            peerread_baseline=baseline,
            current_metrics=current
        )
        
        assert metrics.accuracy_scores["acceptance_rate"] == 1.0  # Perfect match
        assert metrics.accuracy_scores["avg_review_score"] == 1.0
        assert metrics.overall_realism_score == 1.0


class TestVenueStats:
    """Test VenueStats class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def venue_stats(self, temp_dir):
        """Create venue statistics instance for testing."""
        return VenueStats(data_dir=temp_dir)
    
    @pytest.fixture
    def mock_peerread_loader(self):
        """Create mock PeerRead loader."""
        loader = Mock()
        venue_chars = VenueCharacteristics(
            name="ACL",
            venue_type="conference",
            field="NLP",
            total_papers=1000,
            accepted_papers=250,
            acceptance_rate=0.25,
            avg_reviews_per_paper=3.2,
            avg_review_length=650,
            impact_mean=3.2,
            substance_mean=3.4,
            soundness_mean=3.5,
            originality_mean=3.1,
            clarity_mean=3.6,
            meaningful_comparison_mean=3.3
        )
        loader.load_all_venues.return_value = {"ACL": venue_chars}
        return loader
    
    @pytest.fixture
    def venue_stats_with_peerread(self, temp_dir, mock_peerread_loader):
        """Create venue statistics with mocked PeerRead integration."""
        with patch('src.enhancements.venue_statistics.PeerReadLoader') as mock_class:
            mock_class.return_value = mock_peerread_loader
            stats = VenueStats(data_dir=temp_dir, peerread_path="../PeerRead")
            return stats
    
    def test_venue_stats_initialization(self, temp_dir):
        """Test venue statistics initialization."""
        stats = VenueStats(data_dir=temp_dir)
        
        assert stats.data_dir == temp_dir
        assert temp_dir.exists()
        assert len(stats.submission_records) == 0
        assert len(stats.acceptance_records) == 0
        assert stats.realism_threshold == 0.8
        assert stats.acceptance_rate_tolerance == 0.05
    
    def test_venue_stats_with_peerread_integration(self, venue_stats_with_peerread):
        """Test venue statistics with PeerRead integration."""
        stats = venue_stats_with_peerread
        
        assert stats.peerread_loader is not None
        assert "ACL" in stats.peerread_baselines
        assert stats.peerread_baselines["ACL"].acceptance_rate == 0.25
    
    def test_record_submission(self, venue_stats):
        """Test recording a paper submission."""
        submission_date = datetime(2024, 1, 15, 10, 30)
        record = venue_stats.record_submission("paper-123", "venue-456", submission_date)
        
        assert record.paper_id == "paper-123"
        assert record.venue_id == "venue-456"
        assert record.submission_date == submission_date
        assert len(venue_stats.submission_records["venue-456"]) == 1
    
    def test_record_submission_default_date(self, venue_stats):
        """Test recording submission with default date."""
        before_time = datetime.now()
        record = venue_stats.record_submission("paper-123", "venue-456")
        after_time = datetime.now()
        
        assert before_time <= record.submission_date <= after_time
    
    def test_record_decision_success(self, venue_stats):
        """Test recording a review decision."""
        # First record submission
        venue_stats.record_submission("paper-123", "venue-456", datetime(2024, 1, 15))
        
        # Then record decision
        decision_date = datetime(2024, 2, 15)
        review_scores = [7.5, 8.0, 6.5]
        success = venue_stats.record_decision(
            "paper-123", "venue-456", "accept", review_scores, decision_date
        )
        
        assert success is True
        
        # Check that submission record was updated
        record = venue_stats.submission_records["venue-456"][0]
        assert record.decision == "accept"
        assert record.decision_date == decision_date
        assert record.review_scores == review_scores
        assert record.average_score == 7.333333333333333
        assert record.reviewer_count == 3
    
    def test_record_decision_no_submission(self, venue_stats):
        """Test recording decision for non-existent submission."""
        success = venue_stats.record_decision(
            "paper-123", "venue-456", "accept", [7.5, 8.0]
        )
        
        assert success is False
    
    def test_calculate_dynamic_acceptance_rate(self, venue_stats):
        """Test calculating dynamic acceptance rate."""
        venue_id = "venue-123"
        
        # Record submissions and decisions
        base_date = datetime.now() - timedelta(days=30)
        
        # Record 10 submissions, 3 accepted
        for i in range(10):
            submission_date = base_date + timedelta(days=i)
            venue_stats.record_submission(f"paper-{i}", venue_id, submission_date)
            
            decision = "accept" if i < 3 else "reject"
            decision_date = submission_date + timedelta(days=7)
            venue_stats.record_decision(f"paper-{i}", venue_id, decision, [7.0], decision_date)
        
        acceptance_rate = venue_stats.calculate_dynamic_acceptance_rate(venue_id)
        assert acceptance_rate == 0.3  # 3/10 = 0.3
    
    def test_calculate_dynamic_acceptance_rate_no_data(self, venue_stats):
        """Test calculating acceptance rate with no data."""
        acceptance_rate = venue_stats.calculate_dynamic_acceptance_rate("venue-123")
        assert acceptance_rate == 0.0
    
    def test_calculate_dynamic_acceptance_rate_with_baseline(self, venue_stats_with_peerread):
        """Test calculating acceptance rate falling back to baseline."""
        stats = venue_stats_with_peerread
        
        # No submissions recorded, should fall back to PeerRead baseline
        acceptance_rate = stats.calculate_dynamic_acceptance_rate("ACL")
        assert acceptance_rate == 0.25  # PeerRead baseline
    
    def test_get_historical_trends(self, venue_stats):
        """Test getting historical trend data."""
        venue_id = "venue-123"
        
        # Record some submissions to generate trends
        base_date = datetime.now() - timedelta(days=90)
        for i in range(5):
            submission_date = base_date + timedelta(days=i * 20)
            venue_stats.record_submission(f"paper-{i}", venue_id, submission_date)
        
        # Get trend data
        trend_data = venue_stats.get_historical_trends(venue_id, "submission_count")
        
        assert trend_data is not None
        assert trend_data.venue_id == venue_id
        assert trend_data.metric_name == "submission_count"
        assert len(trend_data.time_series) > 0
    
    def test_get_historical_trends_no_data(self, venue_stats):
        """Test getting trends with no data."""
        trend_data = venue_stats.get_historical_trends("venue-123", "acceptance_rate")
        assert trend_data is None
    
    def test_validate_venue_realism(self, venue_stats_with_peerread):
        """Test validating venue realism."""
        stats = venue_stats_with_peerread
        venue_id = "ACL"
        
        # Record some data that matches baseline reasonably well
        base_date = datetime.now() - timedelta(days=30)
        for i in range(10):
            submission_date = base_date + timedelta(days=i)
            stats.record_submission(f"paper-{i}", venue_id, submission_date)
            
            decision = "accept" if i < 2 else "reject"  # 20% acceptance (close to 25% baseline)
            decision_date = submission_date + timedelta(days=7)
            stats.record_decision(f"paper-{i}", venue_id, decision, [3.4, 3.5, 3.3], decision_date)
        
        realism_metrics = stats.validate_venue_realism(venue_id)
        
        assert realism_metrics.venue_id == venue_id
        assert len(realism_metrics.peerread_baseline) > 0
        assert len(realism_metrics.current_metrics) > 0
        assert realism_metrics.overall_realism_score > 0.5  # Should be reasonably realistic
    
    def test_validate_venue_realism_no_baseline(self, venue_stats):
        """Test validating realism without PeerRead baseline."""
        realism_metrics = venue_stats.validate_venue_realism("venue-123")
        
        assert realism_metrics.venue_id == "venue-123"
        assert len(realism_metrics.peerread_baseline) == 0
        assert realism_metrics.overall_realism_score == 0.0
    
    def test_recalibrate_venue(self, venue_stats_with_peerread):
        """Test recalibrating venue parameters."""
        stats = venue_stats_with_peerread
        venue_id = "ACL"
        
        # Create a venue with different acceptance rate
        venue = EnhancedVenue(
            id=venue_id,
            name="ACL",
            venue_type=VenueType.TOP_CONFERENCE,
            field="NLP",
            acceptance_rate=0.40  # Different from baseline 0.25
        )
        
        # Record some data
        base_date = datetime.now() - timedelta(days=30)
        for i in range(10):
            submission_date = base_date + timedelta(days=i)
            stats.record_submission(f"paper-{i}", venue_id, submission_date)
            
            decision = "accept" if i < 4 else "reject"  # 40% acceptance
            decision_date = submission_date + timedelta(days=7)
            stats.record_decision(f"paper-{i}", venue_id, decision, [7.0], decision_date)
        
        # Recalibrate
        success = stats.recalibrate_venue(venue_id, venue)
        
        assert success is True
        # Venue acceptance rate should be adjusted towards baseline
        assert venue.acceptance_rate < 0.40
        assert venue.acceptance_rate > 0.25
    
    def test_recalibrate_venue_no_baseline(self, venue_stats):
        """Test recalibrating venue without baseline."""
        venue = EnhancedVenue(
            id="venue-123",
            name="Test Venue",
            venue_type=VenueType.MID_CONFERENCE,
            field="CS"
        )
        
        success = venue_stats.recalibrate_venue("venue-123", venue)
        assert success is False
    
    def test_get_venue_statistics(self, venue_stats):
        """Test getting comprehensive venue statistics."""
        venue_id = "venue-123"
        
        # Record some data
        base_date = datetime.now() - timedelta(days=30)
        for i in range(5):
            submission_date = base_date + timedelta(days=i * 5)
            venue_stats.record_submission(f"paper-{i}", venue_id, submission_date)
            
            decision = "accept" if i < 2 else "reject"
            decision_date = submission_date + timedelta(days=7)
            scores = [7.0 + i * 0.5, 6.5 + i * 0.5, 7.5 + i * 0.5]
            venue_stats.record_decision(f"paper-{i}", venue_id, decision, scores, decision_date)
        
        stats = venue_stats.get_venue_statistics(venue_id)
        
        assert stats["venue_id"] == venue_id
        assert stats["total_submissions"] == 5
        assert stats["submissions_with_decisions"] == 5
        assert stats["acceptance_rate"] == 0.4  # 2/5
        assert stats["average_review_score"] > 0
        assert stats["average_review_count"] == 3.0
        assert "trends" in stats
        assert "last_updated" in stats
    
    def test_get_venue_statistics_no_data(self, venue_stats):
        """Test getting statistics for venue with no data."""
        stats = venue_stats.get_venue_statistics("venue-123")
        
        assert stats["venue_id"] == "venue-123"
        assert stats["total_submissions"] == 0
        assert stats["submissions_with_decisions"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["average_review_score"] == 0.0
    
    def test_save_and_load_data(self, temp_dir):
        """Test saving and loading statistics data."""
        # Create first instance and add data
        stats1 = VenueStats(data_dir=temp_dir)
        stats1.record_submission("paper-123", "venue-456", datetime(2024, 1, 15))
        stats1.record_decision("paper-123", "venue-456", "accept", [7.5, 8.0])
        stats1.save_data()
        
        # Create second instance and check data is loaded
        stats2 = VenueStats(data_dir=temp_dir)
        
        assert len(stats2.submission_records["venue-456"]) == 1
        record = stats2.submission_records["venue-456"][0]
        assert record.paper_id == "paper-123"
        assert record.decision == "accept"
        assert record.review_scores == [7.5, 8.0]
    
    def test_cleanup_old_data(self, venue_stats):
        """Test cleaning up old statistics data."""
        venue_id = "venue-123"
        
        # Record old and new data
        old_date = datetime.now() - timedelta(days=400)  # Older than max_history_days
        new_date = datetime.now() - timedelta(days=30)
        
        venue_stats.record_submission("old-paper", venue_id, old_date)
        venue_stats.record_submission("new-paper", venue_id, new_date)
        
        assert len(venue_stats.submission_records[venue_id]) == 2
        
        # Clean up old data
        venue_stats.cleanup_old_data(max_age_days=365)
        
        # Only new data should remain
        assert len(venue_stats.submission_records[venue_id]) == 1
        assert venue_stats.submission_records[venue_id][0].paper_id == "new-paper"
    
    def test_update_submission_trends(self, venue_stats):
        """Test updating submission trends."""
        venue_id = "venue-123"
        
        # Record submissions in different months
        jan_date = datetime(2024, 1, 15)
        feb_date = datetime(2024, 2, 15)
        
        venue_stats.record_submission("paper-1", venue_id, jan_date)
        venue_stats.record_submission("paper-2", venue_id, jan_date)
        venue_stats.record_submission("paper-3", venue_id, feb_date)
        
        # Check that trend data was created
        assert "submission_count" in venue_stats.trend_data[venue_id]
        trend_data = venue_stats.trend_data[venue_id]["submission_count"]
        assert len(trend_data.time_series) >= 1
    
    def test_update_acceptance_trends(self, venue_stats):
        """Test updating acceptance trends."""
        venue_id = "venue-123"
        
        # Record submissions and decisions
        submission_date = datetime(2024, 1, 15)
        decision_date = datetime(2024, 1, 25)
        
        venue_stats.record_submission("paper-1", venue_id, submission_date)
        venue_stats.record_submission("paper-2", venue_id, submission_date)
        
        venue_stats.record_decision("paper-1", venue_id, "accept", [7.0], decision_date)
        venue_stats.record_decision("paper-2", venue_id, "reject", [5.0], decision_date)
        
        # Check that trend data was created
        assert "acceptance_rate" in venue_stats.trend_data[venue_id]
        trend_data = venue_stats.trend_data[venue_id]["acceptance_rate"]
        assert len(trend_data.time_series) >= 1


class TestVenueStatsIntegration:
    """Integration tests for venue statistics system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_full_venue_statistics_workflow(self, temp_dir):
        """Test complete venue statistics workflow."""
        # Initialize with mock PeerRead data
        mock_loader = Mock()
        venue_chars = VenueCharacteristics(
            name="TestConf",
            venue_type="conference",
            field="CS",
            total_papers=500,
            accepted_papers=125,
            acceptance_rate=0.25,
            avg_reviews_per_paper=3.0,
            avg_review_length=600,
            impact_mean=3.2,
            substance_mean=3.4
        )
        mock_loader.load_all_venues.return_value = {"TestConf": venue_chars}
        
        with patch('src.enhancements.venue_statistics.PeerReadLoader') as mock_class:
            mock_class.return_value = mock_loader
            stats = VenueStats(data_dir=temp_dir, peerread_path="../PeerRead")
        
        venue_id = "TestConf"
        
        # Simulate a conference cycle
        base_date = datetime.now() - timedelta(days=60)
        
        # Record submissions over time
        for i in range(20):
            submission_date = base_date + timedelta(days=i * 2)
            stats.record_submission(f"paper-{i}", venue_id, submission_date)
        
        # Record decisions with realistic acceptance rate
        for i in range(20):
            decision_date = base_date + timedelta(days=i * 2 + 30)  # 30 days later
            decision = "accept" if i < 5 else "reject"  # 25% acceptance rate
            scores = [3.0 + (i % 3) * 0.5, 3.2 + (i % 3) * 0.3, 3.4 + (i % 3) * 0.4]
            stats.record_decision(f"paper-{i}", venue_id, decision, scores, decision_date)
        
        # Test dynamic acceptance rate calculation
        acceptance_rate = stats.calculate_dynamic_acceptance_rate(venue_id)
        assert 0.20 <= acceptance_rate <= 0.30  # Should be close to 25%
        
        # Test historical trends
        submission_trend = stats.get_historical_trends(venue_id, "submission_count")
        acceptance_trend = stats.get_historical_trends(venue_id, "acceptance_rate")
        
        assert submission_trend is not None
        assert acceptance_trend is not None
        assert len(submission_trend.time_series) > 0
        assert len(acceptance_trend.time_series) > 0
        
        # Test realism validation
        realism_metrics = stats.validate_venue_realism(venue_id)
        assert realism_metrics.overall_realism_score > 0.7  # Should be realistic
        
        # Test venue recalibration
        venue = EnhancedVenue(
            id=venue_id,
            name="TestConf",
            venue_type=VenueType.TOP_CONFERENCE,
            field="CS",
            acceptance_rate=0.35  # More significantly off from baseline (0.25)
        )
        
        recalibration_success = stats.recalibrate_venue(venue_id, venue)
        assert recalibration_success is True
        assert venue.acceptance_rate < 0.35  # Should be adjusted towards baseline
        assert venue.acceptance_rate > 0.25  # But not all the way to baseline
        
        # Test comprehensive statistics
        venue_statistics = stats.get_venue_statistics(venue_id)
        assert venue_statistics["total_submissions"] == 20
        assert venue_statistics["submissions_with_decisions"] == 20
        assert venue_statistics["acceptance_rate"] == 0.25
        assert venue_statistics["average_review_count"] == 3.0
        assert venue_statistics["realism_score"] > 0.7
        
        # Test data persistence
        stats.save_data()
        
        # Create new instance and verify data is loaded
        with patch('src.enhancements.venue_statistics.PeerReadLoader') as mock_class2:
            mock_class2.return_value = mock_loader
            stats2 = VenueStats(data_dir=temp_dir, peerread_path="../PeerRead")
        
        assert len(stats2.submission_records[venue_id]) == 20
        
        # Test data cleanup
        stats2.cleanup_old_data(max_age_days=30)  # Clean data older than 30 days
        remaining_records = len(stats2.submission_records[venue_id])
        assert remaining_records < 20  # Some records should be cleaned up


if __name__ == "__main__":
    pytest.main([__file__])