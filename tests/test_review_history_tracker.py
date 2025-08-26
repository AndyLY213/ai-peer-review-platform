"""
Unit Tests for Review History Tracking System

This module contains comprehensive unit tests for the review history tracking system,
including ReviewQualityMetric tracking, reliability scoring, and historical data management.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import json

from src.enhancements.review_history_tracker import (
    ReviewHistoryTracker, ReviewPerformanceMetrics, ReliabilityCategory
)
from src.data.enhanced_models import (
    StructuredReview, ReviewQualityMetric, EnhancedReviewCriteria,
    ReviewDecision, DetailedStrength, DetailedWeakness
)


class TestReviewHistoryTracker(unittest.TestCase):
    """Test cases for ReviewHistoryTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = ReviewHistoryTracker(data_dir=self.temp_dir)
        
        # Create sample review data
        self.sample_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            criteria_scores=EnhancedReviewCriteria(
                novelty=7.0,
                technical_quality=8.0,
                clarity=6.0,
                significance=7.5,
                reproducibility=6.5,
                related_work=7.0
            ),
            confidence_level=4,
            recommendation=ReviewDecision.MINOR_REVISION,
            executive_summary="This paper presents interesting work on AI peer review.",
            detailed_strengths=[
                DetailedStrength(category="Technical", description="Strong methodology", importance=4),
                DetailedStrength(category="Novelty", description="Novel approach", importance=3)
            ],
            detailed_weaknesses=[
                DetailedWeakness(category="Presentation", description="Some clarity issues", severity=2)
            ],
            technical_comments="The technical approach is sound but could be improved.",
            presentation_comments="Generally well written with minor issues.",
            questions_for_authors=["How does this compare to method X?"],
            suggestions_for_improvement=["Consider adding more experiments"],
            review_length=450,
            time_spent_minutes=120,
            submission_timestamp=datetime.now()
        )
        
        # Sample deadline
        self.sample_deadline = datetime.now() + timedelta(days=1)  # Review submitted 1 day early
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ReviewHistoryTracker initialization."""
        self.assertIsInstance(self.tracker, ReviewHistoryTracker)
        self.assertTrue(self.temp_dir.exists())
        self.assertEqual(len(self.tracker.performance_metrics), 0)
    
    def test_track_review_quality_basic(self):
        """Test basic review quality tracking."""
        quality_metric = self.tracker.track_review_quality(self.sample_review, self.sample_deadline)
        
        # Check quality metric creation
        self.assertIsInstance(quality_metric, ReviewQualityMetric)
        self.assertEqual(quality_metric.review_id, self.sample_review.review_id)
        self.assertGreater(quality_metric.quality_score, 0.0)
        self.assertGreater(quality_metric.timeliness_score, 0.0)
        
        # Check performance metrics update
        self.assertIn("reviewer_001", self.tracker.performance_metrics)
        metrics = self.tracker.performance_metrics["reviewer_001"]
        self.assertEqual(metrics.total_reviews, 1)
        self.assertEqual(metrics.on_time_reviews, 1)
        self.assertEqual(metrics.late_reviews, 0)
    
    def test_timeliness_score_calculation(self):
        """Test timeliness score calculation for different scenarios."""
        # Test early submission (2 days early)
        early_deadline = datetime.now() + timedelta(days=2)
        early_review = StructuredReview(
            reviewer_id="reviewer_002",
            paper_id="paper_002",
            venue_id="venue_001",
            submission_timestamp=datetime.now()
        )
        
        quality_metric = self.tracker.track_review_quality(early_review, early_deadline)
        self.assertEqual(quality_metric.timeliness_score, 1.0)  # Excellent
        
        # Test late submission (5 days late)
        late_deadline = datetime.now() - timedelta(days=5)
        late_review = StructuredReview(
            reviewer_id="reviewer_003",
            paper_id="paper_003",
            venue_id="venue_001",
            submission_timestamp=datetime.now()
        )
        
        quality_metric = self.tracker.track_review_quality(late_review, late_deadline)
        self.assertLess(quality_metric.timeliness_score, 0.5)  # Poor
    
    def test_performance_metrics_calculation(self):
        """Test comprehensive performance metrics calculation."""
        reviewer_id = "reviewer_004"
        
        # Submit multiple reviews with varying quality and timeliness
        reviews_data = [
            (8.0, datetime.now() - timedelta(days=1)),  # High quality, 1 day late
            (6.0, datetime.now() + timedelta(days=1)),  # Medium quality, 1 day early
            (4.0, datetime.now() - timedelta(days=3)),  # Low quality, 3 days late
            (9.0, datetime.now()),                      # High quality, on time
            (7.0, datetime.now() + timedelta(days=2))   # High quality, 2 days early
        ]
        
        for i, (quality, deadline) in enumerate(reviews_data):
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id=f"paper_{i}",
                venue_id="venue_001",
                submission_timestamp=datetime.now()
            )
            review.quality_score = quality / 10.0  # Normalize to 0-1
            
            self.tracker.track_review_quality(review, deadline)
        
        # Check calculated metrics
        metrics = self.tracker.performance_metrics[reviewer_id]
        self.assertEqual(metrics.total_reviews, 5)
        self.assertEqual(metrics.high_quality_reviews, 2)  # Scores >= 0.8
        self.assertEqual(metrics.medium_quality_reviews, 2)  # Scores 0.5-0.8
        self.assertEqual(metrics.low_quality_reviews, 1)   # Scores < 0.5
        self.assertEqual(metrics.on_time_reviews, 3)       # On time or early
        self.assertEqual(metrics.late_reviews, 2)          # Late submissions
    
    def test_reliability_category_determination(self):
        """Test reliability category determination."""
        # Test excellent reliability
        excellent_metrics = ReviewPerformanceMetrics(reviewer_id="test")
        excellent_metrics.reliability_score = 0.95
        category = self.tracker._determine_reliability_category(excellent_metrics.reliability_score)
        self.assertEqual(category, ReliabilityCategory.EXCELLENT)
        
        # Test poor reliability
        poor_metrics = ReviewPerformanceMetrics(reviewer_id="test")
        poor_metrics.reliability_score = 0.45
        category = self.tracker._determine_reliability_category(poor_metrics.reliability_score)
        self.assertEqual(category, ReliabilityCategory.POOR)
        
        # Test unreliable
        unreliable_metrics = ReviewPerformanceMetrics(reviewer_id="test")
        unreliable_metrics.reliability_score = 0.25
        category = self.tracker._determine_reliability_category(unreliable_metrics.reliability_score)
        self.assertEqual(category, ReliabilityCategory.UNRELIABLE)
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        # Test improving trend
        improving_quality = [0.5, 0.6, 0.7, 0.8, 0.9]
        improving_timeliness = [0.6, 0.7, 0.8, 0.9, 1.0]
        trend = self.tracker._analyze_performance_trend(improving_quality, improving_timeliness)
        self.assertEqual(trend, "improving")
        
        # Test declining trend
        declining_quality = [0.9, 0.8, 0.7, 0.6, 0.5]
        declining_timeliness = [1.0, 0.9, 0.8, 0.7, 0.6]
        trend = self.tracker._analyze_performance_trend(declining_quality, declining_timeliness)
        self.assertEqual(trend, "declining")
        
        # Test stable trend
        stable_quality = [0.7, 0.7, 0.7, 0.7, 0.7]
        stable_timeliness = [0.8, 0.8, 0.8, 0.8, 0.8]
        trend = self.tracker._analyze_performance_trend(stable_quality, stable_timeliness)
        self.assertEqual(trend, "stable")
    
    def test_get_reviewer_performance(self):
        """Test getting reviewer performance metrics."""
        # Track a review first
        self.tracker.track_review_quality(self.sample_review, self.sample_deadline)
        
        # Get performance metrics
        metrics = self.tracker.get_reviewer_performance("reviewer_001")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.reviewer_id, "reviewer_001")
        self.assertEqual(metrics.total_reviews, 1)
        
        # Test non-existent reviewer
        metrics = self.tracker.get_reviewer_performance("non_existent")
        self.assertIsNone(metrics)
    
    def test_get_reliability_score(self):
        """Test getting reliability score for a reviewer."""
        # Track a review first
        self.tracker.track_review_quality(self.sample_review, self.sample_deadline)
        
        # Get reliability score
        score = self.tracker.get_reliability_score("reviewer_001")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test non-existent reviewer
        score = self.tracker.get_reliability_score("non_existent")
        self.assertEqual(score, 0.0)
    
    def test_get_top_reviewers(self):
        """Test getting top reviewers by reliability score."""
        # Create multiple reviewers with different performance
        reviewers_data = [
            ("reviewer_A", 0.95, 6),  # Excellent, 6 reviews
            ("reviewer_B", 0.85, 5),  # Good, 5 reviews
            ("reviewer_C", 0.75, 8),  # Average, 8 reviews
            ("reviewer_D", 0.65, 3),  # Below threshold, 3 reviews
            ("reviewer_E", 0.90, 2)   # Excellent but too few reviews
        ]
        
        for reviewer_id, reliability, review_count in reviewers_data:
            metrics = ReviewPerformanceMetrics(reviewer_id=reviewer_id)
            metrics.reliability_score = reliability
            metrics.total_reviews = review_count
            self.tracker.performance_metrics[reviewer_id] = metrics
        
        # Get top reviewers (minimum 5 reviews)
        top_reviewers = self.tracker.get_top_reviewers(limit=3, min_reviews=5)
        
        self.assertEqual(len(top_reviewers), 3)
        self.assertEqual(top_reviewers[0][0], "reviewer_A")  # Highest score
        self.assertEqual(top_reviewers[1][0], "reviewer_B")  # Second highest
        self.assertEqual(top_reviewers[2][0], "reviewer_C")  # Third highest
    
    def test_get_reviewers_needing_improvement(self):
        """Test getting reviewers needing improvement."""
        # Create reviewers with different reliability scores
        reviewers_data = [
            ("reviewer_good", 0.85, 5),
            ("reviewer_poor", 0.45, 4),
            ("reviewer_unreliable", 0.25, 6),
            ("reviewer_new", 0.30, 2)  # Too few reviews to be included
        ]
        
        for reviewer_id, reliability, review_count in reviewers_data:
            metrics = ReviewPerformanceMetrics(reviewer_id=reviewer_id)
            metrics.reliability_score = reliability
            metrics.total_reviews = review_count
            self.tracker.performance_metrics[reviewer_id] = metrics
        
        # Get reviewers needing improvement (threshold 0.6)
        needing_improvement = self.tracker.get_reviewers_needing_improvement(threshold=0.6)
        
        self.assertEqual(len(needing_improvement), 2)
        reviewer_ids = [reviewer_id for reviewer_id, _ in needing_improvement]
        self.assertIn("reviewer_poor", reviewer_ids)
        self.assertIn("reviewer_unreliable", reviewer_ids)
        self.assertNotIn("reviewer_new", reviewer_ids)  # Too few reviews
    
    def test_generate_performance_report(self):
        """Test generating comprehensive performance report."""
        # Track multiple reviews for a reviewer
        reviewer_id = "reviewer_report_test"
        
        for i in range(5):
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id=f"paper_{i}",
                venue_id="venue_001",
                submission_timestamp=datetime.now()
            )
            review.quality_score = 0.7 + (i * 0.05)  # Improving quality
            
            deadline = datetime.now() + timedelta(days=1)
            self.tracker.track_review_quality(review, deadline)
        
        # Generate report
        report = self.tracker.generate_performance_report(reviewer_id)
        
        # Check report structure
        self.assertIn('reviewer_id', report)
        self.assertIn('summary', report)
        self.assertIn('quality_metrics', report)
        self.assertIn('timeliness_metrics', report)
        self.assertIn('percentiles', report)
        self.assertIn('recommendations', report)
        
        # Check summary data
        self.assertEqual(report['summary']['total_reviews'], 5)
        self.assertGreater(report['summary']['reliability_score'], 0.0)
        
        # Test non-existent reviewer
        report = self.tracker.generate_performance_report("non_existent")
        self.assertIn('error', report)
    
    def test_update_review_helpfulness(self):
        """Test updating review helpfulness rating."""
        # Track a review first
        quality_metric = self.tracker.track_review_quality(self.sample_review, self.sample_deadline)
        
        # Create history file manually for testing
        reviewer_id = self.sample_review.reviewer_id
        history_file = self.temp_dir / f"{reviewer_id}_history.json"
        history_data = {
            'reviews': [{
                'review_id': quality_metric.review_id,
                'quality_score': quality_metric.quality_score,
                'timeliness_score': quality_metric.timeliness_score,
                'timestamp': quality_metric.timestamp.isoformat()
            }]
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f)
        
        # Update helpfulness rating
        self.tracker.update_review_helpfulness(quality_metric.review_id, 0.85)
        
        # Verify update
        with open(history_file, 'r') as f:
            updated_data = json.load(f)
        
        self.assertEqual(updated_data['reviews'][0]['helpfulness_rating'], 0.85)
    
    def test_data_persistence(self):
        """Test saving and loading performance data."""
        # Track some reviews
        self.tracker.track_review_quality(self.sample_review, self.sample_deadline)
        
        # Create new tracker instance with same data directory
        new_tracker = ReviewHistoryTracker(data_dir=self.temp_dir)
        
        # Check that data was loaded
        self.assertIn("reviewer_001", new_tracker.performance_metrics)
        metrics = new_tracker.performance_metrics["reviewer_001"]
        self.assertEqual(metrics.total_reviews, 1)
    
    def test_system_statistics(self):
        """Test getting system-wide statistics."""
        # Create multiple reviewers with different performance
        reviewers_data = [
            ("reviewer_1", 0.95, 6, ReliabilityCategory.EXCELLENT),
            ("reviewer_2", 0.85, 5, ReliabilityCategory.GOOD),
            ("reviewer_3", 0.65, 4, ReliabilityCategory.AVERAGE),
            ("reviewer_4", 0.45, 3, ReliabilityCategory.POOR)
        ]
        
        for reviewer_id, reliability, review_count, category in reviewers_data:
            metrics = ReviewPerformanceMetrics(reviewer_id=reviewer_id)
            metrics.reliability_score = reliability
            metrics.total_reviews = review_count
            metrics.reliability_category = category
            self.tracker.performance_metrics[reviewer_id] = metrics
        
        # Get system statistics
        stats = self.tracker.get_system_statistics()
        
        self.assertEqual(stats['total_reviewers'], 4)
        self.assertEqual(stats['total_reviews'], 18)  # 6+5+4+3
        self.assertGreater(stats['avg_reliability_score'], 0.0)
        self.assertEqual(stats['reviewers_with_good_reliability'], 2)  # >= 0.8
        self.assertEqual(stats['reviewers_needing_improvement'], 1)    # < 0.6
        
        # Check reliability distribution
        self.assertEqual(stats['reliability_distribution']['Excellent'], 1)
        self.assertEqual(stats['reliability_distribution']['Good'], 1)
        self.assertEqual(stats['reliability_distribution']['Average'], 1)
        self.assertEqual(stats['reliability_distribution']['Poor'], 1)
    
    def test_improvement_recommendations(self):
        """Test generation of improvement recommendations."""
        # Create metrics with various issues
        poor_quality_metrics = ReviewPerformanceMetrics(reviewer_id="poor_quality")
        poor_quality_metrics.avg_quality_score = 0.4
        poor_quality_metrics.avg_timeliness_score = 0.8
        poor_quality_metrics.quality_consistency = 0.4
        poor_quality_metrics.reliability_category = ReliabilityCategory.POOR
        
        recommendations = self.tracker._generate_improvement_recommendations(poor_quality_metrics)
        
        # Check that appropriate recommendations are generated
        self.assertTrue(any("detailed and constructive feedback" in rec for rec in recommendations))
        self.assertTrue(any("consistent review quality" in rec for rec in recommendations))
        self.assertTrue(any("mentorship or training" in rec for rec in recommendations))
        
        # Test for timeliness issues
        late_metrics = ReviewPerformanceMetrics(reviewer_id="late_reviewer")
        late_metrics.avg_quality_score = 0.8
        late_metrics.avg_timeliness_score = 0.5
        late_metrics.on_time_reviews = 2
        late_metrics.late_reviews = 8
        
        recommendations = self.tracker._generate_improvement_recommendations(late_metrics)
        self.assertTrue(any("timeliness" in rec for rec in recommendations))
        self.assertTrue(any("deadline" in rec for rec in recommendations))


class TestReviewPerformanceMetrics(unittest.TestCase):
    """Test cases for ReviewPerformanceMetrics class."""
    
    def test_initialization(self):
        """Test ReviewPerformanceMetrics initialization."""
        metrics = ReviewPerformanceMetrics(reviewer_id="test_reviewer")
        
        self.assertEqual(metrics.reviewer_id, "test_reviewer")
        self.assertEqual(metrics.total_reviews, 0)
        self.assertEqual(metrics.avg_quality_score, 0.0)
        self.assertEqual(metrics.reliability_category, ReliabilityCategory.AVERAGE)
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        original_metrics = ReviewPerformanceMetrics(
            reviewer_id="test_reviewer",
            total_reviews=10,
            avg_quality_score=0.85,
            reliability_category=ReliabilityCategory.GOOD,
            recent_performance_trend="improving"
        )
        
        # Convert to dict and back
        metrics_dict = original_metrics.to_dict()
        restored_metrics = ReviewPerformanceMetrics.from_dict(metrics_dict)
        
        # Check that all fields are preserved
        self.assertEqual(restored_metrics.reviewer_id, original_metrics.reviewer_id)
        self.assertEqual(restored_metrics.total_reviews, original_metrics.total_reviews)
        self.assertEqual(restored_metrics.avg_quality_score, original_metrics.avg_quality_score)
        self.assertEqual(restored_metrics.reliability_category, original_metrics.reliability_category)
        self.assertEqual(restored_metrics.recent_performance_trend, original_metrics.recent_performance_trend)


if __name__ == '__main__':
    unittest.main()