"""
Unit tests for ReviewTradingDetector

Tests review trading detection functionality including pattern detection,
statistical analysis, and suspicious behavior identification.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.enhancements.review_trading_detector import (
    ReviewTradingDetector, ReviewExchange, TradingPattern,
    ResearcherTradingProfile
)
from src.data.enhanced_models import ReviewDecision
from src.core.exceptions import ValidationError


class TestReviewExchange:
    """Test ReviewExchange functionality."""
    
    def test_exchange_creation(self):
        """Test creating a review exchange."""
        date_a = datetime.now()
        date_b = date_a + timedelta(days=5)
        
        exchange = ReviewExchange(
            reviewer_a="researcher_001",
            reviewer_b="researcher_002",
            paper_a="paper_001",
            paper_b="paper_002",
            review_a_date=date_a,
            review_b_date=date_b,
            score_a=7.5,
            score_b=8.0,
            decision_a=ReviewDecision.ACCEPT,
            decision_b=ReviewDecision.MINOR_REVISION
        )
        
        assert exchange.reviewer_a == "researcher_001"
        assert exchange.reviewer_b == "researcher_002"
        assert exchange.paper_a == "paper_001"
        assert exchange.paper_b == "paper_002"
        assert exchange.score_a == 7.5
        assert exchange.score_b == 8.0
        assert exchange.decision_a == ReviewDecision.ACCEPT
        assert exchange.decision_b == ReviewDecision.MINOR_REVISION
        assert exchange.time_gap_days == 5
    
    def test_time_gap_calculation(self):
        """Test time gap calculation between reviews."""
        date_a = datetime(2024, 1, 1)
        date_b = datetime(2024, 1, 10)
        
        exchange = ReviewExchange(
            reviewer_a="researcher_001",
            reviewer_b="researcher_002",
            paper_a="paper_001",
            paper_b="paper_002",
            review_a_date=date_a,
            review_b_date=date_b
        )
        
        assert exchange.time_gap_days == 9
        
        # Test reverse order
        exchange_reverse = ReviewExchange(
            reviewer_a="researcher_001",
            reviewer_b="researcher_002",
            paper_a="paper_001",
            paper_b="paper_002",
            review_a_date=date_b,
            review_b_date=date_a
        )
        
        assert exchange_reverse.time_gap_days == 9  # Should be absolute value


class TestTradingPattern:
    """Test TradingPattern functionality."""
    
    def test_empty_pattern(self):
        """Test pattern with no exchanges."""
        pattern = TradingPattern(
            researcher_a="researcher_001",
            researcher_b="researcher_002"
        )
        
        assert pattern.total_exchanges == 0
        assert pattern.mutual_favorable_count == 0
        assert pattern.average_time_gap == 0.0
        assert pattern.trading_score == 0.0
        assert pattern.confidence_level == 0.0
    
    def test_single_exchange_pattern(self):
        """Test pattern with single exchange."""
        exchange = ReviewExchange(
            reviewer_a="researcher_001",
            reviewer_b="researcher_002",
            paper_a="paper_001",
            paper_b="paper_002",
            review_a_date=datetime.now(),
            review_b_date=datetime.now() + timedelta(days=3),
            score_a=8.0,
            score_b=7.5,
            decision_a=ReviewDecision.ACCEPT,
            decision_b=ReviewDecision.ACCEPT
        )
        
        pattern = TradingPattern(
            researcher_a="researcher_001",
            researcher_b="researcher_002",
            exchanges=[exchange]
        )
        
        assert pattern.total_exchanges == 1
        assert pattern.mutual_favorable_count == 1  # Both accepted
        assert pattern.average_time_gap == 3.0
        assert pattern.trading_score > 0.0
    
    def test_multiple_exchanges_pattern(self):
        """Test pattern with multiple exchanges."""
        base_date = datetime.now()
        exchanges = []
        
        # Create 3 exchanges with varying characteristics
        for i in range(3):
            exchange = ReviewExchange(
                reviewer_a="researcher_001",
                reviewer_b="researcher_002",
                paper_a=f"paper_a_{i}",
                paper_b=f"paper_b_{i}",
                review_a_date=base_date + timedelta(days=i*30),
                review_b_date=base_date + timedelta(days=i*30 + 5),
                score_a=7.0 + i * 0.5,
                score_b=7.5 + i * 0.3,
                decision_a=ReviewDecision.ACCEPT if i < 2 else ReviewDecision.REJECT,
                decision_b=ReviewDecision.ACCEPT if i < 2 else ReviewDecision.MAJOR_REVISION
            )
            exchanges.append(exchange)
        
        pattern = TradingPattern(
            researcher_a="researcher_001",
            researcher_b="researcher_002",
            exchanges=exchanges
        )
        
        assert pattern.total_exchanges == 3
        assert pattern.mutual_favorable_count == 2  # First two are mutual accepts
        assert pattern.average_time_gap == 5.0  # All have 5-day gaps
        assert pattern.trading_score > 0.0
        assert pattern.confidence_level > 0.0
    
    def test_high_trading_score_scenario(self):
        """Test scenario that should produce high trading score."""
        base_date = datetime.now()
        exchanges = []
        
        # Create multiple exchanges with favorable reviews and close timing
        for i in range(5):
            exchange = ReviewExchange(
                reviewer_a="researcher_001",
                reviewer_b="researcher_002",
                paper_a=f"paper_a_{i}",
                paper_b=f"paper_b_{i}",
                review_a_date=base_date + timedelta(days=i*20),
                review_b_date=base_date + timedelta(days=i*20 + 2),  # Very close timing
                score_a=8.5,  # High scores
                score_b=8.7,
                decision_a=ReviewDecision.ACCEPT,
                decision_b=ReviewDecision.ACCEPT
            )
            exchanges.append(exchange)
        
        pattern = TradingPattern(
            researcher_a="researcher_001",
            researcher_b="researcher_002",
            exchanges=exchanges
        )
        
        assert pattern.total_exchanges == 5
        assert pattern.mutual_favorable_count == 5  # All mutual accepts
        assert pattern.average_time_gap == 2.0  # Very close timing
        assert pattern.trading_score > 0.7  # Should be high
        assert pattern.confidence_level > 0.5


class TestResearcherTradingProfile:
    """Test ResearcherTradingProfile functionality."""
    
    def test_empty_profile(self):
        """Test profile with no trading patterns."""
        profile = ResearcherTradingProfile(researcher_id="researcher_001")
        profile.update_profile([])
        
        assert profile.total_trading_partners == 0
        assert profile.total_exchanges == 0
        assert profile.average_trading_score == 0.0
        assert profile.trading_tendency == 0.0
        assert len(profile.most_frequent_partners) == 0
        assert len(profile.suspicious_patterns) == 0
    
    def test_profile_with_patterns(self):
        """Test profile update with trading patterns."""
        # Create mock trading patterns
        pattern1 = TradingPattern(
            researcher_a="researcher_001",
            researcher_b="researcher_002",
            exchanges=[ReviewExchange(
                reviewer_a="researcher_001",
                reviewer_b="researcher_002",
                paper_a="paper_001",
                paper_b="paper_002",
                score_a=8.0,
                score_b=8.5,
                decision_a=ReviewDecision.ACCEPT,
                decision_b=ReviewDecision.ACCEPT
            )]
        )
        
        pattern2 = TradingPattern(
            researcher_a="researcher_001",
            researcher_b="researcher_003",
            exchanges=[ReviewExchange(
                reviewer_a="researcher_001",
                reviewer_b="researcher_003",
                paper_a="paper_003",
                paper_b="paper_004",
                score_a=7.0,
                score_b=7.5,
                decision_a=ReviewDecision.MINOR_REVISION,
                decision_b=ReviewDecision.ACCEPT
            )]
        )
        
        profile = ResearcherTradingProfile(researcher_id="researcher_001")
        profile.update_profile([pattern1, pattern2])
        
        assert profile.total_trading_partners == 2
        assert profile.total_exchanges == 2
        assert profile.average_trading_score > 0.0
        assert len(profile.most_frequent_partners) == 2


class TestReviewTradingDetector:
    """Test ReviewTradingDetector functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = ReviewTradingDetector(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector.data_dir.exists()
        assert len(self.detector.review_records) == 0
        assert len(self.detector.researcher_reviews) == 0
        assert len(self.detector.paper_reviews) == 0
        assert len(self.detector.trading_patterns) == 0
        assert len(self.detector.researcher_profiles) == 0
    
    def test_record_single_review(self):
        """Test recording a single review."""
        review_date = datetime.now()
        
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_001",
            author_id="researcher_002",
            review_date=review_date,
            score=7.5,
            decision=ReviewDecision.ACCEPT
        )
        
        assert len(self.detector.review_records) == 1
        assert "review_001" in self.detector.review_records
        assert len(self.detector.researcher_reviews["researcher_001"]) == 1
        assert len(self.detector.paper_reviews["paper_001"]) == 1
    
    def test_detect_reciprocal_reviews(self):
        """Test detection of reciprocal review patterns."""
        base_date = datetime.now()
        
        # Researcher A reviews B's paper
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        # Researcher B reviews A's paper (reciprocal)
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=5),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # Should detect trading pattern
        patterns = self.detector.detect_trading_patterns(min_score=0.0)
        assert len(patterns) >= 1
        
        pattern = patterns[0]
        assert pattern.total_exchanges == 1
        assert pattern.mutual_favorable_count == 1
    
    def test_no_trading_detection_for_distant_reviews(self):
        """Test that distant reviews don't trigger trading detection."""
        base_date = datetime.now()
        
        # Researcher A reviews B's paper
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        # Researcher B reviews A's paper much later (beyond threshold)
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=100),  # Beyond max_time_gap_days
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # Should not detect trading pattern
        patterns = self.detector.detect_trading_patterns(min_score=0.0)
        assert len(patterns) == 0
    
    def test_multiple_exchanges_pattern(self):
        """Test detection of multiple exchanges between same researchers."""
        base_date = datetime.now()
        
        # First exchange
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=3),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # Second exchange
        self.detector.record_review(
            review_id="review_003",
            reviewer_id="researcher_001",
            paper_id="paper_004",
            author_id="researcher_002",
            review_date=base_date + timedelta(days=30),
            score=7.5,
            decision=ReviewDecision.MINOR_REVISION
        )
        
        self.detector.record_review(
            review_id="review_004",
            reviewer_id="researcher_002",
            paper_id="paper_003",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=32),
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        patterns = self.detector.detect_trading_patterns(min_score=0.0)
        assert len(patterns) >= 1
        
        pattern = patterns[0]
        assert pattern.total_exchanges == 2
    
    def test_researcher_trading_behavior(self):
        """Test getting researcher trading behavior."""
        # Create some trading patterns first
        base_date = datetime.now()
        
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=3),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # Get behavior profile
        profile = self.detector.get_researcher_trading_behavior("researcher_001")
        assert profile is not None
        assert profile.researcher_id == "researcher_001"
        assert profile.total_trading_partners >= 1
        assert profile.total_exchanges >= 1
    
    def test_network_cluster_analysis(self):
        """Test network cluster analysis."""
        base_date = datetime.now()
        
        # Create a small trading network: A-B, B-C, forming a cluster
        # A-B exchange
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=3),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # B-C exchange
        self.detector.record_review(
            review_id="review_003",
            reviewer_id="researcher_002",
            paper_id="paper_003",
            author_id="researcher_003",
            review_date=base_date + timedelta(days=10),
            score=8.2,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_004",
            reviewer_id="researcher_003",
            paper_id="paper_004",
            author_id="researcher_002",
            review_date=base_date + timedelta(days=12),
            score=8.3,
            decision=ReviewDecision.ACCEPT
        )
        
        clusters = self.detector.analyze_network_clusters()
        # Should find at least one cluster if trading scores are high enough
        assert isinstance(clusters, dict)
    
    def test_trading_statistics(self):
        """Test calculation of trading statistics."""
        # Initially empty
        stats = self.detector.calculate_trading_statistics()
        assert stats['total_patterns'] == 0
        assert stats['suspicious_patterns'] == 0
        assert stats['trading_rate'] == 0.0
        
        # Add some trading patterns
        base_date = datetime.now()
        
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=3),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        stats = self.detector.calculate_trading_statistics()
        assert stats['total_patterns'] >= 1
        assert stats['total_researchers_involved'] >= 2
    
    def test_trading_likelihood_prediction(self):
        """Test prediction of trading likelihood."""
        # Test with no history
        likelihood = self.detector.predict_trading_likelihood("researcher_001", "researcher_002")
        assert 0.0 <= likelihood <= 1.0
        
        # Create some trading history
        base_date = datetime.now()
        
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=3),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # Should have higher likelihood now
        new_likelihood = self.detector.predict_trading_likelihood("researcher_001", "researcher_002")
        assert new_likelihood > likelihood
    
    def test_generate_trading_report(self):
        """Test generation of trading reports."""
        # Test global report with no data
        report = self.detector.generate_trading_report()
        assert 'statistics' in report
        assert 'network_clusters' in report
        assert 'suspicious_patterns' in report
        assert 'top_traders' in report
        
        # Test researcher-specific report with no data
        report = self.detector.generate_trading_report("researcher_001")
        assert 'error' in report
        
        # Add some data and test again
        base_date = datetime.now()
        
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=3),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # Test researcher-specific report with data
        report = self.detector.generate_trading_report("researcher_001")
        assert 'researcher_id' in report
        assert 'profile' in report
        assert 'patterns' in report
        assert 'risk_level' in report
    
    def test_save_and_load_data(self):
        """Test saving and loading data."""
        # Add some data
        base_date = datetime.now()
        
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=8.0,
            decision=ReviewDecision.ACCEPT
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=3),
            score=8.5,
            decision=ReviewDecision.ACCEPT
        )
        
        # Save data
        self.detector.save_data("test_data.json")
        
        # Create new detector and load data
        new_detector = ReviewTradingDetector(data_dir=self.temp_dir)
        new_detector.load_data("test_data.json")
        
        # Verify data was loaded correctly
        assert len(new_detector.review_records) == 2
        assert len(new_detector.trading_patterns) >= 1
        assert len(new_detector.researcher_profiles) >= 2
    
    def test_configuration_parameters(self):
        """Test configuration parameter effects."""
        # Test with different min_trading_score
        base_date = datetime.now()
        
        # Create low-score trading pattern
        self.detector.record_review(
            review_id="review_001",
            reviewer_id="researcher_001",
            paper_id="paper_002",
            author_id="researcher_002",
            review_date=base_date,
            score=6.0,  # Lower score
            decision=ReviewDecision.MAJOR_REVISION
        )
        
        self.detector.record_review(
            review_id="review_002",
            reviewer_id="researcher_002",
            paper_id="paper_001",
            author_id="researcher_001",
            review_date=base_date + timedelta(days=20),  # Longer gap
            score=6.5,
            decision=ReviewDecision.MINOR_REVISION
        )
        
        # Should detect with low threshold
        patterns_low = self.detector.detect_trading_patterns(min_score=0.1)
        patterns_high = self.detector.detect_trading_patterns(min_score=0.8)
        
        # Low threshold should find more patterns than high threshold
        assert len(patterns_low) >= len(patterns_high)