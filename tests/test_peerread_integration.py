"""
Unit tests for PeerRead dataset integration utilities.

Tests cover PeerReadLoader class functionality, data parsing, venue characteristics extraction,
validation logic, and statistical calculations.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.peerread_loader import (
    PeerReadLoader, PeerReadReview, PeerReadPaper, VenueCharacteristics
)
from src.data.validation_metrics import ValidationMetrics, StatisticalComparison, BaselineStatistics
from src.data.enhanced_models import EnhancedResearcher, StructuredReview, EnhancedReviewCriteria
from src.core.exceptions import DatasetError, ValidationError


class TestPeerReadReview(unittest.TestCase):
    """Test PeerReadReview data class and dimension mapping."""
    
    def test_review_creation(self):
        """Test basic review creation."""
        review = PeerReadReview(paper_id="test_paper")
        self.assertEqual(review.paper_id, "test_paper")
        self.assertEqual(review.reviewer_id, "anonymous")
        self.assertIsNone(review.impact)
    
    def test_dimension_mapping(self):
        """Test mapping of PeerRead dimensions to enhanced system."""
        review = PeerReadReview(
            paper_id="test_paper",
            impact=4,
            substance=3,
            soundness_correctness=5,
            originality=2,
            clarity=4,
            meaningful_comparison=3
        )
        
        # Check mapping (1-5 scale to 1-10 scale)
        self.assertEqual(review.significance, 8.0)  # impact * 2
        self.assertEqual(review.technical_quality, 8.0)  # (substance + soundness) / 2 * 2
        self.assertEqual(review.novelty, 4.0)  # originality * 2
        self.assertEqual(review.clarity, 8.0)  # clarity * 2
        self.assertEqual(review.related_work, 6.0)  # meaningful_comparison * 2
        self.assertIsNotNone(review.reproducibility)  # Should be set based on technical quality
    
    def test_dimension_mapping_partial_data(self):
        """Test dimension mapping with partial data."""
        review = PeerReadReview(
            paper_id="test_paper",
            impact=3,
            originality=4
        )
        
        self.assertEqual(review.significance, 6.0)
        self.assertEqual(review.novelty, 8.0)
        self.assertIsNone(review.technical_quality)  # No substance or soundness data
        self.assertIsNone(review.related_work)  # No meaningful_comparison data
    
    def test_dimension_mapping_soundness_only(self):
        """Test mapping when only soundness_correctness is available."""
        review = PeerReadReview(
            paper_id="test_paper",
            soundness_correctness=4
        )
        
        self.assertEqual(review.technical_quality, 8.0)  # soundness * 2
    
    def test_dimension_mapping_both_substance_and_soundness(self):
        """Test mapping when both substance and soundness are available."""
        review = PeerReadReview(
            paper_id="test_paper",
            substance=3,
            soundness_correctness=5
        )
        
        # Should average the two: (3*2 + 5*2) / 2 = 8.0
        self.assertEqual(review.technical_quality, 8.0)


class TestPeerReadPaper(unittest.TestCase):
    """Test PeerReadPaper data class."""
    
    def test_paper_creation(self):
        """Test basic paper creation."""
        paper = PeerReadPaper(
            id="test_paper",
            title="Test Paper",
            abstract="Test abstract"
        )
        
        self.assertEqual(paper.id, "test_paper")
        self.assertEqual(paper.title, "Test Paper")
        self.assertEqual(paper.abstract, "Test abstract")
        self.assertEqual(len(paper.reviews), 0)
        self.assertEqual(len(paper.authors), 0)
    
    def test_paper_with_reviews(self):
        """Test paper with reviews."""
        review1 = PeerReadReview(paper_id="test_paper", impact=4)
        review2 = PeerReadReview(paper_id="test_paper", substance=3)
        
        paper = PeerReadPaper(
            id="test_paper",
            title="Test Paper",
            abstract="Test abstract",
            reviews=[review1, review2]
        )
        
        self.assertEqual(len(paper.reviews), 2)
        self.assertEqual(paper.reviews[0].significance, 8.0)
        self.assertEqual(paper.reviews[1].technical_quality, 6.0)


class TestVenueCharacteristics(unittest.TestCase):
    """Test VenueCharacteristics data class."""
    
    def test_venue_creation(self):
        """Test basic venue characteristics creation."""
        venue = VenueCharacteristics(
            name="ACL",
            venue_type="conference",
            field="NLP"
        )
        
        self.assertEqual(venue.name, "ACL")
        self.assertEqual(venue.venue_type, "conference")
        self.assertEqual(venue.field, "NLP")
        self.assertEqual(venue.total_papers, 0)
        self.assertEqual(venue.acceptance_rate, 0.0)
    
    def test_venue_with_statistics(self):
        """Test venue with calculated statistics."""
        venue = VenueCharacteristics(
            name="ICLR",
            venue_type="conference",
            field="AI",
            total_papers=100,
            accepted_papers=30,
            impact_scores=[3, 4, 3, 5, 4],
            substance_scores=[4, 4, 3, 4, 5]
        )
        
        venue.acceptance_rate = venue.accepted_papers / venue.total_papers
        venue.impact_mean = sum(venue.impact_scores) / len(venue.impact_scores)
        venue.substance_mean = sum(venue.substance_scores) / len(venue.substance_scores)
        
        self.assertEqual(venue.acceptance_rate, 0.3)
        self.assertEqual(venue.impact_mean, 3.8)
        self.assertEqual(venue.substance_mean, 4.0)


class TestPeerReadLoader(unittest.TestCase):
    """Test PeerReadLoader class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.peerread_path = Path(self.temp_dir) / "PeerRead"
        self.data_path = self.peerread_path / "data"
        self.data_path.mkdir(parents=True)
        
        # Create mock venue directory structure
        self.acl_path = self.data_path / "acl_2017"
        self.acl_train_path = self.acl_path / "train"
        self.acl_reviews_path = self.acl_train_path / "reviews"
        self.acl_pdfs_path = self.acl_train_path / "parsed_pdfs"
        
        self.acl_reviews_path.mkdir(parents=True)
        self.acl_pdfs_path.mkdir(parents=True)
        
        # Create sample review file
        sample_review = {
            "id": "104",
            "title": "Test Paper Title",
            "abstract": "Test abstract content",
            "authors": ["Author One", "Author Two"],
            "accepted": True,
            "reviews": [
                {
                    "IMPACT": "4",
                    "SUBSTANCE": "3",
                    "SOUNDNESS_CORRECTNESS": "4",
                    "ORIGINALITY": "3",
                    "CLARITY": "4",
                    "MEANINGFUL_COMPARISON": "2",
                    "RECOMMENDATION": "3",
                    "REVIEWER_CONFIDENCE": "4",
                    "comments": "This is a test review with detailed comments about the paper.",
                    "is_meta_review": False
                }
            ]
        }
        
        with open(self.acl_reviews_path / "104.json", 'w') as f:
            json.dump(sample_review, f)
        
        # Create sample parsed PDF file
        sample_pdf = {
            "metadata": {
                "title": "Test Paper Title",
                "abstractText": "Test abstract content",
                "authors": ["Author One", "Author Two"],
                "year": 2017,
                "sections": [
                    {
                        "heading": "Introduction",
                        "text": "This is the introduction section."
                    },
                    {
                        "heading": "Methods",
                        "text": "This is the methods section."
                    }
                ]
            }
        }
        
        with open(self.acl_pdfs_path / "104.pdf.json", 'w') as f:
            json.dump(sample_pdf, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_loader_initialization(self):
        """Test PeerReadLoader initialization."""
        loader = PeerReadLoader(str(self.peerread_path))
        self.assertEqual(loader.peerread_path, self.peerread_path)
        self.assertEqual(len(loader.papers), 0)
        self.assertEqual(len(loader.venues), 0)
    
    def test_loader_initialization_invalid_path(self):
        """Test loader initialization with invalid path."""
        with self.assertRaises(DatasetError):
            PeerReadLoader("/nonexistent/path")
    
    def test_load_venue(self):
        """Test loading a specific venue."""
        loader = PeerReadLoader(str(self.peerread_path))
        venue_chars = loader.load_venue("acl_2017")
        
        self.assertIsNotNone(venue_chars)
        self.assertEqual(venue_chars.name, "ACL")
        self.assertEqual(venue_chars.venue_type, "conference")
        self.assertEqual(venue_chars.field, "NLP")
        self.assertEqual(venue_chars.total_papers, 1)
        self.assertEqual(venue_chars.accepted_papers, 1)
        self.assertEqual(venue_chars.acceptance_rate, 0.25)  # Known acceptance rate
    
    def test_load_unknown_venue(self):
        """Test loading unknown venue."""
        loader = PeerReadLoader(str(self.peerread_path))
        venue_chars = loader.load_venue("unknown_venue")
        
        self.assertIsNone(venue_chars)
    
    def test_load_review_file(self):
        """Test loading and parsing review file."""
        loader = PeerReadLoader(str(self.peerread_path))
        loader._load_review_file(self.acl_reviews_path / "104.json", "ACL")
        
        self.assertEqual(len(loader.papers), 1)
        paper = loader.papers["104"]
        self.assertEqual(paper.title, "Test Paper Title")
        self.assertEqual(paper.venue, "ACL")
        self.assertTrue(paper.accepted)
        self.assertEqual(len(paper.reviews), 1)
        
        review = paper.reviews[0]
        self.assertEqual(review.impact, 4)
        self.assertEqual(review.substance, 3)
        self.assertEqual(review.significance, 8.0)  # impact * 2
        self.assertEqual(review.technical_quality, 7.0)  # (substance + soundness) / 2 * 2
    
    def test_load_paper_file(self):
        """Test loading and parsing PDF file."""
        loader = PeerReadLoader(str(self.peerread_path))
        loader._load_paper_file(self.acl_pdfs_path / "104.pdf.json", "ACL")
        
        self.assertEqual(len(loader.papers), 1)
        paper = loader.papers["104"]
        self.assertEqual(paper.title, "Test Paper Title")
        self.assertEqual(paper.year, 2017)
        self.assertIn("Introduction", paper.content)
        self.assertIn("Methods", paper.content)
    
    def test_parse_review(self):
        """Test parsing individual review data."""
        loader = PeerReadLoader(str(self.peerread_path))
        review_data = {
            "IMPACT": "4",
            "SUBSTANCE": "3",
            "ORIGINALITY": "5",
            "comments": "Test review comments",
            "REVIEWER_CONFIDENCE": "3"
        }
        
        review = loader._parse_review(review_data, "test_paper", "reviewer_1")
        
        self.assertEqual(review.paper_id, "test_paper")
        self.assertEqual(review.reviewer_id, "reviewer_1")
        self.assertEqual(review.impact, 4)
        self.assertEqual(review.substance, 3)
        self.assertEqual(review.originality, 5)
        self.assertEqual(review.comments, "Test review comments")
        self.assertEqual(review.reviewer_confidence, 3)
        
        # Check mapped dimensions
        self.assertEqual(review.significance, 8.0)
        self.assertEqual(review.novelty, 10.0)
    
    def test_parse_review_invalid_scores(self):
        """Test parsing review with invalid score values."""
        loader = PeerReadLoader(str(self.peerread_path))
        review_data = {
            "IMPACT": "invalid",
            "SUBSTANCE": None,
            "ORIGINALITY": "3",
            "comments": "Test review"
        }
        
        review = loader._parse_review(review_data, "test_paper", "reviewer_1")
        
        self.assertIsNone(review.impact)
        self.assertIsNone(review.substance)
        self.assertEqual(review.originality, 3)
        self.assertIsNone(review.significance)  # Should be None due to invalid impact
        self.assertEqual(review.novelty, 6.0)  # originality * 2
    
    def test_calculate_venue_statistics(self):
        """Test calculation of venue statistics."""
        loader = PeerReadLoader(str(self.peerread_path))
        
        # Add test data
        paper1 = PeerReadPaper(id="1", title="Paper 1", abstract="Abstract 1", venue="ACL", accepted=True)
        paper1.reviews = [
            PeerReadReview(paper_id="1", impact=4, substance=3),
            PeerReadReview(paper_id="1", impact=3, substance=4)
        ]
        
        paper2 = PeerReadPaper(id="2", title="Paper 2", abstract="Abstract 2", venue="ACL", accepted=False)
        paper2.reviews = [
            PeerReadReview(paper_id="2", impact=2, substance=2)
        ]
        
        loader.papers = {"1": paper1, "2": paper2}
        loader.venues = {"ACL": VenueCharacteristics(name="ACL", venue_type="conference", field="NLP")}
        
        loader._calculate_venue_statistics()
        
        venue_chars = loader.venues["ACL"]
        self.assertEqual(venue_chars.total_papers, 2)
        self.assertEqual(venue_chars.accepted_papers, 1)
        self.assertEqual(venue_chars.acceptance_rate, 0.25)  # Known rate overrides calculated
        self.assertEqual(venue_chars.avg_reviews_per_paper, 1.5)  # (2 + 1) / 2
        
        # Check score distributions
        self.assertEqual(len(venue_chars.impact_scores), 3)  # 4, 3, 2
        self.assertEqual(venue_chars.impact_mean, 3.0)  # (4 + 3 + 2) / 3
    
    def test_get_papers_by_venue(self):
        """Test getting papers by venue."""
        loader = PeerReadLoader(str(self.peerread_path))
        
        paper1 = PeerReadPaper(id="1", title="Paper 1", abstract="Abstract 1", venue="ACL")
        paper2 = PeerReadPaper(id="2", title="Paper 2", abstract="Abstract 2", venue="ICLR")
        paper3 = PeerReadPaper(id="3", title="Paper 3", abstract="Abstract 3", venue="ACL")
        
        loader.papers = {"1": paper1, "2": paper2, "3": paper3}
        
        acl_papers = loader.get_papers_by_venue("ACL")
        self.assertEqual(len(acl_papers), 2)
        self.assertEqual(acl_papers[0].id, "1")
        self.assertEqual(acl_papers[1].id, "3")
        
        iclr_papers = loader.get_papers_by_venue("ICLR")
        self.assertEqual(len(iclr_papers), 1)
        self.assertEqual(iclr_papers[0].id, "2")
    
    def test_validate_data_format(self):
        """Test data format validation."""
        loader = PeerReadLoader(str(self.peerread_path))
        
        # Add test data with various completeness levels
        paper1 = PeerReadPaper(id="1", title="Paper 1", abstract="Abstract 1", venue="ACL", content="Content 1")
        paper1.reviews = [PeerReadReview(paper_id="1", impact=4, comments="Review 1")]
        
        paper2 = PeerReadPaper(id="2", title="", abstract="Abstract 2", venue="ACL")  # Missing title
        
        paper3 = PeerReadPaper(id="3", title="Paper 3", abstract="", venue="ACL")  # Missing abstract
        
        loader.papers = {"1": paper1, "2": paper2, "3": paper3}
        loader.venues = {"ACL": VenueCharacteristics(name="ACL", venue_type="conference", field="NLP", total_papers=3)}
        
        validation_results = loader.validate_data_format()
        
        self.assertEqual(validation_results["total_papers"], 3)
        self.assertEqual(validation_results["papers_with_reviews"], 1)
        self.assertEqual(validation_results["papers_with_content"], 1)
        self.assertEqual(validation_results["reviews_with_scores"], 1)
        self.assertEqual(len(validation_results["missing_data"]), 2)  # Missing title and abstract
    
    def test_export_statistics(self):
        """Test exporting comprehensive statistics."""
        loader = PeerReadLoader(str(self.peerread_path))
        
        # Add test data
        paper1 = PeerReadPaper(id="1", title="Paper 1", abstract="Abstract 1", venue="ACL")
        paper1.reviews = [
            PeerReadReview(paper_id="1", impact=4, substance=3, comments="Review 1"),
            PeerReadReview(paper_id="1", impact=3, substance=4, comments="Review 2")
        ]
        
        loader.papers = {"1": paper1}
        loader.venues = {
            "ACL": VenueCharacteristics(
                name="ACL", venue_type="conference", field="NLP",
                total_papers=1, accepted_papers=1, acceptance_rate=0.25,
                avg_reviews_per_paper=2.0, avg_review_length=8.0,
                impact_mean=3.5, substance_mean=3.5
            )
        }
        
        stats = loader.export_statistics()
        
        self.assertEqual(stats["dataset_summary"]["total_papers"], 1)
        self.assertEqual(stats["dataset_summary"]["total_venues"], 1)
        self.assertEqual(stats["dataset_summary"]["total_reviews"], 2)
        
        acl_stats = stats["venues"]["ACL"]
        self.assertEqual(acl_stats["type"], "conference")
        self.assertEqual(acl_stats["field"], "NLP")
        self.assertEqual(acl_stats["acceptance_rate"], 0.25)
        self.assertEqual(acl_stats["score_means"]["impact"], 3.5)
        
        # Check review dimension statistics
        self.assertIn("impact", stats["review_dimension_stats"])
        impact_stats = stats["review_dimension_stats"]["impact"]
        self.assertEqual(impact_stats["mean"], 3.5)
        self.assertEqual(impact_stats["count"], 2)


class TestValidationMetrics(unittest.TestCase):
    """Test ValidationMetrics class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock PeerRead loader with test data
        self.mock_loader = MagicMock()
        
        # Mock venue characteristics
        venue_chars = VenueCharacteristics(
            name="ACL",
            venue_type="conference",
            field="NLP",
            total_papers=100,
            accepted_papers=25,
            acceptance_rate=0.25,
            avg_reviews_per_paper=3.0,
            avg_review_length=500.0,
            impact_mean=3.2,
            substance_mean=3.4
        )
        
        self.mock_loader.venues = {"ACL": venue_chars}
        
        # Mock papers with reviews
        paper1 = PeerReadPaper(id="1", title="Paper 1", abstract="Abstract 1", venue="ACL")
        paper1.reviews = [
            PeerReadReview(paper_id="1", impact=4, substance=3, comments="Review 1" * 50),
            PeerReadReview(paper_id="1", impact=3, substance=4, comments="Review 2" * 40)
        ]
        
        paper2 = PeerReadPaper(id="2", title="Paper 2", abstract="Abstract 2", venue="ACL")
        paper2.reviews = [
            PeerReadReview(paper_id="2", impact=2, substance=3, comments="Review 3" * 60)
        ]
        
        self.mock_loader.get_papers_by_venue.return_value = [paper1, paper2]
        
        self.validation_metrics = ValidationMetrics(self.mock_loader)
    
    def test_initialization(self):
        """Test ValidationMetrics initialization."""
        self.assertIsNotNone(self.validation_metrics.peerread_loader)
        self.assertEqual(len(self.validation_metrics.baseline_stats), 1)
        self.assertIn("ACL", self.validation_metrics.baseline_stats)
    
    def test_baseline_statistics_calculation(self):
        """Test calculation of baseline statistics."""
        baseline = self.validation_metrics.baseline_stats["ACL"]
        
        self.assertEqual(baseline.venue_name, "ACL")
        self.assertGreater(baseline.avg_review_length, 0)
        self.assertEqual(baseline.reviews_per_paper_mean, 1.5)  # (2 + 1) / 2
        self.assertEqual(baseline.acceptance_rate, 0.25)
        
        # Check score statistics
        self.assertIn("mean", baseline.impact_stats)
        self.assertEqual(baseline.impact_stats["mean"], 3.0)  # (4 + 3 + 2) / 3
    
    def test_statistical_comparison(self):
        """Test StatisticalComparison class."""
        sim_data = [3.5, 4.0, 3.0, 3.8, 4.2]
        real_data = [3.2, 3.8, 3.1, 3.9, 4.0]
        
        comparison = StatisticalComparison("test_metric", sim_data, real_data)
        
        self.assertEqual(comparison.metric_name, "test_metric")
        self.assertIsNotNone(comparison.sim_mean)
        self.assertIsNotNone(comparison.real_mean)
        self.assertIsNotNone(comparison.wasserstein_distance)
        self.assertIsNotNone(comparison.ks_statistic)
        self.assertIsNotNone(comparison.similarity_score)
        self.assertGreater(comparison.similarity_score, 0)
        self.assertLessEqual(comparison.similarity_score, 1)
    
    def test_compare_distributions(self):
        """Test distribution comparison."""
        simulation_data = {
            "impact_scores": [3.5, 4.0, 3.0, 3.8],
            "substance_scores": [3.2, 3.8, 3.5, 4.0],
            "review_lengths": [450, 520, 480, 510]
        }
        
        comparisons = self.validation_metrics.compare_distributions(simulation_data, "ACL")
        
        self.assertIn("impact_scores", comparisons)
        self.assertIn("substance_scores", comparisons)
        self.assertIn("review_lengths", comparisons)
        
        impact_comparison = comparisons["impact_scores"]
        self.assertIsInstance(impact_comparison, StatisticalComparison)
        self.assertEqual(len(impact_comparison.simulation_values), 4)
        self.assertGreater(impact_comparison.similarity_score, 0)
    
    def test_kl_divergence_calculation(self):
        """Test KL divergence calculation."""
        sim_data = [3.0, 3.5, 4.0, 3.2, 3.8]
        real_data = [3.1, 3.4, 3.9, 3.3, 3.7]
        
        kl_div = self.validation_metrics.calculate_kl_divergence(sim_data, real_data)
        
        self.assertIsInstance(kl_div, float)
        self.assertGreaterEqual(kl_div, 0)  # KL divergence is non-negative
    
    def test_wasserstein_distance_calculation(self):
        """Test Wasserstein distance calculation."""
        sim_data = [3.0, 3.5, 4.0, 3.2, 3.8]
        real_data = [3.1, 3.4, 3.9, 3.3, 3.7]
        
        wd = self.validation_metrics.calculate_wasserstein_distance(sim_data, real_data)
        
        self.assertIsInstance(wd, float)
        self.assertGreaterEqual(wd, 0)  # Wasserstein distance is non-negative
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        sim_data = {
            "metric1": [1, 2, 3, 4, 5],
            "metric2": [2, 4, 6, 8, 10]
        }
        real_data = {
            "metric1": [1.1, 2.1, 2.9, 4.1, 4.9],
            "metric2": [2.2, 3.8, 6.1, 7.9, 10.1]
        }
        
        correlations = self.validation_metrics.calculate_correlation_analysis(sim_data, real_data)
        
        self.assertIn("metric1", correlations)
        self.assertIn("metric2", correlations)
        self.assertGreater(correlations["metric1"], 0.9)  # Should be highly correlated
        self.assertGreater(correlations["metric2"], 0.9)  # Should be highly correlated
    
    def test_realism_indicators(self):
        """Test creation of realism indicators."""
        simulation_data = {
            "avg_review_scores": [3.2, 3.5, 3.8, 3.1],
            "review_lengths": [480, 520, 450, 510],
            "acceptance_rate": 0.28
        }
        
        indicators = self.validation_metrics.create_realism_indicators(simulation_data, "ACL")
        
        self.assertIn("review_score_realism", indicators)
        self.assertIn("review_length_realism", indicators)
        self.assertIn("acceptance_rate_realism", indicators)
        
        score_indicator = indicators["review_score_realism"]
        self.assertEqual(score_indicator.category, "review_quality")
        self.assertGreater(score_indicator.similarity_score, 0)
        self.assertIn(score_indicator.status, ["OK", "WARNING", "CRITICAL"])
    
    def test_continuous_validation_monitoring(self):
        """Test continuous validation monitoring."""
        simulation_results = {
            "avg_review_scores": [3.0, 3.2, 3.5],
            "review_lengths": [400, 450, 500],
            "acceptance_rate": 0.30
        }
        
        report = self.validation_metrics.monitor_continuous_validation(simulation_results, "ACL")
        
        self.assertIn("timestamp", report)
        self.assertIn("venue_name", report)
        self.assertIn("alerts", report)
        self.assertIn("warnings", report)
        self.assertIn("realism_scores", report)
        self.assertIn("overall_realism", report)
        
        self.assertEqual(report["venue_name"], "ACL")
        self.assertIsInstance(report["alerts"], list)
        self.assertIsInstance(report["warnings"], list)
        self.assertGreaterEqual(report["overall_realism"], 0)
        self.assertLessEqual(report["overall_realism"], 1)
    
    def test_baseline_statistics_export(self):
        """Test getting baseline statistics."""
        baseline = self.validation_metrics.get_baseline_statistics("ACL")
        
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.venue_name, "ACL")
        self.assertEqual(baseline.acceptance_rate, 0.25)
        
        # Test non-existent venue
        baseline_none = self.validation_metrics.get_baseline_statistics("NonExistent")
        self.assertIsNone(baseline_none)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.peerread_path = Path(self.temp_dir) / "PeerRead"
        self.data_path = self.peerread_path / "data"
        
        # Create comprehensive test dataset
        self._create_comprehensive_test_dataset()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_comprehensive_test_dataset(self):
        """Create a comprehensive test dataset with multiple venues."""
        # Create ACL venue
        acl_path = self.data_path / "acl_2017" / "train"
        acl_reviews_path = acl_path / "reviews"
        acl_pdfs_path = acl_path / "parsed_pdfs"
        acl_reviews_path.mkdir(parents=True)
        acl_pdfs_path.mkdir(parents=True)
        
        # Create ICLR venue
        iclr_path = self.data_path / "iclr_2017" / "train"
        iclr_reviews_path = iclr_path / "reviews"
        iclr_pdfs_path = iclr_path / "parsed_pdfs"
        iclr_reviews_path.mkdir(parents=True)
        iclr_pdfs_path.mkdir(parents=True)
        
        # Create sample papers for ACL
        for i in range(5):
            paper_id = f"acl_{i}"
            
            # Review file
            review_data = {
                "id": paper_id,
                "title": f"ACL Paper {i}",
                "abstract": f"Abstract for ACL paper {i}",
                "authors": [f"Author {i}A", f"Author {i}B"],
                "accepted": i < 2,  # First 2 papers accepted
                "reviews": [
                    {
                        "IMPACT": str(3 + i % 3),
                        "SUBSTANCE": str(3 + (i + 1) % 3),
                        "SOUNDNESS_CORRECTNESS": str(3 + (i + 2) % 3),
                        "ORIGINALITY": str(2 + i % 4),
                        "CLARITY": str(3 + i % 3),
                        "MEANINGFUL_COMPARISON": str(2 + i % 3),
                        "RECOMMENDATION": str(2 + i % 3),
                        "REVIEWER_CONFIDENCE": str(3 + i % 3),
                        "comments": f"Review comments for ACL paper {i}. " * (10 + i * 5),
                        "is_meta_review": False
                    },
                    {
                        "IMPACT": str(2 + i % 4),
                        "SUBSTANCE": str(3 + i % 3),
                        "ORIGINALITY": str(3 + i % 3),
                        "CLARITY": str(2 + i % 4),
                        "comments": f"Second review for ACL paper {i}. " * (8 + i * 3),
                        "is_meta_review": False
                    }
                ]
            }
            
            with open(acl_reviews_path / f"{paper_id}.json", 'w') as f:
                json.dump(review_data, f)
            
            # PDF file
            pdf_data = {
                "metadata": {
                    "title": f"ACL Paper {i}",
                    "abstractText": f"Abstract for ACL paper {i}",
                    "authors": [f"Author {i}A", f"Author {i}B"],
                    "year": 2017,
                    "sections": [
                        {
                            "heading": "Introduction",
                            "text": f"Introduction content for paper {i}."
                        },
                        {
                            "heading": "Methods",
                            "text": f"Methods content for paper {i}."
                        }
                    ]
                }
            }
            
            with open(acl_pdfs_path / f"{paper_id}.pdf.json", 'w') as f:
                json.dump(pdf_data, f)
        
        # Create sample papers for ICLR (similar structure)
        for i in range(3):
            paper_id = f"iclr_{i}"
            
            review_data = {
                "id": paper_id,
                "title": f"ICLR Paper {i}",
                "abstract": f"Abstract for ICLR paper {i}",
                "authors": [f"Author {i}X", f"Author {i}Y"],
                "accepted": i < 1,  # First paper accepted
                "reviews": [
                    {
                        "IMPACT": str(4 + i % 2),
                        "SUBSTANCE": str(4 + i % 2),
                        "SOUNDNESS_CORRECTNESS": str(3 + i % 3),
                        "ORIGINALITY": str(3 + i % 3),
                        "CLARITY": str(4 + i % 2),
                        "MEANINGFUL_COMPARISON": str(3 + i % 2),
                        "comments": f"ICLR review comments for paper {i}. " * (15 + i * 8),
                        "is_meta_review": False
                    }
                ]
            }
            
            with open(iclr_reviews_path / f"{paper_id}.json", 'w') as f:
                json.dump(review_data, f)
    
    def test_full_pipeline_integration(self):
        """Test full pipeline from loading to validation."""
        # Load PeerRead data
        loader = PeerReadLoader(str(self.peerread_path))
        venues = loader.load_all_venues()
        
        # Verify venues loaded
        self.assertIn("ACL", venues)
        self.assertIn("ICLR", venues)
        
        acl_venue = venues["ACL"]
        self.assertEqual(acl_venue.total_papers, 5)
        self.assertEqual(acl_venue.accepted_papers, 2)
        self.assertEqual(acl_venue.field, "NLP")
        
        iclr_venue = venues["ICLR"]
        self.assertEqual(iclr_venue.total_papers, 3)
        self.assertEqual(iclr_venue.accepted_papers, 1)
        self.assertEqual(iclr_venue.field, "AI")
        
        # Validate data format
        validation_results = loader.validate_data_format()
        self.assertEqual(validation_results["total_papers"], 8)  # 5 ACL + 3 ICLR
        self.assertEqual(validation_results["total_venues"], 2)
        self.assertGreater(validation_results["papers_with_reviews"], 0)
        
        # Create validation metrics
        validation_metrics = ValidationMetrics(loader)
        
        # Test baseline statistics
        acl_baseline = validation_metrics.get_baseline_statistics("ACL")
        self.assertIsNotNone(acl_baseline)
        self.assertEqual(acl_baseline.venue_name, "ACL")
        self.assertGreater(acl_baseline.avg_review_length, 0)
        
        # Test distribution comparison
        simulation_data = {
            "impact_scores": [3.5, 4.0, 3.2, 3.8, 4.1],
            "substance_scores": [3.8, 3.5, 4.0, 3.6, 3.9],
            "review_lengths": [450, 520, 480, 510, 495]
        }
        
        comparisons = validation_metrics.compare_distributions(simulation_data, "ACL")
        self.assertGreater(len(comparisons), 0)
        
        for comparison in comparisons.values():
            self.assertIsInstance(comparison, StatisticalComparison)
            self.assertGreater(comparison.similarity_score, 0)
        
        # Test continuous monitoring
        simulation_results = {
            "avg_review_scores": [3.4, 3.6, 3.8],
            "review_lengths": [480, 520, 450],
            "acceptance_rate": 0.35
        }
        
        report = validation_metrics.monitor_continuous_validation(simulation_results, "ACL")
        self.assertIn("overall_realism", report)
        self.assertGreaterEqual(report["overall_realism"], 0)
        
        # Export statistics
        stats = loader.export_statistics()
        self.assertEqual(stats["dataset_summary"]["total_papers"], 8)
        self.assertEqual(stats["dataset_summary"]["total_venues"], 2)
        self.assertIn("ACL", stats["venues"])
        self.assertIn("ICLR", stats["venues"])
    
    def test_dimension_mapping_consistency(self):
        """Test consistency of dimension mapping across the pipeline."""
        loader = PeerReadLoader(str(self.peerread_path))
        loader.load_all_venues()
        
        # Get a paper with reviews
        acl_papers = loader.get_papers_by_venue("ACL")
        self.assertGreater(len(acl_papers), 0)
        
        paper = acl_papers[0]
        self.assertGreater(len(paper.reviews), 0)
        
        review = paper.reviews[0]
        
        # Test dimension mapping consistency
        if review.impact is not None:
            expected_significance = review.impact * 2.0
            self.assertEqual(review.significance, expected_significance)
        
        if review.originality is not None:
            expected_novelty = review.originality * 2.0
            self.assertEqual(review.novelty, expected_novelty)
        
        if review.substance is not None and review.soundness_correctness is not None:
            expected_technical = (review.substance * 2.0 + review.soundness_correctness * 2.0) / 2.0
            self.assertEqual(review.technical_quality, expected_technical)
        elif review.substance is not None:
            expected_technical = review.substance * 2.0
            self.assertEqual(review.technical_quality, expected_technical)
        elif review.soundness_correctness is not None:
            expected_technical = review.soundness_correctness * 2.0
            self.assertEqual(review.technical_quality, expected_technical)
    
    def test_venue_calibration_accuracy(self):
        """Test accuracy of venue calibration with known acceptance rates."""
        loader = PeerReadLoader(str(self.peerread_path))
        venues = loader.load_all_venues()
        
        # Test known acceptance rates are applied
        acl_venue = venues["ACL"]
        self.assertEqual(acl_venue.acceptance_rate, 0.25)  # Known ACL rate
        
        iclr_venue = venues["ICLR"]
        self.assertEqual(iclr_venue.acceptance_rate, 0.30)  # Known ICLR rate
        
        # Test that calculated rates are overridden by known rates
        # ACL has 2/5 = 0.4 calculated, but should use 0.25 known rate
        # ICLR has 1/3 = 0.33 calculated, but should use 0.30 known rate


if __name__ == '__main__':
    unittest.main()