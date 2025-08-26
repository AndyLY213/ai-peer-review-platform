"""
Unit tests for the Availability Bias Model.

Tests the AvailabilityBiasModel class and its ability to model bias based on
recent exposure to similar work affecting reviewer judgment.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.enhancements.availability_bias_model import (
    AvailabilityBiasModel, RecentExposure, TopicSimilarity
)
from src.enhancements.bias_engine import BiasConfiguration, BiasType
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, BiasEffect,
    ResearcherLevel, ReviewDecision
)


class TestRecentExposure:
    """Test cases for RecentExposure dataclass."""
    
    def test_recent_exposure_creation(self):
        """Test creating a recent exposure."""
        exposure = RecentExposure(
            paper_id="test_paper",
            title="Test Paper Title",
            abstract="This is a test abstract about machine learning",
            keywords=["machine learning", "AI", "neural networks"],
            exposure_date=datetime.now() - timedelta(days=5),
            exposure_type="reviewed"
        )
        
        assert exposure.paper_id == "test_paper"
        assert exposure.title == "Test Paper Title"
        assert exposure.exposure_type == "reviewed"
        assert len(exposure.keywords) == 3
        assert exposure.relevance_score > 0  # Should be calculated automatically
    
    def test_relevance_score_calculation(self):
        """Test relevance score calculation based on exposure type and recency."""
        now = datetime.now()
        
        # Recent reviewed paper (high relevance)
        recent_reviewed = RecentExposure(
            paper_id="recent",
            title="Recent Paper",
            abstract="Abstract",
            keywords=["AI"],
            exposure_date=now - timedelta(days=1),
            exposure_type="reviewed"
        )
        
        # Old discussed paper (low relevance)
        old_discussed = RecentExposure(
            paper_id="old",
            title="Old Paper",
            abstract="Abstract",
            keywords=["AI"],
            exposure_date=now - timedelta(days=25),
            exposure_type="discussed"
        )
        
        assert recent_reviewed.relevance_score > old_discussed.relevance_score
        assert 0.0 <= old_discussed.relevance_score <= 1.0
        assert 0.0 <= recent_reviewed.relevance_score <= 1.0


class TestTopicSimilarity:
    """Test cases for TopicSimilarity dataclass."""
    
    def test_topic_similarity_creation(self):
        """Test creating a topic similarity object."""
        similarity = TopicSimilarity(
            similarity_score=0.8,
            matching_keywords=["machine learning", "neural networks"],
            semantic_similarity=0.7
        )
        
        assert similarity.similarity_score == 0.8
        assert len(similarity.matching_keywords) == 2
        assert similarity.semantic_similarity == 0.7
        assert similarity.get_overall_similarity() == 0.8  # Currently just returns similarity_score


class TestAvailabilityBiasModel:
    """Test cases for AvailabilityBiasModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BiasConfiguration(
            bias_type=BiasType.AVAILABILITY,
            base_strength=0.5,
            parameters={
                'recency_window_days': 30,
                'similarity_threshold': 0.6,
                'max_adjustment': 1.0
            }
        )
        self.model = AvailabilityBiasModel(self.config)
        
        self.reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI",
            cognitive_biases={"availability": 0.6}
        )
        
        self.review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id="test_paper",
            venue_id="test_venue"
        )
    
    def test_model_initialization(self):
        """Test availability bias model initialization."""
        assert self.model.bias_type == BiasType.AVAILABILITY
        assert self.model.recency_window_days == 30
        assert self.model.similarity_threshold == 0.6
        assert self.model.max_adjustment == 1.0
        assert len(self.model.reviewer_exposures) == 0  # Should start empty
        assert len(self.model.research_keywords) > 0  # Should have keyword sets
    
    def test_research_keywords_initialization(self):
        """Test that research keywords are properly initialized."""
        keywords = self.model.research_keywords
        
        assert "AI" in keywords
        assert "NLP" in keywords
        assert "Computer Vision" in keywords
        assert "Machine Learning" in keywords
        
        # Check that each field has keywords
        for field, keyword_set in keywords.items():
            assert len(keyword_set) > 0
            assert isinstance(keyword_set, set)
    
    def test_is_applicable_with_exposures(self):
        """Test is_applicable with recent exposures available."""
        # Add a recent exposure
        self.model.add_exposure(
            reviewer_id="test_reviewer",
            paper_id="similar_paper",
            title="Similar AI Paper",
            abstract="This paper discusses machine learning techniques",
            keywords=["machine learning", "AI"],
            exposure_type="reviewed"
        )
        
        context = {
            'paper_content': {
                'title': 'New AI Research',
                'abstract': 'This paper presents novel machine learning approaches',
                'keywords': ['machine learning', 'neural networks']
            }
        }
        
        assert self.model.is_applicable(self.reviewer, context) is True
    
    def test_is_applicable_without_exposures(self):
        """Test is_applicable without recent exposures."""
        context = {
            'paper_content': {
                'title': 'New AI Research',
                'abstract': 'This paper presents novel approaches'
            }
        }
        
        assert self.model.is_applicable(self.reviewer, context) is False
    
    def test_is_applicable_no_paper_content(self):
        """Test is_applicable without paper content."""
        context = {}
        
        assert self.model.is_applicable(self.reviewer, context) is False
    
    def test_is_applicable_no_susceptibility(self):
        """Test is_applicable with reviewer having no availability susceptibility."""
        reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI",
            cognitive_biases={"availability": 0.0}  # No susceptibility
        )
        
        context = {
            'paper_content': {
                'title': 'New AI Research',
                'abstract': 'This paper presents novel approaches'
            }
        }
        
        assert self.model.is_applicable(reviewer, context) is False
    
    def test_extract_keywords_from_text(self):
        """Test extracting research keywords from text."""
        text = "This paper discusses machine learning and neural networks for computer vision tasks"
        
        keywords = self.model._extract_keywords_from_text(text)
        
        assert "machine learning" in keywords
        assert "neural networks" in keywords
        assert "computer vision" in keywords
        assert len(keywords) >= 3
    
    def test_calculate_topic_similarity(self):
        """Test calculating similarity between topics."""
        keywords1 = {"machine learning", "neural networks", "AI"}
        keywords2 = {"machine learning", "deep learning", "AI"}
        text1 = "machine learning neural networks artificial intelligence"
        text2 = "machine learning deep learning artificial intelligence"
        
        similarity = self.model._calculate_topic_similarity(keywords1, keywords2, text1, text2)
        
        assert isinstance(similarity, TopicSimilarity)
        assert similarity.similarity_score > 0
        assert "machine learning" in similarity.matching_keywords
        assert "AI" in similarity.matching_keywords
        assert len(similarity.matching_keywords) >= 2
    
    def test_calculate_topic_similarity_no_overlap(self):
        """Test calculating similarity with no keyword overlap."""
        keywords1 = {"biology", "genetics"}
        keywords2 = {"physics", "quantum"}
        text1 = "biology genetics research"
        text2 = "physics quantum mechanics"
        
        similarity = self.model._calculate_topic_similarity(keywords1, keywords2, text1, text2)
        
        assert similarity.similarity_score == 0.0
        assert len(similarity.matching_keywords) == 0
    
    def test_get_recent_exposures(self):
        """Test getting recent exposures within time window."""
        reviewer_id = "test_reviewer"
        now = datetime.now()
        
        # Add exposures at different times
        self.model.add_exposure(
            reviewer_id, "paper1", "Recent Paper", "Abstract", ["AI"],
            "reviewed", now - timedelta(days=5)
        )
        self.model.add_exposure(
            reviewer_id, "paper2", "Old Paper", "Abstract", ["AI"],
            "read", now - timedelta(days=40)  # Outside window
        )
        
        recent_exposures = self.model._get_recent_exposures(reviewer_id)
        
        assert len(recent_exposures) == 1
        assert recent_exposures[0].paper_id == "paper1"
    
    def test_find_most_similar_exposure(self):
        """Test finding the most similar exposure to current paper."""
        # Add some exposures
        exposures = [
            RecentExposure(
                "paper1", "Machine Learning Paper", 
                "This paper discusses neural networks and deep learning",
                ["machine learning", "neural networks"], 
                datetime.now(), "reviewed"
            ),
            RecentExposure(
                "paper2", "Computer Vision Paper",
                "This paper discusses image processing and object detection",
                ["computer vision", "image processing"],
                datetime.now(), "read"
            )
        ]
        
        # Current paper about machine learning
        paper_title = "Deep Neural Networks for Classification"
        paper_abstract = "We present a novel neural network architecture for machine learning"
        paper_keywords = ["neural networks", "machine learning", "classification"]
        
        most_similar, similarity = self.model._find_most_similar_exposure(
            paper_title, paper_abstract, paper_keywords, exposures
        )
        
        assert most_similar.paper_id == "paper1"  # Should match ML paper
        assert similarity.get_overall_similarity() > 0.4  # Adjusted threshold
        assert "machine learning" in similarity.matching_keywords
    
    def test_determine_exposure_valence(self):
        """Test determining whether exposure was positive or negative."""
        reviewed_exposure = RecentExposure(
            "paper1", "Title", "Abstract", ["AI"], datetime.now(), "reviewed"
        )
        cited_exposure = RecentExposure(
            "paper2", "Title", "Abstract", ["AI"], datetime.now(), "cited"
        )
        discussed_exposure = RecentExposure(
            "paper3", "Title", "Abstract", ["AI"], datetime.now(), "discussed"
        )
        
        reviewed_valence = self.model._determine_exposure_valence(reviewed_exposure, {})
        cited_valence = self.model._determine_exposure_valence(cited_exposure, {})
        discussed_valence = self.model._determine_exposure_valence(discussed_exposure, {})
        
        assert cited_valence > reviewed_valence > discussed_valence
        assert -1.0 <= discussed_valence <= 1.0
        assert -1.0 <= reviewed_valence <= 1.0
        assert -1.0 <= cited_valence <= 1.0
    
    def test_calculate_bias_effect_with_similar_exposure(self):
        """Test calculating bias effect with similar recent exposure."""
        # Add a similar exposure
        self.model.add_exposure(
            reviewer_id="test_reviewer",
            paper_id="similar_paper",
            title="Machine Learning for Classification",
            abstract="This paper presents neural network approaches for classification tasks",
            keywords=["machine learning", "neural networks", "classification"],
            exposure_type="cited"
        )
        
        context = {
            'paper_content': {
                'title': 'Deep Learning Classification Methods',
                'abstract': 'Novel neural network architectures for machine learning classification',
                'keywords': ['deep learning', 'neural networks', 'machine learning']
            }
        }
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "availability"
        assert bias_effect.strength > 0
        assert bias_effect.score_adjustment != 0  # Should have some adjustment
        assert "availability bias" in bias_effect.description.lower()
    
    def test_calculate_bias_effect_no_similar_exposure(self):
        """Test calculating bias effect without similar exposures."""
        # Add a dissimilar exposure
        self.model.add_exposure(
            reviewer_id="test_reviewer",
            paper_id="different_paper",
            title="Biology Research Paper",
            abstract="This paper discusses genetic analysis and molecular biology",
            keywords=["biology", "genetics", "molecular"],
            exposure_type="read"
        )
        
        context = {
            'paper_content': {
                'title': 'Machine Learning Classification',
                'abstract': 'Neural networks for classification tasks',
                'keywords': ['machine learning', 'neural networks']
            }
        }
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "availability"
        assert bias_effect.score_adjustment == 0.0  # No similar exposure
        assert "no sufficiently similar" in bias_effect.description.lower()
    
    def test_calculate_bias_effect_no_paper_content(self):
        """Test calculating bias effect without paper content."""
        context = {}
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "availability"
        assert bias_effect.strength == 0.0
        assert bias_effect.score_adjustment == 0.0
        assert "no paper content" in bias_effect.description.lower()
    
    def test_generate_bias_description(self):
        """Test bias description generation."""
        exposure = RecentExposure(
            "paper1", "Similar Paper", "Abstract", ["AI"],
            datetime.now() - timedelta(days=10), "reviewed"
        )
        similarity = TopicSimilarity(similarity_score=0.8, matching_keywords=["AI", "ML"])
        
        # Positive adjustment
        desc = self.model._generate_bias_description(exposure, similarity, 0.3)
        assert "availability bias" in desc.lower()
        assert "highly similar" in desc.lower()  # 0.8 > 0.75 threshold
        assert "positively" in desc.lower()
        assert "10 days ago" in desc
        
        # Negative adjustment
        desc = self.model._generate_bias_description(exposure, similarity, -0.2)
        assert "negatively" in desc.lower()
        
        # Minimal adjustment
        desc = self.model._generate_bias_description(exposure, similarity, 0.02)
        assert "minimal" in desc.lower()
    
    def test_add_exposure(self):
        """Test adding exposure for a reviewer."""
        reviewer_id = "test_reviewer"
        initial_count = len(self.model.reviewer_exposures.get(reviewer_id, []))
        
        self.model.add_exposure(
            reviewer_id=reviewer_id,
            paper_id="new_paper",
            title="New Paper Title",
            abstract="This is a new paper about AI",
            keywords=["AI", "machine learning"],
            exposure_type="read"
        )
        
        assert reviewer_id in self.model.reviewer_exposures
        assert len(self.model.reviewer_exposures[reviewer_id]) == initial_count + 1
        
        added_exposure = self.model.reviewer_exposures[reviewer_id][-1]
        assert added_exposure.paper_id == "new_paper"
        assert added_exposure.title == "New Paper Title"
        assert added_exposure.exposure_type == "read"
    
    def test_add_exposure_limit(self):
        """Test that exposure history is limited to prevent unbounded growth."""
        reviewer_id = "test_reviewer"
        
        # Add more than the limit (50) exposures
        for i in range(55):
            self.model.add_exposure(
                reviewer_id, f"paper_{i}", f"Title {i}", "Abstract", ["AI"], "read"
            )
        
        # Should be limited to 50 exposures
        assert len(self.model.reviewer_exposures[reviewer_id]) == 50
        
        # Should keep the most recent exposures
        last_exposure = self.model.reviewer_exposures[reviewer_id][-1]
        assert last_exposure.paper_id == "paper_54"  # Most recent
    
    def test_get_reviewer_exposures(self):
        """Test getting exposures for a specific reviewer."""
        reviewer_id = "test_reviewer"
        now = datetime.now()
        
        # Add exposures at different times
        self.model.add_exposure(
            reviewer_id, "paper1", "Recent", "Abstract", ["AI"], "read",
            now - timedelta(days=5)
        )
        self.model.add_exposure(
            reviewer_id, "paper2", "Old", "Abstract", ["AI"], "read",
            now - timedelta(days=40)
        )
        
        # Get recent exposures (default window)
        recent = self.model.get_reviewer_exposures(reviewer_id)
        assert len(recent) == 1
        assert recent[0].paper_id == "paper1"
        
        # Get all exposures (larger window)
        all_exposures = self.model.get_reviewer_exposures(reviewer_id, days_back=50)
        assert len(all_exposures) == 2
    
    def test_clear_reviewer_exposures(self):
        """Test clearing exposures for a specific reviewer."""
        reviewer_id = "test_reviewer"
        
        # Add exposure
        self.model.add_exposure(reviewer_id, "paper1", "Title", "Abstract", ["AI"], "read")
        assert len(self.model.reviewer_exposures[reviewer_id]) == 1
        
        # Clear exposures
        self.model.clear_reviewer_exposures(reviewer_id)
        assert reviewer_id not in self.model.reviewer_exposures
    
    def test_get_availability_statistics_empty(self):
        """Test getting statistics with no exposures."""
        stats = self.model.get_availability_statistics()
        
        assert stats["total_reviewers_with_exposures"] == 0
        assert stats["total_exposures"] == 0
        assert stats["average_exposures_per_reviewer"] == 0.0
        assert stats["exposure_types"] == {}
        assert stats["average_relevance_score"] == 0.0
    
    def test_get_availability_statistics_with_data(self):
        """Test getting statistics with exposure data."""
        # Add exposures for multiple reviewers
        for reviewer_id in ["reviewer1", "reviewer2"]:
            for i in range(3):
                self.model.add_exposure(
                    reviewer_id, f"paper_{i}", f"Title {i}", "Abstract", ["AI"],
                    "reviewed" if i == 0 else "read"
                )
        
        stats = self.model.get_availability_statistics()
        
        assert stats["total_reviewers_with_exposures"] == 2
        assert stats["total_exposures"] == 6
        assert stats["average_exposures_per_reviewer"] == 3.0
        assert stats["exposure_types"]["reviewed"] == 2
        assert stats["exposure_types"]["read"] == 4
        assert stats["average_relevance_score"] > 0
        assert stats["recent_exposures"] == 6  # All should be recent
    
    def test_cleanup_old_exposures(self):
        """Test cleaning up old exposures."""
        reviewer_id = "test_reviewer"
        now = datetime.now()
        
        # Add old and recent exposures
        self.model.add_exposure(
            reviewer_id, "old_paper", "Old", "Abstract", ["AI"], "read",
            now - timedelta(days=100)  # Very old
        )
        self.model.add_exposure(
            reviewer_id, "recent_paper", "Recent", "Abstract", ["AI"], "read",
            now - timedelta(days=5)  # Recent
        )
        
        assert len(self.model.reviewer_exposures[reviewer_id]) == 2
        
        # Cleanup old exposures
        self.model.cleanup_old_exposures()
        
        # Should keep only recent exposure
        assert len(self.model.reviewer_exposures[reviewer_id]) == 1
        assert self.model.reviewer_exposures[reviewer_id][0].paper_id == "recent_paper"
    
    def test_reset_all_exposures(self):
        """Test resetting all exposure history."""
        # Add some exposures
        self.model.add_exposure("reviewer1", "paper1", "Title", "Abstract", ["AI"], "read")
        self.model.add_exposure("reviewer2", "paper2", "Title", "Abstract", ["AI"], "read")
        
        assert len(self.model.reviewer_exposures) == 2
        
        # Reset
        self.model.reset_all_exposures()
        assert len(self.model.reviewer_exposures) == 0


if __name__ == "__main__":
    pytest.main([__file__])