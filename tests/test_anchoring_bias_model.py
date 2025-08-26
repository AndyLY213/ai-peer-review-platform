"""
Unit tests for the Anchoring Bias Model.

Tests the AnchoringBiasModel class and its ability to model sequential review bias
where later reviewers are influenced by earlier review scores.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.enhancements.anchoring_bias_model import (
    AnchoringBiasModel, ReviewAnchor
)
from src.enhancements.bias_engine import BiasConfiguration, BiasType
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, BiasEffect,
    ResearcherLevel, ReviewDecision, EnhancedReviewCriteria
)


class TestReviewAnchor:
    """Test cases for ReviewAnchor dataclass."""
    
    def test_review_anchor_creation(self):
        """Test creating a review anchor."""
        anchor = ReviewAnchor(
            review_id="test_review",
            reviewer_id="test_reviewer",
            overall_score=7.5,
            confidence_level=4,
            submission_time=datetime.now()
        )
        
        assert anchor.review_id == "test_review"
        assert anchor.reviewer_id == "test_reviewer"
        assert anchor.overall_score == 7.5
        assert anchor.confidence_level == 4
        assert anchor.anchor_strength > 0  # Should be calculated automatically
    
    def test_anchor_strength_calculation(self):
        """Test anchor strength calculation based on confidence."""
        # High confidence anchor
        high_confidence = ReviewAnchor(
            review_id="high_conf",
            reviewer_id="reviewer1",
            overall_score=8.0,
            confidence_level=5,
            submission_time=datetime.now()
        )
        
        # Low confidence anchor
        low_confidence = ReviewAnchor(
            review_id="low_conf",
            reviewer_id="reviewer2",
            overall_score=8.0,
            confidence_level=1,
            submission_time=datetime.now()
        )
        
        assert high_confidence.anchor_strength > low_confidence.anchor_strength
        assert 0.0 <= low_confidence.anchor_strength <= 1.0
        assert 0.0 <= high_confidence.anchor_strength <= 1.0


class TestAnchoringBiasModel:
    """Test cases for AnchoringBiasModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BiasConfiguration(
            bias_type=BiasType.ANCHORING,
            base_strength=0.6,
            parameters={
                'influence_decay': 0.8,
                'confidence_weight': 0.3,
                'max_influence': 1.0
            }
        )
        self.model = AnchoringBiasModel(self.config)
        
        self.reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI",
            cognitive_biases={"anchoring": 0.7}
        )
        
        self.review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id="test_paper",
            venue_id="test_venue"
        )
    
    def test_model_initialization(self):
        """Test anchoring bias model initialization."""
        assert self.model.bias_type == BiasType.ANCHORING
        assert self.model.influence_decay == 0.8
        assert self.model.confidence_weight == 0.3
        assert self.model.max_influence == 1.0
        assert len(self.model.review_history) == 0  # Should start empty
    
    def test_is_applicable_with_previous_reviews(self):
        """Test is_applicable with previous reviews available."""
        previous_reviews = [
            {
                'review_id': 'prev_review_1',
                'reviewer_id': 'prev_reviewer_1',
                'overall_score': 7.0,
                'confidence_level': 4,
                'submission_time': datetime.now().isoformat()
            }
        ]
        
        context = {'previous_reviews': previous_reviews}
        
        assert self.model.is_applicable(self.reviewer, context) is True
    
    def test_is_applicable_without_previous_reviews(self):
        """Test is_applicable without previous reviews."""
        context = {}
        
        assert self.model.is_applicable(self.reviewer, context) is False
    
    def test_is_applicable_no_susceptibility(self):
        """Test is_applicable with reviewer having no anchoring susceptibility."""
        reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI",
            cognitive_biases={"anchoring": 0.0}  # No susceptibility
        )
        
        context = {'previous_reviews': [{'overall_score': 7.0}]}
        
        assert self.model.is_applicable(reviewer, context) is False
    
    def test_convert_to_anchors_structured_review(self):
        """Test converting StructuredReview objects to anchors."""
        review = StructuredReview(
            reviewer_id="prev_reviewer",
            paper_id="test_paper",
            venue_id="test_venue",
            confidence_level=4
        )
        review.criteria_scores = EnhancedReviewCriteria(
            novelty=7.0, technical_quality=8.0, clarity=6.0,
            significance=7.5, reproducibility=6.5, related_work=7.0
        )
        
        anchors = self.model._convert_to_anchors([review])
        
        assert len(anchors) == 1
        assert anchors[0].reviewer_id == "prev_reviewer"
        assert anchors[0].confidence_level == 4
        assert abs(anchors[0].overall_score - 7.0) < 0.1  # Average of criteria scores
    
    def test_convert_to_anchors_dict_format(self):
        """Test converting dictionary format reviews to anchors."""
        review_dict = {
            'review_id': 'test_review',
            'reviewer_id': 'test_reviewer',
            'overall_score': 6.5,
            'confidence_level': 3,
            'submission_time': datetime.now().isoformat()
        }
        
        anchors = self.model._convert_to_anchors([review_dict])
        
        assert len(anchors) == 1
        assert anchors[0].review_id == 'test_review'
        assert anchors[0].reviewer_id == 'test_reviewer'
        assert anchors[0].overall_score == 6.5
        assert anchors[0].confidence_level == 3
    
    def test_convert_to_anchors_with_criteria_scores(self):
        """Test converting dict with criteria scores to anchors."""
        review_dict = {
            'review_id': 'test_review',
            'reviewer_id': 'test_reviewer',
            'criteria_scores': {
                'novelty': 8.0,
                'technical_quality': 7.0,
                'clarity': 6.0,
                'significance': 8.5,
                'reproducibility': 7.5,
                'related_work': 7.0
            },
            'confidence_level': 4,
            'submission_time': datetime.now().isoformat()
        }
        
        anchors = self.model._convert_to_anchors([review_dict])
        
        assert len(anchors) == 1
        # Should calculate average from criteria scores
        expected_avg = (8.0 + 7.0 + 6.0 + 8.5 + 7.5 + 7.0) / 6
        assert abs(anchors[0].overall_score - expected_avg) < 0.01
    
    def test_calculate_anchor_score_single_anchor(self):
        """Test calculating anchor score with single previous review."""
        anchor = ReviewAnchor(
            review_id="test",
            reviewer_id="reviewer",
            overall_score=7.5,
            confidence_level=4,
            submission_time=datetime.now()
        )
        
        anchor_score = self.model._calculate_anchor_score([anchor])
        
        assert anchor_score == 7.5  # Should equal the single anchor score
    
    def test_calculate_anchor_score_multiple_anchors(self):
        """Test calculating anchor score with multiple previous reviews."""
        now = datetime.now()
        
        anchors = [
            ReviewAnchor("r1", "rev1", 8.0, 5, now - timedelta(minutes=10)),  # Most recent, high confidence
            ReviewAnchor("r2", "rev2", 6.0, 3, now - timedelta(minutes=20)),  # Older, lower confidence
            ReviewAnchor("r3", "rev3", 7.0, 4, now - timedelta(minutes=30))   # Oldest
        ]
        
        anchor_score = self.model._calculate_anchor_score(anchors)
        
        # Should be weighted toward more recent, higher confidence reviews
        assert 6.0 < anchor_score < 8.0  # Should be between the extremes
        assert anchor_score > 7.0  # Should be closer to the recent high-confidence review
    
    def test_calculate_anchor_score_empty_list(self):
        """Test calculating anchor score with empty anchor list."""
        anchor_score = self.model._calculate_anchor_score([])
        
        assert anchor_score == 5.0  # Should return default neutral score
    
    def test_calculate_score_adjustment_toward_higher_anchor(self):
        """Test score adjustment when anchor is higher than intended score."""
        intended_score = 6.0
        anchor_score = 8.0
        effective_strength = 0.5
        anchors = [ReviewAnchor("r1", "rev1", 8.0, 4, datetime.now())]
        
        adjustment = self.model._calculate_score_adjustment(
            intended_score, anchor_score, effective_strength, anchors
        )
        
        assert adjustment > 0  # Should adjust upward toward anchor
        assert adjustment <= self.model.max_influence
    
    def test_calculate_score_adjustment_toward_lower_anchor(self):
        """Test score adjustment when anchor is lower than intended score."""
        intended_score = 8.0
        anchor_score = 5.0
        effective_strength = 0.6
        anchors = [ReviewAnchor("r1", "rev1", 5.0, 3, datetime.now())]
        
        adjustment = self.model._calculate_score_adjustment(
            intended_score, anchor_score, effective_strength, anchors
        )
        
        assert adjustment < 0  # Should adjust downward toward anchor
        assert adjustment >= -self.model.max_influence
    
    def test_calculate_score_adjustment_max_influence_limit(self):
        """Test that score adjustment respects maximum influence limit."""
        intended_score = 2.0
        anchor_score = 9.0  # Very large difference
        effective_strength = 1.0  # Maximum strength
        anchors = [ReviewAnchor("r1", "rev1", 9.0, 5, datetime.now())]
        
        adjustment = self.model._calculate_score_adjustment(
            intended_score, anchor_score, effective_strength, anchors
        )
        
        assert adjustment <= self.model.max_influence
        assert adjustment >= -self.model.max_influence
    
    def test_calculate_anchor_consensus_high_consensus(self):
        """Test consensus calculation with similar anchor scores."""
        anchors = [
            ReviewAnchor("r1", "rev1", 7.0, 4, datetime.now()),
            ReviewAnchor("r2", "rev2", 7.2, 4, datetime.now()),
            ReviewAnchor("r3", "rev3", 6.8, 4, datetime.now())
        ]
        
        consensus = self.model._calculate_anchor_consensus(anchors)
        
        assert consensus > 0.8  # High consensus due to similar scores
    
    def test_calculate_anchor_consensus_low_consensus(self):
        """Test consensus calculation with diverse anchor scores."""
        anchors = [
            ReviewAnchor("r1", "rev1", 3.0, 4, datetime.now()),
            ReviewAnchor("r2", "rev2", 7.0, 4, datetime.now()),
            ReviewAnchor("r3", "rev3", 9.0, 4, datetime.now())
        ]
        
        consensus = self.model._calculate_anchor_consensus(anchors)
        
        assert consensus < 0.5  # Low consensus due to diverse scores
    
    def test_calculate_anchor_consensus_single_anchor(self):
        """Test consensus calculation with single anchor."""
        anchors = [ReviewAnchor("r1", "rev1", 7.0, 4, datetime.now())]
        
        consensus = self.model._calculate_anchor_consensus(anchors)
        
        assert consensus == 1.0  # Perfect consensus with single anchor
    
    def test_calculate_bias_effect_with_anchors(self):
        """Test calculating bias effect with previous reviews."""
        previous_reviews = [
            {
                'review_id': 'prev_review_1',
                'reviewer_id': 'prev_reviewer_1',
                'overall_score': 8.0,
                'confidence_level': 4,
                'submission_time': datetime.now().isoformat()
            }
        ]
        
        context = {
            'previous_reviews': previous_reviews,
            'intended_score': 6.0  # Lower than anchor
        }
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "anchoring"
        assert bias_effect.strength > 0
        assert bias_effect.score_adjustment > 0  # Should adjust upward toward anchor
        assert "anchoring bias" in bias_effect.description.lower()
    
    def test_calculate_bias_effect_no_previous_reviews(self):
        """Test calculating bias effect without previous reviews."""
        context = {}
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "anchoring"
        assert bias_effect.strength == 0.0
        assert bias_effect.score_adjustment == 0.0
        assert "no previous reviews" in bias_effect.description.lower()
    
    def test_generate_bias_description(self):
        """Test bias description generation."""
        # Positive adjustment toward high anchor
        desc = self.model._generate_bias_description(8.0, 0.5, 2)
        assert "toward" in desc.lower()
        assert "high anchor" in desc.lower()
        assert "2 previous" in desc
        assert "+0.50" in desc
        
        # Negative adjustment toward low anchor
        desc = self.model._generate_bias_description(3.0, -0.3, 1)
        assert "away from" in desc.lower()
        assert "low anchor" in desc.lower()
        assert "1 previous" in desc
        assert "-0.30" in desc
        
        # Minimal effect
        desc = self.model._generate_bias_description(6.0, 0.02, 3)
        assert "minimal" in desc.lower()
    
    def test_add_review_to_history(self):
        """Test adding a review to the anchoring history."""
        paper_id = "test_paper"
        review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id=paper_id,
            venue_id="test_venue",
            confidence_level=4
        )
        
        initial_count = len(self.model.review_history.get(paper_id, []))
        
        self.model.add_review_to_history(paper_id, review)
        
        assert paper_id in self.model.review_history
        assert len(self.model.review_history[paper_id]) == initial_count + 1
        
        added_anchor = self.model.review_history[paper_id][-1]
        assert added_anchor.reviewer_id == "test_reviewer"
        assert added_anchor.confidence_level == 4
    
    def test_add_review_to_history_limit(self):
        """Test that review history is limited to prevent unbounded growth."""
        paper_id = "test_paper"
        
        # Add more than the limit (10) reviews
        for i in range(15):
            review = StructuredReview(
                reviewer_id=f"reviewer_{i}",
                paper_id=paper_id,
                venue_id="test_venue"
            )
            self.model.add_review_to_history(paper_id, review)
        
        # Should be limited to 10 reviews
        assert len(self.model.review_history[paper_id]) == 10
        
        # Should keep the most recent reviews
        last_anchor = self.model.review_history[paper_id][-1]
        assert last_anchor.reviewer_id == "reviewer_14"  # Most recent
    
    def test_get_paper_review_history(self):
        """Test getting review history for a specific paper."""
        paper_id = "test_paper"
        review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id=paper_id,
            venue_id="test_venue"
        )
        
        # Initially empty
        history = self.model.get_paper_review_history(paper_id)
        assert len(history) == 0
        
        # Add review and check
        self.model.add_review_to_history(paper_id, review)
        history = self.model.get_paper_review_history(paper_id)
        assert len(history) == 1
        assert history[0].reviewer_id == "test_reviewer"
    
    def test_clear_paper_history(self):
        """Test clearing review history for a specific paper."""
        paper_id = "test_paper"
        review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id=paper_id,
            venue_id="test_venue"
        )
        
        # Add review
        self.model.add_review_to_history(paper_id, review)
        assert len(self.model.review_history[paper_id]) == 1
        
        # Clear history
        self.model.clear_paper_history(paper_id)
        assert paper_id not in self.model.review_history
    
    def test_get_anchoring_statistics_empty(self):
        """Test getting statistics with empty history."""
        stats = self.model.get_anchoring_statistics()
        
        assert stats["total_papers_with_history"] == 0
        assert stats["total_reviews_in_history"] == 0
        assert stats["average_reviews_per_paper"] == 0.0
        assert stats["average_anchor_strength"] == 0.0
    
    def test_get_anchoring_statistics_with_data(self):
        """Test getting statistics with review history."""
        # Add some reviews
        for paper_id in ["paper1", "paper2"]:
            for i in range(3):
                review = StructuredReview(
                    reviewer_id=f"reviewer_{i}",
                    paper_id=paper_id,
                    venue_id="test_venue",
                    confidence_level=4
                )
                self.model.add_review_to_history(paper_id, review)
        
        stats = self.model.get_anchoring_statistics()
        
        assert stats["total_papers_with_history"] == 2
        assert stats["total_reviews_in_history"] == 6
        assert stats["average_reviews_per_paper"] == 3.0
        assert stats["average_anchor_strength"] > 0
        assert stats["papers_with_multiple_reviews"] == 2
    
    def test_simulate_review_order_effect(self):
        """Test simulating the effect of review order."""
        base_scores = [7.0, 6.0, 8.0, 5.0]
        susceptibilities = [0.0, 0.5, 0.7, 0.3]  # First reviewer has no bias
        
        adjusted_scores = self.model.simulate_review_order_effect(
            base_scores, susceptibilities
        )
        
        assert len(adjusted_scores) == 4
        assert adjusted_scores[0] == 7.0  # First reviewer unchanged
        
        # Later reviewers should be influenced by earlier ones
        # The exact values depend on the anchoring algorithm
        for score in adjusted_scores:
            assert 1.0 <= score <= 10.0  # Should be within reasonable range
    
    def test_simulate_review_order_effect_mismatched_lengths(self):
        """Test simulation with mismatched input lengths."""
        base_scores = [7.0, 6.0]
        susceptibilities = [0.5, 0.7, 0.3]  # Different length
        
        with pytest.raises(ValueError):
            self.model.simulate_review_order_effect(base_scores, susceptibilities)
    
    def test_reset_history(self):
        """Test resetting all review history."""
        # Add some reviews
        paper_id = "test_paper"
        review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id=paper_id,
            venue_id="test_venue"
        )
        self.model.add_review_to_history(paper_id, review)
        
        assert len(self.model.review_history) > 0
        
        # Reset
        self.model.reset_history()
        assert len(self.model.review_history) == 0


if __name__ == "__main__":
    pytest.main([__file__])