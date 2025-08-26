"""
Unit tests for the Confirmation Bias Model.

Tests the ConfirmationBiasModel class and its ability to model belief-based bias
in peer review based on alignment between paper content and reviewer beliefs.
"""

import pytest
from unittest.mock import Mock, patch

from src.enhancements.confirmation_bias_model import (
    ConfirmationBiasModel, ResearchBelief
)
from src.enhancements.bias_engine import BiasConfiguration, BiasType
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, BiasEffect,
    ResearcherLevel, ReviewDecision
)


class TestResearchBelief:
    """Test cases for ResearchBelief dataclass."""
    
    def test_research_belief_creation(self):
        """Test creating a research belief."""
        belief = ResearchBelief(
            topic="interpretability",
            position="interpretability_crucial",
            strength=0.8,
            keywords=["explainable AI", "interpretability", "transparency"]
        )
        
        assert belief.topic == "interpretability"
        assert belief.position == "interpretability_crucial"
        assert belief.strength == 0.8
        assert len(belief.keywords) == 3
        assert "explainable AI" in belief.keywords


class TestConfirmationBiasModel:
    """Test cases for ConfirmationBiasModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BiasConfiguration(
            bias_type=BiasType.CONFIRMATION,
            base_strength=0.5,
            parameters={
                'belief_alignment_threshold': 0.7,
                'max_score_adjustment': 1.5
            }
        )
        self.model = ConfirmationBiasModel(self.config)
        
        self.reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI",
            cognitive_biases={"confirmation": 0.6}
        )
        
        self.review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id="test_paper",
            venue_id="test_venue"
        )
    
    def test_model_initialization(self):
        """Test confirmation bias model initialization."""
        assert self.model.bias_type == BiasType.CONFIRMATION
        assert self.model.belief_alignment_threshold == 0.7
        assert self.model.max_score_adjustment == 1.5
        assert len(self.model.research_beliefs) > 0
        assert "AI" in self.model.research_beliefs
    
    def test_research_beliefs_database(self):
        """Test that research beliefs database is properly initialized."""
        ai_beliefs = self.model.research_beliefs["AI"]
        assert len(ai_beliefs) > 0
        
        # Check that beliefs have required fields
        for belief in ai_beliefs:
            assert hasattr(belief, 'topic')
            assert hasattr(belief, 'position')
            assert hasattr(belief, 'strength')
            assert hasattr(belief, 'keywords')
            assert 0.0 <= belief.strength <= 1.0
            assert len(belief.keywords) > 0
    
    def test_is_applicable_with_valid_context(self):
        """Test is_applicable with valid context."""
        context = {
            'paper_content': {
                'title': 'Deep Learning for Computer Vision',
                'abstract': 'This paper presents a novel neural network architecture...',
                'keywords': ['deep learning', 'neural networks', 'computer vision']
            }
        }
        
        assert self.model.is_applicable(self.reviewer, context) is True
    
    def test_is_applicable_without_paper_content(self):
        """Test is_applicable without paper content."""
        context = {}
        
        assert self.model.is_applicable(self.reviewer, context) is False
    
    def test_is_applicable_unknown_specialty(self):
        """Test is_applicable with unknown specialty."""
        reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="Unknown Field"
        )
        
        context = {
            'paper_content': {
                'title': 'Test Paper',
                'abstract': 'Test abstract'
            }
        }
        
        assert self.model.is_applicable(reviewer, context) is False
    
    def test_calculate_belief_alignment_positive(self):
        """Test belief alignment calculation with supportive paper."""
        paper_title = "Explainable AI for Medical Diagnosis"
        paper_abstract = "This paper focuses on interpretability and transparency in AI systems"
        paper_keywords = ["explainable AI", "interpretability", "transparency"]
        
        alignment = self.model._calculate_belief_alignment(
            self.reviewer, paper_title, paper_abstract, paper_keywords
        )
        
        # Should be positive since paper supports interpretability beliefs
        assert alignment > 0
    
    def test_calculate_belief_alignment_negative(self):
        """Test belief alignment calculation with contradictory paper."""
        paper_title = "Black Box Deep Learning Achieves State-of-the-Art Performance"
        paper_abstract = "We show that performance is more important than interpretability"
        paper_keywords = ["performance", "accuracy", "state-of-the-art", "black box"]
        
        alignment = self.model._calculate_belief_alignment(
            self.reviewer, paper_title, paper_abstract, paper_keywords
        )
        
        # Should be negative since paper contradicts interpretability beliefs
        assert alignment < 0
    
    def test_calculate_belief_alignment_neutral(self):
        """Test belief alignment calculation with neutral paper."""
        paper_title = "A Survey of Machine Learning Techniques"
        paper_abstract = "This paper provides a comprehensive survey of various ML methods"
        paper_keywords = ["machine learning", "survey", "methods"]
        
        alignment = self.model._calculate_belief_alignment(
            self.reviewer, paper_title, paper_abstract, paper_keywords
        )
        
        # Should be close to neutral
        assert abs(alignment) < 0.3
    
    def test_determine_alignment_direction_supportive(self):
        """Test alignment direction determination for supportive content."""
        belief = ResearchBelief(
            topic="interpretability",
            position="interpretability_crucial",
            strength=0.8,
            keywords=["explainable AI", "interpretability", "transparency"]
        )
        
        paper_text = "explainable ai and interpretability are crucial for trust"
        
        direction = self.model._determine_alignment_direction(belief, paper_text)
        assert direction == 1.0  # Supportive
    
    def test_determine_alignment_direction_contradictory(self):
        """Test alignment direction determination for contradictory content."""
        belief = ResearchBelief(
            topic="interpretability",
            position="interpretability_crucial",
            strength=0.8,
            keywords=["explainable AI", "interpretability", "transparency"]
        )
        
        paper_text = "performance and accuracy are more important than interpretability"
        
        direction = self.model._determine_alignment_direction(belief, paper_text)
        assert direction == -1.0  # Contradictory
    
    def test_calculate_score_adjustment_positive_alignment(self):
        """Test score adjustment calculation for positive alignment."""
        alignment_score = 0.8  # Strong positive alignment
        effective_strength = 0.6
        
        adjustment = self.model._calculate_score_adjustment(alignment_score, effective_strength)
        
        assert adjustment > 0  # Should boost score
        assert adjustment <= self.model.max_score_adjustment
    
    def test_calculate_score_adjustment_negative_alignment(self):
        """Test score adjustment calculation for negative alignment."""
        alignment_score = -0.7  # Strong negative alignment
        effective_strength = 0.5
        
        adjustment = self.model._calculate_score_adjustment(alignment_score, effective_strength)
        
        assert adjustment < 0  # Should reduce score
        assert adjustment >= -self.model.max_score_adjustment
    
    def test_calculate_score_adjustment_neutral_alignment(self):
        """Test score adjustment calculation for neutral alignment."""
        alignment_score = 0.0  # Neutral alignment
        effective_strength = 0.5
        
        adjustment = self.model._calculate_score_adjustment(alignment_score, effective_strength)
        
        assert abs(adjustment) < 0.1  # Should be minimal adjustment
    
    def test_calculate_bias_effect_positive(self):
        """Test calculating bias effect with positive alignment."""
        context = {
            'paper_content': {
                'title': 'Explainable AI for Healthcare',
                'abstract': 'This paper focuses on interpretability and transparency',
                'keywords': ['explainable AI', 'interpretability', 'healthcare']
            }
        }
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "confirmation"
        assert bias_effect.strength > 0
        assert bias_effect.score_adjustment > 0  # Positive adjustment
        assert "positive" in bias_effect.description.lower()
    
    def test_calculate_bias_effect_negative(self):
        """Test calculating bias effect with negative alignment."""
        context = {
            'paper_content': {
                'title': 'Black Box Deep Learning Performance',
                'abstract': 'Performance is more important than interpretability',
                'keywords': ['performance', 'accuracy', 'black box']
            }
        }
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "confirmation"
        assert bias_effect.strength > 0
        assert bias_effect.score_adjustment < 0  # Negative adjustment
        assert "negative" in bias_effect.description.lower()
    
    def test_calculate_bias_effect_error_handling(self):
        """Test bias effect calculation with invalid context."""
        context = {}  # Missing paper content
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "confirmation"
        assert bias_effect.strength == 0.0
        assert bias_effect.score_adjustment == 0.0
        assert "no paper content" in bias_effect.description.lower()
    
    def test_generate_bias_description(self):
        """Test bias description generation."""
        # Strong positive bias
        desc = self.model._generate_bias_description(0.8, 1.2)
        assert "strong positive" in desc.lower()
        assert "+1.20" in desc
        
        # Strong negative bias
        desc = self.model._generate_bias_description(-0.7, -1.0)
        assert "strong negative" in desc.lower()
        assert "-1.00" in desc
        
        # Minimal bias (below 0.05 threshold)
        desc = self.model._generate_bias_description(0.1, 0.03)
        assert "minimal" in desc.lower()
    
    def test_add_research_belief(self):
        """Test adding a new research belief."""
        initial_count = len(self.model.research_beliefs.get("Test Field", []))
        
        self.model.add_research_belief(
            specialty="Test Field",
            topic="test_topic",
            position="test_position",
            strength=0.7,
            keywords=["test", "keywords"]
        )
        
        beliefs = self.model.research_beliefs["Test Field"]
        assert len(beliefs) == initial_count + 1
        
        new_belief = beliefs[-1]
        assert new_belief.topic == "test_topic"
        assert new_belief.position == "test_position"
        assert new_belief.strength == 0.7
        assert new_belief.keywords == ["test", "keywords"]
    
    def test_get_researcher_beliefs(self):
        """Test getting researcher beliefs by specialty."""
        beliefs = self.model.get_researcher_beliefs(self.reviewer)
        
        assert len(beliefs) > 0
        assert all(isinstance(belief, ResearchBelief) for belief in beliefs)
    
    def test_get_researcher_beliefs_unknown_specialty(self):
        """Test getting beliefs for unknown specialty."""
        reviewer = EnhancedResearcher(
            id="test",
            name="Test",
            specialty="Unknown Field"
        )
        
        beliefs = self.model.get_researcher_beliefs(reviewer)
        assert len(beliefs) == 0
    
    def test_update_belief_strength(self):
        """Test updating belief strength."""
        # First add a belief
        self.model.add_research_belief(
            specialty="Test Field",
            topic="test_topic",
            position="test_position",
            strength=0.5,
            keywords=["test"]
        )
        
        # Update its strength
        self.model.update_belief_strength("Test Field", "test_topic", "test_position", 0.9)
        
        # Verify update
        beliefs = self.model.research_beliefs["Test Field"]
        updated_belief = next(b for b in beliefs if b.topic == "test_topic")
        assert updated_belief.strength == 0.9
    
    def test_update_belief_strength_not_found(self):
        """Test updating strength for non-existent belief."""
        # Should not raise error, just log warning
        self.model.update_belief_strength("Unknown", "unknown", "unknown", 0.5)
        # Test passes if no exception is raised


if __name__ == "__main__":
    pytest.main([__file__])