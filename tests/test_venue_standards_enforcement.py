"""
Unit Tests for Venue-Specific Review Standards Enforcement

Tests for ReviewRequirements class, QualityStandards validator, and venue-specific
review length enforcement logic.
"""

import pytest
from datetime import datetime
from typing import List

from src.enhancements.venue_standards_enforcement import (
    ReviewRequirementsManager, QualityStandardsValidator, VenueStandardsEnforcer,
    VenueStandardsConfig
)
from src.data.enhanced_models import (
    EnhancedVenue, StructuredReview, VenueType, ReviewDecision,
    EnhancedReviewCriteria, DetailedStrength, DetailedWeakness,
    ReviewRequirements, QualityStandards
)


def generate_text(word_count: int) -> str:
    """Generate text with specified word count."""
    base_words = ["This", "is", "a", "detailed", "review", "comment", "that", "provides", "comprehensive", "analysis"]
    words = []
    while len(words) < word_count:
        words.extend(base_words)
    return " ".join(words[:word_count])


def create_test_review(word_count: int = 500, strengths_count: int = 2, 
                      weaknesses_count: int = 1, confidence: int = 3,
                      has_questions: bool = True, has_suggestions: bool = False,
                      venue_id: str = "test_venue") -> StructuredReview:
    """Create a test review with specified parameters."""
    
    # Distribute words across sections
    executive_words = max(50, word_count // 6)
    technical_words = max(100, word_count // 2)
    presentation_words = max(30, word_count // 8)
    strength_words = max(20, word_count // (8 * max(1, strengths_count)))
    weakness_words = max(20, word_count // (8 * max(1, weaknesses_count)))
    
    strengths = [
        DetailedStrength(
            category="Technical",
            description=f"Strength {i+1}: {generate_text(strength_words)}",
            importance=4
        ) for i in range(strengths_count)
    ]
    
    weaknesses = [
        DetailedWeakness(
            category="Technical", 
            description=f"Weakness {i+1}: {generate_text(weakness_words)}",
            severity=3
        ) for i in range(weaknesses_count)
    ]
    
    questions = ["What about X?", "How does Y work?"] if has_questions else []
    suggestions = ["Improve Z", "Consider W"] if has_suggestions else []
    
    return StructuredReview(
        review_id="test_review_001",
        reviewer_id="reviewer_001",
        paper_id="paper_001",
        venue_id=venue_id,
        criteria_scores=EnhancedReviewCriteria(
            novelty=7.0, technical_quality=8.0, clarity=6.0,
            significance=7.5, reproducibility=6.5, related_work=7.0
        ),
        confidence_level=confidence,
        recommendation=ReviewDecision.MINOR_REVISION,
        executive_summary=generate_text(executive_words),
        detailed_strengths=strengths,
        detailed_weaknesses=weaknesses,
        technical_comments=generate_text(technical_words),
        presentation_comments=generate_text(presentation_words),
        questions_for_authors=questions,
        suggestions_for_improvement=suggestions,
        submission_timestamp=datetime.now()
    )


class TestReviewRequirementsManager:
    """Test ReviewRequirements class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ReviewRequirementsManager()
    
    def test_initialization(self):
        """Test manager initializes with correct venue standards."""
        assert len(self.manager.venue_standards) == 6
        assert VenueType.TOP_CONFERENCE in self.manager.venue_standards
        assert VenueType.MID_CONFERENCE in self.manager.venue_standards
        assert VenueType.LOW_CONFERENCE in self.manager.venue_standards
        assert VenueType.TOP_JOURNAL in self.manager.venue_standards
        assert VenueType.SPECIALIZED_JOURNAL in self.manager.venue_standards
        assert VenueType.GENERAL_JOURNAL in self.manager.venue_standards
    
    def test_create_review_requirements_top_conference(self):
        """Test creating requirements for top conference."""
        requirements = self.manager.create_review_requirements(VenueType.TOP_CONFERENCE)
        
        assert requirements.min_word_count == 600
        assert requirements.max_word_count == 1000
        assert requirements.min_strengths == 3
        assert requirements.min_weaknesses == 2
        assert requirements.requires_questions == True
        assert requirements.requires_suggestions == True
    
    def test_create_review_requirements_low_conference(self):
        """Test creating requirements for low conference."""
        requirements = self.manager.create_review_requirements(VenueType.LOW_CONFERENCE)
        
        assert requirements.min_word_count == 300
        assert requirements.max_word_count == 600
        assert requirements.min_strengths == 2
        assert requirements.min_weaknesses == 1
        assert requirements.requires_questions == False
        assert requirements.requires_suggestions == False
    
    def test_get_venue_standards(self):
        """Test getting venue standards configuration."""
        standards = self.manager.get_venue_standards(VenueType.TOP_CONFERENCE)
        
        assert isinstance(standards, VenueStandardsConfig)
        assert standards.venue_type == VenueType.TOP_CONFERENCE
        assert standards.min_word_count == 600
        assert standards.max_word_count == 1000
        assert standards.min_confidence_level == 4
        assert standards.acceptance_threshold == 8.5


class TestQualityStandardsValidator:
    """Test QualityStandards validator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = QualityStandardsValidator()
        self.venue = EnhancedVenue(
            id="test_venue",
            name="Test Mid Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="Computer Science"
        )
    
    def test_validate_word_count_valid(self):
        """Test word count validation with valid review."""
        review = create_test_review(word_count=500)
        is_valid, errors = self.validator._validate_word_count(
            review, self.validator.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        )
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_word_count_too_short(self):
        """Test word count validation with review too short."""
        review = create_test_review(word_count=200)  # Below 400 minimum for mid conference
        is_valid, errors = self.validator._validate_word_count(
            review, self.validator.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        )
        
        assert is_valid == False
        assert len(errors) == 1
        assert "below minimum" in errors[0]
    
    def test_validate_word_count_too_long(self):
        """Test word count validation with review too long."""
        review = create_test_review(word_count=1000)  # Above 800 maximum for mid conference
        is_valid, errors = self.validator._validate_word_count(
            review, self.validator.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        )
        
        assert is_valid == False
        assert len(errors) == 1
        assert "exceeds maximum" in errors[0]
    
    def test_validate_structure_requirements_valid(self):
        """Test structure validation with valid review."""
        review = create_test_review(strengths_count=2, weaknesses_count=1)
        is_valid, errors = self.validator._validate_structure_requirements(
            review, self.validator.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        )
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_structure_requirements_insufficient_strengths(self):
        """Test structure validation with insufficient strengths."""
        review = create_test_review(strengths_count=1, weaknesses_count=1)  # Need 2 strengths
        is_valid, errors = self.validator._validate_structure_requirements(
            review, self.validator.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        )
        
        assert is_valid == False
        assert len(errors) == 1
        assert "detailed strengths" in errors[0]
    
    def test_validate_confidence_level_valid(self):
        """Test confidence level validation with valid level."""
        review = create_test_review(confidence=3)  # Meets minimum for mid conference
        is_valid, errors = self.validator._validate_confidence_level(
            review, self.validator.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        )
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_confidence_level_too_low(self):
        """Test confidence level validation with level too low."""
        review = create_test_review(confidence=2)  # Below minimum 3 for mid conference
        is_valid, errors = self.validator._validate_confidence_level(
            review, self.validator.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        )
        
        assert is_valid == False
        assert len(errors) == 1
        assert "below minimum" in errors[0]
    
    def test_validate_complete_review_valid(self):
        """Test complete review validation with valid review."""
        review = create_test_review(
            word_count=500, strengths_count=2, weaknesses_count=1, 
            confidence=3, has_questions=True
        )
        is_valid, errors = self.validator.validate_review_against_venue_standards(review, self.venue)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_enforce_minimum_review_length_valid(self):
        """Test minimum review length enforcement with valid length."""
        review = create_test_review(word_count=500)
        is_valid, message = self.validator.enforce_minimum_review_length(review, VenueType.MID_CONFERENCE)
        
        assert is_valid == True
        assert "meets minimum length" in message


class TestVenueStandardsEnforcer:
    """Test VenueStandardsEnforcer main functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enforcer = VenueStandardsEnforcer()
        self.venue = EnhancedVenue(
            id="test_venue",
            name="Test Mid Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="Computer Science"
        )
    
    def test_enforce_venue_standards_valid(self):
        """Test venue standards enforcement with valid review."""
        review = create_test_review(word_count=500, venue_id=self.venue.id)
        is_valid, errors, details = self.enforcer.enforce_venue_standards(review, self.venue)
        
        assert is_valid == True
        assert len(errors) == 0
        assert details['venue_id'] == self.venue.id
        assert details['venue_type'] == 'Mid Conference'
        assert details['validation_results']['is_valid'] == True
        assert 'standards_applied' in details
    
    def test_enforce_venue_standards_invalid(self):
        """Test venue standards enforcement with invalid review."""
        review = create_test_review(word_count=200, strengths_count=1, confidence=2, venue_id=self.venue.id)
        is_valid, errors, details = self.enforcer.enforce_venue_standards(review, self.venue)
        
        assert is_valid == False
        assert len(errors) > 0
        assert details['validation_results']['is_valid'] == False
        assert details['validation_results']['error_count'] > 0
    
    def test_get_venue_requirements_summary(self):
        """Test getting venue requirements summary."""
        summary = self.enforcer.get_venue_requirements_summary(VenueType.TOP_CONFERENCE)
        
        assert summary['venue_type'] == 'Top Conference'
        assert summary['word_count_range'] == '600-1000 words'
        assert summary['structural_requirements']['min_strengths'] == 3
        assert summary['structural_requirements']['min_weaknesses'] == 2
        assert summary['structural_requirements']['requires_questions'] == True
        assert summary['quality_requirements']['min_confidence_level'] == 4
        assert summary['quality_requirements']['acceptance_threshold'] == 8.5
    
    def test_compare_venue_standards(self):
        """Test comparing standards across venue types."""
        venue_types = [VenueType.TOP_CONFERENCE, VenueType.MID_CONFERENCE, VenueType.LOW_CONFERENCE]
        comparison = self.enforcer.compare_venue_standards(venue_types)
        
        assert len(comparison['venue_types']) == 3
        assert 'comparison_matrix' in comparison
        assert 'strictness_ranking' in comparison
        
        # Check that top conference is strictest
        assert comparison['strictness_ranking'][0] == 'Top Conference'


class TestVenueStandardsIntegration:
    """Integration tests for venue standards enforcement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enforcer = VenueStandardsEnforcer()
    
    def test_all_venue_types_have_standards(self):
        """Test that all venue types have defined standards."""
        for venue_type in VenueType:
            if venue_type in [VenueType.WORKSHOP, VenueType.PREPRINT]:
                continue  # Skip venue types not in standards
            
            standards = self.enforcer.requirements_manager.get_venue_standards(venue_type)
            assert standards is not None
            assert standards.venue_type == venue_type
            assert standards.min_word_count > 0
            assert standards.max_word_count > standards.min_word_count
            assert standards.acceptance_threshold > 0
    
    def test_venue_standards_progression(self):
        """Test that venue standards follow expected progression."""
        # Get standards for different venue types
        top_conf = self.enforcer.requirements_manager.get_venue_standards(VenueType.TOP_CONFERENCE)
        mid_conf = self.enforcer.requirements_manager.get_venue_standards(VenueType.MID_CONFERENCE)
        low_conf = self.enforcer.requirements_manager.get_venue_standards(VenueType.LOW_CONFERENCE)
        
        # Top conference should be strictest
        assert top_conf.min_word_count >= mid_conf.min_word_count >= low_conf.min_word_count
        assert top_conf.min_strengths >= mid_conf.min_strengths >= low_conf.min_strengths
        assert top_conf.min_confidence_level >= mid_conf.min_confidence_level >= low_conf.min_confidence_level
        assert top_conf.acceptance_threshold >= mid_conf.acceptance_threshold >= low_conf.acceptance_threshold


if __name__ == "__main__":
    pytest.main([__file__])