"""
Unit tests for the venue standards system module.

Tests venue-specific standards and thresholds with PeerRead data,
including acceptance threshold calculation, reviewer selection criteria,
and venue-specific review standards enforcement.
"""

import pytest
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhancements.venue_standards_system import (
    VenueStandardsManager, AcceptanceThreshold, ReviewerSelectionCriteria,
    VenueStandards, get_venue_standards_manager
)
from src.data.enhanced_models import (
    EnhancedVenue, VenueType, EnhancedResearcher, StructuredReview,
    ReviewDecision, ResearcherLevel, EnhancedReviewCriteria
)
from src.core.exceptions import ValidationError


class TestAcceptanceThreshold:
    """Test AcceptanceThreshold dataclass."""
    
    def test_acceptance_threshold_creation(self):
        """Test creating an acceptance threshold."""
        threshold = AcceptanceThreshold(
            base_threshold=3.5,
            dimension_weights={
                "novelty": 0.20,
                "technical_quality": 0.25,
                "clarity": 0.15,
                "significance": 0.25,
                "reproducibility": 0.10,
                "related_work": 0.05
            },
            minimum_score=2.5,
            confidence_requirement=3,
            reviewer_agreement_threshold=0.7
        )
        
        assert threshold.base_threshold == 3.5
        assert threshold.dimension_weights["novelty"] == 0.20
        assert threshold.minimum_score == 2.5
        assert threshold.confidence_requirement == 3
        assert threshold.reviewer_agreement_threshold == 0.7


class TestReviewerSelectionCriteria:
    """Test ReviewerSelectionCriteria dataclass."""
    
    def test_reviewer_selection_criteria_creation(self):
        """Test creating reviewer selection criteria."""
        criteria = ReviewerSelectionCriteria(
            min_h_index=15,
            min_years_experience=5,
            preferred_institution_tiers=[1, 2],
            min_reputation_score=0.6,
            max_reviews_per_reviewer=3,
            field_expertise_required=True,
            conflict_avoidance_years=3
        )
        
        assert criteria.min_h_index == 15
        assert criteria.min_years_experience == 5
        assert criteria.preferred_institution_tiers == [1, 2]
        assert criteria.min_reputation_score == 0.6
        assert criteria.max_reviews_per_reviewer == 3
        assert criteria.field_expertise_required is True
        assert criteria.conflict_avoidance_years == 3


class TestVenueStandards:
    """Test VenueStandards dataclass."""
    
    def test_venue_standards_creation(self):
        """Test creating venue standards."""
        acceptance_threshold = AcceptanceThreshold(
            base_threshold=3.5,
            dimension_weights={"novelty": 0.5, "technical_quality": 0.5},
            minimum_score=2.5,
            confidence_requirement=3,
            reviewer_agreement_threshold=0.7
        )
        
        reviewer_criteria = ReviewerSelectionCriteria(
            min_h_index=15,
            min_years_experience=5,
            preferred_institution_tiers=[1, 2],
            min_reputation_score=0.6,
            max_reviews_per_reviewer=3,
            field_expertise_required=True
        )
        
        standards = VenueStandards(
            venue_id="test-venue",
            venue_name="Test Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            acceptance_threshold=acceptance_threshold,
            reviewer_criteria=reviewer_criteria,
            min_review_length=400,
            max_review_length=800,
            required_technical_depth=3.5,
            novelty_threshold=3.0,
            significance_threshold=3.5,
            min_reviewer_count=3,
            preferred_reviewer_count=3,
            review_deadline_weeks=6,
            late_penalty_per_day=0.1
        )
        
        assert standards.venue_id == "test-venue"
        assert standards.venue_name == "Test Conference"
        assert standards.venue_type == VenueType.TOP_CONFERENCE
        assert standards.min_review_length == 400
        assert standards.max_review_length == 800
        assert standards.required_technical_depth == 3.5


class TestVenueStandardsManager:
    """Test VenueStandardsManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create venue standards manager for testing."""
        return VenueStandardsManager()
    
    @pytest.fixture
    def sample_venue_acl(self):
        """Create sample ACL venue for testing."""
        return EnhancedVenue(
            id="acl-venue",
            name="ACL",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Natural Language Processing"
        )
    
    @pytest.fixture
    def sample_venue_neurips(self):
        """Create sample NeurIPS venue for testing."""
        return EnhancedVenue(
            id="neurips-venue",
            name="NeurIPS",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Machine Learning"
        )
    
    @pytest.fixture
    def sample_venue_conll(self):
        """Create sample CoNLL venue for testing."""
        return EnhancedVenue(
            id="conll-venue",
            name="CoNLL",
            venue_type=VenueType.MID_CONFERENCE,
            field="Natural Language Processing"
        )
    
    @pytest.fixture
    def sample_researcher_qualified(self):
        """Create qualified researcher for testing."""
        return EnhancedResearcher(
            id="qualified-researcher",
            name="Dr. Qualified",
            specialty="Machine Learning",
            level=ResearcherLevel.ASSISTANT_PROF,
            h_index=20,
            years_active=8,
            institution_tier=1,
            reputation_score=0.7
        )
    
    @pytest.fixture
    def sample_researcher_unqualified(self):
        """Create unqualified researcher for testing."""
        return EnhancedResearcher(
            id="unqualified-researcher",
            name="Dr. Junior",
            specialty="Machine Learning",
            level=ResearcherLevel.GRADUATE_STUDENT,
            h_index=3,
            years_active=1,
            institution_tier=3,
            reputation_score=0.2
        )
    
    @pytest.fixture
    def sample_review_good(self):
        """Create good quality review for testing."""
        criteria = EnhancedReviewCriteria(
            novelty=4.0,
            technical_quality=4.5,
            clarity=4.0,
            significance=4.2,
            reproducibility=3.8,
            related_work=3.5
        )
        
        return StructuredReview(
            reviewer_id="reviewer-1",
            paper_id="paper-1",
            venue_id="venue-1",
            criteria_scores=criteria,
            confidence_level=4,
            recommendation=ReviewDecision.ACCEPT,
            executive_summary="This is a good paper with solid contributions.",
            detailed_strengths=["Strong technical approach", "Clear presentation"],
            detailed_weaknesses=["Minor issues with evaluation"],
            technical_comments="The technical approach is sound and well-executed.",
            presentation_comments="The paper is well-written and clear.",
            questions_for_authors=["How does this compare to recent work?"],
            suggestions_for_improvement=["Add more baseline comparisons"]
        )
    
    @pytest.fixture
    def sample_review_poor(self):
        """Create poor quality review for testing."""
        criteria = EnhancedReviewCriteria(
            novelty=2.0,
            technical_quality=2.5,
            clarity=2.0,
            significance=2.2,
            reproducibility=1.8,
            related_work=2.5
        )
        
        return StructuredReview(
            reviewer_id="reviewer-2",
            paper_id="paper-1",
            venue_id="venue-1",
            criteria_scores=criteria,
            confidence_level=2,
            recommendation=ReviewDecision.REJECT,
            executive_summary="This paper has significant issues.",
            detailed_strengths=["Some interesting ideas"],
            detailed_weaknesses=["Poor execution", "Unclear presentation"],
            technical_comments="Technical approach needs work.",
            presentation_comments="Paper is hard to follow.",
            questions_for_authors=["Can you clarify the methodology?"],
            suggestions_for_improvement=["Rewrite for clarity"]
        )
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert len(manager._venue_standards) == 4  # ACL, NeurIPS, ICLR, CoNLL
        assert len(manager._peerread_thresholds) == 4
        
        # Check PeerRead thresholds
        assert manager._peerread_thresholds["ACL"] == 3.5
        assert manager._peerread_thresholds["NeurIPS"] == 4.0
        assert manager._peerread_thresholds["ICLR"] == 3.5
        assert manager._peerread_thresholds["CoNLL"] == 3.0
    
    def test_get_venue_score_thresholds(self, manager):
        """Test getting venue score thresholds."""
        thresholds = manager.get_venue_score_thresholds()
        
        assert thresholds["ACL"] == 3.5
        assert thresholds["NeurIPS"] == 4.0
        assert thresholds["ICLR"] == 3.5
        assert thresholds["CoNLL"] == 3.0
    
    def test_list_supported_venues(self, manager):
        """Test listing supported venues."""
        venues = manager.list_supported_venues()
        
        assert len(venues) == 4
        assert "ACL" in venues
        assert "NeurIPS" in venues
        assert "ICLR" in venues
        assert "CoNLL" in venues
    
    def test_get_venue_standards_acl(self, manager):
        """Test getting ACL venue standards."""
        standards = manager.get_venue_standards("ACL")
        
        assert standards is not None
        assert standards.venue_name == "ACL"
        assert standards.venue_type == VenueType.TOP_CONFERENCE
        assert standards.acceptance_threshold.base_threshold == 3.5
        assert standards.reviewer_criteria.min_h_index == 15
        assert standards.min_review_length == 400
        assert standards.max_review_length == 800
        assert standards.min_reviewer_count == 3
    
    def test_get_venue_standards_neurips(self, manager):
        """Test getting NeurIPS venue standards."""
        standards = manager.get_venue_standards("NeurIPS")
        
        assert standards is not None
        assert standards.venue_name == "NeurIPS"
        assert standards.venue_type == VenueType.TOP_CONFERENCE
        assert standards.acceptance_threshold.base_threshold == 4.0
        assert standards.reviewer_criteria.min_h_index == 20
        assert standards.min_review_length == 500
        assert standards.max_review_length == 1000
        assert standards.preferred_reviewer_count == 4
    
    def test_get_venue_standards_conll(self, manager):
        """Test getting CoNLL venue standards."""
        standards = manager.get_venue_standards("CoNLL")
        
        assert standards is not None
        assert standards.venue_name == "CoNLL"
        assert standards.venue_type == VenueType.MID_CONFERENCE
        assert standards.acceptance_threshold.base_threshold == 3.0
        assert standards.reviewer_criteria.min_h_index == 8
        assert standards.min_review_length == 300
        assert standards.max_review_length == 600
        assert standards.min_reviewer_count == 2
    
    def test_get_venue_standards_not_found(self, manager):
        """Test getting standards for non-existent venue."""
        standards = manager.get_venue_standards("NonExistentVenue")
        assert standards is None
    
    def test_calculate_acceptance_threshold_single_review(self, manager, sample_venue_acl, sample_review_good):
        """Test calculating acceptance threshold with single review."""
        threshold = manager.calculate_acceptance_threshold(sample_venue_acl, [sample_review_good])
        
        # Should be based on weighted average of review scores
        assert isinstance(threshold, float)
        assert threshold > 0.0
        assert threshold <= 5.0
    
    def test_calculate_acceptance_threshold_multiple_reviews(self, manager, sample_venue_acl, sample_review_good, sample_review_poor):
        """Test calculating acceptance threshold with multiple reviews."""
        reviews = [sample_review_good, sample_review_poor]
        threshold = manager.calculate_acceptance_threshold(sample_venue_acl, reviews)
        
        assert isinstance(threshold, float)
        assert threshold > 0.0
        assert threshold <= 5.0
    
    def test_calculate_acceptance_threshold_invalid_venue(self, manager, sample_review_good):
        """Test calculating threshold for unsupported venue."""
        invalid_venue = EnhancedVenue(
            id="invalid",
            name="InvalidVenue",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Test"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            manager.calculate_acceptance_threshold(invalid_venue, [sample_review_good])
        
        assert "venue.name" in str(exc_info.value)
    
    def test_calculate_acceptance_threshold_empty_reviews(self, manager, sample_venue_acl):
        """Test calculating threshold with empty reviews list."""
        with pytest.raises(ValidationError) as exc_info:
            manager.calculate_acceptance_threshold(sample_venue_acl, [])
        
        assert "reviews" in str(exc_info.value)
    
    def test_get_reviewer_selection_criteria_acl(self, manager, sample_venue_acl):
        """Test getting reviewer selection criteria for ACL."""
        criteria = manager.get_reviewer_selection_criteria(sample_venue_acl)
        
        assert criteria.min_h_index == 15
        assert criteria.min_years_experience == 5
        assert criteria.preferred_institution_tiers == [1, 2]
        assert criteria.min_reputation_score == 0.6
        assert criteria.max_reviews_per_reviewer == 3
        assert criteria.field_expertise_required is True
    
    def test_get_reviewer_selection_criteria_neurips(self, manager, sample_venue_neurips):
        """Test getting reviewer selection criteria for NeurIPS."""
        criteria = manager.get_reviewer_selection_criteria(sample_venue_neurips)
        
        assert criteria.min_h_index == 20
        assert criteria.min_years_experience == 7
        assert criteria.preferred_institution_tiers == [1]
        assert criteria.min_reputation_score == 0.7
        assert criteria.max_reviews_per_reviewer == 3
    
    def test_get_reviewer_selection_criteria_conll(self, manager, sample_venue_conll):
        """Test getting reviewer selection criteria for CoNLL."""
        criteria = manager.get_reviewer_selection_criteria(sample_venue_conll)
        
        assert criteria.min_h_index == 8
        assert criteria.min_years_experience == 3
        assert criteria.preferred_institution_tiers == [1, 2, 3]
        assert criteria.min_reputation_score == 0.3
        assert criteria.max_reviews_per_reviewer == 4
    
    def test_get_reviewer_selection_criteria_invalid_venue(self, manager):
        """Test getting criteria for unsupported venue."""
        invalid_venue = EnhancedVenue(
            id="invalid",
            name="InvalidVenue",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Test"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            manager.get_reviewer_selection_criteria(invalid_venue)
        
        assert "venue.name" in str(exc_info.value)
    
    def test_enforce_venue_review_standards_valid_review(self, manager, sample_venue_acl, sample_review_good):
        """Test enforcing standards with valid review."""
        is_valid, violations = manager.enforce_venue_review_standards(sample_venue_acl, sample_review_good)
        
        # Review should be valid for ACL standards
        assert is_valid is True
        assert len(violations) == 0
    
    def test_enforce_venue_review_standards_invalid_review(self, manager, sample_venue_neurips, sample_review_poor):
        """Test enforcing standards with invalid review."""
        is_valid, violations = manager.enforce_venue_review_standards(sample_venue_neurips, sample_review_poor)
        
        # Review should fail NeurIPS standards (higher requirements)
        assert is_valid is False
        assert len(violations) > 0
    
    def test_enforce_venue_review_standards_invalid_venue(self, manager, sample_review_good):
        """Test enforcing standards for unsupported venue."""
        invalid_venue = EnhancedVenue(
            id="invalid",
            name="InvalidVenue",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Test"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            manager.enforce_venue_review_standards(invalid_venue, sample_review_good)
        
        assert "venue.name" in str(exc_info.value)
    
    def test_get_minimum_reviewer_requirements_acl(self, manager, sample_venue_acl):
        """Test getting minimum reviewer requirements for ACL."""
        requirements = manager.get_minimum_reviewer_requirements(sample_venue_acl)
        
        assert requirements["min_reviewer_count"] == 3
        assert requirements["preferred_reviewer_count"] == 3
        assert requirements["min_h_index"] == 15
        assert requirements["min_years_experience"] == 5
        assert requirements["max_reviews_per_reviewer"] == 3
    
    def test_get_minimum_reviewer_requirements_neurips(self, manager, sample_venue_neurips):
        """Test getting minimum reviewer requirements for NeurIPS."""
        requirements = manager.get_minimum_reviewer_requirements(sample_venue_neurips)
        
        assert requirements["min_reviewer_count"] == 3
        assert requirements["preferred_reviewer_count"] == 4
        assert requirements["min_h_index"] == 20
        assert requirements["min_years_experience"] == 7
        assert requirements["max_reviews_per_reviewer"] == 3
    
    def test_get_minimum_reviewer_requirements_conll(self, manager, sample_venue_conll):
        """Test getting minimum reviewer requirements for CoNLL."""
        requirements = manager.get_minimum_reviewer_requirements(sample_venue_conll)
        
        assert requirements["min_reviewer_count"] == 2
        assert requirements["preferred_reviewer_count"] == 3
        assert requirements["min_h_index"] == 8
        assert requirements["min_years_experience"] == 3
        assert requirements["max_reviews_per_reviewer"] == 4
    
    def test_get_minimum_reviewer_requirements_invalid_venue(self, manager):
        """Test getting requirements for unsupported venue."""
        invalid_venue = EnhancedVenue(
            id="invalid",
            name="InvalidVenue",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Test"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            manager.get_minimum_reviewer_requirements(invalid_venue)
        
        assert "venue.name" in str(exc_info.value)
    
    def test_validate_reviewer_qualifications_qualified(self, manager, sample_venue_acl, sample_researcher_qualified):
        """Test validating qualified reviewer."""
        is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue_acl, sample_researcher_qualified)
        
        assert is_qualified is True
        assert len(issues) == 0
    
    def test_validate_reviewer_qualifications_unqualified(self, manager, sample_venue_neurips, sample_researcher_unqualified):
        """Test validating unqualified reviewer."""
        is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue_neurips, sample_researcher_unqualified)
        
        assert is_qualified is False
        assert len(issues) > 0
        
        # Check specific issues
        issue_text = " ".join(issues)
        assert "H-index too low" in issue_text
        assert "Insufficient experience" in issue_text
        assert "Reputation score too low" in issue_text
    
    def test_validate_reviewer_qualifications_partial(self, manager, sample_venue_conll, sample_researcher_qualified):
        """Test validating reviewer with partial qualifications."""
        # Qualified researcher should meet CoNLL requirements
        is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue_conll, sample_researcher_qualified)
        
        assert is_qualified is True
        assert len(issues) == 0
    
    def test_validate_reviewer_qualifications_invalid_venue(self, manager, sample_researcher_qualified):
        """Test validating reviewer for unsupported venue."""
        invalid_venue = EnhancedVenue(
            id="invalid",
            name="InvalidVenue",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Test"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            manager.validate_reviewer_qualifications(invalid_venue, sample_researcher_qualified)
        
        assert "venue.name" in str(exc_info.value)
    
    def test_calculate_reviewer_agreement_single_review(self, manager, sample_review_good):
        """Test calculating agreement with single review."""
        agreement = manager.calculate_reviewer_agreement([sample_review_good])
        
        assert agreement == 1.0  # Perfect agreement with single review
    
    def test_calculate_reviewer_agreement_multiple_reviews_same(self, manager, sample_review_good):
        """Test calculating agreement with identical reviews."""
        # Create identical reviews
        review1 = sample_review_good
        review2 = sample_review_good
        
        agreement = manager.calculate_reviewer_agreement([review1, review2])
        
        assert agreement == 1.0  # Perfect agreement
    
    def test_calculate_reviewer_agreement_multiple_reviews_different(self, manager, sample_review_good, sample_review_poor):
        """Test calculating agreement with different reviews."""
        agreement = manager.calculate_reviewer_agreement([sample_review_good, sample_review_poor])
        
        assert 0.0 <= agreement <= 1.0
        assert agreement < 1.0  # Should not be perfect agreement
    
    def test_calculate_reviewer_agreement_empty_reviews(self, manager):
        """Test calculating agreement with empty reviews."""
        agreement = manager.calculate_reviewer_agreement([])
        
        assert agreement == 1.0  # Default to perfect agreement


class TestGlobalVenueStandardsManager:
    """Test global venue standards manager functions."""
    
    def setup_method(self):
        """Reset global manager before each test."""
        import src.enhancements.venue_standards_system
        src.enhancements.venue_standards_system._venue_standards_manager = None
    
    def test_get_venue_standards_manager_singleton(self):
        """Test that get_venue_standards_manager returns singleton."""
        manager1 = get_venue_standards_manager()
        manager2 = get_venue_standards_manager()
        
        assert manager1 is manager2
    
    def test_get_venue_standards_manager_initialization(self):
        """Test that manager is properly initialized."""
        manager = get_venue_standards_manager()
        
        assert manager is not None
        assert len(manager.list_supported_venues()) == 4
        
        thresholds = manager.get_venue_score_thresholds()
        assert thresholds["ACL"] == 3.5
        assert thresholds["NeurIPS"] == 4.0
        assert thresholds["ICLR"] == 3.5
        assert thresholds["CoNLL"] == 3.0


class TestVenueStandardsSystemIntegration:
    """Integration tests for venue standards system."""
    
    @pytest.fixture
    def manager(self):
        """Create venue standards manager for testing."""
        return VenueStandardsManager()
    
    def test_full_venue_standards_workflow(self, manager):
        """Test complete venue standards workflow."""
        # Create venues
        acl_venue = EnhancedVenue(
            id="acl",
            name="ACL",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Natural Language Processing"
        )
        
        neurips_venue = EnhancedVenue(
            id="neurips",
            name="NeurIPS",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Machine Learning"
        )
        
        # Create researchers with different qualifications
        senior_researcher = EnhancedResearcher(
            id="senior",
            name="Dr. Senior",
            specialty="Machine Learning",
            level=ResearcherLevel.FULL_PROF,
            h_index=25,
            years_active=15,
            institution_tier=1,
            reputation_score=0.8
        )
        
        junior_researcher = EnhancedResearcher(
            id="junior",
            name="Dr. Junior",
            specialty="Machine Learning",
            level=ResearcherLevel.ASSISTANT_PROF,
            h_index=10,
            years_active=3,
            institution_tier=2,
            reputation_score=0.4
        )
        
        # Test reviewer qualifications
        acl_senior_qualified, acl_senior_issues = manager.validate_reviewer_qualifications(acl_venue, senior_researcher)
        acl_junior_qualified, acl_junior_issues = manager.validate_reviewer_qualifications(acl_venue, junior_researcher)
        
        neurips_senior_qualified, neurips_senior_issues = manager.validate_reviewer_qualifications(neurips_venue, senior_researcher)
        neurips_junior_qualified, neurips_junior_issues = manager.validate_reviewer_qualifications(neurips_venue, junior_researcher)
        
        # Senior researcher should qualify for both venues
        assert acl_senior_qualified is True
        assert neurips_senior_qualified is True
        
        # Junior researcher should qualify for ACL but not NeurIPS (higher standards)
        assert acl_junior_qualified is False  # h_index 10 < 15 required for ACL
        assert neurips_junior_qualified is False  # h_index 10 < 20 required for NeurIPS
        
        # Test venue requirements
        acl_requirements = manager.get_minimum_reviewer_requirements(acl_venue)
        neurips_requirements = manager.get_minimum_reviewer_requirements(neurips_venue)
        
        # NeurIPS should have higher requirements
        assert neurips_requirements["min_h_index"] > acl_requirements["min_h_index"]
        assert neurips_requirements["min_years_experience"] > acl_requirements["min_years_experience"]
        assert neurips_requirements["preferred_reviewer_count"] >= acl_requirements["preferred_reviewer_count"]
        
        # Test score thresholds
        thresholds = manager.get_venue_score_thresholds()
        assert thresholds["NeurIPS"] > thresholds["ACL"]  # NeurIPS has higher threshold
        
        # Test venue standards
        acl_standards = manager.get_venue_standards("ACL")
        neurips_standards = manager.get_venue_standards("NeurIPS")
        
        assert acl_standards is not None
        assert neurips_standards is not None
        
        # NeurIPS should have higher standards
        assert neurips_standards.acceptance_threshold.base_threshold > acl_standards.acceptance_threshold.base_threshold
        assert neurips_standards.required_technical_depth > acl_standards.required_technical_depth
        assert neurips_standards.min_review_length > acl_standards.min_review_length


if __name__ == "__main__":
    pytest.main([__file__])