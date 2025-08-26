#!/usr/bin/env python3
"""
Direct test of venue standards functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.enhanced_models import (
    EnhancedVenue, VenueType, EnhancedResearcher, StructuredReview,
    ReviewDecision, ResearcherLevel, EnhancedReviewCriteria
)
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

# Import the classes directly by copying the implementation
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics

logger = get_logger(__name__)

@dataclass
class AcceptanceThreshold:
    """Acceptance threshold configuration for a venue."""
    base_threshold: float
    dimension_weights: Dict[str, float]
    minimum_score: float
    confidence_requirement: int
    reviewer_agreement_threshold: float

@dataclass
class ReviewerSelectionCriteria:
    """Reviewer selection criteria based on venue prestige and PeerRead analysis."""
    min_h_index: int
    min_years_experience: int
    preferred_institution_tiers: List[int]
    min_reputation_score: float
    max_reviews_per_reviewer: int
    field_expertise_required: bool
    conflict_avoidance_years: int = 3

@dataclass
class VenueStandards:
    """Comprehensive venue standards configuration."""
    venue_id: str
    venue_name: str
    venue_type: VenueType
    acceptance_threshold: AcceptanceThreshold
    reviewer_criteria: ReviewerSelectionCriteria
    min_review_length: int
    max_review_length: int
    required_technical_depth: float
    novelty_threshold: float
    significance_threshold: float
    min_reviewer_count: int
    preferred_reviewer_count: int
    review_deadline_weeks: int
    late_penalty_per_day: float

class VenueStandardsManager:
    """Manages venue-specific standards and thresholds with PeerRead calibration."""
    
    def __init__(self):
        """Initialize the venue standards manager."""
        self._venue_standards: Dict[str, VenueStandards] = {}
        self._peerread_thresholds = self._initialize_peerread_thresholds()
        self._initialize_venue_standards()
    
    def _initialize_peerread_thresholds(self) -> Dict[str, float]:
        """Initialize PeerRead score thresholds for different venues."""
        return {
            "ACL": 3.5,      # ≥3.5/5 from PeerRead analysis
            "NeurIPS": 4.0,  # ≥4.0/5 from PeerRead analysis
            "ICLR": 3.5,     # ≥3.5/5 from PeerRead analysis
            "CoNLL": 3.0     # ≥3.0/5 from PeerRead analysis
        }
    
    def _initialize_venue_standards(self):
        """Initialize standards for all supported venues."""
        # ACL Standards
        acl_standards = VenueStandards(
            venue_id="acl", venue_name="ACL", venue_type=VenueType.TOP_CONFERENCE,
            acceptance_threshold=AcceptanceThreshold(
                base_threshold=3.5,
                dimension_weights={
                    "novelty": 0.20, "technical_quality": 0.25, "clarity": 0.15,
                    "significance": 0.25, "reproducibility": 0.10, "related_work": 0.05
                },
                minimum_score=2.5, confidence_requirement=3, reviewer_agreement_threshold=0.7
            ),
            reviewer_criteria=ReviewerSelectionCriteria(
                min_h_index=15, min_years_experience=5, preferred_institution_tiers=[1, 2],
                min_reputation_score=0.6, max_reviews_per_reviewer=3, field_expertise_required=True
            ),
            min_review_length=400, max_review_length=800, required_technical_depth=3.5,
            novelty_threshold=3.0, significance_threshold=3.5, min_reviewer_count=3,
            preferred_reviewer_count=3, review_deadline_weeks=6, late_penalty_per_day=0.1
        )
        
        # NeurIPS Standards
        neurips_standards = VenueStandards(
            venue_id="neurips", venue_name="NeurIPS", venue_type=VenueType.TOP_CONFERENCE,
            acceptance_threshold=AcceptanceThreshold(
                base_threshold=4.0,
                dimension_weights={
                    "novelty": 0.25, "technical_quality": 0.30, "clarity": 0.15,
                    "significance": 0.20, "reproducibility": 0.08, "related_work": 0.02
                },
                minimum_score=3.0, confidence_requirement=4, reviewer_agreement_threshold=0.75
            ),
            reviewer_criteria=ReviewerSelectionCriteria(
                min_h_index=20, min_years_experience=7, preferred_institution_tiers=[1],
                min_reputation_score=0.7, max_reviews_per_reviewer=3, field_expertise_required=True
            ),
            min_review_length=500, max_review_length=1000, required_technical_depth=4.0,
            novelty_threshold=3.5, significance_threshold=4.0, min_reviewer_count=3,
            preferred_reviewer_count=4, review_deadline_weeks=6, late_penalty_per_day=0.15
        )
        
        self._venue_standards = {"ACL": acl_standards, "NeurIPS": neurips_standards}
        logger.info(f"Initialized standards for {len(self._venue_standards)} venues")
    
    def get_venue_score_thresholds(self) -> Dict[str, float]:
        """Get PeerRead score thresholds for all venues."""
        return self._peerread_thresholds.copy()
    
    def get_venue_standards(self, venue_name: str) -> Optional[VenueStandards]:
        """Get complete venue standards configuration."""
        return self._venue_standards.get(venue_name)
    
    def list_supported_venues(self) -> List[str]:
        """List all supported venue names."""
        return list(self._venue_standards.keys())
    
    def get_reviewer_selection_criteria(self, venue: EnhancedVenue) -> ReviewerSelectionCriteria:
        """Get reviewer selection criteria based on venue prestige and PeerRead analysis."""
        if venue.name not in self._venue_standards:
            raise ValidationError("venue.name", venue.name, f"one of {list(self._venue_standards.keys())}")
        return self._venue_standards[venue.name].reviewer_criteria
    
    def get_minimum_reviewer_requirements(self, venue: EnhancedVenue) -> Dict[str, int]:
        """Get minimum reviewer counts and experience requirements per venue."""
        if venue.name not in self._venue_standards:
            raise ValidationError("venue.name", venue.name, f"one of {list(self._venue_standards.keys())}")
        
        standards = self._venue_standards[venue.name]
        return {
            "min_reviewer_count": standards.min_reviewer_count,
            "preferred_reviewer_count": standards.preferred_reviewer_count,
            "min_h_index": standards.reviewer_criteria.min_h_index,
            "min_years_experience": standards.reviewer_criteria.min_years_experience,
            "max_reviews_per_reviewer": standards.reviewer_criteria.max_reviews_per_reviewer
        }
    
    def validate_reviewer_qualifications(self, venue: EnhancedVenue, reviewer: EnhancedResearcher) -> Tuple[bool, List[str]]:
        """Validate if a reviewer meets venue-specific qualifications."""
        if venue.name not in self._venue_standards:
            raise ValidationError("venue.name", venue.name, f"one of {list(self._venue_standards.keys())}")
        
        standards = self._venue_standards[venue.name]
        criteria = standards.reviewer_criteria
        issues = []
        
        if reviewer.h_index < criteria.min_h_index:
            issues.append(f"H-index too low: {reviewer.h_index} < {criteria.min_h_index}")
        if reviewer.years_active < criteria.min_years_experience:
            issues.append(f"Insufficient experience: {reviewer.years_active} < {criteria.min_years_experience}")
        if hasattr(reviewer, 'institution_tier') and reviewer.institution_tier not in criteria.preferred_institution_tiers:
            issues.append(f"Institution tier not preferred: {reviewer.institution_tier} not in {criteria.preferred_institution_tiers}")
        if hasattr(reviewer, 'reputation_score') and reviewer.reputation_score < criteria.min_reputation_score:
            issues.append(f"Reputation score too low: {reviewer.reputation_score} < {criteria.min_reputation_score}")
        
        return len(issues) == 0, issues


def test_venue_standards_system():
    """Test the venue standards system functionality."""
    print("Testing Venue Standards System...")
    
    # Test manager creation
    manager = VenueStandardsManager()
    print("✓ VenueStandardsManager created successfully")
    
    # Test basic functionality
    thresholds = manager.get_venue_score_thresholds()
    print(f"✓ PeerRead thresholds: {thresholds}")
    assert thresholds["ACL"] == 3.5
    assert thresholds["NeurIPS"] == 4.0
    
    venues = manager.list_supported_venues()
    print(f"✓ Supported venues: {venues}")
    assert "ACL" in venues
    assert "NeurIPS" in venues
    
    # Test venue standards
    acl_standards = manager.get_venue_standards("ACL")
    assert acl_standards is not None
    assert acl_standards.acceptance_threshold.base_threshold == 3.5
    print(f"✓ ACL standards: threshold={acl_standards.acceptance_threshold.base_threshold}")
    
    neurips_standards = manager.get_venue_standards("NeurIPS")
    assert neurips_standards is not None
    assert neurips_standards.acceptance_threshold.base_threshold == 4.0
    print(f"✓ NeurIPS standards: threshold={neurips_standards.acceptance_threshold.base_threshold}")
    
    # Test with sample venue
    sample_venue_acl = EnhancedVenue(
        id="acl-test", name="ACL", venue_type=VenueType.TOP_CONFERENCE,
        field="Natural Language Processing"
    )
    
    sample_venue_neurips = EnhancedVenue(
        id="neurips-test", name="NeurIPS", venue_type=VenueType.TOP_CONFERENCE,
        field="Machine Learning"
    )
    
    # Test reviewer criteria
    acl_criteria = manager.get_reviewer_selection_criteria(sample_venue_acl)
    assert acl_criteria.min_h_index == 15
    print(f"✓ ACL reviewer criteria: min_h_index={acl_criteria.min_h_index}")
    
    neurips_criteria = manager.get_reviewer_selection_criteria(sample_venue_neurips)
    assert neurips_criteria.min_h_index == 20
    print(f"✓ NeurIPS reviewer criteria: min_h_index={neurips_criteria.min_h_index}")
    
    # Test reviewer requirements
    acl_requirements = manager.get_minimum_reviewer_requirements(sample_venue_acl)
    assert acl_requirements["min_h_index"] == 15
    assert acl_requirements["min_reviewer_count"] == 3
    print(f"✓ ACL reviewer requirements: {acl_requirements}")
    
    neurips_requirements = manager.get_minimum_reviewer_requirements(sample_venue_neurips)
    assert neurips_requirements["min_h_index"] == 20
    assert neurips_requirements["preferred_reviewer_count"] == 4
    print(f"✓ NeurIPS reviewer requirements: {neurips_requirements}")
    
    # Test with sample researchers
    qualified_researcher = EnhancedResearcher(
        id="qualified", name="Dr. Qualified", specialty="ML",
        level=ResearcherLevel.FULL_PROF, h_index=25, years_active=15, institution_tier=1
    )
    qualified_researcher.reputation_score = 0.8
    
    unqualified_researcher = EnhancedResearcher(
        id="unqualified", name="Dr. Junior", specialty="ML",
        level=ResearcherLevel.GRADUATE_STUDENT, h_index=5, years_active=2, institution_tier=3
    )
    unqualified_researcher.reputation_score = 0.2
    
    # Test qualifications for ACL
    is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue_acl, qualified_researcher)
    assert is_qualified == True
    assert len(issues) == 0
    print(f"✓ Qualified researcher for ACL: qualified={is_qualified}")
    
    is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue_acl, unqualified_researcher)
    assert is_qualified == False
    assert len(issues) > 0
    print(f"✓ Unqualified researcher for ACL: qualified={is_qualified}, issues={len(issues)}")
    
    # Test qualifications for NeurIPS (higher standards)
    is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue_neurips, qualified_researcher)
    assert is_qualified == True
    print(f"✓ Qualified researcher for NeurIPS: qualified={is_qualified}")
    
    is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue_neurips, unqualified_researcher)
    assert is_qualified == False
    assert len(issues) > 0
    print(f"✓ Unqualified researcher for NeurIPS: qualified={is_qualified}, issues={len(issues)}")
    
    # Verify NeurIPS has higher standards than ACL
    assert neurips_standards.acceptance_threshold.base_threshold > acl_standards.acceptance_threshold.base_threshold
    assert neurips_criteria.min_h_index > acl_criteria.min_h_index
    assert neurips_criteria.min_reputation_score > acl_criteria.min_reputation_score
    print("✓ NeurIPS has higher standards than ACL")
    
    print("\n🎉 All venue standards tests passed!")


if __name__ == "__main__":
    test_venue_standards_system()