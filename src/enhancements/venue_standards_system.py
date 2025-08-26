"""
Venue-Specific Standards and Thresholds System

This module implements venue-specific standards and thresholds with PeerRead data,
including acceptance threshold calculation, reviewer selection criteria, and
venue-specific review standards enforcement.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics

from src.data.enhanced_models import (
    EnhancedVenue, VenueType, EnhancedResearcher, StructuredReview,
    ReviewDecision, ResearcherLevel
)
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

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
    """
    Manages venue-specific standards and thresholds with PeerRead calibration.
    
    This class implements acceptance threshold calculation using PeerRead score
    distributions, reviewer selection criteria based on venue prestige, and
    enforcement of venue-specific review standards.
    """
    
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
            venue_id="acl",
            venue_name="ACL",
            venue_type=VenueType.TOP_CONFERENCE,
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
        
        # ICLR Standards
        iclr_standards = VenueStandards(
            venue_id="iclr", venue_name="ICLR", venue_type=VenueType.TOP_CONFERENCE,
            acceptance_threshold=AcceptanceThreshold(
                base_threshold=3.5,
                dimension_weights={
                    "novelty": 0.22, "technical_quality": 0.28, "clarity": 0.18,
                    "significance": 0.22, "reproducibility": 0.08, "related_work": 0.02
                },
                minimum_score=2.5, confidence_requirement=3, reviewer_agreement_threshold=0.65
            ),
            reviewer_criteria=ReviewerSelectionCriteria(
                min_h_index=12, min_years_experience=4, preferred_institution_tiers=[1, 2],
                min_reputation_score=0.5, max_reviews_per_reviewer=3, field_expertise_required=True
            ),
            min_review_length=450, max_review_length=900, required_technical_depth=3.5,
            novelty_threshold=3.0, significance_threshold=3.5, min_reviewer_count=3,
            preferred_reviewer_count=3, review_deadline_weeks=8, late_penalty_per_day=0.08
        )
        
        # CoNLL Standards
        conll_standards = VenueStandards(
            venue_id="conll", venue_name="CoNLL", venue_type=VenueType.MID_CONFERENCE,
            acceptance_threshold=AcceptanceThreshold(
                base_threshold=3.0,
                dimension_weights={
                    "novelty": 0.18, "technical_quality": 0.25, "clarity": 0.20,
                    "significance": 0.20, "reproducibility": 0.12, "related_work": 0.05
                },
                minimum_score=2.0, confidence_requirement=2, reviewer_agreement_threshold=0.6
            ),
            reviewer_criteria=ReviewerSelectionCriteria(
                min_h_index=8, min_years_experience=3, preferred_institution_tiers=[1, 2, 3],
                min_reputation_score=0.3, max_reviews_per_reviewer=4, field_expertise_required=True
            ),
            min_review_length=300, max_review_length=600, required_technical_depth=3.0,
            novelty_threshold=2.5, significance_threshold=3.0, min_reviewer_count=2,
            preferred_reviewer_count=3, review_deadline_weeks=4, late_penalty_per_day=0.05
        )
        
        self._venue_standards = {
            "ACL": acl_standards, "NeurIPS": neurips_standards,
            "ICLR": iclr_standards, "CoNLL": conll_standards
        }
        logger.info(f"Initialized standards for {len(self._venue_standards)} venues")
    
    def calculate_acceptance_threshold(self, venue: EnhancedVenue, reviews: List[StructuredReview]) -> float:
        """Calculate acceptance threshold using PeerRead score distributions."""
        if venue.name not in self._venue_standards:
            raise ValidationError("venue.name", venue.name, f"one of {list(self._venue_standards.keys())}")
        if not reviews:
            raise ValidationError("reviews", reviews, "non-empty list")
        
        standards = self._venue_standards[venue.name]
        threshold_config = standards.acceptance_threshold
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for review in reviews:
            if not hasattr(review, 'criteria_scores'):
                continue
            
            review_score = 0.0
            review_weight = 0.0
            criteria_scores = review.criteria_scores
            
            for dimension, weight in threshold_config.dimension_weights.items():
                if hasattr(criteria_scores, dimension):
                    score = getattr(criteria_scores, dimension)
                    review_score += score * weight
                    review_weight += weight
            
            if review_weight > 0:
                review_score /= review_weight
                confidence_weight = getattr(review, 'confidence_level', 3) / 5.0
                total_weighted_score += review_score * confidence_weight
                total_weight += confidence_weight
        
        if total_weight == 0:
            return threshold_config.base_threshold
        
        calculated_threshold = total_weighted_score / total_weight
        final_threshold = max(calculated_threshold, threshold_config.minimum_score)
        
        logger.debug(f"Calculated acceptance threshold for {venue.name}: {final_threshold}")
        return final_threshold
    
    def get_reviewer_selection_criteria(self, venue: EnhancedVenue) -> ReviewerSelectionCriteria:
        """Get reviewer selection criteria based on venue prestige and PeerRead analysis."""
        if venue.name not in self._venue_standards:
            raise ValidationError("venue.name", venue.name, f"one of {list(self._venue_standards.keys())}")
        return self._venue_standards[venue.name].reviewer_criteria
    
    def enforce_venue_review_standards(self, venue: EnhancedVenue, review: StructuredReview) -> Tuple[bool, List[str]]:
        """Enforce venue-specific review standards including technical depth and novelty thresholds."""
        if venue.name not in self._venue_standards:
            raise ValidationError("venue.name", venue.name, f"one of {list(self._venue_standards.keys())}")
        
        standards = self._venue_standards[venue.name]
        violations = []
        
        # Check review length (approximate based on content)
        review_length = 0
        if hasattr(review, 'detailed_strengths') and review.detailed_strengths:
            review_length += sum(len(str(s)) for s in review.detailed_strengths)
        if hasattr(review, 'detailed_weaknesses') and review.detailed_weaknesses:
            review_length += sum(len(str(w)) for w in review.detailed_weaknesses)
        if hasattr(review, 'technical_comments') and review.technical_comments:
            review_length += len(str(review.technical_comments))
        
        if review_length < standards.min_review_length:
            violations.append(f"Review too short: {review_length} < {standards.min_review_length}")
        elif review_length > standards.max_review_length:
            violations.append(f"Review too long: {review_length} > {standards.max_review_length}")
        
        # Check technical depth, novelty, and significance thresholds
        if hasattr(review, 'criteria_scores'):
            cs = review.criteria_scores
            if hasattr(cs, 'technical_quality') and cs.technical_quality < standards.required_technical_depth:
                violations.append(f"Insufficient technical depth: {cs.technical_quality} < {standards.required_technical_depth}")
            if hasattr(cs, 'novelty') and cs.novelty < standards.novelty_threshold:
                violations.append(f"Novelty below threshold: {cs.novelty} < {standards.novelty_threshold}")
            if hasattr(cs, 'significance') and cs.significance < standards.significance_threshold:
                violations.append(f"Significance below threshold: {cs.significance} < {standards.significance_threshold}")
        
        # Check confidence requirement
        if hasattr(review, 'confidence_level') and review.confidence_level < standards.acceptance_threshold.confidence_requirement:
            violations.append(f"Confidence too low: {review.confidence_level} < {standards.acceptance_threshold.confidence_requirement}")
        
        # Check required sections
        for section in ["detailed_strengths", "detailed_weaknesses", "technical_comments"]:
            if not hasattr(review, section) or not getattr(review, section):
                violations.append(f"Missing required section: {section}")
        
        is_valid = len(violations) == 0
        if not is_valid:
            logger.warning(f"Review validation failed for {venue.name}: {violations}")
        
        return is_valid, violations
    
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
        
        is_qualified = len(issues) == 0
        if not is_qualified:
            logger.debug(f"Reviewer {reviewer.id} not qualified for {venue.name}: {issues}")
        
        return is_qualified, issues
    
    def get_venue_score_thresholds(self) -> Dict[str, float]:
        """Get PeerRead score thresholds for all venues."""
        return self._peerread_thresholds.copy()
    
    def get_venue_standards(self, venue_name: str) -> Optional[VenueStandards]:
        """Get complete venue standards configuration."""
        return self._venue_standards.get(venue_name)
    
    def list_supported_venues(self) -> List[str]:
        """List all supported venue names."""
        return list(self._venue_standards.keys())
    
    def calculate_reviewer_agreement(self, reviews: List[StructuredReview]) -> float:
        """Calculate reviewer agreement score."""
        if len(reviews) < 2:
            return 1.0
        
        agreements = []
        for i in range(len(reviews)):
            for j in range(i + 1, len(reviews)):
                review1, review2 = reviews[i], reviews[j]
                
                if hasattr(review1, 'recommendation') and hasattr(review2, 'recommendation'):
                    rec_agreement = 1.0 if review1.recommendation == review2.recommendation else 0.0
                    agreements.append(rec_agreement)
                
                if hasattr(review1, 'criteria_scores') and hasattr(review2, 'criteria_scores'):
                    score_diffs = []
                    for attr in ['novelty', 'technical_quality', 'clarity', 'significance']:
                        if hasattr(review1.criteria_scores, attr) and hasattr(review2.criteria_scores, attr):
                            score1 = getattr(review1.criteria_scores, attr)
                            score2 = getattr(review2.criteria_scores, attr)
                            score_diffs.append(abs(score1 - score2))
                    
                    if score_diffs:
                        avg_diff = sum(score_diffs) / len(score_diffs)
                        score_agreement = max(0.0, 1.0 - (avg_diff / 5.0))
                        agreements.append(score_agreement)
        
        return sum(agreements) / len(agreements) if agreements else 0.0


# Global venue standards manager instance
_venue_standards_manager: Optional[VenueStandardsManager] = None


def get_venue_standards_manager() -> VenueStandardsManager:
    """Get the global venue standards manager instance."""
    global _venue_standards_manager
    if _venue_standards_manager is None:
        _venue_standards_manager = VenueStandardsManager()
    return _venue_standards_manager