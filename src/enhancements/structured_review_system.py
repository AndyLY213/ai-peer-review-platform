"""
Structured Review Validation System

This module implements the enhanced multi-dimensional review system with PeerRead patterns,
including structured review validation, venue-specific standards, and score calibration.
"""

import re
import statistics
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.data.enhanced_models import (
    EnhancedReviewCriteria, StructuredReview, DetailedStrength, DetailedWeakness,
    VenueType, ReviewDecision, EnhancedVenue
)
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class PeerReadDimension(Enum):
    """PeerRead review dimensions mapped to our system."""
    IMPACT = "significance"  # IMPACT → significance
    SUBSTANCE = "technical_quality"  # SUBSTANCE → technical quality
    SOUNDNESS_CORRECTNESS = "technical_quality"  # SOUNDNESS_CORRECTNESS → technical quality
    ORIGINALITY = "novelty"  # ORIGINALITY → novelty
    CLARITY = "clarity"  # CLARITY → clarity
    MEANINGFUL_COMPARISON = "related_work"  # MEANINGFUL_COMPARISON → related work


@dataclass
class PeerReadCalibration:
    """PeerRead score distributions for calibration."""
    # Mean scores from PeerRead analysis
    impact_mean: float = 3.2
    substance_mean: float = 3.4
    soundness_mean: float = 3.6
    originality_mean: float = 3.1
    clarity_mean: float = 3.5
    comparison_mean: float = 3.3
    
    # Standard deviations
    impact_std: float = 1.1
    substance_std: float = 1.0
    soundness_std: float = 0.9
    originality_std: float = 1.2
    clarity_std: float = 1.0
    comparison_std: float = 1.1


@dataclass
class ReviewLanguagePatterns:
    """Common language patterns from PeerRead analysis by section."""
    
    # Executive summary patterns
    summary_starters: List[str] = field(default_factory=lambda: [
        "This paper presents",
        "The authors propose",
        "This work introduces",
        "The paper describes",
        "This study investigates"
    ])
    
    # Strength patterns by category
    technical_strengths: List[str] = field(default_factory=lambda: [
        "The methodology is sound",
        "The experimental design is rigorous",
        "The technical approach is well-motivated",
        "The implementation is thorough",
        "The evaluation is comprehensive"
    ])
    
    novelty_strengths: List[str] = field(default_factory=lambda: [
        "The approach is novel",
        "This is an innovative solution",
        "The idea is original",
        "The contribution is significant",
        "The work advances the state-of-the-art"
    ])
    
    clarity_strengths: List[str] = field(default_factory=lambda: [
        "The paper is well-written",
        "The presentation is clear",
        "The writing is easy to follow",
        "The structure is logical",
        "The figures are informative"
    ])
    
    # Weakness patterns by category
    technical_weaknesses: List[str] = field(default_factory=lambda: [
        "The methodology has limitations",
        "The experimental setup is insufficient",
        "The evaluation is incomplete",
        "The baselines are weak",
        "The statistical analysis is lacking"
    ])
    
    novelty_weaknesses: List[str] = field(default_factory=lambda: [
        "The novelty is limited",
        "The contribution is incremental",
        "Similar work exists",
        "The differences from prior work are unclear",
        "The innovation is marginal"
    ])
    
    clarity_weaknesses: List[str] = field(default_factory=lambda: [
        "The writing needs improvement",
        "The presentation is unclear",
        "The paper is hard to follow",
        "The structure is confusing",
        "The figures are poorly designed"
    ])
    
    # Common questions for authors
    common_questions: List[str] = field(default_factory=lambda: [
        "Can you provide more details on the methodology?",
        "How does this compare to recent work by [X]?",
        "What are the limitations of your approach?",
        "Can you discuss the computational complexity?",
        "Will you make the code and data available?",
        "How sensitive are the results to hyperparameters?",
        "Can you provide error bars or confidence intervals?",
        "How does performance vary across different datasets?"
    ])


class StructuredReviewValidator:
    """Validates structured reviews against venue requirements and PeerRead patterns."""
    
    def __init__(self):
        self.peerread_calibration = PeerReadCalibration()
        self.language_patterns = ReviewLanguagePatterns()
        self.venue_word_requirements = self._initialize_venue_requirements()
    
    def _initialize_venue_requirements(self) -> Dict[VenueType, Dict[str, int]]:
        """Initialize venue-specific word count requirements based on PeerRead analysis."""
        return {
            VenueType.TOP_CONFERENCE: {
                'min_words': 600,
                'max_words': 1000,
                'target_words': 800
            },
            VenueType.MID_CONFERENCE: {
                'min_words': 400,
                'max_words': 800,
                'target_words': 600
            },
            VenueType.LOW_CONFERENCE: {
                'min_words': 300,
                'max_words': 600,
                'target_words': 450
            },
            VenueType.TOP_JOURNAL: {
                'min_words': 800,
                'max_words': 1200,
                'target_words': 1000
            },
            VenueType.SPECIALIZED_JOURNAL: {
                'min_words': 500,
                'max_words': 900,
                'target_words': 700
            },
            VenueType.GENERAL_JOURNAL: {
                'min_words': 400,
                'max_words': 700,
                'target_words': 550
            }
        }
    
    def validate_six_dimensional_scoring(self, review: StructuredReview) -> Tuple[bool, List[str]]:
        """Validate that all six dimensions are scored within valid ranges."""
        errors = []
        criteria = review.criteria_scores
        
        # Check all dimensions are present and in valid range (1-10)
        dimensions = ['novelty', 'technical_quality', 'clarity', 'significance', 'reproducibility', 'related_work']
        
        for dim in dimensions:
            score = getattr(criteria, dim)
            if not (1.0 <= score <= 10.0):
                errors.append(f"{dim} score {score} is outside valid range [1.0, 10.0]")
        
        return len(errors) == 0, errors
    
    def validate_structured_sections(self, review: StructuredReview) -> Tuple[bool, List[str]]:
        """Validate that review has all required structured sections."""
        errors = []
        
        # Check executive summary
        if not review.executive_summary or len(review.executive_summary.strip()) < 50:
            errors.append("Executive summary is missing or too short (minimum 50 characters)")
        
        # Check minimum strengths (at least 2)
        if len(review.detailed_strengths) < 2:
            errors.append(f"Review must have at least 2 detailed strengths, found {len(review.detailed_strengths)}")
        
        # Check minimum weaknesses (at least 1)
        if len(review.detailed_weaknesses) < 1:
            errors.append("Review must have at least 1 detailed weakness")
        
        # Check detailed comments
        if not review.technical_comments or len(review.technical_comments.strip()) < 100:
            errors.append("Technical comments are missing or too short (minimum 100 characters)")
        
        # Check questions for authors (recommended but not required)
        if len(review.questions_for_authors) == 0:
            logger.warning(f"Review {review.review_id} has no questions for authors")
        
        return len(errors) == 0, errors
    
    def validate_venue_word_requirements(self, review: StructuredReview, venue: EnhancedVenue) -> Tuple[bool, List[str]]:
        """Validate review meets venue-specific word count requirements."""
        errors = []
        
        # Calculate total word count
        total_text = " ".join([
            review.executive_summary,
            review.technical_comments,
            review.presentation_comments,
            " ".join(s.description for s in review.detailed_strengths),
            " ".join(w.description for w in review.detailed_weaknesses),
            " ".join(review.questions_for_authors),
            " ".join(review.suggestions_for_improvement)
        ])
        
        word_count = len(total_text.split())
        
        # Get venue requirements
        requirements = self.venue_word_requirements.get(venue.venue_type)
        if not requirements:
            logger.warning(f"No word requirements found for venue type {venue.venue_type}")
            return True, []
        
        min_words = requirements['min_words']
        max_words = requirements['max_words']
        
        if word_count < min_words:
            errors.append(f"Review word count {word_count} is below minimum {min_words} for {venue.venue_type.value}")
        
        if word_count > max_words:
            errors.append(f"Review word count {word_count} exceeds maximum {max_words} for {venue.venue_type.value}")
        
        return len(errors) == 0, errors
    
    def validate_confidence_and_recommendation(self, review: StructuredReview) -> Tuple[bool, List[str]]:
        """Validate confidence level and recommendation are present and valid."""
        errors = []
        
        # Check confidence level (1-5 scale)
        if not (1 <= review.confidence_level <= 5):
            errors.append(f"Confidence level {review.confidence_level} is outside valid range [1, 5]")
        
        # Check recommendation is valid enum value
        if not isinstance(review.recommendation, ReviewDecision):
            errors.append(f"Invalid recommendation: {review.recommendation}")
        
        # Check consistency between scores and recommendation
        avg_score = review.criteria_scores.get_average_score()
        
        # Define score thresholds for recommendations
        if review.recommendation == ReviewDecision.ACCEPT and avg_score < 7.0:
            errors.append(f"Accept recommendation inconsistent with average score {avg_score:.1f}")
        elif review.recommendation == ReviewDecision.REJECT and avg_score > 4.0:
            errors.append(f"Reject recommendation inconsistent with average score {avg_score:.1f}")
        
        return len(errors) == 0, errors
    
    def calibrate_scores_with_peerread(self, review: StructuredReview) -> StructuredReview:
        """Calibrate review scores using PeerRead distributions."""
        calibrated_review = review
        
        # Map our dimensions to PeerRead means (converting from 5-point to 10-point scale)
        peerread_means_10pt = {
            'significance': self.peerread_calibration.impact_mean * 2,  # 3.2 * 2 = 6.4
            'technical_quality': (self.peerread_calibration.substance_mean + 
                                self.peerread_calibration.soundness_mean) / 2 * 2,  # ~7.0
            'novelty': self.peerread_calibration.originality_mean * 2,  # 3.1 * 2 = 6.2
            'clarity': self.peerread_calibration.clarity_mean * 2,  # 3.5 * 2 = 7.0
            'related_work': self.peerread_calibration.comparison_mean * 2,  # 3.3 * 2 = 6.6
            'reproducibility': 6.0  # Default as not directly measured in PeerRead
        }
        
        # Apply calibration by adjusting scores toward PeerRead means
        calibration_strength = 0.3  # How much to adjust toward PeerRead means
        
        criteria = calibrated_review.criteria_scores
        
        # Calibrate each dimension
        for dimension, peerread_mean in peerread_means_10pt.items():
            current_score = getattr(criteria, dimension)
            calibrated_score = (current_score * (1 - calibration_strength) + 
                              peerread_mean * calibration_strength)
            
            # Ensure score stays in valid range
            calibrated_score = max(1.0, min(10.0, calibrated_score))
            setattr(criteria, dimension, calibrated_score)
        
        return calibrated_review
    
    def integrate_peerread_language_patterns(self, review: StructuredReview, 
                                           enhance_language: bool = True) -> StructuredReview:
        """Integrate PeerRead review language patterns into review content."""
        if not enhance_language:
            return review
        
        # Create a copy to avoid modifying the original
        import copy
        enhanced_review = copy.deepcopy(review)
        
        # Enhance executive summary if too generic
        if len(enhanced_review.executive_summary) < 100:
            starter = self.language_patterns.summary_starters[0]  # Use first pattern
            enhanced_review.executive_summary = f"{starter} a method for addressing the research problem. {enhanced_review.executive_summary}"
        
        # Enhance strengths with pattern-based language
        for i, strength in enumerate(enhanced_review.detailed_strengths):
            if len(strength.description) < 50:  # If too short, enhance
                if strength.category.lower() in ['technical', 'methodology']:
                    pattern = self.language_patterns.technical_strengths[i % len(self.language_patterns.technical_strengths)]
                elif strength.category.lower() in ['novelty', 'originality']:
                    pattern = self.language_patterns.novelty_strengths[i % len(self.language_patterns.novelty_strengths)]
                else:
                    pattern = self.language_patterns.clarity_strengths[i % len(self.language_patterns.clarity_strengths)]
                
                strength.description = f"{pattern}. {strength.description}"
        
        # Enhance weaknesses with pattern-based language
        for i, weakness in enumerate(enhanced_review.detailed_weaknesses):
            if len(weakness.description) < 50:  # If too short, enhance
                if weakness.category.lower() in ['technical', 'methodology']:
                    pattern = self.language_patterns.technical_weaknesses[i % len(self.language_patterns.technical_weaknesses)]
                elif weakness.category.lower() in ['novelty', 'originality']:
                    pattern = self.language_patterns.novelty_weaknesses[i % len(self.language_patterns.novelty_weaknesses)]
                else:
                    pattern = self.language_patterns.clarity_weaknesses[i % len(self.language_patterns.clarity_weaknesses)]
                
                weakness.description = f"{pattern}. {weakness.description}"
        
        # Add common questions if none exist
        if len(enhanced_review.questions_for_authors) == 0:
            enhanced_review.questions_for_authors = self.language_patterns.common_questions[:2]
        
        return enhanced_review
    
    def validate_complete_review(self, review: StructuredReview, venue: EnhancedVenue) -> Tuple[bool, List[str]]:
        """Perform complete validation of a structured review."""
        all_errors = []
        
        # Validate six-dimensional scoring
        valid_scores, score_errors = self.validate_six_dimensional_scoring(review)
        all_errors.extend(score_errors)
        
        # Validate structured sections
        valid_sections, section_errors = self.validate_structured_sections(review)
        all_errors.extend(section_errors)
        
        # Validate venue requirements
        valid_venue, venue_errors = self.validate_venue_word_requirements(review, venue)
        all_errors.extend(venue_errors)
        
        # Validate confidence and recommendation
        valid_confidence, confidence_errors = self.validate_confidence_and_recommendation(review)
        all_errors.extend(confidence_errors)
        
        is_valid = len(all_errors) == 0
        
        if is_valid:
            logger.info(f"Review {review.review_id} passed all validation checks")
        else:
            logger.warning(f"Review {review.review_id} failed validation: {all_errors}")
        
        return is_valid, all_errors
    
    def enhance_review_with_peerread_patterns(self, review: StructuredReview) -> StructuredReview:
        """Apply PeerRead calibration and language patterns to enhance review quality."""
        # First calibrate scores
        calibrated_review = self.calibrate_scores_with_peerread(review)
        
        # Then enhance language patterns
        enhanced_review = self.integrate_peerread_language_patterns(calibrated_review)
        
        # Recalculate quality scores after enhancements
        enhanced_review._calculate_quality_scores()
        
        logger.info(f"Enhanced review {review.review_id} with PeerRead patterns")
        
        return enhanced_review


class ReviewRequirementsManager:
    """Manages venue-specific review requirements and standards."""
    
    def __init__(self):
        self.validator = StructuredReviewValidator()
    
    def get_venue_requirements(self, venue_type: VenueType) -> Dict[str, any]:
        """Get comprehensive requirements for a venue type."""
        base_requirements = {
            VenueType.TOP_CONFERENCE: {
                'min_word_count': 600,
                'max_word_count': 1000,
                'min_strengths': 3,
                'min_weaknesses': 2,
                'requires_questions': True,
                'requires_suggestions': True,
                'min_confidence': 4,
                'detailed_technical_comments': True
            },
            VenueType.MID_CONFERENCE: {
                'min_word_count': 400,
                'max_word_count': 800,
                'min_strengths': 2,
                'min_weaknesses': 1,
                'requires_questions': True,
                'requires_suggestions': False,
                'min_confidence': 3,
                'detailed_technical_comments': True
            },
            VenueType.LOW_CONFERENCE: {
                'min_word_count': 300,
                'max_word_count': 600,
                'min_strengths': 2,
                'min_weaknesses': 1,
                'requires_questions': False,
                'requires_suggestions': False,
                'min_confidence': 2,
                'detailed_technical_comments': False
            },
            VenueType.TOP_JOURNAL: {
                'min_word_count': 800,
                'max_word_count': 1200,
                'min_strengths': 3,
                'min_weaknesses': 2,
                'requires_questions': True,
                'requires_suggestions': True,
                'min_confidence': 4,
                'detailed_technical_comments': True
            },
            VenueType.SPECIALIZED_JOURNAL: {
                'min_word_count': 500,
                'max_word_count': 900,
                'min_strengths': 2,
                'min_weaknesses': 2,
                'requires_questions': True,
                'requires_suggestions': True,
                'min_confidence': 3,
                'detailed_technical_comments': True
            },
            VenueType.GENERAL_JOURNAL: {
                'min_word_count': 400,
                'max_word_count': 700,
                'min_strengths': 2,
                'min_weaknesses': 1,
                'requires_questions': False,
                'requires_suggestions': False,
                'min_confidence': 3,
                'detailed_technical_comments': False
            }
        }
        
        return base_requirements.get(venue_type, base_requirements[VenueType.MID_CONFERENCE])
    
    def check_review_meets_requirements(self, review: StructuredReview, 
                                      venue: EnhancedVenue) -> Tuple[bool, List[str]]:
        """Check if review meets all venue-specific requirements."""
        requirements = self.get_venue_requirements(venue.venue_type)
        errors = []
        
        # Use the validator for comprehensive checking
        is_valid, validation_errors = self.validator.validate_complete_review(review, venue)
        errors.extend(validation_errors)
        
        # Additional venue-specific checks
        if requirements['requires_questions'] and len(review.questions_for_authors) == 0:
            errors.append(f"Venue {venue.venue_type.value} requires questions for authors")
        
        if requirements['requires_suggestions'] and len(review.suggestions_for_improvement) == 0:
            errors.append(f"Venue {venue.venue_type.value} requires suggestions for improvement")
        
        if review.confidence_level < requirements['min_confidence']:
            errors.append(f"Confidence level {review.confidence_level} below minimum {requirements['min_confidence']} for {venue.venue_type.value}")
        
        if requirements['detailed_technical_comments'] and len(review.technical_comments) < 200:
            errors.append(f"Venue {venue.venue_type.value} requires detailed technical comments (minimum 200 characters)")
        
        return len(errors) == 0, errors