"""
Venue-Specific Review Standards Enforcement System

This module implements venue-specific review standards enforcement including
ReviewRequirements class for venue-specific standards, QualityStandards validator
for different venue types, and logic to enforce minimum review lengths based on venue.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from src.data.enhanced_models import (
    EnhancedVenue, StructuredReview, VenueType, ReviewRequirements, 
    QualityStandards, ReviewerCriteria, EnhancedResearcher
)
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VenueStandardsConfig:
    """Configuration for venue-specific standards."""
    venue_type: VenueType
    min_word_count: int
    max_word_count: int
    min_strengths: int
    min_weaknesses: int
    requires_questions: bool
    requires_suggestions: bool
    min_confidence_level: int
    requires_detailed_technical_comments: bool
    min_technical_comment_length: int
    acceptance_threshold: float


class ReviewRequirementsManager:
    """Manages venue-specific review requirements and standards."""
    
    def __init__(self):
        self.venue_standards = self._initialize_venue_standards()
        logger.info("ReviewRequirementsManager initialized with venue-specific standards")
    
    def _initialize_venue_standards(self) -> Dict[VenueType, VenueStandardsConfig]:
        """Initialize venue-specific standards based on requirements analysis."""
        return {
            VenueType.TOP_CONFERENCE: VenueStandardsConfig(
                venue_type=VenueType.TOP_CONFERENCE,
                min_word_count=600,
                max_word_count=1000,
                min_strengths=3,
                min_weaknesses=2,
                requires_questions=True,
                requires_suggestions=True,
                min_confidence_level=4,
                requires_detailed_technical_comments=True,
                min_technical_comment_length=200,
                acceptance_threshold=8.5
            ),
            VenueType.MID_CONFERENCE: VenueStandardsConfig(
                venue_type=VenueType.MID_CONFERENCE,
                min_word_count=400,
                max_word_count=800,
                min_strengths=2,
                min_weaknesses=1,
                requires_questions=True,
                requires_suggestions=False,
                min_confidence_level=3,
                requires_detailed_technical_comments=True,
                min_technical_comment_length=150,
                acceptance_threshold=6.5
            ),
            VenueType.LOW_CONFERENCE: VenueStandardsConfig(
                venue_type=VenueType.LOW_CONFERENCE,
                min_word_count=300,
                max_word_count=600,
                min_strengths=2,
                min_weaknesses=1,
                requires_questions=False,
                requires_suggestions=False,
                min_confidence_level=2,
                requires_detailed_technical_comments=False,
                min_technical_comment_length=100,
                acceptance_threshold=5.0
            ),
            VenueType.TOP_JOURNAL: VenueStandardsConfig(
                venue_type=VenueType.TOP_JOURNAL,
                min_word_count=800,
                max_word_count=1200,
                min_strengths=3,
                min_weaknesses=2,
                requires_questions=True,
                requires_suggestions=True,
                min_confidence_level=4,
                requires_detailed_technical_comments=True,
                min_technical_comment_length=250,
                acceptance_threshold=9.0
            ),
            VenueType.SPECIALIZED_JOURNAL: VenueStandardsConfig(
                venue_type=VenueType.SPECIALIZED_JOURNAL,
                min_word_count=500,
                max_word_count=900,
                min_strengths=2,
                min_weaknesses=2,
                requires_questions=True,
                requires_suggestions=True,
                min_confidence_level=3,
                requires_detailed_technical_comments=True,
                min_technical_comment_length=180,
                acceptance_threshold=7.0
            ),
            VenueType.GENERAL_JOURNAL: VenueStandardsConfig(
                venue_type=VenueType.GENERAL_JOURNAL,
                min_word_count=400,
                max_word_count=700,
                min_strengths=2,
                min_weaknesses=1,
                requires_questions=False,
                requires_suggestions=False,
                min_confidence_level=3,
                requires_detailed_technical_comments=False,
                min_technical_comment_length=120,
                acceptance_threshold=6.0
            )
        }
    
    def create_review_requirements(self, venue_type: VenueType) -> ReviewRequirements:
        """Create ReviewRequirements object for a specific venue type."""
        standards = self.venue_standards.get(venue_type)
        if not standards:
            logger.warning(f"No standards found for venue type {venue_type}, using default")
            standards = self.venue_standards[VenueType.MID_CONFERENCE]
        
        requirements = ReviewRequirements(
            min_word_count=standards.min_word_count,
            max_word_count=standards.max_word_count,
            required_sections=["summary", "strengths", "weaknesses", "detailed_comments"],
            min_strengths=standards.min_strengths,
            min_weaknesses=standards.min_weaknesses,
            requires_questions=standards.requires_questions,
            requires_suggestions=standards.requires_suggestions
        )
        
        logger.debug(f"Created ReviewRequirements for {venue_type.value}: {requirements}")
        return requirements
    
    def get_venue_standards(self, venue_type: VenueType) -> VenueStandardsConfig:
        """Get venue-specific standards configuration."""
        standards = self.venue_standards.get(venue_type)
        if not standards:
            logger.warning(f"No standards found for venue type {venue_type}, using MID_CONFERENCE defaults")
            return self.venue_standards[VenueType.MID_CONFERENCE]
        return standards


class QualityStandardsValidator:
    """Validates reviews against venue-specific quality standards."""
    
    def __init__(self):
        self.requirements_manager = ReviewRequirementsManager()
        logger.info("QualityStandardsValidator initialized")
    
    def validate_review_against_venue_standards(self, review: StructuredReview, 
                                              venue: EnhancedVenue) -> Tuple[bool, List[str]]:
        """Validate review against comprehensive venue-specific standards."""
        errors = []
        standards = self.requirements_manager.get_venue_standards(venue.venue_type)
        
        # Validate word count requirements
        word_count_valid, word_errors = self._validate_word_count(review, standards)
        errors.extend(word_errors)
        
        # Validate structural requirements
        structure_valid, structure_errors = self._validate_structure_requirements(review, standards)
        errors.extend(structure_errors)
        
        # Validate confidence level
        confidence_valid, confidence_errors = self._validate_confidence_level(review, standards)
        errors.extend(confidence_errors)
        
        # Validate technical comments if required
        tech_comments_valid, tech_errors = self._validate_technical_comments(review, standards)
        errors.extend(tech_errors)
        
        # Validate questions and suggestions if required
        questions_valid, question_errors = self._validate_questions_and_suggestions(review, standards)
        errors.extend(question_errors)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"Review {review.review_id} passed venue standards validation for {venue.venue_type.value}")
        else:
            logger.warning(f"Review {review.review_id} failed venue standards validation: {errors}")
        
        return is_valid, errors
    
    def _validate_word_count(self, review: StructuredReview, 
                           standards: VenueStandardsConfig) -> Tuple[bool, List[str]]:
        """Validate review meets word count requirements."""
        errors = []
        
        # Calculate total word count from all text sections
        total_text = " ".join([
            review.executive_summary or "",
            review.technical_comments or "",
            review.presentation_comments or "",
            " ".join(s.description for s in review.detailed_strengths),
            " ".join(w.description for w in review.detailed_weaknesses),
            " ".join(review.questions_for_authors),
            " ".join(review.suggestions_for_improvement)
        ])
        
        word_count = len(total_text.split())
        
        if word_count < standards.min_word_count:
            errors.append(
                f"Review word count {word_count} is below minimum {standards.min_word_count} "
                f"for {standards.venue_type.value}"
            )
        
        if word_count > standards.max_word_count:
            errors.append(
                f"Review word count {word_count} exceeds maximum {standards.max_word_count} "
                f"for {standards.venue_type.value}"
            )
        
        return len(errors) == 0, errors
    
    def _validate_structure_requirements(self, review: StructuredReview, 
                                       standards: VenueStandardsConfig) -> Tuple[bool, List[str]]:
        """Validate review meets structural requirements."""
        errors = []
        
        # Check minimum strengths
        if len(review.detailed_strengths) < standards.min_strengths:
            errors.append(
                f"Review must have at least {standards.min_strengths} detailed strengths, "
                f"found {len(review.detailed_strengths)} for {standards.venue_type.value}"
            )
        
        # Check minimum weaknesses
        if len(review.detailed_weaknesses) < standards.min_weaknesses:
            errors.append(
                f"Review must have at least {standards.min_weaknesses} detailed weaknesses, "
                f"found {len(review.detailed_weaknesses)} for {standards.venue_type.value}"
            )
        
        # Check executive summary
        if not review.executive_summary or len(review.executive_summary.strip()) < 50:
            errors.append(
                f"Executive summary is missing or too short (minimum 50 characters) "
                f"for {standards.venue_type.value}"
            )
        
        return len(errors) == 0, errors
    
    def _validate_confidence_level(self, review: StructuredReview, 
                                 standards: VenueStandardsConfig) -> Tuple[bool, List[str]]:
        """Validate confidence level meets venue requirements."""
        errors = []
        
        if review.confidence_level < standards.min_confidence_level:
            errors.append(
                f"Confidence level {review.confidence_level} is below minimum "
                f"{standards.min_confidence_level} for {standards.venue_type.value}"
            )
        
        return len(errors) == 0, errors
    
    def _validate_technical_comments(self, review: StructuredReview, 
                                   standards: VenueStandardsConfig) -> Tuple[bool, List[str]]:
        """Validate technical comments meet venue requirements."""
        errors = []
        
        if standards.requires_detailed_technical_comments:
            if not review.technical_comments:
                errors.append(
                    f"Venue {standards.venue_type.value} requires detailed technical comments"
                )
            elif len(review.technical_comments) < standards.min_technical_comment_length:
                errors.append(
                    f"Technical comments length {len(review.technical_comments)} is below "
                    f"minimum {standards.min_technical_comment_length} for {standards.venue_type.value}"
                )
        
        return len(errors) == 0, errors
    
    def _validate_questions_and_suggestions(self, review: StructuredReview, 
                                          standards: VenueStandardsConfig) -> Tuple[bool, List[str]]:
        """Validate questions and suggestions meet venue requirements."""
        errors = []
        
        if standards.requires_questions and len(review.questions_for_authors) == 0:
            errors.append(
                f"Venue {standards.venue_type.value} requires questions for authors"
            )
        
        if standards.requires_suggestions and len(review.suggestions_for_improvement) == 0:
            errors.append(
                f"Venue {standards.venue_type.value} requires suggestions for improvement"
            )
        
        return len(errors) == 0, errors
    
    def enforce_minimum_review_length(self, review: StructuredReview, 
                                    venue_type: VenueType) -> Tuple[bool, str]:
        """Enforce minimum review length based on venue type (300-1000 words)."""
        standards = self.requirements_manager.get_venue_standards(venue_type)
        
        # Calculate total word count
        total_text = " ".join([
            review.executive_summary or "",
            review.technical_comments or "",
            review.presentation_comments or "",
            " ".join(s.description for s in review.detailed_strengths),
            " ".join(w.description for w in review.detailed_weaknesses),
            " ".join(review.questions_for_authors),
            " ".join(review.suggestions_for_improvement)
        ])
        
        word_count = len(total_text.split())
        
        if word_count < standards.min_word_count:
            message = (
                f"Review length {word_count} words is below minimum {standards.min_word_count} "
                f"words required for {venue_type.value}"
            )
            logger.warning(f"Review {review.review_id}: {message}")
            return False, message
        
        logger.info(f"Review {review.review_id} meets minimum length requirement for {venue_type.value}")
        return True, f"Review meets minimum length requirement ({word_count} words)"
    
    def validate_venue_type_standards(self, venue_type: VenueType, 
                                    reviews: List[StructuredReview]) -> Dict[str, any]:
        """Validate multiple reviews against venue type standards and return summary."""
        standards = self.requirements_manager.get_venue_standards(venue_type)
        
        validation_results = {
            'venue_type': venue_type.value,
            'total_reviews': len(reviews),
            'passed_reviews': 0,
            'failed_reviews': 0,
            'validation_details': [],
            'common_failures': {},
            'average_word_count': 0,
            'standards_applied': {
                'min_word_count': standards.min_word_count,
                'max_word_count': standards.max_word_count,
                'min_strengths': standards.min_strengths,
                'min_weaknesses': standards.min_weaknesses,
                'requires_questions': standards.requires_questions,
                'requires_suggestions': standards.requires_suggestions,
                'min_confidence_level': standards.min_confidence_level
            }
        }
        
        word_counts = []
        failure_types = {}
        
        for review in reviews:
            # Create a mock venue for validation
            from src.data.enhanced_models import EnhancedVenue
            mock_venue = EnhancedVenue(
                id=f"mock_{venue_type.value}",
                name=f"Mock {venue_type.value}",
                venue_type=venue_type,
                field="Computer Science"
            )
            
            is_valid, errors = self.validate_review_against_venue_standards(review, mock_venue)
            
            # Calculate word count for this review
            total_text = " ".join([
                review.executive_summary or "",
                review.technical_comments or "",
                review.presentation_comments or "",
                " ".join(s.description for s in review.detailed_strengths),
                " ".join(w.description for w in review.detailed_weaknesses),
                " ".join(review.questions_for_authors),
                " ".join(review.suggestions_for_improvement)
            ])
            word_count = len(total_text.split())
            word_counts.append(word_count)
            
            if is_valid:
                validation_results['passed_reviews'] += 1
            else:
                validation_results['failed_reviews'] += 1
                
                # Track failure types
                for error in errors:
                    error_type = error.split(':')[0] if ':' in error else error.split(' ')[0]
                    failure_types[error_type] = failure_types.get(error_type, 0) + 1
            
            validation_results['validation_details'].append({
                'review_id': review.review_id,
                'is_valid': is_valid,
                'errors': errors,
                'word_count': word_count
            })
        
        validation_results['common_failures'] = failure_types
        validation_results['average_word_count'] = sum(word_counts) / len(word_counts) if word_counts else 0
        
        logger.info(
            f"Validated {len(reviews)} reviews for {venue_type.value}: "
            f"{validation_results['passed_reviews']} passed, {validation_results['failed_reviews']} failed"
        )
        
        return validation_results


class VenueStandardsEnforcer:
    """Main class for enforcing venue-specific review standards."""
    
    def __init__(self):
        self.requirements_manager = ReviewRequirementsManager()
        self.quality_validator = QualityStandardsValidator()
        logger.info("VenueStandardsEnforcer initialized")
    
    def enforce_venue_standards(self, review: StructuredReview, 
                              venue: EnhancedVenue) -> Tuple[bool, List[str], Dict[str, any]]:
        """Comprehensive enforcement of venue-specific standards."""
        
        # Validate against venue standards
        is_valid, errors = self.quality_validator.validate_review_against_venue_standards(review, venue)
        
        # Get enforcement details
        standards = self.requirements_manager.get_venue_standards(venue.venue_type)
        enforcement_details = {
            'venue_id': venue.id,
            'venue_type': venue.venue_type.value,
            'review_id': review.review_id,
            'standards_applied': {
                'min_word_count': standards.min_word_count,
                'max_word_count': standards.max_word_count,
                'min_strengths': standards.min_strengths,
                'min_weaknesses': standards.min_weaknesses,
                'requires_questions': standards.requires_questions,
                'requires_suggestions': standards.requires_suggestions,
                'min_confidence_level': standards.min_confidence_level,
                'requires_detailed_technical_comments': standards.requires_detailed_technical_comments
            },
            'validation_results': {
                'is_valid': is_valid,
                'errors': errors,
                'error_count': len(errors)
            }
        }
        
        if is_valid:
            logger.info(f"Review {review.review_id} successfully meets all standards for venue {venue.id}")
        else:
            logger.warning(f"Review {review.review_id} failed venue standards enforcement: {errors}")
        
        return is_valid, errors, enforcement_details
    
    def get_venue_requirements_summary(self, venue_type: VenueType) -> Dict[str, any]:
        """Get a summary of requirements for a venue type."""
        standards = self.requirements_manager.get_venue_standards(venue_type)
        requirements = self.requirements_manager.create_review_requirements(venue_type)
        
        return {
            'venue_type': venue_type.value,
            'word_count_range': f"{standards.min_word_count}-{standards.max_word_count} words",
            'structural_requirements': {
                'min_strengths': standards.min_strengths,
                'min_weaknesses': standards.min_weaknesses,
                'requires_questions': standards.requires_questions,
                'requires_suggestions': standards.requires_suggestions
            },
            'quality_requirements': {
                'min_confidence_level': standards.min_confidence_level,
                'requires_detailed_technical_comments': standards.requires_detailed_technical_comments,
                'min_technical_comment_length': standards.min_technical_comment_length,
                'acceptance_threshold': standards.acceptance_threshold
            },
            'required_sections': requirements.required_sections
        }
    
    def compare_venue_standards(self, venue_types: List[VenueType]) -> Dict[str, any]:
        """Compare standards across multiple venue types."""
        comparison = {
            'venue_types': [vt.value for vt in venue_types],
            'comparison_matrix': {},
            'strictness_ranking': []
        }
        
        standards_data = {}
        for venue_type in venue_types:
            standards = self.requirements_manager.get_venue_standards(venue_type)
            standards_data[venue_type.value] = {
                'min_word_count': standards.min_word_count,
                'max_word_count': standards.max_word_count,
                'min_strengths': standards.min_strengths,
                'min_weaknesses': standards.min_weaknesses,
                'min_confidence_level': standards.min_confidence_level,
                'acceptance_threshold': standards.acceptance_threshold,
                'strictness_score': self._calculate_strictness_score(standards)
            }
        
        comparison['comparison_matrix'] = standards_data
        
        # Rank by strictness
        sorted_venues = sorted(
            standards_data.items(), 
            key=lambda x: x[1]['strictness_score'], 
            reverse=True
        )
        comparison['strictness_ranking'] = [venue for venue, _ in sorted_venues]
        
        return comparison
    
    def _calculate_strictness_score(self, standards: VenueStandardsConfig) -> float:
        """Calculate a strictness score for venue standards."""
        score = 0.0
        
        # Word count requirements (normalized)
        score += standards.min_word_count / 1000.0  # Max expected 1000
        
        # Structural requirements
        score += standards.min_strengths * 0.1
        score += standards.min_weaknesses * 0.1
        
        # Boolean requirements
        score += 0.2 if standards.requires_questions else 0.0
        score += 0.2 if standards.requires_suggestions else 0.0
        score += 0.3 if standards.requires_detailed_technical_comments else 0.0
        
        # Confidence and acceptance thresholds
        score += standards.min_confidence_level * 0.1
        score += standards.acceptance_threshold / 10.0  # Normalize to 0-1
        
        return score