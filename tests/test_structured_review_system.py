"""
Unit tests for the Structured Review Validation System.

Tests all components of the enhanced multi-dimensional review system including
validation, PeerRead calibration, and venue-specific requirements.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.enhancements.structured_review_system import (
    StructuredReviewValidator, ReviewRequirementsManager, PeerReadCalibration,
    ReviewLanguagePatterns, PeerReadDimension
)
from src.data.enhanced_models import (
    StructuredReview, EnhancedReviewCriteria, DetailedStrength, DetailedWeakness,
    EnhancedVenue, VenueType, ReviewDecision
)


class TestStructuredReviewValidator:
    """Test cases for StructuredReviewValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = StructuredReviewValidator()
        
        # Create a sample review for testing
        self.sample_criteria = EnhancedReviewCriteria(
            novelty=7.0,
            technical_quality=8.0,
            clarity=6.0,
            significance=7.5,
            reproducibility=6.5,
            related_work=6.0
        )
        
        self.sample_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            criteria_scores=self.sample_criteria,
            confidence_level=4,
            recommendation=ReviewDecision.MINOR_REVISION,
            executive_summary="This paper presents a novel and comprehensive approach to machine learning optimization that addresses important limitations in current methods and provides significant advances over existing techniques. The work builds upon solid theoretical foundations while introducing substantial innovations that advance the state-of-the-art in this important and rapidly evolving research area. The authors provide comprehensive theoretical analysis, rigorous mathematical derivations, and extensive experimental validation across multiple datasets and scenarios to support their claims and demonstrate the effectiveness of their proposed approach.",
            detailed_strengths=[
                DetailedStrength(category="Technical", description="The methodology is sound and well-motivated with clear theoretical foundations. The authors provide rigorous mathematical analysis and demonstrate deep understanding of the underlying principles."),
                DetailedStrength(category="Novelty", description="The approach introduces innovative techniques that advance the state-of-the-art. The combination of existing methods with novel algorithmic contributions creates a significant advancement in the field.")
            ],
            detailed_weaknesses=[
                DetailedWeakness(category="Evaluation", description="The experimental evaluation could be more comprehensive with additional baselines. While the current experiments are solid, comparison with more recent methods would strengthen the paper significantly.")
            ],
            technical_comments="The technical approach is generally solid and well-executed throughout the paper, demonstrating a comprehensive understanding of the problem domain and existing methodologies. The authors provide extensive theoretical analysis with rigorous mathematical foundations that clearly demonstrate deep understanding of the underlying principles and their practical implications. The implementation appears correct and efficient, with careful attention to algorithmic details, optimization considerations, and computational efficiency aspects that are crucial for practical deployment. However, some details about the computational complexity could be clarified more thoroughly, particularly regarding the worst-case scenarios and average-case performance characteristics. The scalability analysis could be expanded to cover larger datasets and more diverse scenarios that would be encountered in real-world applications. The mathematical formulations are clear and well-presented throughout the document, with proofs that appear sound and complete, though some intermediate steps could benefit from additional explanation. The experimental methodology is appropriate and follows established best practices in the field, with careful consideration of statistical significance, proper baseline comparisons, and comprehensive evaluation protocols. The results are convincing and well-presented with appropriate visualizations, detailed statistical analysis, and thorough discussion of the findings and their implications for future research directions.",
            presentation_comments="The paper is generally well-written and clearly structured, making it accessible to readers familiar with the field. The figures and tables are informative and well-designed, effectively supporting the textual content. However, some sections could benefit from improved organization and clearer transitions between concepts.",
            questions_for_authors=["Can you provide more details on computational complexity?", "How does this compare to recent work by Smith et al.?", "What are the memory requirements?", "How does performance vary with different hyperparameters?"]
        )
        
        # Create sample venue
        self.sample_venue = EnhancedVenue(
            id="venue_001",
            name="Test Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="Machine Learning"
        )
    
    def test_validate_six_dimensional_scoring_valid(self):
        """Test validation of valid six-dimensional scoring."""
        is_valid, errors = self.validator.validate_six_dimensional_scoring(self.sample_review)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_six_dimensional_scoring_invalid_range(self):
        """Test validation with scores outside valid range."""
        # Create a review with valid criteria first, then manually set invalid scores
        invalid_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001"
        )
        
        # Manually set invalid scores to bypass __post_init__ validation
        invalid_review.criteria_scores.novelty = 11.0  # Invalid: > 10
        invalid_review.criteria_scores.technical_quality = 0.5  # Invalid: < 1
        
        is_valid, errors = self.validator.validate_six_dimensional_scoring(invalid_review)
        
        assert is_valid is False
        assert len(errors) == 2
        assert "novelty score 11.0 is outside valid range" in errors[0]
        assert "technical_quality score 0.5 is outside valid range" in errors[1]
    
    def test_validate_structured_sections_valid(self):
        """Test validation of valid structured sections."""
        is_valid, errors = self.validator.validate_structured_sections(self.sample_review)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_structured_sections_missing_summary(self):
        """Test validation with missing executive summary."""
        invalid_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            executive_summary="",  # Too short
            detailed_strengths=self.sample_review.detailed_strengths,
            detailed_weaknesses=self.sample_review.detailed_weaknesses,
            technical_comments=self.sample_review.technical_comments
        )
        
        is_valid, errors = self.validator.validate_structured_sections(invalid_review)
        
        assert is_valid is False
        assert any("Executive summary is missing or too short" in error for error in errors)
    
    def test_validate_structured_sections_insufficient_strengths(self):
        """Test validation with insufficient strengths."""
        invalid_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            executive_summary=self.sample_review.executive_summary,
            detailed_strengths=[self.sample_review.detailed_strengths[0]],  # Only 1 strength
            detailed_weaknesses=self.sample_review.detailed_weaknesses,
            technical_comments=self.sample_review.technical_comments
        )
        
        is_valid, errors = self.validator.validate_structured_sections(invalid_review)
        
        assert is_valid is False
        assert any("Review must have at least 2 detailed strengths" in error for error in errors)
    
    def test_validate_structured_sections_no_weaknesses(self):
        """Test validation with no weaknesses."""
        invalid_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            executive_summary=self.sample_review.executive_summary,
            detailed_strengths=self.sample_review.detailed_strengths,
            detailed_weaknesses=[],  # No weaknesses
            technical_comments=self.sample_review.technical_comments
        )
        
        is_valid, errors = self.validator.validate_structured_sections(invalid_review)
        
        assert is_valid is False
        assert any("Review must have at least 1 detailed weakness" in error for error in errors)
    
    def test_validate_venue_word_requirements_valid(self):
        """Test validation of venue word requirements with valid review."""
        is_valid, errors = self.validator.validate_venue_word_requirements(
            self.sample_review, self.sample_venue
        )
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_venue_word_requirements_too_short(self):
        """Test validation with review too short for venue."""
        short_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            executive_summary="Short summary.",
            detailed_strengths=[DetailedStrength(category="Test", description="Short.")],
            detailed_weaknesses=[DetailedWeakness(category="Test", description="Short.")],
            technical_comments="Short comment."
        )
        
        # Test with top conference (higher requirements)
        top_venue = EnhancedVenue(
            id="venue_002",
            name="Top Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="AI"
        )
        
        is_valid, errors = self.validator.validate_venue_word_requirements(short_review, top_venue)
        
        assert is_valid is False
        assert any("word count" in error and "below minimum" in error for error in errors)
    
    def test_validate_confidence_and_recommendation_valid(self):
        """Test validation of valid confidence and recommendation."""
        is_valid, errors = self.validator.validate_confidence_and_recommendation(self.sample_review)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_confidence_and_recommendation_invalid_confidence(self):
        """Test validation with invalid confidence level."""
        # Create valid review first, then manually set invalid confidence
        invalid_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            recommendation=ReviewDecision.ACCEPT
        )
        
        # Manually set invalid confidence to bypass __post_init__ validation
        invalid_review.confidence_level = 6  # Invalid: > 5
        
        is_valid, errors = self.validator.validate_confidence_and_recommendation(invalid_review)
        
        assert is_valid is False
        assert any("Confidence level 6 is outside valid range" in error for error in errors)
    
    def test_validate_confidence_and_recommendation_inconsistent(self):
        """Test validation with inconsistent recommendation and scores."""
        # High scores but reject recommendation
        inconsistent_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            criteria_scores=EnhancedReviewCriteria(
                novelty=9.0, technical_quality=9.0, clarity=8.0,
                significance=9.0, reproducibility=8.0, related_work=8.0
            ),
            confidence_level=4,
            recommendation=ReviewDecision.REJECT  # Inconsistent with high scores
        )
        
        is_valid, errors = self.validator.validate_confidence_and_recommendation(inconsistent_review)
        
        assert is_valid is False
        assert any("Reject recommendation inconsistent" in error for error in errors)
    
    def test_calibrate_scores_with_peerread(self):
        """Test PeerRead score calibration."""
        original_significance = self.sample_review.criteria_scores.significance
        
        calibrated_review = self.validator.calibrate_scores_with_peerread(self.sample_review)
        
        # Scores should be adjusted toward PeerRead means
        assert calibrated_review.criteria_scores.significance != original_significance
        
        # All scores should still be in valid range
        criteria = calibrated_review.criteria_scores
        for dim in ['novelty', 'technical_quality', 'clarity', 'significance', 'reproducibility', 'related_work']:
            score = getattr(criteria, dim)
            assert 1.0 <= score <= 10.0
    
    def test_integrate_peerread_language_patterns(self):
        """Test integration of PeerRead language patterns."""
        # Create review with minimal content
        minimal_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            executive_summary="Short.",
            detailed_strengths=[DetailedStrength(category="Technical", description="Good.")],
            detailed_weaknesses=[DetailedWeakness(category="Technical", description="Bad.")],
            questions_for_authors=[]
        )
        
        enhanced_review = self.validator.integrate_peerread_language_patterns(minimal_review)
        
        # Content should be enhanced
        assert len(enhanced_review.executive_summary) > len(minimal_review.executive_summary)
        assert len(enhanced_review.detailed_strengths[0].description) > len(minimal_review.detailed_strengths[0].description)
        assert len(enhanced_review.questions_for_authors) > 0
    
    def test_validate_complete_review_valid(self):
        """Test complete validation of a valid review."""
        is_valid, errors = self.validator.validate_complete_review(self.sample_review, self.sample_venue)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_complete_review_multiple_errors(self):
        """Test complete validation with multiple errors."""
        # Create a valid review first, then manually introduce errors
        invalid_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            executive_summary="",  # Missing summary
            detailed_strengths=[],  # No strengths
            detailed_weaknesses=[],  # No weaknesses
            technical_comments=""  # No comments
        )
        
        # Manually set invalid values to bypass __post_init__ validation
        invalid_review.criteria_scores.novelty = 11.0  # Invalid score
        invalid_review.confidence_level = 6  # Invalid confidence
        
        is_valid, errors = self.validator.validate_complete_review(invalid_review, self.sample_venue)
        
        assert is_valid is False
        assert len(errors) > 3  # Multiple validation errors
    
    def test_enhance_review_with_peerread_patterns(self):
        """Test complete enhancement with PeerRead patterns."""
        original_quality = self.sample_review.quality_score
        
        enhanced_review = self.validator.enhance_review_with_peerread_patterns(self.sample_review)
        
        # Review should be enhanced
        assert enhanced_review is not None
        # Quality score should be recalculated
        assert hasattr(enhanced_review, 'quality_score')


class TestReviewRequirementsManager:
    """Test cases for ReviewRequirementsManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ReviewRequirementsManager()
        
        self.sample_venue = EnhancedVenue(
            id="venue_001",
            name="Test Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="AI"
        )
        
        self.sample_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            criteria_scores=EnhancedReviewCriteria(
                novelty=8.0, technical_quality=8.5, clarity=7.5,
                significance=8.0, reproducibility=7.0, related_work=7.5
            ),
            confidence_level=4,
            recommendation=ReviewDecision.ACCEPT,
            executive_summary="This paper presents a comprehensive and innovative approach to the challenging problem of machine learning optimization with solid theoretical foundations and extensive experimental validation across multiple domains and application scenarios. The work demonstrates significant advances over existing methods through novel algorithmic contributions and provides valuable insights for the research community. The authors tackle an important and timely problem in the field, addressing key limitations of current approaches while introducing practical solutions that can be readily adopted by practitioners. The theoretical analysis is rigorous and the experimental evaluation is thorough, covering multiple datasets and comparison scenarios that demonstrate the effectiveness and generalizability of the proposed approach.",
            detailed_strengths=[
                DetailedStrength(category="Technical", description="The methodology is rigorous and well-designed with clear theoretical justification and comprehensive mathematical analysis that demonstrates deep understanding of the underlying principles. The algorithmic design is elegant and efficient, with careful consideration of computational complexity and practical implementation details."),
                DetailedStrength(category="Novelty", description="The approach introduces significant innovations that advance the field substantially, combining existing techniques in novel ways while introducing original algorithmic contributions. The theoretical insights are valuable and the practical improvements are substantial and measurable."),
                DetailedStrength(category="Evaluation", description="The experimental evaluation is thorough and convincing, with comprehensive comparisons against state-of-the-art baselines and detailed analysis of results across multiple datasets and scenarios. The statistical analysis is appropriate and the results are clearly presented with proper error bars and significance testing.")
            ],
            detailed_weaknesses=[
                DetailedWeakness(category="Presentation", description="Some sections could be clearer and better organized, particularly the related work section which could benefit from more structured presentation of prior approaches and clearer positioning of the current work within the broader research landscape."),
                DetailedWeakness(category="Limitations", description="The limitations discussion could be more comprehensive and detailed, addressing potential failure cases and boundary conditions more thoroughly. The authors should also discuss computational requirements and scalability limitations in more detail.")
            ],
            technical_comments="The technical approach is sound and well-executed throughout the paper, demonstrating comprehensive understanding of the problem domain and existing methodologies in the field. The implementation appears correct and efficient, with careful attention to algorithmic details, optimization considerations, and computational efficiency aspects that are crucial for practical deployment in real-world scenarios. The theoretical analysis is thorough and rigorous, providing solid mathematical foundations for the proposed methods with clear derivations and proofs that support the main claims. The experimental methodology follows established best practices in the field with appropriate statistical analysis, comprehensive evaluation protocols, and careful consideration of potential confounding factors that could affect the results. The results are convincing and well-presented with clear visualizations, detailed statistical analysis, and thorough discussion of findings and their implications for future research directions in this important area. The computational complexity analysis is adequate though could be expanded to cover worst-case scenarios and average-case performance characteristics more thoroughly. The scalability analysis demonstrates good performance across different problem sizes and configurations, showing promise for practical applications.",
            presentation_comments="The paper is generally well-written and clearly structured, making it accessible to readers familiar with the field. The figures and tables are informative and well-designed, effectively supporting the textual content and providing clear visualization of the experimental results. However, some sections could benefit from improved organization and clearer transitions between concepts, particularly in the methodology section where the flow between different algorithmic components could be enhanced.",
            questions_for_authors=["Can you discuss the computational complexity in more detail?", "How does this scale to larger datasets?", "What are the memory requirements for the proposed approach?", "How sensitive are the results to hyperparameter choices?", "Can you provide confidence intervals for the reported results?"],
            suggestions_for_improvement=["Consider adding more baseline comparisons with recent methods", "Expand the limitations discussion to cover edge cases", "Provide more detailed complexity analysis", "Include comprehensive ablation studies to better understand component contributions", "Add thorough discussion of potential negative societal impacts"]
        )
    
    def test_get_venue_requirements_top_conference(self):
        """Test getting requirements for top conference."""
        requirements = self.manager.get_venue_requirements(VenueType.TOP_CONFERENCE)
        
        assert requirements['min_word_count'] == 600
        assert requirements['max_word_count'] == 1000
        assert requirements['min_strengths'] == 3
        assert requirements['min_weaknesses'] == 2
        assert requirements['requires_questions'] is True
        assert requirements['requires_suggestions'] is True
        assert requirements['min_confidence'] == 4
    
    def test_get_venue_requirements_low_conference(self):
        """Test getting requirements for low conference."""
        requirements = self.manager.get_venue_requirements(VenueType.LOW_CONFERENCE)
        
        assert requirements['min_word_count'] == 300
        assert requirements['max_word_count'] == 600
        assert requirements['min_strengths'] == 2
        assert requirements['min_weaknesses'] == 1
        assert requirements['requires_questions'] is False
        assert requirements['requires_suggestions'] is False
        assert requirements['min_confidence'] == 2
    
    def test_check_review_meets_requirements_valid(self):
        """Test checking valid review against requirements."""
        meets_requirements, errors = self.manager.check_review_meets_requirements(
            self.sample_review, self.sample_venue
        )
        
        assert meets_requirements is True
        assert len(errors) == 0
    
    def test_check_review_meets_requirements_missing_questions(self):
        """Test checking review missing required questions."""
        review_no_questions = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            criteria_scores=self.sample_review.criteria_scores,
            confidence_level=4,
            recommendation=ReviewDecision.ACCEPT,
            executive_summary=self.sample_review.executive_summary,
            detailed_strengths=self.sample_review.detailed_strengths,
            detailed_weaknesses=self.sample_review.detailed_weaknesses,
            technical_comments=self.sample_review.technical_comments,
            questions_for_authors=[],  # Missing required questions
            suggestions_for_improvement=self.sample_review.suggestions_for_improvement
        )
        
        meets_requirements, errors = self.manager.check_review_meets_requirements(
            review_no_questions, self.sample_venue
        )
        
        assert meets_requirements is False
        assert any("requires questions for authors" in error for error in errors)
    
    def test_check_review_meets_requirements_low_confidence(self):
        """Test checking review with confidence below venue minimum."""
        low_confidence_review = StructuredReview(
            reviewer_id="reviewer_001",
            paper_id="paper_001",
            venue_id="venue_001",
            criteria_scores=self.sample_review.criteria_scores,
            confidence_level=2,  # Below minimum of 4 for top conference
            recommendation=ReviewDecision.ACCEPT,
            executive_summary=self.sample_review.executive_summary,
            detailed_strengths=self.sample_review.detailed_strengths,
            detailed_weaknesses=self.sample_review.detailed_weaknesses,
            technical_comments=self.sample_review.technical_comments,
            questions_for_authors=self.sample_review.questions_for_authors,
            suggestions_for_improvement=self.sample_review.suggestions_for_improvement
        )
        
        meets_requirements, errors = self.manager.check_review_meets_requirements(
            low_confidence_review, self.sample_venue
        )
        
        assert meets_requirements is False
        assert any("Confidence level 2 below minimum 4" in error for error in errors)


class TestPeerReadCalibration:
    """Test cases for PeerRead calibration functionality."""
    
    def test_peerread_calibration_initialization(self):
        """Test PeerRead calibration data initialization."""
        calibration = PeerReadCalibration()
        
        # Check that means are reasonable (around 3.0-3.6 on 5-point scale)
        assert 3.0 <= calibration.impact_mean <= 4.0
        assert 3.0 <= calibration.substance_mean <= 4.0
        assert 3.0 <= calibration.originality_mean <= 4.0
        assert 3.0 <= calibration.clarity_mean <= 4.0
        
        # Check that standard deviations are reasonable
        assert 0.5 <= calibration.impact_std <= 2.0
        assert 0.5 <= calibration.substance_std <= 2.0
    
    def test_peerread_dimension_mapping(self):
        """Test PeerRead dimension mapping to our system."""
        assert PeerReadDimension.IMPACT.value == "significance"
        assert PeerReadDimension.SUBSTANCE.value == "technical_quality"
        assert PeerReadDimension.SOUNDNESS_CORRECTNESS.value == "technical_quality"
        assert PeerReadDimension.ORIGINALITY.value == "novelty"
        assert PeerReadDimension.CLARITY.value == "clarity"
        assert PeerReadDimension.MEANINGFUL_COMPARISON.value == "related_work"


class TestReviewLanguagePatterns:
    """Test cases for review language patterns."""
    
    def test_language_patterns_initialization(self):
        """Test language patterns initialization."""
        patterns = ReviewLanguagePatterns()
        
        # Check that patterns are populated
        assert len(patterns.summary_starters) > 0
        assert len(patterns.technical_strengths) > 0
        assert len(patterns.novelty_strengths) > 0
        assert len(patterns.clarity_strengths) > 0
        assert len(patterns.technical_weaknesses) > 0
        assert len(patterns.common_questions) > 0
        
        # Check that patterns are reasonable
        assert "This paper presents" in patterns.summary_starters
        assert any("methodology" in strength.lower() for strength in patterns.technical_strengths)
        assert any("novel" in strength.lower() for strength in patterns.novelty_strengths)


if __name__ == "__main__":
    pytest.main([__file__])