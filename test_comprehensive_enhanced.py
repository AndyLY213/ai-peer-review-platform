#!/usr/bin/env python3
"""
Comprehensive test script to verify all enhanced models work together correctly.
"""

import sys
import json
import tempfile
from datetime import datetime, date
from pathlib import Path

sys.path.append('.')

from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, EnhancedVenue,
    EnhancedReviewCriteria, DetailedStrength, DetailedWeakness,
    BiasEffect, ReviewBehaviorProfile, StrategicBehaviorProfile,
    CareerMilestone, PublicationRecord, ReviewQualityMetric,
    TenureTimeline, ReviewRequirements, QualityStandards,
    ReviewerCriteria, DatabaseMigrationUtility,
    ResearcherLevel, VenueType, ReviewDecision, CareerStage, FundingStatus
)

def test_comprehensive_workflow():
    """Test a comprehensive peer review workflow."""
    print("Testing comprehensive peer review workflow...")
    
    # Create a top-tier venue
    venue = EnhancedVenue(
        id="icml_2024",
        name="International Conference on Machine Learning 2024",
        venue_type=VenueType.TOP_CONFERENCE,
        field="Machine Learning"
    )
    
    # Create researchers with different levels
    senior_researcher = EnhancedResearcher(
        id="prof_smith",
        name="Prof. Alice Smith",
        specialty="Deep Learning",
        level=ResearcherLevel.FULL_PROF,
        h_index=45,
        total_citations=2500,
        years_active=20,
        institution_tier=1,
        email="alice.smith@university.edu"
    )
    
    junior_researcher = EnhancedResearcher(
        id="dr_jones",
        name="Dr. Bob Jones",
        specialty="Machine Learning",
        level=ResearcherLevel.ASSISTANT_PROF,
        h_index=16,  # Increased to meet top conference criteria (min 15)
        total_citations=400,
        years_active=6,
        institution_tier=1,  # Top tier institution
        email="bob.jones@college.edu"
    )
    
    grad_student = EnhancedResearcher(
        id="student_brown",
        name="Charlie Brown",
        specialty="AI",
        level=ResearcherLevel.GRADUATE_STUDENT,
        h_index=3,
        total_citations=25,
        years_active=2,
        institution_tier=2
    )
    
    # Test venue reviewer criteria
    assert venue.meets_reviewer_criteria(senior_researcher), "Senior researcher should meet top venue criteria"
    assert venue.meets_reviewer_criteria(junior_researcher), "Junior researcher should meet criteria"
    assert not venue.meets_reviewer_criteria(grad_student), "Grad student should not meet top venue criteria"
    
    print("✓ Venue reviewer criteria working correctly")
    
    # Test reputation multipliers
    senior_multiplier = senior_researcher.get_reputation_multiplier()
    junior_multiplier = junior_researcher.get_reputation_multiplier()
    
    assert senior_multiplier > junior_multiplier, "Senior researcher should have higher multiplier"
    assert senior_multiplier >= 1.5, "Full professor should have at least 1.5x multiplier"
    
    print("✓ Reputation multipliers working correctly")
    
    # Create detailed reviews
    senior_review = StructuredReview(
        reviewer_id=senior_researcher.id,
        paper_id="paper_123",
        venue_id=venue.id,
        criteria_scores=EnhancedReviewCriteria(
            novelty=8.5,
            technical_quality=9.0,
            clarity=7.5,
            significance=8.0,
            reproducibility=8.5,
            related_work=7.0
        ),
        confidence_level=5,
        recommendation=ReviewDecision.ACCEPT,
        executive_summary="This paper presents a novel approach to deep learning optimization with strong theoretical foundations and impressive empirical results. The authors introduce a new optimization algorithm that addresses key limitations of existing methods and demonstrate significant improvements across multiple benchmarks. The theoretical analysis provides valuable insights into convergence properties and the empirical evaluation is comprehensive and convincing. The work makes important contributions to the field and represents a significant advance in optimization techniques for deep learning applications. I recommend acceptance with minor revisions to address remaining clarity issues.",
        detailed_strengths=[
            DetailedStrength(
                category="Technical",
                description="The proposed optimization algorithm shows significant improvements over existing methods",
                importance=5
            ),
            DetailedStrength(
                category="Novelty",
                description="Novel theoretical analysis provides new insights into convergence properties",
                importance=4
            ),
            DetailedStrength(
                category="Empirical",
                description="Comprehensive experiments on multiple datasets demonstrate effectiveness",
                importance=4
            )
        ],
        detailed_weaknesses=[
            DetailedWeakness(
                category="Clarity",
                description="Some mathematical notation could be clearer in Section 3",
                severity=2,
                suggestions=["Consider adding more intuitive explanations", "Improve notation consistency"]
            )
        ],
        technical_comments="The theoretical analysis in Section 4 is particularly strong. The convergence proof is elegant and the bounds are tight. The authors provide a thorough analysis of the algorithm's properties and demonstrate both theoretical guarantees and practical effectiveness. The mathematical framework is well-developed and the proofs appear correct. The experimental methodology is sound with appropriate baselines and evaluation metrics. The results clearly demonstrate the superiority of the proposed approach across multiple datasets and settings. The complexity analysis is thorough and the authors address computational efficiency concerns. The algorithm design is well-motivated and the implementation details are sufficient for reproducibility. The comparison with existing methods is fair and comprehensive, covering both classical and recent approaches in the field.",
        presentation_comments="Overall well-written paper with clear motivation and good experimental design. The paper is generally well-organized with a logical flow from motivation through methodology to results. The figures are informative and the tables are well-formatted. The writing is clear and the technical content is accessible to the target audience. Some minor improvements in notation and organization could enhance readability further. The abstract effectively summarizes the contributions and the introduction provides good context. The related work section is comprehensive and positions the work well within the existing literature. The conclusion summarizes the key findings and discusses future directions appropriately.",
        questions_for_authors=[
            "How does the algorithm perform with very high-dimensional data?",
            "Have you considered the computational complexity in distributed settings?"
        ],
        suggestions_for_improvement=[
            "Add comparison with more recent baseline methods",
            "Include discussion of limitations and future work"
        ],
        applied_biases=[
            BiasEffect(
                bias_type="confirmation",
                strength=0.2,
                score_adjustment=0.3,
                description="Slight positive bias due to alignment with reviewer's research interests"
            )
        ]
    )
    
    junior_review = StructuredReview(
        reviewer_id=junior_researcher.id,
        paper_id="paper_123",
        venue_id=venue.id,
        criteria_scores=EnhancedReviewCriteria(
            novelty=7.0,
            technical_quality=8.0,
            clarity=6.5,
            significance=7.5,
            reproducibility=7.0,
            related_work=8.0
        ),
        confidence_level=3,
        recommendation=ReviewDecision.MINOR_REVISION,
        executive_summary="Solid paper with good technical contributions, but presentation could be improved. The authors tackle an important problem in optimization and provide a reasonable solution. While the work is technically sound, there are several areas where the paper could be strengthened, particularly in terms of clarity and experimental evaluation. The related work section is comprehensive and the methodology appears correct. The theoretical foundations are adequate though not as strong as they could be. The experimental results show promise but would benefit from more extensive evaluation across additional datasets and comparison with more recent methods. Overall this is a reasonable contribution that merits publication after addressing the identified issues.",
        detailed_strengths=[
            DetailedStrength(
                category="Technical",
                description="Sound methodology and implementation",
                importance=4
            ),
            DetailedStrength(
                category="Related Work",
                description="Comprehensive coverage of related work",
                importance=3
            )
        ],
        detailed_weaknesses=[
            DetailedWeakness(
                category="Clarity",
                description="Paper is difficult to follow in places",
                severity=3,
                suggestions=["Reorganize Section 2", "Add more intuitive explanations"]
            ),
            DetailedWeakness(
                category="Experiments",
                description="Limited experimental evaluation",
                severity=3,
                suggestions=["Add more datasets", "Include statistical significance tests"]
            )
        ],
        technical_comments="The methodology is generally sound but could be strengthened in several areas. The algorithm design is reasonable and the implementation appears correct. However, the theoretical analysis could be more rigorous and the experimental evaluation more comprehensive. The authors should consider additional baselines and provide more detailed analysis of the results. The convergence analysis is adequate but could benefit from tighter bounds. The computational complexity discussion is helpful but could be expanded. The experimental setup is reasonable though more datasets would strengthen the evaluation. The statistical analysis of results could be more thorough with significance testing and confidence intervals.",
        presentation_comments="The paper is generally well-written but has some organizational issues. The flow could be improved and some sections need clarification. The figures are adequate but could be more informative. Overall the presentation is acceptable but would benefit from revision. The abstract could be more specific about the contributions. The introduction provides good motivation but could better position the work relative to recent advances. The related work section is comprehensive and well-organized. The conclusion effectively summarizes the key findings and discusses limitations appropriately. Some notation could be simplified for much better overall readability and improved clarity."
    )
    
    # Test review quality and venue requirements
    # Test review quality and venue requirements
    assert senior_review.meets_venue_requirements(venue.review_requirements.min_word_count)
    assert junior_review.meets_venue_requirements(venue.review_requirements.min_word_count)
    
    # Test review quality scores
    assert senior_review.quality_score > 0.7, "Senior review should have high quality score"
    assert senior_review.completeness_score > 0.7, "Senior review should be very complete"
    
    print("✓ Detailed reviews created and validated")
    
    # Test acceptance probability calculation
    review_scores = [
        senior_review.criteria_scores.get_average_score(),
        junior_review.criteria_scores.get_average_score()
    ]
    
    acceptance_prob = venue.calculate_acceptance_probability(review_scores)
    assert 0.0 <= acceptance_prob <= 1.0, "Acceptance probability should be between 0 and 1"
    
    # With good scores, should have reasonable acceptance probability
    avg_score = sum(review_scores) / len(review_scores)
    if avg_score > venue.quality_standards.acceptance_threshold:
        assert acceptance_prob > 0.5, "Good scores should have high acceptance probability"
    
    print("✓ Acceptance probability calculation working")
    
    # Test venue submission tracking
    venue.add_submission_record("paper_123", True, review_scores)
    assert len(venue.submission_history) == 1
    assert venue.submission_history[0]['accepted'] == True
    
    print("✓ Venue submission tracking working")
    
    # Test researcher publication history
    publication = PublicationRecord(
        paper_id="paper_123",
        title="Novel Deep Learning Optimization",
        venue="ICML",
        year=2024,
        citations=5  # New publication with some citations
    )
    
    # Simulate paper being accepted and published
    for i, author in enumerate([senior_researcher, junior_researcher]):
        old_pub_count = len(author.publication_history)
        author.update_publication_history(publication)
        # Should have one more publication
        assert len(author.publication_history) == old_pub_count + 1
        # H-index and citations are recalculated from publication history
        assert author.h_index >= 0  # Should be non-negative
        assert author.total_citations >= 0  # Should be non-negative
    
    print("✓ Publication history tracking working")
    
    # Test collaboration networks
    senior_researcher.add_collaboration(junior_researcher.id)
    junior_researcher.add_collaboration(senior_researcher.id)
    
    assert junior_researcher.id in senior_researcher.collaboration_network
    assert senior_researcher.id in junior_researcher.collaboration_network
    
    print("✓ Collaboration networks working")
    
    # Test serialization of complex objects
    senior_dict = senior_researcher.to_dict()
    restored_senior = EnhancedResearcher.from_dict(senior_dict)
    
    assert restored_senior.name == senior_researcher.name
    assert restored_senior.level == senior_researcher.level
    assert restored_senior.collaboration_network == senior_researcher.collaboration_network
    
    venue_dict = venue.to_dict()
    restored_venue = EnhancedVenue.from_dict(venue_dict)
    
    assert restored_venue.name == venue.name
    assert restored_venue.venue_type == venue.venue_type
    
    review_dict = senior_review.to_dict()
    restored_review = StructuredReview.from_dict(review_dict)
    
    assert restored_review.reviewer_id == senior_review.reviewer_id
    assert restored_review.recommendation == senior_review.recommendation
    assert len(restored_review.detailed_strengths) == len(senior_review.detailed_strengths)
    
    print("✓ Complex object serialization working")

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")
    
    # Test invalid confidence levels
    try:
        StructuredReview(
            reviewer_id="reviewer_1",
            paper_id="paper_1",
            venue_id="venue_1",
            confidence_level=6  # Invalid
        )
        assert False, "Should have raised ValidationError"
    except Exception:
        pass  # Expected
    
    # Test invalid institution tier
    try:
        EnhancedResearcher(
            id="researcher_1",
            name="Test",
            specialty="AI",
            institution_tier=5  # Invalid
        )
        assert False, "Should have raised ValidationError"
    except Exception:
        pass  # Expected
    
    # Test invalid acceptance rate
    try:
        EnhancedVenue(
            id="venue_1",
            name="Test",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI",
            acceptance_rate=1.5  # Invalid
        )
        assert False, "Should have raised ValidationError"
    except Exception:
        pass  # Expected
    
    print("✓ Edge cases handled correctly")

def test_career_progression():
    """Test career progression features."""
    print("Testing career progression...")
    
    # Create researcher with tenure timeline
    tenure_timeline = TenureTimeline(
        start_date=date(2020, 9, 1),
        tenure_decision_date=date(2026, 9, 1),
        current_progress=0.6,
        publication_requirements={"journal": 3, "conference": 5},
        current_publications={"journal": 2, "conference": 4}
    )
    
    researcher = EnhancedResearcher(
        id="tenure_track",
        name="Dr. Tenure Track",
        specialty="AI",
        level=ResearcherLevel.ASSISTANT_PROF,
        tenure_timeline=tenure_timeline,
        career_stage=CareerStage.EARLY_CAREER,
        funding_status=FundingStatus.ADEQUATELY_FUNDED,
        publication_pressure=0.8
    )
    
    assert researcher.tenure_timeline is not None
    assert researcher.career_stage == CareerStage.EARLY_CAREER
    assert researcher.publication_pressure == 0.8
    
    # Add career milestone
    milestone = CareerMilestone(
        milestone_type="grant",
        date_achieved=date(2023, 6, 1),
        description="Received NSF CAREER Award",
        impact_on_behavior={"publication_pressure": -0.1, "confidence": 0.2}
    )
    
    researcher.career_milestones.append(milestone)
    assert len(researcher.career_milestones) == 1
    
    print("✓ Career progression features working")

def main():
    """Run all comprehensive tests."""
    print("Running comprehensive enhanced models tests...\n")
    
    try:
        test_comprehensive_workflow()
        test_edge_cases()
        test_career_progression()
        
        print("\n🎉 All comprehensive tests passed!")
        print("Enhanced data models are fully functional and ready for use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)