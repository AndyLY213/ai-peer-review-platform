#!/usr/bin/env python3
"""
Simple test script to verify enhanced models work correctly.
"""

import sys
sys.path.append('.')

from src.data.enhanced_models import (
    EnhancedReviewCriteria, StructuredReview, EnhancedResearcher, EnhancedVenue,
    ResearcherLevel, VenueType, ReviewDecision, DetailedStrength, DetailedWeakness
)
from src.core.exceptions import ValidationError

def test_enhanced_review_criteria():
    """Test EnhancedReviewCriteria functionality."""
    print("Testing EnhancedReviewCriteria...")
    
    # Test valid creation
    criteria = EnhancedReviewCriteria(
        novelty=8.0,
        technical_quality=7.5,
        clarity=6.0,
        significance=9.0,
        reproducibility=5.5,
        related_work=7.0
    )
    
    assert criteria.novelty == 8.0
    assert criteria.technical_quality == 7.5
    
    # Test average calculation
    avg = criteria.get_average_score()
    expected = (8.0 + 7.5 + 6.0 + 9.0 + 5.5 + 7.0) / 6.0
    assert abs(avg - expected) < 0.001
    
    # Test validation
    try:
        EnhancedReviewCriteria(novelty=11.0)  # Should fail
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected
    
    print("✓ EnhancedReviewCriteria tests passed")

def test_structured_review():
    """Test StructuredReview functionality."""
    print("Testing StructuredReview...")
    
    review = StructuredReview(
        reviewer_id="reviewer_1",
        paper_id="paper_1",
        venue_id="venue_1",
        executive_summary="This is a good paper.",
        detailed_strengths=[
            DetailedStrength(category="Technical", description="Strong methodology"),
            DetailedStrength(category="Novelty", description="Novel approach")
        ],
        detailed_weaknesses=[
            DetailedWeakness(category="Clarity", description="Some unclear sections")
        ]
    )
    
    assert review.reviewer_id == "reviewer_1"
    assert len(review.detailed_strengths) == 2
    assert len(review.detailed_weaknesses) == 1
    assert review.review_length > 0
    
    # Test venue requirements
    assert review.meets_venue_requirements(min_word_count=5)
    
    print("✓ StructuredReview tests passed")

def test_enhanced_researcher():
    """Test EnhancedResearcher functionality."""
    print("Testing EnhancedResearcher...")
    
    researcher = EnhancedResearcher(
        id="researcher_1",
        name="Dr. Jane Smith",
        specialty="Machine Learning",
        level=ResearcherLevel.ASSOCIATE_PROF,
        h_index=15,
        total_citations=300,
        years_active=8
    )
    
    assert researcher.name == "Dr. Jane Smith"
    assert researcher.level == ResearcherLevel.ASSOCIATE_PROF
    assert researcher.reputation_score > 0
    
    # Test reputation multiplier
    multiplier = researcher.get_reputation_multiplier()
    assert multiplier > 1.0  # Associate prof should have > 1.0 multiplier
    
    # Test review capacity
    assert researcher.can_accept_review()
    
    print("✓ EnhancedResearcher tests passed")

def test_enhanced_venue():
    """Test EnhancedVenue functionality."""
    print("Testing EnhancedVenue...")
    
    venue = EnhancedVenue(
        id="venue_1",
        name="Top AI Conference",
        venue_type=VenueType.TOP_CONFERENCE,
        field="Artificial Intelligence"
    )
    
    assert venue.name == "Top AI Conference"
    assert venue.venue_type == VenueType.TOP_CONFERENCE
    assert venue.acceptance_rate == 0.05  # Should be set by defaults
    assert venue.prestige_score == 9
    
    # Test reviewer criteria
    good_researcher = EnhancedResearcher(
        id="researcher_1",
        name="Dr. Smith",
        specialty="AI",
        level=ResearcherLevel.FULL_PROF,
        h_index=20,
        years_active=15,
        institution_tier=1
    )
    
    assert venue.meets_reviewer_criteria(good_researcher)
    
    print("✓ EnhancedVenue tests passed")

def test_serialization():
    """Test serialization/deserialization."""
    print("Testing serialization...")
    
    # Test researcher serialization
    researcher = EnhancedResearcher(
        id="researcher_1",
        name="Dr. Smith",
        specialty="AI",
        level=ResearcherLevel.FULL_PROF
    )
    
    researcher_dict = researcher.to_dict()
    assert isinstance(researcher_dict, dict)
    assert researcher_dict['name'] == "Dr. Smith"
    assert researcher_dict['level'] == "Full Prof"
    
    restored = EnhancedResearcher.from_dict(researcher_dict)
    assert restored.name == researcher.name
    assert restored.level == researcher.level
    
    print("✓ Serialization tests passed")

def main():
    """Run all tests."""
    print("Running enhanced models tests...\n")
    
    try:
        test_enhanced_review_criteria()
        test_structured_review()
        test_enhanced_researcher()
        test_enhanced_venue()
        test_serialization()
        
        print("\n🎉 All enhanced models tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)