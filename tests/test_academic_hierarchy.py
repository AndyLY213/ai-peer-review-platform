"""
Unit tests for Academic Hierarchy Management System

Tests the AcademicHierarchy class functionality including seniority level definitions,
reputation multiplier calculations, promotion eligibility, and hierarchy statistics.
"""

import pytest
from datetime import date
from typing import List

from src.enhancements.academic_hierarchy import AcademicHierarchy
from src.data.enhanced_models import (
    EnhancedResearcher, ResearcherLevel, CareerStage, FundingStatus,
    PublicationRecord, CareerMilestone
)
from src.core.exceptions import ValidationError


class TestAcademicHierarchy:
    """Test cases for AcademicHierarchy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = AcademicHierarchy()
        
        # Create sample researchers for testing
        self.grad_student = EnhancedResearcher(
            id="grad_001",
            name="Alice Graduate",
            specialty="Machine Learning",
            level=ResearcherLevel.GRADUATE_STUDENT,
            institution_tier=2,
            h_index=3,
            total_citations=25,
            years_active=2
        )
        
        self.postdoc = EnhancedResearcher(
            id="postdoc_001",
            name="Bob Postdoc",
            specialty="Natural Language Processing",
            level=ResearcherLevel.POSTDOC,
            institution_tier=1,
            h_index=8,
            total_citations=150,
            years_active=3
        )
        
        self.assistant_prof = EnhancedResearcher(
            id="asst_001",
            name="Carol Assistant",
            specialty="Computer Vision",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_tier=1,
            h_index=15,
            total_citations=400,
            years_active=6
        )
        
        self.associate_prof = EnhancedResearcher(
            id="assoc_001",
            name="David Associate",
            specialty="Robotics",
            level=ResearcherLevel.ASSOCIATE_PROF,
            institution_tier=2,
            h_index=25,
            total_citations=800,
            years_active=12
        )
        
        self.full_prof = EnhancedResearcher(
            id="full_001",
            name="Eve Full",
            specialty="Artificial Intelligence",
            level=ResearcherLevel.FULL_PROF,
            institution_tier=1,
            h_index=45,
            total_citations=2000,
            years_active=20
        )
        
        self.emeritus = EnhancedResearcher(
            id="emer_001",
            name="Frank Emeritus",
            specialty="Machine Learning",
            level=ResearcherLevel.EMERITUS,
            institution_tier=1,
            h_index=60,
            total_citations=3500,
            years_active=35
        )
    
    def test_initialization(self):
        """Test AcademicHierarchy initialization."""
        hierarchy = AcademicHierarchy()
        assert hierarchy is not None
        assert hasattr(hierarchy, 'LEVEL_DEFINITIONS')
        assert hasattr(hierarchy, 'PROMOTION_REQUIREMENTS')
    
    def test_get_level_definition(self):
        """Test getting level definitions."""
        # Test valid level
        grad_def = self.hierarchy.get_level_definition(ResearcherLevel.GRADUATE_STUDENT)
        assert grad_def['description'] == 'PhD student conducting research under supervision'
        assert grad_def['base_reputation_multiplier'] == 0.3
        assert grad_def['max_reviews_per_month'] == 2
        assert grad_def['can_be_senior_reviewer'] is False
        
        # Test Full Professor definition (requirement: 1.5x multiplier)
        full_prof_def = self.hierarchy.get_level_definition(ResearcherLevel.FULL_PROF)
        assert full_prof_def['base_reputation_multiplier'] == 1.5
        assert full_prof_def['can_be_senior_reviewer'] is True
        
        # Test Assistant Professor baseline (requirement: 1.0x multiplier)
        asst_prof_def = self.hierarchy.get_level_definition(ResearcherLevel.ASSISTANT_PROF)
        assert asst_prof_def['base_reputation_multiplier'] == 1.0
    
    def test_get_all_levels(self):
        """Test getting all levels in hierarchical order."""
        levels = self.hierarchy.get_all_levels()
        expected_order = [
            ResearcherLevel.GRADUATE_STUDENT,
            ResearcherLevel.POSTDOC,
            ResearcherLevel.ASSISTANT_PROF,
            ResearcherLevel.ASSOCIATE_PROF,
            ResearcherLevel.FULL_PROF,
            ResearcherLevel.EMERITUS
        ]
        assert levels == expected_order
        assert len(levels) == 6
    
    def test_calculate_reputation_multiplier_baseline(self):
        """Test reputation multiplier calculation for baseline cases."""
        # Test Assistant Professor baseline (should be 1.0)
        asst_multiplier = self.hierarchy.calculate_reputation_multiplier(self.assistant_prof)
        assert 0.9 <= asst_multiplier <= 1.4  # Allow for performance adjustments
        
        # Test Full Professor (should be 1.5x baseline)
        full_multiplier = self.hierarchy.calculate_reputation_multiplier(self.full_prof)
        assert full_multiplier > asst_multiplier
        assert 1.4 <= full_multiplier <= 2.0  # Should be around 1.5 with adjustments
        
        # Test Graduate Student (should be much lower)
        grad_multiplier = self.hierarchy.calculate_reputation_multiplier(self.grad_student)
        assert grad_multiplier < asst_multiplier
        assert 0.1 <= grad_multiplier <= 0.6
    
    def test_calculate_reputation_multiplier_all_levels(self):
        """Test reputation multiplier calculation for all levels."""
        researchers = [
            self.grad_student, self.postdoc, self.assistant_prof,
            self.associate_prof, self.full_prof, self.emeritus
        ]
        
        multipliers = [self.hierarchy.calculate_reputation_multiplier(r) for r in researchers]
        
        # Check that multipliers are in reasonable ranges
        assert all(0.1 <= m <= 2.0 for m in multipliers)
        
        # Check general hierarchy (with some flexibility for performance adjustments)
        # Graduate student should be lowest
        assert multipliers[0] < multipliers[2]  # Grad < Assistant
        # Full professor should be higher than assistant
        assert multipliers[4] > multipliers[2]  # Full > Assistant
    
    def test_calculate_reputation_multiplier_performance_bonus(self):
        """Test reputation multiplier with performance bonuses."""
        # Create high-performing assistant professor
        high_performer = EnhancedResearcher(
            id="high_001",
            name="High Performer",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_tier=1,  # Top tier institution
            h_index=30,  # Very high for assistant prof
            total_citations=1000,
            years_active=7
        )
        
        # Create low-performing assistant professor
        low_performer = EnhancedResearcher(
            id="low_001",
            name="Low Performer",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_tier=3,  # Lower tier institution
            h_index=3,  # Low for assistant prof
            total_citations=50,
            years_active=8
        )
        
        high_multiplier = self.hierarchy.calculate_reputation_multiplier(high_performer)
        low_multiplier = self.hierarchy.calculate_reputation_multiplier(low_performer)
        
        # High performer should have higher multiplier
        assert high_multiplier > low_multiplier
        assert high_multiplier >= 1.0  # Should be at or above baseline
        assert low_multiplier < 1.0   # Should be below baseline
    
    def test_get_max_reviews_per_month(self):
        """Test maximum reviews per month by level."""
        assert self.hierarchy.get_max_reviews_per_month(ResearcherLevel.GRADUATE_STUDENT) == 2
        assert self.hierarchy.get_max_reviews_per_month(ResearcherLevel.POSTDOC) == 3
        assert self.hierarchy.get_max_reviews_per_month(ResearcherLevel.ASSISTANT_PROF) == 4
        assert self.hierarchy.get_max_reviews_per_month(ResearcherLevel.ASSOCIATE_PROF) == 6
        assert self.hierarchy.get_max_reviews_per_month(ResearcherLevel.FULL_PROF) == 8
        assert self.hierarchy.get_max_reviews_per_month(ResearcherLevel.EMERITUS) == 3
    
    def test_get_review_weight(self):
        """Test review weight by level."""
        assert self.hierarchy.get_review_weight(ResearcherLevel.GRADUATE_STUDENT) == 0.5
        assert self.hierarchy.get_review_weight(ResearcherLevel.POSTDOC) == 0.7
        assert self.hierarchy.get_review_weight(ResearcherLevel.ASSISTANT_PROF) == 1.0
        assert self.hierarchy.get_review_weight(ResearcherLevel.ASSOCIATE_PROF) == 1.3
        assert self.hierarchy.get_review_weight(ResearcherLevel.FULL_PROF) == 1.5
        assert self.hierarchy.get_review_weight(ResearcherLevel.EMERITUS) == 1.4
    
    def test_can_be_senior_reviewer(self):
        """Test senior reviewer eligibility by level."""
        assert not self.hierarchy.can_be_senior_reviewer(ResearcherLevel.GRADUATE_STUDENT)
        assert not self.hierarchy.can_be_senior_reviewer(ResearcherLevel.POSTDOC)
        assert self.hierarchy.can_be_senior_reviewer(ResearcherLevel.ASSISTANT_PROF)
        assert self.hierarchy.can_be_senior_reviewer(ResearcherLevel.ASSOCIATE_PROF)
        assert self.hierarchy.can_be_senior_reviewer(ResearcherLevel.FULL_PROF)
        assert self.hierarchy.can_be_senior_reviewer(ResearcherLevel.EMERITUS)
    
    def test_can_mentor(self):
        """Test mentoring eligibility by level."""
        assert not self.hierarchy.can_mentor(ResearcherLevel.GRADUATE_STUDENT)
        assert self.hierarchy.can_mentor(ResearcherLevel.POSTDOC)
        assert self.hierarchy.can_mentor(ResearcherLevel.ASSISTANT_PROF)
        assert self.hierarchy.can_mentor(ResearcherLevel.ASSOCIATE_PROF)
        assert self.hierarchy.can_mentor(ResearcherLevel.FULL_PROF)
        assert self.hierarchy.can_mentor(ResearcherLevel.EMERITUS)
    
    def test_get_typical_publication_pressure(self):
        """Test publication pressure by level."""
        # Graduate students and postdocs should have high pressure
        assert self.hierarchy.get_typical_publication_pressure(ResearcherLevel.GRADUATE_STUDENT) == 0.8
        assert self.hierarchy.get_typical_publication_pressure(ResearcherLevel.POSTDOC) == 0.9
        
        # Assistant professors should have very high pressure (tenure track)
        assert self.hierarchy.get_typical_publication_pressure(ResearcherLevel.ASSISTANT_PROF) == 0.9
        
        # Senior levels should have lower pressure
        assert self.hierarchy.get_typical_publication_pressure(ResearcherLevel.ASSOCIATE_PROF) == 0.6
        assert self.hierarchy.get_typical_publication_pressure(ResearcherLevel.FULL_PROF) == 0.4
        assert self.hierarchy.get_typical_publication_pressure(ResearcherLevel.EMERITUS) == 0.1
    
    def test_is_promotion_eligible_positive_cases(self):
        """Test promotion eligibility for eligible candidates."""
        # Create a researcher eligible for promotion from postdoc to assistant prof
        eligible_postdoc = EnhancedResearcher(
            id="eligible_001",
            name="Eligible Postdoc",
            specialty="AI",
            level=ResearcherLevel.POSTDOC,
            institution_tier=1,
            h_index=8,
            total_citations=200,
            years_active=3,
            publication_history=[
                PublicationRecord(f"paper_{i}", f"Title {i}", "Venue", 2020+i, 10)
                for i in range(6)  # 6 publications
            ]
        )
        
        is_eligible, missing_reqs = self.hierarchy.is_promotion_eligible(
            eligible_postdoc, ResearcherLevel.ASSISTANT_PROF
        )
        
        assert is_eligible
        assert len(missing_reqs) == 0
    
    def test_is_promotion_eligible_negative_cases(self):
        """Test promotion eligibility for ineligible candidates."""
        # Create a researcher not eligible for promotion (insufficient experience)
        ineligible_postdoc = EnhancedResearcher(
            id="ineligible_001",
            name="Ineligible Postdoc",
            specialty="AI",
            level=ResearcherLevel.POSTDOC,
            institution_tier=1,
            h_index=2,  # Too low
            total_citations=50,
            years_active=0,  # Too few years
            publication_history=[
                PublicationRecord("paper_1", "Title 1", "Venue", 2023, 5)
            ]  # Too few publications
        )
        
        is_eligible, missing_reqs = self.hierarchy.is_promotion_eligible(
            ineligible_postdoc, ResearcherLevel.ASSISTANT_PROF
        )
        
        assert not is_eligible
        assert len(missing_reqs) > 0
        
        # Check that specific requirements are mentioned
        missing_text = " ".join(missing_reqs)
        assert "years experience" in missing_text or "publications" in missing_text or "h-index" in missing_text
    
    def test_is_promotion_eligible_wrong_level(self):
        """Test promotion eligibility from wrong starting level."""
        is_eligible, missing_reqs = self.hierarchy.is_promotion_eligible(
            self.grad_student, ResearcherLevel.FULL_PROF  # Can't skip levels
        )
        
        assert not is_eligible
        assert len(missing_reqs) > 0
        assert any("Must be" in req for req in missing_reqs)
    
    def test_get_promotion_requirements(self):
        """Test getting promotion requirements."""
        reqs = self.hierarchy.get_promotion_requirements(ResearcherLevel.ASSISTANT_PROF)
        
        assert 'from_level' in reqs
        assert 'min_years_experience' in reqs
        assert 'min_publications' in reqs
        assert 'min_h_index' in reqs
        assert 'requirements' in reqs
        
        assert reqs['from_level'] == ResearcherLevel.POSTDOC
        assert isinstance(reqs['min_years_experience'], int)
        assert isinstance(reqs['min_publications'], int)
        assert isinstance(reqs['min_h_index'], int)
        assert isinstance(reqs['requirements'], list)
    
    def test_get_promotion_requirements_invalid_level(self):
        """Test getting promotion requirements for invalid level."""
        # Graduate student is the starting level, no promotion requirements
        with pytest.raises(ValidationError):
            self.hierarchy.get_promotion_requirements(ResearcherLevel.GRADUATE_STUDENT)
    
    def test_suggest_career_progression(self):
        """Test career progression suggestions."""
        suggestion = self.hierarchy.suggest_career_progression(self.postdoc)
        
        assert 'current_level' in suggestion
        assert 'next_level' in suggestion
        assert 'is_eligible' in suggestion
        assert 'missing_requirements' in suggestion
        assert 'promotion_timeline' in suggestion
        assert 'recommendations' in suggestion
        
        assert suggestion['current_level'] == ResearcherLevel.POSTDOC.value
        assert suggestion['next_level'] == ResearcherLevel.ASSISTANT_PROF.value
        assert isinstance(suggestion['is_eligible'], bool)
        assert isinstance(suggestion['missing_requirements'], list)
        assert isinstance(suggestion['recommendations'], list)
    
    def test_suggest_career_progression_emeritus(self):
        """Test career progression suggestions for emeritus (highest level)."""
        suggestion = self.hierarchy.suggest_career_progression(self.emeritus)
        
        assert suggestion['current_level'] == ResearcherLevel.EMERITUS.value
        assert suggestion['next_level'] is None
        assert not suggestion['is_eligible']
        assert 'message' in suggestion
        assert 'highest academic level' in suggestion['message']
    
    def test_get_hierarchy_statistics(self):
        """Test hierarchy statistics calculation."""
        researchers = [
            self.grad_student, self.postdoc, self.assistant_prof,
            self.associate_prof, self.full_prof, self.emeritus
        ]
        
        stats = self.hierarchy.get_hierarchy_statistics(researchers)
        
        assert 'total_researchers' in stats
        assert 'level_distribution' in stats
        assert 'level_metrics' in stats
        assert 'hierarchy_diversity' in stats
        
        assert stats['total_researchers'] == 6
        assert stats['hierarchy_diversity'] == 6  # All levels represented
        
        # Check level distribution
        level_dist = stats['level_distribution']
        assert level_dist['Graduate Student'] == 1
        assert level_dist['Postdoc'] == 1
        assert level_dist['Assistant Prof'] == 1
        assert level_dist['Associate Prof'] == 1
        assert level_dist['Full Prof'] == 1
        assert level_dist['Emeritus'] == 1
        
        # Check level metrics structure
        level_metrics = stats['level_metrics']
        for level_name in level_dist.keys():
            assert level_name in level_metrics
            metrics = level_metrics[level_name]
            assert 'count' in metrics
            assert 'avg_h_index' in metrics
            assert 'avg_years_active' in metrics
            assert 'avg_reputation_score' in metrics
            assert 'avg_reputation_multiplier' in metrics
    
    def test_get_hierarchy_statistics_empty_levels(self):
        """Test hierarchy statistics with some empty levels."""
        researchers = [self.assistant_prof, self.full_prof]  # Only two levels
        
        stats = self.hierarchy.get_hierarchy_statistics(researchers)
        
        assert stats['total_researchers'] == 2
        assert stats['hierarchy_diversity'] == 2  # Only 2 levels represented
        
        # Check that empty levels have zero counts
        level_dist = stats['level_distribution']
        assert level_dist['Graduate Student'] == 0
        assert level_dist['Postdoc'] == 0
        assert level_dist['Assistant Prof'] == 1
        assert level_dist['Associate Prof'] == 0
        assert level_dist['Full Prof'] == 1
        assert level_dist['Emeritus'] == 0
        
        # Check that empty levels have zero metrics
        level_metrics = stats['level_metrics']
        grad_metrics = level_metrics['Graduate Student']
        assert grad_metrics['count'] == 0
        assert grad_metrics['avg_h_index'] == 0
        assert grad_metrics['avg_reputation_multiplier'] == 0
    
    def test_reputation_multiplier_bounds(self):
        """Test that reputation multipliers are within reasonable bounds."""
        # Create extreme cases
        extreme_high = EnhancedResearcher(
            id="extreme_high",
            name="Extreme High",
            specialty="AI",
            level=ResearcherLevel.FULL_PROF,
            institution_tier=1,
            h_index=100,  # Extremely high
            total_citations=10000,
            years_active=30
        )
        
        extreme_low = EnhancedResearcher(
            id="extreme_low",
            name="Extreme Low",
            specialty="AI",
            level=ResearcherLevel.GRADUATE_STUDENT,
            institution_tier=3,
            h_index=0,  # Very low
            total_citations=0,
            years_active=1
        )
        
        high_multiplier = self.hierarchy.calculate_reputation_multiplier(extreme_high)
        low_multiplier = self.hierarchy.calculate_reputation_multiplier(extreme_low)
        
        # Check bounds (should be capped at 0.1 to 2.0)
        assert 0.1 <= high_multiplier <= 2.0
        assert 0.1 <= low_multiplier <= 2.0
        
        # High should still be higher than low
        assert high_multiplier > low_multiplier
    
    def test_level_definitions_completeness(self):
        """Test that all level definitions have required fields."""
        required_fields = [
            'description', 'typical_years_experience', 'typical_h_index_range',
            'base_reputation_multiplier', 'max_reviews_per_month', 'review_weight',
            'can_be_senior_reviewer', 'can_mentor', 'typical_publication_pressure'
        ]
        
        for level in self.hierarchy.get_all_levels():
            definition = self.hierarchy.get_level_definition(level)
            for field in required_fields:
                assert field in definition, f"Missing field {field} in {level.value} definition"
    
    def test_promotion_requirements_completeness(self):
        """Test that all promotion requirements have required fields."""
        required_fields = [
            'from_level', 'min_years_experience', 'min_publications',
            'min_h_index', 'requirements'
        ]
        
        # All levels except graduate student should have promotion requirements
        promotable_levels = [
            ResearcherLevel.POSTDOC, ResearcherLevel.ASSISTANT_PROF,
            ResearcherLevel.ASSOCIATE_PROF, ResearcherLevel.FULL_PROF,
            ResearcherLevel.EMERITUS
        ]
        
        for level in promotable_levels:
            requirements = self.hierarchy.get_promotion_requirements(level)
            for field in required_fields:
                assert field in requirements, f"Missing field {field} in {level.value} promotion requirements"


if __name__ == "__main__":
    pytest.main([__file__])