"""
Unit tests for Multi-Institutional Collaboration Bonus System

Tests the collaboration incentive system for multi-institutional projects,
funding and publication success bonuses, collaborative project formation
algorithms, and collaboration bonus calculations.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch

from src.enhancements.multi_institutional_collaboration import (
    MultiInstitutionalCollaborationSystem,
    InstitutionProfile,
    CollaborationProject,
    CollaborationBonus,
    CollaborationType,
    CollaborationStatus
)
from src.core.exceptions import ValidationError


class TestInstitutionProfile:
    """Test InstitutionProfile class."""
    
    def test_institution_profile_creation(self):
        """Test creating a valid institution profile."""
        institution = InstitutionProfile(
            name="MIT",
            tier=1,
            country="USA",
            institution_type="Academic",
            research_strengths=["AI", "Robotics"],
            reputation_score=0.9
        )
        
        assert institution.name == "MIT"
        assert institution.tier == 1
        assert institution.country == "USA"
        assert institution.institution_type == "Academic"
        assert "AI" in institution.research_strengths
        assert institution.reputation_score == 0.9
    
    def test_institution_profile_validation_tier(self):
        """Test institution profile tier validation."""
        with pytest.raises(ValidationError) as exc_info:
            InstitutionProfile(name="Test", tier=0)
        assert "tier" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            InstitutionProfile(name="Test", tier=4)
        assert "tier" in str(exc_info.value)
    
    def test_institution_profile_validation_reputation(self):
        """Test institution profile reputation validation."""
        with pytest.raises(ValidationError) as exc_info:
            InstitutionProfile(name="Test", reputation_score=-0.1)
        assert "reputation_score" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            InstitutionProfile(name="Test", reputation_score=1.1)
        assert "reputation_score" in str(exc_info.value)


class TestCollaborationProject:
    """Test CollaborationProject class."""
    
    def test_collaboration_project_creation(self):
        """Test creating a valid collaboration project."""
        project = CollaborationProject(
            title="AI Research Project",
            description="Joint AI research",
            participating_institutions=["inst1", "inst2"],
            participating_researchers=["res1", "res2"],
            lead_institution="inst1",
            lead_researcher="res1",
            research_areas=["AI", "ML"]
        )
        
        assert project.title == "AI Research Project"
        assert len(project.participating_institutions) == 2
        assert project.lead_institution == "inst1"
        assert project.collaboration_type == CollaborationType.BILATERAL
    
    def test_collaboration_project_validation_institutions(self):
        """Test collaboration project institution validation."""
        with pytest.raises(ValidationError) as exc_info:
            CollaborationProject(
                title="Test",
                participating_institutions=["inst1"]  # Only one institution
            )
        assert "participating_institutions" in str(exc_info.value)
    
    def test_collaboration_project_validation_lead_institution(self):
        """Test collaboration project lead institution validation."""
        with pytest.raises(ValidationError) as exc_info:
            CollaborationProject(
                title="Test",
                participating_institutions=["inst1", "inst2"],
                lead_institution="inst3"  # Not in participating institutions
            )
        assert "lead_institution" in str(exc_info.value)
    
    def test_get_collaboration_complexity(self):
        """Test collaboration complexity calculation."""
        # Simple bilateral project
        project = CollaborationProject(
            participating_institutions=["inst1", "inst2"],
            collaboration_type=CollaborationType.BILATERAL,
            research_areas=["AI"]
        )
        complexity = project.get_collaboration_complexity()
        assert 0.0 <= complexity <= 1.0
        
        # Complex international project
        project = CollaborationProject(
            participating_institutions=["inst1", "inst2", "inst3", "inst4"],
            collaboration_type=CollaborationType.INTERNATIONAL,
            research_areas=["AI", "Robotics", "NLP", "Vision", "Ethics"]
        )
        complex_score = project.get_collaboration_complexity()
        assert complex_score > complexity
        assert 0.0 <= complex_score <= 1.0
    
    def test_is_active(self):
        """Test project active status check."""
        project = CollaborationProject(
            participating_institutions=["inst1", "inst2"],
            status=CollaborationStatus.PROPOSED
        )
        assert not project.is_active()
        
        project.status = CollaborationStatus.ACTIVE
        assert project.is_active()
    
    def test_get_duration_months(self):
        """Test project duration calculation."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 7, 1)  # 6 months
        
        project = CollaborationProject(
            participating_institutions=["inst1", "inst2"],
            start_date=start_date,
            planned_end_date=end_date
        )
        
        duration = project.get_duration_months()
        assert duration == 6


class TestCollaborationBonus:
    """Test CollaborationBonus class."""
    
    def test_collaboration_bonus_creation(self):
        """Test creating a collaboration bonus."""
        bonus = CollaborationBonus(
            project_id="proj1",
            researcher_id="res1",
            institution_id="inst1",
            funding_success_bonus=0.15,
            publication_success_bonus=0.10,
            reputation_bonus=0.05,
            network_expansion_bonus=0.02
        )
        
        assert bonus.project_id == "proj1"
        assert bonus.funding_success_bonus == 0.15
        assert bonus.publication_success_bonus == 0.10
        assert bonus.is_active
    
    def test_get_total_bonus_value(self):
        """Test total bonus value calculation."""
        bonus = CollaborationBonus(
            funding_success_bonus=0.15,
            publication_success_bonus=0.10,
            reputation_bonus=0.05,
            network_expansion_bonus=0.02
        )
        
        total = bonus.get_total_bonus_value()
        assert total == 0.32
    
    def test_is_expired(self):
        """Test bonus expiry check."""
        # Non-expiring bonus
        bonus = CollaborationBonus()
        assert not bonus.is_expired()
        
        # Expired bonus
        bonus.expiry_date = date.today() - timedelta(days=1)
        assert bonus.is_expired()
        
        # Future expiry
        bonus.expiry_date = date.today() + timedelta(days=1)
        assert not bonus.is_expired()


class TestMultiInstitutionalCollaborationSystem:
    """Test MultiInstitutionalCollaborationSystem class."""
    
    @pytest.fixture
    def collaboration_system(self):
        """Create a collaboration system for testing."""
        return MultiInstitutionalCollaborationSystem()
    
    @pytest.fixture
    def sample_institutions(self, collaboration_system):
        """Create sample institutions for testing."""
        institutions = []
        
        # Academic institution
        mit = InstitutionProfile(
            name="MIT",
            tier=1,
            country="USA",
            institution_type="Academic",
            research_strengths=["AI", "Robotics", "Computer Science"],
            reputation_score=0.95
        )
        institutions.append(mit)
        collaboration_system.register_institution(mit)
        
        # Industry institution
        google = InstitutionProfile(
            name="Google Research",
            tier=1,
            country="USA",
            institution_type="Industry",
            research_strengths=["AI", "Machine Learning", "NLP"],
            reputation_score=0.90
        )
        institutions.append(google)
        collaboration_system.register_institution(google)
        
        # International institution
        oxford = InstitutionProfile(
            name="Oxford University",
            tier=1,
            country="UK",
            institution_type="Academic",
            research_strengths=["AI", "Philosophy", "Ethics"],
            reputation_score=0.92
        )
        institutions.append(oxford)
        collaboration_system.register_institution(oxford)
        
        return institutions
    
    def test_register_institution(self, collaboration_system):
        """Test institution registration."""
        institution = InstitutionProfile(name="Test University", tier=2)
        inst_id = collaboration_system.register_institution(institution)
        
        assert inst_id == institution.institution_id
        assert collaboration_system.get_institution(inst_id) == institution
    
    def test_create_collaboration_project(self, collaboration_system, sample_institutions):
        """Test collaboration project creation."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="AI Ethics Research",
            description="Joint research on AI ethics",
            participating_institutions=inst_ids,
            participating_researchers=["res1", "res2"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI", "Ethics"],
            total_funding=500000
        )
        
        project = collaboration_system.projects[project_id]
        assert project.title == "AI Ethics Research"
        assert len(project.participating_institutions) == 2
        assert project.total_funding == 500000
        assert project.status == CollaborationStatus.PROPOSED
    
    def test_create_collaboration_project_invalid_institution(self, collaboration_system):
        """Test collaboration project creation with invalid institution."""
        with pytest.raises(ValidationError) as exc_info:
            collaboration_system.create_collaboration_project(
                title="Test Project",
                description="Test",
                participating_institutions=["invalid_id"],
                participating_researchers=["res1"],
                lead_institution="invalid_id",
                lead_researcher="res1",
                research_areas=["AI"]
            )
        assert "institution_id" in str(exc_info.value)
    
    def test_determine_collaboration_type(self, collaboration_system, sample_institutions):
        """Test collaboration type determination."""
        # Bilateral (same country, both academic)
        bilateral_ids = [sample_institutions[0].institution_id, sample_institutions[0].institution_id]
        # Create a second academic US institution for proper bilateral test
        us_academic = InstitutionProfile(
            name="Stanford", tier=1, country="USA", institution_type="Academic"
        )
        collaboration_system.register_institution(us_academic)
        bilateral_ids = [sample_institutions[0].institution_id, us_academic.institution_id]
        
        collab_type = collaboration_system._determine_collaboration_type(bilateral_ids)
        assert collab_type == CollaborationType.BILATERAL
        
        # International (different countries)
        international_ids = [sample_institutions[0].institution_id, sample_institutions[2].institution_id]
        collab_type = collaboration_system._determine_collaboration_type(international_ids)
        assert collab_type == CollaborationType.INTERNATIONAL
        
        # Industry-Academic
        industry_academic_ids = [sample_institutions[0].institution_id, sample_institutions[1].institution_id]
        collab_type = collaboration_system._determine_collaboration_type(industry_academic_ids)
        assert collab_type == CollaborationType.INDUSTRY_ACADEMIC
    
    def test_activate_project(self, collaboration_system, sample_institutions):
        """Test project activation and bonus application."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1", "res2"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        # Activate project
        success = collaboration_system.activate_project(project_id)
        assert success
        
        project = collaboration_system.projects[project_id]
        assert project.status == CollaborationStatus.ACTIVE
        
        # Check that bonuses were created
        researcher_bonuses = collaboration_system.get_researcher_bonuses("res1")
        assert len(researcher_bonuses) > 0
        assert researcher_bonuses[0].funding_success_bonus > 0
    
    def test_activate_project_invalid_status(self, collaboration_system, sample_institutions):
        """Test activating project with invalid status."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        # Activate once
        collaboration_system.activate_project(project_id)
        
        # Try to activate again
        success = collaboration_system.activate_project(project_id)
        assert not success
    
    def test_calculate_collaboration_bonus(self, collaboration_system, sample_institutions):
        """Test collaboration bonus calculation."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project = CollaborationProject(
            title="Test Project",
            participating_institutions=inst_ids,
            participating_researchers=["res1", "res2"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            collaboration_type=CollaborationType.INDUSTRY_ACADEMIC,
            research_areas=["AI"]
        )
        
        complexity_score = project.get_collaboration_complexity()
        bonus = collaboration_system._calculate_collaboration_bonus(project, "res1", complexity_score)
        
        assert bonus.funding_success_bonus > collaboration_system.base_funding_bonus
        assert bonus.publication_success_bonus > collaboration_system.base_publication_bonus
        assert bonus.researcher_id == "res1"
        assert bonus.project_id == project.project_id
        
        # Lead researcher should get leadership bonus
        lead_bonus = collaboration_system._calculate_collaboration_bonus(project, "res1", complexity_score)
        non_lead_bonus = collaboration_system._calculate_collaboration_bonus(project, "res2", complexity_score)
        
        assert lead_bonus.funding_success_bonus >= non_lead_bonus.funding_success_bonus
    
    def test_get_researcher_bonuses(self, collaboration_system, sample_institutions):
        """Test getting researcher bonuses."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1", "res2"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id)
        
        # Get bonuses for researcher
        bonuses = collaboration_system.get_researcher_bonuses("res1")
        assert len(bonuses) == 1
        assert bonuses[0].researcher_id == "res1"
        
        # Test active_only parameter
        bonuses[0].is_active = False
        active_bonuses = collaboration_system.get_researcher_bonuses("res1", active_only=True)
        assert len(active_bonuses) == 0
        
        all_bonuses = collaboration_system.get_researcher_bonuses("res1", active_only=False)
        assert len(all_bonuses) == 1
    
    def test_calculate_funding_success_multiplier(self, collaboration_system, sample_institutions):
        """Test funding success multiplier calculation."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id)
        
        multiplier = collaboration_system.calculate_funding_success_multiplier("res1")
        assert multiplier > 1.0  # Should have bonus
        
        # Researcher with no bonuses
        no_bonus_multiplier = collaboration_system.calculate_funding_success_multiplier("res_no_bonus")
        assert no_bonus_multiplier == 1.0
    
    def test_calculate_publication_success_multiplier(self, collaboration_system, sample_institutions):
        """Test publication success multiplier calculation."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id)
        
        multiplier = collaboration_system.calculate_publication_success_multiplier("res1")
        assert multiplier > 1.0  # Should have bonus
    
    def test_suggest_collaboration_partners(self, collaboration_system, sample_institutions):
        """Test collaboration partner suggestions."""
        suggestions = collaboration_system.suggest_collaboration_partners(
            researcher_id="res1",
            research_areas=["AI", "Machine Learning"],
            max_suggestions=3
        )
        
        assert len(suggestions) <= 3
        for inst_id, score in suggestions:
            assert inst_id in collaboration_system.institutions
            assert 0.0 <= score <= 1.0
        
        # Should be sorted by compatibility score
        if len(suggestions) > 1:
            assert suggestions[0][1] >= suggestions[1][1]
    
    def test_complete_project(self, collaboration_system, sample_institutions):
        """Test project completion."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id)
        
        outcomes = {
            "publications": ["paper1", "paper2"],
            "patents": ["patent1"],
            "other_outcomes": ["award1"]
        }
        
        success = collaboration_system.complete_project(project_id, outcomes)
        assert success
        
        project = collaboration_system.projects[project_id]
        assert project.status == CollaborationStatus.COMPLETED
        assert len(project.publications) == 2
        assert len(project.patents) == 1
        assert project.actual_end_date is not None
    
    def test_complete_project_invalid_status(self, collaboration_system, sample_institutions):
        """Test completing project with invalid status."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        # Try to complete without activating
        success = collaboration_system.complete_project(project_id, {})
        assert not success
    
    def test_get_collaboration_statistics(self, collaboration_system, sample_institutions):
        """Test collaboration statistics."""
        inst_ids = [inst.institution_id for inst in sample_institutions[:2]]
        
        # Create and activate a project
        project_id = collaboration_system.create_collaboration_project(
            title="Test Project",
            description="Test",
            participating_institutions=inst_ids,
            participating_researchers=["res1"],
            lead_institution=inst_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id)
        
        stats = collaboration_system.get_collaboration_statistics()
        
        assert stats["total_institutions"] == len(sample_institutions)
        assert stats["total_projects"] == 1
        assert stats["active_projects"] == 1
        assert stats["completed_projects"] == 0
        assert stats["total_bonuses"] > 0
        assert stats["active_bonuses"] > 0
        assert "collaboration_type_distribution" in stats
        assert stats["average_funding_bonus"] > 0
        assert stats["average_publication_bonus"] > 0
    
    def test_international_collaboration_bonus(self, collaboration_system, sample_institutions):
        """Test that international collaborations get additional bonuses."""
        # Create international collaboration (USA + UK)
        international_ids = [sample_institutions[0].institution_id, sample_institutions[2].institution_id]
        
        project_id = collaboration_system.create_collaboration_project(
            title="International Project",
            description="Test",
            participating_institutions=international_ids,
            participating_researchers=["res1"],
            lead_institution=international_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id)
        
        bonuses = collaboration_system.get_researcher_bonuses("res1")
        international_bonus = bonuses[0]
        
        # Create domestic collaboration for comparison
        domestic_ids = [sample_institutions[0].institution_id, sample_institutions[1].institution_id]
        
        project_id_2 = collaboration_system.create_collaboration_project(
            title="Domestic Project",
            description="Test",
            participating_institutions=domestic_ids,
            participating_researchers=["res2"],
            lead_institution=domestic_ids[0],
            lead_researcher="res2",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id_2)
        
        domestic_bonuses = collaboration_system.get_researcher_bonuses("res2")
        domestic_bonus = domestic_bonuses[0]
        
        # International collaboration should have higher bonus
        assert international_bonus.funding_success_bonus > domestic_bonus.funding_success_bonus
    
    def test_industry_academic_collaboration_bonus(self, collaboration_system, sample_institutions):
        """Test that industry-academic collaborations get additional bonuses."""
        # Create industry-academic collaboration
        industry_academic_ids = [sample_institutions[0].institution_id, sample_institutions[1].institution_id]
        
        project_id = collaboration_system.create_collaboration_project(
            title="Industry-Academic Project",
            description="Test",
            participating_institutions=industry_academic_ids,
            participating_researchers=["res1"],
            lead_institution=industry_academic_ids[0],
            lead_researcher="res1",
            research_areas=["AI"]
        )
        
        collaboration_system.activate_project(project_id)
        
        bonuses = collaboration_system.get_researcher_bonuses("res1")
        industry_bonus = bonuses[0]
        
        # Should have industry collaboration bonus
        assert "type_bonus" in industry_bonus.bonus_factors
        assert industry_bonus.bonus_factors["type_bonus"] > 0
        assert industry_bonus.funding_success_bonus > collaboration_system.base_funding_bonus


if __name__ == "__main__":
    pytest.main([__file__])