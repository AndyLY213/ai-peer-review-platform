"""
Unit tests for Career Transition Management System.

Tests the CareerTransitionManager class functionality including:
- Career transition modeling for academic-industry transitions
- Different incentive structures across career paths
- Transition probability calculations based on researcher profiles
- Transition planning and outcome tracking
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch

from src.enhancements.career_transition_manager import (
    CareerTransitionManager, CareerPath, TransitionReason, TransitionOutcome,
    IncentiveStructure, TransitionProfile, TransitionPlan, TransitionOutcomeRecord
)
from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel, CareerStage
from src.core.exceptions import ValidationError, CareerSystemError


class TestIncentiveStructure:
    """Test IncentiveStructure data class."""
    
    def test_valid_incentive_structure(self):
        """Test creating valid incentive structure."""
        structure = IncentiveStructure(
            career_path=CareerPath.INDUSTRY_RESEARCH,
            salary_range=(100000, 180000),
            job_security_score=0.5,
            research_freedom_score=0.6,
            work_life_balance_score=0.6,
            advancement_opportunities=0.8,
            impact_potential=0.7,
            publication_importance=0.4,
            collaboration_opportunities=0.7,
            geographic_flexibility=0.7
        )
        
        assert structure.career_path == CareerPath.INDUSTRY_RESEARCH
        assert structure.salary_range == (100000, 180000)
        assert structure.job_security_score == 0.5
    
    def test_invalid_scores_raise_validation_error(self):
        """Test that invalid scores raise ValidationError."""
        with pytest.raises(ValidationError):
            IncentiveStructure(
                career_path=CareerPath.INDUSTRY_RESEARCH,
                salary_range=(100000, 180000),
                job_security_score=1.5,  # Invalid score > 1.0
                research_freedom_score=0.6,
                work_life_balance_score=0.6,
                advancement_opportunities=0.8,
                impact_potential=0.7,
                publication_importance=0.4,
                collaboration_opportunities=0.7,
                geographic_flexibility=0.7
            )


class TestTransitionProfile:
    """Test TransitionProfile data class."""
    
    def test_valid_transition_profile(self):
        """Test creating valid transition profile."""
        profile = TransitionProfile(
            researcher_id="researcher_1",
            current_path=CareerPath.ACADEMIC_RESEARCH,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH, CareerPath.GOVERNMENT_LAB],
            priority_factors={"salary": 0.3, "security": 0.4, "freedom": 0.3},
            constraints=["geographic_immobility"],
            risk_tolerance=0.6,
            salary_requirements=(80000, 120000),
            timeline_flexibility=12
        )
        
        assert profile.researcher_id == "researcher_1"
        assert profile.current_path == CareerPath.ACADEMIC_RESEARCH
        assert len(profile.preferred_paths) == 2
        assert profile.risk_tolerance == 0.6
    
    def test_invalid_risk_tolerance_raises_validation_error(self):
        """Test that invalid risk tolerance raises ValidationError."""
        with pytest.raises(ValidationError):
            TransitionProfile(
                researcher_id="researcher_1",
                current_path=CareerPath.ACADEMIC_RESEARCH,
                preferred_paths=[CareerPath.INDUSTRY_RESEARCH],
                priority_factors={"salary": 1.0},
                constraints=[],
                risk_tolerance=1.5,  # Invalid risk tolerance > 1.0
                salary_requirements=(80000, 120000),
                timeline_flexibility=12
            )
    
    def test_invalid_salary_requirements_raise_validation_error(self):
        """Test that invalid salary requirements raise ValidationError."""
        with pytest.raises(ValidationError):
            TransitionProfile(
                researcher_id="researcher_1",
                current_path=CareerPath.ACADEMIC_RESEARCH,
                preferred_paths=[CareerPath.INDUSTRY_RESEARCH],
                priority_factors={"salary": 1.0},
                constraints=[],
                risk_tolerance=0.5,
                salary_requirements=(120000, 80000),  # Min > preferred
                timeline_flexibility=12
            )


class TestTransitionPlan:
    """Test TransitionPlan data class."""
    
    def test_valid_transition_plan(self):
        """Test creating valid transition plan."""
        plan = TransitionPlan(
            researcher_id="researcher_1",
            source_path=CareerPath.ACADEMIC_RESEARCH,
            target_path=CareerPath.INDUSTRY_RESEARCH,
            transition_probability=0.7,
            estimated_timeline=6,
            required_skills=["Project management", "Business acumen"],
            skill_gaps=["Business acumen"],
            preparation_steps=["Take business course", "Network with industry"],
            networking_requirements=["Industry conferences", "Alumni network"],
            financial_considerations={"salary_change": 25.0, "transition_costs": 5000},
            risk_factors=["Market volatility"],
            success_factors=["Strong technical background"]
        )
        
        assert plan.researcher_id == "researcher_1"
        assert plan.transition_probability == 0.7
        assert plan.estimated_timeline == 6
        assert len(plan.skill_gaps) == 1
    
    def test_invalid_probability_raises_validation_error(self):
        """Test that invalid probability raises ValidationError."""
        with pytest.raises(ValidationError):
            TransitionPlan(
                researcher_id="researcher_1",
                source_path=CareerPath.ACADEMIC_RESEARCH,
                target_path=CareerPath.INDUSTRY_RESEARCH,
                transition_probability=1.5,  # Invalid probability > 1.0
                estimated_timeline=6,
                required_skills=[],
                skill_gaps=[],
                preparation_steps=[],
                networking_requirements=[],
                financial_considerations={},
                risk_factors=[],
                success_factors=[]
            )


class TestCareerTransitionManager:
    """Test CareerTransitionManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create CareerTransitionManager instance for testing."""
        return CareerTransitionManager()
    
    @pytest.fixture
    def sample_researcher(self):
        """Create sample researcher for testing."""
        return EnhancedResearcher(
            id="researcher_1",
            name="Dr. Jane Smith",
            specialty="computer_science",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_tier=2,
            h_index=12,
            total_citations=250,
            years_active=6,
            reputation_score=0.7,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=CareerStage.EARLY_CAREER,
            funding_status=None,
            publication_pressure=0.6,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[
                {"title": "Paper 1", "year": 2020, "venue_type": "conference", "first_author": True},
                {"title": "Paper 2", "year": 2021, "venue_type": "journal", "first_author": True},
                {"title": "Paper 3", "year": 2022, "venue_type": "conference", "first_author": False}
            ],
            career_milestones=[]
        )
    
    def test_initialization(self, manager):
        """Test CareerTransitionManager initialization."""
        assert len(manager.incentive_structures) == len(CareerPath)
        assert len(manager.transition_profiles) == 0
        assert len(manager.transition_plans) == 0
        assert len(manager.transition_outcomes) == 0
    
    def test_get_incentive_structure(self, manager):
        """Test getting incentive structure for career path."""
        structure = manager.get_incentive_structure(CareerPath.INDUSTRY_RESEARCH)
        
        assert structure.career_path == CareerPath.INDUSTRY_RESEARCH
        assert isinstance(structure.salary_range, tuple)
        assert 0.0 <= structure.job_security_score <= 1.0
        assert 0.0 <= structure.research_freedom_score <= 1.0
    
    def test_get_incentive_structure_default(self, manager):
        """Test getting incentive structure returns default for unknown path."""
        # Remove a path to test default behavior
        del manager.incentive_structures[CareerPath.CONSULTING]
        
        structure = manager.get_incentive_structure(CareerPath.CONSULTING)
        assert structure.career_path == CareerPath.ACADEMIC_RESEARCH  # Default
    
    def test_create_transition_profile(self, manager, sample_researcher):
        """Test creating transition profile for researcher."""
        preferred_paths = [CareerPath.INDUSTRY_RESEARCH, CareerPath.GOVERNMENT_LAB]
        
        profile = manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=preferred_paths,
            risk_tolerance=0.6
        )
        
        assert profile.researcher_id == sample_researcher.id
        assert profile.current_path == CareerPath.ACADEMIC_RESEARCH
        assert profile.preferred_paths == preferred_paths
        assert profile.risk_tolerance == 0.6
        assert isinstance(profile.salary_requirements, tuple)
        assert profile.timeline_flexibility > 0
        
        # Check it's stored
        assert sample_researcher.id in manager.transition_profiles
    
    def test_create_transition_profile_with_custom_priorities(self, manager, sample_researcher):
        """Test creating transition profile with custom priority factors."""
        priority_factors = {
            'salary': 0.4,
            'job_security': 0.3,
            'work_life_balance': 0.3
        }
        
        profile = manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH],
            priority_factors=priority_factors,
            constraints=["geographic_immobility"],
            risk_tolerance=0.8
        )
        
        assert profile.priority_factors == priority_factors
        assert "geographic_immobility" in profile.constraints
        assert profile.risk_tolerance == 0.8
    
    def test_calculate_transition_probability(self, manager, sample_researcher):
        """Test calculating transition probability."""
        # Create profile first
        manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH]
        )
        
        probability = manager.calculate_transition_probability(
            researcher=sample_researcher,
            target_path=CareerPath.INDUSTRY_RESEARCH
        )
        
        assert 0.0 <= probability <= 1.0
        assert isinstance(probability, float)
    
    def test_calculate_transition_probability_without_profile(self, manager, sample_researcher):
        """Test calculating transition probability creates default profile."""
        probability = manager.calculate_transition_probability(
            researcher=sample_researcher,
            target_path=CareerPath.INDUSTRY_RESEARCH
        )
        
        assert 0.0 <= probability <= 1.0
        # Should have created a profile
        assert sample_researcher.id in manager.transition_profiles
    
    def test_calculate_transition_probability_varies_by_path(self, manager, sample_researcher):
        """Test that transition probability varies by target path."""
        manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH, CareerPath.ENTREPRENEURSHIP]
        )
        
        prob_industry = manager.calculate_transition_probability(
            researcher=sample_researcher,
            target_path=CareerPath.INDUSTRY_RESEARCH
        )
        
        prob_entrepreneurship = manager.calculate_transition_probability(
            researcher=sample_researcher,
            target_path=CareerPath.ENTREPRENEURSHIP
        )
        
        # Probabilities should be different (industry typically easier than entrepreneurship)
        assert prob_industry != prob_entrepreneurship
    
    def test_create_transition_plan(self, manager, sample_researcher):
        """Test creating detailed transition plan."""
        manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH]
        )
        
        plan = manager.create_transition_plan(
            researcher=sample_researcher,
            target_path=CareerPath.INDUSTRY_RESEARCH
        )
        
        assert plan.researcher_id == sample_researcher.id
        assert plan.source_path == CareerPath.ACADEMIC_RESEARCH
        assert plan.target_path == CareerPath.INDUSTRY_RESEARCH
        assert 0.0 <= plan.transition_probability <= 1.0
        assert plan.estimated_timeline > 0
        assert isinstance(plan.required_skills, list)
        assert isinstance(plan.skill_gaps, list)
        assert isinstance(plan.preparation_steps, list)
        assert isinstance(plan.networking_requirements, list)
        assert isinstance(plan.financial_considerations, dict)
        assert isinstance(plan.risk_factors, list)
        assert isinstance(plan.success_factors, list)
        
        # Check it's stored
        assert sample_researcher.id in manager.transition_plans
        assert len(manager.transition_plans[sample_researcher.id]) == 1
    
    def test_create_transition_plan_without_profile(self, manager, sample_researcher):
        """Test creating transition plan creates default profile if needed."""
        plan = manager.create_transition_plan(
            researcher=sample_researcher,
            target_path=CareerPath.INDUSTRY_RESEARCH
        )
        
        assert plan.researcher_id == sample_researcher.id
        # Should have created a profile
        assert sample_researcher.id in manager.transition_profiles
    
    def test_record_transition_outcome(self, manager, sample_researcher):
        """Test recording transition outcome."""
        # Create profile first
        manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH]
        )
        
        outcome_record = manager.record_transition_outcome(
            researcher_id=sample_researcher.id,
            target_path=CareerPath.INDUSTRY_RESEARCH,
            outcome=TransitionOutcome.SUCCESSFUL,
            actual_timeline=8,
            salary_change=25.0,
            satisfaction_score=0.8,
            lessons_learned=["Networking was crucial", "Technical skills transferred well"],
            challenges_faced=["Adapting to corporate culture", "Learning business processes"]
        )
        
        assert outcome_record.researcher_id == sample_researcher.id
        assert outcome_record.target_path == CareerPath.INDUSTRY_RESEARCH
        assert outcome_record.outcome == TransitionOutcome.SUCCESSFUL
        assert outcome_record.actual_timeline == 8
        assert outcome_record.salary_change == 25.0
        assert outcome_record.satisfaction_score == 0.8
        assert len(outcome_record.lessons_learned) == 2
        assert len(outcome_record.challenges_faced) == 2
        
        # Check it's stored
        assert sample_researcher.id in manager.transition_outcomes
        assert len(manager.transition_outcomes[sample_researcher.id]) == 1
    
    def test_record_transition_outcome_without_profile_raises_error(self, manager, sample_researcher):
        """Test recording outcome without profile raises error."""
        with pytest.raises(CareerSystemError):
            manager.record_transition_outcome(
                researcher_id=sample_researcher.id,
                target_path=CareerPath.INDUSTRY_RESEARCH,
                outcome=TransitionOutcome.SUCCESSFUL,
                actual_timeline=8,
                salary_change=25.0,
                satisfaction_score=0.8
            )
    
    def test_analyze_transition_patterns_no_data(self, manager):
        """Test analyzing transition patterns with no data."""
        analysis = manager.analyze_transition_patterns()
        
        assert 'error' in analysis
        assert analysis['error'] == 'No transition data available'
    
    def test_analyze_transition_patterns_with_data(self, manager, sample_researcher):
        """Test analyzing transition patterns with data."""
        # Create profile and record some outcomes
        manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH]
        )
        
        # Record multiple outcomes
        manager.record_transition_outcome(
            researcher_id=sample_researcher.id,
            target_path=CareerPath.INDUSTRY_RESEARCH,
            outcome=TransitionOutcome.SUCCESSFUL,
            actual_timeline=6,
            salary_change=20.0,
            satisfaction_score=0.8
        )
        
        manager.record_transition_outcome(
            researcher_id=sample_researcher.id,
            target_path=CareerPath.CONSULTING,
            outcome=TransitionOutcome.PARTIALLY_SUCCESSFUL,
            actual_timeline=12,
            salary_change=15.0,
            satisfaction_score=0.6
        )
        
        analysis = manager.analyze_transition_patterns()
        
        assert 'total_transitions' in analysis
        assert analysis['total_transitions'] == 2
        assert 'success_rates_by_path' in analysis
        assert 'average_timelines_months' in analysis
        assert 'salary_change_percentages' in analysis
        assert 'satisfaction_scores' in analysis
        assert 'overall_success_rate' in analysis
        
        # Check specific values
        assert CareerPath.INDUSTRY_RESEARCH.value in analysis['success_rates_by_path']
        assert analysis['success_rates_by_path'][CareerPath.INDUSTRY_RESEARCH.value] == 1.0
    
    def test_get_transition_recommendations(self, manager, sample_researcher):
        """Test getting personalized transition recommendations."""
        recommendations = manager.get_transition_recommendations(sample_researcher)
        
        assert 'researcher_id' in recommendations
        assert recommendations['researcher_id'] == sample_researcher.id
        assert 'current_path' in recommendations
        assert 'current_path_benefits' in recommendations
        assert 'top_recommendations' in recommendations
        assert 'transition_readiness' in recommendations
        assert 'general_advice' in recommendations
        
        # Check top recommendations structure
        top_recs = recommendations['top_recommendations']
        assert len(top_recs) <= 3
        
        for rec in top_recs:
            assert 'career_path' in rec
            assert 'success_probability' in rec
            assert 'salary_range' in rec
            assert 'key_benefits' in rec
            assert 'main_challenges' in rec
            assert 'preparation_time' in rec
            assert 0.0 <= rec['success_probability'] <= 1.0
    
    def test_get_transition_recommendations_readiness_assessment(self, manager):
        """Test transition readiness assessment for different researcher profiles."""
        # Test junior researcher (low readiness)
        junior_researcher = EnhancedResearcher(
            id="junior_1",
            name="Junior Researcher",
            specialty="computer_science",
            level=ResearcherLevel.GRADUATE_STUDENT,
            institution_tier=3,
            h_index=2,
            total_citations=10,
            years_active=2,
            reputation_score=0.3,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=CareerStage.EARLY_CAREER,
            funding_status=None,
            publication_pressure=0.8,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[{"title": "Paper 1", "year": 2023}],
            career_milestones=[]
        )
        
        junior_recs = manager.get_transition_recommendations(junior_researcher)
        assert "Should build more experience" in junior_recs['transition_readiness']
        
        # Test senior researcher (high readiness)
        senior_researcher = EnhancedResearcher(
            id="senior_1",
            name="Senior Researcher",
            specialty="computer_science",
            level=ResearcherLevel.ASSOCIATE_PROF,
            institution_tier=1,
            h_index=25,
            total_citations=800,
            years_active=12,
            reputation_score=0.9,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=CareerStage.MID_CAREER,
            funding_status=None,
            publication_pressure=0.4,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[{"title": f"Paper {i}", "year": 2020+i} for i in range(20)],
            career_milestones=[]
        )
        
        senior_recs = manager.get_transition_recommendations(senior_researcher)
        assert "Highly ready" in senior_recs['transition_readiness']
    
    def test_private_methods_salary_calculation(self, manager, sample_researcher):
        """Test private methods for salary calculation."""
        salary_req = manager._calculate_salary_requirements(sample_researcher)
        
        assert isinstance(salary_req, tuple)
        assert len(salary_req) == 2
        assert salary_req[0] <= salary_req[1]  # Min <= preferred
        assert salary_req[0] > 0  # Positive salaries
    
    def test_private_methods_timeline_flexibility(self, manager, sample_researcher):
        """Test private methods for timeline flexibility estimation."""
        timeline = manager._estimate_timeline_flexibility(sample_researcher)
        
        assert isinstance(timeline, int)
        assert timeline > 0
        assert timeline <= 36  # Reasonable upper bound
    
    def test_private_methods_qualification_factor(self, manager, sample_researcher):
        """Test qualification factor calculation."""
        factor = manager._calculate_qualification_factor(sample_researcher, CareerPath.INDUSTRY_RESEARCH)
        
        assert isinstance(factor, float)
        assert factor > 0
        assert factor <= 1.5  # Should be capped at 1.5x
    
    def test_private_methods_market_factor(self, manager):
        """Test market factor calculation."""
        factor = manager._calculate_market_factor(CareerPath.INDUSTRY_RESEARCH)
        
        assert isinstance(factor, float)
        assert 0.0 <= factor <= 1.0
    
    def test_private_methods_skill_identification(self, manager, sample_researcher):
        """Test skill identification methods."""
        required_skills = manager._identify_required_skills(CareerPath.INDUSTRY_RESEARCH)
        
        assert isinstance(required_skills, list)
        assert len(required_skills) > 0
        assert all(isinstance(skill, str) for skill in required_skills)
        
        # Test skill gap identification
        skill_gaps = manager._identify_skill_gaps(sample_researcher, required_skills)
        
        assert isinstance(skill_gaps, list)
        # Should identify some gaps for academic researcher transitioning to industry
        assert len(skill_gaps) > 0
    
    def test_private_methods_financial_considerations(self, manager, sample_researcher):
        """Test financial considerations calculation."""
        financial = manager._calculate_financial_considerations(
            researcher=sample_researcher,
            source_path=CareerPath.ACADEMIC_RESEARCH,
            target_path=CareerPath.INDUSTRY_RESEARCH
        )
        
        assert isinstance(financial, dict)
        assert 'current_salary_estimate' in financial
        assert 'target_salary_estimate' in financial
        assert 'salary_change_percentage' in financial
        assert 'transition_costs' in financial
        assert 'payback_period_months' in financial
        
        # Check values are reasonable
        assert financial['current_salary_estimate'] > 0
        assert financial['target_salary_estimate'] > 0
        assert financial['transition_costs'] > 0
    
    def test_edge_cases_empty_publication_history(self, manager):
        """Test handling researcher with no publications."""
        researcher_no_pubs = EnhancedResearcher(
            id="no_pubs_1",
            name="No Publications",
            specialty="computer_science",
            level=ResearcherLevel.GRADUATE_STUDENT,
            institution_tier=3,
            h_index=0,
            total_citations=0,
            years_active=1,
            reputation_score=0.1,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=CareerStage.EARLY_CAREER,
            funding_status=None,
            publication_pressure=0.9,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[],  # Empty publication history
            career_milestones=[]
        )
        
        # Should still work without errors
        probability = manager.calculate_transition_probability(
            researcher=researcher_no_pubs,
            target_path=CareerPath.INDUSTRY_APPLIED
        )
        
        assert 0.0 <= probability <= 1.0
        
        recommendations = manager.get_transition_recommendations(researcher_no_pubs)
        assert 'top_recommendations' in recommendations
    
    def test_edge_cases_all_career_paths(self, manager, sample_researcher):
        """Test transition probability calculation for all career paths."""
        for path in CareerPath:
            if path != CareerPath.ACADEMIC_RESEARCH:  # Skip current path
                probability = manager.calculate_transition_probability(
                    researcher=sample_researcher,
                    target_path=path
                )
                
                assert 0.0 <= probability <= 1.0, f"Invalid probability for {path.value}"
    
    def test_multiple_transition_plans(self, manager, sample_researcher):
        """Test creating multiple transition plans for same researcher."""
        manager.create_transition_profile(
            researcher=sample_researcher,
            preferred_paths=[CareerPath.INDUSTRY_RESEARCH, CareerPath.CONSULTING]
        )
        
        # Create multiple plans
        plan1 = manager.create_transition_plan(sample_researcher, CareerPath.INDUSTRY_RESEARCH)
        plan2 = manager.create_transition_plan(sample_researcher, CareerPath.CONSULTING)
        
        assert len(manager.transition_plans[sample_researcher.id]) == 2
        assert plan1.target_path != plan2.target_path
        assert plan1.transition_probability != plan2.transition_probability
    
    @patch('random.randint')
    def test_deterministic_timeline_estimation(self, mock_randint, manager, sample_researcher):
        """Test timeline estimation with mocked randomness for deterministic testing."""
        mock_randint.return_value = 12  # Fixed return value
        
        timeline = manager._estimate_timeline_flexibility(sample_researcher)
        assert timeline == 12
        
        # Test transition timeline estimation
        transition_timeline = manager._estimate_transition_timeline(
            researcher=sample_researcher,
            target_path=CareerPath.INDUSTRY_RESEARCH,
            probability=0.7
        )
        
        assert isinstance(transition_timeline, int)
        assert 1 <= transition_timeline <= 36


if __name__ == "__main__":
    pytest.main([__file__])