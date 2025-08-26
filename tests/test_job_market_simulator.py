"""
Unit Tests for Job Market Simulator

This module contains comprehensive unit tests for the JobMarketSimulator class,
testing postdoc and faculty competition modeling, position scarcity dynamics,
and job market outcome prediction functionality.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch
import random

from src.enhancements.job_market_simulator import (
    JobMarketSimulator, JobPosition, JobApplication, JobMarketCandidate,
    JobMarketResults, MarketTrends, PositionType, InstitutionTier,
    ApplicationOutcome
)
from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel, CareerStage
from src.core.exceptions import ValidationError, CareerSystemError


class TestJobPosition:
    """Test JobPosition data class."""
    
    def test_job_position_creation(self):
        """Test creating a valid job position."""
        position = JobPosition(
            position_id="pos_1",
            position_type=PositionType.ASSISTANT_PROF,
            institution_name="Test University",
            institution_tier=InstitutionTier.R1_MID,
            field="Computer Science",
            location="Boston, MA",
            salary_range=(70000, 90000),
            required_qualifications=["PhD", "Research experience"],
            preferred_qualifications=["Postdoc experience"],
            application_deadline=date(2024, 12, 1),
            start_date=date(2025, 8, 15),
            expected_applicants=100
        )
        
        assert position.position_id == "pos_1"
        assert position.position_type == PositionType.ASSISTANT_PROF
        assert position.institution_name == "Test University"
        assert position.salary_range == (70000, 90000)
        assert "PhD" in position.required_qualifications
    
    def test_job_position_validation(self):
        """Test job position validation."""
        # Test invalid salary range
        with pytest.raises(ValidationError):
            JobPosition(
                position_id="pos_1",
                position_type=PositionType.ASSISTANT_PROF,
                institution_name="Test University",
                institution_tier=InstitutionTier.R1_MID,
                field="Computer Science",
                location="Boston, MA",
                salary_range=(90000, 70000),  # Invalid: min > max
                required_qualifications=["PhD"],
                preferred_qualifications=[],
                application_deadline=date(2024, 12, 1),
                start_date=date(2025, 8, 15)
            )
        
        # Test invalid number of positions
        with pytest.raises(ValidationError):
            JobPosition(
                position_id="pos_1",
                position_type=PositionType.ASSISTANT_PROF,
                institution_name="Test University",
                institution_tier=InstitutionTier.R1_MID,
                field="Computer Science",
                location="Boston, MA",
                salary_range=(70000, 90000),
                required_qualifications=["PhD"],
                preferred_qualifications=[],
                application_deadline=date(2024, 12, 1),
                start_date=date(2025, 8, 15),
                number_of_positions=0  # Invalid
            )


class TestJobApplication:
    """Test JobApplication data class."""
    
    def test_job_application_creation(self):
        """Test creating a valid job application."""
        application = JobApplication(
            application_id="app_1",
            applicant_id="researcher_1",
            position_id="pos_1",
            application_date=date(2024, 6, 1),
            materials_submitted=["CV", "Cover Letter", "Research Statement"]
        )
        
        assert application.application_id == "app_1"
        assert application.applicant_id == "researcher_1"
        assert application.position_id == "pos_1"
        assert application.outcome == ApplicationOutcome.PENDING
        assert "CV" in application.materials_submitted


class TestJobMarketCandidate:
    """Test JobMarketCandidate data class."""
    
    @pytest.fixture
    def sample_researcher(self):
        """Create a sample researcher for testing."""
        return EnhancedResearcher(
            id="researcher_1",
            name="Dr. Test Candidate",
            specialty="Computer Science",
            level=ResearcherLevel.POSTDOC,
            institution_tier=1,
            h_index=8,
            total_citations=120,
            years_active=3,
            reputation_score=0.6,
            cognitive_biases={},
            review_behavior={},
            strategic_behavior={},
            career_stage=CareerStage.EARLY_CAREER,
            funding_status="ADEQUATELY_FUNDED",
            publication_pressure=0.7,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[
                {'title': 'Paper 1', 'venue_type': 'conference'},
                {'title': 'Paper 2', 'venue_type': 'journal'}
            ],
            career_milestones=[]
        )
    
    def test_job_market_candidate_creation(self, sample_researcher):
        """Test creating a valid job market candidate."""
        candidate = JobMarketCandidate(
            researcher=sample_researcher,
            target_positions=[PositionType.ASSISTANT_PROF, PositionType.POSTDOC],
            geographic_preferences=["Boston", "San Francisco"],
            salary_expectations=(70000, 90000),
            mobility_constraints=[],
            application_strategy="selective",
            market_competitiveness=0.7
        )
        
        assert candidate.researcher.id == "researcher_1"
        assert PositionType.ASSISTANT_PROF in candidate.target_positions
        assert "Boston" in candidate.geographic_preferences
        assert candidate.market_competitiveness == 0.7
    
    def test_job_market_candidate_validation(self, sample_researcher):
        """Test job market candidate validation."""
        # Test invalid competitiveness score
        with pytest.raises(ValidationError):
            JobMarketCandidate(
                researcher=sample_researcher,
                target_positions=[PositionType.ASSISTANT_PROF],
                geographic_preferences=["Any"],
                salary_expectations=(70000, 90000),
                mobility_constraints=[],
                application_strategy="selective",
                market_competitiveness=1.5  # Invalid: > 1.0
            )


class TestJobMarketSimulator:
    """Test JobMarketSimulator class."""
    
    @pytest.fixture
    def simulator(self):
        """Create a JobMarketSimulator instance for testing."""
        return JobMarketSimulator()
    
    @pytest.fixture
    def sample_researcher(self):
        """Create a sample researcher for testing."""
        return EnhancedResearcher(
            id="researcher_1",
            name="Dr. Test Researcher",
            specialty="Computer Science",
            level=ResearcherLevel.POSTDOC,
            institution_tier=1,
            h_index=10,
            total_citations=150,
            years_active=3,
            reputation_score=0.7,
            cognitive_biases={},
            review_behavior={},
            strategic_behavior={},
            career_stage=CareerStage.EARLY_CAREER,
            funding_status="ADEQUATELY_FUNDED",
            publication_pressure=0.7,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[
                {'title': 'Paper 1', 'venue_type': 'conference'},
                {'title': 'Paper 2', 'venue_type': 'journal'}
            ],
            career_milestones=[]
        )
    
    def test_simulator_initialization(self, simulator):
        """Test JobMarketSimulator initialization."""
        assert isinstance(simulator.available_positions, dict)
        assert isinstance(simulator.job_candidates, dict)
        assert isinstance(simulator.applications, dict)
        assert isinstance(simulator.market_history, list)
        assert len(simulator.available_positions) == 0
        assert len(simulator.job_candidates) == 0
    
    def test_create_job_position(self, simulator):
        """Test creating job positions."""
        position = simulator.create_job_position(
            position_type=PositionType.ASSISTANT_PROF,
            institution_tier=InstitutionTier.R1_MID,
            field="Computer Science",
            location="Boston, MA"
        )
        
        assert position.position_type == PositionType.ASSISTANT_PROF
        assert position.institution_tier == InstitutionTier.R1_MID
        assert position.field == "Computer Science"
        assert position.location == "Boston, MA"
        assert position.position_id in simulator.available_positions
        assert len(simulator.available_positions) == 1
    
    def test_register_job_candidate(self, simulator, sample_researcher):
        """Test registering job candidates."""
        candidate = simulator.register_job_candidate(
            researcher=sample_researcher,
            target_positions=[PositionType.ASSISTANT_PROF, PositionType.POSTDOC],
            geographic_preferences=["Boston", "San Francisco"],
            application_strategy="selective"
        )
        
        assert candidate.researcher.id == sample_researcher.id
        assert PositionType.ASSISTANT_PROF in candidate.target_positions
        assert candidate.application_strategy == "selective"
        assert 0.0 <= candidate.market_competitiveness <= 1.0
        assert sample_researcher.id in simulator.job_candidates
    
    def test_calculate_market_competitiveness(self, simulator, sample_researcher):
        """Test market competitiveness calculation."""
        competitiveness = simulator._calculate_market_competitiveness(sample_researcher)
        
        assert 0.0 <= competitiveness <= 1.0
        assert isinstance(competitiveness, float)
        
        # Test with high-performing researcher
        high_performer = EnhancedResearcher(
            id="high_performer",
            name="Dr. High Performer",
            specialty="Computer Science",
            level=ResearcherLevel.FULL_PROF,
            institution_tier=1,
            h_index=25,
            total_citations=1000,
            years_active=10,
            reputation_score=0.9,
            cognitive_biases={},
            review_behavior={},
            strategic_behavior={},
            career_stage=CareerStage.ESTABLISHED,
            funding_status="WELL_FUNDED",
            publication_pressure=0.3,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[{'title': f'Paper {i}', 'venue_type': 'journal'} for i in range(25)],
            career_milestones=[]
        )
        
        high_competitiveness = simulator._calculate_market_competitiveness(high_performer)
        assert high_competitiveness > competitiveness
    
    def test_simulate_job_market_cycle(self, simulator, sample_researcher):
        """Test job market cycle simulation."""
        # Register a candidate
        simulator.register_job_candidate(
            researcher=sample_researcher,
            target_positions=[PositionType.POSTDOC],
            application_strategy="selective"
        )
        
        # Simulate market cycle
        results = simulator.simulate_job_market_cycle(
            year=2024,
            field="computer_science"
        )
        
        assert isinstance(results, JobMarketResults)
        assert results.year == 2024
        assert results.total_positions > 0
        assert results.total_candidates >= 1
        assert results.competition_ratio > 0
        assert 0.0 <= results.placement_rate <= 1.0
        assert isinstance(results.position_outcomes, dict)
        assert isinstance(results.candidate_outcomes, dict)
        assert isinstance(results.market_trends, dict)
        assert isinstance(results.field_analysis, dict)
    
    def test_predict_job_market_outcome(self, simulator, sample_researcher):
        """Test job market outcome prediction."""
        # Register candidate
        candidate = simulator.register_job_candidate(
            researcher=sample_researcher,
            target_positions=[PositionType.ASSISTANT_PROF, PositionType.POSTDOC],
            application_strategy="selective"
        )
        
        # Predict outcomes
        prediction = simulator.predict_job_market_outcome(candidate)
        
        assert prediction['candidate_id'] == sample_researcher.id
        assert 'target_year' in prediction
        assert 'field' in prediction
        assert 'position_predictions' in prediction
        assert 'overall_assessment' in prediction
        assert 'market_conditions' in prediction
        
        # Check position predictions
        for position_type in candidate.target_positions:
            pos_pred = prediction['position_predictions'][position_type.value]
            assert 'success_probability' in pos_pred
            assert 'expected_timeline_months' in pos_pred
            assert 'predicted_salary_range' in pos_pred
            assert 'competition_level' in pos_pred
            assert 'recommended_applications' in pos_pred
            assert 'key_strengths' in pos_pred
            assert 'improvement_areas' in pos_pred
            
            assert 0.0 <= pos_pred['success_probability'] <= 1.0
            assert pos_pred['expected_timeline_months'] > 0
            assert isinstance(pos_pred['predicted_salary_range'], tuple)
            assert pos_pred['competition_level'] > 0
    
    def test_salary_range_calculation(self, simulator):
        """Test salary range calculation."""
        # Test different position types
        postdoc_range = simulator._calculate_salary_range(
            PositionType.POSTDOC, InstitutionTier.R1_MID
        )
        prof_range = simulator._calculate_salary_range(
            PositionType.ASSISTANT_PROF, InstitutionTier.R1_MID
        )
        
        assert isinstance(postdoc_range, tuple)
        assert isinstance(prof_range, tuple)
        assert len(postdoc_range) == 2
        assert len(prof_range) == 2
        assert postdoc_range[0] < postdoc_range[1]
        assert prof_range[0] < prof_range[1]
        assert prof_range[0] > postdoc_range[0]  # Professors earn more than postdocs
        
        # Test tier differences
        r1_top_range = simulator._calculate_salary_range(
            PositionType.ASSISTANT_PROF, InstitutionTier.R1_TOP
        )
        teaching_range = simulator._calculate_salary_range(
            PositionType.ASSISTANT_PROF, InstitutionTier.TEACHING
        )
        
        assert r1_top_range[0] > teaching_range[0]  # Top tier pays more
    
    def test_market_statistics(self, simulator, sample_researcher):
        """Test market statistics generation."""
        # Initially no data
        stats = simulator.get_market_statistics()
        assert 'error' in stats or stats['total_positions'] == 0
        
        # Add some data
        simulator.create_job_position(
            PositionType.ASSISTANT_PROF, InstitutionTier.R1_MID, "Computer Science"
        )
        simulator.register_job_candidate(
            sample_researcher, [PositionType.ASSISTANT_PROF]
        )
        
        # Run simulation to generate history
        simulator.simulate_job_market_cycle(field="computer_science")
        
        stats = simulator.get_market_statistics()
        assert 'total_positions' in stats
        assert 'total_candidates' in stats
        assert 'recent_placement_rate' in stats
        assert 'recent_competition_ratio' in stats
        assert 'position_distribution' in stats
        assert 'candidate_competitiveness_distribution' in stats
    
    def test_export_market_data(self, simulator, sample_researcher):
        """Test market data export."""
        # Add some data
        simulator.create_job_position(
            PositionType.ASSISTANT_PROF, InstitutionTier.R1_MID, "Computer Science"
        )
        simulator.register_job_candidate(
            sample_researcher, [PositionType.ASSISTANT_PROF]
        )
        
        # Export data
        export_data = simulator.export_market_data()
        
        assert 'market_results' in export_data
        assert 'current_positions' in export_data
        assert 'registered_candidates' in export_data
        assert isinstance(export_data['market_results'], list)
        assert isinstance(export_data['current_positions'], list)
        assert isinstance(export_data['registered_candidates'], list)
        
        # Check position data structure
        if export_data['current_positions']:
            pos_data = export_data['current_positions'][0]
            assert 'position_id' in pos_data
            assert 'position_type' in pos_data
            assert 'institution_tier' in pos_data
            assert 'field' in pos_data
            assert 'salary_range' in pos_data
        
        # Check candidate data structure
        if export_data['registered_candidates']:
            cand_data = export_data['registered_candidates'][0]
            assert 'candidate_id' in cand_data
            assert 'researcher_level' in cand_data
            assert 'target_positions' in cand_data
            assert 'market_competitiveness' in cand_data
            assert 'application_strategy' in cand_data
    
    def test_application_strategy_effects(self, simulator, sample_researcher):
        """Test different application strategies."""
        strategies = ['aggressive', 'selective', 'conservative']
        
        for strategy in strategies:
            candidate = JobMarketCandidate(
                researcher=sample_researcher,
                target_positions=[PositionType.ASSISTANT_PROF],
                geographic_preferences=["Any"],
                salary_expectations=(70000, 90000),
                mobility_constraints=[],
                application_strategy=strategy,
                market_competitiveness=0.7
            )
            
            prediction = simulator.predict_job_market_outcome(candidate)
            
            # Aggressive strategy should recommend more applications
            pos_pred = prediction['position_predictions'][PositionType.ASSISTANT_PROF.value]
            if strategy == 'aggressive':
                assert pos_pred['recommended_applications'] >= 10
            elif strategy == 'conservative':
                assert pos_pred['recommended_applications'] <= 15
    
    @patch('random.randint')
    @patch('random.uniform')
    def test_position_generation_deterministic(self, mock_uniform, mock_randint, simulator):
        """Test position generation with mocked randomness."""
        mock_uniform.return_value = 1.0  # Maximum multiplier
        mock_randint.return_value = 100  # Fixed applicant count
        
        positions = simulator._generate_annual_positions("computer_science", 2024)
        
        assert len(positions) > 0
        assert all(isinstance(pos, JobPosition) for pos in positions)
        assert all(pos.field == "computer_science" for pos in positions)
    
    def test_field_specific_parameters(self, simulator):
        """Test field-specific market parameters."""
        # Test different fields
        fields = ['computer_science', 'biology', 'physics']
        
        for field in fields:
            assert field in simulator.MARKET_PARAMETERS
            params = simulator.MARKET_PARAMETERS[field]
            
            assert 'postdoc_positions_per_year' in params
            assert 'assistant_prof_positions_per_year' in params
            assert 'candidates_per_position' in params
            assert 'placement_rates' in params
            
            # Check that placement rates are reasonable (0-1)
            for pos_type, rate in params['placement_rates'].items():
                assert 0.0 <= rate <= 1.0
            
            # Check that competition ratios are reasonable (>1)
            for pos_type, ratio in params['candidates_per_position'].items():
                assert ratio >= 1.0
