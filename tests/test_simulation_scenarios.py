"""
Tests for Comprehensive Simulation Scenarios

This module contains integration tests for the simulation scenarios system,
testing scenario configuration, execution logic, and validation of results.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

from src.enhancements.simulation_scenarios import (
    SimulationScenarios, ScenarioConfiguration, ScenarioResults,
    ScenarioType, ScenarioComplexity
)
from src.data.enhanced_models import ResearcherLevel, VenueType
from src.core.exceptions import ValidationError, SimulationError


class TestSimulationScenarios:
    """Test cases for the SimulationScenarios class."""
    
    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock simulation coordinator."""
        coordinator = Mock()
        coordinator.config = Mock()
        coordinator.state = Mock()
        coordinator.config.bias_strengths = {}
        coordinator.state.bias_system_active = True
        coordinator.state.network_system_active = True
        coordinator.state.career_system_active = True
        coordinator.state.funding_system_active = True
        return coordinator
    
    @pytest.fixture
    def simulation_scenarios(self, mock_coordinator):
        """Create a SimulationScenarios instance for testing."""
        return SimulationScenarios(simulation_coordinator=mock_coordinator)
    
    def test_initialization(self, simulation_scenarios):
        """Test proper initialization of simulation scenarios."""
        assert simulation_scenarios is not None
        assert len(simulation_scenarios.scenarios) > 0
        assert simulation_scenarios.execution_history == []
        
        # Check that predefined scenarios are loaded
        expected_scenarios = [
            "basic_peer_review", "bias_demonstration", "network_effects",
            "strategic_behavior", "career_progression", "funding_impact",
            "venue_dynamics", "meta_science_evolution", "comprehensive_ecosystem",
            "reproducibility_crisis"
        ]
        
        for scenario_id in expected_scenarios:
            assert scenario_id in simulation_scenarios.scenarios
    
    def test_get_available_scenarios(self, simulation_scenarios):
        """Test getting available scenarios."""
        scenarios = simulation_scenarios.get_available_scenarios()
        
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0
        
        # Check that all scenarios have proper configuration
        for scenario_id, config in scenarios.items():
            assert isinstance(config, ScenarioConfiguration)
            assert config.scenario_id == scenario_id
            assert config.name is not None
            assert config.description is not None
            assert isinstance(config.scenario_type, ScenarioType)
            assert isinstance(config.complexity, ScenarioComplexity)
    
    def test_get_specific_scenario(self, simulation_scenarios):
        """Test getting a specific scenario."""
        # Test existing scenario
        scenario = simulation_scenarios.get_scenario("basic_peer_review")
        assert scenario is not None
        assert scenario.scenario_id == "basic_peer_review"
        assert scenario.scenario_type == ScenarioType.BASIC_PEER_REVIEW
        
        # Test non-existing scenario
        scenario = simulation_scenarios.get_scenario("non_existent")
        assert scenario is None
    
    def test_basic_peer_review_scenario_configuration(self, simulation_scenarios):
        """Test the basic peer review scenario configuration."""
        scenario = simulation_scenarios.get_scenario("basic_peer_review")
        
        assert scenario.scenario_type == ScenarioType.BASIC_PEER_REVIEW
        assert scenario.complexity == ScenarioComplexity.SIMPLE
        assert scenario.duration_days == 30
        assert scenario.num_researchers == 20
        assert scenario.num_papers == 10
        assert scenario.num_venues == 3
        
        # Check that advanced features are disabled
        assert not scenario.enable_biases
        assert not scenario.enable_networks
        assert not scenario.enable_strategic_behavior
        assert not scenario.enable_career_progression
        assert not scenario.enable_funding_system
        assert not scenario.enable_meta_science
        
        # Check parameters
        assert scenario.parameters["review_deadline_weeks"] == 4
        assert scenario.parameters["min_reviews_per_paper"] == 3
        
        # Check expected outcomes
        assert scenario.expected_outcomes["reviews_completed"] == 20
        assert scenario.expected_outcomes["papers_decided"] == 10
    
    def test_bias_demonstration_scenario_configuration(self, simulation_scenarios):
        """Test the bias demonstration scenario configuration."""
        scenario = simulation_scenarios.get_scenario("bias_demonstration")
        
        assert scenario.scenario_type == ScenarioType.BIAS_DEMONSTRATION
        assert scenario.complexity == ScenarioComplexity.MODERATE
        assert scenario.enable_biases
        assert scenario.enable_networks
        assert scenario.enable_career_progression
        
        # Check bias parameters
        bias_strengths = scenario.parameters["bias_strengths"]
        assert "confirmation" in bias_strengths
        assert "halo_effect" in bias_strengths
        assert "anchoring" in bias_strengths
        assert "availability" in bias_strengths
        
        # Check expected outcomes
        assert scenario.expected_outcomes["bias_effect_detected"]
        assert "prestigious_author_advantage" in scenario.expected_outcomes
    
    def test_comprehensive_ecosystem_scenario_configuration(self, simulation_scenarios):
        """Test the comprehensive ecosystem scenario configuration."""
        scenario = simulation_scenarios.get_scenario("comprehensive_ecosystem")
        
        assert scenario.scenario_type == ScenarioType.COMPREHENSIVE_ECOSYSTEM
        assert scenario.complexity == ScenarioComplexity.COMPREHENSIVE
        assert scenario.duration_days == 3650  # 10 years
        
        # Check that all systems are enabled
        assert scenario.enable_biases
        assert scenario.enable_networks
        assert scenario.enable_strategic_behavior
        assert scenario.enable_career_progression
        assert scenario.enable_funding_system
        assert scenario.enable_meta_science
        
        # Check scale
        assert scenario.num_researchers == 200
        assert scenario.num_papers == 500
        assert scenario.num_venues == 20
    
    @patch('src.enhancements.simulation_scenarios.datetime')
    def test_execute_basic_scenario(self, mock_datetime, simulation_scenarios):
        """Test executing a basic scenario."""
        # Mock datetime
        mock_now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Execute basic scenario
        result = simulation_scenarios.execute_scenario("basic_peer_review")
        
        assert isinstance(result, ScenarioResults)
        assert result.scenario_id == "basic_peer_review"
        assert result.success
        assert result.total_reviews_generated > 0
        assert result.total_papers_processed > 0
        assert result.total_researchers_active > 0
        assert result.execution_time_seconds >= 0
        
        # Check that result is stored in history
        assert len(simulation_scenarios.execution_history) == 1
        assert simulation_scenarios.execution_history[0] == result
    
    def test_execute_scenario_with_custom_parameters(self, simulation_scenarios):
        """Test executing a scenario with custom parameters."""
        custom_params = {
            "num_researchers": 30,
            "duration_days": 45,
            "bias_strengths": {
                "confirmation": 0.6,
                "halo_effect": 0.7
            }
        }
        
        result = simulation_scenarios.execute_scenario(
            "bias_demonstration", 
            custom_parameters=custom_params
        )
        
        assert result.success
        assert result.total_researchers_active >= 30  # Should reflect custom parameter
    
    def test_execute_nonexistent_scenario(self, simulation_scenarios):
        """Test executing a non-existent scenario."""
        with pytest.raises(ValidationError, match="Scenario non_existent not found"):
            simulation_scenarios.execute_scenario("non_existent")
    
    def test_execute_scenario_batch_sequential(self, simulation_scenarios):
        """Test executing multiple scenarios in batch (sequential)."""
        scenario_ids = ["basic_peer_review", "bias_demonstration"]
        
        results = simulation_scenarios.execute_scenario_batch(
            scenario_ids, 
            parallel=False
        )
        
        assert len(results) == 2
        assert all(isinstance(r, ScenarioResults) for r in results)
        assert results[0].scenario_id == "basic_peer_review"
        assert results[1].scenario_id == "bias_demonstration"
        
        # Check execution history
        assert len(simulation_scenarios.execution_history) == 2
    
    def test_create_custom_scenario(self, simulation_scenarios):
        """Test creating a custom scenario."""
        custom_config = ScenarioConfiguration(
            scenario_id="custom_test",
            name="Custom Test Scenario",
            description="A custom scenario for testing",
            scenario_type=ScenarioType.BASIC_PEER_REVIEW,
            complexity=ScenarioComplexity.SIMPLE,
            duration_days=15,
            num_researchers=10,
            num_papers=5,
            num_venues=2,
            parameters={"test_param": "test_value"},
            expected_outcomes={"test_outcome": True}
        )
        
        scenario_id = simulation_scenarios.create_custom_scenario(custom_config)
        
        assert scenario_id == "custom_test"
        assert "custom_test" in simulation_scenarios.scenarios
        
        retrieved_scenario = simulation_scenarios.get_scenario("custom_test")
        assert retrieved_scenario.name == "Custom Test Scenario"
        assert retrieved_scenario.parameters["test_param"] == "test_value"
    
    def test_create_duplicate_custom_scenario(self, simulation_scenarios):
        """Test creating a custom scenario with duplicate ID."""
        custom_config = ScenarioConfiguration(
            scenario_id="basic_peer_review",  # Duplicate ID
            name="Duplicate Scenario",
            description="This should fail",
            scenario_type=ScenarioType.BASIC_PEER_REVIEW,
            complexity=ScenarioComplexity.SIMPLE,
            duration_days=15,
            num_researchers=10,
            num_papers=5,
            num_venues=2
        )
        
        with pytest.raises(ValidationError, match="Scenario basic_peer_review already exists"):
            simulation_scenarios.create_custom_scenario(custom_config)
    
    def test_get_execution_history(self, simulation_scenarios):
        """Test getting execution history."""
        # Initially empty
        history = simulation_scenarios.get_execution_history()
        assert history == []
        
        # Execute a scenario
        simulation_scenarios.execute_scenario("basic_peer_review")
        
        # Check history
        history = simulation_scenarios.get_execution_history()
        assert len(history) == 1
        assert history[0].scenario_id == "basic_peer_review"
    
    def test_get_scenario_statistics_empty(self, simulation_scenarios):
        """Test getting statistics with no execution history."""
        stats = simulation_scenarios.get_scenario_statistics()
        
        assert stats["total_executions"] == 0
    
    def test_get_scenario_statistics_with_executions(self, simulation_scenarios):
        """Test getting statistics with execution history."""
        # Execute multiple scenarios
        simulation_scenarios.execute_scenario("basic_peer_review")
        simulation_scenarios.execute_scenario("bias_demonstration")
        
        stats = simulation_scenarios.get_scenario_statistics()
        
        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 2
        assert stats["failed_executions"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["average_execution_time"] > 0
        assert stats["total_reviews_generated"] > 0
        assert stats["total_papers_processed"] > 0
        assert "scenarios_by_type" in stats
        assert "complexity_distribution" in stats
    
    def test_scenario_environment_initialization(self, simulation_scenarios):
        """Test scenario environment initialization."""
        scenario = simulation_scenarios.get_scenario("basic_peer_review")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        assert "researchers" in environment
        assert "papers" in environment
        assert "venues" in environment
        assert "timeline" in environment
        assert "system_configuration" in environment
        
        # Check researchers
        researchers = environment["researchers"]
        assert len(researchers) == scenario.num_researchers
        assert all("id" in r for r in researchers)
        assert all("level" in r for r in researchers)
        assert all("specialty" in r for r in researchers)
        
        # Check papers
        papers = environment["papers"]
        assert len(papers) == scenario.num_papers
        assert all("id" in p for p in papers)
        assert all("title" in p for p in papers)
        assert all("authors" in p for p in papers)
        
        # Check venues
        venues = environment["venues"]
        assert len(venues) == scenario.num_venues
        assert all("id" in v for v in venues)
        assert all("venue_type" in v for v in venues)
        assert all("acceptance_rate" in v for v in venues)
    
    def test_researcher_creation_distribution(self, simulation_scenarios):
        """Test that researchers are created with realistic level distribution."""
        scenario = simulation_scenarios.get_scenario("comprehensive_ecosystem")
        researchers = simulation_scenarios._create_scenario_researchers(scenario)
        
        # Count levels
        level_counts = {}
        for researcher in researchers:
            level = researcher["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Check that we have a reasonable distribution
        assert ResearcherLevel.GRADUATE_STUDENT in level_counts
        assert ResearcherLevel.POSTDOC in level_counts
        assert ResearcherLevel.ASSISTANT_PROF in level_counts
        
        # Graduate students should be most common
        assert level_counts[ResearcherLevel.GRADUATE_STUDENT] > level_counts.get(ResearcherLevel.EMERITUS, 0)
    
    def test_venue_creation_with_realistic_characteristics(self, simulation_scenarios):
        """Test that venues are created with realistic characteristics."""
        scenario = simulation_scenarios.get_scenario("venue_dynamics")
        venues = simulation_scenarios._create_scenario_venues(scenario)
        
        # Check venue types and acceptance rates
        for venue in venues:
            venue_type = venue["venue_type"]
            acceptance_rate = venue["acceptance_rate"]
            
            # Top venues should have lower acceptance rates
            if venue_type == VenueType.TOP_CONFERENCE:
                assert acceptance_rate <= 0.1
            elif venue_type == VenueType.TOP_JOURNAL:
                assert acceptance_rate <= 0.05
            elif venue_type == VenueType.LOW_CONFERENCE:
                assert acceptance_rate >= 0.4
    
    def test_paper_review_processing(self, simulation_scenarios):
        """Test paper review processing logic."""
        scenario = simulation_scenarios.get_scenario("basic_peer_review")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        review_results = simulation_scenarios._process_paper_reviews(scenario, environment)
        
        assert "total_reviews" in review_results
        assert "outcomes" in review_results
        assert review_results["total_reviews"] > 0
        assert len(review_results["outcomes"]) == scenario.num_papers
        
        # Check review outcomes
        for outcome in review_results["outcomes"]:
            assert "paper_id" in outcome
            assert "venue_id" in outcome
            assert "reviews" in outcome
            assert "decision" in outcome
            # Note: Some papers might not get reviews if no matching reviewers are available
            # This is realistic behavior, so we just check the structure
    
    def test_bias_effects_simulation(self, simulation_scenarios):
        """Test bias effects simulation."""
        scenario = simulation_scenarios.get_scenario("bias_demonstration")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        bias_effects = simulation_scenarios._simulate_bias_effects(scenario, environment)
        
        assert "confirmation_bias_instances" in bias_effects
        assert "halo_effect_instances" in bias_effects
        assert "anchoring_bias_instances" in bias_effects
        assert "overall_bias_impact" in bias_effects
        
        assert bias_effects["confirmation_bias_instances"] > 0
        assert 0 <= bias_effects["overall_bias_impact"] <= 1
    
    def test_network_effects_simulation(self, simulation_scenarios):
        """Test network effects simulation."""
        scenario = simulation_scenarios.get_scenario("network_effects")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        network_effects = simulation_scenarios._simulate_network_effects(scenario, environment)
        
        assert "collaboration_networks_formed" in network_effects
        assert "citation_networks_identified" in network_effects
        assert "conflict_of_interest_cases" in network_effects
        assert "network_influence_score" in network_effects
        
        assert network_effects["collaboration_networks_formed"] >= 0
        assert 0 <= network_effects["network_influence_score"] <= 1
    
    def test_strategic_behavior_simulation(self, simulation_scenarios):
        """Test strategic behavior simulation."""
        scenario = simulation_scenarios.get_scenario("strategic_behavior")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        strategic_behavior = simulation_scenarios._simulate_strategic_behaviors(scenario, environment)
        
        assert "venue_shopping_instances" in strategic_behavior
        assert "review_trading_detected" in strategic_behavior
        assert "citation_cartels_formed" in strategic_behavior
        assert "salami_slicing_cases" in strategic_behavior
        
        assert strategic_behavior["venue_shopping_instances"] >= 0
        assert strategic_behavior["citation_cartels_formed"] >= 0
    
    def test_career_progression_simulation(self, simulation_scenarios):
        """Test career progression simulation."""
        scenario = simulation_scenarios.get_scenario("career_progression")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        career_progression = simulation_scenarios._simulate_career_progression(scenario, environment)
        
        assert "tenure_evaluations" in career_progression
        assert "promotions_achieved" in career_progression
        assert "job_market_movements" in career_progression
        assert "career_pressure_impact" in career_progression
        
        assert career_progression["tenure_evaluations"] >= 0
        assert 0 <= career_progression["career_pressure_impact"] <= 1
    
    def test_funding_impact_simulation(self, simulation_scenarios):
        """Test funding impact simulation."""
        scenario = simulation_scenarios.get_scenario("funding_impact")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        funding_impact = simulation_scenarios._simulate_funding_impact(scenario, environment)
        
        assert "funding_cycles_completed" in funding_impact
        assert "publication_pressure_correlation" in funding_impact
        assert "collaboration_incentive_effect" in funding_impact
        assert "resource_constraint_impact" in funding_impact
        
        assert funding_impact["funding_cycles_completed"] >= 0
        assert 0 <= funding_impact["publication_pressure_correlation"] <= 1
    
    def test_meta_science_evolution_simulation(self, simulation_scenarios):
        """Test meta-science evolution simulation."""
        scenario = simulation_scenarios.get_scenario("meta_science_evolution")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        meta_science = simulation_scenarios._simulate_meta_science_evolution(scenario, environment)
        
        assert "reproducibility_trend" in meta_science
        assert "open_science_adoption" in meta_science
        assert "ai_impact_detected" in meta_science
        assert "reform_effectiveness" in meta_science
        
        assert -1 <= meta_science["reproducibility_trend"] <= 1
        assert 0 <= meta_science["open_science_adoption"] <= 1
        assert isinstance(meta_science["ai_impact_detected"], bool)
    
    def test_comprehensive_ecosystem_simulation(self, simulation_scenarios):
        """Test comprehensive ecosystem simulation."""
        scenario = simulation_scenarios.get_scenario("comprehensive_ecosystem")
        environment = simulation_scenarios._initialize_scenario_environment(scenario)
        
        ecosystem = simulation_scenarios._simulate_comprehensive_ecosystem(scenario, environment)
        
        assert "system_interactions" in ecosystem
        assert "emergent_behaviors" in ecosystem
        assert "ecosystem_stability" in ecosystem
        assert "complexity_metrics" in ecosystem
        
        complexity_metrics = ecosystem["complexity_metrics"]
        assert "interaction_density" in complexity_metrics
        assert "behavioral_diversity" in complexity_metrics
        assert "system_resilience" in complexity_metrics
        
        assert ecosystem["system_interactions"] > 0
        assert 0 <= ecosystem["ecosystem_stability"] <= 1
    
    def test_scenario_validation(self, simulation_scenarios):
        """Test scenario result validation."""
        scenario = simulation_scenarios.get_scenario("basic_peer_review")
        
        # Mock execution results
        execution_results = {
            "total_reviews": 30,
            "total_papers": 10,
            "review_outcomes": [
                {
                    "reviews": [
                        {"quality_score": 3.5},
                        {"quality_score": 4.0},
                        {"quality_score": 3.2}
                    ]
                }
            ]
        }
        
        validation_results = simulation_scenarios._validate_scenario_results(scenario, execution_results)
        
        assert "reviews_completed" in validation_results
        assert "papers_decided" in validation_results
        assert "average_review_quality" in validation_results
        
        # Should pass validation with these results
        assert validation_results["reviews_completed"]
        assert validation_results["papers_decided"]
    
    def test_realistic_h_index_generation(self, simulation_scenarios):
        """Test realistic h-index generation for different career levels."""
        # Test different career levels
        grad_h_index = simulation_scenarios._generate_realistic_h_index(ResearcherLevel.GRADUATE_STUDENT)
        prof_h_index = simulation_scenarios._generate_realistic_h_index(ResearcherLevel.FULL_PROF)
        
        assert 0 <= grad_h_index <= 3
        assert 20 <= prof_h_index <= 60
        assert prof_h_index > grad_h_index
    
    def test_cognitive_bias_generation(self, simulation_scenarios):
        """Test cognitive bias generation for researchers."""
        scenario = simulation_scenarios.get_scenario("bias_demonstration")
        biases = simulation_scenarios._generate_cognitive_biases(scenario)
        
        assert "confirmation" in biases
        assert "halo_effect" in biases
        assert "anchoring" in biases
        assert "availability" in biases
        
        # All bias strengths should be between 0 and 1
        for bias_strength in biases.values():
            assert 0 <= bias_strength <= 1
    
    def test_strategic_behavior_generation(self, simulation_scenarios):
        """Test strategic behavior generation for researchers."""
        scenario = simulation_scenarios.get_scenario("strategic_behavior")
        behaviors = simulation_scenarios._generate_strategic_behavior(scenario)
        
        assert "venue_shopping" in behaviors
        assert "review_trading" in behaviors
        assert "citation_gaming" in behaviors
        assert "salami_slicing" in behaviors
        
        # All behavior tendencies should be between 0 and 1
        for behavior_strength in behaviors.values():
            assert 0 <= behavior_strength <= 1
    
    def test_venue_acceptance_rates(self, simulation_scenarios):
        """Test venue acceptance rate assignment."""
        top_conf_rate = simulation_scenarios._get_venue_acceptance_rate(VenueType.TOP_CONFERENCE)
        low_conf_rate = simulation_scenarios._get_venue_acceptance_rate(VenueType.LOW_CONFERENCE)
        top_journal_rate = simulation_scenarios._get_venue_acceptance_rate(VenueType.TOP_JOURNAL)
        
        assert top_conf_rate == 0.05
        assert low_conf_rate == 0.50
        assert top_journal_rate == 0.02
        
        # Top venues should have lower acceptance rates
        assert top_conf_rate < low_conf_rate
        assert top_journal_rate < top_conf_rate
    
    def test_venue_prestige_scores(self, simulation_scenarios):
        """Test venue prestige score assignment."""
        top_conf_score = simulation_scenarios._get_venue_prestige_score(VenueType.TOP_CONFERENCE)
        low_conf_score = simulation_scenarios._get_venue_prestige_score(VenueType.LOW_CONFERENCE)
        
        assert top_conf_score == 10
        assert low_conf_score == 4
        assert top_conf_score > low_conf_score
    
    def test_review_bias_application(self, simulation_scenarios):
        """Test application of biases to review scores."""
        scenario = simulation_scenarios.get_scenario("bias_demonstration")
        
        base_scores = {
            "novelty": 5.0,
            "technical_quality": 6.0,
            "clarity": 4.0,
            "significance": 5.5,
            "reproducibility": 5.0,
            "related_work": 4.5
        }
        
        paper = {"field": "AI", "authors": ["researcher_001"]}
        reviewer = {
            "id": "reviewer_001",
            "specialty": "AI",
            "cognitive_biases": {
                "confirmation": 0.3,
                "halo_effect": 0.4,
                "anchoring": 0.2
            }
        }
        
        adjustments = simulation_scenarios._apply_review_biases(base_scores, paper, reviewer, scenario)
        
        # Should have some adjustments
        assert len(adjustments) > 0
        
        # Confirmation bias should favor matching field
        if "significance" in adjustments:
            assert adjustments["significance"] > 0  # Positive adjustment for matching field
    
    def test_recommendation_determination(self, simulation_scenarios):
        """Test review recommendation determination."""
        venue = {"acceptance_rate": 0.25}  # 25% acceptance rate
        
        # High score should get accept
        high_score_rec = simulation_scenarios._determine_recommendation(8.0, venue)
        assert high_score_rec in ["Accept", "Minor Revision"]
        
        # Low score should get reject
        low_score_rec = simulation_scenarios._determine_recommendation(2.0, venue)
        assert low_score_rec in ["Reject", "Major Revision"]
    
    def test_paper_decision_making(self, simulation_scenarios):
        """Test final paper decision making."""
        venue = {"acceptance_rate": 0.25}
        
        # High average score reviews
        high_reviews = [
            {"overall_score": 7.5},
            {"overall_score": 8.0},
            {"overall_score": 7.8}
        ]
        
        # Low average score reviews
        low_reviews = [
            {"overall_score": 2.0},
            {"overall_score": 2.5},
            {"overall_score": 1.8}
        ]
        
        paper = {"id": "test_paper"}
        
        high_decision = simulation_scenarios._make_paper_decision(paper, venue, high_reviews)
        low_decision = simulation_scenarios._make_paper_decision(paper, venue, low_reviews)
        
        assert high_decision == "Accept"
        assert low_decision == "Reject"
    
    def test_scenario_configuration_validation(self, simulation_scenarios):
        """Test scenario configuration validation."""
        # Valid configuration
        valid_config = ScenarioConfiguration(
            scenario_id="test_valid",
            name="Valid Test",
            description="Valid configuration",
            scenario_type=ScenarioType.BASIC_PEER_REVIEW,
            complexity=ScenarioComplexity.SIMPLE,
            duration_days=30,
            num_researchers=10,
            num_papers=5,
            num_venues=2
        )
        
        # Should not raise exception
        simulation_scenarios._validate_scenario_configuration(valid_config)
        
        # Invalid configurations
        invalid_configs = [
            # Negative researchers
            ScenarioConfiguration(
                scenario_id="test_invalid1", name="Invalid", description="Invalid",
                scenario_type=ScenarioType.BASIC_PEER_REVIEW, complexity=ScenarioComplexity.SIMPLE,
                duration_days=30, num_researchers=-1, num_papers=5, num_venues=2
            ),
            # Zero papers
            ScenarioConfiguration(
                scenario_id="test_invalid2", name="Invalid", description="Invalid",
                scenario_type=ScenarioType.BASIC_PEER_REVIEW, complexity=ScenarioComplexity.SIMPLE,
                duration_days=30, num_researchers=10, num_papers=0, num_venues=2
            ),
            # Negative duration
            ScenarioConfiguration(
                scenario_id="test_invalid3", name="Invalid", description="Invalid",
                scenario_type=ScenarioType.BASIC_PEER_REVIEW, complexity=ScenarioComplexity.SIMPLE,
                duration_days=-10, num_researchers=10, num_papers=5, num_venues=2
            )
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValidationError):
                simulation_scenarios._validate_scenario_configuration(invalid_config)
    
    @patch('src.enhancements.simulation_scenarios.logger')
    def test_logging_during_execution(self, mock_logger, simulation_scenarios):
        """Test that appropriate logging occurs during scenario execution."""
        simulation_scenarios.execute_scenario("basic_peer_review")
        
        # Check that info logs were called
        mock_logger.info.assert_called()
        
        # Check for specific log messages
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Executing scenario" in msg for msg in log_calls)
        assert any("executed successfully" in msg for msg in log_calls)


class TestScenarioConfiguration:
    """Test cases for ScenarioConfiguration dataclass."""
    
    def test_scenario_configuration_creation(self):
        """Test creating a scenario configuration."""
        config = ScenarioConfiguration(
            scenario_id="test_scenario",
            name="Test Scenario",
            description="A test scenario",
            scenario_type=ScenarioType.BASIC_PEER_REVIEW,
            complexity=ScenarioComplexity.SIMPLE,
            duration_days=30,
            num_researchers=20,
            num_papers=10,
            num_venues=3
        )
        
        assert config.scenario_id == "test_scenario"
        assert config.name == "Test Scenario"
        assert config.scenario_type == ScenarioType.BASIC_PEER_REVIEW
        assert config.complexity == ScenarioComplexity.SIMPLE
        assert config.enable_biases  # Default True
        assert config.parameters == {}  # Default empty dict
        assert config.expected_outcomes == {}  # Default empty dict
    
    def test_scenario_configuration_with_custom_parameters(self):
        """Test creating a scenario configuration with custom parameters."""
        custom_params = {"test_param": "test_value", "numeric_param": 42}
        expected_outcomes = {"test_outcome": True, "numeric_outcome": 3.14}
        
        config = ScenarioConfiguration(
            scenario_id="test_scenario",
            name="Test Scenario",
            description="A test scenario",
            scenario_type=ScenarioType.BIAS_DEMONSTRATION,
            complexity=ScenarioComplexity.MODERATE,
            duration_days=60,
            num_researchers=50,
            num_papers=25,
            num_venues=5,
            enable_biases=True,
            enable_networks=False,
            parameters=custom_params,
            expected_outcomes=expected_outcomes
        )
        
        assert config.parameters == custom_params
        assert config.expected_outcomes == expected_outcomes
        assert config.enable_biases
        assert not config.enable_networks


class TestScenarioResults:
    """Test cases for ScenarioResults dataclass."""
    
    def test_scenario_results_creation(self):
        """Test creating scenario results."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)
        
        results = ScenarioResults(
            scenario_id="test_scenario",
            execution_id="exec_123",
            start_time=start_time,
            end_time=end_time,
            success=True,
            total_reviews_generated=30,
            total_papers_processed=10,
            total_researchers_active=20,
            execution_time_seconds=30.0,
            memory_usage_mb=128.5,
            error_count=0
        )
        
        assert results.scenario_id == "test_scenario"
        assert results.execution_id == "exec_123"
        assert results.success
        assert results.total_reviews_generated == 30
        assert results.execution_time_seconds == 30.0
        assert results.scenario_metrics == {}  # Default empty dict
        assert results.validation_results == {}  # Default empty dict
        assert results.execution_log == []  # Default empty list
    
    def test_scenario_results_with_detailed_data(self):
        """Test creating scenario results with detailed data."""
        scenario_metrics = {
            "bias_effects": {"confirmation": 5, "halo_effect": 3},
            "network_effects": {"collaborations": 8}
        }
        
        validation_results = {
            "reviews_completed": True,
            "papers_decided": True,
            "average_quality": False
        }
        
        execution_log = [
            "Scenario started",
            "Processing papers",
            "Scenario completed"
        ]
        
        results = ScenarioResults(
            scenario_id="detailed_test",
            execution_id="exec_456",
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
            total_reviews_generated=45,
            total_papers_processed=15,
            total_researchers_active=30,
            execution_time_seconds=45.5,
            memory_usage_mb=256.0,
            error_count=1,
            scenario_metrics=scenario_metrics,
            validation_results=validation_results,
            execution_log=execution_log
        )
        
        assert results.scenario_metrics == scenario_metrics
        assert results.validation_results == validation_results
        assert results.execution_log == execution_log
        assert results.error_count == 1


if __name__ == "__main__":
    pytest.main([__file__])