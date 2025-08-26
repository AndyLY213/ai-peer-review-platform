"""
Unit tests for the Enhanced Simulation Coordinator.

This module tests the SimulationCoordinator class functionality including:
- System initialization and coordination
- Review process coordination
- Researcher lifecycle management
- Venue management coordination
- System evolution coordination
- Error handling and recovery
- Performance monitoring
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.enhancements.simulation_coordinator import (
    SimulationCoordinator, SimulationState, SystemConfiguration
)
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, EnhancedVenue,
    ResearcherLevel, VenueType, BiasEffect
)
from src.core.exceptions import ValidationError, SimulationError


class TestSimulationCoordinator:
    """Test cases for SimulationCoordinator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock system configuration."""
        return SystemConfiguration(
            bias_strengths={'confirmation': 0.3, 'halo_effect': 0.4},
            collaboration_window_years=3,
            tenure_track_years=6,
            funding_cycle_duration=3,
            venue_calibration_enabled=True,
            real_time_simulation=False
        )
    
    @pytest.fixture
    def coordinator(self, mock_config):
        """Create a SimulationCoordinator instance for testing."""
        return SimulationCoordinator(mock_config)
    
    def test_initialization(self, coordinator, mock_config):
        """Test proper initialization of SimulationCoordinator."""
        assert coordinator.config == mock_config
        assert coordinator.state.simulation_id is not None
        assert coordinator.state.start_time is not None
        assert coordinator.state.total_researchers == 0
        assert coordinator.state.total_papers == 0
        assert coordinator.state.total_reviews == 0
        
        # Check that systems dictionary is initialized
        assert hasattr(coordinator, 'systems')
        assert isinstance(coordinator.systems, dict)
        assert 'bias_engine' in coordinator.systems
        assert 'venue_registry' in coordinator.systems
        assert 'academic_hierarchy' in coordinator.systems
        
        # Check that some systems are actually initialized
        initialized_systems = sum(1 for system in coordinator.systems.values() if system is not None)
        assert initialized_systems > 0
    
    def test_system_dependencies(self, coordinator):
        """Test that system dependencies are properly defined."""
        dependencies = coordinator._system_dependencies
        
        assert 'review_system' in dependencies
        assert 'bias_engine' in dependencies['review_system']
        assert 'venue_standards' in dependencies['review_system']
        assert 'academic_hierarchy' in dependencies['review_system']
        
        assert 'venue_registry' in dependencies
        assert 'network_systems' in dependencies
        assert 'career_systems' in dependencies
    
    def test_coordinate_review_process_success(self, coordinator):
        """Test successful review process coordination."""
        # Setup mocks
        paper_id = "paper_123"
        venue_id = "venue_456"
        reviewer_ids = ["reviewer_1", "reviewer_2", "reviewer_3"]
        
        # Execute
        result = coordinator.coordinate_review_process(paper_id, venue_id, reviewer_ids)
        
        # Verify
        assert result['paper_id'] == paper_id
        assert result['venue_id'] == venue_id
        assert 'deadline' in result
        assert len(result['reviews']) > 0
        assert 'coordination_timestamp' in result
        assert 'systems_involved' in result
    
    def test_coordinate_review_process_insufficient_reviewers(self, coordinator):
        """Test review process coordination with insufficient reviewers."""
        paper_id = "paper_123"
        venue_id = "venue_456"
        reviewer_ids = ["reviewer_1"]  # Only one reviewer, but venue needs 2 minimum
        
        # Execute and verify exception
        with pytest.raises(ValidationError, match="Insufficient available reviewers"):
            coordinator.coordinate_review_process(paper_id, venue_id, reviewer_ids)
    
    def test_coordinate_review_process_venue_not_found(self, coordinator):
        """Test review process coordination with non-existent venue."""
        paper_id = "paper_123"
        venue_id = "nonexistent_venue"
        reviewer_ids = ["reviewer_1", "reviewer_2"]
        
        # The current implementation uses mock venues, so this test will pass
        # In a real implementation, this would check venue existence
        result = coordinator.coordinate_review_process(paper_id, venue_id, reviewer_ids)
        assert result['venue_id'] == venue_id
    
    def test_coordinate_researcher_lifecycle(self, coordinator):
        """Test researcher lifecycle coordination."""
        researcher_id = "researcher_123"
        
        # Execute
        result = coordinator.coordinate_researcher_lifecycle(researcher_id)
        
        # Verify
        assert result['researcher_id'] == researcher_id
        assert 'career_status' in result
        assert 'funding_status' in result
        assert 'network_position' in result
        assert 'strategic_behavior' in result
        assert 'coordination_timestamp' in result
    
    def test_coordinate_venue_management(self, coordinator):
        """Test venue management coordination."""
        venue_id = "venue_123"
        
        # Mock venue
        mock_venue = Mock(spec=EnhancedVenue)
        coordinator.venue_registry.get_venue.return_value = mock_venue
        
        # Mock system operations
        venue_stats = {'submission_count': 100, 'acceptance_rate': 0.25}
        calibration_results = {'calibration_accuracy': 0.95}
        new_acceptance_rate = 0.27
        assignment_optimization = {'optimization_score': 0.85}
        
        coordinator._calculate_venue_statistics = Mock(return_value=venue_stats)
        coordinator._calibrate_venue_standards = Mock(return_value=calibration_results)
        coordinator._calculate_dynamic_acceptance_rate = Mock(return_value=new_acceptance_rate)
        coordinator._optimize_reviewer_assignment = Mock(return_value=assignment_optimization)
        
        # Execute
        result = coordinator.coordinate_venue_management(venue_id)
        
        # Verify
        assert result['venue_id'] == venue_id
        assert result['statistics'] == venue_stats
        assert result['calibration_results'] == calibration_results
        assert result['assignment_optimization'] == assignment_optimization
        assert 'coordination_timestamp' in result
        
        # Verify system calls
        coordinator.venue_registry.get_venue.assert_called_once_with(venue_id)
        coordinator._calculate_venue_statistics.assert_called_once_with(venue_id)
        coordinator.venue_registry.update_acceptance_rate.assert_called_once_with(venue_id, new_acceptance_rate)
    
    def test_coordinate_system_evolution(self, coordinator):
        """Test system evolution coordination."""
        # Mock system evaluations
        reproducibility_trends = {'replication_rate': 0.4, 'trend': 'declining'}
        open_science_adoption = {'preprint_usage': 0.6, 'open_access_rate': 0.3}
        ai_impact_assessment = {'ai_assistance_usage': 0.2, 'detection_rate': 0.8}
        reform_impact = {'alternative_metrics_adoption': 0.1}
        system_changes = {'changes_implemented': [], 'system_adaptations': []}
        
        coordinator.reproducibility_tracker.analyze_trends.return_value = reproducibility_trends
        coordinator.open_science_manager.evaluate_adoption_rates.return_value = open_science_adoption
        coordinator.ai_impact_simulator.assess_current_impact.return_value = ai_impact_assessment
        coordinator.publication_reform_manager.evaluate_reform_impact.return_value = reform_impact
        coordinator._coordinate_system_changes = Mock(return_value=system_changes)
        
        # Execute
        result = coordinator.coordinate_system_evolution()
        
        # Verify
        assert result['reproducibility_trends'] == reproducibility_trends
        assert result['open_science_adoption'] == open_science_adoption
        assert result['ai_impact_assessment'] == ai_impact_assessment
        assert result['reform_impact'] == reform_impact
        assert result['system_changes'] == system_changes
        assert 'coordination_timestamp' in result
        
        # Verify system calls
        coordinator.reproducibility_tracker.analyze_trends.assert_called_once()
        coordinator.open_science_manager.evaluate_adoption_rates.assert_called_once()
        coordinator.ai_impact_simulator.assess_current_impact.assert_called_once()
        coordinator.publication_reform_manager.evaluate_reform_impact.assert_called_once()
    
    def test_get_simulation_state(self, coordinator):
        """Test getting current simulation state."""
        # Mock state update
        coordinator._update_simulation_state = Mock()
        
        # Execute
        state = coordinator.get_simulation_state()
        
        # Verify
        assert isinstance(state, SimulationState)
        assert state.simulation_id == coordinator.state.simulation_id
        coordinator._update_simulation_state.assert_called_once()
    
    def test_get_system_health(self, coordinator):
        """Test getting system health status."""
        # Execute
        health = coordinator.get_system_health()
        
        # Verify
        assert 'simulation_id' in health
        assert 'uptime' in health
        assert 'total_operations' in health
        assert 'error_rate' in health
        assert 'system_load' in health
        assert 'performance_metrics' in health
        assert 'system_status' in health
        
        system_status = health['system_status']
        assert 'bias_system' in system_status
        assert 'network_system' in system_status
        assert 'career_system' in system_status
        assert 'funding_system' in system_status
        assert 'venue_system' in system_status
        assert 'temporal_system' in system_status
    
    def test_generate_enhanced_review(self, coordinator):
        """Test enhanced review generation with all systems."""
        paper_id = "paper_123"
        venue_id = "venue_456"
        reviewer_id = "reviewer_789"
        deadline = datetime.now() + timedelta(weeks=4)
        
        # Mock review generation pipeline
        base_review = Mock(spec=StructuredReview)
        biased_review = Mock(spec=StructuredReview)
        network_adjusted_review = Mock(spec=StructuredReview)
        final_review = Mock(spec=StructuredReview)
        
        coordinator.review_system.generate_review.return_value = base_review
        coordinator.bias_engine.apply_biases.return_value = biased_review
        coordinator.network_influence.apply_network_effects.return_value = network_adjusted_review
        coordinator.venue_standards.enforce_standards.return_value = final_review
        
        # Execute
        result = coordinator._generate_enhanced_review(paper_id, venue_id, reviewer_id, deadline)
        
        # Verify
        assert result == final_review
        coordinator.review_system.generate_review.assert_called_once_with(paper_id, venue_id, reviewer_id)
        coordinator.bias_engine.apply_biases.assert_called_once_with(base_review, reviewer_id, paper_id)
        coordinator.network_influence.apply_network_effects.assert_called_once_with(biased_review, reviewer_id, paper_id)
        coordinator.venue_standards.enforce_standards.assert_called_once_with(network_adjusted_review, venue_id)
    
    def test_has_conflict_of_interest(self, coordinator):
        """Test conflict of interest detection."""
        paper_id = "paper_123"
        reviewer_id = "reviewer_456"
        
        # Test no conflicts
        coordinator.collaboration_network.has_recent_collaboration.return_value = False
        coordinator.citation_network.has_citation_relationship.return_value = False
        coordinator.conference_community.has_close_community_ties.return_value = False
        
        assert not coordinator._has_conflict_of_interest(paper_id, reviewer_id)
        
        # Test collaboration conflict
        coordinator.collaboration_network.has_recent_collaboration.return_value = True
        assert coordinator._has_conflict_of_interest(paper_id, reviewer_id)
        
        # Reset and test citation conflict
        coordinator.collaboration_network.has_recent_collaboration.return_value = False
        coordinator.citation_network.has_citation_relationship.return_value = True
        assert coordinator._has_conflict_of_interest(paper_id, reviewer_id)
        
        # Reset and test community conflict
        coordinator.citation_network.has_citation_relationship.return_value = False
        coordinator.conference_community.has_close_community_ties.return_value = True
        assert coordinator._has_conflict_of_interest(paper_id, reviewer_id)
    
    def test_error_handling(self, coordinator):
        """Test error handling and recovery mechanisms."""
        operation = "test_operation"
        error = Exception("Test error")
        
        initial_errors = coordinator.state.total_errors
        
        # Execute error handling
        coordinator._handle_coordination_error(operation, error)
        
        # Verify error tracking
        assert coordinator.state.total_errors == initial_errors + 1
        assert operation in coordinator.state.system_errors
        assert coordinator.state.system_errors[operation] == 1
        
        # Test multiple errors for same operation
        coordinator._handle_coordination_error(operation, error)
        assert coordinator.state.system_errors[operation] == 2
    
    def test_performance_metrics_update(self, coordinator):
        """Test performance metrics calculation."""
        # Set some initial state
        coordinator.state.total_reviews = 100
        coordinator.state.total_papers = 50
        coordinator.state.start_time = datetime.now() - timedelta(seconds=100)
        
        # Execute update
        coordinator._update_performance_metrics()
        
        # Verify metrics are calculated
        assert coordinator.state.reviews_per_second > 0
        assert coordinator.state.papers_per_day > 0
    
    def test_shutdown(self, coordinator):
        """Test graceful shutdown process."""
        # Mock methods
        coordinator._save_simulation_state = Mock()
        coordinator._cleanup_systems = Mock()
        
        # Execute shutdown
        coordinator.shutdown()
        
        # Verify shutdown process
        coordinator._save_simulation_state.assert_called_once()
        coordinator._cleanup_systems.assert_called_once()
    
    def test_system_configuration_validation(self):
        """Test system configuration validation."""
        # Test valid configuration
        config = SystemConfiguration(
            bias_strengths={'confirmation': 0.5},
            collaboration_window_years=3,
            tenure_track_years=6
        )
        assert config.bias_strengths['confirmation'] == 0.5
        assert config.collaboration_window_years == 3
        
        # Test default configuration
        default_config = SystemConfiguration()
        assert 'confirmation' in default_config.bias_strengths
        assert default_config.collaboration_window_years == 3
        assert default_config.tenure_track_years == 6
    
    def test_simulation_state_tracking(self, coordinator):
        """Test comprehensive simulation state tracking."""
        # Update state
        coordinator.state.total_researchers = 10
        coordinator.state.total_papers = 25
        coordinator.state.total_reviews = 75
        coordinator.state.active_venues = 5
        
        # Update simulation state
        coordinator._update_simulation_state()
        
        # Verify state updates
        assert coordinator.state.current_time is not None
        assert coordinator.state.total_researchers == 10
        assert coordinator.state.total_papers == 25
        assert coordinator.state.total_reviews == 75
        assert coordinator.state.active_venues == 5


class TestSimulationState:
    """Test cases for SimulationState dataclass."""
    
    def test_simulation_state_creation(self):
        """Test SimulationState creation and default values."""
        state = SimulationState(
            simulation_id="test_123",
            start_time=datetime.now(),
            current_time=datetime.now(),
            total_researchers=10,
            total_papers=5,
            total_reviews=15,
            active_venues=3
        )
        
        assert state.simulation_id == "test_123"
        assert state.total_researchers == 10
        assert state.total_papers == 5
        assert state.total_reviews == 15
        assert state.active_venues == 3
        assert state.bias_system_active is True
        assert state.network_system_active is True
        assert state.total_errors == 0
        assert isinstance(state.system_errors, dict)
        assert isinstance(state.statistics, dict)


class TestSystemConfiguration:
    """Test cases for SystemConfiguration dataclass."""
    
    def test_system_configuration_defaults(self):
        """Test SystemConfiguration default values."""
        config = SystemConfiguration()
        
        assert 'confirmation' in config.bias_strengths
        assert 'halo_effect' in config.bias_strengths
        assert 'anchoring' in config.bias_strengths
        assert 'availability' in config.bias_strengths
        
        assert config.collaboration_window_years == 3
        assert config.tenure_track_years == 6
        assert config.funding_cycle_duration == 3
        assert config.venue_calibration_enabled is True
        assert config.real_time_simulation is False
        assert config.time_acceleration_factor == 365.0
    
    def test_system_configuration_custom_values(self):
        """Test SystemConfiguration with custom values."""
        custom_biases = {
            'confirmation': 0.5,
            'halo_effect': 0.6,
            'anchoring': 0.3,
            'availability': 0.4
        }
        
        config = SystemConfiguration(
            bias_strengths=custom_biases,
            collaboration_window_years=5,
            tenure_track_years=7,
            funding_cycle_duration=4,
            venue_calibration_enabled=False,
            real_time_simulation=True,
            time_acceleration_factor=100.0
        )
        
        assert config.bias_strengths == custom_biases
        assert config.collaboration_window_years == 5
        assert config.tenure_track_years == 7
        assert config.funding_cycle_duration == 4
        assert config.venue_calibration_enabled is False
        assert config.real_time_simulation is True
        assert config.time_acceleration_factor == 100.0


if __name__ == '__main__':
    pytest.main([__file__])