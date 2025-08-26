"""
Unit tests for Resource Constraint Manager

Tests the ResourceConstraint, StudentFunding, CollaborationIncentive, 
ResearcherResourceProfile, and ResourceConstraintManager classes functionality
including resource allocation, student funding, collaboration incentives,
and research output impact calculations.
"""

import pytest
from datetime import date, timedelta
from typing import Dict, List

from src.enhancements.resource_constraint_manager import (
    ResourceConstraintManager, ResourceConstraint, StudentFunding, 
    CollaborationIncentive, ResearcherResourceProfile,
    ResourceType, ResourceStatus, CollaborationType
)
from src.data.enhanced_models import ResearcherLevel
from src.core.exceptions import ValidationError


class TestResourceConstraint:
    """Test cases for ResourceConstraint class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constraint = ResourceConstraint(
            resource_type=ResourceType.LAB_EQUIPMENT,
            availability_level=ResourceStatus.ADEQUATE,
            cost_per_unit=5000.0,
            maintenance_cost_per_month=200.0,
            utilization_capacity=100,
            current_utilization=30,
            shared_access=True,
            institutional_priority=4
        )
    
    def test_initialization(self):
        """Test ResourceConstraint initialization."""
        assert self.constraint.resource_type == ResourceType.LAB_EQUIPMENT
        assert self.constraint.availability_level == ResourceStatus.ADEQUATE
        assert self.constraint.cost_per_unit == 5000.0
        assert self.constraint.utilization_capacity == 100
        assert self.constraint.current_utilization == 30
        assert self.constraint.shared_access is True
        assert self.constraint.institutional_priority == 4
        assert len(self.constraint.waiting_list) == 0
    
    def test_initialization_invalid_utilization(self):
        """Test ResourceConstraint initialization with invalid utilization."""
        with pytest.raises(ValidationError):
            ResourceConstraint(
                utilization_capacity=50,
                current_utilization=60  # Invalid: exceeds capacity
            )
    
    def test_initialization_invalid_priority(self):
        """Test ResourceConstraint initialization with invalid priority."""
        with pytest.raises(ValidationError):
            ResourceConstraint(
                institutional_priority=6  # Invalid: must be 1-5
            )
    
    def test_get_availability_score_adequate(self):
        """Test availability score calculation for adequate resources."""
        # 30% utilization with adequate status
        score = self.constraint.get_availability_score()
        expected = 0.8 * (1.0 - 0.3)  # 0.8 base * 0.7 availability
        assert abs(score - expected) < 0.001
    
    def test_get_availability_score_abundant(self):
        """Test availability score calculation for abundant resources."""
        self.constraint.availability_level = ResourceStatus.ABUNDANT
        self.constraint.current_utilization = 10  # 10% utilization
        
        score = self.constraint.get_availability_score()
        expected = 1.0 * (1.0 - 0.1)  # 1.0 base * 0.9 availability
        assert abs(score - expected) < 0.001
    
    def test_get_availability_score_unavailable(self):
        """Test availability score calculation for unavailable resources."""
        self.constraint.availability_level = ResourceStatus.UNAVAILABLE
        
        score = self.constraint.get_availability_score()
        assert score == 0.0
    
    def test_get_availability_score_full_utilization(self):
        """Test availability score with full utilization."""
        self.constraint.current_utilization = 100  # Full capacity
        
        score = self.constraint.get_availability_score()
        assert score == 0.0  # No availability when fully utilized
    
    def test_can_allocate_success(self):
        """Test can_allocate with available capacity."""
        # Current: 30, Capacity: 100, Request: 50
        assert self.constraint.can_allocate(50) is True
        assert self.constraint.can_allocate(70) is True  # Exactly at capacity
    
    def test_can_allocate_failure_capacity(self):
        """Test can_allocate exceeding capacity."""
        # Current: 30, Capacity: 100, Request: 80 (would exceed)
        assert self.constraint.can_allocate(80) is False
    
    def test_can_allocate_failure_unavailable(self):
        """Test can_allocate with unavailable resource."""
        self.constraint.availability_level = ResourceStatus.UNAVAILABLE
        assert self.constraint.can_allocate(10) is False
    
    def test_allocate_resource_success(self):
        """Test successful resource allocation."""
        initial_utilization = self.constraint.current_utilization
        
        result = self.constraint.allocate_resource("researcher_001", 20)
        
        assert result is True
        assert self.constraint.current_utilization == initial_utilization + 20
        assert "researcher_001" not in self.constraint.waiting_list
    
    def test_allocate_resource_failure_adds_to_waiting_list(self):
        """Test failed allocation adds researcher to waiting list."""
        result = self.constraint.allocate_resource("researcher_001", 80)  # Would exceed capacity
        
        assert result is False
        assert self.constraint.current_utilization == 30  # Unchanged
        assert "researcher_001" in self.constraint.waiting_list
    
    def test_allocate_resource_removes_from_waiting_list(self):
        """Test successful allocation removes researcher from waiting list."""
        # First add to waiting list
        self.constraint.waiting_list.append("researcher_001")
        
        # Then successfully allocate
        result = self.constraint.allocate_resource("researcher_001", 20)
        
        assert result is True
        assert "researcher_001" not in self.constraint.waiting_list
    
    def test_release_resource_partial(self):
        """Test partial resource release."""
        initial_utilization = self.constraint.current_utilization
        
        released = self.constraint.release_resource(10)
        
        assert released == 10
        assert self.constraint.current_utilization == initial_utilization - 10
    
    def test_release_resource_exceeds_current(self):
        """Test releasing more than currently allocated."""
        initial_utilization = self.constraint.current_utilization
        
        released = self.constraint.release_resource(50)  # More than current 30
        
        assert released == initial_utilization  # Should release all available
        assert self.constraint.current_utilization == 0


class TestStudentFunding:
    """Test cases for StudentFunding class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.funding = StudentFunding(
            researcher_id="researcher_001",
            total_budget=100000.0,
            allocated_budget=20000.0,
            max_students=3,
            current_students=1,
            funding_duration_months=12,
            stipend_per_student_per_month=2000.0,
            overhead_rate=0.3
        )
    
    def test_initialization(self):
        """Test StudentFunding initialization."""
        assert self.funding.researcher_id == "researcher_001"
        assert self.funding.total_budget == 100000.0
        assert self.funding.allocated_budget == 20000.0
        assert self.funding.max_students == 3
        assert self.funding.current_students == 1
        assert self.funding.stipend_per_student_per_month == 2000.0
        assert self.funding.overhead_rate == 0.3
    
    def test_initialization_invalid_budget(self):
        """Test StudentFunding initialization with invalid budget allocation."""
        with pytest.raises(ValidationError):
            StudentFunding(
                total_budget=50000.0,
                allocated_budget=60000.0  # Invalid: exceeds total
            )
    
    def test_initialization_invalid_students(self):
        """Test StudentFunding initialization with invalid student count."""
        with pytest.raises(ValidationError):
            StudentFunding(
                max_students=2,
                current_students=3  # Invalid: exceeds max
            )
    
    def test_get_available_budget(self):
        """Test available budget calculation."""
        available = self.funding.get_available_budget()
        assert available == 80000.0  # 100000 - 20000
    
    def test_can_fund_student_success(self):
        """Test can_fund_student with sufficient resources."""
        can_fund, reason = self.funding.can_fund_student()
        
        assert can_fund is True
        assert "Can fund student" in reason
    
    def test_can_fund_student_max_capacity(self):
        """Test can_fund_student at maximum student capacity."""
        self.funding.current_students = 3  # At max capacity
        
        can_fund, reason = self.funding.can_fund_student()
        
        assert can_fund is False
        assert "Maximum student capacity" in reason
    
    def test_can_fund_student_insufficient_budget(self):
        """Test can_fund_student with insufficient budget."""
        self.funding.allocated_budget = 95000.0  # Only 5000 left
        
        can_fund, reason = self.funding.can_fund_student()
        
        assert can_fund is False
        assert "Insufficient budget" in reason
    
    def test_fund_student_success(self):
        """Test successful student funding."""
        initial_students = self.funding.current_students
        initial_allocated = self.funding.allocated_budget
        
        result = self.funding.fund_student()
        
        assert result is True
        assert self.funding.current_students == initial_students + 1
        
        # Check budget allocation (2000 * 1.3 * 12 = 31200)
        expected_cost = 2000.0 * (1 + 0.3) * 12
        assert abs(self.funding.allocated_budget - (initial_allocated + expected_cost)) < 0.01
    
    def test_fund_student_failure(self):
        """Test failed student funding."""
        self.funding.current_students = 3  # At max capacity
        initial_allocated = self.funding.allocated_budget
        
        result = self.funding.fund_student()
        
        assert result is False
        assert self.funding.current_students == 3  # Unchanged
        assert self.funding.allocated_budget == initial_allocated  # Unchanged
    
    def test_release_student_success(self):
        """Test successful student release."""
        initial_students = self.funding.current_students
        initial_allocated = self.funding.allocated_budget
        
        result = self.funding.release_student()
        
        assert result is True
        assert self.funding.current_students == initial_students - 1
        assert self.funding.allocated_budget < initial_allocated  # Should decrease
    
    def test_release_student_no_students(self):
        """Test releasing student when none are funded."""
        self.funding.current_students = 0
        initial_allocated = self.funding.allocated_budget
        
        result = self.funding.release_student()
        
        assert result is False
        assert self.funding.current_students == 0
        assert self.funding.allocated_budget == initial_allocated  # Unchanged


class TestCollaborationIncentive:
    """Test cases for CollaborationIncentive class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.incentive = CollaborationIncentive(
            collaboration_type=CollaborationType.EXTERNAL,
            funding_multiplier=1.3,
            resource_sharing_bonus=0.4,
            publication_bonus=1.2,
            min_collaborators=2,
            max_collaborators=4,
            coordination_overhead=0.15
        )
    
    def test_initialization(self):
        """Test CollaborationIncentive initialization."""
        assert self.incentive.collaboration_type == CollaborationType.EXTERNAL
        assert self.incentive.funding_multiplier == 1.3
        assert self.incentive.resource_sharing_bonus == 0.4
        assert self.incentive.publication_bonus == 1.2
        assert self.incentive.min_collaborators == 2
        assert self.incentive.max_collaborators == 4
        assert self.incentive.coordination_overhead == 0.15
    
    def test_initialization_invalid_collaborators(self):
        """Test CollaborationIncentive initialization with invalid collaborator range."""
        with pytest.raises(ValidationError):
            CollaborationIncentive(
                min_collaborators=5,
                max_collaborators=3  # Invalid: min > max
            )
    
    def test_initialization_invalid_overhead(self):
        """Test CollaborationIncentive initialization with invalid overhead."""
        with pytest.raises(ValidationError):
            CollaborationIncentive(
                coordination_overhead=1.5  # Invalid: must be 0-1
            )
    
    def test_calculate_net_benefit_valid_size(self):
        """Test net benefit calculation for valid collaboration size."""
        # 3 collaborators (within 2-4 range)
        benefit = self.incentive.calculate_net_benefit(3)
        
        # Expected: (0.3 + 0.4 + 0.2) / (1 + 0.15 * 2) = 0.9 / 1.3
        expected = 0.9 / 1.3
        assert abs(benefit - expected) < 0.001
    
    def test_calculate_net_benefit_minimum_size(self):
        """Test net benefit calculation for minimum collaboration size."""
        benefit = self.incentive.calculate_net_benefit(2)
        
        # Expected: (0.3 + 0.4 + 0.2) / (1 + 0.15 * 1) = 0.9 / 1.15
        expected = 0.9 / 1.15
        assert abs(benefit - expected) < 0.001
    
    def test_calculate_net_benefit_maximum_size(self):
        """Test net benefit calculation for maximum collaboration size."""
        benefit = self.incentive.calculate_net_benefit(4)
        
        # Expected: (0.3 + 0.4 + 0.2) / (1 + 0.15 * 3) = 0.9 / 1.45
        expected = 0.9 / 1.45
        assert abs(benefit - expected) < 0.001
    
    def test_calculate_net_benefit_invalid_size(self):
        """Test net benefit calculation for invalid collaboration size."""
        # Too few collaborators
        assert self.incentive.calculate_net_benefit(1) == 0.0
        
        # Too many collaborators
        assert self.incentive.calculate_net_benefit(5) == 0.0
    
    def test_is_collaboration_viable_true(self):
        """Test collaboration viability for beneficial collaboration."""
        assert self.incentive.is_collaboration_viable(3) is True
    
    def test_is_collaboration_viable_false(self):
        """Test collaboration viability for non-beneficial collaboration."""
        # Test with invalid collaboration size (outside min/max range)
        assert self.incentive.is_collaboration_viable(1) is False  # Too few
        assert self.incentive.is_collaboration_viable(6) is False  # Too many


class TestResearcherResourceProfile:
    """Test cases for ResearcherResourceProfile class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profile = ResearcherResourceProfile(
            researcher_id="researcher_001",
            career_level=ResearcherLevel.ASSISTANT_PROF
        )
        
        # Create some test constraints
        self.constraints = {
            "constraint_1": ResourceConstraint(
                resource_type=ResourceType.LAB_EQUIPMENT,
                availability_level=ResourceStatus.ADEQUATE,
                utilization_capacity=100,
                current_utilization=30
            ),
            "constraint_2": ResourceConstraint(
                resource_type=ResourceType.COMPUTING_RESOURCES,
                availability_level=ResourceStatus.ABUNDANT,
                utilization_capacity=200,
                current_utilization=50
            )
        }
    
    def test_initialization(self):
        """Test ResearcherResourceProfile initialization."""
        assert self.profile.researcher_id == "researcher_001"
        assert self.profile.career_level == ResearcherLevel.ASSISTANT_PROF
        assert len(self.profile.resource_needs) > 0  # Should have default needs
        assert self.profile.resource_efficiency == 1.0
        assert self.profile.priority_score == 0.5
    
    def test_default_resource_needs_assistant_prof(self):
        """Test default resource needs for assistant professor."""
        needs = self.profile.resource_needs
        
        # Should have needs for all resource types
        assert ResourceType.LAB_EQUIPMENT in needs
        assert ResourceType.COMPUTING_RESOURCES in needs
        assert ResourceType.STUDENT_FUNDING in needs
        
        # Assistant prof should have base multiplier (1.0)
        assert needs[ResourceType.LAB_EQUIPMENT] == 10  # Base value
    
    def test_default_resource_needs_full_prof(self):
        """Test default resource needs for full professor."""
        full_prof_profile = ResearcherResourceProfile(
            researcher_id="full_prof_001",
            career_level=ResearcherLevel.FULL_PROF
        )
        
        needs = full_prof_profile.resource_needs
        
        # Full prof should have 2x multiplier
        assert needs[ResourceType.LAB_EQUIPMENT] == 20  # 10 * 2.0
        assert needs[ResourceType.COMPUTING_RESOURCES] == 40  # 20 * 2.0
    
    def test_default_resource_needs_graduate_student(self):
        """Test default resource needs for graduate student."""
        grad_profile = ResearcherResourceProfile(
            researcher_id="grad_001",
            career_level=ResearcherLevel.GRADUATE_STUDENT
        )
        
        needs = grad_profile.resource_needs
        
        # Graduate student should have 0.3x multiplier
        assert needs[ResourceType.LAB_EQUIPMENT] == 3  # 10 * 0.3
        assert needs[ResourceType.COMPUTING_RESOURCES] == 6  # 20 * 0.3
    
    def test_get_total_resource_satisfaction_full(self):
        """Test resource satisfaction calculation with full availability."""
        satisfaction = self.profile.get_total_resource_satisfaction(self.constraints)
        
        # Should be > 0 since we have some constraints available
        assert 0.0 < satisfaction <= 1.0
    
    def test_get_total_resource_satisfaction_no_constraints(self):
        """Test resource satisfaction with no constraints available."""
        satisfaction = self.profile.get_total_resource_satisfaction({})
        
        # Should be 0 when no constraints are available
        assert satisfaction == 0.0
    
    def test_calculate_productivity_impact_high_satisfaction(self):
        """Test productivity impact with high resource satisfaction."""
        impact = self.profile.calculate_productivity_impact(0.9)
        
        # High satisfaction should give full productivity
        assert impact == 1.0
    
    def test_calculate_productivity_impact_medium_satisfaction(self):
        """Test productivity impact with medium resource satisfaction."""
        impact = self.profile.calculate_productivity_impact(0.5)
        
        # Medium satisfaction should reduce productivity
        assert 0.5 < impact < 1.0
    
    def test_calculate_productivity_impact_low_satisfaction(self):
        """Test productivity impact with low resource satisfaction."""
        impact = self.profile.calculate_productivity_impact(0.1)
        
        # Low satisfaction should significantly reduce productivity
        assert impact < 0.5
    
    def test_calculate_productivity_impact_with_efficiency(self):
        """Test productivity impact with resource efficiency modifier."""
        self.profile.resource_efficiency = 1.5
        
        impact = self.profile.calculate_productivity_impact(0.5)
        
        # Should be higher due to efficiency multiplier
        base_impact = 0.7  # Expected base for 0.5 satisfaction
        expected = min(1.0, base_impact * 1.5)
        assert abs(impact - expected) < 0.001


class TestResourceConstraintManager:
    """Test cases for ResourceConstraintManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ResourceConstraintManager()
    
    def test_initialization(self):
        """Test ResourceConstraintManager initialization."""
        # Should have default constraints
        assert len(self.manager.constraints) > 0
        
        # Should have default collaboration incentives
        assert len(self.manager.collaboration_incentives) > 0
        
        # Should start with no researcher profiles
        assert len(self.manager.researcher_profiles) == 0
        
        # Should start with no active collaborations
        assert len(self.manager.active_collaborations) == 0
    
    def test_default_constraints_created(self):
        """Test that default resource constraints are created."""
        # Check for lab equipment constraint
        lab_constraints = self.manager.get_constraints_by_type(ResourceType.LAB_EQUIPMENT)
        assert len(lab_constraints) >= 1
        
        # Check for computing resources constraint
        computing_constraints = self.manager.get_constraints_by_type(ResourceType.COMPUTING_RESOURCES)
        assert len(computing_constraints) >= 1
        
        # Check for research materials constraint
        materials_constraints = self.manager.get_constraints_by_type(ResourceType.RESEARCH_MATERIALS)
        assert len(materials_constraints) >= 1
    
    def test_default_incentives_created(self):
        """Test that default collaboration incentives are created."""
        # Should have external, international, and industry incentives
        incentive_types = [inc.collaboration_type for inc in self.manager.collaboration_incentives.values()]
        
        assert CollaborationType.EXTERNAL in incentive_types
        assert CollaborationType.INTERNATIONAL in incentive_types
        assert CollaborationType.INDUSTRY in incentive_types
    
    def test_add_constraint(self):
        """Test adding a new resource constraint."""
        new_constraint = ResourceConstraint(
            resource_type=ResourceType.TRAVEL_FUNDING,
            availability_level=ResourceStatus.LIMITED,
            utilization_capacity=20
        )
        
        constraint_id = self.manager.add_constraint(new_constraint)
        
        assert constraint_id == new_constraint.constraint_id
        assert self.manager.get_constraint(constraint_id) == new_constraint
    
    def test_get_constraint_valid(self):
        """Test getting constraint by valid ID."""
        # Get a default constraint
        lab_constraints = self.manager.get_constraints_by_type(ResourceType.LAB_EQUIPMENT)
        constraint = lab_constraints[0]
        
        retrieved = self.manager.get_constraint(constraint.constraint_id)
        assert retrieved == constraint
    
    def test_get_constraint_invalid(self):
        """Test getting constraint by invalid ID."""
        retrieved = self.manager.get_constraint("invalid_id")
        assert retrieved is None
    
    def test_get_constraints_by_type(self):
        """Test getting constraints by resource type."""
        lab_constraints = self.manager.get_constraints_by_type(ResourceType.LAB_EQUIPMENT)
        
        assert len(lab_constraints) >= 1
        assert all(c.resource_type == ResourceType.LAB_EQUIPMENT for c in lab_constraints)
    
    def test_add_researcher_profile_assistant_prof(self):
        """Test adding researcher profile for assistant professor."""
        researcher_id = self.manager.add_researcher_profile(
            "researcher_001", 
            ResearcherLevel.ASSISTANT_PROF
        )
        
        assert researcher_id == "researcher_001"
        
        profile = self.manager.get_researcher_profile(researcher_id)
        assert profile is not None
        assert profile.career_level == ResearcherLevel.ASSISTANT_PROF
        assert profile.student_funding is not None  # Should have student funding
        assert profile.student_funding.total_budget == 75000  # Assistant prof amount
    
    def test_add_researcher_profile_graduate_student(self):
        """Test adding researcher profile for graduate student."""
        researcher_id = self.manager.add_researcher_profile(
            "grad_001", 
            ResearcherLevel.GRADUATE_STUDENT
        )
        
        profile = self.manager.get_researcher_profile(researcher_id)
        assert profile is not None
        assert profile.career_level == ResearcherLevel.GRADUATE_STUDENT
        assert profile.student_funding is None  # Should not have student funding
    
    def test_add_researcher_profile_full_prof(self):
        """Test adding researcher profile for full professor."""
        researcher_id = self.manager.add_researcher_profile(
            "full_prof_001", 
            ResearcherLevel.FULL_PROF
        )
        
        profile = self.manager.get_researcher_profile(researcher_id)
        assert profile is not None
        assert profile.student_funding is not None
        assert profile.student_funding.total_budget == 250000  # Full prof amount
        assert profile.student_funding.max_students == 6  # Full prof capacity
    
    def test_add_researcher_profile_custom_needs(self):
        """Test adding researcher profile with custom resource needs."""
        custom_needs = {
            ResourceType.LAB_EQUIPMENT: 50,
            ResourceType.COMPUTING_RESOURCES: 100
        }
        
        researcher_id = self.manager.add_researcher_profile(
            "custom_001", 
            ResearcherLevel.ASSISTANT_PROF,
            custom_needs
        )
        
        profile = self.manager.get_researcher_profile(researcher_id)
        assert profile.resource_needs == custom_needs
    
    def test_allocate_resources_success(self):
        """Test successful resource allocation."""
        # Add researcher
        researcher_id = self.manager.add_researcher_profile(
            "researcher_001", 
            ResearcherLevel.ASSISTANT_PROF
        )
        
        # Request resources
        requests = {
            ResourceType.LAB_EQUIPMENT: 10,
            ResourceType.COMPUTING_RESOURCES: 20
        }
        
        results = self.manager.allocate_resources(researcher_id, requests)
        
        # Should succeed for available resources
        assert results[ResourceType.LAB_EQUIPMENT] is True
        assert results[ResourceType.COMPUTING_RESOURCES] is True
        
        # Check that allocations are tracked
        profile = self.manager.get_researcher_profile(researcher_id)
        assert len(profile.current_allocations) > 0
    
    def test_allocate_resources_partial_failure(self):
        """Test resource allocation with some failures."""
        # Add researcher
        researcher_id = self.manager.add_researcher_profile(
            "researcher_001", 
            ResearcherLevel.ASSISTANT_PROF
        )
        
        # Request excessive resources for scarce materials
        requests = {
            ResourceType.LAB_EQUIPMENT: 10,  # Should succeed
            ResourceType.RESEARCH_MATERIALS: 50  # Should fail (capacity is 30)
        }
        
        results = self.manager.allocate_resources(researcher_id, requests)
        
        # Lab equipment should succeed, materials should fail
        assert results[ResourceType.LAB_EQUIPMENT] is True
        assert results[ResourceType.RESEARCH_MATERIALS] is False
    
    def test_allocate_resources_no_profile(self):
        """Test resource allocation for non-existent researcher."""
        requests = {ResourceType.LAB_EQUIPMENT: 10}
        
        results = self.manager.allocate_resources("nonexistent", requests)
        
        # Should fail for all requests
        assert all(not success for success in results.values())
    
    def test_release_resources(self):
        """Test resource release."""
        # Add researcher and allocate resources
        researcher_id = self.manager.add_researcher_profile(
            "researcher_001", 
            ResearcherLevel.ASSISTANT_PROF
        )
        
        requests = {ResourceType.LAB_EQUIPMENT: 10}
        self.manager.allocate_resources(researcher_id, requests)
        
        # Get constraint ID for release
        profile = self.manager.get_researcher_profile(researcher_id)
        constraint_id = list(profile.current_allocations.keys())[0]
        
        # Release resources
        releases = {constraint_id: 5}
        results = self.manager.release_resources(researcher_id, releases)
        
        assert results[constraint_id] == 5
        assert profile.current_allocations[constraint_id] == 5  # 10 - 5 = 5
    
    def test_create_collaboration_success(self):
        """Test successful collaboration creation."""
        # Add researchers
        researcher_ids = []
        for i in range(3):
            researcher_id = self.manager.add_researcher_profile(
                f"researcher_{i:03d}", 
                ResearcherLevel.ASSISTANT_PROF
            )
            researcher_ids.append(researcher_id)
        
        # Create collaboration
        collab_id = self.manager.create_collaboration(
            researcher_ids, 
            CollaborationType.EXTERNAL
        )
        
        assert collab_id is not None
        assert collab_id in self.manager.active_collaborations
        assert self.manager.active_collaborations[collab_id] == researcher_ids
        
        # Check that researcher profiles are updated
        for researcher_id in researcher_ids:
            profile = self.manager.get_researcher_profile(researcher_id)
            assert collab_id in profile.collaboration_history
            assert profile.resource_efficiency > 1.0  # Should have efficiency bonus
    
    def test_create_collaboration_too_few_researchers(self):
        """Test collaboration creation with too few researchers."""
        researcher_id = self.manager.add_researcher_profile(
            "researcher_001", 
            ResearcherLevel.ASSISTANT_PROF
        )
        
        collab_id = self.manager.create_collaboration(
            [researcher_id], 
            CollaborationType.EXTERNAL
        )
        
        assert collab_id is None
    
    def test_create_collaboration_no_matching_incentive(self):
        """Test collaboration creation with no matching incentive."""
        # Add many researchers (more than max for any incentive)
        researcher_ids = []
        for i in range(10):  # More than any incentive allows
            researcher_id = self.manager.add_researcher_profile(
                f"researcher_{i:03d}", 
                ResearcherLevel.ASSISTANT_PROF
            )
            researcher_ids.append(researcher_id)
        
        collab_id = self.manager.create_collaboration(
            researcher_ids, 
            CollaborationType.EXTERNAL
        )
        
        assert collab_id is None
    
    def test_calculate_research_output_impact(self):
        """Test research output impact calculation."""
        # Add researcher
        researcher_id = self.manager.add_researcher_profile(
            "researcher_001", 
            ResearcherLevel.ASSISTANT_PROF
        )
        
        # Allocate some resources
        requests = {ResourceType.LAB_EQUIPMENT: 5}
        self.manager.allocate_resources(researcher_id, requests)
        
        impact = self.manager.calculate_research_output_impact(researcher_id)
        
        # Should have all expected keys
        assert "productivity_multiplier" in impact
        assert "resource_satisfaction" in impact
        assert "student_funding_impact" in impact
        assert "collaboration_impact" in impact
        assert "base_productivity" in impact
        
        # Values should be reasonable
        assert 0.0 <= impact["productivity_multiplier"] <= 2.0
        assert 0.0 <= impact["resource_satisfaction"] <= 1.0
        assert impact["student_funding_impact"] >= 0.8  # Should have some student funding
        assert impact["collaboration_impact"] == 1.0  # No collaborations yet
    
    def test_calculate_research_output_impact_with_collaboration(self):
        """Test research output impact with active collaboration."""
        # Add researchers and create collaboration
        researcher_ids = []
        for i in range(3):
            researcher_id = self.manager.add_researcher_profile(
                f"researcher_{i:03d}", 
                ResearcherLevel.ASSISTANT_PROF
            )
            researcher_ids.append(researcher_id)
        
        self.manager.create_collaboration(researcher_ids, CollaborationType.EXTERNAL)
        
        # Calculate impact for first researcher
        impact = self.manager.calculate_research_output_impact(researcher_ids[0])
        
        # Should have collaboration bonus
        assert impact["collaboration_impact"] > 1.0
    
    def test_calculate_research_output_impact_no_profile(self):
        """Test research output impact for non-existent researcher."""
        impact = self.manager.calculate_research_output_impact("nonexistent")
        
        assert impact["productivity_multiplier"] == 0.0
        assert impact["resource_satisfaction"] == 0.0
    
    def test_get_system_statistics(self):
        """Test comprehensive system statistics."""
        # Add some researchers and allocate resources
        for i in range(3):
            researcher_id = self.manager.add_researcher_profile(
                f"researcher_{i:03d}", 
                ResearcherLevel.ASSISTANT_PROF
            )
            
            requests = {ResourceType.LAB_EQUIPMENT: 5 + i}
            self.manager.allocate_resources(researcher_id, requests)
        
        stats = self.manager.get_system_statistics()
        
        # Should have all major sections
        assert "resource_statistics" in stats
        assert "researcher_statistics" in stats
        assert "collaboration_statistics" in stats
        assert "system_health" in stats
        
        # Resource statistics should have entries for each resource type
        resource_stats = stats["resource_statistics"]
        assert len(resource_stats) > 0
        
        # Researcher statistics should reflect our 3 researchers
        researcher_stats = stats["researcher_statistics"]
        assert researcher_stats["total_researchers"] == 3
        assert 0.0 <= researcher_stats["average_resource_satisfaction"] <= 1.0
        assert 0.0 <= researcher_stats["average_productivity_multiplier"] <= 2.0
        
        # System health should have overall metrics
        system_health = stats["system_health"]
        assert "overall_resource_utilization" in system_health
        assert "researcher_satisfaction_level" in system_health
        assert system_health["researcher_satisfaction_level"] in ["High", "Medium", "Low"]