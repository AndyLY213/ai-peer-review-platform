"""
Resource Constraint Manager

This module implements resource constraint modeling for lab equipment, student funding,
and collaboration incentives affecting research output. It models how resource availability
impacts researcher behavior and research productivity.
"""

import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import random

from src.core.exceptions import ValidationError
from src.data.enhanced_models import ResearcherLevel

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of research resources."""
    LAB_EQUIPMENT = "Lab Equipment"
    COMPUTING_RESOURCES = "Computing Resources"
    STUDENT_FUNDING = "Student Funding"
    RESEARCH_MATERIALS = "Research Materials"
    TRAVEL_FUNDING = "Travel Funding"
    PUBLICATION_FUNDS = "Publication Funds"


class ResourceStatus(Enum):
    """Status of resource availability."""
    ABUNDANT = "Abundant"
    ADEQUATE = "Adequate"
    LIMITED = "Limited"
    SCARCE = "Scarce"
    UNAVAILABLE = "Unavailable"


class CollaborationType(Enum):
    """Types of research collaborations."""
    INTERNAL = "Internal"  # Within same institution
    EXTERNAL = "External"  # Different institutions
    INTERNATIONAL = "International"  # Different countries
    INDUSTRY = "Industry"  # Academic-industry collaboration


@dataclass
class ResourceConstraint:
    """Represents a specific resource constraint affecting research."""
    
    constraint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.LAB_EQUIPMENT
    availability_level: ResourceStatus = ResourceStatus.ADEQUATE
    cost_per_unit: float = 1000.0
    maintenance_cost_per_month: float = 100.0
    utilization_capacity: int = 100  # Max usage units per month
    current_utilization: int = 0
    waiting_list: List[str] = field(default_factory=list)  # Researcher IDs waiting
    shared_access: bool = True
    institutional_priority: int = 3  # 1-5 scale, 5 = highest priority
    
    def __post_init__(self):
        """Validate resource constraint parameters."""
        if not (0 <= self.current_utilization <= self.utilization_capacity):
            raise ValidationError("current_utilization", self.current_utilization, 
                                f"0-{self.utilization_capacity}")
        
        if not (1 <= self.institutional_priority <= 5):
            raise ValidationError("institutional_priority", self.institutional_priority, "1-5")
    
    def get_availability_score(self) -> float:
        """Calculate availability score based on utilization and status."""
        utilization_ratio = self.current_utilization / max(self.utilization_capacity, 1)
        
        # Base score from status
        status_scores = {
            ResourceStatus.ABUNDANT: 1.0,
            ResourceStatus.ADEQUATE: 0.8,
            ResourceStatus.LIMITED: 0.6,
            ResourceStatus.SCARCE: 0.4,
            ResourceStatus.UNAVAILABLE: 0.0
        }
        
        base_score = status_scores[self.availability_level]
        
        # Adjust for utilization
        availability_multiplier = max(0.0, 1.0 - utilization_ratio)
        
        return base_score * availability_multiplier
    
    def can_allocate(self, requested_units: int) -> bool:
        """Check if resource can be allocated for requested units."""
        if self.availability_level == ResourceStatus.UNAVAILABLE:
            return False
        
        return (self.current_utilization + requested_units) <= self.utilization_capacity
    
    def allocate_resource(self, researcher_id: str, units: int) -> bool:
        """Allocate resource units to a researcher."""
        if not self.can_allocate(units):
            # Add to waiting list if not already there
            if researcher_id not in self.waiting_list:
                self.waiting_list.append(researcher_id)
            return False
        
        self.current_utilization += units
        
        # Remove from waiting list if they were waiting
        if researcher_id in self.waiting_list:
            self.waiting_list.remove(researcher_id)
        
        return True
    
    def release_resource(self, units: int) -> int:
        """Release resource units and return actual units released."""
        actual_released = min(units, self.current_utilization)
        self.current_utilization -= actual_released
        return actual_released


@dataclass
class StudentFunding:
    """Represents student funding availability and constraints."""
    
    funding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    researcher_id: str = ""
    total_budget: float = 50000.0
    allocated_budget: float = 0.0
    max_students: int = 3
    current_students: int = 0
    funding_duration_months: int = 12
    stipend_per_student_per_month: float = 2000.0
    overhead_rate: float = 0.3  # 30% overhead
    renewal_probability: float = 0.8
    
    def __post_init__(self):
        """Validate student funding parameters."""
        if self.allocated_budget > self.total_budget:
            raise ValidationError("allocated_budget", self.allocated_budget, 
                                f"<= {self.total_budget}")
        
        if self.current_students > self.max_students:
            raise ValidationError("current_students", self.current_students, 
                                f"<= {self.max_students}")
    
    def get_available_budget(self) -> float:
        """Get remaining available budget."""
        return self.total_budget - self.allocated_budget
    
    def can_fund_student(self) -> Tuple[bool, str]:
        """Check if can fund an additional student."""
        if self.current_students >= self.max_students:
            return False, "Maximum student capacity reached"
        
        monthly_cost = self.stipend_per_student_per_month * (1 + self.overhead_rate)
        total_cost = monthly_cost * self.funding_duration_months
        
        if total_cost > self.get_available_budget():
            return False, "Insufficient budget for full funding duration"
        
        return True, "Can fund student"
    
    def fund_student(self) -> bool:
        """Fund an additional student if possible."""
        can_fund, reason = self.can_fund_student()
        if not can_fund:
            return False
        
        monthly_cost = self.stipend_per_student_per_month * (1 + self.overhead_rate)
        total_cost = monthly_cost * self.funding_duration_months
        
        self.allocated_budget += total_cost
        self.current_students += 1
        
        return True
    
    def release_student(self) -> bool:
        """Release a student and free up budget."""
        if self.current_students <= 0:
            return False
        
        monthly_cost = self.stipend_per_student_per_month * (1 + self.overhead_rate)
        remaining_months = max(1, self.funding_duration_months // 2)  # Assume halfway through
        cost_to_release = monthly_cost * remaining_months
        
        self.allocated_budget = max(0, self.allocated_budget - cost_to_release)
        self.current_students -= 1
        
        return True


@dataclass
class CollaborationIncentive:
    """Represents incentives for research collaboration."""
    
    incentive_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    collaboration_type: CollaborationType = CollaborationType.EXTERNAL
    funding_multiplier: float = 1.2  # 20% bonus for collaboration
    resource_sharing_bonus: float = 0.3  # 30% more effective resource usage
    publication_bonus: float = 1.1  # 10% higher publication success rate
    min_collaborators: int = 2
    max_collaborators: int = 5
    duration_months: int = 12
    coordination_overhead: float = 0.1  # 10% overhead for coordination
    
    def __post_init__(self):
        """Validate collaboration incentive parameters."""
        if self.min_collaborators > self.max_collaborators:
            raise ValidationError("min_collaborators", self.min_collaborators, 
                                f"<= {self.max_collaborators}")
        
        if not (0.0 <= self.coordination_overhead <= 1.0):
            raise ValidationError("coordination_overhead", self.coordination_overhead, "0.0-1.0")
    
    def calculate_net_benefit(self, num_collaborators: int) -> float:
        """Calculate net benefit of collaboration considering overhead."""
        if not (self.min_collaborators <= num_collaborators <= self.max_collaborators):
            return 0.0
        
        # Base benefits
        funding_benefit = self.funding_multiplier - 1.0
        resource_benefit = self.resource_sharing_bonus
        publication_benefit = self.publication_bonus - 1.0
        
        total_benefit = funding_benefit + resource_benefit + publication_benefit
        
        # Apply coordination overhead (increases with more collaborators)
        overhead_multiplier = 1.0 + (self.coordination_overhead * (num_collaborators - 1))
        net_benefit = total_benefit / overhead_multiplier
        
        return max(0.0, net_benefit)
    
    def is_collaboration_viable(self, num_collaborators: int) -> bool:
        """Check if collaboration is viable given the number of collaborators."""
        return self.calculate_net_benefit(num_collaborators) > 0.0


@dataclass
class ResearcherResourceProfile:
    """Profile of a researcher's resource needs and constraints."""
    
    researcher_id: str = ""
    career_level: ResearcherLevel = ResearcherLevel.ASSISTANT_PROF
    resource_needs: Dict[ResourceType, int] = field(default_factory=dict)
    current_allocations: Dict[str, int] = field(default_factory=dict)  # constraint_id -> units
    student_funding: Optional[StudentFunding] = None
    collaboration_history: List[str] = field(default_factory=list)  # collaboration IDs
    resource_efficiency: float = 1.0  # Multiplier for resource effectiveness
    priority_score: float = 0.5  # 0-1 scale for resource allocation priority
    
    def __post_init__(self):
        """Initialize default resource needs based on career level."""
        if not self.resource_needs:
            self.resource_needs = self._get_default_resource_needs()
    
    def _get_default_resource_needs(self) -> Dict[ResourceType, int]:
        """Get default resource needs based on career level."""
        base_needs = {
            ResourceType.LAB_EQUIPMENT: 10,
            ResourceType.COMPUTING_RESOURCES: 20,
            ResourceType.STUDENT_FUNDING: 1,
            ResourceType.RESEARCH_MATERIALS: 15,
            ResourceType.TRAVEL_FUNDING: 5,
            ResourceType.PUBLICATION_FUNDS: 3
        }
        
        # Scale based on career level
        career_multipliers = {
            ResearcherLevel.GRADUATE_STUDENT: 0.3,
            ResearcherLevel.POSTDOC: 0.5,
            ResearcherLevel.ASSISTANT_PROF: 1.0,
            ResearcherLevel.ASSOCIATE_PROF: 1.5,
            ResearcherLevel.FULL_PROF: 2.0,
            ResearcherLevel.EMERITUS: 0.8
        }
        
        multiplier = career_multipliers.get(self.career_level, 1.0)
        
        return {resource_type: int(need * multiplier) 
                for resource_type, need in base_needs.items()}
    
    def get_total_resource_satisfaction(self, constraints: Dict[str, ResourceConstraint]) -> float:
        """Calculate overall resource satisfaction score (0-1)."""
        if not self.resource_needs:
            return 1.0
        
        total_satisfaction = 0.0
        total_weight = 0.0
        
        for resource_type, needed_units in self.resource_needs.items():
            if needed_units <= 0:
                continue
            
            # Find relevant constraints for this resource type
            relevant_constraints = [
                constraint for constraint in constraints.values()
                if constraint.resource_type == resource_type
            ]
            
            if not relevant_constraints:
                # No constraints means no availability
                satisfaction = 0.0
            else:
                # Calculate satisfaction based on best available constraint
                best_availability = max(
                    constraint.get_availability_score() 
                    for constraint in relevant_constraints
                )
                satisfaction = best_availability
            
            total_satisfaction += satisfaction * needed_units
            total_weight += needed_units
        
        return total_satisfaction / max(total_weight, 1.0)
    
    def calculate_productivity_impact(self, resource_satisfaction: float) -> float:
        """Calculate impact of resource constraints on research productivity."""
        # Base productivity impact curve (diminishing returns)
        if resource_satisfaction >= 0.8:
            productivity_multiplier = 1.0
        elif resource_satisfaction >= 0.6:
            productivity_multiplier = 0.9
        elif resource_satisfaction >= 0.4:
            productivity_multiplier = 0.7
        elif resource_satisfaction >= 0.2:
            productivity_multiplier = 0.5
        else:
            productivity_multiplier = 0.3
        
        # Apply resource efficiency
        final_multiplier = productivity_multiplier * self.resource_efficiency
        
        return min(1.0, final_multiplier)


class ResourceConstraintManager:
    """Main class for managing resource constraints and their effects on research output."""
    
    def __init__(self):
        """Initialize the resource constraint manager."""
        self.constraints: Dict[str, ResourceConstraint] = {}
        self.researcher_profiles: Dict[str, ResearcherResourceProfile] = {}
        self.collaboration_incentives: Dict[str, CollaborationIncentive] = {}
        self.active_collaborations: Dict[str, List[str]] = {}  # collaboration_id -> researcher_ids
        self._create_default_constraints()
        self._create_default_incentives()
    
    def _create_default_constraints(self):
        """Create default resource constraints for common research resources."""
        # Lab Equipment Constraint
        lab_equipment = ResourceConstraint(
            resource_type=ResourceType.LAB_EQUIPMENT,
            availability_level=ResourceStatus.LIMITED,
            cost_per_unit=5000.0,
            maintenance_cost_per_month=200.0,
            utilization_capacity=50,
            shared_access=True,
            institutional_priority=4
        )
        self.add_constraint(lab_equipment)
        
        # Computing Resources Constraint
        computing = ResourceConstraint(
            resource_type=ResourceType.COMPUTING_RESOURCES,
            availability_level=ResourceStatus.ADEQUATE,
            cost_per_unit=100.0,
            maintenance_cost_per_month=50.0,
            utilization_capacity=200,
            shared_access=True,
            institutional_priority=5
        )
        self.add_constraint(computing)
        
        # Research Materials Constraint
        materials = ResourceConstraint(
            resource_type=ResourceType.RESEARCH_MATERIALS,
            availability_level=ResourceStatus.SCARCE,
            cost_per_unit=500.0,
            maintenance_cost_per_month=0.0,
            utilization_capacity=30,
            shared_access=False,
            institutional_priority=3
        )
        self.add_constraint(materials)
        
        # Travel Funding Constraint
        travel = ResourceConstraint(
            resource_type=ResourceType.TRAVEL_FUNDING,
            availability_level=ResourceStatus.LIMITED,
            cost_per_unit=2000.0,
            maintenance_cost_per_month=0.0,
            utilization_capacity=20,
            shared_access=False,
            institutional_priority=2
        )
        self.add_constraint(travel)
    
    def _create_default_incentives(self):
        """Create default collaboration incentives."""
        # External collaboration incentive
        external = CollaborationIncentive(
            collaboration_type=CollaborationType.EXTERNAL,
            funding_multiplier=1.3,
            resource_sharing_bonus=0.4,
            publication_bonus=1.2,
            min_collaborators=2,
            max_collaborators=4,
            coordination_overhead=0.15
        )
        self.add_collaboration_incentive(external)
        
        # International collaboration incentive
        international = CollaborationIncentive(
            collaboration_type=CollaborationType.INTERNATIONAL,
            funding_multiplier=1.5,
            resource_sharing_bonus=0.5,
            publication_bonus=1.3,
            min_collaborators=2,
            max_collaborators=3,
            coordination_overhead=0.25
        )
        self.add_collaboration_incentive(international)
        
        # Industry collaboration incentive
        industry = CollaborationIncentive(
            collaboration_type=CollaborationType.INDUSTRY,
            funding_multiplier=2.0,
            resource_sharing_bonus=0.6,
            publication_bonus=1.1,
            min_collaborators=2,
            max_collaborators=5,
            coordination_overhead=0.2
        )
        self.add_collaboration_incentive(industry)
    
    def add_constraint(self, constraint: ResourceConstraint) -> str:
        """Add a resource constraint to the system."""
        self.constraints[constraint.constraint_id] = constraint
        return constraint.constraint_id
    
    def get_constraint(self, constraint_id: str) -> Optional[ResourceConstraint]:
        """Get a resource constraint by ID."""
        return self.constraints.get(constraint_id)
    
    def get_constraints_by_type(self, resource_type: ResourceType) -> List[ResourceConstraint]:
        """Get all constraints for a specific resource type."""
        return [constraint for constraint in self.constraints.values()
                if constraint.resource_type == resource_type]
    
    def add_researcher_profile(self, researcher_id: str, career_level: ResearcherLevel,
                             custom_needs: Optional[Dict[ResourceType, int]] = None) -> str:
        """Add a researcher resource profile."""
        profile = ResearcherResourceProfile(
            researcher_id=researcher_id,
            career_level=career_level
        )
        
        if custom_needs:
            profile.resource_needs = custom_needs
        
        # Create student funding if appropriate career level
        if career_level in [ResearcherLevel.ASSISTANT_PROF, ResearcherLevel.ASSOCIATE_PROF, 
                           ResearcherLevel.FULL_PROF]:
            funding_amounts = {
                ResearcherLevel.ASSISTANT_PROF: 75000,
                ResearcherLevel.ASSOCIATE_PROF: 150000,
                ResearcherLevel.FULL_PROF: 250000
            }
            
            max_students = {
                ResearcherLevel.ASSISTANT_PROF: 2,
                ResearcherLevel.ASSOCIATE_PROF: 4,
                ResearcherLevel.FULL_PROF: 6
            }
            
            profile.student_funding = StudentFunding(
                researcher_id=researcher_id,
                total_budget=funding_amounts[career_level],
                max_students=max_students[career_level]
            )
        
        self.researcher_profiles[researcher_id] = profile
        return researcher_id
    
    def get_researcher_profile(self, researcher_id: str) -> Optional[ResearcherResourceProfile]:
        """Get researcher resource profile."""
        return self.researcher_profiles.get(researcher_id)
    
    def allocate_resources(self, researcher_id: str, 
                          resource_requests: Dict[ResourceType, int]) -> Dict[ResourceType, bool]:
        """
        Attempt to allocate requested resources to a researcher.
        
        Args:
            researcher_id: ID of the researcher requesting resources
            resource_requests: Dictionary mapping resource types to requested units
            
        Returns:
            Dictionary mapping resource types to allocation success (True/False)
        """
        profile = self.get_researcher_profile(researcher_id)
        if not profile:
            return {resource_type: False for resource_type in resource_requests.keys()}
        
        allocation_results = {}
        
        for resource_type, requested_units in resource_requests.items():
            # Find best available constraint for this resource type
            available_constraints = self.get_constraints_by_type(resource_type)
            
            if not available_constraints:
                allocation_results[resource_type] = False
                continue
            
            # Sort by availability score (best first)
            available_constraints.sort(
                key=lambda c: c.get_availability_score(), 
                reverse=True
            )
            
            allocated = False
            for constraint in available_constraints:
                if constraint.allocate_resource(researcher_id, requested_units):
                    # Track allocation in researcher profile
                    profile.current_allocations[constraint.constraint_id] = (
                        profile.current_allocations.get(constraint.constraint_id, 0) + 
                        requested_units
                    )
                    allocated = True
                    break
            
            allocation_results[resource_type] = allocated
        
        return allocation_results
    
    def release_resources(self, researcher_id: str, 
                         resource_releases: Dict[str, int]) -> Dict[str, int]:
        """
        Release resources allocated to a researcher.
        
        Args:
            researcher_id: ID of the researcher releasing resources
            resource_releases: Dictionary mapping constraint IDs to units to release
            
        Returns:
            Dictionary mapping constraint IDs to actual units released
        """
        profile = self.get_researcher_profile(researcher_id)
        if not profile:
            return {}
        
        release_results = {}
        
        for constraint_id, units_to_release in resource_releases.items():
            constraint = self.get_constraint(constraint_id)
            if not constraint:
                release_results[constraint_id] = 0
                continue
            
            # Check how many units the researcher actually has allocated
            allocated_units = profile.current_allocations.get(constraint_id, 0)
            actual_release = min(units_to_release, allocated_units)
            
            if actual_release > 0:
                released = constraint.release_resource(actual_release)
                profile.current_allocations[constraint_id] = allocated_units - released
                
                # Remove entry if no units left
                if profile.current_allocations[constraint_id] <= 0:
                    del profile.current_allocations[constraint_id]
                
                release_results[constraint_id] = released
            else:
                release_results[constraint_id] = 0
        
        return release_results
    
    def add_collaboration_incentive(self, incentive: CollaborationIncentive) -> str:
        """Add a collaboration incentive to the system."""
        self.collaboration_incentives[incentive.incentive_id] = incentive
        return incentive.incentive_id
    
    def create_collaboration(self, researcher_ids: List[str], 
                           collaboration_type: CollaborationType) -> Optional[str]:
        """
        Create a new research collaboration.
        
        Args:
            researcher_ids: List of researcher IDs participating
            collaboration_type: Type of collaboration
            
        Returns:
            Collaboration ID if successful, None otherwise
        """
        if len(researcher_ids) < 2:
            return None
        
        # Find appropriate incentive
        incentive = None
        for inc in self.collaboration_incentives.values():
            if (inc.collaboration_type == collaboration_type and
                inc.min_collaborators <= len(researcher_ids) <= inc.max_collaborators):
                incentive = inc
                break
        
        if not incentive:
            return None
        
        # Check if collaboration is viable
        if not incentive.is_collaboration_viable(len(researcher_ids)):
            return None
        
        collaboration_id = str(uuid.uuid4())
        self.active_collaborations[collaboration_id] = researcher_ids.copy()
        
        # Update researcher profiles
        for researcher_id in researcher_ids:
            profile = self.get_researcher_profile(researcher_id)
            if profile:
                profile.collaboration_history.append(collaboration_id)
                # Apply resource efficiency bonus
                net_benefit = incentive.calculate_net_benefit(len(researcher_ids))
                profile.resource_efficiency = min(2.0, profile.resource_efficiency + net_benefit * 0.1)
        
        return collaboration_id
    
    def calculate_research_output_impact(self, researcher_id: str) -> Dict[str, float]:
        """
        Calculate the impact of resource constraints on research output.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Dictionary with impact metrics
        """
        profile = self.get_researcher_profile(researcher_id)
        if not profile:
            return {"productivity_multiplier": 0.0, "resource_satisfaction": 0.0}
        
        # Calculate resource satisfaction
        resource_satisfaction = profile.get_total_resource_satisfaction(self.constraints)
        
        # Calculate productivity impact
        productivity_multiplier = profile.calculate_productivity_impact(resource_satisfaction)
        
        # Factor in student funding impact
        student_impact = 1.0
        if profile.student_funding:
            funding_ratio = (profile.student_funding.total_budget - 
                           profile.student_funding.allocated_budget) / profile.student_funding.total_budget
            student_impact = 0.8 + (0.4 * funding_ratio)  # 0.8 to 1.2 multiplier
        
        # Factor in collaboration benefits
        collaboration_impact = 1.0
        if profile.collaboration_history:
            # Recent collaborations provide ongoing benefits
            recent_collaborations = len([
                collab_id for collab_id in profile.collaboration_history
                if collab_id in self.active_collaborations
            ])
            collaboration_impact = 1.0 + (recent_collaborations * 0.1)  # 10% per active collaboration
        
        final_productivity = productivity_multiplier * student_impact * collaboration_impact
        
        return {
            "productivity_multiplier": min(2.0, final_productivity),
            "resource_satisfaction": resource_satisfaction,
            "student_funding_impact": student_impact,
            "collaboration_impact": collaboration_impact,
            "base_productivity": productivity_multiplier
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the resource constraint system."""
        # Resource utilization statistics
        resource_stats = {}
        for resource_type in ResourceType:
            constraints = self.get_constraints_by_type(resource_type)
            if constraints:
                total_capacity = sum(c.utilization_capacity for c in constraints)
                total_utilization = sum(c.current_utilization for c in constraints)
                avg_availability = sum(c.get_availability_score() for c in constraints) / len(constraints)
                
                resource_stats[resource_type.value] = {
                    "total_capacity": total_capacity,
                    "total_utilization": total_utilization,
                    "utilization_rate": total_utilization / max(total_capacity, 1),
                    "average_availability": avg_availability,
                    "num_constraints": len(constraints)
                }
        
        # Researcher statistics
        total_researchers = len(self.researcher_profiles)
        avg_satisfaction = 0.0
        avg_productivity = 0.0
        
        if total_researchers > 0:
            satisfactions = []
            productivities = []
            
            for researcher_id in self.researcher_profiles.keys():
                impact = self.calculate_research_output_impact(researcher_id)
                satisfactions.append(impact["resource_satisfaction"])
                productivities.append(impact["productivity_multiplier"])
            
            avg_satisfaction = sum(satisfactions) / len(satisfactions)
            avg_productivity = sum(productivities) / len(productivities)
        
        # Collaboration statistics
        active_collaborations = len(self.active_collaborations)
        total_participants = sum(len(participants) for participants in self.active_collaborations.values())
        
        return {
            "resource_statistics": resource_stats,
            "researcher_statistics": {
                "total_researchers": total_researchers,
                "average_resource_satisfaction": avg_satisfaction,
                "average_productivity_multiplier": avg_productivity
            },
            "collaboration_statistics": {
                "active_collaborations": active_collaborations,
                "total_participants": total_participants,
                "average_collaboration_size": total_participants / max(active_collaborations, 1)
            },
            "system_health": {
                "overall_resource_utilization": sum(
                    stats["utilization_rate"] for stats in resource_stats.values()
                ) / max(len(resource_stats), 1),
                "researcher_satisfaction_level": "High" if avg_satisfaction > 0.7 else 
                                               "Medium" if avg_satisfaction > 0.4 else "Low"
            }
        }