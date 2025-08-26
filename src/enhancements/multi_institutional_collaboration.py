"""
Multi-Institutional Collaboration Bonus System

This module implements the collaboration incentive system for multi-institutional projects,
providing funding and publication success bonuses, implementing collaborative project
formation algorithms, and managing collaboration bonus calculations.
"""

import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from src.core.exceptions import ValidationError, NetworkError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel, FundingStatus


logger = get_logger(__name__)


class CollaborationType(Enum):
    """Types of multi-institutional collaborations."""
    BILATERAL = "Bilateral"  # 2 institutions
    MULTILATERAL = "Multilateral"  # 3+ institutions
    INTERNATIONAL = "International"  # Cross-country
    INDUSTRY_ACADEMIC = "Industry-Academic"  # Industry + Academic
    GOVERNMENT_ACADEMIC = "Government-Academic"  # Government + Academic


class CollaborationStatus(Enum):
    """Status of collaboration projects."""
    PROPOSED = "Proposed"
    ACTIVE = "Active"
    COMPLETED = "Completed"
    SUSPENDED = "Suspended"


@dataclass
class InstitutionProfile:
    """Profile of an institution for collaboration purposes."""
    institution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tier: int = 2  # 1-3 (1 = top tier)
    country: str = "USA"
    institution_type: str = "Academic"  # Academic, Industry, Government
    research_strengths: List[str] = field(default_factory=list)
    collaboration_history: List[str] = field(default_factory=list)  # Past collaboration IDs
    reputation_score: float = 0.5  # 0-1 scale
    
    def __post_init__(self):
        """Validate institution profile."""
        if not (1 <= self.tier <= 3):
            raise ValidationError("tier", self.tier, "integer between 1 and 3")
        
        if not (0.0 <= self.reputation_score <= 1.0):
            raise ValidationError("reputation_score", self.reputation_score, "float between 0.0 and 1.0")


@dataclass
class CollaborationProject:
    """Represents a multi-institutional collaboration project."""
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    participating_institutions: List[str] = field(default_factory=list)  # Institution IDs
    participating_researchers: List[str] = field(default_factory=list)  # Researcher IDs
    lead_institution: str = ""
    lead_researcher: str = ""
    
    # Project characteristics
    collaboration_type: CollaborationType = CollaborationType.BILATERAL
    research_areas: List[str] = field(default_factory=list)
    status: CollaborationStatus = CollaborationStatus.PROPOSED
    
    # Timeline
    start_date: date = field(default_factory=date.today)
    planned_end_date: date = field(default_factory=lambda: date.today() + timedelta(days=365))
    actual_end_date: Optional[date] = None
    
    # Funding and resources
    total_funding: int = 0
    funding_sources: List[str] = field(default_factory=list)
    resource_sharing_agreements: Dict[str, Any] = field(default_factory=dict)
    
    # Outcomes
    publications: List[str] = field(default_factory=list)  # Paper IDs
    patents: List[str] = field(default_factory=list)
    other_outcomes: List[str] = field(default_factory=list)
    
    # Bonus tracking
    funding_bonus_applied: bool = False
    publication_bonus_applied: bool = False
    bonus_amounts: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate collaboration project."""
        if len(self.participating_institutions) < 2:
            raise ValidationError("participating_institutions", len(self.participating_institutions), 
                                "at least 2 institutions")
        
        if self.lead_institution and self.lead_institution not in self.participating_institutions:
            raise ValidationError("lead_institution", self.lead_institution, 
                                "must be in participating institutions")
    
    def get_collaboration_complexity(self) -> float:
        """Calculate collaboration complexity score (0-1 scale)."""
        # Base complexity from number of institutions
        institution_complexity = min(1.0, (len(self.participating_institutions) - 2) / 3.0)
        
        # Type complexity
        type_complexity = {
            CollaborationType.BILATERAL: 0.2,
            CollaborationType.MULTILATERAL: 0.5,
            CollaborationType.INTERNATIONAL: 0.8,
            CollaborationType.INDUSTRY_ACADEMIC: 0.6,
            CollaborationType.GOVERNMENT_ACADEMIC: 0.7
        }
        
        # Research area diversity
        area_complexity = min(1.0, len(self.research_areas) / 5.0)
        
        return (institution_complexity * 0.4 + 
                type_complexity.get(self.collaboration_type, 0.5) * 0.4 + 
                area_complexity * 0.2)
    
    def is_active(self) -> bool:
        """Check if collaboration is currently active."""
        return self.status == CollaborationStatus.ACTIVE
    
    def get_duration_months(self) -> int:
        """Get project duration in months."""
        end_date = self.actual_end_date or self.planned_end_date
        return max(1, (end_date - self.start_date).days // 30)


@dataclass
class CollaborationBonus:
    """Represents bonuses awarded for multi-institutional collaboration."""
    bonus_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    researcher_id: str = ""
    institution_id: str = ""
    
    # Bonus types and amounts
    funding_success_bonus: float = 0.0  # Multiplier for funding success rate
    publication_success_bonus: float = 0.0  # Multiplier for publication acceptance
    reputation_bonus: float = 0.0  # Bonus to reputation score
    network_expansion_bonus: float = 0.0  # Bonus for network growth
    
    # Bonus justification
    bonus_factors: Dict[str, float] = field(default_factory=dict)
    calculation_details: str = ""
    
    # Temporal information
    awarded_date: date = field(default_factory=date.today)
    expiry_date: Optional[date] = None
    is_active: bool = True
    
    def get_total_bonus_value(self) -> float:
        """Calculate total bonus value."""
        return (self.funding_success_bonus + self.publication_success_bonus + 
                self.reputation_bonus + self.network_expansion_bonus)
    
    def is_expired(self) -> bool:
        """Check if bonus has expired."""
        if self.expiry_date is None:
            return False
        return date.today() > self.expiry_date


class MultiInstitutionalCollaborationSystem:
    """
    Main system for managing multi-institutional collaboration bonuses.
    
    This system provides:
    - Collaboration incentive system for multi-institutional projects
    - Funding and publication success bonuses
    - Collaborative project formation algorithms
    - Bonus calculation and tracking
    """
    
    def __init__(self):
        """Initialize the collaboration system."""
        self.institutions: Dict[str, InstitutionProfile] = {}
        self.projects: Dict[str, CollaborationProject] = {}
        self.bonuses: Dict[str, CollaborationBonus] = {}
        self.collaboration_history: Dict[str, List[str]] = defaultdict(list)  # researcher_id -> project_ids
        
        # Configuration parameters
        self.base_funding_bonus = 0.15  # 15% base bonus for multi-institutional projects
        self.base_publication_bonus = 0.10  # 10% base bonus for publication success
        self.complexity_multiplier = 2.0  # Multiplier for complex collaborations
        self.international_bonus = 0.05  # Additional 5% for international collaborations
        self.industry_bonus = 0.08  # Additional 8% for industry collaborations
        
        logger.info("Initialized MultiInstitutionalCollaborationSystem")
    
    def register_institution(self, institution: InstitutionProfile) -> str:
        """
        Register a new institution in the system.
        
        Args:
            institution: Institution profile to register
            
        Returns:
            str: Institution ID
        """
        self.institutions[institution.institution_id] = institution
        logger.info(f"Registered institution: {institution.name} (ID: {institution.institution_id})")
        return institution.institution_id
    
    def get_institution(self, institution_id: str) -> Optional[InstitutionProfile]:
        """Get institution by ID."""
        return self.institutions.get(institution_id)
    
    def create_collaboration_project(self, title: str, description: str,
                                   participating_institutions: List[str],
                                   participating_researchers: List[str],
                                   lead_institution: str, lead_researcher: str,
                                   research_areas: List[str],
                                   total_funding: int = 0) -> str:
        """
        Create a new multi-institutional collaboration project.
        
        Args:
            title: Project title
            description: Project description
            participating_institutions: List of institution IDs
            participating_researchers: List of researcher IDs
            lead_institution: Lead institution ID
            lead_researcher: Lead researcher ID
            research_areas: List of research areas
            total_funding: Total project funding
            
        Returns:
            str: Project ID
            
        Raises:
            ValidationError: If project parameters are invalid
        """
        # Validate institutions exist
        for inst_id in participating_institutions:
            if inst_id not in self.institutions:
                raise ValidationError("institution_id", inst_id, "registered institution")
        
        # Determine collaboration type
        collaboration_type = self._determine_collaboration_type(participating_institutions)
        
        project = CollaborationProject(
            title=title,
            description=description,
            participating_institutions=participating_institutions,
            participating_researchers=participating_researchers,
            lead_institution=lead_institution,
            lead_researcher=lead_researcher,
            collaboration_type=collaboration_type,
            research_areas=research_areas,
            total_funding=total_funding,
            status=CollaborationStatus.PROPOSED
        )
        
        self.projects[project.project_id] = project
        
        # Update collaboration history
        for researcher_id in participating_researchers:
            self.collaboration_history[researcher_id].append(project.project_id)
        
        logger.info(f"Created collaboration project: {title} (ID: {project.project_id})")
        return project.project_id
    
    def _determine_collaboration_type(self, institution_ids: List[str]) -> CollaborationType:
        """Determine collaboration type based on participating institutions."""
        institutions = [self.institutions[inst_id] for inst_id in institution_ids 
                       if inst_id in self.institutions]
        
        if not institutions:
            return CollaborationType.BILATERAL
        
        # Check for international collaboration
        countries = set(inst.country for inst in institutions)
        if len(countries) > 1:
            return CollaborationType.INTERNATIONAL
        
        # Check for industry-academic collaboration
        types = set(inst.institution_type for inst in institutions)
        if "Industry" in types and "Academic" in types:
            return CollaborationType.INDUSTRY_ACADEMIC
        
        if "Government" in types and "Academic" in types:
            return CollaborationType.GOVERNMENT_ACADEMIC
        
        # Check number of institutions
        if len(institutions) == 2:
            return CollaborationType.BILATERAL
        else:
            return CollaborationType.MULTILATERAL
    
    def activate_project(self, project_id: str) -> bool:
        """
        Activate a collaboration project and apply initial bonuses.
        
        Args:
            project_id: ID of the project to activate
            
        Returns:
            bool: True if successful, False otherwise
        """
        project = self.projects.get(project_id)
        if not project:
            logger.error(f"Project not found: {project_id}")
            return False
        
        if project.status != CollaborationStatus.PROPOSED:
            logger.warning(f"Project {project_id} is not in PROPOSED status")
            return False
        
        project.status = CollaborationStatus.ACTIVE
        
        # Apply initial collaboration bonuses
        self._apply_collaboration_bonuses(project)
        
        logger.info(f"Activated collaboration project: {project.title}")
        return True
    
    def _apply_collaboration_bonuses(self, project: CollaborationProject):
        """Apply collaboration bonuses to all participating researchers."""
        complexity_score = project.get_collaboration_complexity()
        
        for researcher_id in project.participating_researchers:
            bonus = self._calculate_collaboration_bonus(project, researcher_id, complexity_score)
            self.bonuses[bonus.bonus_id] = bonus
            
            logger.debug(f"Applied collaboration bonus to researcher {researcher_id}: "
                        f"funding={bonus.funding_success_bonus:.3f}, "
                        f"publication={bonus.publication_success_bonus:.3f}")
    
    def _calculate_collaboration_bonus(self, project: CollaborationProject, 
                                     researcher_id: str, complexity_score: float) -> CollaborationBonus:
        """Calculate collaboration bonus for a specific researcher."""
        # Base bonuses
        funding_bonus = self.base_funding_bonus
        publication_bonus = self.base_publication_bonus
        
        # Complexity multiplier
        complexity_multiplier = 1.0 + (complexity_score * self.complexity_multiplier)
        
        # Type-specific bonuses
        type_bonus = 0.0
        if project.collaboration_type == CollaborationType.INTERNATIONAL:
            type_bonus += self.international_bonus
        elif project.collaboration_type in [CollaborationType.INDUSTRY_ACADEMIC, 
                                          CollaborationType.GOVERNMENT_ACADEMIC]:
            type_bonus += self.industry_bonus
        
        # Leadership bonus
        leadership_bonus = 0.02 if researcher_id == project.lead_researcher else 0.0
        
        # Institution tier bonus (average of participating institutions)
        institution_tiers = []
        for inst_id in project.participating_institutions:
            if inst_id in self.institutions:
                institution_tiers.append(self.institutions[inst_id].tier)
        
        avg_tier = sum(institution_tiers) / len(institution_tiers) if institution_tiers else 2.0
        tier_bonus = (4 - avg_tier) / 10.0  # Higher tier (lower number) = higher bonus
        
        # Apply all multipliers and bonuses
        final_funding_bonus = (funding_bonus + type_bonus + leadership_bonus + tier_bonus) * complexity_multiplier
        final_publication_bonus = (publication_bonus + type_bonus + leadership_bonus + tier_bonus) * complexity_multiplier
        
        # Reputation and network bonuses
        reputation_bonus = complexity_score * 0.05  # Up to 5% reputation bonus
        network_bonus = len(project.participating_institutions) * 0.01  # 1% per institution
        
        bonus = CollaborationBonus(
            project_id=project.project_id,
            researcher_id=researcher_id,
            institution_id=self._get_researcher_institution(researcher_id, project),
            funding_success_bonus=final_funding_bonus,
            publication_success_bonus=final_publication_bonus,
            reputation_bonus=reputation_bonus,
            network_expansion_bonus=network_bonus,
            bonus_factors={
                "base_funding": funding_bonus,
                "base_publication": publication_bonus,
                "complexity_multiplier": complexity_multiplier,
                "type_bonus": type_bonus,
                "leadership_bonus": leadership_bonus,
                "tier_bonus": tier_bonus,
                "complexity_score": complexity_score
            },
            calculation_details=f"Multi-institutional collaboration bonus for {project.collaboration_type.value} project",
            expiry_date=project.planned_end_date + timedelta(days=365)  # Bonus lasts 1 year after project end
        )
        
        return bonus
    
    def _get_researcher_institution(self, researcher_id: str, project: CollaborationProject) -> str:
        """Get the primary institution for a researcher in a project."""
        # This is a simplified implementation - in practice, you'd look up the researcher's affiliation
        # For now, return the lead institution if it's the lead researcher, otherwise the first institution
        if researcher_id == project.lead_researcher:
            return project.lead_institution
        return project.participating_institutions[0] if project.participating_institutions else ""
    
    def get_researcher_bonuses(self, researcher_id: str, active_only: bool = True) -> List[CollaborationBonus]:
        """
        Get all collaboration bonuses for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            active_only: Whether to return only active bonuses
            
        Returns:
            List[CollaborationBonus]: List of bonuses
        """
        bonuses = [bonus for bonus in self.bonuses.values() 
                  if bonus.researcher_id == researcher_id]
        
        if active_only:
            bonuses = [bonus for bonus in bonuses 
                      if bonus.is_active and not bonus.is_expired()]
        
        return bonuses
    
    def calculate_funding_success_multiplier(self, researcher_id: str) -> float:
        """
        Calculate the funding success multiplier for a researcher based on active bonuses.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            float: Funding success multiplier (1.0 = no bonus)
        """
        bonuses = self.get_researcher_bonuses(researcher_id, active_only=True)
        total_bonus = sum(bonus.funding_success_bonus for bonus in bonuses)
        return 1.0 + total_bonus
    
    def calculate_publication_success_multiplier(self, researcher_id: str) -> float:
        """
        Calculate the publication success multiplier for a researcher based on active bonuses.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            float: Publication success multiplier (1.0 = no bonus)
        """
        bonuses = self.get_researcher_bonuses(researcher_id, active_only=True)
        total_bonus = sum(bonus.publication_success_bonus for bonus in bonuses)
        return 1.0 + total_bonus
    
    def suggest_collaboration_partners(self, researcher_id: str, 
                                     research_areas: List[str],
                                     max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest potential collaboration partners for a researcher.
        
        Args:
            researcher_id: ID of the researcher seeking collaborations
            research_areas: Research areas of interest
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List[Tuple[str, float]]: List of (institution_id, compatibility_score) tuples
        """
        suggestions = []
        
        for inst_id, institution in self.institutions.items():
            # Skip if researcher is already affiliated with this institution
            if self._is_researcher_affiliated(researcher_id, inst_id):
                continue
            
            # Calculate compatibility score
            compatibility = self._calculate_institution_compatibility(
                institution, research_areas, researcher_id
            )
            
            if compatibility > 0.1:  # Minimum threshold
                suggestions.append((inst_id, compatibility))
        
        # Sort by compatibility score and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def _is_researcher_affiliated(self, researcher_id: str, institution_id: str) -> bool:
        """Check if researcher is already affiliated with an institution."""
        # This would need to be implemented based on your researcher data structure
        # For now, return False as a placeholder
        return False
    
    def _calculate_institution_compatibility(self, institution: InstitutionProfile,
                                           research_areas: List[str],
                                           researcher_id: str) -> float:
        """Calculate compatibility score between researcher and institution."""
        # Research area overlap
        area_overlap = len(set(research_areas).intersection(set(institution.research_strengths)))
        area_score = area_overlap / max(len(research_areas), 1)
        
        # Institution reputation
        reputation_score = institution.reputation_score
        
        # Collaboration history bonus (institutions with successful collaborations)
        history_bonus = min(0.2, len(institution.collaboration_history) * 0.02)
        
        # Tier bonus (prefer diverse tier collaborations)
        tier_score = 0.1  # Base score for tier diversity
        
        return (area_score * 0.4 + reputation_score * 0.3 + 
                history_bonus * 0.2 + tier_score * 0.1)
    
    def complete_project(self, project_id: str, outcomes: Dict[str, Any]) -> bool:
        """
        Complete a collaboration project and apply final bonuses.
        
        Args:
            project_id: ID of the project to complete
            outcomes: Dictionary of project outcomes (publications, patents, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        project = self.projects.get(project_id)
        if not project:
            logger.error(f"Project not found: {project_id}")
            return False
        
        if project.status != CollaborationStatus.ACTIVE:
            logger.warning(f"Project {project_id} is not in ACTIVE status")
            return False
        
        project.status = CollaborationStatus.COMPLETED
        project.actual_end_date = date.today()
        
        # Update project outcomes
        project.publications = outcomes.get("publications", [])
        project.patents = outcomes.get("patents", [])
        project.other_outcomes = outcomes.get("other_outcomes", [])
        
        # Apply completion bonuses based on outcomes
        self._apply_completion_bonuses(project, outcomes)
        
        logger.info(f"Completed collaboration project: {project.title}")
        return True
    
    def _apply_completion_bonuses(self, project: CollaborationProject, outcomes: Dict[str, Any]):
        """Apply additional bonuses based on project completion and outcomes."""
        # Calculate success metrics
        publication_count = len(project.publications)
        patent_count = len(project.patents)
        
        # Success multiplier based on outcomes
        success_multiplier = 1.0
        if publication_count > 0:
            success_multiplier += publication_count * 0.05  # 5% per publication
        if patent_count > 0:
            success_multiplier += patent_count * 0.1  # 10% per patent
        
        # Apply completion bonuses to existing bonuses
        for bonus in self.bonuses.values():
            if (bonus.project_id == project.project_id and 
                bonus.is_active and not bonus.is_expired()):
                
                # Extend bonus duration for successful projects
                if success_multiplier > 1.2:  # Successful project
                    bonus.expiry_date = date.today() + timedelta(days=730)  # 2 years
                
                # Add success bonus
                bonus.bonus_factors["completion_success"] = success_multiplier
                bonus.calculation_details += f" | Completion bonus: {success_multiplier:.2f}x"
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the collaboration system.
        
        Returns:
            Dict[str, Any]: Collaboration statistics
        """
        total_projects = len(self.projects)
        active_projects = len([p for p in self.projects.values() 
                              if p.status == CollaborationStatus.ACTIVE])
        completed_projects = len([p for p in self.projects.values() 
                                 if p.status == CollaborationStatus.COMPLETED])
        
        # Collaboration type distribution
        type_distribution = defaultdict(int)
        for project in self.projects.values():
            type_distribution[project.collaboration_type.value] += 1
        
        # Average bonuses
        active_bonuses = [bonus for bonus in self.bonuses.values() 
                         if bonus.is_active and not bonus.is_expired()]
        
        avg_funding_bonus = (sum(b.funding_success_bonus for b in active_bonuses) / 
                           len(active_bonuses)) if active_bonuses else 0.0
        avg_publication_bonus = (sum(b.publication_success_bonus for b in active_bonuses) / 
                               len(active_bonuses)) if active_bonuses else 0.0
        
        return {
            "total_institutions": len(self.institutions),
            "total_projects": total_projects,
            "active_projects": active_projects,
            "completed_projects": completed_projects,
            "total_bonuses": len(self.bonuses),
            "active_bonuses": len(active_bonuses),
            "collaboration_type_distribution": dict(type_distribution),
            "average_funding_bonus": avg_funding_bonus,
            "average_publication_bonus": avg_publication_bonus,
            "total_funding": sum(p.total_funding for p in self.projects.values()),
            "researchers_with_collaborations": len(self.collaboration_history)
        }