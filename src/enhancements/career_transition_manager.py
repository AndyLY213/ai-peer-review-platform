"""
Career Transition Management System

This module implements career transition modeling for academic-industry transitions,
modeling different incentive structures across career paths, and creating transition
probability calculations based on researcher profiles.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging
import random
import math

from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel, CareerStage
from src.core.exceptions import ValidationError, CareerSystemError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class CareerPath(Enum):
    """Different career path options."""
    ACADEMIC_RESEARCH = "Academic Research"
    ACADEMIC_TEACHING = "Academic Teaching"
    INDUSTRY_RESEARCH = "Industry Research"
    INDUSTRY_APPLIED = "Industry Applied"
    GOVERNMENT_LAB = "Government Lab"
    CONSULTING = "Consulting"
    ENTREPRENEURSHIP = "Entrepreneurship"
    NON_PROFIT = "Non-Profit Research"
    SCIENCE_POLICY = "Science Policy"
    SCIENCE_COMMUNICATION = "Science Communication"


class TransitionReason(Enum):
    """Reasons for career transitions."""
    BETTER_COMPENSATION = "Better Compensation"
    WORK_LIFE_BALANCE = "Work-Life Balance"
    JOB_SECURITY = "Job Security"
    RESEARCH_FREEDOM = "Research Freedom"
    IMPACT_OPPORTUNITY = "Impact Opportunity"
    GEOGRAPHIC_PREFERENCE = "Geographic Preference"
    FAMILY_CONSIDERATIONS = "Family Considerations"
    CAREER_ADVANCEMENT = "Career Advancement"
    MARKET_CONDITIONS = "Market Conditions"
    PERSONAL_INTEREST = "Personal Interest"


class TransitionOutcome(Enum):
    """Possible transition outcomes."""
    SUCCESSFUL = "Successful"
    PARTIALLY_SUCCESSFUL = "Partially Successful"
    UNSUCCESSFUL = "Unsuccessful"
    RETURNED_TO_ACADEMIA = "Returned to Academia"
    PENDING = "Pending"


@dataclass
class IncentiveStructure:
    """Represents incentive structure for different career paths."""
    career_path: CareerPath
    salary_range: Tuple[int, int]  # Min, max annual salary
    job_security_score: float  # 0-1 scale
    research_freedom_score: float  # 0-1 scale
    work_life_balance_score: float  # 0-1 scale
    advancement_opportunities: float  # 0-1 scale
    impact_potential: float  # 0-1 scale
    publication_importance: float  # 0-1 scale (how much publications matter)
    collaboration_opportunities: float  # 0-1 scale
    geographic_flexibility: float  # 0-1 scale
    
    def __post_init__(self):
        """Validate incentive structure scores."""
        scores = [
            self.job_security_score, self.research_freedom_score, 
            self.work_life_balance_score, self.advancement_opportunities,
            self.impact_potential, self.publication_importance,
            self.collaboration_opportunities, self.geographic_flexibility
        ]
        for i, score in enumerate(scores):
            if not (0.0 <= score <= 1.0):
                raise ValidationError(f"score_{i}", score, "value between 0 and 1")


@dataclass
class TransitionProfile:
    """Profile of researcher's transition preferences and constraints."""
    researcher_id: str
    current_path: CareerPath
    preferred_paths: List[CareerPath]
    priority_factors: Dict[str, float]  # What matters most to them
    constraints: List[str]  # Geographic, family, etc.
    risk_tolerance: float  # 0-1 scale (willingness to take career risks)
    salary_requirements: Tuple[int, int]  # Min, preferred salary
    timeline_flexibility: int  # Months willing to wait for transition
    
    def __post_init__(self):
        """Validate transition profile."""
        if not (0.0 <= self.risk_tolerance <= 1.0):
            raise ValidationError("risk_tolerance", self.risk_tolerance, "value between 0 and 1")
        if self.salary_requirements[0] > self.salary_requirements[1]:
            raise ValidationError("salary_requirements", self.salary_requirements, "min <= preferred")


@dataclass
class TransitionPlan:
    """Detailed plan for career transition."""
    researcher_id: str
    source_path: CareerPath
    target_path: CareerPath
    transition_probability: float  # 0-1 probability of success
    estimated_timeline: int  # Months to complete transition
    required_skills: List[str]  # Skills needed for transition
    skill_gaps: List[str]  # Current skill gaps
    preparation_steps: List[str]  # Steps to prepare for transition
    networking_requirements: List[str]  # Networking needs
    financial_considerations: Dict[str, float]  # Salary changes, costs, etc.
    risk_factors: List[str]  # Potential risks
    success_factors: List[str]  # Factors that increase success probability
    
    def __post_init__(self):
        """Validate transition plan."""
        if not (0.0 <= self.transition_probability <= 1.0):
            raise ValidationError("transition_probability", self.transition_probability, "value between 0 and 1")
        if self.estimated_timeline < 0:
            raise ValidationError("estimated_timeline", self.estimated_timeline, "non-negative integer")


@dataclass
class TransitionOutcomeRecord:
    """Record of completed career transition."""
    researcher_id: str
    transition_date: date
    source_path: CareerPath
    target_path: CareerPath
    outcome: TransitionOutcome
    actual_timeline: int  # Months taken
    salary_change: float  # Percentage change
    satisfaction_score: float  # 0-1 scale
    lessons_learned: List[str]
    challenges_faced: List[str]
    success_factors: List[str]
    
    def __post_init__(self):
        """Validate outcome record."""
        if not (0.0 <= self.satisfaction_score <= 1.0):
            raise ValidationError("satisfaction_score", self.satisfaction_score, "value between 0 and 1")


class CareerTransitionManager:
    """
    Manages career transition modeling for academic-industry transitions with
    different incentive structures and transition probability calculations.
    
    This class provides functionality to:
    - Model different incentive structures across career paths
    - Calculate transition probabilities based on researcher profiles
    - Create detailed transition plans with skill gap analysis
    - Track transition outcomes and success factors
    - Analyze career path trends and patterns
    """
    
    # Default incentive structures for different career paths
    DEFAULT_INCENTIVE_STRUCTURES = {
        CareerPath.ACADEMIC_RESEARCH: IncentiveStructure(
            career_path=CareerPath.ACADEMIC_RESEARCH,
            salary_range=(65000, 120000),
            job_security_score=0.6,  # Tenure provides security but competitive
            research_freedom_score=0.9,  # High research freedom
            work_life_balance_score=0.7,  # Flexible but demanding
            advancement_opportunities=0.5,  # Limited positions available
            impact_potential=0.8,  # High potential for scientific impact
            publication_importance=1.0,  # Publications are critical
            collaboration_opportunities=0.8,  # Strong academic networks
            geographic_flexibility=0.4  # Limited by university locations
        ),
        CareerPath.ACADEMIC_TEACHING: IncentiveStructure(
            career_path=CareerPath.ACADEMIC_TEACHING,
            salary_range=(55000, 85000),
            job_security_score=0.7,  # More stable than research track
            research_freedom_score=0.3,  # Limited research opportunities
            work_life_balance_score=0.8,  # Better work-life balance
            advancement_opportunities=0.4,  # Limited advancement paths
            impact_potential=0.6,  # Educational impact
            publication_importance=0.3,  # Less emphasis on publications
            collaboration_opportunities=0.6,  # Teaching communities
            geographic_flexibility=0.6  # More teaching positions available
        ),
        CareerPath.INDUSTRY_RESEARCH: IncentiveStructure(
            career_path=CareerPath.INDUSTRY_RESEARCH,
            salary_range=(100000, 180000),
            job_security_score=0.5,  # Market dependent
            research_freedom_score=0.6,  # Directed by business needs
            work_life_balance_score=0.6,  # Varies by company
            advancement_opportunities=0.8,  # Clear advancement paths
            impact_potential=0.7,  # Product/technology impact
            publication_importance=0.4,  # Some publications valued
            collaboration_opportunities=0.7,  # Industry networks
            geographic_flexibility=0.7  # Multiple locations
        ),
        CareerPath.INDUSTRY_APPLIED: IncentiveStructure(
            career_path=CareerPath.INDUSTRY_APPLIED,
            salary_range=(90000, 160000),
            job_security_score=0.6,  # Stable with performance
            research_freedom_score=0.4,  # Applied focus
            work_life_balance_score=0.7,  # Generally better than academia
            advancement_opportunities=0.8,  # Clear career progression
            impact_potential=0.6,  # Product development impact
            publication_importance=0.2,  # Publications less important
            collaboration_opportunities=0.6,  # Team-based work
            geographic_flexibility=0.8  # Many locations available
        ),
        CareerPath.GOVERNMENT_LAB: IncentiveStructure(
            career_path=CareerPath.GOVERNMENT_LAB,
            salary_range=(75000, 130000),
            job_security_score=0.9,  # Very stable employment
            research_freedom_score=0.7,  # Mission-directed research
            work_life_balance_score=0.8,  # Good work-life balance
            advancement_opportunities=0.6,  # Structured advancement
            impact_potential=0.8,  # Policy and societal impact
            publication_importance=0.7,  # Publications valued
            collaboration_opportunities=0.7,  # Government networks
            geographic_flexibility=0.3  # Limited locations
        ),
        CareerPath.CONSULTING: IncentiveStructure(
            career_path=CareerPath.CONSULTING,
            salary_range=(80000, 200000),
            job_security_score=0.4,  # Project-based work
            research_freedom_score=0.5,  # Client-driven work
            work_life_balance_score=0.4,  # Often demanding hours
            advancement_opportunities=0.7,  # Performance-based advancement
            impact_potential=0.6,  # Business impact
            publication_importance=0.1,  # Publications not important
            collaboration_opportunities=0.8,  # Diverse client networks
            geographic_flexibility=0.9  # Travel opportunities
        ),
        CareerPath.ENTREPRENEURSHIP: IncentiveStructure(
            career_path=CareerPath.ENTREPRENEURSHIP,
            salary_range=(0, 500000),  # Highly variable
            job_security_score=0.2,  # High risk
            research_freedom_score=0.9,  # Complete freedom
            work_life_balance_score=0.3,  # Very demanding
            advancement_opportunities=1.0,  # Unlimited potential
            impact_potential=0.9,  # High potential impact
            publication_importance=0.2,  # Publications less relevant
            collaboration_opportunities=0.7,  # Startup ecosystems
            geographic_flexibility=0.8  # Location flexibility
        ),
        CareerPath.NON_PROFIT: IncentiveStructure(
            career_path=CareerPath.NON_PROFIT,
            salary_range=(50000, 90000),
            job_security_score=0.5,  # Funding dependent
            research_freedom_score=0.6,  # Mission-aligned research
            work_life_balance_score=0.7,  # Generally good balance
            advancement_opportunities=0.5,  # Limited by organization size
            impact_potential=0.9,  # High social impact

            publication_importance=0.6,  # Publications help with funding
            collaboration_opportunities=0.8,  # Strong mission-driven networks
            geographic_flexibility=0.6  # Varies by organization
        ),
        CareerPath.SCIENCE_POLICY: IncentiveStructure(
            career_path=CareerPath.SCIENCE_POLICY,
            salary_range=(70000, 140000),
            job_security_score=0.6,  # Political cycles affect stability
            research_freedom_score=0.4,  # Policy-constrained
            work_life_balance_score=0.6,  # Varies by role
            advancement_opportunities=0.7,  # Political advancement possible
            impact_potential=0.9,  # High policy impact
            publication_importance=0.5,  # Some publications valued
            collaboration_opportunities=0.8,  # Policy networks
            geographic_flexibility=0.4  # Concentrated in policy centers
        ),
        CareerPath.SCIENCE_COMMUNICATION: IncentiveStructure(
            career_path=CareerPath.SCIENCE_COMMUNICATION,
            salary_range=(45000, 100000),
            job_security_score=0.4,  # Media industry volatility
            research_freedom_score=0.6,  # Editorial constraints
            work_life_balance_score=0.6,  # Deadline-driven
            advancement_opportunities=0.6,  # Media career paths
            impact_potential=0.8,  # Public education impact
            publication_importance=0.3,  # Different publication types
            collaboration_opportunities=0.7,  # Media and science networks
            geographic_flexibility=0.7  # Many media markets
        )
    }
    
    # Transition difficulty matrix (source -> target difficulty score 0-1)
    TRANSITION_DIFFICULTY = {
        CareerPath.ACADEMIC_RESEARCH: {
            CareerPath.ACADEMIC_TEACHING: 0.3,
            CareerPath.INDUSTRY_RESEARCH: 0.4,
            CareerPath.INDUSTRY_APPLIED: 0.6,
            CareerPath.GOVERNMENT_LAB: 0.3,
            CareerPath.CONSULTING: 0.7,
            CareerPath.ENTREPRENEURSHIP: 0.8,
            CareerPath.NON_PROFIT: 0.4,
            CareerPath.SCIENCE_POLICY: 0.5,
            CareerPath.SCIENCE_COMMUNICATION: 0.6
        },
        CareerPath.ACADEMIC_TEACHING: {
            CareerPath.ACADEMIC_RESEARCH: 0.7,
            CareerPath.INDUSTRY_RESEARCH: 0.8,
            CareerPath.INDUSTRY_APPLIED: 0.7,
            CareerPath.GOVERNMENT_LAB: 0.6,
            CareerPath.CONSULTING: 0.6,
            CareerPath.ENTREPRENEURSHIP: 0.8,
            CareerPath.NON_PROFIT: 0.4,
            CareerPath.SCIENCE_POLICY: 0.5,
            CareerPath.SCIENCE_COMMUNICATION: 0.4
        },
        CareerPath.INDUSTRY_RESEARCH: {
            CareerPath.ACADEMIC_RESEARCH: 0.6,
            CareerPath.ACADEMIC_TEACHING: 0.7,
            CareerPath.INDUSTRY_APPLIED: 0.3,
            CareerPath.GOVERNMENT_LAB: 0.4,
            CareerPath.CONSULTING: 0.4,
            CareerPath.ENTREPRENEURSHIP: 0.5,
            CareerPath.NON_PROFIT: 0.5,
            CareerPath.SCIENCE_POLICY: 0.6,
            CareerPath.SCIENCE_COMMUNICATION: 0.6
        },
        CareerPath.INDUSTRY_APPLIED: {
            CareerPath.ACADEMIC_RESEARCH: 0.8,
            CareerPath.ACADEMIC_TEACHING: 0.6,
            CareerPath.INDUSTRY_RESEARCH: 0.4,
            CareerPath.GOVERNMENT_LAB: 0.5,
            CareerPath.CONSULTING: 0.3,
            CareerPath.ENTREPRENEURSHIP: 0.4,
            CareerPath.NON_PROFIT: 0.6,
            CareerPath.SCIENCE_POLICY: 0.7,
            CareerPath.SCIENCE_COMMUNICATION: 0.7
        }
    }
    
    def __init__(self):
        """Initialize the career transition manager."""
        logger.info("Initializing Career Transition Management System")
        self.incentive_structures: Dict[CareerPath, IncentiveStructure] = self.DEFAULT_INCENTIVE_STRUCTURES.copy()
        self.transition_profiles: Dict[str, TransitionProfile] = {}
        self.transition_plans: Dict[str, List[TransitionPlan]] = {}
        self.transition_outcomes: Dict[str, List[TransitionOutcomeRecord]] = {}
    
    def get_incentive_structure(self, career_path: CareerPath) -> IncentiveStructure:
        """
        Get incentive structure for a specific career path.
        
        Args:
            career_path: The career path to get incentives for
            
        Returns:
            IncentiveStructure object
        """
        return self.incentive_structures.get(career_path, 
                                           self.incentive_structures[CareerPath.ACADEMIC_RESEARCH])
    
    def create_transition_profile(self, researcher: EnhancedResearcher,
                                preferred_paths: List[CareerPath],
                                priority_factors: Optional[Dict[str, float]] = None,
                                constraints: Optional[List[str]] = None,
                                risk_tolerance: float = 0.5) -> TransitionProfile:
        """
        Create a transition profile for a researcher.
        
        Args:
            researcher: The researcher considering transition
            preferred_paths: List of preferred career paths
            priority_factors: What factors matter most (salary, balance, etc.)
            constraints: Any constraints (geographic, family, etc.)
            risk_tolerance: Willingness to take career risks (0-1)
            
        Returns:
            TransitionProfile object
        """
        if priority_factors is None:
            # Default priorities based on career stage
            if researcher.level in [ResearcherLevel.GRADUATE_STUDENT, ResearcherLevel.POSTDOC]:
                priority_factors = {
                    'salary': 0.3,
                    'job_security': 0.25,
                    'research_freedom': 0.2,
                    'advancement': 0.15,
                    'work_life_balance': 0.1
                }
            else:
                priority_factors = {
                    'salary': 0.2,
                    'job_security': 0.2,
                    'research_freedom': 0.25,
                    'work_life_balance': 0.2,
                    'impact': 0.15
                }
        
        if constraints is None:
            constraints = []
        
        # Determine current career path
        current_path = self._determine_current_path(researcher)
        
        # Calculate salary requirements based on current situation
        salary_requirements = self._calculate_salary_requirements(researcher)
        
        # Estimate timeline flexibility based on career stage
        timeline_flexibility = self._estimate_timeline_flexibility(researcher)
        
        profile = TransitionProfile(
            researcher_id=researcher.id,
            current_path=current_path,
            preferred_paths=preferred_paths,
            priority_factors=priority_factors,
            constraints=constraints,
            risk_tolerance=risk_tolerance,
            salary_requirements=salary_requirements,
            timeline_flexibility=timeline_flexibility
        )
        
        self.transition_profiles[researcher.id] = profile
        
        logger.info(f"Created transition profile for {researcher.name}: "
                   f"{current_path.value} → {[p.value for p in preferred_paths]}")
        
        return profile
    
    def calculate_transition_probability(self, researcher: EnhancedResearcher,
                                       target_path: CareerPath) -> float:
        """
        Calculate probability of successful transition to target career path.
        
        Args:
            researcher: The researcher considering transition
            target_path: Target career path
            
        Returns:
            Probability of successful transition (0.0 to 1.0)
        """
        profile = self.transition_profiles.get(researcher.id)
        if not profile:
            # Create default profile
            profile = self.create_transition_profile(researcher, [target_path])
        
        # Base probability from transition difficulty
        current_path = profile.current_path
        base_difficulty = self.TRANSITION_DIFFICULTY.get(current_path, {}).get(target_path, 0.5)
        base_probability = 1.0 - base_difficulty
        
        # Adjust for researcher qualifications
        qualification_factor = self._calculate_qualification_factor(researcher, target_path)
        
        # Adjust for market conditions
        market_factor = self._calculate_market_factor(target_path)
        
        # Adjust for researcher characteristics
        personal_factor = self._calculate_personal_factor(researcher, profile, target_path)
        
        # Combine factors
        final_probability = base_probability * qualification_factor * market_factor * personal_factor
        
        # Apply constraints
        if self._has_blocking_constraints(profile, target_path):
            final_probability *= 0.3  # Significant reduction for constraints
        
        return max(0.0, min(1.0, final_probability))
    
    def create_transition_plan(self, researcher: EnhancedResearcher,
                             target_path: CareerPath) -> TransitionPlan:
        """
        Create detailed transition plan for researcher to target career path.
        
        Args:
            researcher: The researcher planning transition
            target_path: Target career path
            
        Returns:
            TransitionPlan with detailed guidance
        """
        profile = self.transition_profiles.get(researcher.id)
        if not profile:
            profile = self.create_transition_profile(researcher, [target_path])
        
        # Calculate transition probability
        probability = self.calculate_transition_probability(researcher, target_path)
        
        # Estimate timeline
        timeline = self._estimate_transition_timeline(researcher, target_path, probability)
        
        # Identify required skills and gaps
        required_skills = self._identify_required_skills(target_path)
        skill_gaps = self._identify_skill_gaps(researcher, required_skills)
        
        # Generate preparation steps
        preparation_steps = self._generate_preparation_steps(researcher, target_path, skill_gaps)
        
        # Identify networking requirements
        networking_requirements = self._identify_networking_requirements(target_path)
        
        # Calculate financial considerations
        financial_considerations = self._calculate_financial_considerations(
            researcher, profile.current_path, target_path
        )
        
        # Identify risk and success factors
        risk_factors = self._identify_risk_factors(researcher, target_path)
        success_factors = self._identify_success_factors(researcher, target_path)
        
        plan = TransitionPlan(
            researcher_id=researcher.id,
            source_path=profile.current_path,
            target_path=target_path,
            transition_probability=probability,
            estimated_timeline=timeline,
            required_skills=required_skills,
            skill_gaps=skill_gaps,
            preparation_steps=preparation_steps,
            networking_requirements=networking_requirements,
            financial_considerations=financial_considerations,
            risk_factors=risk_factors,
            success_factors=success_factors
        )
        
        # Store plan
        if researcher.id not in self.transition_plans:
            self.transition_plans[researcher.id] = []
        self.transition_plans[researcher.id].append(plan)
        
        logger.info(f"Created transition plan for {researcher.name}: "
                   f"{profile.current_path.value} → {target_path.value} "
                   f"(probability: {probability:.2f})")
        
        return plan
    
    def _determine_current_path(self, researcher: EnhancedResearcher) -> CareerPath:
        """Determine researcher's current career path."""
        if researcher.level in [ResearcherLevel.GRADUATE_STUDENT, ResearcherLevel.POSTDOC]:
            return CareerPath.ACADEMIC_RESEARCH
        elif researcher.level in [ResearcherLevel.ASSISTANT_PROF, ResearcherLevel.ASSOCIATE_PROF, 
                                ResearcherLevel.FULL_PROF, ResearcherLevel.EMERITUS]:
            # Could be research or teaching focused - assume research for simulation
            return CareerPath.ACADEMIC_RESEARCH
        else:
            return CareerPath.ACADEMIC_RESEARCH  # Default
    
    def _calculate_salary_requirements(self, researcher: EnhancedResearcher) -> Tuple[int, int]:
        """Calculate researcher's salary requirements."""
        # Base on current level and experience
        level_salaries = {
            ResearcherLevel.GRADUATE_STUDENT: (40000, 60000),
            ResearcherLevel.POSTDOC: (50000, 70000),
            ResearcherLevel.ASSISTANT_PROF: (70000, 100000),
            ResearcherLevel.ASSOCIATE_PROF: (80000, 120000),
            ResearcherLevel.FULL_PROF: (100000, 150000),
            ResearcherLevel.EMERITUS: (60000, 100000)
        }
        
        base_min, base_max = level_salaries.get(researcher.level, (50000, 80000))
        
        # Adjust for reputation and experience
        reputation_multiplier = 0.8 + (researcher.reputation_score * 0.4)
        experience_multiplier = 1.0 + (researcher.years_active * 0.02)
        
        min_salary = int(base_min * reputation_multiplier * experience_multiplier)
        preferred_salary = int(base_max * reputation_multiplier * experience_multiplier)
        
        return (min_salary, preferred_salary)
    
    def _estimate_timeline_flexibility(self, researcher: EnhancedResearcher) -> int:
        """Estimate how long researcher is willing to wait for transition."""
        # Younger researchers typically more flexible
        if researcher.level in [ResearcherLevel.GRADUATE_STUDENT, ResearcherLevel.POSTDOC]:
            return random.randint(6, 18)  # 6-18 months
        elif researcher.level == ResearcherLevel.ASSISTANT_PROF:
            return random.randint(12, 24)  # 1-2 years
        else:
            return random.randint(6, 12)  # 6-12 months (more urgent)
    
    def _calculate_qualification_factor(self, researcher: EnhancedResearcher, 
                                      target_path: CareerPath) -> float:
        """Calculate how well researcher's qualifications match target path."""
        factor = 0.5  # Base factor
        
        # Publications factor
        pub_count = len(researcher.publication_history)
        if target_path in [CareerPath.ACADEMIC_RESEARCH, CareerPath.GOVERNMENT_LAB]:
            factor += min(0.3, pub_count * 0.02)  # Publications very important
        elif target_path in [CareerPath.INDUSTRY_RESEARCH, CareerPath.NON_PROFIT]:
            factor += min(0.2, pub_count * 0.015)  # Publications somewhat important
        else:
            factor += min(0.1, pub_count * 0.01)  # Publications less important
        
        # H-index factor
        if researcher.h_index > 10:
            factor += 0.1
        if researcher.h_index > 20:
            factor += 0.1
        
        # Experience factor
        experience_bonus = min(0.2, researcher.years_active * 0.03)
        factor += experience_bonus
        
        # Level factor
        if researcher.level in [ResearcherLevel.ASSOCIATE_PROF, ResearcherLevel.FULL_PROF]:
            factor += 0.1  # Senior level helps with most transitions
        
        return min(1.5, factor)  # Cap at 1.5x
    
    def _calculate_market_factor(self, target_path: CareerPath) -> float:
        """Calculate market conditions factor for target path."""
        # Market demand by career path (simplified)
        market_demand = {
            CareerPath.ACADEMIC_RESEARCH: 0.3,  # Tight academic job market
            CareerPath.ACADEMIC_TEACHING: 0.5,  # Moderate demand
            CareerPath.INDUSTRY_RESEARCH: 0.8,  # High demand
            CareerPath.INDUSTRY_APPLIED: 0.9,   # Very high demand
            CareerPath.GOVERNMENT_LAB: 0.6,     # Moderate demand
            CareerPath.CONSULTING: 0.7,         # Good demand
            CareerPath.ENTREPRENEURSHIP: 1.0,   # Always possible (but risky)
            CareerPath.NON_PROFIT: 0.4,         # Limited positions
            CareerPath.SCIENCE_POLICY: 0.5,     # Moderate demand
            CareerPath.SCIENCE_COMMUNICATION: 0.6  # Growing field
        }
        
        return market_demand.get(target_path, 0.5)
    
    def _calculate_personal_factor(self, researcher: EnhancedResearcher,
                                 profile: TransitionProfile, target_path: CareerPath) -> float:
        """Calculate personal factors affecting transition success."""
        factor = 1.0
        
        # Risk tolerance factor
        target_incentives = self.get_incentive_structure(target_path)
        if target_incentives.job_security_score < 0.5 and profile.risk_tolerance < 0.5:
            factor *= 0.7  # Risk-averse person considering risky path
        elif target_incentives.job_security_score > 0.7 and profile.risk_tolerance > 0.7:
            factor *= 1.2  # Risk-tolerant person considering safe path
        
        # Age/career stage factor
        if researcher.level in [ResearcherLevel.GRADUATE_STUDENT, ResearcherLevel.POSTDOC]:
            factor *= 1.3  # Easier to transition early in career
        elif researcher.level in [ResearcherLevel.ASSOCIATE_PROF, ResearcherLevel.FULL_PROF]:
            factor *= 0.8  # Harder to transition later in career
        
        return factor
    
    def _has_blocking_constraints(self, profile: TransitionProfile, 
                                target_path: CareerPath) -> bool:
        """Check if profile has constraints that block transition."""
        target_incentives = self.get_incentive_structure(target_path)
        
        # Geographic constraints
        if "geographic_immobility" in profile.constraints:
            if target_incentives.geographic_flexibility < 0.5:
                return True
        
        # Salary constraints
        if target_incentives.salary_range[1] < profile.salary_requirements[0]:
            return True  # Target path can't meet minimum salary needs
        
        return False
    
    def _estimate_transition_timeline(self, researcher: EnhancedResearcher,
                                    target_path: CareerPath, probability: float) -> int:
        """Estimate timeline for transition in months."""
        # Base timeline by transition type
        base_timelines = {
            CareerPath.ACADEMIC_RESEARCH: 12,
            CareerPath.ACADEMIC_TEACHING: 6,
            CareerPath.INDUSTRY_RESEARCH: 4,
            CareerPath.INDUSTRY_APPLIED: 3,
            CareerPath.GOVERNMENT_LAB: 8,
            CareerPath.CONSULTING: 2,
            CareerPath.ENTREPRENEURSHIP: 6,
            CareerPath.NON_PROFIT: 6,
            CareerPath.SCIENCE_POLICY: 10,
            CareerPath.SCIENCE_COMMUNICATION: 4
        }
        
        base_timeline = base_timelines.get(target_path, 6)
        
        # Adjust for probability (lower probability = longer timeline)
        probability_factor = 2.0 - probability  # 1.0 to 2.0 multiplier
        
        # Adjust for researcher level
        level_factor = 1.0
        if researcher.level in [ResearcherLevel.ASSOCIATE_PROF, ResearcherLevel.FULL_PROF]:
            level_factor = 0.8  # Senior researchers may transition faster
        
        final_timeline = int(base_timeline * probability_factor * level_factor)
        return max(1, min(36, final_timeline))  # Cap between 1 and 36 months
    
    def _identify_required_skills(self, target_path: CareerPath) -> List[str]:
        """Identify skills required for target career path."""
        skill_requirements = {
            CareerPath.ACADEMIC_RESEARCH: [
                "Research methodology", "Grant writing", "Publication writing",
                "Teaching", "Mentoring", "Collaboration"
            ],
            CareerPath.ACADEMIC_TEACHING: [
                "Curriculum development", "Pedagogical training", "Assessment design",
                "Classroom management", "Educational technology"
            ],
            CareerPath.INDUSTRY_RESEARCH: [
                "Applied research", "Project management", "Technology transfer",
                "Intellectual property", "Business acumen", "Team leadership"
            ],
            CareerPath.INDUSTRY_APPLIED: [
                "Product development", "Software engineering", "Data analysis",
                "Project management", "Customer focus", "Agile methodologies"
            ],
            CareerPath.GOVERNMENT_LAB: [
                "Policy understanding", "Regulatory knowledge", "Report writing",
                "Stakeholder engagement", "Security clearance"
            ],
            CareerPath.CONSULTING: [
                "Business analysis", "Client management", "Presentation skills",
                "Problem solving", "Industry knowledge", "Sales skills"
            ],
            CareerPath.ENTREPRENEURSHIP: [
                "Business planning", "Fundraising", "Marketing", "Leadership",
                "Risk management", "Financial management", "Networking"
            ],
            CareerPath.NON_PROFIT: [
                "Mission alignment", "Fundraising", "Grant writing",
                "Community engagement", "Program management"
            ],
            CareerPath.SCIENCE_POLICY: [
                "Policy analysis", "Government relations", "Communication",
                "Stakeholder engagement", "Regulatory knowledge"
            ],
            CareerPath.SCIENCE_COMMUNICATION: [
                "Writing skills", "Media relations", "Public speaking",
                "Social media", "Journalism", "Visual communication"
            ]
        }
        
        return skill_requirements.get(target_path, [])
    
    def _identify_skill_gaps(self, researcher: EnhancedResearcher, 
                           required_skills: List[str]) -> List[str]:
        """Identify skill gaps for researcher."""
        # Simplified skill gap analysis
        # In a real system, this would analyze researcher's actual skills
        
        # Assume researchers have academic skills
        academic_skills = {
            "Research methodology", "Publication writing", "Teaching", 
            "Mentoring", "Collaboration", "Grant writing"
        }
        
        # Identify gaps
        gaps = []
        for skill in required_skills:
            if skill not in academic_skills:
                gaps.append(skill)
        
        # Add some academic skills as gaps for junior researchers
        if researcher.level in [ResearcherLevel.GRADUATE_STUDENT, ResearcherLevel.POSTDOC]:
            if "Grant writing" in required_skills:
                gaps.append("Grant writing")
            if "Mentoring" in required_skills:
                gaps.append("Mentoring")
        
        return gaps
    
    def _generate_preparation_steps(self, researcher: EnhancedResearcher,
                                  target_path: CareerPath, skill_gaps: List[str]) -> List[str]:
        """Generate preparation steps for transition."""
        steps = []
        
        # Skill development steps
        for gap in skill_gaps:
            if gap == "Business analysis":
                steps.append("Take business analysis course or certification")
            elif gap == "Project management":
                steps.append("Obtain PMP or similar project management certification")
            elif gap == "Software engineering":
                steps.append("Complete software development bootcamp or courses")
            elif gap == "Marketing":
                steps.append("Take marketing courses and develop marketing portfolio")
            elif gap == "Financial management":
                steps.append("Complete finance courses or MBA program")
            else:
                steps.append(f"Develop {gap.lower()} skills through training or experience")
        
        # General preparation steps
        steps.extend([
            "Update resume for target industry",
            "Build professional network in target field",
            "Conduct informational interviews",
            "Attend industry conferences and events",
            "Consider internship or consulting projects"
        ])
        
        # Path-specific steps
        if target_path == CareerPath.INDUSTRY_RESEARCH:
            steps.append("Highlight applied research experience")
            steps.append("Emphasize technology transfer potential")
        elif target_path == CareerPath.CONSULTING:
            steps.append("Develop case study portfolio")
            steps.append("Practice business presentation skills")
        elif target_path == CareerPath.ENTREPRENEURSHIP:
            steps.append("Develop business plan")
            steps.append("Identify potential co-founders")
            steps.append("Research funding opportunities")
        
        return steps
    
    def _identify_networking_requirements(self, target_path: CareerPath) -> List[str]:
        """Identify networking requirements for target path."""
        networking_needs = {
            CareerPath.ACADEMIC_RESEARCH: [
                "Academic conferences", "Research collaborators", "Funding agencies"
            ],
            CareerPath.ACADEMIC_TEACHING: [
                "Teaching conferences", "Educational organizations", "Curriculum committees"
            ],
            CareerPath.INDUSTRY_RESEARCH: [
                "Industry conferences", "R&D professionals", "Technology transfer offices"
            ],
            CareerPath.INDUSTRY_APPLIED: [
                "Professional associations", "Industry meetups", "Alumni networks"
            ],
            CareerPath.GOVERNMENT_LAB: [
                "Government scientists", "Policy makers", "Regulatory agencies"
            ],
            CareerPath.CONSULTING: [
                "Consulting firms", "Business networks", "Client industries"
            ],
            CareerPath.ENTREPRENEURSHIP: [
                "Startup ecosystems", "Investors", "Accelerators", "Mentors"
            ],
            CareerPath.NON_PROFIT: [
                "Non-profit organizations", "Foundation networks", "Mission-aligned groups"
            ],
            CareerPath.SCIENCE_POLICY: [
                "Policy organizations", "Government relations", "Think tanks"
            ],
            CareerPath.SCIENCE_COMMUNICATION: [
                "Media professionals", "Science writers", "Communication organizations"
            ]
        }
        
        return networking_needs.get(target_path, [])
    
    def _calculate_financial_considerations(self, researcher: EnhancedResearcher,
                                         source_path: CareerPath, 
                                         target_path: CareerPath) -> Dict[str, float]:
        """Calculate financial implications of transition."""
        source_incentives = self.get_incentive_structure(source_path)
        target_incentives = self.get_incentive_structure(target_path)
        
        # Estimate current salary (midpoint of range)
        current_salary = (source_incentives.salary_range[0] + source_incentives.salary_range[1]) / 2
        
        # Estimate target salary (midpoint of range)
        target_salary = (target_incentives.salary_range[0] + target_incentives.salary_range[1]) / 2
        
        # Calculate percentage change
        salary_change = ((target_salary - current_salary) / current_salary) * 100
        
        # Estimate transition costs
        transition_costs = 5000  # Base cost for job search, training, etc.
        if target_path == CareerPath.ENTREPRENEURSHIP:
            transition_costs = 50000  # Higher costs for starting business
        elif target_path in [CareerPath.CONSULTING, CareerPath.INDUSTRY_APPLIED]:
            transition_costs = 2000  # Lower costs for similar transitions
        
        return {
            'current_salary_estimate': current_salary,
            'target_salary_estimate': target_salary,
            'salary_change_percentage': salary_change,
            'transition_costs': transition_costs,
            'payback_period_months': max(1, transition_costs / max(1, (target_salary - current_salary) / 12))
        }
    
    def _identify_risk_factors(self, researcher: EnhancedResearcher, 
                             target_path: CareerPath) -> List[str]:
        """Identify risk factors for transition."""
        risks = []
        
        target_incentives = self.get_incentive_structure(target_path)
        
        # Job security risks
        if target_incentives.job_security_score < 0.5:
            risks.append("Lower job security in target field")
        
        # Salary risks
        if target_incentives.salary_range[0] < 50000:
            risks.append("Potential for lower compensation")
        
        # Career progression risks
        if target_incentives.advancement_opportunities < 0.5:
            risks.append("Limited advancement opportunities")
        
        # Research freedom risks
        if (target_incentives.research_freedom_score < 0.5 and 
            researcher.level in [ResearcherLevel.ASSISTANT_PROF, ResearcherLevel.ASSOCIATE_PROF, ResearcherLevel.FULL_PROF]):
            risks.append("Reduced research autonomy")
        
        # Age/experience risks
        if researcher.years_active > 15:
            risks.append("Age discrimination in new field")
        
        # Path-specific risks
        if target_path == CareerPath.ENTREPRENEURSHIP:
            risks.extend(["High failure rate", "Financial instability", "Work-life balance challenges"])
        elif target_path == CareerPath.CONSULTING:
            risks.append("Travel requirements and irregular schedule")
        elif target_path in [CareerPath.INDUSTRY_RESEARCH, CareerPath.INDUSTRY_APPLIED]:
            risks.append("Corporate restructuring and layoffs")
        
        return risks
    
    def _identify_success_factors(self, researcher: EnhancedResearcher,
                                target_path: CareerPath) -> List[str]:
        """Identify factors that increase transition success probability."""
        factors = []
        
        # Publication record
        if len(researcher.publication_history) > 10:
            factors.append("Strong publication record demonstrates expertise")
        
        # H-index and citations
        if researcher.h_index > 15:
            factors.append("High research impact and visibility")
        
        # Experience level
        if researcher.years_active > 5:
            factors.append("Substantial research experience")
        
        # Reputation
        if researcher.reputation_score > 0.7:
            factors.append("Strong professional reputation")
        
        # Institution tier
        if researcher.institution_tier <= 2:
            factors.append("Prestigious institutional affiliation")
        
        # Path-specific success factors
        if target_path in [CareerPath.INDUSTRY_RESEARCH, CareerPath.INDUSTRY_APPLIED]:
            factors.extend([
                "Technical skills transferable to industry",
                "Growing demand for PhD-level talent in industry"
            ])
        elif target_path == CareerPath.GOVERNMENT_LAB:
            factors.append("Mission-driven research aligns with government priorities")
        elif target_path == CareerPath.CONSULTING:
            factors.append("Analytical and problem-solving skills valued in consulting")
        elif target_path == CareerPath.ENTREPRENEURSHIP:
            factors.extend([
                "Deep technical expertise for technology ventures",
                "Research experience in identifying and solving problems"
            ])
        
        return factors
    
    def record_transition_outcome(self, researcher_id: str, target_path: CareerPath,
                                outcome: TransitionOutcome, actual_timeline: int,
                                salary_change: float, satisfaction_score: float,
                                lessons_learned: Optional[List[str]] = None,
                                challenges_faced: Optional[List[str]] = None) -> TransitionOutcomeRecord:
        """
        Record the outcome of a career transition.
        
        Args:
            researcher_id: ID of the researcher
            target_path: Target career path
            outcome: Actual outcome of transition
            actual_timeline: Actual time taken (months)
            salary_change: Actual salary change (percentage)
            satisfaction_score: Satisfaction with transition (0-1)
            lessons_learned: Lessons from the transition
            challenges_faced: Challenges encountered
            
        Returns:
            TransitionOutcomeRecord object
        """
        profile = self.transition_profiles.get(researcher_id)
        if not profile:
            raise CareerSystemError(f"No transition profile found for researcher {researcher_id}")
        
        if lessons_learned is None:
            lessons_learned = []
        if challenges_faced is None:
            challenges_faced = []
        
        # Generate success factors based on outcome
        success_factors = []
        if outcome == TransitionOutcome.SUCCESSFUL:
            success_factors = ["Strong preparation", "Good market timing", "Effective networking"]
        elif outcome == TransitionOutcome.PARTIALLY_SUCCESSFUL:
            success_factors = ["Adequate preparation", "Some market challenges"]
        
        record = TransitionOutcomeRecord(
            researcher_id=researcher_id,
            transition_date=date.today(),
            source_path=profile.current_path,
            target_path=target_path,
            outcome=outcome,
            actual_timeline=actual_timeline,
            salary_change=salary_change,
            satisfaction_score=satisfaction_score,
            lessons_learned=lessons_learned,
            challenges_faced=challenges_faced,
            success_factors=success_factors
        )
        
        # Store outcome
        if researcher_id not in self.transition_outcomes:
            self.transition_outcomes[researcher_id] = []
        self.transition_outcomes[researcher_id].append(record)
        
        logger.info(f"Recorded transition outcome for researcher {researcher_id}: "
                   f"{profile.current_path.value} → {target_path.value} = {outcome.value}")
        
        return record
    
    def analyze_transition_patterns(self, field: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze patterns in career transitions.
        
        Args:
            field: Optional field filter
            
        Returns:
            Dictionary with transition pattern analysis
        """
        all_outcomes = []
        for outcomes in self.transition_outcomes.values():
            all_outcomes.extend(outcomes)
        
        if not all_outcomes:
            return {'error': 'No transition data available'}
        
        # Success rates by target path
        success_rates = {}
        for path in CareerPath:
            path_outcomes = [o for o in all_outcomes if o.target_path == path]
            if path_outcomes:
                successful = len([o for o in path_outcomes if o.outcome == TransitionOutcome.SUCCESSFUL])
                success_rates[path.value] = successful / len(path_outcomes)
        
        # Average timelines by path
        avg_timelines = {}
        for path in CareerPath:
            path_outcomes = [o for o in all_outcomes if o.target_path == path]
            if path_outcomes:
                avg_timelines[path.value] = sum(o.actual_timeline for o in path_outcomes) / len(path_outcomes)
        
        # Salary change analysis
        salary_changes = {}
        for path in CareerPath:
            path_outcomes = [o for o in all_outcomes if o.target_path == path]
            if path_outcomes:
                avg_change = sum(o.salary_change for o in path_outcomes) / len(path_outcomes)
                salary_changes[path.value] = avg_change
        
        # Satisfaction scores
        satisfaction_scores = {}
        for path in CareerPath:
            path_outcomes = [o for o in all_outcomes if o.target_path == path]
            if path_outcomes:
                avg_satisfaction = sum(o.satisfaction_score for o in path_outcomes) / len(path_outcomes)
                satisfaction_scores[path.value] = avg_satisfaction
        
        # Most common challenges
        all_challenges = []
        for outcome in all_outcomes:
            all_challenges.extend(outcome.challenges_faced)
        
        challenge_counts = {}
        for challenge in all_challenges:
            challenge_counts[challenge] = challenge_counts.get(challenge, 0) + 1
        
        common_challenges = sorted(challenge_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_transitions': len(all_outcomes),
            'success_rates_by_path': success_rates,
            'average_timelines_months': avg_timelines,
            'salary_change_percentages': salary_changes,
            'satisfaction_scores': satisfaction_scores,
            'most_common_challenges': common_challenges,
            'overall_success_rate': len([o for o in all_outcomes if o.outcome == TransitionOutcome.SUCCESSFUL]) / len(all_outcomes)
        }
    
    def get_transition_recommendations(self, researcher: EnhancedResearcher) -> Dict[str, Any]:
        """
        Get personalized transition recommendations for researcher.
        
        Args:
            researcher: The researcher to provide recommendations for
            
        Returns:
            Dictionary with personalized recommendations
        """
        # Calculate probabilities for all career paths
        probabilities = {}
        for path in CareerPath:
            if path != self._determine_current_path(researcher):  # Don't recommend staying
                probabilities[path.value] = self.calculate_transition_probability(researcher, path)
        
        # Sort by probability
        sorted_paths = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 3 recommendations
        top_recommendations = []
        for path_name, probability in sorted_paths[:3]:
            path = CareerPath(path_name)
            incentives = self.get_incentive_structure(path)
            
            recommendation = {
                'career_path': path_name,
                'success_probability': probability,
                'salary_range': incentives.salary_range,
                'key_benefits': self._get_key_benefits(incentives),
                'main_challenges': self._get_main_challenges(researcher, path),
                'preparation_time': self._estimate_transition_timeline(researcher, path, probability)
            }
            top_recommendations.append(recommendation)
        
        # Overall assessment
        profile = self.transition_profiles.get(researcher.id)
        current_path = self._determine_current_path(researcher)
        current_incentives = self.get_incentive_structure(current_path)
        
        return {
            'researcher_id': researcher.id,
            'current_path': current_path.value,
            'current_path_benefits': self._get_key_benefits(current_incentives),
            'top_recommendations': top_recommendations,
            'transition_readiness': self._assess_transition_readiness(researcher),
            'general_advice': self._generate_general_advice(researcher, sorted_paths)
        }
    
    def _get_key_benefits(self, incentives: IncentiveStructure) -> List[str]:
        """Get key benefits of a career path."""
        benefits = []
        
        if incentives.salary_range[1] > 120000:
            benefits.append("High earning potential")
        if incentives.job_security_score > 0.7:
            benefits.append("Strong job security")
        if incentives.research_freedom_score > 0.7:
            benefits.append("High research autonomy")
        if incentives.work_life_balance_score > 0.7:
            benefits.append("Good work-life balance")
        if incentives.advancement_opportunities > 0.7:
            benefits.append("Clear advancement paths")
        if incentives.impact_potential > 0.7:
            benefits.append("High impact potential")
        
        return benefits[:3]  # Return top 3 benefits
    
    def _get_main_challenges(self, researcher: EnhancedResearcher, 
                           target_path: CareerPath) -> List[str]:
        """Get main challenges for transition."""
        challenges = []
        
        current_path = self._determine_current_path(researcher)
        difficulty = self.TRANSITION_DIFFICULTY.get(current_path, {}).get(target_path, 0.5)
        
        if difficulty > 0.6:
            challenges.append("Significant career change required")
        
        target_incentives = self.get_incentive_structure(target_path)
        
        if target_incentives.publication_importance < 0.3:
            challenges.append("Academic publications less valued")
        if target_incentives.job_security_score < 0.5:
            challenges.append("Lower job security")
        if target_incentives.research_freedom_score < 0.5:
            challenges.append("Reduced research autonomy")
        
        return challenges[:3]  # Return top 3 challenges
    
    def _assess_transition_readiness(self, researcher: EnhancedResearcher) -> str:
        """Assess researcher's readiness for career transition."""
        readiness_score = 0
        
        # Experience factor
        if researcher.years_active > 3:
            readiness_score += 1
        if researcher.years_active > 8:
            readiness_score += 1
        
        # Publication record
        if len(researcher.publication_history) > 5:
            readiness_score += 1
        if len(researcher.publication_history) > 15:
            readiness_score += 1
        
        # Reputation
        if researcher.reputation_score > 0.6:
            readiness_score += 1
        
        # Career stage
        if researcher.level in [ResearcherLevel.POSTDOC, ResearcherLevel.ASSISTANT_PROF]:
            readiness_score += 1  # Good time for transition
        
        if readiness_score >= 5:
            return "Highly ready for transition"
        elif readiness_score >= 3:
            return "Moderately ready for transition"
        else:
            return "Should build more experience before transitioning"
    
    def _generate_general_advice(self, researcher: EnhancedResearcher,
                               sorted_paths: List[Tuple[str, float]]) -> List[str]:
        """Generate general transition advice."""
        advice = []
        
        # Based on career stage
        if researcher.level in [ResearcherLevel.GRADUATE_STUDENT, ResearcherLevel.POSTDOC]:
            advice.append("Consider gaining more research experience before major career change")
            advice.append("Explore internships or consulting projects to test interest")
        elif researcher.level == ResearcherLevel.ASSISTANT_PROF:
            advice.append("Evaluate tenure prospects before considering transition")
            advice.append("Industry transition may be easier before tenure decision")
        else:
            advice.append("Leverage senior experience and network for transition")
        
        # Based on top recommendations
        if sorted_paths and sorted_paths[0][1] > 0.7:
            advice.append(f"Strong prospects for {sorted_paths[0][0]} transition")
        elif sorted_paths and sorted_paths[0][1] < 0.4:
            advice.append("Consider building additional skills before transitioning")
        
        # General advice
        advice.extend([
            "Network actively in target industries",
            "Develop transferable skills through side projects",
            "Consider gradual transition through consulting or part-time work"
        ])
        
        return advice