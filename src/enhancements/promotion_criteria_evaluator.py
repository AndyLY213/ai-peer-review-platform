"""
Promotion Criteria Evaluation System

This module implements promotion criteria evaluation for teaching/service/research balance,
promotion readiness evaluation with field-specific weights, promotion timeline and
requirement tracking for the peer review simulation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging
import json

from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel, CareerStage
from src.core.exceptions import ValidationError, CareerSystemError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class PromotionType(Enum):
    """Types of academic promotions."""
    ASSISTANT_TO_ASSOCIATE = "Assistant to Associate Professor"
    ASSOCIATE_TO_FULL = "Associate to Full Professor"
    LECTURER_TO_ASSISTANT = "Lecturer to Assistant Professor"
    CLINICAL_PROMOTION = "Clinical Track Promotion"
    TEACHING_TRACK_PROMOTION = "Teaching Track Promotion"


class PromotionOutcome(Enum):
    """Possible promotion outcomes."""
    APPROVED = "Approved"
    DENIED = "Denied"
    DEFERRED = "Deferred"
    CONDITIONAL = "Conditional Approval"
    WITHDRAWN = "Withdrawn"


@dataclass
class PromotionWeights:
    """Field-specific weights for promotion criteria."""
    research_weight: float = 0.6  # Research contribution weight
    teaching_weight: float = 0.25  # Teaching effectiveness weight
    service_weight: float = 0.15  # Service contribution weight
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.research_weight + self.teaching_weight + self.service_weight
        if abs(total - 1.0) > 0.01:
            raise ValidationError("promotion_weights", total, "weights must sum to 1.0")


@dataclass
class ResearchCriteria:
    """Research evaluation criteria."""
    min_publications: int
    min_first_author_publications: int
    min_journal_publications: int
    min_h_index: int
    min_citations: int
    min_external_funding: float  # in dollars
    required_collaborations: int
    international_visibility: bool = False
    
    def __post_init__(self):
        """Validate research criteria."""
        if self.min_publications < 0:
            raise ValidationError("min_publications", self.min_publications, "non-negative integer")
        if self.min_external_funding < 0:
            raise ValidationError("min_external_funding", self.min_external_funding, "non-negative number")


@dataclass
class TeachingCriteria:
    """Teaching evaluation criteria."""
    min_courses_taught: int
    min_student_evaluation_score: float  # 1-5 scale
    required_course_development: int  # number of new courses developed
    mentoring_requirements: int  # number of students mentored
    teaching_innovation_required: bool = False
    
    def __post_init__(self):
        """Validate teaching criteria."""
        if not (1.0 <= self.min_student_evaluation_score <= 5.0):
            raise ValidationError("min_student_evaluation_score", 
                                self.min_student_evaluation_score, "score between 1.0 and 5.0")


@dataclass
class ServiceCriteria:
    """Service evaluation criteria."""
    min_internal_service_roles: int
    min_external_service_roles: int
    editorial_board_experience: bool = False
    conference_organization_required: bool = False
    grant_review_experience: bool = False
    leadership_roles_required: int = 0
    
    def __post_init__(self):
        """Validate service criteria."""
        if self.min_internal_service_roles < 0:
            raise ValidationError("min_internal_service_roles", 
                                self.min_internal_service_roles, "non-negative integer")


@dataclass
class PromotionRequirements:
    """Complete promotion requirements for a specific level and field."""
    promotion_type: PromotionType
    field: str
    research_criteria: ResearchCriteria
    teaching_criteria: TeachingCriteria
    service_criteria: ServiceCriteria
    weights: PromotionWeights
    minimum_years_in_rank: int
    external_letters_required: int = 6
    
    def __post_init__(self):
        """Validate promotion requirements."""
        if self.minimum_years_in_rank < 0:
            raise ValidationError("minimum_years_in_rank", 
                                self.minimum_years_in_rank, "non-negative integer")


@dataclass
class PromotionEvaluation:
    """Results of promotion criteria evaluation."""
    researcher_id: str
    promotion_type: PromotionType
    evaluation_date: date
    research_score: float  # 0-100
    teaching_score: float  # 0-100
    service_score: float   # 0-100
    overall_score: float   # 0-100
    meets_minimum_requirements: bool
    recommendation: PromotionOutcome
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[str]
    timeline_assessment: str
    committee_notes: str = ""
    
    def __post_init__(self):
        """Validate evaluation scores."""
        for score_name in ['research_score', 'teaching_score', 'service_score', 'overall_score']:
            score = getattr(self, score_name)
            if not (0 <= score <= 100):
                raise ValidationError(score_name, score, "score between 0 and 100")


@dataclass
class PromotionTimeline:
    """Tracks promotion timeline and milestones."""
    researcher_id: str
    current_rank: ResearcherLevel
    target_rank: ResearcherLevel
    rank_start_date: date
    earliest_eligible_date: date
    recommended_application_date: date
    milestones_completed: List[str] = field(default_factory=list)
    milestones_pending: List[str] = field(default_factory=list)
    
    def get_years_in_rank(self) -> float:
        """Calculate years in current rank."""
        today = date.today()
        days_in_rank = (today - self.rank_start_date).days
        return days_in_rank / 365.25
    
    def is_eligible_for_promotion(self) -> bool:
        """Check if researcher is eligible for promotion."""
        return date.today() >= self.earliest_eligible_date


class PromotionCriteriaEvaluator:
    """
    Evaluates promotion readiness for teaching/service/research balance with
    field-specific weights, promotion timeline tracking, and requirement assessment.
    
    This class provides functionality to:
    - Evaluate promotion readiness with field-specific criteria
    - Balance teaching, service, and research contributions
    - Track promotion timeline and requirements
    - Generate comprehensive promotion assessments
    - Provide improvement recommendations
    """
    
    # Default promotion requirements by field and promotion type
    DEFAULT_REQUIREMENTS = {
        'computer_science': {
            PromotionType.ASSISTANT_TO_ASSOCIATE: PromotionRequirements(
                promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
                field='computer_science',
                research_criteria=ResearchCriteria(
                    min_publications=15,
                    min_first_author_publications=8,
                    min_journal_publications=5,
                    min_h_index=12,
                    min_citations=200,
                    min_external_funding=150000,
                    required_collaborations=3,
                    international_visibility=True
                ),
                teaching_criteria=TeachingCriteria(
                    min_courses_taught=6,
                    min_student_evaluation_score=3.5,
                    required_course_development=1,
                    mentoring_requirements=2,
                    teaching_innovation_required=False
                ),
                service_criteria=ServiceCriteria(
                    min_internal_service_roles=3,
                    min_external_service_roles=2,
                    editorial_board_experience=False,
                    conference_organization_required=False,
                    grant_review_experience=True,
                    leadership_roles_required=1
                ),
                weights=PromotionWeights(research_weight=0.65, teaching_weight=0.20, service_weight=0.15),
                minimum_years_in_rank=6,
                external_letters_required=6
            ),
            PromotionType.ASSOCIATE_TO_FULL: PromotionRequirements(
                promotion_type=PromotionType.ASSOCIATE_TO_FULL,
                field='computer_science',
                research_criteria=ResearchCriteria(
                    min_publications=30,
                    min_first_author_publications=15,
                    min_journal_publications=12,
                    min_h_index=20,
                    min_citations=500,
                    min_external_funding=500000,
                    required_collaborations=5,
                    international_visibility=True
                ),
                teaching_criteria=TeachingCriteria(
                    min_courses_taught=12,
                    min_student_evaluation_score=3.7,
                    required_course_development=2,
                    mentoring_requirements=5,
                    teaching_innovation_required=True
                ),
                service_criteria=ServiceCriteria(
                    min_internal_service_roles=5,
                    min_external_service_roles=4,
                    editorial_board_experience=True,
                    conference_organization_required=True,
                    grant_review_experience=True,
                    leadership_roles_required=2
                ),
                weights=PromotionWeights(research_weight=0.60, teaching_weight=0.20, service_weight=0.20),
                minimum_years_in_rank=6,
                external_letters_required=8
            )
        },
        'biology': {
            PromotionType.ASSISTANT_TO_ASSOCIATE: PromotionRequirements(
                promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
                field='biology',
                research_criteria=ResearchCriteria(
                    min_publications=20,
                    min_first_author_publications=10,
                    min_journal_publications=15,
                    min_h_index=15,
                    min_citations=400,
                    min_external_funding=300000,
                    required_collaborations=4,
                    international_visibility=True
                ),
                teaching_criteria=TeachingCriteria(
                    min_courses_taught=8,
                    min_student_evaluation_score=3.6,
                    required_course_development=1,
                    mentoring_requirements=3,
                    teaching_innovation_required=False
                ),
                service_criteria=ServiceCriteria(
                    min_internal_service_roles=4,
                    min_external_service_roles=3,
                    editorial_board_experience=False,
                    conference_organization_required=False,
                    grant_review_experience=True,
                    leadership_roles_required=1
                ),
                weights=PromotionWeights(research_weight=0.70, teaching_weight=0.18, service_weight=0.12),
                minimum_years_in_rank=6,
                external_letters_required=7
            ),
            PromotionType.ASSOCIATE_TO_FULL: PromotionRequirements(
                promotion_type=PromotionType.ASSOCIATE_TO_FULL,
                field='biology',
                research_criteria=ResearchCriteria(
                    min_publications=40,
                    min_first_author_publications=20,
                    min_journal_publications=30,
                    min_h_index=25,
                    min_citations=800,
                    min_external_funding=800000,
                    required_collaborations=6,
                    international_visibility=True
                ),
                teaching_criteria=TeachingCriteria(
                    min_courses_taught=15,
                    min_student_evaluation_score=3.8,
                    required_course_development=3,
                    mentoring_requirements=8,
                    teaching_innovation_required=True
                ),
                service_criteria=ServiceCriteria(
                    min_internal_service_roles=6,
                    min_external_service_roles=5,
                    editorial_board_experience=True,
                    conference_organization_required=True,
                    grant_review_experience=True,
                    leadership_roles_required=3
                ),
                weights=PromotionWeights(research_weight=0.65, teaching_weight=0.18, service_weight=0.17),
                minimum_years_in_rank=6,
                external_letters_required=9
            )
        },
        'physics': {
            PromotionType.ASSISTANT_TO_ASSOCIATE: PromotionRequirements(
                promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
                field='physics',
                research_criteria=ResearchCriteria(
                    min_publications=18,
                    min_first_author_publications=9,
                    min_journal_publications=12,
                    min_h_index=14,
                    min_citations=350,
                    min_external_funding=250000,
                    required_collaborations=4,
                    international_visibility=True
                ),
                teaching_criteria=TeachingCriteria(
                    min_courses_taught=7,
                    min_student_evaluation_score=3.5,
                    required_course_development=1,
                    mentoring_requirements=2,
                    teaching_innovation_required=False
                ),
                service_criteria=ServiceCriteria(
                    min_internal_service_roles=3,
                    min_external_service_roles=2,
                    editorial_board_experience=False,
                    conference_organization_required=False,
                    grant_review_experience=True,
                    leadership_roles_required=1
                ),
                weights=PromotionWeights(research_weight=0.68, teaching_weight=0.19, service_weight=0.13),
                minimum_years_in_rank=6,
                external_letters_required=6
            ),
            PromotionType.ASSOCIATE_TO_FULL: PromotionRequirements(
                promotion_type=PromotionType.ASSOCIATE_TO_FULL,
                field='physics',
                research_criteria=ResearchCriteria(
                    min_publications=35,
                    min_first_author_publications=18,
                    min_journal_publications=25,
                    min_h_index=22,
                    min_citations=700,
                    min_external_funding=600000,
                    required_collaborations=6,
                    international_visibility=True
                ),
                teaching_criteria=TeachingCriteria(
                    min_courses_taught=14,
                    min_student_evaluation_score=3.7,
                    required_course_development=2,
                    mentoring_requirements=6,
                    teaching_innovation_required=True
                ),
                service_criteria=ServiceCriteria(
                    min_internal_service_roles=5,
                    min_external_service_roles=4,
                    editorial_board_experience=True,
                    conference_organization_required=True,
                    grant_review_experience=True,
                    leadership_roles_required=2
                ),
                weights=PromotionWeights(research_weight=0.63, teaching_weight=0.19, service_weight=0.18),
                minimum_years_in_rank=6,
                external_letters_required=8
            )
        }
    }
    
    def __init__(self):
        """Initialize the promotion criteria evaluator."""
        logger.info("Initializing Promotion Criteria Evaluation System")
        self.promotion_requirements: Dict[str, Dict[PromotionType, PromotionRequirements]] = self.DEFAULT_REQUIREMENTS.copy()
        self.promotion_timelines: Dict[str, PromotionTimeline] = {}
        self.evaluations: Dict[str, List[PromotionEvaluation]] = {}
    
    def create_promotion_timeline(self, researcher: EnhancedResearcher,
                                rank_start_date: Optional[date] = None) -> PromotionTimeline:
        """
        Create promotion timeline for a researcher.
        
        Args:
            researcher: The researcher to create timeline for
            rank_start_date: Date when current rank started (defaults to estimated date)
            
        Returns:
            PromotionTimeline object
            
        Raises:
            CareerSystemError: If researcher level is not eligible for promotion
        """
        if researcher.level == ResearcherLevel.EMERITUS:
            raise CareerSystemError("Emeritus researchers are not eligible for promotion")
        
        # Determine target rank
        target_rank_map = {
            ResearcherLevel.GRADUATE_STUDENT: ResearcherLevel.POSTDOC,
            ResearcherLevel.POSTDOC: ResearcherLevel.ASSISTANT_PROF,
            ResearcherLevel.ASSISTANT_PROF: ResearcherLevel.ASSOCIATE_PROF,
            ResearcherLevel.ASSOCIATE_PROF: ResearcherLevel.FULL_PROF,
            ResearcherLevel.FULL_PROF: ResearcherLevel.FULL_PROF  # Already at top
        }
        
        target_rank = target_rank_map.get(researcher.level)
        if not target_rank or target_rank == researcher.level:
            raise CareerSystemError(f"No promotion path available for {researcher.level.value}")
        
        # Estimate rank start date if not provided
        if rank_start_date is None:
            # Estimate based on years active and typical career progression
            years_per_rank = {
                ResearcherLevel.GRADUATE_STUDENT: 5,
                ResearcherLevel.POSTDOC: 3,
                ResearcherLevel.ASSISTANT_PROF: 6,
                ResearcherLevel.ASSOCIATE_PROF: 8
            }
            
            estimated_years_in_rank = min(researcher.years_active, 
                                        years_per_rank.get(researcher.level, 6))
            rank_start_date = date.today() - timedelta(days=int(estimated_years_in_rank * 365.25))
        
        # Calculate minimum years required for promotion
        min_years_map = {
            ResearcherLevel.ASSISTANT_PROF: 6,  # Typical tenure track
            ResearcherLevel.ASSOCIATE_PROF: 6,  # Typical associate to full
            ResearcherLevel.POSTDOC: 2,         # Minimum postdoc experience
            ResearcherLevel.GRADUATE_STUDENT: 4  # Minimum PhD time
        }
        
        min_years = min_years_map.get(researcher.level, 6)
        earliest_eligible_date = rank_start_date + timedelta(days=int(min_years * 365.25))
        
        # Recommend application date (typically 1 year after eligible)
        recommended_date = earliest_eligible_date + timedelta(days=365)
        
        # Generate standard milestones
        milestones = self._generate_promotion_milestones(researcher.level, target_rank)
        
        timeline = PromotionTimeline(
            researcher_id=researcher.id,
            current_rank=researcher.level,
            target_rank=target_rank,
            rank_start_date=rank_start_date,
            earliest_eligible_date=earliest_eligible_date,
            recommended_application_date=recommended_date,
            milestones_pending=milestones
        )
        
        self.promotion_timelines[researcher.id] = timeline
        
        logger.info(f"Created promotion timeline for {researcher.name}: "
                   f"{researcher.level.value} → {target_rank.value}")
        
        return timeline
    
    def evaluate_promotion_readiness(self, researcher: EnhancedResearcher,
                                   field: str = "computer_science") -> PromotionEvaluation:
        """
        Evaluate researcher's readiness for promotion with field-specific weights.
        
        Args:
            researcher: The researcher to evaluate
            field: Academic field for evaluation criteria
            
        Returns:
            PromotionEvaluation with comprehensive assessment
            
        Raises:
            CareerSystemError: If no promotion requirements found for field/level
        """
        # Determine promotion type
        promotion_type = self._get_promotion_type(researcher.level)
        if not promotion_type:
            raise CareerSystemError(f"No promotion available for {researcher.level.value}")
        
        # Get requirements
        field_requirements = self.promotion_requirements.get(field)
        if not field_requirements:
            raise CareerSystemError(f"No promotion requirements defined for field: {field}")
        
        requirements = field_requirements.get(promotion_type)
        if not requirements:
            raise CareerSystemError(f"No requirements for {promotion_type.value} in {field}")
        
        # Evaluate each component
        research_score = self._evaluate_research_criteria(researcher, requirements.research_criteria)
        teaching_score = self._evaluate_teaching_criteria(researcher, requirements.teaching_criteria)
        service_score = self._evaluate_service_criteria(researcher, requirements.service_criteria)
        
        # Calculate weighted overall score
        weights = requirements.weights
        overall_score = (
            research_score * weights.research_weight +
            teaching_score * weights.teaching_weight +
            service_score * weights.service_weight
        )
        
        # Check minimum requirements
        meets_minimum = self._check_minimum_requirements(researcher, requirements)
        
        # Determine recommendation
        recommendation = self._determine_promotion_recommendation(
            overall_score, meets_minimum, researcher
        )
        
        # Generate analysis
        strengths, weaknesses, improvements = self._analyze_promotion_readiness(
            researcher, requirements, research_score, teaching_score, service_score
        )
        
        # Timeline assessment
        timeline = self.promotion_timelines.get(researcher.id)
        timeline_assessment = self._assess_promotion_timeline(timeline, meets_minimum)
        
        evaluation = PromotionEvaluation(
            researcher_id=researcher.id,
            promotion_type=promotion_type,
            evaluation_date=date.today(),
            research_score=research_score,
            teaching_score=teaching_score,
            service_score=service_score,
            overall_score=overall_score,
            meets_minimum_requirements=meets_minimum,
            recommendation=recommendation,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_areas=improvements,
            timeline_assessment=timeline_assessment,
            committee_notes=f"Evaluation for {promotion_type.value} in {field}"
        )
        
        # Store evaluation
        if researcher.id not in self.evaluations:
            self.evaluations[researcher.id] = []
        self.evaluations[researcher.id].append(evaluation)
        
        logger.info(f"Completed promotion evaluation for {researcher.name}: "
                   f"{recommendation.value} (score: {overall_score:.1f})")
        
        return evaluation
    
    def _get_promotion_type(self, current_level: ResearcherLevel) -> Optional[PromotionType]:
        """Determine promotion type based on current level."""
        promotion_map = {
            ResearcherLevel.ASSISTANT_PROF: PromotionType.ASSISTANT_TO_ASSOCIATE,
            ResearcherLevel.ASSOCIATE_PROF: PromotionType.ASSOCIATE_TO_FULL
        }
        return promotion_map.get(current_level)
    
    def _evaluate_research_criteria(self, researcher: EnhancedResearcher,
                                  criteria: ResearchCriteria) -> float:
        """Evaluate research component (0-100 score)."""
        score = 0.0
        
        # Publications (30 points)
        pub_count = len(researcher.publication_history)
        pub_score = min(30.0, (pub_count / criteria.min_publications) * 30.0)
        score += pub_score
        
        # First author publications (20 points)
        first_author_count = len([p for p in researcher.publication_history 
                                if p.get('first_author', False)])
        first_author_score = min(20.0, (first_author_count / criteria.min_first_author_publications) * 20.0)
        score += first_author_score
        
        # H-index (25 points)
        h_index_score = min(25.0, (researcher.h_index / criteria.min_h_index) * 25.0)
        score += h_index_score
        
        # Citations (15 points)
        citation_score = min(15.0, (researcher.total_citations / criteria.min_citations) * 15.0)
        score += citation_score
        
        # External funding (10 points) - placeholder, would need funding data
        funding_score = 5.0  # Assume partial funding for simulation
        score += funding_score
        
        return min(100.0, score)
    
    def _evaluate_teaching_criteria(self, researcher: EnhancedResearcher,
                                  criteria: TeachingCriteria) -> float:
        """Evaluate teaching component (0-100 score)."""
        # Placeholder implementation - in real system would use teaching data
        base_score = 70.0  # Assume adequate teaching
        
        # Adjust based on experience and level
        experience_bonus = min(15.0, researcher.years_active * 2.0)
        
        # Senior researchers typically better at teaching
        level_bonus = 0.0
        if researcher.level in [ResearcherLevel.ASSOCIATE_PROF, ResearcherLevel.FULL_PROF]:
            level_bonus = 10.0
        
        # Add some variation for realism
        import random
        variation = random.uniform(-5, 10)
        
        total_score = base_score + experience_bonus + level_bonus + variation
        return max(0.0, min(100.0, total_score))
    
    def _evaluate_service_criteria(self, researcher: EnhancedResearcher,
                                 criteria: ServiceCriteria) -> float:
        """Evaluate service component (0-100 score)."""
        # Placeholder implementation - in real system would use service data
        base_score = 60.0  # Assume minimal service
        
        # Adjust based on years active (more service expected over time)
        experience_bonus = min(25.0, researcher.years_active * 3.0)
        
        # Senior levels expected to do more service
        level_bonus = 0.0
        if researcher.level == ResearcherLevel.ASSOCIATE_PROF:
            level_bonus = 10.0
        elif researcher.level == ResearcherLevel.FULL_PROF:
            level_bonus = 15.0
        
        total_score = base_score + experience_bonus + level_bonus
        return max(0.0, min(100.0, total_score))
    
    def _check_minimum_requirements(self, researcher: EnhancedResearcher,
                                  requirements: PromotionRequirements) -> bool:
        """Check if researcher meets minimum requirements."""
        research_criteria = requirements.research_criteria
        
        # Check key research minimums
        if len(researcher.publication_history) < research_criteria.min_publications:
            return False
        
        if researcher.h_index < research_criteria.min_h_index:
            return False
        
        if researcher.total_citations < research_criteria.min_citations:
            return False
        
        # Check time in rank
        timeline = self.promotion_timelines.get(researcher.id)
        if timeline and timeline.get_years_in_rank() < requirements.minimum_years_in_rank:
            return False
        
        return True
    
    def _determine_promotion_recommendation(self, overall_score: float,
                                         meets_minimum: bool,
                                         researcher: EnhancedResearcher) -> PromotionOutcome:
        """Determine promotion recommendation based on scores and requirements."""
        if not meets_minimum:
            return PromotionOutcome.DENIED
        
        if overall_score >= 85:
            return PromotionOutcome.APPROVED
        elif overall_score >= 70:
            return PromotionOutcome.CONDITIONAL
        elif overall_score >= 55:
            return PromotionOutcome.DEFERRED
        else:
            return PromotionOutcome.DENIED
    
    def _analyze_promotion_readiness(self, researcher: EnhancedResearcher,
                                   requirements: PromotionRequirements,
                                   research_score: float, teaching_score: float,
                                   service_score: float) -> Tuple[List[str], List[str], List[str]]:
        """Analyze strengths, weaknesses, and improvement areas."""
        strengths = []
        weaknesses = []
        improvements = []
        
        # Research analysis
        if research_score >= 80:
            strengths.append("Excellent research record")
        elif research_score < 60:
            weaknesses.append("Research record below expectations")
            improvements.append("Increase publication output and citation impact")
        
        # Teaching analysis
        if teaching_score >= 80:
            strengths.append("Strong teaching performance")
        elif teaching_score < 60:
            weaknesses.append("Teaching performance needs improvement")
            improvements.append("Focus on teaching effectiveness and student mentoring")
        
        # Service analysis
        if service_score >= 80:
            strengths.append("Excellent service contribution")
        elif service_score < 60:
            weaknesses.append("Service contribution insufficient")
            improvements.append("Take on more significant service roles")
        
        # Publication analysis
        pub_count = len(researcher.publication_history)
        if pub_count > requirements.research_criteria.min_publications * 1.2:
            strengths.append("Strong publication productivity")
        elif pub_count < requirements.research_criteria.min_publications:
            improvements.append("Increase publication output")
        
        # Citation impact analysis
        if researcher.h_index > requirements.research_criteria.min_h_index * 1.3:
            strengths.append("High citation impact")
        elif researcher.h_index < requirements.research_criteria.min_h_index:
            improvements.append("Work on increasing citation impact and visibility")
        
        return strengths, weaknesses, improvements
    
    def _assess_promotion_timeline(self, timeline: Optional[PromotionTimeline],
                                 meets_minimum: bool) -> str:
        """Assess promotion timeline and readiness."""
        if not timeline:
            return "No timeline established"
        
        years_in_rank = timeline.get_years_in_rank()
        is_eligible = timeline.is_eligible_for_promotion()
        
        if not is_eligible:
            years_remaining = (timeline.earliest_eligible_date - date.today()).days / 365.25
            return f"Not yet eligible. {years_remaining:.1f} years remaining until eligibility."
        
        if meets_minimum and years_in_rank >= 6:
            return "Ready for promotion application"
        elif meets_minimum:
            return "Meets requirements but could benefit from additional time"
        else:
            return "Not ready - minimum requirements not met"
    
    def _generate_promotion_milestones(self, current_level: ResearcherLevel,
                                     target_level: ResearcherLevel) -> List[str]:
        """Generate standard promotion milestones."""
        if current_level == ResearcherLevel.ASSISTANT_PROF:
            return [
                "Establish independent research program",
                "Secure external funding",
                "Develop teaching portfolio",
                "Take on service roles",
                "Build national visibility",
                "Prepare promotion dossier"
            ]
        elif current_level == ResearcherLevel.ASSOCIATE_PROF:
            return [
                "Demonstrate research leadership",
                "Secure major grants",
                "Mentor junior faculty",
                "Lead major service initiatives",
                "Achieve international recognition",
                "Prepare full professor dossier"
            ]
        else:
            return ["Complete current position requirements"]
    
    def track_promotion_timeline(self, researcher_id: str) -> Dict[str, Any]:
        """
        Track promotion timeline and milestone progress.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Dictionary with timeline tracking information
        """
        timeline = self.promotion_timelines.get(researcher_id)
        if not timeline:
            return {'error': 'No promotion timeline found'}
        
        years_in_rank = timeline.get_years_in_rank()
        is_eligible = timeline.is_eligible_for_promotion()
        
        completed_count = len(timeline.milestones_completed)
        total_count = completed_count + len(timeline.milestones_pending)
        
        return {
            'current_rank': timeline.current_rank.value,
            'target_rank': timeline.target_rank.value,
            'years_in_rank': years_in_rank,
            'is_eligible': is_eligible,
            'earliest_eligible_date': timeline.earliest_eligible_date.isoformat(),
            'recommended_application_date': timeline.recommended_application_date.isoformat(),
            'milestones_completed': completed_count,
            'milestones_total': total_count,
            'milestone_completion_rate': (completed_count / total_count * 100) if total_count > 0 else 0,
            'completed_milestones': timeline.milestones_completed,
            'pending_milestones': timeline.milestones_pending
        }
    
    def complete_milestone(self, researcher_id: str, milestone: str) -> bool:
        """
        Mark a promotion milestone as completed.
        
        Args:
            researcher_id: ID of the researcher
            milestone: Milestone description to complete
            
        Returns:
            True if milestone was completed, False otherwise
        """
        timeline = self.promotion_timelines.get(researcher_id)
        if not timeline:
            return False
        
        if milestone in timeline.milestones_pending:
            timeline.milestones_pending.remove(milestone)
            timeline.milestones_completed.append(milestone)
            logger.info(f"Completed milestone for researcher {researcher_id}: {milestone}")
            return True
        
        return False
    
    def set_custom_requirements(self, field: str, promotion_type: PromotionType,
                              requirements: PromotionRequirements):
        """
        Set custom promotion requirements for a field and promotion type.
        
        Args:
            field: Academic field name
            promotion_type: Type of promotion
            requirements: PromotionRequirements object
        """
        if field not in self.promotion_requirements:
            self.promotion_requirements[field] = {}
        
        self.promotion_requirements[field][promotion_type] = requirements
        logger.info(f"Set custom promotion requirements for {field}: {promotion_type.value}")
    
    def get_promotion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about promotion evaluations and timelines.
        
        Returns:
            Dictionary with promotion statistics
        """
        if not self.evaluations:
            return {'total_evaluations': 0}
        
        all_evaluations = []
        for eval_list in self.evaluations.values():
            all_evaluations.extend(eval_list)
        
        # Outcome distribution
        outcome_counts = {}
        for outcome in PromotionOutcome:
            outcome_counts[outcome.value] = len([e for e in all_evaluations 
                                               if e.recommendation == outcome])
        
        # Average scores by component
        avg_research = sum(e.research_score for e in all_evaluations) / len(all_evaluations)
        avg_teaching = sum(e.teaching_score for e in all_evaluations) / len(all_evaluations)
        avg_service = sum(e.service_score for e in all_evaluations) / len(all_evaluations)
        avg_overall = sum(e.overall_score for e in all_evaluations) / len(all_evaluations)
        
        # Timeline statistics
        timeline_stats = {}
        if self.promotion_timelines:
            timelines = list(self.promotion_timelines.values())
            avg_years_in_rank = sum(t.get_years_in_rank() for t in timelines) / len(timelines)
            eligible_count = len([t for t in timelines if t.is_eligible_for_promotion()])
            
            timeline_stats = {
                'total_timelines': len(timelines),
                'average_years_in_rank': avg_years_in_rank,
                'eligible_for_promotion': eligible_count,
                'eligibility_rate': eligible_count / len(timelines) * 100
            }
        
        return {
            'total_evaluations': len(all_evaluations),
            'outcome_distribution': outcome_counts,
            'average_scores': {
                'research': avg_research,
                'teaching': avg_teaching,
                'service': avg_service,
                'overall': avg_overall
            },
            'timeline_statistics': timeline_stats,
            'approval_rate': outcome_counts.get('Approved', 0) / len(all_evaluations) * 100 if all_evaluations else 0
        }
    
    def export_evaluation_data(self, researcher_id: str) -> Dict[str, Any]:
        """
        Export promotion evaluation data for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Dictionary with exportable evaluation data
        """
        evaluations = self.evaluations.get(researcher_id, [])
        timeline = self.promotion_timelines.get(researcher_id)
        
        evaluation_data = []
        for eval in evaluations:
            evaluation_data.append({
                'evaluation_date': eval.evaluation_date.isoformat(),
                'promotion_type': eval.promotion_type.value,
                'research_score': eval.research_score,
                'teaching_score': eval.teaching_score,
                'service_score': eval.service_score,
                'overall_score': eval.overall_score,
                'recommendation': eval.recommendation.value,
                'meets_minimum_requirements': eval.meets_minimum_requirements,
                'strengths': eval.strengths,
                'weaknesses': eval.weaknesses,
                'improvement_areas': eval.improvement_areas,
                'timeline_assessment': eval.timeline_assessment
            })
        
        timeline_data = {}
        if timeline:
            timeline_data = {
                'current_rank': timeline.current_rank.value,
                'target_rank': timeline.target_rank.value,
                'rank_start_date': timeline.rank_start_date.isoformat(),
                'years_in_rank': timeline.get_years_in_rank(),
                'is_eligible': timeline.is_eligible_for_promotion(),
                'milestones_completed': timeline.milestones_completed,
                'milestones_pending': timeline.milestones_pending
            }
        
        return {
            'researcher_id': researcher_id,
            'evaluations': evaluation_data,
            'timeline': timeline_data,
            'export_date': datetime.now().isoformat()
        }