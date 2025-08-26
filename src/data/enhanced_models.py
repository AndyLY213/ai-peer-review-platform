"""
Enhanced Data Model Classes

This module provides enhanced data model classes for researchers, reviews, and venues
with validation and serialization methods for new data structures, database migration
utilities, and comprehensive data management capabilities.
"""

import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

from src.core.exceptions import ValidationError, DatabaseError
from src.core.logging_config import get_logger


logger = get_logger(__name__)


class ResearcherLevel(Enum):
    """Academic hierarchy levels."""
    GRADUATE_STUDENT = "Graduate Student"
    POSTDOC = "Postdoc"
    ASSISTANT_PROF = "Assistant Prof"
    ASSOCIATE_PROF = "Associate Prof"
    FULL_PROF = "Full Prof"
    EMERITUS = "Emeritus"


class VenueType(Enum):
    """Types of publication venues."""
    TOP_CONFERENCE = "Top Conference"
    MID_CONFERENCE = "Mid Conference"
    LOW_CONFERENCE = "Low Conference"
    TOP_JOURNAL = "Top Journal"
    SPECIALIZED_JOURNAL = "Specialized Journal"
    GENERAL_JOURNAL = "General Journal"
    WORKSHOP = "Workshop"
    PREPRINT = "Preprint"


class ReviewDecision(Enum):
    """Review recommendation categories."""
    ACCEPT = "Accept"
    MINOR_REVISION = "Minor Revision"
    MAJOR_REVISION = "Major Revision"
    REJECT = "Reject"


class CareerStage(Enum):
    """Career progression stages."""
    EARLY_CAREER = "Early Career"
    MID_CAREER = "Mid Career"
    SENIOR_CAREER = "Senior Career"
    EMERITUS_CAREER = "Emeritus Career"


class FundingStatus(Enum):
    """Funding status categories."""
    WELL_FUNDED = "Well Funded"
    ADEQUATELY_FUNDED = "Adequately Funded"
    UNDER_FUNDED = "Under Funded"
    NO_FUNDING = "No Funding"


@dataclass
class EnhancedReviewCriteria:
    """Enhanced review criteria with six mandatory scoring dimensions (1-10 scale)."""
    novelty: float = 5.0  # 1-10 scale
    technical_quality: float = 5.0  # 1-10 scale
    clarity: float = 5.0  # 1-10 scale
    significance: float = 5.0  # 1-10 scale
    reproducibility: float = 5.0  # 1-10 scale
    related_work: float = 5.0  # 1-10 scale
    
    def __post_init__(self):
        """Validate score ranges."""
        for field_name in ['novelty', 'technical_quality', 'clarity', 'significance', 'reproducibility', 'related_work']:
            value = getattr(self, field_name)
            if not (1.0 <= value <= 10.0):
                raise ValidationError(field_name, value, "score between 1.0 and 10.0")
    
    def get_average_score(self) -> float:
        """Calculate average score across all dimensions."""
        return (self.novelty + self.technical_quality + self.clarity + 
                self.significance + self.reproducibility + self.related_work) / 6.0
    
    def get_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted average score."""
        if weights is None:
            weights = {
                'novelty': 1.0,
                'technical_quality': 1.5,  # Higher weight for technical quality
                'clarity': 1.0,
                'significance': 1.5,  # Higher weight for significance
                'reproducibility': 1.0,
                'related_work': 0.8  # Lower weight for related work
            }
        
        total_weighted = sum(getattr(self, dim) * weights.get(dim, 1.0) 
                           for dim in ['novelty', 'technical_quality', 'clarity', 
                                     'significance', 'reproducibility', 'related_work'])
        total_weights = sum(weights.values())
        
        return total_weighted / total_weights if total_weights > 0 else self.get_average_score()


@dataclass
class DetailedStrength:
    """Detailed strength in a review."""
    category: str  # e.g., "Technical", "Methodological", "Presentation"
    description: str
    importance: int = 3  # 1-5 scale


@dataclass
class DetailedWeakness:
    """Detailed weakness in a review."""
    category: str  # e.g., "Technical", "Methodological", "Presentation"
    description: str
    severity: int = 3  # 1-5 scale
    suggestions: List[str] = field(default_factory=list)


@dataclass
class BiasEffect:
    """Represents the effect of a cognitive bias on a review."""
    bias_type: str  # e.g., "confirmation", "halo", "anchoring", "availability"
    strength: float  # 0-1 scale
    score_adjustment: float  # How much the bias adjusted the score
    description: str = ""


@dataclass
class StructuredReview:
    """Enhanced structured review with comprehensive sections and bias tracking."""
    # Basic identification
    reviewer_id: str
    paper_id: str
    venue_id: str
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Enhanced scoring
    criteria_scores: EnhancedReviewCriteria = field(default_factory=EnhancedReviewCriteria)
    confidence_level: int = 3  # 1-5 scale
    recommendation: ReviewDecision = ReviewDecision.MAJOR_REVISION
    
    # Structured content sections
    executive_summary: str = ""
    detailed_strengths: List[DetailedStrength] = field(default_factory=list)
    detailed_weaknesses: List[DetailedWeakness] = field(default_factory=list)
    technical_comments: str = ""
    presentation_comments: str = ""
    questions_for_authors: List[str] = field(default_factory=list)
    suggestions_for_improvement: List[str] = field(default_factory=list)
    
    # Quality and completeness metrics
    review_length: int = 0
    time_spent_minutes: int = 0
    quality_score: float = 0.0  # Calculated based on completeness and detail
    completeness_score: float = 0.0  # How complete the review is
    
    # Bias tracking
    applied_biases: List[BiasEffect] = field(default_factory=list)
    bias_adjusted_scores: Dict[str, float] = field(default_factory=dict)
    
    # Temporal information
    submission_timestamp: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    is_late: bool = False
    revision_round: int = 1
    
    def __post_init__(self):
        """Validate and calculate derived fields."""
        self._validate_confidence_level()
        self._calculate_review_length()
        self._calculate_quality_scores()
    
    def _validate_confidence_level(self):
        """Validate confidence level is in valid range."""
        if not (1 <= self.confidence_level <= 5):
            raise ValidationError("confidence_level", self.confidence_level, "integer between 1 and 5")
    
    def _calculate_review_length(self):
        """Calculate total review length in characters."""
        total_length = len(self.executive_summary) + len(self.technical_comments) + len(self.presentation_comments)
        total_length += sum(len(s.description) for s in self.detailed_strengths)
        total_length += sum(len(w.description) for w in self.detailed_weaknesses)
        total_length += sum(len(q) for q in self.questions_for_authors)
        total_length += sum(len(s) for s in self.suggestions_for_improvement)
        self.review_length = total_length
    
    def _calculate_quality_scores(self):
        """Calculate quality and completeness scores."""
        # Completeness score based on filled sections
        completeness_factors = [
            1.0 if self.executive_summary else 0.0,
            1.0 if self.detailed_strengths else 0.0,
            1.0 if self.detailed_weaknesses else 0.0,
            1.0 if self.technical_comments else 0.0,
            0.5 if self.presentation_comments else 0.0,
            0.5 if self.questions_for_authors else 0.0,
            0.5 if self.suggestions_for_improvement else 0.0
        ]
        self.completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Quality score based on length, detail, and completeness
        length_factor = min(1.0, self.review_length / 500.0)  # Normalize to 500 chars
        detail_factor = (len(self.detailed_strengths) + len(self.detailed_weaknesses)) / 5.0
        detail_factor = min(1.0, detail_factor)
        
        self.quality_score = (self.completeness_score * 0.4 + length_factor * 0.3 + detail_factor * 0.3)
    
    def meets_venue_requirements(self, min_word_count: int) -> bool:
        """Check if review meets venue-specific requirements."""
        word_count = len(self.executive_summary.split()) + len(self.technical_comments.split()) + \
                    len(self.presentation_comments.split())
        
        return (word_count >= min_word_count and 
                len(self.detailed_strengths) >= 2 and 
                len(self.detailed_weaknesses) >= 1 and
                self.executive_summary.strip() != "")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data['recommendation'] = self.recommendation.value
        data['submission_timestamp'] = self.submission_timestamp.isoformat()
        if self.deadline:
            data['deadline'] = self.deadline.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredReview':
        """Create from dictionary."""
        # Convert string enums back
        if 'recommendation' in data:
            data['recommendation'] = ReviewDecision(data['recommendation'])
        
        # Convert timestamps
        if 'submission_timestamp' in data:
            data['submission_timestamp'] = datetime.fromisoformat(data['submission_timestamp'])
        if 'deadline' in data and data['deadline']:
            data['deadline'] = datetime.fromisoformat(data['deadline'])
        
        # Handle nested objects
        if 'criteria_scores' in data and isinstance(data['criteria_scores'], dict):
            data['criteria_scores'] = EnhancedReviewCriteria(**data['criteria_scores'])
        
        if 'detailed_strengths' in data:
            data['detailed_strengths'] = [DetailedStrength(**s) if isinstance(s, dict) else s 
                                        for s in data['detailed_strengths']]
        
        if 'detailed_weaknesses' in data:
            data['detailed_weaknesses'] = [DetailedWeakness(**w) if isinstance(w, dict) else w 
                                         for w in data['detailed_weaknesses']]
        
        if 'applied_biases' in data:
            data['applied_biases'] = [BiasEffect(**b) if isinstance(b, dict) else b 
                                    for b in data['applied_biases']]
        
        return cls(**data)


@dataclass
class ReviewBehaviorProfile:
    """Profile of reviewer behavior patterns."""
    avg_review_length: int = 500
    review_thoroughness: float = 0.7  # 0-1 scale
    bias_susceptibility: Dict[str, float] = field(default_factory=lambda: {
        'confirmation': 0.3,
        'halo': 0.2,
        'anchoring': 0.4,
        'availability': 0.3
    })
    review_speed: float = 0.5  # 0-1 scale (higher = faster)
    consistency: float = 0.8  # How consistent across dimensions


@dataclass
class StrategicBehaviorProfile:
    """Profile of strategic behavior patterns."""
    venue_shopping_tendency: float = 0.3  # 0-1 scale
    review_trading_willingness: float = 0.1  # 0-1 scale
    citation_cartel_participation: float = 0.05  # 0-1 scale
    salami_slicing_tendency: float = 0.2  # 0-1 scale


@dataclass
class CareerMilestone:
    """Represents a career milestone."""
    milestone_type: str  # e.g., "tenure", "promotion", "award"
    date_achieved: date
    description: str
    impact_on_behavior: Dict[str, float] = field(default_factory=dict)


@dataclass
class PublicationRecord:
    """Record of a publication."""
    paper_id: str
    title: str
    venue: str
    year: int
    citations: int = 0
    h_index_contribution: float = 0.0


@dataclass
class ReviewQualityMetric:
    """Metric for tracking review quality over time."""
    review_id: str
    quality_score: float
    timeliness_score: float  # Based on submission time vs deadline
    helpfulness_rating: Optional[float] = None  # From authors if available
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TenureTimeline:
    """Timeline for tenure track progression."""
    start_date: date
    tenure_decision_date: date
    current_progress: float = 0.0  # 0-1 scale
    publication_requirements: Dict[str, int] = field(default_factory=dict)
    current_publications: Dict[str, int] = field(default_factory=dict)


@dataclass
class EnhancedResearcher:
    """Enhanced researcher model with comprehensive academic profile."""
    # Basic identification
    id: str
    name: str
    specialty: str
    email: str = ""
    
    # Academic hierarchy and reputation
    level: ResearcherLevel = ResearcherLevel.ASSISTANT_PROF
    institution_name: str = ""
    institution_tier: int = 2  # 1-3 (1 = top tier)
    h_index: int = 10
    total_citations: int = 100
    years_active: int = 5
    reputation_score: float = 0.0  # Calculated field
    
    # Behavioral profiles
    cognitive_biases: Dict[str, float] = field(default_factory=lambda: {
        'confirmation': 0.3,
        'halo': 0.2,
        'anchoring': 0.4,
        'availability': 0.3
    })
    review_behavior: ReviewBehaviorProfile = field(default_factory=ReviewBehaviorProfile)
    strategic_behavior: StrategicBehaviorProfile = field(default_factory=StrategicBehaviorProfile)
    
    # Career and funding
    career_stage: CareerStage = CareerStage.EARLY_CAREER
    funding_status: FundingStatus = FundingStatus.ADEQUATELY_FUNDED
    publication_pressure: float = 0.5  # 0-1 scale
    tenure_timeline: Optional[TenureTimeline] = None
    
    # Networks and relationships
    collaboration_network: Set[str] = field(default_factory=set)
    citation_network: Set[str] = field(default_factory=set)
    institutional_affiliations: List[str] = field(default_factory=list)
    advisor_relationships: List[str] = field(default_factory=list)
    
    # Performance tracking
    review_quality_history: List[ReviewQualityMetric] = field(default_factory=list)
    publication_history: List[PublicationRecord] = field(default_factory=list)
    career_milestones: List[CareerMilestone] = field(default_factory=list)
    
    # Workload and availability
    current_review_load: int = 0
    max_reviews_per_month: int = 4
    availability_status: bool = True
    
    def __post_init__(self):
        """Calculate derived fields and validate data."""
        self._calculate_reputation_score()
        self._determine_career_stage()
        self._set_max_reviews_per_month()
        self._validate_institution_tier()
    
    def _calculate_reputation_score(self):
        """Calculate reputation score based on multiple factors."""
        # Base score from h-index (normalized)
        h_index_score = min(1.0, self.h_index / 50.0)
        
        # Citations score (normalized)
        citation_score = min(1.0, self.total_citations / 1000.0)
        
        # Experience score
        experience_score = min(1.0, self.years_active / 20.0)
        
        # Institution tier bonus (higher tier = lower number = higher bonus)
        institution_bonus = (4 - self.institution_tier) / 3.0
        
        # Level multiplier
        level_multipliers = {
            ResearcherLevel.GRADUATE_STUDENT: 0.3,
            ResearcherLevel.POSTDOC: 0.5,
            ResearcherLevel.ASSISTANT_PROF: 0.7,
            ResearcherLevel.ASSOCIATE_PROF: 0.9,
            ResearcherLevel.FULL_PROF: 1.0,
            ResearcherLevel.EMERITUS: 1.2
        }
        
        level_multiplier = level_multipliers.get(self.level, 0.7)
        
        # Combine all factors
        self.reputation_score = (h_index_score * 0.3 + citation_score * 0.3 + 
                               experience_score * 0.2 + institution_bonus * 0.2) * level_multiplier
    
    def _determine_career_stage(self):
        """Determine career stage based on level and years active."""
        if self.level in [ResearcherLevel.GRADUATE_STUDENT, ResearcherLevel.POSTDOC]:
            self.career_stage = CareerStage.EARLY_CAREER
        elif self.level == ResearcherLevel.ASSISTANT_PROF or self.years_active < 10:
            self.career_stage = CareerStage.EARLY_CAREER
        elif self.level == ResearcherLevel.ASSOCIATE_PROF or self.years_active < 20:
            self.career_stage = CareerStage.MID_CAREER
        elif self.level == ResearcherLevel.FULL_PROF:
            self.career_stage = CareerStage.SENIOR_CAREER
        else:  # EMERITUS
            self.career_stage = CareerStage.EMERITUS_CAREER
    
    def _set_max_reviews_per_month(self):
        """Set maximum reviews per month based on seniority level."""
        level_limits = {
            ResearcherLevel.GRADUATE_STUDENT: 2,
            ResearcherLevel.POSTDOC: 3,
            ResearcherLevel.ASSISTANT_PROF: 4,
            ResearcherLevel.ASSOCIATE_PROF: 6,
            ResearcherLevel.FULL_PROF: 8,
            ResearcherLevel.EMERITUS: 3
        }
        self.max_reviews_per_month = level_limits.get(self.level, 4)
    
    def _validate_institution_tier(self):
        """Validate institution tier is in valid range."""
        if not (1 <= self.institution_tier <= 3):
            raise ValidationError("institution_tier", self.institution_tier, "integer between 1 and 3")
    
    def get_reputation_multiplier(self) -> float:
        """Get reputation multiplier for review influence."""
        base_multipliers = {
            ResearcherLevel.GRADUATE_STUDENT: 0.5,
            ResearcherLevel.POSTDOC: 0.7,
            ResearcherLevel.ASSISTANT_PROF: 1.0,  # Baseline
            ResearcherLevel.ASSOCIATE_PROF: 1.3,
            ResearcherLevel.FULL_PROF: 1.5,
            ResearcherLevel.EMERITUS: 1.2
        }
        
        base_multiplier = base_multipliers.get(self.level, 1.0)
        reputation_bonus = self.reputation_score * 0.5  # Up to 50% bonus
        
        return base_multiplier + reputation_bonus
    
    def can_accept_review(self) -> bool:
        """Check if researcher can accept another review."""
        return (self.availability_status and 
                self.current_review_load < self.max_reviews_per_month)
    
    def add_collaboration(self, researcher_id: str):
        """Add a collaboration relationship."""
        self.collaboration_network.add(researcher_id)
    
    def add_citation_relationship(self, researcher_id: str):
        """Add a citation relationship."""
        self.citation_network.add(researcher_id)
    
    def update_publication_history(self, publication: PublicationRecord):
        """Add a publication to history and update metrics."""
        self.publication_history.append(publication)
        
        # Update h-index (simplified calculation)
        citation_counts = sorted([p.citations for p in self.publication_history], reverse=True)
        h_index = 0
        for i, citations in enumerate(citation_counts, 1):
            if citations >= i:
                h_index = i
            else:
                break
        self.h_index = h_index
        
        # Update total citations
        self.total_citations = sum(p.citations for p in self.publication_history)
        
        # Recalculate reputation score
        self._calculate_reputation_score()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        
        # Convert enums to strings
        data['level'] = self.level.value
        data['career_stage'] = self.career_stage.value
        data['funding_status'] = self.funding_status.value
        
        # Convert sets to lists
        data['collaboration_network'] = list(self.collaboration_network)
        data['citation_network'] = list(self.citation_network)
        
        # Handle nested objects
        if self.tenure_timeline:
            data['tenure_timeline']['start_date'] = self.tenure_timeline.start_date.isoformat()
            data['tenure_timeline']['tenure_decision_date'] = self.tenure_timeline.tenure_decision_date.isoformat()
        
        # Convert dates in career milestones
        for milestone in data['career_milestones']:
            if isinstance(milestone['date_achieved'], date):
                milestone['date_achieved'] = milestone['date_achieved'].isoformat()
        
        # Convert timestamps in review quality history
        for metric in data['review_quality_history']:
            if isinstance(metric['timestamp'], datetime):
                metric['timestamp'] = metric['timestamp'].isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedResearcher':
        """Create from dictionary."""
        # Convert string enums back
        if 'level' in data:
            data['level'] = ResearcherLevel(data['level'])
        if 'career_stage' in data:
            data['career_stage'] = CareerStage(data['career_stage'])
        if 'funding_status' in data:
            data['funding_status'] = FundingStatus(data['funding_status'])
        
        # Convert lists to sets
        if 'collaboration_network' in data:
            data['collaboration_network'] = set(data['collaboration_network'])
        if 'citation_network' in data:
            data['citation_network'] = set(data['citation_network'])
        
        # Handle nested objects
        if 'review_behavior' in data and isinstance(data['review_behavior'], dict):
            data['review_behavior'] = ReviewBehaviorProfile(**data['review_behavior'])
        
        if 'strategic_behavior' in data and isinstance(data['strategic_behavior'], dict):
            data['strategic_behavior'] = StrategicBehaviorProfile(**data['strategic_behavior'])
        
        if 'tenure_timeline' in data and data['tenure_timeline']:
            timeline_data = data['tenure_timeline']
            if 'start_date' in timeline_data:
                timeline_data['start_date'] = date.fromisoformat(timeline_data['start_date'])
            if 'tenure_decision_date' in timeline_data:
                timeline_data['tenure_decision_date'] = date.fromisoformat(timeline_data['tenure_decision_date'])
            data['tenure_timeline'] = TenureTimeline(**timeline_data)
        
        # Convert dates and timestamps
        if 'career_milestones' in data:
            for milestone in data['career_milestones']:
                if isinstance(milestone, dict) and 'date_achieved' in milestone:
                    milestone['date_achieved'] = date.fromisoformat(milestone['date_achieved'])
            data['career_milestones'] = [CareerMilestone(**m) if isinstance(m, dict) else m 
                                       for m in data['career_milestones']]
        
        if 'review_quality_history' in data:
            for metric in data['review_quality_history']:
                if isinstance(metric, dict) and 'timestamp' in metric:
                    metric['timestamp'] = datetime.fromisoformat(metric['timestamp'])
            data['review_quality_history'] = [ReviewQualityMetric(**m) if isinstance(m, dict) else m 
                                            for m in data['review_quality_history']]
        
        if 'publication_history' in data:
            data['publication_history'] = [PublicationRecord(**p) if isinstance(p, dict) else p 
                                         for p in data['publication_history']]
        
        return cls(**data)


@dataclass
class ReviewRequirements:
    """Requirements for reviews at a specific venue."""
    min_word_count: int = 300
    max_word_count: int = 1000
    required_sections: List[str] = field(default_factory=lambda: [
        "summary", "strengths", "weaknesses", "detailed_comments"
    ])
    min_strengths: int = 2
    min_weaknesses: int = 1
    requires_questions: bool = False
    requires_suggestions: bool = False


@dataclass
class QualityStandards:
    """Quality standards for a venue."""
    min_technical_depth: float = 5.0  # 1-10 scale
    min_novelty_threshold: float = 4.0  # 1-10 scale
    min_significance_threshold: float = 5.0  # 1-10 scale
    acceptance_threshold: float = 6.0  # Overall score threshold
    requires_reproducibility: bool = True


@dataclass
class ReviewerCriteria:
    """Criteria for reviewer selection at a venue."""
    min_h_index: int = 5
    min_years_experience: int = 2
    preferred_institution_tiers: List[int] = field(default_factory=lambda: [1, 2, 3])
    min_reputation_score: float = 0.3
    max_reviews_per_reviewer: int = 3


@dataclass
class EnhancedVenue:
    """Enhanced venue model with realistic characteristics and standards."""
    # Basic identification
    id: str
    name: str
    venue_type: VenueType
    field: str  # Research field (AI, NLP, Vision, etc.)
    
    # Venue characteristics
    acceptance_rate: float = 0.25  # Default 25%
    prestige_score: int = 5  # 1-10 scale
    impact_factor: Optional[float] = None
    
    # Review standards and requirements
    review_requirements: ReviewRequirements = field(default_factory=ReviewRequirements)
    quality_standards: QualityStandards = field(default_factory=QualityStandards)
    reviewer_selection_criteria: ReviewerCriteria = field(default_factory=ReviewerCriteria)
    
    # Temporal constraints
    review_deadline_weeks: int = 4  # Weeks for review
    revision_cycles_allowed: int = 1
    
    # Historical data and statistics
    submission_history: List[Dict[str, Any]] = field(default_factory=list)
    acceptance_trends: List[Dict[str, Any]] = field(default_factory=list)
    reviewer_pool: Set[str] = field(default_factory=set)
    
    # Calibration data from PeerRead
    score_distributions: Dict[str, List[float]] = field(default_factory=dict)
    review_length_stats: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate venue data and set defaults based on venue type."""
        self._validate_acceptance_rate()
        self._set_type_specific_defaults()
    
    def _validate_acceptance_rate(self):
        """Validate acceptance rate is in valid range."""
        if not (0.0 <= self.acceptance_rate <= 1.0):
            raise ValidationError("acceptance_rate", self.acceptance_rate, "float between 0.0 and 1.0")
    
    def _set_type_specific_defaults(self):
        """Set defaults based on venue type."""
        # Skip defaults if loading from dict or if _from_dict flag is set
        if hasattr(self, '_from_dict') and self._from_dict:
            return
        
        type_defaults = {
            VenueType.TOP_CONFERENCE: {
                'acceptance_rate': 0.05,
                'prestige_score': 9,
                'min_h_index': 15,
                'acceptance_threshold': 8.5,
                'review_deadline_weeks': 6
            },
            VenueType.MID_CONFERENCE: {
                'acceptance_rate': 0.25,
                'prestige_score': 6,
                'min_h_index': 8,
                'acceptance_threshold': 6.5,
                'review_deadline_weeks': 4
            },
            VenueType.LOW_CONFERENCE: {
                'acceptance_rate': 0.50,
                'prestige_score': 4,
                'min_h_index': 3,
                'acceptance_threshold': 5.0,
                'review_deadline_weeks': 3
            },
            VenueType.TOP_JOURNAL: {
                'acceptance_rate': 0.02,
                'prestige_score': 10,
                'min_h_index': 20,
                'acceptance_threshold': 9.0,
                'review_deadline_weeks': 8
            },
            VenueType.SPECIALIZED_JOURNAL: {
                'acceptance_rate': 0.15,
                'prestige_score': 7,
                'min_h_index': 10,
                'acceptance_threshold': 7.0,
                'review_deadline_weeks': 6
            },
            VenueType.GENERAL_JOURNAL: {
                'acceptance_rate': 0.40,
                'prestige_score': 5,
                'min_h_index': 5,
                'acceptance_threshold': 6.0,
                'review_deadline_weeks': 4
            }
        }
        
        defaults = type_defaults.get(self.venue_type, {})
        
        # Apply defaults if not already set
        for key, value in defaults.items():
            if key == 'acceptance_rate' and self.acceptance_rate == 0.25:  # Default value
                self.acceptance_rate = value
            elif key == 'prestige_score' and self.prestige_score == 5:  # Default value
                self.prestige_score = value
            elif key == 'min_h_index':
                self.reviewer_selection_criteria.min_h_index = value
            elif key == 'acceptance_threshold':
                self.quality_standards.acceptance_threshold = value
            elif key == 'review_deadline_weeks' and self.review_deadline_weeks == 4:  # Default value
                self.review_deadline_weeks = value
    
    def meets_reviewer_criteria(self, researcher: EnhancedResearcher) -> bool:
        """Check if researcher meets reviewer criteria for this venue."""
        criteria = self.reviewer_selection_criteria
        
        # Check h-index requirement
        if researcher.h_index < criteria.min_h_index:
            return False
        
        # Check years of experience
        if researcher.years_active < criteria.min_years_experience:
            return False
        
        # Check institution tier preference
        if researcher.institution_tier not in criteria.preferred_institution_tiers:
            return False
        
        # Check reputation score
        if researcher.reputation_score < criteria.min_reputation_score:
            return False
        
        return True
        
        return (researcher.h_index >= criteria.min_h_index and
                researcher.years_active >= criteria.min_years_experience and
                researcher.institution_tier in criteria.preferred_institution_tiers and
                researcher.reputation_score >= criteria.min_reputation_score)
    
    def calculate_acceptance_probability(self, review_scores: List[float]) -> float:
        """Calculate acceptance probability based on review scores."""
        if not review_scores:
            return 0.0
        
        avg_score = sum(review_scores) / len(review_scores)
        threshold = self.quality_standards.acceptance_threshold
        
        # Sigmoid function for smooth probability transition
        import math
        probability = 1.0 / (1.0 + math.exp(-(avg_score - threshold)))
        
        return probability
    
    def add_submission_record(self, paper_id: str, accepted: bool, scores: List[float]):
        """Add a submission record to history."""
        record = {
            'paper_id': paper_id,
            'accepted': accepted,
            'scores': scores,
            'timestamp': datetime.now().isoformat()
        }
        self.submission_history.append(record)
        
        # Update acceptance trends
        current_year = datetime.now().year
        year_records = [r for r in self.submission_history 
                       if datetime.fromisoformat(r['timestamp']).year == current_year]
        
        if year_records:
            year_acceptance_rate = sum(1 for r in year_records if r['accepted']) / len(year_records)
            
            # Update or add trend for current year
            existing_trend = next((t for t in self.acceptance_trends if t['year'] == current_year), None)
            if existing_trend:
                existing_trend['acceptance_rate'] = year_acceptance_rate
                existing_trend['submissions'] = len(year_records)
            else:
                self.acceptance_trends.append({
                    'year': current_year,
                    'acceptance_rate': year_acceptance_rate,
                    'submissions': len(year_records)
                })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        
        # Convert enums to strings
        data['venue_type'] = self.venue_type.value
        
        # Convert sets to lists
        data['reviewer_pool'] = list(self.reviewer_pool)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedVenue':
        """Create from dictionary."""
        # Convert string enums back
        if 'venue_type' in data:
            data['venue_type'] = VenueType(data['venue_type'])
        
        # Convert lists to sets
        if 'reviewer_pool' in data:
            data['reviewer_pool'] = set(data['reviewer_pool'])
        
        # Handle nested objects
        if 'review_requirements' in data and isinstance(data['review_requirements'], dict):
            data['review_requirements'] = ReviewRequirements(**data['review_requirements'])
        
        if 'quality_standards' in data and isinstance(data['quality_standards'], dict):
            data['quality_standards'] = QualityStandards(**data['quality_standards'])
        
        if 'reviewer_selection_criteria' in data and isinstance(data['reviewer_selection_criteria'], dict):
            data['reviewer_selection_criteria'] = ReviewerCriteria(**data['reviewer_selection_criteria'])
        
        # Create instance using __new__ to avoid __init__ and __post_init__
        instance = cls.__new__(cls)
        
        # Set _from_dict flag before setting attributes
        instance._from_dict = True
        
        # Set all attributes directly
        for key, value in data.items():
            setattr(instance, key, value)
        
        # Only validate, don't set defaults
        instance._validate_acceptance_rate()
        
        return instance


class DatabaseMigrationUtility:
    """Utility for migrating existing data to enhanced models."""
    
    def __init__(self, data_directory: str = "peer_review_workspace"):
        """Initialize migration utility."""
        self.data_directory = Path(data_directory)
        self.backup_directory = self.data_directory / "backups"
        self.backup_directory.mkdir(exist_ok=True)
        
    def migrate_researchers(self, old_researchers_file: str, new_researchers_file: str):
        """Migrate researchers from old format to enhanced format."""
        logger.info(f"Migrating researchers from {old_researchers_file} to {new_researchers_file}")
        
        # Create backup
        backup_file = self.backup_directory / f"researchers_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if Path(old_researchers_file).exists():
            import shutil
            shutil.copy2(old_researchers_file, backup_file)
            logger.info(f"Backup created: {backup_file}")
        
        # Load old data
        old_researchers = {}
        if Path(old_researchers_file).exists():
            with open(old_researchers_file, 'r') as f:
                old_data = json.load(f)
                old_researchers = old_data.get('researchers', {})
        
        # Convert to enhanced format
        enhanced_researchers = {}
        for researcher_id, old_data in old_researchers.items():
            try:
                enhanced = self._convert_researcher_to_enhanced(researcher_id, old_data)
                enhanced_researchers[researcher_id] = enhanced.to_dict()
            except Exception as e:
                logger.error(f"Failed to migrate researcher {researcher_id}: {e}")
                continue
        
        # Save enhanced data
        new_data = {
            'researchers': enhanced_researchers,
            'migration_timestamp': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        with open(new_researchers_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        logger.info(f"Successfully migrated {len(enhanced_researchers)} researchers")
    
    def _convert_researcher_to_enhanced(self, researcher_id: str, old_data: Dict[str, Any]) -> EnhancedResearcher:
        """Convert old researcher format to enhanced format."""
        # Map old fields to new fields
        enhanced_data = {
            'id': researcher_id,
            'name': old_data.get('name', f'Researcher_{researcher_id}'),
            'specialty': old_data.get('specialty', 'General'),
            'email': old_data.get('email', ''),
        }
        
        # Map level if available
        old_level = old_data.get('level', 'Assistant Prof')
        level_mapping = {
            'Graduate Student': ResearcherLevel.GRADUATE_STUDENT,
            'Postdoc': ResearcherLevel.POSTDOC,
            'Assistant Prof': ResearcherLevel.ASSISTANT_PROF,
            'Associate Prof': ResearcherLevel.ASSOCIATE_PROF,
            'Full Prof': ResearcherLevel.FULL_PROF,
            'Emeritus': ResearcherLevel.EMERITUS
        }
        enhanced_data['level'] = level_mapping.get(old_level, ResearcherLevel.ASSISTANT_PROF)
        
        # Set defaults for new fields
        enhanced_data['h_index'] = old_data.get('h_index', 10)
        enhanced_data['total_citations'] = old_data.get('citations', 100)
        enhanced_data['years_active'] = old_data.get('years_active', 5)
        enhanced_data['institution_tier'] = old_data.get('institution_tier', 2)
        
        return EnhancedResearcher(**enhanced_data)
    
    def migrate_papers(self, old_papers_file: str, new_papers_file: str):
        """Migrate papers to include enhanced review structures."""
        logger.info(f"Migrating papers from {old_papers_file} to {new_papers_file}")
        
        # Create backup
        backup_file = self.backup_directory / f"papers_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if Path(old_papers_file).exists():
            import shutil
            shutil.copy2(old_papers_file, backup_file)
            logger.info(f"Backup created: {backup_file}")
        
        # Load and migrate papers
        old_papers = {}
        if Path(old_papers_file).exists():
            with open(old_papers_file, 'r') as f:
                old_data = json.load(f)
                old_papers = old_data.get('papers', {})
        
        # Convert reviews to enhanced format
        enhanced_papers = {}
        for paper_id, paper_data in old_papers.items():
            enhanced_paper = paper_data.copy()
            
            # Convert reviews
            old_reviews = paper_data.get('reviews', [])
            enhanced_reviews = []
            
            for old_review in old_reviews:
                try:
                    enhanced_review = self._convert_review_to_enhanced(paper_id, old_review)
                    enhanced_reviews.append(enhanced_review.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to migrate review for paper {paper_id}: {e}")
                    continue
            
            enhanced_paper['reviews'] = enhanced_reviews
            enhanced_papers[paper_id] = enhanced_paper
        
        # Save enhanced data
        new_data = {
            'papers': enhanced_papers,
            'migration_timestamp': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        with open(new_papers_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        logger.info(f"Successfully migrated {len(enhanced_papers)} papers")
    
    def _convert_review_to_enhanced(self, paper_id: str, old_review: Dict[str, Any]) -> StructuredReview:
        """Convert old review format to enhanced structured review."""
        # Create enhanced review criteria from old scores
        criteria = EnhancedReviewCriteria()
        
        # Map old single score to multiple dimensions (if available)
        old_rating = old_review.get('rating', 5.0)
        if isinstance(old_rating, (int, float)):
            # Distribute the old rating across dimensions with some variation
            import random
            base_score = float(old_rating)
            criteria.novelty = max(1.0, min(10.0, base_score + random.uniform(-1, 1)))
            criteria.technical_quality = max(1.0, min(10.0, base_score + random.uniform(-0.5, 0.5)))
            criteria.clarity = max(1.0, min(10.0, base_score + random.uniform(-1, 1)))
            criteria.significance = max(1.0, min(10.0, base_score + random.uniform(-0.5, 0.5)))
            criteria.reproducibility = max(1.0, min(10.0, base_score + random.uniform(-1, 1)))
            criteria.related_work = max(1.0, min(10.0, base_score + random.uniform(-1, 1)))
        
        # Create structured review
        enhanced_review = StructuredReview(
            reviewer_id=old_review.get('reviewer_id', 'unknown'),
            paper_id=paper_id,
            venue_id=old_review.get('venue_id', 'unknown'),
            criteria_scores=criteria,
            executive_summary=old_review.get('text', '')[:200] + "..." if len(old_review.get('text', '')) > 200 else old_review.get('text', ''),
            technical_comments=old_review.get('text', ''),
            confidence_level=old_review.get('confidence', 3)
        )
        
        # Parse old review text for strengths and weaknesses if possible
        review_text = old_review.get('text', '')
        if 'strengths' in review_text.lower() or 'weakness' in review_text.lower():
            # Simple parsing - in practice, this could be more sophisticated
            if 'strengths' in review_text.lower():
                enhanced_review.detailed_strengths.append(
                    DetailedStrength(category="General", description="Extracted from original review")
                )
            if 'weakness' in review_text.lower():
                enhanced_review.detailed_weaknesses.append(
                    DetailedWeakness(category="General", description="Extracted from original review")
                )
        
        return enhanced_review