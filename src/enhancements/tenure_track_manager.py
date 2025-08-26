"""
Tenure Track Management System

This module implements tenure track modeling with 6-year tenure timeline management,
publication requirement tracking, tenure evaluation criteria, and milestone tracking
for the peer review simulation.
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


class TenureStatus(Enum):
    """Tenure track status categories."""
    NOT_ON_TRACK = "Not on Tenure Track"
    ON_TRACK = "On Tenure Track"
    TENURE_REVIEW = "Under Tenure Review"
    TENURED = "Tenured"
    TENURE_DENIED = "Tenure Denied"


class TenureYear(Enum):
    """Tenure track year markers."""
    YEAR_1 = 1
    YEAR_2 = 2
    YEAR_3 = 3
    YEAR_4 = 4
    YEAR_5 = 5
    YEAR_6 = 6


@dataclass
class TenureTimeline:
    """Represents a 6-year tenure track timeline."""
    start_date: date
    expected_review_date: date
    current_year: int
    status: TenureStatus
    milestones_completed: List[str] = field(default_factory=list)
    milestones_pending: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate tenure timeline data."""
        if not (1 <= self.current_year <= 6):
            raise ValidationError("current_year", self.current_year, "year between 1 and 6")
        
        if self.expected_review_date <= self.start_date:
            raise ValidationError("expected_review_date", self.expected_review_date, 
                                "date after start_date")
    
    def get_years_remaining(self) -> float:
        """Calculate years remaining until tenure review."""
        today = date.today()
        days_remaining = (self.expected_review_date - today).days
        return max(0, days_remaining / 365.25)
    
    def get_progress_percentage(self) -> float:
        """Calculate progress through tenure track as percentage."""
        return min(100.0, (self.current_year / 6.0) * 100.0)


@dataclass
class TenureRequirements:
    """Tenure evaluation requirements and criteria."""
    min_publications: int
    min_first_author_publications: int
    min_journal_publications: int
    min_conference_publications: int
    min_h_index: int
    min_total_citations: int
    min_external_funding: float  # in dollars
    required_service_roles: List[str]
    required_teaching_evaluations: float  # minimum average rating
    external_letters_required: int
    
    def __post_init__(self):
        """Validate tenure requirements."""
        if self.min_publications < 0:
            raise ValidationError("min_publications", self.min_publications, "non-negative integer")
        if self.min_h_index < 0:
            raise ValidationError("min_h_index", self.min_h_index, "non-negative integer")
        if self.min_external_funding < 0:
            raise ValidationError("min_external_funding", self.min_external_funding, "non-negative number")


@dataclass
class TenureMilestone:
    """Represents a tenure track milestone."""
    year: int
    milestone_type: str
    description: str
    required: bool
    completed: bool = False
    completion_date: Optional[date] = None
    notes: str = ""
    
    def __post_init__(self):
        """Validate milestone data."""
        if not (1 <= self.year <= 6):
            raise ValidationError("year", self.year, "year between 1 and 6")


@dataclass
class TenureEvaluation:
    """Represents a tenure evaluation result."""
    researcher_id: str
    evaluation_date: date
    research_score: float  # 0-100
    teaching_score: float  # 0-100
    service_score: float   # 0-100
    overall_score: float   # 0-100
    recommendation: str    # "Grant Tenure", "Deny Tenure", "Extend Review"
    strengths: List[str]
    weaknesses: List[str]
    committee_notes: str
    external_letter_summary: str
    
    def __post_init__(self):
        """Validate evaluation scores."""
        for score_name in ['research_score', 'teaching_score', 'service_score', 'overall_score']:
            score = getattr(self, score_name)
            if not (0 <= score <= 100):
                raise ValidationError(score_name, score, "score between 0 and 100")


class TenureTrackManager:
    """
    Manages tenure track modeling with 6-year timeline, publication requirements,
    and evaluation criteria tracking.
    
    This class provides functionality to:
    - Create and manage 6-year tenure timelines
    - Track publication requirements and progress
    - Implement tenure evaluation criteria
    - Monitor milestone completion
    - Calculate tenure success probability
    """
    
    # Standard tenure requirements by field (can be customized)
    DEFAULT_REQUIREMENTS = {
        'computer_science': TenureRequirements(
            min_publications=15,
            min_first_author_publications=8,
            min_journal_publications=5,
            min_conference_publications=10,
            min_h_index=12,
            min_total_citations=200,
            min_external_funding=150000,
            required_service_roles=['reviewer', 'program_committee'],
            required_teaching_evaluations=3.5,
            external_letters_required=6
        ),
        'biology': TenureRequirements(
            min_publications=20,
            min_first_author_publications=10,
            min_journal_publications=15,
            min_conference_publications=5,
            min_h_index=15,
            min_total_citations=400,
            min_external_funding=300000,
            required_service_roles=['reviewer', 'editorial_board'],
            required_teaching_evaluations=3.5,
            external_letters_required=8
        ),
        'physics': TenureRequirements(
            min_publications=18,
            min_first_author_publications=9,
            min_journal_publications=12,
            min_conference_publications=6,
            min_h_index=14,
            min_total_citations=350,
            min_external_funding=250000,
            required_service_roles=['reviewer', 'conference_organizer'],
            required_teaching_evaluations=3.5,
            external_letters_required=7
        )
    }
    
    # Standard milestones for 6-year tenure track
    STANDARD_MILESTONES = [
        TenureMilestone(1, "setup", "Establish research program and lab", True),
        TenureMilestone(1, "teaching", "Complete first year of teaching", True),
        TenureMilestone(2, "research", "Submit first major grant application", True),
        TenureMilestone(2, "publication", "Publish first papers from new position", True),
        TenureMilestone(3, "service", "Take on significant service role", True),
        TenureMilestone(3, "funding", "Secure external funding", True),
        TenureMilestone(4, "visibility", "Establish national visibility in field", True),
        TenureMilestone(4, "mentoring", "Begin mentoring graduate students", True),
        TenureMilestone(5, "dossier", "Prepare tenure dossier", True),
        TenureMilestone(5, "external", "Solicit external letters", True),
        TenureMilestone(6, "review", "Undergo tenure review", True),
    ]
    
    def __init__(self):
        """Initialize the tenure track manager."""
        logger.info("Initializing Tenure Track Management System")
        self.active_timelines: Dict[str, TenureTimeline] = {}
        self.tenure_requirements: Dict[str, TenureRequirements] = self.DEFAULT_REQUIREMENTS.copy()
        self.evaluations: Dict[str, List[TenureEvaluation]] = {}
    
    def create_tenure_timeline(self, researcher: EnhancedResearcher, 
                             start_date: Optional[date] = None,
                             field: str = "computer_science") -> TenureTimeline:
        """
        Create a new 6-year tenure timeline for a researcher.
        
        Args:
            researcher: The researcher starting tenure track
            start_date: Start date of tenure track (defaults to today)
            field: Academic field for requirements (defaults to computer_science)
            
        Returns:
            TenureTimeline object
            
        Raises:
            CareerSystemError: If researcher is not eligible for tenure track
        """
        if researcher.level != ResearcherLevel.ASSISTANT_PROF:
            raise CareerSystemError(f"Only Assistant Professors can be on tenure track, "
                                  f"researcher is {researcher.level.value}")
        
        if start_date is None:
            start_date = date.today()
        
        # Calculate expected review date (6 years from start)
        expected_review_date = start_date + timedelta(days=6*365)
        
        timeline = TenureTimeline(
            start_date=start_date,
            expected_review_date=expected_review_date,
            current_year=1,
            status=TenureStatus.ON_TRACK,
            milestones_pending=[m.description for m in self.STANDARD_MILESTONES]
        )
        
        self.active_timelines[researcher.id] = timeline
        
        logger.info(f"Created tenure timeline for {researcher.name}, "
                   f"review expected on {expected_review_date}")
        
        return timeline
    
    def get_tenure_timeline(self, researcher_id: str) -> Optional[TenureTimeline]:
        """
        Get tenure timeline for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            TenureTimeline if exists, None otherwise
        """
        return self.active_timelines.get(researcher_id)
    
    def update_tenure_year(self, researcher_id: str) -> bool:
        """
        Update the current year in tenure timeline based on elapsed time.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            True if year was updated, False otherwise
        """
        timeline = self.active_timelines.get(researcher_id)
        if not timeline:
            return False
        
        # Calculate current year based on elapsed time
        today = date.today()
        days_elapsed = (today - timeline.start_date).days
        current_year = min(6, max(1, int(days_elapsed / 365.25) + 1))
        
        if current_year != timeline.current_year:
            old_year = timeline.current_year
            timeline.current_year = current_year
            logger.info(f"Updated tenure year for researcher {researcher_id} "
                       f"from year {old_year} to year {current_year}")
            return True
        
        return False
    
    def track_publication_requirements(self, researcher: EnhancedResearcher,
                                     field: str = "computer_science") -> Dict[str, Any]:
        """
        Track publication requirements and progress for tenure.
        
        Args:
            researcher: The researcher to track
            field: Academic field for requirements
            
        Returns:
            Dictionary with requirement tracking information
        """
        requirements = self.tenure_requirements.get(field, self.DEFAULT_REQUIREMENTS['computer_science'])
        
        # Count publications by type
        total_pubs = len(researcher.publication_history)
        first_author_pubs = len([p for p in researcher.publication_history 
                               if p.get('first_author', False)])
        journal_pubs = len([p for p in researcher.publication_history 
                          if p.get('venue_type') == 'journal'])
        conference_pubs = len([p for p in researcher.publication_history 
                             if p.get('venue_type') == 'conference'])
        
        # Calculate progress percentages
        progress = {
            'total_publications': {
                'current': total_pubs,
                'required': requirements.min_publications,
                'percentage': min(100, (total_pubs / requirements.min_publications) * 100),
                'on_track': total_pubs >= requirements.min_publications
            },
            'first_author_publications': {
                'current': first_author_pubs,
                'required': requirements.min_first_author_publications,
                'percentage': min(100, (first_author_pubs / requirements.min_first_author_publications) * 100),
                'on_track': first_author_pubs >= requirements.min_first_author_publications
            },
            'journal_publications': {
                'current': journal_pubs,
                'required': requirements.min_journal_publications,
                'percentage': min(100, (journal_pubs / requirements.min_journal_publications) * 100),
                'on_track': journal_pubs >= requirements.min_journal_publications
            },
            'conference_publications': {
                'current': conference_pubs,
                'required': requirements.min_conference_publications,
                'percentage': min(100, (conference_pubs / requirements.min_conference_publications) * 100),
                'on_track': conference_pubs >= requirements.min_conference_publications
            },
            'h_index': {
                'current': researcher.h_index,
                'required': requirements.min_h_index,
                'percentage': min(100, (researcher.h_index / requirements.min_h_index) * 100),
                'on_track': researcher.h_index >= requirements.min_h_index
            },
            'citations': {
                'current': researcher.total_citations,
                'required': requirements.min_total_citations,
                'percentage': min(100, (researcher.total_citations / requirements.min_total_citations) * 100),
                'on_track': researcher.total_citations >= requirements.min_total_citations
            }
        }
        
        # Calculate overall progress
        on_track_count = sum(1 for metric in progress.values() if metric['on_track'])
        overall_progress = (on_track_count / len(progress)) * 100
        
        return {
            'field': field,
            'requirements': requirements,
            'progress': progress,
            'overall_progress': overall_progress,
            'on_track': overall_progress >= 70,  # 70% threshold for being "on track"
            'areas_needing_attention': [key for key, value in progress.items() 
                                      if not value['on_track']]
        }
    
    def evaluate_tenure_readiness(self, researcher: EnhancedResearcher,
                                field: str = "computer_science") -> TenureEvaluation:
        """
        Evaluate researcher's readiness for tenure based on comprehensive criteria.
        
        Args:
            researcher: The researcher to evaluate
            field: Academic field for evaluation criteria
            
        Returns:
            TenureEvaluation object with detailed assessment
        """
        timeline = self.get_tenure_timeline(researcher.id)
        if not timeline:
            raise CareerSystemError(f"No tenure timeline found for researcher {researcher.id}")
        
        # Track publication progress
        pub_progress = self.track_publication_requirements(researcher, field)
        
        # Calculate research score (0-100)
        research_score = self._calculate_research_score(researcher, pub_progress)
        
        # Calculate teaching score (placeholder - would integrate with teaching data)
        teaching_score = self._calculate_teaching_score(researcher)
        
        # Calculate service score (placeholder - would integrate with service data)
        service_score = self._calculate_service_score(researcher)
        
        # Calculate overall score
        overall_score = (research_score * 0.6 + teaching_score * 0.25 + service_score * 0.15)
        
        # Determine recommendation
        if overall_score >= 80:
            recommendation = "Grant Tenure"
        elif overall_score >= 60:
            recommendation = "Extend Review"
        else:
            recommendation = "Deny Tenure"
        
        # Generate strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(researcher, pub_progress)
        
        evaluation = TenureEvaluation(
            researcher_id=researcher.id,
            evaluation_date=date.today(),
            research_score=research_score,
            teaching_score=teaching_score,
            service_score=service_score,
            overall_score=overall_score,
            recommendation=recommendation,
            strengths=strengths,
            weaknesses=weaknesses,
            committee_notes=f"Evaluation based on {timeline.current_year} years of tenure track progress",
            external_letter_summary="External letters pending" if timeline.current_year < 5 else "External letters received"
        )
        
        # Store evaluation
        if researcher.id not in self.evaluations:
            self.evaluations[researcher.id] = []
        self.evaluations[researcher.id].append(evaluation)
        
        logger.info(f"Completed tenure evaluation for {researcher.name}: "
                   f"{recommendation} (score: {overall_score:.1f})")
        
        return evaluation
    
    def _calculate_research_score(self, researcher: EnhancedResearcher, 
                                pub_progress: Dict[str, Any]) -> float:
        """Calculate research component score for tenure evaluation."""
        # Base score from publication progress (0-60 points)
        base_score = pub_progress['overall_progress'] * 0.6
        
        # Bonus for exceeding requirements
        if pub_progress['overall_progress'] > 100:
            base_score += min(20, (pub_progress['overall_progress'] - 100) * 0.2)
        
        # H-index component (0-25 points)
        h_index_score = min(25, researcher.h_index * 1.0)
        
        # Citation impact component (0-15 points)
        citation_score = min(15, researcher.total_citations / 20)
        
        total_score = min(100.0, base_score + h_index_score + citation_score)
        return total_score
    
    def _calculate_teaching_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate teaching component score (placeholder implementation)."""
        # In a real system, this would integrate with teaching evaluation data
        # For simulation, use a reasonable default based on researcher characteristics
        base_score = 75.0  # Assume adequate teaching
        
        # Senior researchers might be better teachers
        if researcher.level in [ResearcherLevel.ASSOCIATE_PROF, ResearcherLevel.FULL_PROF]:
            base_score += 10
        
        # Add some randomness for realism
        import random
        variation = random.uniform(-10, 15)
        
        return max(0, min(100, base_score + variation))
    
    def _calculate_service_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate service component score (placeholder implementation)."""
        # In a real system, this would integrate with service record data
        # For simulation, base on years active and level
        base_score = min(80.0, researcher.years_active * 10.0)
        
        # Bonus for senior levels (more service expected)
        if researcher.level == ResearcherLevel.ASSOCIATE_PROF:
            base_score += 10.0
        elif researcher.level == ResearcherLevel.FULL_PROF:
            base_score += 15.0
        
        return max(0.0, min(100.0, base_score))
    
    def _analyze_strengths_weaknesses(self, researcher: EnhancedResearcher,
                                    pub_progress: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze researcher strengths and weaknesses for tenure evaluation."""
        strengths = []
        weaknesses = []
        
        # Analyze publication record
        if pub_progress['progress']['total_publications']['on_track']:
            strengths.append("Strong publication record")
        else:
            weaknesses.append("Publication count below expectations")
        
        if pub_progress['progress']['h_index']['on_track']:
            strengths.append("Good citation impact")
        else:
            weaknesses.append("Limited citation impact")
        
        # Analyze career stage appropriateness
        if researcher.h_index > 15:
            strengths.append("Excellent research impact for career stage")
        elif researcher.h_index < 8:
            weaknesses.append("Research impact below expectations")
        
        # Analyze productivity
        years_active = max(1, researcher.years_active)
        pubs_per_year = len(researcher.publication_history) / years_active
        if pubs_per_year > 3:
            strengths.append("High research productivity")
        elif pubs_per_year < 2:
            weaknesses.append("Research productivity could be improved")
        
        return strengths, weaknesses
    
    def complete_milestone(self, researcher_id: str, milestone_description: str,
                         completion_date: Optional[date] = None,
                         notes: str = "") -> bool:
        """
        Mark a tenure milestone as completed.
        
        Args:
            researcher_id: ID of the researcher
            milestone_description: Description of the milestone
            completion_date: Date of completion (defaults to today)
            notes: Additional notes about completion
            
        Returns:
            True if milestone was marked complete, False otherwise
        """
        timeline = self.active_timelines.get(researcher_id)
        if not timeline:
            return False
        
        if milestone_description in timeline.milestones_pending:
            timeline.milestones_pending.remove(milestone_description)
            timeline.milestones_completed.append(milestone_description)
            
            logger.info(f"Completed milestone for researcher {researcher_id}: {milestone_description}")
            return True
        
        return False
    
    def get_milestone_progress(self, researcher_id: str) -> Dict[str, Any]:
        """
        Get milestone completion progress for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Dictionary with milestone progress information
        """
        timeline = self.active_timelines.get(researcher_id)
        if not timeline:
            return {'error': 'No tenure timeline found'}
        
        total_milestones = len(self.STANDARD_MILESTONES)
        completed_count = len(timeline.milestones_completed)
        
        return {
            'total_milestones': total_milestones,
            'completed_milestones': completed_count,
            'pending_milestones': len(timeline.milestones_pending),
            'completion_percentage': (completed_count / total_milestones) * 100,
            'completed_list': timeline.milestones_completed,
            'pending_list': timeline.milestones_pending,
            'current_year': timeline.current_year,
            'years_remaining': timeline.get_years_remaining(),
            'on_schedule': completed_count >= (timeline.current_year * 2)  # Expect ~2 milestones per year
        }
    
    def calculate_tenure_success_probability(self, researcher: EnhancedResearcher,
                                           field: str = "computer_science") -> float:
        """
        Calculate probability of tenure success based on current progress.
        
        Args:
            researcher: The researcher to analyze
            field: Academic field for evaluation
            
        Returns:
            Probability of tenure success (0.0 to 1.0)
        """
        timeline = self.get_tenure_timeline(researcher.id)
        if not timeline:
            return 0.0
        
        # Get publication progress
        pub_progress = self.track_publication_requirements(researcher, field)
        
        # Get milestone progress
        milestone_progress = self.get_milestone_progress(researcher.id)
        
        # Calculate base probability from publication progress
        pub_probability = pub_progress['overall_progress'] / 100.0
        
        # Adjust for milestone completion
        milestone_probability = milestone_progress['completion_percentage'] / 100.0
        
        # Adjust for time remaining
        years_remaining = timeline.get_years_remaining()
        time_factor = 1.0
        if years_remaining < 1:
            time_factor = 0.8  # Less time to improve
        elif years_remaining > 3:
            time_factor = 1.2  # More time to improve
        
        # Combine factors
        overall_probability = (pub_probability * 0.6 + milestone_probability * 0.4) * time_factor
        
        # Cap at reasonable bounds
        return max(0.0, min(1.0, overall_probability))
    
    def get_tenure_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all active tenure timelines.
        
        Returns:
            Dictionary with tenure track statistics
        """
        if not self.active_timelines:
            return {'total_timelines': 0}
        
        timelines = list(self.active_timelines.values())
        
        # Status distribution
        status_counts = {}
        for status in TenureStatus:
            status_counts[status.value] = len([t for t in timelines if t.status == status])
        
        # Year distribution
        year_counts = {}
        for year in range(1, 7):
            year_counts[f"Year {year}"] = len([t for t in timelines if t.current_year == year])
        
        # Average progress
        avg_progress = sum(t.get_progress_percentage() for t in timelines) / len(timelines)
        
        return {
            'total_timelines': len(timelines),
            'status_distribution': status_counts,
            'year_distribution': year_counts,
            'average_progress': avg_progress,
            'timelines_on_track': len([t for t in timelines if t.status == TenureStatus.ON_TRACK]),
            'timelines_under_review': len([t for t in timelines if t.status == TenureStatus.TENURE_REVIEW])
        }
    
    def set_custom_requirements(self, field: str, requirements: TenureRequirements):
        """
        Set custom tenure requirements for a specific field.
        
        Args:
            field: Academic field name
            requirements: TenureRequirements object
        """
        self.tenure_requirements[field] = requirements
        logger.info(f"Set custom tenure requirements for field: {field}")
    
    def export_timeline_data(self, researcher_id: str) -> Dict[str, Any]:
        """
        Export tenure timeline data for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Dictionary with exportable timeline data
        """
        timeline = self.active_timelines.get(researcher_id)
        if not timeline:
            return {'error': 'No timeline found'}
        
        return {
            'researcher_id': researcher_id,
            'start_date': timeline.start_date.isoformat(),
            'expected_review_date': timeline.expected_review_date.isoformat(),
            'current_year': timeline.current_year,
            'status': timeline.status.value,
            'progress_percentage': timeline.get_progress_percentage(),
            'years_remaining': timeline.get_years_remaining(),
            'milestones_completed': timeline.milestones_completed,
            'milestones_pending': timeline.milestones_pending,
            'evaluations': [
                {
                    'date': eval.evaluation_date.isoformat(),
                    'overall_score': eval.overall_score,
                    'recommendation': eval.recommendation
                }
                for eval in self.evaluations.get(researcher_id, [])
            ]
        }