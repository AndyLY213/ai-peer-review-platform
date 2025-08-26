"""
Venue Shopping Tracker

This module implements venue shopping tracking functionality to monitor submission patterns
and detect strategic submission behavior where researchers resubmit rejected papers to
progressively lower-tier venues.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from src.data.enhanced_models import VenueType, EnhancedVenue
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class SubmissionOutcome(Enum):
    """Possible outcomes for paper submissions."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


@dataclass
class SubmissionRecord:
    """Record of a paper submission to a venue."""
    submission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str = ""
    researcher_id: str = ""
    venue_id: str = ""
    venue_type: VenueType = VenueType.MID_CONFERENCE
    venue_prestige: int = 5  # 1-10 scale
    submission_date: datetime = field(default_factory=datetime.now)
    outcome: SubmissionOutcome = SubmissionOutcome.PENDING
    outcome_date: Optional[datetime] = None
    review_scores: List[float] = field(default_factory=list)
    average_score: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.review_scores:
            self.average_score = sum(self.review_scores) / len(self.review_scores)


@dataclass
class VenueShoppingPattern:
    """Pattern of venue shopping behavior for a paper."""
    paper_id: str
    researcher_id: str
    submission_sequence: List[SubmissionRecord] = field(default_factory=list)
    venue_downgrade_count: int = 0
    total_venues_tried: int = 0
    time_span_days: int = 0
    final_acceptance_venue: Optional[str] = None
    shopping_score: float = 0.0  # 0-1 scale indicating shopping intensity
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self._calculate_venue_downgrades()
        self._calculate_time_span()
        self._calculate_shopping_score()
    
    def _calculate_venue_downgrades(self):
        """Calculate number of venue downgrades."""
        self.total_venues_tried = len(self.submission_sequence)
        
        if len(self.submission_sequence) < 2:
            return
        
        downgrades = 0
        for i in range(1, len(self.submission_sequence)):
            prev_prestige = self.submission_sequence[i-1].venue_prestige
            curr_prestige = self.submission_sequence[i].venue_prestige
            if curr_prestige < prev_prestige:
                downgrades += 1
        
        self.venue_downgrade_count = downgrades
    
    def _calculate_time_span(self):
        """Calculate time span of shopping behavior."""
        if len(self.submission_sequence) < 2:
            return
        
        first_submission = min(self.submission_sequence, key=lambda x: x.submission_date)
        last_submission = max(self.submission_sequence, key=lambda x: x.submission_date)
        
        self.time_span_days = (last_submission.submission_date - first_submission.submission_date).days
    
    def _calculate_shopping_score(self):
        """Calculate shopping intensity score (0-1 scale)."""
        if len(self.submission_sequence) <= 1:
            self.shopping_score = 0.0
            return
        
        # Factors contributing to shopping score
        venue_count_factor = min(1.0, (self.total_venues_tried - 1) / 5.0)  # Normalize to 5 venues
        downgrade_factor = min(1.0, self.venue_downgrade_count / 3.0)  # Normalize to 3 downgrades
        time_factor = min(1.0, self.time_span_days / 365.0)  # Normalize to 1 year
        
        # Weight the factors
        self.shopping_score = (venue_count_factor * 0.4 + 
                              downgrade_factor * 0.4 + 
                              time_factor * 0.2)


@dataclass
class ResearcherShoppingProfile:
    """Profile of a researcher's venue shopping behavior."""
    researcher_id: str
    total_papers_submitted: int = 0
    papers_with_shopping: int = 0
    average_venues_per_paper: float = 1.0
    average_shopping_score: float = 0.0
    preferred_venue_types: List[VenueType] = field(default_factory=list)
    shopping_tendency: float = 0.0  # 0-1 scale
    
    def update_profile(self, shopping_patterns: List[VenueShoppingPattern]):
        """Update profile based on shopping patterns."""
        if not shopping_patterns:
            return
        
        self.total_papers_submitted = len(shopping_patterns)
        self.papers_with_shopping = sum(1 for p in shopping_patterns if p.shopping_score > 0.1)
        
        total_venues = sum(p.total_venues_tried for p in shopping_patterns)
        self.average_venues_per_paper = total_venues / len(shopping_patterns)
        
        total_shopping_score = sum(p.shopping_score for p in shopping_patterns)
        self.average_shopping_score = total_shopping_score / len(shopping_patterns)
        
        # Calculate shopping tendency
        self.shopping_tendency = self.papers_with_shopping / self.total_papers_submitted
        
        # Identify preferred venue types
        venue_type_counts = {}
        for pattern in shopping_patterns:
            for submission in pattern.submission_sequence:
                venue_type = submission.venue_type
                venue_type_counts[venue_type] = venue_type_counts.get(venue_type, 0) + 1
        
        # Sort by frequency and take top 3
        sorted_types = sorted(venue_type_counts.items(), key=lambda x: x[1], reverse=True)
        self.preferred_venue_types = [vtype for vtype, _ in sorted_types[:3]]


class VenueShoppingTracker:
    """
    Tracks venue shopping behavior patterns and detects strategic submission behavior.
    
    This class monitors paper submissions across venues and identifies patterns where
    researchers resubmit rejected papers to progressively lower-tier venues.
    """
    
    def __init__(self, data_dir: str = "data/venue_shopping"):
        """
        Initialize the venue shopping tracker.
        
        Args:
            data_dir: Directory to store venue shopping data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for tracking data
        self.submission_records: Dict[str, List[SubmissionRecord]] = {}  # paper_id -> submissions
        self.shopping_patterns: Dict[str, VenueShoppingPattern] = {}  # paper_id -> pattern
        self.researcher_profiles: Dict[str, ResearcherShoppingProfile] = {}  # researcher_id -> profile
        
        # Configuration
        self.min_shopping_score = 0.2  # Minimum score to consider as shopping
        self.max_time_between_submissions = 365  # Days
        
        logger.info(f"VenueShoppingTracker initialized with data directory: {self.data_dir}")
    
    def record_submission(self, paper_id: str, researcher_id: str, venue_id: str, 
                         venue_type: VenueType, venue_prestige: int) -> str:
        """
        Record a new paper submission.
        
        Args:
            paper_id: Unique identifier for the paper
            researcher_id: Unique identifier for the researcher
            venue_id: Unique identifier for the venue
            venue_type: Type of venue
            venue_prestige: Prestige score of venue (1-10)
            
        Returns:
            Submission ID
        """
        submission = SubmissionRecord(
            paper_id=paper_id,
            researcher_id=researcher_id,
            venue_id=venue_id,
            venue_type=venue_type,
            venue_prestige=venue_prestige
        )
        
        # Add to records
        if paper_id not in self.submission_records:
            self.submission_records[paper_id] = []
        self.submission_records[paper_id].append(submission)
        
        logger.debug(f"Recorded submission {submission.submission_id} for paper {paper_id} to venue {venue_id}")
        
        return submission.submission_id
    
    def update_submission_outcome(self, paper_id: str, submission_id: str, 
                                outcome: SubmissionOutcome, review_scores: List[float] = None):
        """
        Update the outcome of a submission.
        
        Args:
            paper_id: Paper identifier
            submission_id: Submission identifier
            outcome: Outcome of the submission
            review_scores: List of review scores if available
        """
        if paper_id not in self.submission_records:
            raise ValidationError("paper_id", paper_id, "existing paper ID")
        
        # Find and update the submission
        for submission in self.submission_records[paper_id]:
            if submission.submission_id == submission_id:
                submission.outcome = outcome
                submission.outcome_date = datetime.now()
                if review_scores:
                    submission.review_scores = review_scores
                    submission.average_score = sum(review_scores) / len(review_scores)
                
                logger.debug(f"Updated submission {submission_id} outcome to {outcome.value}")
                
                # Update shopping pattern if this is a rejection or acceptance
                if outcome in [SubmissionOutcome.ACCEPTED, SubmissionOutcome.REJECTED]:
                    self._update_shopping_pattern(paper_id)
                
                return
        
        raise ValidationError("submission_id", submission_id, "existing submission ID")
    
    def _update_shopping_pattern(self, paper_id: str):
        """Update the shopping pattern for a paper."""
        if paper_id not in self.submission_records:
            return
        
        submissions = self.submission_records[paper_id]
        if not submissions:
            return
        
        # Get researcher ID from first submission
        researcher_id = submissions[0].researcher_id
        
        # Create or update shopping pattern
        if paper_id not in self.shopping_patterns:
            self.shopping_patterns[paper_id] = VenueShoppingPattern(
                paper_id=paper_id,
                researcher_id=researcher_id
            )
        
        pattern = self.shopping_patterns[paper_id]
        pattern.submission_sequence = sorted(submissions, key=lambda x: x.submission_date)
        
        # Recalculate metrics
        pattern.__post_init__()
        
        # Check for final acceptance
        accepted_submissions = [s for s in submissions if s.outcome == SubmissionOutcome.ACCEPTED]
        if accepted_submissions:
            latest_acceptance = max(accepted_submissions, key=lambda x: x.submission_date)
            pattern.final_acceptance_venue = latest_acceptance.venue_id
        
        # Update researcher profile
        self._update_researcher_profile(researcher_id)
    
    def _update_researcher_profile(self, researcher_id: str):
        """Update the shopping profile for a researcher."""
        # Get all patterns for this researcher
        researcher_patterns = [p for p in self.shopping_patterns.values() 
                             if p.researcher_id == researcher_id]
        
        if not researcher_patterns:
            return
        
        # Create or update profile
        if researcher_id not in self.researcher_profiles:
            self.researcher_profiles[researcher_id] = ResearcherShoppingProfile(
                researcher_id=researcher_id
            )
        
        profile = self.researcher_profiles[researcher_id]
        profile.update_profile(researcher_patterns)
    
    def detect_venue_shopping(self, paper_id: str) -> Optional[VenueShoppingPattern]:
        """
        Detect venue shopping pattern for a specific paper.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            VenueShoppingPattern if shopping detected, None otherwise
        """
        if paper_id not in self.shopping_patterns:
            return None
        
        pattern = self.shopping_patterns[paper_id]
        
        # Consider it shopping if score is above threshold and multiple venues tried
        if pattern.shopping_score >= self.min_shopping_score and pattern.total_venues_tried > 1:
            return pattern
        
        return None
    
    def get_researcher_shopping_behavior(self, researcher_id: str) -> Optional[ResearcherShoppingProfile]:
        """
        Get venue shopping behavior profile for a researcher.
        
        Args:
            researcher_id: Researcher identifier
            
        Returns:
            ResearcherShoppingProfile if available, None otherwise
        """
        return self.researcher_profiles.get(researcher_id)
    
    def analyze_venue_downgrade_patterns(self) -> Dict[str, List[Tuple[VenueType, VenueType]]]:
        """
        Analyze common venue downgrade patterns.
        
        Returns:
            Dictionary mapping researcher IDs to lists of venue type transitions
        """
        downgrade_patterns = {}
        
        for pattern in self.shopping_patterns.values():
            if pattern.venue_downgrade_count == 0:
                continue
            
            researcher_id = pattern.researcher_id
            if researcher_id not in downgrade_patterns:
                downgrade_patterns[researcher_id] = []
            
            # Extract venue type transitions
            for i in range(1, len(pattern.submission_sequence)):
                prev_venue = pattern.submission_sequence[i-1]
                curr_venue = pattern.submission_sequence[i]
                
                if curr_venue.venue_prestige < prev_venue.venue_prestige:
                    transition = (prev_venue.venue_type, curr_venue.venue_type)
                    downgrade_patterns[researcher_id].append(transition)
        
        return downgrade_patterns
    
    def get_shopping_statistics(self) -> Dict[str, float]:
        """
        Get overall venue shopping statistics.
        
        Returns:
            Dictionary with shopping statistics
        """
        if not self.shopping_patterns:
            return {
                'total_papers': 0,
                'papers_with_shopping': 0,
                'shopping_rate': 0.0,
                'average_venues_per_paper': 0.0,
                'average_shopping_score': 0.0
            }
        
        total_papers = len(self.shopping_patterns)
        papers_with_shopping = sum(1 for p in self.shopping_patterns.values() 
                                 if p.shopping_score >= self.min_shopping_score)
        
        total_venues = sum(p.total_venues_tried for p in self.shopping_patterns.values())
        total_shopping_score = sum(p.shopping_score for p in self.shopping_patterns.values())
        
        return {
            'total_papers': total_papers,
            'papers_with_shopping': papers_with_shopping,
            'shopping_rate': papers_with_shopping / total_papers if total_papers > 0 else 0.0,
            'average_venues_per_paper': total_venues / total_papers if total_papers > 0 else 0.0,
            'average_shopping_score': total_shopping_score / total_papers if total_papers > 0 else 0.0
        }
    
    def predict_next_venue_choice(self, researcher_id: str, current_venue_prestige: int, 
                                rejection_score: float) -> List[Tuple[VenueType, float]]:
        """
        Predict likely next venue choices for a researcher after rejection.
        
        Args:
            researcher_id: Researcher identifier
            current_venue_prestige: Prestige of current venue (1-10)
            rejection_score: Average rejection score
            
        Returns:
            List of (venue_type, probability) tuples
        """
        profile = self.researcher_profiles.get(researcher_id)
        if not profile:
            # Default prediction based on prestige downgrade
            target_prestige = max(1, current_venue_prestige - 2)
            return [(VenueType.MID_CONFERENCE, 0.6), (VenueType.LOW_CONFERENCE, 0.4)]
        
        # Analyze historical patterns
        venue_transitions = {}
        for pattern in self.shopping_patterns.values():
            if pattern.researcher_id != researcher_id:
                continue
            
            for i in range(1, len(pattern.submission_sequence)):
                prev_submission = pattern.submission_sequence[i-1]
                curr_submission = pattern.submission_sequence[i]
                
                if prev_submission.outcome == SubmissionOutcome.REJECTED:
                    prev_prestige = prev_submission.venue_prestige
                    curr_type = curr_submission.venue_type
                    
                    # Group by prestige ranges
                    prestige_range = self._get_prestige_range(prev_prestige)
                    if prestige_range not in venue_transitions:
                        venue_transitions[prestige_range] = {}
                    
                    if curr_type not in venue_transitions[prestige_range]:
                        venue_transitions[prestige_range][curr_type] = 0
                    venue_transitions[prestige_range][curr_type] += 1
        
        # Get predictions for current prestige range
        current_range = self._get_prestige_range(current_venue_prestige)
        if current_range in venue_transitions:
            total_transitions = sum(venue_transitions[current_range].values())
            predictions = [(vtype, count / total_transitions) 
                          for vtype, count in venue_transitions[current_range].items()]
            return sorted(predictions, key=lambda x: x[1], reverse=True)
        
        # Fallback prediction
        return [(VenueType.MID_CONFERENCE, 0.5), (VenueType.LOW_CONFERENCE, 0.3), 
                (VenueType.GENERAL_JOURNAL, 0.2)]
    
    def _get_prestige_range(self, prestige: int) -> str:
        """Get prestige range category."""
        if prestige >= 8:
            return "high"
        elif prestige >= 5:
            return "medium"
        else:
            return "low"
    
    def save_data(self, filename: str = "venue_shopping_data.json"):
        """Save venue shopping data to file."""
        data = {
            'submission_records': {
                paper_id: [self._submission_to_dict(s) for s in submissions]
                for paper_id, submissions in self.submission_records.items()
            },
            'shopping_patterns': {
                paper_id: self._pattern_to_dict(pattern)
                for paper_id, pattern in self.shopping_patterns.items()
            },
            'researcher_profiles': {
                researcher_id: self._profile_to_dict(profile)
                for researcher_id, profile in self.researcher_profiles.items()
            }
        }
        
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Venue shopping data saved to {filepath}")
    
    def load_data(self, filename: str = "venue_shopping_data.json"):
        """Load venue shopping data from file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"Venue shopping data file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load submission records
            self.submission_records = {}
            for paper_id, submissions_data in data.get('submission_records', {}).items():
                self.submission_records[paper_id] = [
                    self._dict_to_submission(s) for s in submissions_data
                ]
            
            # Load shopping patterns
            self.shopping_patterns = {}
            for paper_id, pattern_data in data.get('shopping_patterns', {}).items():
                self.shopping_patterns[paper_id] = self._dict_to_pattern(pattern_data)
            
            # Load researcher profiles
            self.researcher_profiles = {}
            for researcher_id, profile_data in data.get('researcher_profiles', {}).items():
                self.researcher_profiles[researcher_id] = self._dict_to_profile(profile_data)
            
            logger.info(f"Venue shopping data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading venue shopping data: {e}")
            raise
    
    def _submission_to_dict(self, submission: SubmissionRecord) -> Dict:
        """Convert submission record to dictionary."""
        return {
            'submission_id': submission.submission_id,
            'paper_id': submission.paper_id,
            'researcher_id': submission.researcher_id,
            'venue_id': submission.venue_id,
            'venue_type': submission.venue_type.value,
            'venue_prestige': submission.venue_prestige,
            'submission_date': submission.submission_date.isoformat(),
            'outcome': submission.outcome.value,
            'outcome_date': submission.outcome_date.isoformat() if submission.outcome_date else None,
            'review_scores': submission.review_scores,
            'average_score': submission.average_score
        }
    
    def _dict_to_submission(self, data: Dict) -> SubmissionRecord:
        """Convert dictionary to submission record."""
        return SubmissionRecord(
            submission_id=data['submission_id'],
            paper_id=data['paper_id'],
            researcher_id=data['researcher_id'],
            venue_id=data['venue_id'],
            venue_type=VenueType(data['venue_type']),
            venue_prestige=data['venue_prestige'],
            submission_date=datetime.fromisoformat(data['submission_date']),
            outcome=SubmissionOutcome(data['outcome']),
            outcome_date=datetime.fromisoformat(data['outcome_date']) if data['outcome_date'] else None,
            review_scores=data['review_scores'],
            average_score=data['average_score']
        )
    
    def _pattern_to_dict(self, pattern: VenueShoppingPattern) -> Dict:
        """Convert shopping pattern to dictionary."""
        return {
            'paper_id': pattern.paper_id,
            'researcher_id': pattern.researcher_id,
            'submission_sequence': [self._submission_to_dict(s) for s in pattern.submission_sequence],
            'venue_downgrade_count': pattern.venue_downgrade_count,
            'total_venues_tried': pattern.total_venues_tried,
            'time_span_days': pattern.time_span_days,
            'final_acceptance_venue': pattern.final_acceptance_venue,
            'shopping_score': pattern.shopping_score
        }
    
    def _dict_to_pattern(self, data: Dict) -> VenueShoppingPattern:
        """Convert dictionary to shopping pattern."""
        return VenueShoppingPattern(
            paper_id=data['paper_id'],
            researcher_id=data['researcher_id'],
            submission_sequence=[self._dict_to_submission(s) for s in data['submission_sequence']],
            venue_downgrade_count=data['venue_downgrade_count'],
            total_venues_tried=data['total_venues_tried'],
            time_span_days=data['time_span_days'],
            final_acceptance_venue=data['final_acceptance_venue'],
            shopping_score=data['shopping_score']
        )
    
    def _profile_to_dict(self, profile: ResearcherShoppingProfile) -> Dict:
        """Convert researcher profile to dictionary."""
        return {
            'researcher_id': profile.researcher_id,
            'total_papers_submitted': profile.total_papers_submitted,
            'papers_with_shopping': profile.papers_with_shopping,
            'average_venues_per_paper': profile.average_venues_per_paper,
            'average_shopping_score': profile.average_shopping_score,
            'preferred_venue_types': [vt.value for vt in profile.preferred_venue_types],
            'shopping_tendency': profile.shopping_tendency
        }
    
    def _dict_to_profile(self, data: Dict) -> ResearcherShoppingProfile:
        """Convert dictionary to researcher profile."""
        return ResearcherShoppingProfile(
            researcher_id=data['researcher_id'],
            total_papers_submitted=data['total_papers_submitted'],
            papers_with_shopping=data['papers_with_shopping'],
            average_venues_per_paper=data['average_venues_per_paper'],
            average_shopping_score=data['average_shopping_score'],
            preferred_venue_types=[VenueType(vt) for vt in data['preferred_venue_types']],
            shopping_tendency=data['shopping_tendency']
        )