"""
Deadline Management System

This module implements the DeadlineManager class with venue-specific deadlines (2-8 weeks),
logic to track review submission timing, penalty system for late review submissions,
and comprehensive deadline management functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.core.exceptions import ValidationError, DeadlineError
from src.core.logging_config import get_logger
from src.data.enhanced_models import VenueType, EnhancedVenue, StructuredReview, EnhancedResearcher


logger = get_logger(__name__)


class DeadlineStatus(Enum):
    """Status of deadline compliance."""
    ON_TIME = "On Time"
    LATE = "Late"
    VERY_LATE = "Very Late"
    MISSED = "Missed"


@dataclass
class DeadlineAssignment:
    """Represents a deadline assignment for a review."""
    assignment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_id: str = ""
    paper_id: str = ""
    venue_id: str = ""
    assigned_date: datetime = field(default_factory=datetime.now)
    deadline: datetime = field(default_factory=datetime.now)
    submission_date: Optional[datetime] = None
    status: DeadlineStatus = DeadlineStatus.ON_TIME
    days_late: int = 0
    penalty_applied: float = 0.0
    reminder_sent: bool = False
    
    def __post_init__(self):
        """Calculate status and days late if submitted."""
        if self.submission_date:
            self._calculate_status()
    
    def _calculate_status(self):
        """Calculate deadline status based on submission date."""
        if not self.submission_date:
            return
        
        if self.submission_date <= self.deadline:
            self.status = DeadlineStatus.ON_TIME
            self.days_late = 0
        else:
            delta = self.submission_date - self.deadline
            self.days_late = delta.days
            
            if self.days_late <= 3:
                self.status = DeadlineStatus.LATE
            elif self.days_late <= 7:
                self.status = DeadlineStatus.VERY_LATE
            else:
                self.status = DeadlineStatus.MISSED
    
    def is_overdue(self, current_time: Optional[datetime] = None) -> bool:
        """Check if deadline is overdue."""
        if current_time is None:
            current_time = datetime.now()
        return current_time > self.deadline and self.submission_date is None
    
    def days_until_deadline(self, current_time: Optional[datetime] = None) -> int:
        """Calculate days until deadline."""
        if current_time is None:
            current_time = datetime.now()
        delta = self.deadline - current_time
        return max(0, delta.days)


@dataclass
class PenaltyConfiguration:
    """Configuration for late submission penalties."""
    base_penalty_per_day: float = 0.05  # 5% penalty per day late
    max_penalty: float = 0.5  # Maximum 50% penalty
    grace_period_days: int = 1  # 1 day grace period
    escalation_factor: float = 1.5  # Penalty escalation for very late submissions
    
    def calculate_penalty(self, days_late: int) -> float:
        """Calculate penalty based on days late."""
        if days_late <= self.grace_period_days:
            return 0.0
        
        effective_days = days_late - self.grace_period_days
        base_penalty = effective_days * self.base_penalty_per_day
        
        # Apply escalation for very late submissions (>7 days)
        if days_late > 7:
            escalation_days = days_late - 7
            base_penalty += escalation_days * self.base_penalty_per_day * self.escalation_factor
        
        return min(base_penalty, self.max_penalty)


class DeadlineManager:
    """
    Manages review deadlines with venue-specific timing, tracks submission compliance,
    and applies penalties for late submissions.
    """
    
    def __init__(self, penalty_config: Optional[PenaltyConfiguration] = None):
        """Initialize deadline manager."""
        self.penalty_config = penalty_config or PenaltyConfiguration()
        self.deadline_assignments: Dict[str, DeadlineAssignment] = {}
        self.venue_deadline_weeks: Dict[VenueType, int] = {
            VenueType.TOP_CONFERENCE: 6,
            VenueType.MID_CONFERENCE: 4,
            VenueType.LOW_CONFERENCE: 3,
            VenueType.TOP_JOURNAL: 8,
            VenueType.SPECIALIZED_JOURNAL: 6,
            VenueType.GENERAL_JOURNAL: 4,
            VenueType.WORKSHOP: 2,
            VenueType.PREPRINT: 2
        }
        logger.info("DeadlineManager initialized with venue-specific deadlines")
    
    def assign_review_deadline(
        self, 
        reviewer_id: str, 
        paper_id: str, 
        venue: EnhancedVenue,
        assignment_date: Optional[datetime] = None
    ) -> DeadlineAssignment:
        """
        Assign a review deadline based on venue-specific requirements.
        
        Args:
            reviewer_id: ID of the reviewer
            paper_id: ID of the paper to review
            venue: Venue information
            assignment_date: Date of assignment (defaults to now)
        
        Returns:
            DeadlineAssignment object
        """
        if assignment_date is None:
            assignment_date = datetime.now()
        
        # Get venue-specific deadline weeks
        deadline_weeks = self.get_venue_deadline_weeks(venue.venue_type)
        deadline = assignment_date + timedelta(weeks=deadline_weeks)
        
        assignment = DeadlineAssignment(
            reviewer_id=reviewer_id,
            paper_id=paper_id,
            venue_id=venue.id,
            assigned_date=assignment_date,
            deadline=deadline
        )
        
        self.deadline_assignments[assignment.assignment_id] = assignment
        
        logger.info(f"Assigned review deadline for reviewer {reviewer_id}, paper {paper_id}, "
                   f"venue {venue.name} ({deadline_weeks} weeks, due {deadline.strftime('%Y-%m-%d')})")
        
        return assignment
    
    def get_venue_deadline_weeks(self, venue_type: VenueType) -> int:
        """
        Get deadline weeks for a venue type.
        
        Args:
            venue_type: Type of venue
        
        Returns:
            Number of weeks for deadline
        """
        return self.venue_deadline_weeks.get(venue_type, 4)  # Default 4 weeks
    
    def submit_review(
        self, 
        assignment_id: str, 
        review: StructuredReview,
        submission_time: Optional[datetime] = None
    ) -> Tuple[bool, float]:
        """
        Record review submission and calculate any penalties.
        
        Args:
            assignment_id: ID of the deadline assignment
            review: The submitted review
            submission_time: Time of submission (defaults to now)
        
        Returns:
            Tuple of (is_on_time, penalty_applied)
        """
        if assignment_id not in self.deadline_assignments:
            raise ValidationError("assignment_id", assignment_id, "valid assignment ID")
        
        if submission_time is None:
            submission_time = datetime.now()
        
        assignment = self.deadline_assignments[assignment_id]
        assignment.submission_date = submission_time
        assignment._calculate_status()
        
        # Calculate penalty
        penalty = 0.0
        if assignment.days_late > 0:
            penalty = self.penalty_config.calculate_penalty(assignment.days_late)
            assignment.penalty_applied = penalty
            
            # Update review with late submission info
            review.is_late = True
            review.submission_timestamp = submission_time
            
            logger.warning(f"Late review submission: reviewer {assignment.reviewer_id}, "
                         f"{assignment.days_late} days late, penalty: {penalty:.2%}")
        else:
            logger.info(f"On-time review submission: reviewer {assignment.reviewer_id}")
        
        return assignment.status == DeadlineStatus.ON_TIME, penalty
    
    def get_overdue_assignments(self, current_time: Optional[datetime] = None) -> List[DeadlineAssignment]:
        """
        Get all overdue assignments.
        
        Args:
            current_time: Current time (defaults to now)
        
        Returns:
            List of overdue assignments
        """
        if current_time is None:
            current_time = datetime.now()
        
        overdue = []
        for assignment in self.deadline_assignments.values():
            if assignment.is_overdue(current_time):
                overdue.append(assignment)
        
        return overdue
    
    def get_upcoming_deadlines(
        self, 
        days_ahead: int = 7,
        current_time: Optional[datetime] = None
    ) -> List[DeadlineAssignment]:
        """
        Get assignments with deadlines in the next N days.
        
        Args:
            days_ahead: Number of days to look ahead
            current_time: Current time (defaults to now)
        
        Returns:
            List of upcoming deadline assignments
        """
        if current_time is None:
            current_time = datetime.now()
        
        cutoff_time = current_time + timedelta(days=days_ahead)
        upcoming = []
        
        for assignment in self.deadline_assignments.values():
            if (assignment.submission_date is None and 
                current_time <= assignment.deadline <= cutoff_time):
                upcoming.append(assignment)
        
        return sorted(upcoming, key=lambda x: x.deadline)
    
    def apply_reliability_penalty(
        self, 
        reviewer_id: str, 
        researcher: EnhancedResearcher
    ) -> float:
        """
        Apply reliability penalty to researcher based on deadline compliance history.
        
        Args:
            reviewer_id: ID of the reviewer
            researcher: Researcher object to update
        
        Returns:
            Updated reliability score
        """
        # Get all assignments for this reviewer
        reviewer_assignments = [
            a for a in self.deadline_assignments.values() 
            if a.reviewer_id == reviewer_id and a.submission_date is not None
        ]
        
        if not reviewer_assignments:
            return 1.0  # No history, perfect reliability
        
        # Calculate reliability based on on-time submissions
        on_time_count = sum(1 for a in reviewer_assignments if a.status == DeadlineStatus.ON_TIME)
        total_count = len(reviewer_assignments)
        base_reliability = on_time_count / total_count
        
        # Apply additional penalties for very late submissions
        very_late_count = sum(1 for a in reviewer_assignments 
                             if a.status in [DeadlineStatus.VERY_LATE, DeadlineStatus.MISSED])
        very_late_penalty = very_late_count * 0.1  # 10% penalty per very late submission
        
        reliability_score = max(0.0, base_reliability - very_late_penalty)
        
        # Update researcher's review quality history with reliability metric
        from src.data.enhanced_models import ReviewQualityMetric
        reliability_metric = ReviewQualityMetric(
            review_id=f"reliability_{datetime.now().isoformat()}",
            quality_score=reliability_score,
            timeliness_score=base_reliability,
            timestamp=datetime.now()
        )
        researcher.review_quality_history.append(reliability_metric)
        
        logger.info(f"Updated reliability score for reviewer {reviewer_id}: {reliability_score:.2f}")
        return reliability_score
    
    def get_reviewer_deadline_stats(self, reviewer_id: str) -> Dict[str, Any]:
        """
        Get deadline compliance statistics for a reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
        
        Returns:
            Dictionary with deadline statistics
        """
        reviewer_assignments = [
            a for a in self.deadline_assignments.values() 
            if a.reviewer_id == reviewer_id and a.submission_date is not None
        ]
        
        if not reviewer_assignments:
            return {
                'total_assignments': 0,
                'on_time_rate': 1.0,
                'average_days_late': 0.0,
                'total_penalties': 0.0,
                'status_breakdown': {}
            }
        
        total_assignments = len(reviewer_assignments)
        on_time_count = sum(1 for a in reviewer_assignments if a.status == DeadlineStatus.ON_TIME)
        total_days_late = sum(a.days_late for a in reviewer_assignments)
        total_penalties = sum(a.penalty_applied for a in reviewer_assignments)
        
        # Status breakdown
        status_counts = {}
        for status in DeadlineStatus:
            status_counts[status.value] = sum(1 for a in reviewer_assignments if a.status == status)
        
        return {
            'total_assignments': total_assignments,
            'on_time_rate': on_time_count / total_assignments,
            'average_days_late': total_days_late / total_assignments,
            'total_penalties': total_penalties,
            'status_breakdown': status_counts
        }
    
    def send_deadline_reminders(
        self, 
        days_before: int = 3,
        current_time: Optional[datetime] = None
    ) -> List[str]:
        """
        Identify assignments that need deadline reminders.
        
        Args:
            days_before: Send reminder N days before deadline
            current_time: Current time (defaults to now)
        
        Returns:
            List of reviewer IDs who need reminders
        """
        if current_time is None:
            current_time = datetime.now()
        
        reminder_time = current_time + timedelta(days=days_before)
        reminder_needed = []
        
        for assignment in self.deadline_assignments.values():
            if (assignment.submission_date is None and 
                not assignment.reminder_sent and
                assignment.deadline <= reminder_time):
                
                reminder_needed.append(assignment.reviewer_id)
                assignment.reminder_sent = True
                
                logger.info(f"Reminder needed for reviewer {assignment.reviewer_id}, "
                           f"deadline in {assignment.days_until_deadline(current_time)} days")
        
        return reminder_needed
    
    def get_venue_deadline_performance(self, venue_id: str) -> Dict[str, Any]:
        """
        Get deadline performance statistics for a venue.
        
        Args:
            venue_id: ID of the venue
        
        Returns:
            Dictionary with venue deadline performance
        """
        venue_assignments = [
            a for a in self.deadline_assignments.values() 
            if a.venue_id == venue_id and a.submission_date is not None
        ]
        
        if not venue_assignments:
            return {
                'total_reviews': 0,
                'on_time_rate': 1.0,
                'average_days_late': 0.0,
                'status_breakdown': {}
            }
        
        total_reviews = len(venue_assignments)
        on_time_count = sum(1 for a in venue_assignments if a.status == DeadlineStatus.ON_TIME)
        total_days_late = sum(a.days_late for a in venue_assignments)
        
        # Status breakdown
        status_counts = {}
        for status in DeadlineStatus:
            status_counts[status.value] = sum(1 for a in venue_assignments if a.status == status)
        
        return {
            'total_reviews': total_reviews,
            'on_time_rate': on_time_count / total_reviews,
            'average_days_late': total_days_late / total_reviews,
            'status_breakdown': status_counts
        }
    
    def update_venue_deadline_weeks(self, venue_type: VenueType, weeks: int):
        """
        Update deadline weeks for a venue type.
        
        Args:
            venue_type: Type of venue
            weeks: Number of weeks for deadline
        """
        if not (2 <= weeks <= 8):
            raise ValidationError("weeks", weeks, "integer between 2 and 8")
        
        self.venue_deadline_weeks[venue_type] = weeks
        logger.info(f"Updated deadline for {venue_type.value} to {weeks} weeks")
    
    def clear_old_assignments(self, days_old: int = 365):
        """
        Clear assignments older than specified days.
        
        Args:
            days_old: Remove assignments older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        old_assignments = [
            assignment_id for assignment_id, assignment in self.deadline_assignments.items()
            if assignment.assigned_date < cutoff_date
        ]
        
        for assignment_id in old_assignments:
            del self.deadline_assignments[assignment_id]
        
        logger.info(f"Cleared {len(old_assignments)} old deadline assignments")