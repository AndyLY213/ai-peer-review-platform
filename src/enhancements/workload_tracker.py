"""
Workload Tracker System

This module implements the WorkloadTracker class to monitor reviewer capacity,
availability status checking based on current workload, logic for maximum reviews
per month (2-8 based on seniority), and comprehensive availability tracking functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger
from src.data.enhanced_models import ResearcherLevel, EnhancedResearcher


logger = get_logger(__name__)


class AvailabilityStatus(Enum):
    """Reviewer availability status."""
    AVAILABLE = "Available"
    BUSY = "Busy"
    OVERLOADED = "Overloaded"
    UNAVAILABLE = "Unavailable"
    ON_SABBATICAL = "On Sabbatical"


@dataclass
class ReviewAssignment:
    """Represents a review assignment for workload tracking."""
    assignment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_id: str = ""
    paper_id: str = ""
    venue_id: str = ""
    assigned_date: datetime = field(default_factory=datetime.now)
    deadline: datetime = field(default_factory=datetime.now)
    estimated_hours: int = 8  # Estimated hours to complete review
    actual_hours: Optional[int] = None
    is_completed: bool = False
    completion_date: Optional[datetime] = None
    
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """Check if assignment is currently active (not completed and not overdue)."""
        if self.is_completed:
            return False
        
        if current_time is None:
            current_time = datetime.now()
        
        # Consider assignment inactive if significantly overdue (>2 weeks)
        return current_time <= self.deadline + timedelta(weeks=2)
    
    def days_remaining(self, current_time: Optional[datetime] = None) -> int:
        """Calculate days remaining until deadline."""
        if current_time is None:
            current_time = datetime.now()
        
        delta = self.deadline - current_time
        return max(0, delta.days)


@dataclass
class WorkloadConfiguration:
    """Configuration for workload limits and calculations."""
    # Maximum reviews per month by seniority level
    max_reviews_per_month: Dict[ResearcherLevel, int] = field(default_factory=lambda: {
        ResearcherLevel.GRADUATE_STUDENT: 2,
        ResearcherLevel.POSTDOC: 3,
        ResearcherLevel.ASSISTANT_PROF: 4,
        ResearcherLevel.ASSOCIATE_PROF: 6,
        ResearcherLevel.FULL_PROF: 8,
        ResearcherLevel.EMERITUS: 3
    })
    
    # Estimated hours per review by venue type (can be customized)
    base_hours_per_review: int = 8
    venue_hour_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'TOP_CONFERENCE': 1.5,
        'MID_CONFERENCE': 1.0,
        'LOW_CONFERENCE': 0.8,
        'TOP_JOURNAL': 2.0,
        'SPECIALIZED_JOURNAL': 1.3,
        'GENERAL_JOURNAL': 1.0,
        'WORKSHOP': 0.5,
        'PREPRINT': 0.3
    })
    
    # Workload thresholds
    busy_threshold: float = 0.7  # 70% of capacity
    overloaded_threshold: float = 1.0  # 100% of capacity
    
    def get_max_reviews(self, level: ResearcherLevel) -> int:
        """Get maximum reviews per month for a seniority level."""
        return self.max_reviews_per_month.get(level, 4)
    
    def get_estimated_hours(self, venue_type: str) -> int:
        """Get estimated hours for a review based on venue type."""
        multiplier = self.venue_hour_multipliers.get(venue_type, 1.0)
        return int(self.base_hours_per_review * multiplier)


@dataclass
class ReviewerWorkload:
    """Represents current workload for a reviewer."""
    reviewer_id: str
    current_assignments: List[ReviewAssignment] = field(default_factory=list)
    completed_this_month: int = 0
    total_hours_this_month: int = 0
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE
    unavailable_until: Optional[datetime] = None
    sabbatical_period: Optional[Tuple[datetime, datetime]] = None
    custom_max_reviews: Optional[int] = None  # Override default limits
    
    def get_active_assignments(self, current_time: Optional[datetime] = None) -> List[ReviewAssignment]:
        """Get currently active assignments."""
        return [a for a in self.current_assignments if a.is_active(current_time)]
    
    def get_total_estimated_hours(self, current_time: Optional[datetime] = None) -> int:
        """Get total estimated hours for active assignments."""
        active_assignments = self.get_active_assignments(current_time)
        return sum(a.estimated_hours for a in active_assignments)
    
    def get_workload_ratio(self, max_reviews: int, current_time: Optional[datetime] = None) -> float:
        """Calculate current workload as ratio of maximum capacity."""
        active_count = len(self.get_active_assignments(current_time))
        return active_count / max_reviews if max_reviews > 0 else 0.0


class WorkloadTracker:
    """
    Tracks reviewer workload, availability, and capacity to manage review assignments
    based on seniority levels and current commitments.
    """
    
    def __init__(self, config: Optional[WorkloadConfiguration] = None):
        """Initialize workload tracker."""
        self.config = config or WorkloadConfiguration()
        self.reviewer_workloads: Dict[str, ReviewerWorkload] = {}
        logger.info("WorkloadTracker initialized with seniority-based limits")
    
    def register_reviewer(
        self, 
        researcher: EnhancedResearcher,
        custom_max_reviews: Optional[int] = None
    ):
        """
        Register a reviewer for workload tracking.
        
        Args:
            researcher: The researcher to register
            custom_max_reviews: Custom maximum reviews override
        """
        if researcher.id not in self.reviewer_workloads:
            workload = ReviewerWorkload(
                reviewer_id=researcher.id,
                custom_max_reviews=custom_max_reviews
            )
            self.reviewer_workloads[researcher.id] = workload
            logger.info(f"Registered reviewer {researcher.id} for workload tracking")
    
    def assign_review(
        self,
        reviewer_id: str,
        paper_id: str,
        venue_id: str,
        venue_type: str,
        deadline: datetime,
        assignment_date: Optional[datetime] = None
    ) -> ReviewAssignment:
        """
        Assign a review to a reviewer and update their workload.
        
        Args:
            reviewer_id: ID of the reviewer
            paper_id: ID of the paper
            venue_id: ID of the venue
            venue_type: Type of venue for hour estimation
            deadline: Review deadline
            assignment_date: Date of assignment (defaults to now)
        
        Returns:
            ReviewAssignment object
        """
        if assignment_date is None:
            assignment_date = datetime.now()
        
        if reviewer_id not in self.reviewer_workloads:
            raise ValidationError("reviewer_id", reviewer_id, "registered reviewer")
        
        estimated_hours = self.config.get_estimated_hours(venue_type)
        
        assignment = ReviewAssignment(
            reviewer_id=reviewer_id,
            paper_id=paper_id,
            venue_id=venue_id,
            assigned_date=assignment_date,
            deadline=deadline,
            estimated_hours=estimated_hours
        )
        
        workload = self.reviewer_workloads[reviewer_id]
        workload.current_assignments.append(assignment)
        
        # Update availability status
        self._update_availability_status(reviewer_id)
        
        logger.info(f"Assigned review to {reviewer_id}: paper {paper_id}, "
                   f"estimated {estimated_hours} hours, due {deadline.strftime('%Y-%m-%d')}")
        
        return assignment
    
    def complete_review(
        self,
        reviewer_id: str,
        assignment_id: str,
        actual_hours: Optional[int] = None,
        completion_date: Optional[datetime] = None
    ):
        """
        Mark a review as completed and update workload statistics.
        
        Args:
            reviewer_id: ID of the reviewer
            assignment_id: ID of the assignment
            actual_hours: Actual hours spent (optional)
            completion_date: Date of completion (defaults to now)
        """
        if reviewer_id not in self.reviewer_workloads:
            raise ValidationError("reviewer_id", reviewer_id, "registered reviewer")
        
        if completion_date is None:
            completion_date = datetime.now()
        
        workload = self.reviewer_workloads[reviewer_id]
        assignment = None
        
        for a in workload.current_assignments:
            if a.assignment_id == assignment_id:
                assignment = a
                break
        
        if assignment is None:
            raise ValidationError("assignment_id", assignment_id, "valid assignment")
        
        assignment.is_completed = True
        assignment.completion_date = completion_date
        if actual_hours is not None:
            assignment.actual_hours = actual_hours
        
        # Update monthly statistics if completed in current month
        current_month = completion_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        assignment_month = assignment.assigned_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if current_month == assignment_month:
            workload.completed_this_month += 1
            hours_to_add = actual_hours if actual_hours is not None else assignment.estimated_hours
            workload.total_hours_this_month += hours_to_add
        
        # Update availability status
        self._update_availability_status(reviewer_id)
        
        logger.info(f"Completed review for {reviewer_id}: assignment {assignment_id}, "
                   f"actual hours: {actual_hours or 'not specified'}")
    
    def check_availability(
        self, 
        reviewer_id: str, 
        researcher: EnhancedResearcher,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, AvailabilityStatus, str]:
        """
        Check if a reviewer is available for new assignments.
        
        Args:
            reviewer_id: ID of the reviewer
            researcher: Researcher object for seniority info
            current_time: Current time (defaults to now)
        
        Returns:
            Tuple of (is_available, status, reason)
        """
        if current_time is None:
            current_time = datetime.now()
        
        if reviewer_id not in self.reviewer_workloads:
            # Auto-register if not found, but don't override existing status
            self.register_reviewer(researcher)
        
        workload = self.reviewer_workloads[reviewer_id]
        
        # Check explicit unavailability
        if workload.availability_status == AvailabilityStatus.UNAVAILABLE:
            if workload.unavailable_until:
                if current_time < workload.unavailable_until:
                    return False, AvailabilityStatus.UNAVAILABLE, f"Unavailable until {workload.unavailable_until.strftime('%Y-%m-%d')}"
                else:
                    # Unavailable period has passed, make available again
                    workload.availability_status = AvailabilityStatus.AVAILABLE
                    workload.unavailable_until = None
            else:
                # Indefinitely unavailable
                return False, AvailabilityStatus.UNAVAILABLE, "Unavailable indefinitely"
        
        # Check sabbatical
        if workload.sabbatical_period:
            start, end = workload.sabbatical_period
            if start <= current_time <= end:
                return False, AvailabilityStatus.ON_SABBATICAL, f"On sabbatical until {end.strftime('%Y-%m-%d')}"
        
        # Check workload capacity
        max_reviews = workload.custom_max_reviews or self.config.get_max_reviews(researcher.level)
        active_assignments = workload.get_active_assignments(current_time)
        current_load = len(active_assignments)
        
        if current_load >= max_reviews:
            return False, AvailabilityStatus.OVERLOADED, f"At capacity ({current_load}/{max_reviews} reviews)"
        
        workload_ratio = workload.get_workload_ratio(max_reviews, current_time)
        
        if workload_ratio >= self.config.overloaded_threshold:
            return False, AvailabilityStatus.OVERLOADED, f"Overloaded ({workload_ratio:.1%} capacity)"
        elif workload_ratio >= self.config.busy_threshold:
            return True, AvailabilityStatus.BUSY, f"Busy but available ({workload_ratio:.1%} capacity)"
        else:
            return True, AvailabilityStatus.AVAILABLE, f"Available ({workload_ratio:.1%} capacity)"
    
    def get_reviewer_capacity(
        self, 
        reviewer_id: str, 
        researcher: EnhancedResearcher,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get detailed capacity information for a reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
            researcher: Researcher object for seniority info
            current_time: Current time (defaults to now)
        
        Returns:
            Dictionary with capacity information
        """
        if current_time is None:
            current_time = datetime.now()
        
        if reviewer_id not in self.reviewer_workloads:
            self.register_reviewer(researcher)
        
        workload = self.reviewer_workloads[reviewer_id]
        max_reviews = workload.custom_max_reviews or self.config.get_max_reviews(researcher.level)
        active_assignments = workload.get_active_assignments(current_time)
        current_load = len(active_assignments)
        workload_ratio = workload.get_workload_ratio(max_reviews, current_time)
        
        is_available, status, reason = self.check_availability(reviewer_id, researcher, current_time)
        
        return {
            'reviewer_id': reviewer_id,
            'seniority_level': researcher.level.value,
            'max_reviews_per_month': max_reviews,
            'current_active_reviews': current_load,
            'completed_this_month': workload.completed_this_month,
            'workload_ratio': workload_ratio,
            'availability_status': status.value,
            'is_available': is_available,
            'reason': reason,
            'total_estimated_hours': workload.get_total_estimated_hours(current_time),
            'total_hours_this_month': workload.total_hours_this_month,
            'active_assignments': [
                {
                    'assignment_id': a.assignment_id,
                    'paper_id': a.paper_id,
                    'venue_id': a.venue_id,
                    'deadline': a.deadline.isoformat(),
                    'days_remaining': a.days_remaining(current_time),
                    'estimated_hours': a.estimated_hours
                }
                for a in active_assignments
            ]
        }
    
    def set_reviewer_unavailable(
        self,
        reviewer_id: str,
        until_date: Optional[datetime] = None,
        reason: str = "Personal reasons"
    ):
        """
        Mark a reviewer as unavailable.
        
        Args:
            reviewer_id: ID of the reviewer
            until_date: Date until which reviewer is unavailable
            reason: Reason for unavailability
        """
        if reviewer_id not in self.reviewer_workloads:
            raise ValidationError("reviewer_id", reviewer_id, "registered reviewer")
        
        workload = self.reviewer_workloads[reviewer_id]
        workload.availability_status = AvailabilityStatus.UNAVAILABLE
        workload.unavailable_until = until_date
        
        logger.info(f"Set reviewer {reviewer_id} as unavailable until {until_date or 'indefinitely'}: {reason}")
    
    def set_reviewer_sabbatical(
        self,
        reviewer_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Mark a reviewer as on sabbatical.
        
        Args:
            reviewer_id: ID of the reviewer
            start_date: Start date of sabbatical
            end_date: End date of sabbatical
        """
        if reviewer_id not in self.reviewer_workloads:
            raise ValidationError("reviewer_id", reviewer_id, "registered reviewer")
        
        workload = self.reviewer_workloads[reviewer_id]
        workload.sabbatical_period = (start_date, end_date)
        
        logger.info(f"Set reviewer {reviewer_id} on sabbatical from {start_date.strftime('%Y-%m-%d')} "
                   f"to {end_date.strftime('%Y-%m-%d')}")
    
    def set_reviewer_available(self, reviewer_id: str):
        """
        Mark a reviewer as available again.
        
        Args:
            reviewer_id: ID of the reviewer
        """
        if reviewer_id not in self.reviewer_workloads:
            raise ValidationError("reviewer_id", reviewer_id, "registered reviewer")
        
        workload = self.reviewer_workloads[reviewer_id]
        workload.availability_status = AvailabilityStatus.AVAILABLE
        workload.unavailable_until = None
        workload.sabbatical_period = None
        
        logger.info(f"Set reviewer {reviewer_id} as available")
    
    def get_available_reviewers(
        self,
        researchers: List[EnhancedResearcher],
        min_capacity: float = 0.0,
        current_time: Optional[datetime] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get list of available reviewers with their capacity information.
        
        Args:
            researchers: List of researchers to check
            min_capacity: Minimum available capacity (0.0-1.0)
            current_time: Current time (defaults to now)
        
        Returns:
            List of tuples (reviewer_id, capacity_info)
        """
        available_reviewers = []
        
        for researcher in researchers:
            is_available, status, reason = self.check_availability(
                researcher.id, researcher, current_time
            )
            
            if is_available:
                capacity_info = self.get_reviewer_capacity(
                    researcher.id, researcher, current_time
                )
                
                # Check if meets minimum capacity requirement
                available_capacity = 1.0 - capacity_info['workload_ratio']
                if available_capacity >= min_capacity:
                    available_reviewers.append((researcher.id, capacity_info))
        
        # Sort by available capacity (most available first)
        available_reviewers.sort(key=lambda x: x[1]['workload_ratio'])
        
        return available_reviewers
    
    def _update_availability_status(self, reviewer_id: str):
        """Update availability status based on current workload."""
        if reviewer_id not in self.reviewer_workloads:
            return
        
        workload = self.reviewer_workloads[reviewer_id]
        
        # Don't override explicit unavailability or sabbatical
        if workload.availability_status in [AvailabilityStatus.UNAVAILABLE, AvailabilityStatus.ON_SABBATICAL]:
            return
        
        # Get max reviews (need researcher info, so we'll use a default approach)
        # This is a simplified update - full status should be checked with researcher info
        active_count = len(workload.get_active_assignments())
        
        # Use a reasonable default max (will be properly calculated when check_availability is called)
        estimated_max = 4  # Default for assistant professor level
        workload_ratio = active_count / estimated_max
        
        if workload_ratio >= self.config.overloaded_threshold:
            workload.availability_status = AvailabilityStatus.OVERLOADED
        elif workload_ratio >= self.config.busy_threshold:
            workload.availability_status = AvailabilityStatus.BUSY
        else:
            workload.availability_status = AvailabilityStatus.AVAILABLE
    
    def reset_monthly_stats(self, current_time: Optional[datetime] = None):
        """
        Reset monthly statistics for all reviewers.
        
        Args:
            current_time: Current time (defaults to now)
        """
        if current_time is None:
            current_time = datetime.now()
        
        for workload in self.reviewer_workloads.values():
            workload.completed_this_month = 0
            workload.total_hours_this_month = 0
        
        logger.info(f"Reset monthly statistics for all reviewers at {current_time.strftime('%Y-%m-%d')}")
    
    def get_workload_statistics(self) -> Dict[str, Any]:
        """
        Get overall workload statistics across all reviewers.
        
        Returns:
            Dictionary with workload statistics
        """
        total_reviewers = len(self.reviewer_workloads)
        if total_reviewers == 0:
            return {
                'total_reviewers': 0,
                'availability_breakdown': {},
                'average_workload_ratio': 0.0,
                'total_active_assignments': 0,
                'total_completed_this_month': 0
            }
        
        availability_counts = {}
        total_workload_ratio = 0.0
        total_active_assignments = 0
        total_completed_this_month = 0
        
        for workload in self.reviewer_workloads.values():
            status = workload.availability_status.value
            availability_counts[status] = availability_counts.get(status, 0) + 1
            
            active_assignments = workload.get_active_assignments()
            total_active_assignments += len(active_assignments)
            total_completed_this_month += workload.completed_this_month
            
            # Estimate workload ratio (simplified without researcher info)
            estimated_max = 4
            workload_ratio = len(active_assignments) / estimated_max
            total_workload_ratio += workload_ratio
        
        return {
            'total_reviewers': total_reviewers,
            'availability_breakdown': availability_counts,
            'average_workload_ratio': total_workload_ratio / total_reviewers,
            'total_active_assignments': total_active_assignments,
            'total_completed_this_month': total_completed_this_month
        }
    
    def update_max_reviews_for_level(self, level: ResearcherLevel, max_reviews: int):
        """
        Update maximum reviews per month for a seniority level.
        
        Args:
            level: Seniority level
            max_reviews: New maximum reviews per month
        """
        if not (2 <= max_reviews <= 8):
            raise ValidationError("max_reviews", max_reviews, "integer between 2 and 8")
        
        self.config.max_reviews_per_month[level] = max_reviews
        logger.info(f"Updated max reviews for {level.value} to {max_reviews} per month")
    
    def cleanup_old_assignments(self, days_old: int = 90):
        """
        Clean up old completed assignments.
        
        Args:
            days_old: Remove assignments completed more than this many days ago
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        total_removed = 0
        
        for workload in self.reviewer_workloads.values():
            old_assignments = [
                a for a in workload.current_assignments
                if a.is_completed and a.completion_date and a.completion_date < cutoff_date
            ]
            
            for assignment in old_assignments:
                workload.current_assignments.remove(assignment)
                total_removed += 1
        
        logger.info(f"Cleaned up {total_removed} old assignments older than {days_old} days")