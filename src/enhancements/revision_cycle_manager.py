"""
Revision Cycle Manager System

This module implements the RevisionCycleManager for handling revision cycles,
logic to manage re-review processes with updated deadlines, workflow state management
for multi-round reviews, and comprehensive revision cycle management functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger
from src.data.enhanced_models import ReviewDecision, StructuredReview, EnhancedVenue


logger = get_logger(__name__)


class RevisionStatus(Enum):
    """Status of a revision cycle."""
    INITIAL_REVIEW = "Initial Review"
    REVISION_REQUESTED = "Revision Requested"
    REVISION_SUBMITTED = "Revision Submitted"
    RE_REVIEW_IN_PROGRESS = "Re-review In Progress"
    FINAL_DECISION = "Final Decision"
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"


class WorkflowState(Enum):
    """Workflow state for multi-round reviews."""
    SUBMITTED = "Submitted"
    UNDER_REVIEW = "Under Review"
    AWAITING_REVISION = "Awaiting Revision"
    REVISION_UNDER_REVIEW = "Revision Under Review"
    DECISION_PENDING = "Decision Pending"
    COMPLETED = "Completed"


@dataclass
class RevisionRound:
    """Represents a single round of review/revision."""
    round_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    round_number: int = 1
    submission_date: datetime = field(default_factory=datetime.now)
    review_deadline: Optional[datetime] = None
    revision_deadline: Optional[datetime] = None
    reviews: List[StructuredReview] = field(default_factory=list)
    decision: Optional[ReviewDecision] = None
    decision_date: Optional[datetime] = None
    revision_submitted: bool = False
    revision_submission_date: Optional[datetime] = None
    reviewer_assignments: Set[str] = field(default_factory=set)
    
    def is_review_complete(self) -> bool:
        """Check if all reviews for this round are complete."""
        return len(self.reviews) >= len(self.reviewer_assignments) and len(self.reviews) > 0
    
    def get_average_score(self) -> Optional[float]:
        """Get average score across all reviews in this round."""
        if not self.reviews:
            return None
        
        total_score = sum(review.criteria_scores.get_average_score() for review in self.reviews)
        return total_score / len(self.reviews)
    
    def requires_revision(self) -> bool:
        """Check if this round requires revision based on decision."""
        return self.decision in [ReviewDecision.MINOR_REVISION, ReviewDecision.MAJOR_REVISION]


@dataclass
class RevisionCycle:
    """Represents a complete revision cycle for a paper."""
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str = ""
    venue_id: str = ""
    author_ids: Set[str] = field(default_factory=set)
    
    # Workflow state
    current_status: RevisionStatus = RevisionStatus.INITIAL_REVIEW
    workflow_state: WorkflowState = WorkflowState.SUBMITTED
    
    # Revision rounds
    rounds: List[RevisionRound] = field(default_factory=list)
    current_round_number: int = 1
    max_revision_rounds: int = 2  # Maximum number of revision rounds allowed
    
    # Timeline
    initial_submission_date: datetime = field(default_factory=datetime.now)
    final_decision_date: Optional[datetime] = None
    total_review_time: Optional[timedelta] = None
    
    # Final outcome
    final_decision: Optional[ReviewDecision] = None
    acceptance_rate_impact: float = 0.0  # Impact on venue acceptance rate
    
    def get_current_round(self) -> Optional[RevisionRound]:
        """Get the current active round."""
        if not self.rounds:
            return None
        return self.rounds[-1]
    
    def get_total_rounds(self) -> int:
        """Get total number of rounds in this cycle."""
        return len(self.rounds)
    
    def is_complete(self) -> bool:
        """Check if the revision cycle is complete."""
        return self.current_status in [RevisionStatus.ACCEPTED, RevisionStatus.REJECTED, RevisionStatus.FINAL_DECISION]
    
    def can_request_revision(self) -> bool:
        """Check if another revision round can be requested."""
        return (self.get_total_rounds() < self.max_revision_rounds and 
                not self.is_complete())
    
    def get_total_review_duration(self) -> Optional[timedelta]:
        """Calculate total time from submission to final decision."""
        if self.final_decision_date:
            return self.final_decision_date - self.initial_submission_date
        return None


class RevisionCycleManager:
    """
    Manages multi-round review workflows with revision cycles, re-review processes,
    and comprehensive workflow state management.
    """
    
    def __init__(self):
        """Initialize revision cycle manager."""
        self.active_cycles: Dict[str, RevisionCycle] = {}
        self.completed_cycles: Dict[str, RevisionCycle] = {}
        
        # Default timeline configurations
        self.default_review_weeks = 4
        self.default_revision_weeks = 8
        self.revision_review_weeks = 3  # Shorter for re-reviews
        
        logger.info("RevisionCycleManager initialized")
    
    def start_review_cycle(
        self,
        paper_id: str,
        venue: EnhancedVenue,
        author_ids: Set[str],
        reviewer_ids: Set[str],
        submission_date: Optional[datetime] = None
    ) -> RevisionCycle:
        """
        Start a new review cycle for a paper.
        
        Args:
            paper_id: ID of the paper
            venue: Venue information
            author_ids: Set of author IDs
            reviewer_ids: Set of reviewer IDs
            submission_date: Date of submission (defaults to now)
        
        Returns:
            RevisionCycle object
        """
        if submission_date is None:
            submission_date = datetime.now()
        
        # Create revision cycle
        cycle = RevisionCycle(
            paper_id=paper_id,
            venue_id=venue.id,
            author_ids=author_ids,
            initial_submission_date=submission_date,
            max_revision_rounds=venue.revision_cycles_allowed
        )
        
        # Create initial review round
        review_deadline = submission_date + timedelta(weeks=venue.review_deadline_weeks)
        
        initial_round = RevisionRound(
            round_number=1,
            submission_date=submission_date,
            review_deadline=review_deadline,
            reviewer_assignments=reviewer_ids.copy()
        )
        
        cycle.rounds.append(initial_round)
        cycle.workflow_state = WorkflowState.UNDER_REVIEW
        
        self.active_cycles[cycle.cycle_id] = cycle
        
        logger.info(f"Started review cycle for paper {paper_id} at venue {venue.name}, "
                   f"cycle ID: {cycle.cycle_id}, {len(reviewer_ids)} reviewers assigned")
        
        return cycle
    
    def submit_review(
        self,
        cycle_id: str,
        review: StructuredReview
    ) -> bool:
        """
        Submit a review for the current round.
        
        Args:
            cycle_id: ID of the revision cycle
            review: The submitted review
        
        Returns:
            True if review was accepted, False otherwise
        """
        if cycle_id not in self.active_cycles:
            raise ValidationError("cycle_id", cycle_id, "active revision cycle")
        
        cycle = self.active_cycles[cycle_id]
        current_round = cycle.get_current_round()
        
        if current_round is None:
            raise ValidationError("cycle_state", "no current round", "active review round")
        
        # Check if reviewer is assigned to this round
        if review.reviewer_id not in current_round.reviewer_assignments:
            raise ValidationError("reviewer_id", review.reviewer_id, "assigned reviewer for this round")
        
        # Check if review already exists from this reviewer
        existing_review = next(
            (r for r in current_round.reviews if r.reviewer_id == review.reviewer_id),
            None
        )
        
        if existing_review:
            logger.warning(f"Replacing existing review from {review.reviewer_id} in cycle {cycle_id}")
            current_round.reviews.remove(existing_review)
        
        # Add review to current round
        review.revision_round = current_round.round_number
        current_round.reviews.append(review)
        
        logger.info(f"Submitted review from {review.reviewer_id} for cycle {cycle_id}, "
                   f"round {current_round.round_number}")
        
        # Check if all reviews are complete
        if current_round.is_review_complete():
            self._process_round_completion(cycle_id)
        
        return True
    
    def _process_round_completion(self, cycle_id: str):
        """Process completion of a review round and determine next steps."""
        cycle = self.active_cycles[cycle_id]
        current_round = cycle.get_current_round()
        
        if current_round is None:
            return
        
        # Calculate decision based on reviews
        decision = self._calculate_round_decision(current_round, cycle.venue_id)
        current_round.decision = decision
        current_round.decision_date = datetime.now()
        
        logger.info(f"Round {current_round.round_number} completed for cycle {cycle_id}, "
                   f"decision: {decision.value}")
        
        # Update cycle status based on decision
        if decision == ReviewDecision.ACCEPT:
            self._finalize_cycle(cycle_id, ReviewDecision.ACCEPT)
        elif decision == ReviewDecision.REJECT:
            self._finalize_cycle(cycle_id, ReviewDecision.REJECT)
        elif decision in [ReviewDecision.MINOR_REVISION, ReviewDecision.MAJOR_REVISION]:
            if cycle.can_request_revision():
                self._request_revision(cycle_id, decision)
            else:
                # No more revision rounds allowed, make final decision
                final_decision = ReviewDecision.REJECT if decision == ReviewDecision.MAJOR_REVISION else ReviewDecision.ACCEPT
                self._finalize_cycle(cycle_id, final_decision)
    
    def _calculate_round_decision(self, round_obj: RevisionRound, venue_id: str) -> ReviewDecision:
        """
        Calculate decision for a completed round based on reviews.
        
        Args:
            round_obj: The completed round
            venue_id: ID of the venue for decision thresholds
        
        Returns:
            ReviewDecision for the round
        """
        if not round_obj.reviews:
            return ReviewDecision.REJECT
        
        # Get average score
        avg_score = round_obj.get_average_score()
        if avg_score is None:
            return ReviewDecision.REJECT
        
        # Count recommendations
        recommendations = [review.recommendation for review in round_obj.reviews]
        accept_count = recommendations.count(ReviewDecision.ACCEPT)
        minor_revision_count = recommendations.count(ReviewDecision.MINOR_REVISION)
        major_revision_count = recommendations.count(ReviewDecision.MAJOR_REVISION)
        reject_count = recommendations.count(ReviewDecision.REJECT)
        
        total_reviews = len(recommendations)
        
        # Decision logic based on majority and score
        if accept_count >= total_reviews * 0.6:  # 60% accept
            return ReviewDecision.ACCEPT
        elif reject_count >= total_reviews * 0.6:  # 60% reject
            return ReviewDecision.REJECT
        elif (minor_revision_count + major_revision_count) >= total_reviews * 0.5:  # 50% revision
            if major_revision_count > minor_revision_count:
                return ReviewDecision.MAJOR_REVISION
            else:
                return ReviewDecision.MINOR_REVISION
        elif avg_score >= 7.0:  # High score threshold
            return ReviewDecision.ACCEPT
        elif avg_score <= 4.0:  # Low score threshold
            return ReviewDecision.REJECT
        else:
            # Default to revision for borderline cases
            return ReviewDecision.MINOR_REVISION
    
    def _request_revision(self, cycle_id: str, revision_type: ReviewDecision):
        """Request revision and set up next round."""
        cycle = self.active_cycles[cycle_id]
        
        # Update cycle status
        cycle.current_status = RevisionStatus.REVISION_REQUESTED
        cycle.workflow_state = WorkflowState.AWAITING_REVISION
        
        # Set revision deadline
        revision_weeks = self.default_revision_weeks
        if revision_type == ReviewDecision.MINOR_REVISION:
            revision_weeks = 6  # Shorter for minor revisions
        
        current_round = cycle.get_current_round()
        if current_round:
            current_round.revision_deadline = datetime.now() + timedelta(weeks=revision_weeks)
        
        logger.info(f"Requested {revision_type.value} for cycle {cycle_id}, "
                   f"revision due in {revision_weeks} weeks")
    
    def submit_revision(
        self,
        cycle_id: str,
        submission_date: Optional[datetime] = None
    ) -> bool:
        """
        Submit a revision and start the next review round.
        
        Args:
            cycle_id: ID of the revision cycle
            submission_date: Date of revision submission (defaults to now)
        
        Returns:
            True if revision was accepted, False otherwise
        """
        if cycle_id not in self.active_cycles:
            raise ValidationError("cycle_id", cycle_id, "active revision cycle")
        
        if submission_date is None:
            submission_date = datetime.now()
        
        cycle = self.active_cycles[cycle_id]
        
        if cycle.current_status != RevisionStatus.REVISION_REQUESTED:
            raise ValidationError("cycle_status", cycle.current_status.value, "revision requested")
        
        # Mark current round as having revision submitted
        current_round = cycle.get_current_round()
        if current_round:
            current_round.revision_submitted = True
            current_round.revision_submission_date = submission_date
        
        # Start new review round
        cycle.current_round_number += 1
        review_deadline = submission_date + timedelta(weeks=self.revision_review_weeks)
        
        # Use same reviewers for re-review (could be modified to use different reviewers)
        reviewer_ids = current_round.reviewer_assignments if current_round else set()
        
        new_round = RevisionRound(
            round_number=cycle.current_round_number,
            submission_date=submission_date,
            review_deadline=review_deadline,
            reviewer_assignments=reviewer_ids
        )
        
        cycle.rounds.append(new_round)
        cycle.current_status = RevisionStatus.RE_REVIEW_IN_PROGRESS
        cycle.workflow_state = WorkflowState.REVISION_UNDER_REVIEW
        
        logger.info(f"Submitted revision for cycle {cycle_id}, started round {cycle.current_round_number}")
        
        return True
    
    def _finalize_cycle(self, cycle_id: str, final_decision: ReviewDecision):
        """Finalize a revision cycle with a final decision."""
        cycle = self.active_cycles[cycle_id]
        
        cycle.final_decision = final_decision
        cycle.final_decision_date = datetime.now()
        cycle.total_review_time = cycle.get_total_review_duration()
        
        if final_decision == ReviewDecision.ACCEPT:
            cycle.current_status = RevisionStatus.ACCEPTED
        else:
            cycle.current_status = RevisionStatus.REJECTED
        
        cycle.workflow_state = WorkflowState.COMPLETED
        
        # Move to completed cycles
        self.completed_cycles[cycle_id] = cycle
        del self.active_cycles[cycle_id]
        
        logger.info(f"Finalized cycle {cycle_id} with decision: {final_decision.value}, "
                   f"total duration: {cycle.total_review_time}")
    
    def get_cycle_status(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a revision cycle.
        
        Args:
            cycle_id: ID of the revision cycle
        
        Returns:
            Dictionary with cycle status information
        """
        cycle = self.active_cycles.get(cycle_id) or self.completed_cycles.get(cycle_id)
        
        if cycle is None:
            return None
        
        current_round = cycle.get_current_round()
        
        return {
            'cycle_id': cycle_id,
            'paper_id': cycle.paper_id,
            'venue_id': cycle.venue_id,
            'current_status': cycle.current_status.value,
            'workflow_state': cycle.workflow_state.value,
            'current_round_number': cycle.current_round_number,
            'total_rounds': cycle.get_total_rounds(),
            'max_revision_rounds': cycle.max_revision_rounds,
            'is_complete': cycle.is_complete(),
            'can_request_revision': cycle.can_request_revision(),
            'final_decision': cycle.final_decision.value if cycle.final_decision else None,
            'total_review_time_days': cycle.total_review_time.days if cycle.total_review_time else None,
            'current_round_info': {
                'round_number': current_round.round_number if current_round else None,
                'review_deadline': current_round.review_deadline.isoformat() if current_round and current_round.review_deadline else None,
                'revision_deadline': current_round.revision_deadline.isoformat() if current_round and current_round.revision_deadline else None,
                'reviews_received': len(current_round.reviews) if current_round else 0,
                'reviews_expected': len(current_round.reviewer_assignments) if current_round else 0,
                'is_review_complete': current_round.is_review_complete() if current_round else False,
                'average_score': current_round.get_average_score() if current_round else None,
                'decision': current_round.decision.value if current_round and current_round.decision else None
            } if current_round else None
        }
    
    def get_overdue_cycles(self, current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get cycles with overdue deadlines.
        
        Args:
            current_time: Current time (defaults to now)
        
        Returns:
            List of overdue cycle information
        """
        if current_time is None:
            current_time = datetime.now()
        
        overdue_cycles = []
        
        for cycle in self.active_cycles.values():
            current_round = cycle.get_current_round()
            if current_round is None:
                continue
            
            overdue_info = None
            
            # Check review deadline
            if (current_round.review_deadline and 
                current_time > current_round.review_deadline and 
                not current_round.is_review_complete()):
                
                days_overdue = (current_time - current_round.review_deadline).days
                overdue_info = {
                    'type': 'review',
                    'deadline': current_round.review_deadline.isoformat(),
                    'days_overdue': days_overdue
                }
            
            # Check revision deadline
            elif (current_round.revision_deadline and 
                  current_time > current_round.revision_deadline and 
                  not current_round.revision_submitted):
                
                days_overdue = (current_time - current_round.revision_deadline).days
                overdue_info = {
                    'type': 'revision',
                    'deadline': current_round.revision_deadline.isoformat(),
                    'days_overdue': days_overdue
                }
            
            if overdue_info:
                cycle_info = self.get_cycle_status(cycle.cycle_id)
                cycle_info['overdue_info'] = overdue_info
                overdue_cycles.append(cycle_info)
        
        return overdue_cycles
    
    def get_cycles_by_status(self, status: RevisionStatus) -> List[str]:
        """
        Get cycle IDs by status.
        
        Args:
            status: Revision status to filter by
        
        Returns:
            List of cycle IDs with the specified status
        """
        matching_cycles = []
        
        # Check active cycles
        for cycle_id, cycle in self.active_cycles.items():
            if cycle.current_status == status:
                matching_cycles.append(cycle_id)
        
        # Check completed cycles
        for cycle_id, cycle in self.completed_cycles.items():
            if cycle.current_status == status:
                matching_cycles.append(cycle_id)
        
        return matching_cycles
    
    def get_venue_cycle_statistics(self, venue_id: str) -> Dict[str, Any]:
        """
        Get revision cycle statistics for a venue.
        
        Args:
            venue_id: ID of the venue
        
        Returns:
            Dictionary with venue cycle statistics
        """
        venue_cycles = []
        
        # Collect all cycles for this venue
        for cycle in list(self.active_cycles.values()) + list(self.completed_cycles.values()):
            if cycle.venue_id == venue_id:
                venue_cycles.append(cycle)
        
        if not venue_cycles:
            return {
                'venue_id': venue_id,
                'total_cycles': 0,
                'status_breakdown': {},
                'average_rounds': 0.0,
                'average_review_time_days': 0.0,
                'acceptance_rate': 0.0
            }
        
        # Calculate statistics
        total_cycles = len(venue_cycles)
        status_counts = {}
        total_rounds = 0
        total_review_time = timedelta()
        accepted_count = 0
        completed_cycles = []
        
        for cycle in venue_cycles:
            status = cycle.current_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_rounds += cycle.get_total_rounds()
            
            if cycle.is_complete():
                completed_cycles.append(cycle)
                if cycle.final_decision == ReviewDecision.ACCEPT:
                    accepted_count += 1
                
                if cycle.total_review_time:
                    total_review_time += cycle.total_review_time
        
        avg_rounds = total_rounds / total_cycles if total_cycles > 0 else 0.0
        avg_review_time_days = total_review_time.days / len(completed_cycles) if completed_cycles else 0.0
        acceptance_rate = accepted_count / len(completed_cycles) if completed_cycles else 0.0
        
        return {
            'venue_id': venue_id,
            'total_cycles': total_cycles,
            'active_cycles': len([c for c in venue_cycles if not c.is_complete()]),
            'completed_cycles': len(completed_cycles),
            'status_breakdown': status_counts,
            'average_rounds': avg_rounds,
            'average_review_time_days': avg_review_time_days,
            'acceptance_rate': acceptance_rate
        }
    
    def cleanup_old_cycles(self, days_old: int = 365):
        """
        Clean up old completed cycles.
        
        Args:
            days_old: Remove cycles completed more than this many days ago
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        old_cycle_ids = []
        
        for cycle_id, cycle in self.completed_cycles.items():
            if (cycle.final_decision_date and 
                cycle.final_decision_date < cutoff_date):
                old_cycle_ids.append(cycle_id)
        
        for cycle_id in old_cycle_ids:
            del self.completed_cycles[cycle_id]
        
        logger.info(f"Cleaned up {len(old_cycle_ids)} old revision cycles older than {days_old} days")
    
    def force_decision(self, cycle_id: str, decision: ReviewDecision, reason: str = "Administrative decision"):
        """
        Force a decision on a cycle (for administrative purposes).
        
        Args:
            cycle_id: ID of the revision cycle
            decision: Decision to apply
            reason: Reason for forced decision
        """
        if cycle_id not in self.active_cycles:
            raise ValidationError("cycle_id", cycle_id, "active revision cycle")
        
        cycle = self.active_cycles[cycle_id]
        
        # Add decision to current round if exists
        current_round = cycle.get_current_round()
        if current_round:
            current_round.decision = decision
            current_round.decision_date = datetime.now()
        
        self._finalize_cycle(cycle_id, decision)
        
        logger.warning(f"Forced decision {decision.value} on cycle {cycle_id}: {reason}")
    
    def update_reviewer_assignments(
        self, 
        cycle_id: str, 
        new_reviewer_ids: Set[str],
        round_number: Optional[int] = None
    ):
        """
        Update reviewer assignments for a cycle.
        
        Args:
            cycle_id: ID of the revision cycle
            new_reviewer_ids: New set of reviewer IDs
            round_number: Specific round to update (defaults to current round)
        """
        if cycle_id not in self.active_cycles:
            raise ValidationError("cycle_id", cycle_id, "active revision cycle")
        
        cycle = self.active_cycles[cycle_id]
        
        if round_number is None:
            target_round = cycle.get_current_round()
        else:
            target_round = next(
                (r for r in cycle.rounds if r.round_number == round_number),
                None
            )
        
        if target_round is None:
            raise ValidationError("round_number", round_number, "valid round number")
        
        old_reviewers = target_round.reviewer_assignments.copy()
        target_round.reviewer_assignments = new_reviewer_ids.copy()
        
        logger.info(f"Updated reviewer assignments for cycle {cycle_id}, round {target_round.round_number}: "
                   f"removed {old_reviewers - new_reviewer_ids}, added {new_reviewer_ids - old_reviewers}")
    
    def extend_deadline(
        self, 
        cycle_id: str, 
        extension_days: int,
        deadline_type: str = "review"
    ):
        """
        Extend a deadline for a cycle.
        
        Args:
            cycle_id: ID of the revision cycle
            extension_days: Number of days to extend
            deadline_type: Type of deadline ("review" or "revision")
        """
        if cycle_id not in self.active_cycles:
            raise ValidationError("cycle_id", cycle_id, "active revision cycle")
        
        cycle = self.active_cycles[cycle_id]
        current_round = cycle.get_current_round()
        
        if current_round is None:
            raise ValidationError("cycle_state", "no current round", "active review round")
        
        if deadline_type == "review" and current_round.review_deadline:
            old_deadline = current_round.review_deadline
            current_round.review_deadline += timedelta(days=extension_days)
            logger.info(f"Extended review deadline for cycle {cycle_id} by {extension_days} days: "
                       f"{old_deadline.strftime('%Y-%m-%d')} -> {current_round.review_deadline.strftime('%Y-%m-%d')}")
        
        elif deadline_type == "revision" and current_round.revision_deadline:
            old_deadline = current_round.revision_deadline
            current_round.revision_deadline += timedelta(days=extension_days)
            logger.info(f"Extended revision deadline for cycle {cycle_id} by {extension_days} days: "
                       f"{old_deadline.strftime('%Y-%m-%d')} -> {current_round.revision_deadline.strftime('%Y-%m-%d')}")
        
        else:
            raise ValidationError("deadline_type", deadline_type, "valid deadline type with existing deadline")