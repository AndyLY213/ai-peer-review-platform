"""
Unit tests for the RevisionCycleManager class.

Tests RevisionCycleManager for handling revision cycles, logic to manage re-review
processes with updated deadlines, workflow state management for multi-round reviews,
and comprehensive revision cycle management functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.enhancements.revision_cycle_manager import (
    RevisionCycleManager, RevisionCycle, RevisionRound, RevisionStatus, WorkflowState
)
from src.data.enhanced_models import (
    EnhancedVenue, VenueType, StructuredReview, ReviewDecision, 
    EnhancedReviewCriteria, EnhancedResearcher, ResearcherLevel
)
from src.core.exceptions import ValidationError


class TestRevisionRound:
    """Test RevisionRound functionality."""
    
    def test_revision_round_creation(self):
        """Test creating a revision round."""
        now = datetime.now()
        deadline = now + timedelta(weeks=4)
        
        round_obj = RevisionRound(
            round_number=1,
            submission_date=now,
            review_deadline=deadline,
            reviewer_assignments={"reviewer1", "reviewer2"}
        )
        
        assert round_obj.round_number == 1
        assert round_obj.submission_date == now
        assert round_obj.review_deadline == deadline
        assert len(round_obj.reviewer_assignments) == 2
        assert not round_obj.revision_submitted
        assert len(round_obj.reviews) == 0
    
    def test_is_review_complete(self):
        """Test review completion detection."""
        round_obj = RevisionRound(
            round_number=1,
            reviewer_assignments={"reviewer1", "reviewer2"}
        )
        
        # Not complete with no reviews
        assert not round_obj.is_review_complete()
        
        # Add one review
        review1 = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1"
        )
        round_obj.reviews.append(review1)
        
        # Still not complete with only one review
        assert not round_obj.is_review_complete()
        
        # Add second review
        review2 = StructuredReview(
            reviewer_id="reviewer2",
            paper_id="paper1",
            venue_id="venue1"
        )
        round_obj.reviews.append(review2)
        
        # Now complete
        assert round_obj.is_review_complete()
    
    def test_get_average_score(self):
        """Test average score calculation."""
        round_obj = RevisionRound(round_number=1)
        
        # No reviews should return None
        assert round_obj.get_average_score() is None
        
        # Add reviews with different scores
        review1 = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            criteria_scores=EnhancedReviewCriteria(
                novelty=8.0, technical_quality=7.0, clarity=6.0,
                significance=8.0, reproducibility=7.0, related_work=6.0
            )
        )
        review2 = StructuredReview(
            reviewer_id="reviewer2",
            paper_id="paper1",
            venue_id="venue1",
            criteria_scores=EnhancedReviewCriteria(
                novelty=6.0, technical_quality=5.0, clarity=8.0,
                significance=6.0, reproducibility=5.0, related_work=8.0
            )
        )
        
        round_obj.reviews.extend([review1, review2])
        
        # Average should be (7.0 + 6.33) / 2 ≈ 6.67
        avg_score = round_obj.get_average_score()
        assert avg_score is not None
        assert 6.5 <= avg_score <= 6.8
    
    def test_requires_revision(self):
        """Test revision requirement detection."""
        round_obj = RevisionRound(round_number=1)
        
        # No decision should not require revision
        assert not round_obj.requires_revision()
        
        # Accept should not require revision
        round_obj.decision = ReviewDecision.ACCEPT
        assert not round_obj.requires_revision()
        
        # Reject should not require revision
        round_obj.decision = ReviewDecision.REJECT
        assert not round_obj.requires_revision()
        
        # Minor revision should require revision
        round_obj.decision = ReviewDecision.MINOR_REVISION
        assert round_obj.requires_revision()
        
        # Major revision should require revision
        round_obj.decision = ReviewDecision.MAJOR_REVISION
        assert round_obj.requires_revision()


class TestRevisionCycle:
    """Test RevisionCycle functionality."""
    
    def test_revision_cycle_creation(self):
        """Test creating a revision cycle."""
        now = datetime.now()
        
        cycle = RevisionCycle(
            paper_id="paper1",
            venue_id="venue1",
            author_ids={"author1", "author2"},
            initial_submission_date=now
        )
        
        assert cycle.paper_id == "paper1"
        assert cycle.venue_id == "venue1"
        assert len(cycle.author_ids) == 2
        assert cycle.initial_submission_date == now
        assert cycle.current_status == RevisionStatus.INITIAL_REVIEW
        assert cycle.workflow_state == WorkflowState.SUBMITTED
        assert cycle.current_round_number == 1
        assert cycle.max_revision_rounds == 2
        assert len(cycle.rounds) == 0
    
    def test_get_current_round(self):
        """Test getting current round."""
        cycle = RevisionCycle(paper_id="paper1", venue_id="venue1")
        
        # No rounds initially
        assert cycle.get_current_round() is None
        
        # Add a round
        round1 = RevisionRound(round_number=1)
        cycle.rounds.append(round1)
        
        assert cycle.get_current_round() == round1
        
        # Add another round
        round2 = RevisionRound(round_number=2)
        cycle.rounds.append(round2)
        
        # Should return the latest round
        assert cycle.get_current_round() == round2
    
    def test_get_total_rounds(self):
        """Test getting total number of rounds."""
        cycle = RevisionCycle(paper_id="paper1", venue_id="venue1")
        
        assert cycle.get_total_rounds() == 0
        
        cycle.rounds.append(RevisionRound(round_number=1))
        assert cycle.get_total_rounds() == 1
        
        cycle.rounds.append(RevisionRound(round_number=2))
        assert cycle.get_total_rounds() == 2
    
    def test_is_complete(self):
        """Test completion detection."""
        cycle = RevisionCycle(paper_id="paper1", venue_id="venue1")
        
        # Initially not complete
        assert not cycle.is_complete()
        
        # Set to accepted
        cycle.current_status = RevisionStatus.ACCEPTED
        assert cycle.is_complete()
        
        # Set to rejected
        cycle.current_status = RevisionStatus.REJECTED
        assert cycle.is_complete()
        
        # Set to final decision
        cycle.current_status = RevisionStatus.FINAL_DECISION
        assert cycle.is_complete()
        
        # Other statuses should not be complete
        cycle.current_status = RevisionStatus.REVISION_REQUESTED
        assert not cycle.is_complete()
    
    def test_can_request_revision(self):
        """Test revision request capability."""
        cycle = RevisionCycle(paper_id="paper1", venue_id="venue1", max_revision_rounds=2)
        
        # Initially can request revision
        assert cycle.can_request_revision()
        
        # Add one round
        cycle.rounds.append(RevisionRound(round_number=1))
        assert cycle.can_request_revision()
        
        # Add second round (at max)
        cycle.rounds.append(RevisionRound(round_number=2))
        assert not cycle.can_request_revision()
        
        # Complete cycle
        cycle.current_status = RevisionStatus.ACCEPTED
        assert not cycle.can_request_revision()
    
    def test_get_total_review_duration(self):
        """Test total review duration calculation."""
        now = datetime.now()
        cycle = RevisionCycle(
            paper_id="paper1",
            venue_id="venue1",
            initial_submission_date=now
        )
        
        # No final decision yet
        assert cycle.get_total_review_duration() is None
        
        # Set final decision date
        final_date = now + timedelta(days=30)
        cycle.final_decision_date = final_date
        
        duration = cycle.get_total_review_duration()
        assert duration is not None
        assert duration.days == 30


class TestRevisionCycleManager:
    """Test RevisionCycleManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = RevisionCycleManager()
        self.venue = EnhancedVenue(
            id="venue1",
            name="Test Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI",
            review_deadline_weeks=4,
            revision_cycles_allowed=2
        )
        self.author_ids = {"author1", "author2"}
        self.reviewer_ids = {"reviewer1", "reviewer2", "reviewer3"}
    
    def test_initialization(self):
        """Test RevisionCycleManager initialization."""
        manager = RevisionCycleManager()
        
        assert len(manager.active_cycles) == 0
        assert len(manager.completed_cycles) == 0
        assert manager.default_review_weeks == 4
        assert manager.default_revision_weeks == 8
        assert manager.revision_review_weeks == 3
    
    def test_start_review_cycle(self):
        """Test starting a new review cycle."""
        now = datetime.now()
        
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids,
            submission_date=now
        )
        
        assert cycle.paper_id == "paper1"
        assert cycle.venue_id == "venue1"
        assert cycle.author_ids == self.author_ids
        assert cycle.initial_submission_date == now
        assert cycle.workflow_state == WorkflowState.UNDER_REVIEW
        assert len(cycle.rounds) == 1
        
        # Check initial round
        initial_round = cycle.rounds[0]
        assert initial_round.round_number == 1
        assert initial_round.submission_date == now
        assert initial_round.reviewer_assignments == self.reviewer_ids
        
        # Should be in active cycles
        assert cycle.cycle_id in self.manager.active_cycles
    
    def test_submit_review(self):
        """Test submitting a review."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        review = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            recommendation=ReviewDecision.ACCEPT
        )
        
        result = self.manager.submit_review(cycle.cycle_id, review)
        
        assert result
        
        # Check review was added
        current_round = cycle.get_current_round()
        assert len(current_round.reviews) == 1
        assert current_round.reviews[0].reviewer_id == "reviewer1"
        assert current_round.reviews[0].revision_round == 1
    
    def test_submit_review_invalid_cycle(self):
        """Test submitting review to invalid cycle."""
        review = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1"
        )
        
        with pytest.raises(ValidationError):
            self.manager.submit_review("invalid_cycle", review)
    
    def test_submit_review_invalid_reviewer(self):
        """Test submitting review from unassigned reviewer."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        review = StructuredReview(
            reviewer_id="unassigned_reviewer",
            paper_id="paper1",
            venue_id="venue1"
        )
        
        with pytest.raises(ValidationError):
            self.manager.submit_review(cycle.cycle_id, review)
    
    def test_submit_review_replace_existing(self):
        """Test replacing existing review from same reviewer."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        # Submit first review
        review1 = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            recommendation=ReviewDecision.ACCEPT
        )
        self.manager.submit_review(cycle.cycle_id, review1)
        
        # Submit replacement review
        review2 = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            recommendation=ReviewDecision.REJECT
        )
        self.manager.submit_review(cycle.cycle_id, review2)
        
        # Should have only one review from reviewer1
        current_round = cycle.get_current_round()
        reviewer1_reviews = [r for r in current_round.reviews if r.reviewer_id == "reviewer1"]
        assert len(reviewer1_reviews) == 1
        assert reviewer1_reviews[0].recommendation == ReviewDecision.REJECT
    
    def test_complete_round_accept(self):
        """Test completing a round with accept decision."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids={"reviewer1", "reviewer2"}  # Only 2 reviewers for easier testing
        )
        
        # Submit accepting reviews
        for i, reviewer_id in enumerate(["reviewer1", "reviewer2"]):
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id="paper1",
                venue_id="venue1",
                recommendation=ReviewDecision.ACCEPT,
                criteria_scores=EnhancedReviewCriteria(
                    novelty=8.0, technical_quality=8.0, clarity=8.0,
                    significance=8.0, reproducibility=8.0, related_work=8.0
                )
            )
            self.manager.submit_review(cycle.cycle_id, review)
        
        # Cycle should be completed and accepted
        assert cycle.is_complete()
        assert cycle.final_decision == ReviewDecision.ACCEPT
        assert cycle.current_status == RevisionStatus.ACCEPTED
        assert cycle.cycle_id in self.manager.completed_cycles
        assert cycle.cycle_id not in self.manager.active_cycles
    
    def test_complete_round_reject(self):
        """Test completing a round with reject decision."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids={"reviewer1", "reviewer2"}
        )
        
        # Submit rejecting reviews
        for reviewer_id in ["reviewer1", "reviewer2"]:
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id="paper1",
                venue_id="venue1",
                recommendation=ReviewDecision.REJECT,
                criteria_scores=EnhancedReviewCriteria(
                    novelty=3.0, technical_quality=3.0, clarity=3.0,
                    significance=3.0, reproducibility=3.0, related_work=3.0
                )
            )
            self.manager.submit_review(cycle.cycle_id, review)
        
        # Cycle should be completed and rejected
        assert cycle.is_complete()
        assert cycle.final_decision == ReviewDecision.REJECT
        assert cycle.current_status == RevisionStatus.REJECTED
    
    def test_complete_round_revision_requested(self):
        """Test completing a round with revision requested."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids={"reviewer1", "reviewer2"}
        )
        
        # Submit mixed reviews that should result in revision
        review1 = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            recommendation=ReviewDecision.MINOR_REVISION,
            criteria_scores=EnhancedReviewCriteria(
                novelty=6.0, technical_quality=6.0, clarity=6.0,
                significance=6.0, reproducibility=6.0, related_work=6.0
            )
        )
        review2 = StructuredReview(
            reviewer_id="reviewer2",
            paper_id="paper1",
            venue_id="venue1",
            recommendation=ReviewDecision.MINOR_REVISION,
            criteria_scores=EnhancedReviewCriteria(
                novelty=5.0, technical_quality=5.0, clarity=5.0,
                significance=5.0, reproducibility=5.0, related_work=5.0
            )
        )
        
        self.manager.submit_review(cycle.cycle_id, review1)
        self.manager.submit_review(cycle.cycle_id, review2)
        
        # Cycle should request revision
        assert not cycle.is_complete()
        assert cycle.current_status == RevisionStatus.REVISION_REQUESTED
        assert cycle.workflow_state == WorkflowState.AWAITING_REVISION
        
        # Current round should have revision deadline set
        current_round = cycle.get_current_round()
        assert current_round.revision_deadline is not None
    
    def test_submit_revision(self):
        """Test submitting a revision."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids={"reviewer1", "reviewer2"}
        )
        
        # Complete first round with revision request
        for reviewer_id in ["reviewer1", "reviewer2"]:
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id="paper1",
                venue_id="venue1",
                recommendation=ReviewDecision.MINOR_REVISION
            )
            self.manager.submit_review(cycle.cycle_id, review)
        
        # Submit revision
        revision_date = datetime.now() + timedelta(days=30)
        result = self.manager.submit_revision(cycle.cycle_id, revision_date)
        
        assert result
        assert cycle.current_round_number == 2
        assert len(cycle.rounds) == 2
        assert cycle.current_status == RevisionStatus.RE_REVIEW_IN_PROGRESS
        assert cycle.workflow_state == WorkflowState.REVISION_UNDER_REVIEW
        
        # First round should be marked as having revision submitted
        first_round = cycle.rounds[0]
        assert first_round.revision_submitted
        assert first_round.revision_submission_date == revision_date
        
        # Second round should be set up for re-review
        second_round = cycle.rounds[1]
        assert second_round.round_number == 2
        assert second_round.submission_date == revision_date
        assert second_round.review_deadline is not None
    
    def test_submit_revision_invalid_status(self):
        """Test submitting revision when not in revision requested status."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        # Try to submit revision without requesting it first
        with pytest.raises(ValidationError):
            self.manager.submit_revision(cycle.cycle_id)
    
    def test_get_cycle_status(self):
        """Test getting cycle status information."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        status = self.manager.get_cycle_status(cycle.cycle_id)
        
        assert status is not None
        assert status['cycle_id'] == cycle.cycle_id
        assert status['paper_id'] == "paper1"
        assert status['venue_id'] == "venue1"
        assert status['current_status'] == RevisionStatus.INITIAL_REVIEW.value
        assert status['workflow_state'] == WorkflowState.UNDER_REVIEW.value
        assert status['current_round_number'] == 1
        assert status['total_rounds'] == 1
        assert not status['is_complete']
        assert status['can_request_revision']
        assert status['current_round_info'] is not None
    
    def test_get_cycle_status_nonexistent(self):
        """Test getting status for nonexistent cycle."""
        status = self.manager.get_cycle_status("nonexistent")
        assert status is None
    
    def test_get_overdue_cycles(self):
        """Test getting overdue cycles."""
        now = datetime.now()
        past_time = now - timedelta(days=30)
        
        # Create cycle with past deadline
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids,
            submission_date=past_time
        )
        
        # Manually set review deadline to past
        current_round = cycle.get_current_round()
        current_round.review_deadline = now - timedelta(days=5)
        
        overdue = self.manager.get_overdue_cycles(now)
        
        assert len(overdue) == 1
        assert overdue[0]['cycle_id'] == cycle.cycle_id
        assert overdue[0]['overdue_info']['type'] == 'review'
        assert overdue[0]['overdue_info']['days_overdue'] == 5
    
    def test_get_cycles_by_status(self):
        """Test getting cycles by status."""
        # Create cycle in initial review
        cycle1 = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        # Create another cycle and complete it
        cycle2 = self.manager.start_review_cycle(
            paper_id="paper2",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids={"reviewer1", "reviewer2"}
        )
        
        # Complete second cycle
        for reviewer_id in ["reviewer1", "reviewer2"]:
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id="paper2",
                venue_id="venue1",
                recommendation=ReviewDecision.ACCEPT,
                criteria_scores=EnhancedReviewCriteria(
                    novelty=8.0, technical_quality=8.0, clarity=8.0,
                    significance=8.0, reproducibility=8.0, related_work=8.0
                )
            )
            self.manager.submit_review(cycle2.cycle_id, review)
        
        # Test filtering by status
        initial_review_cycles = self.manager.get_cycles_by_status(RevisionStatus.INITIAL_REVIEW)
        accepted_cycles = self.manager.get_cycles_by_status(RevisionStatus.ACCEPTED)
        
        assert len(initial_review_cycles) == 1
        assert cycle1.cycle_id in initial_review_cycles
        
        assert len(accepted_cycles) == 1
        assert cycle2.cycle_id in accepted_cycles
    
    def test_get_venue_cycle_statistics(self):
        """Test getting venue cycle statistics."""
        # Create multiple cycles for the venue
        cycles = []
        for i in range(3):
            cycle = self.manager.start_review_cycle(
                paper_id=f"paper{i}",
                venue=self.venue,
                author_ids=self.author_ids,
                reviewer_ids={"reviewer1", "reviewer2"}
            )
            cycles.append(cycle)
        
        # Complete some cycles
        for i, cycle in enumerate(cycles[:2]):
            decision = ReviewDecision.ACCEPT if i == 0 else ReviewDecision.REJECT
            for reviewer_id in ["reviewer1", "reviewer2"]:
                review = StructuredReview(
                    reviewer_id=reviewer_id,
                    paper_id=cycle.paper_id,
                    venue_id="venue1",
                    recommendation=decision,
                    criteria_scores=EnhancedReviewCriteria(
                        novelty=8.0 if decision == ReviewDecision.ACCEPT else 3.0,
                        technical_quality=8.0 if decision == ReviewDecision.ACCEPT else 3.0,
                        clarity=8.0 if decision == ReviewDecision.ACCEPT else 3.0,
                        significance=8.0 if decision == ReviewDecision.ACCEPT else 3.0,
                        reproducibility=8.0 if decision == ReviewDecision.ACCEPT else 3.0,
                        related_work=8.0 if decision == ReviewDecision.ACCEPT else 3.0
                    )
                )
                self.manager.submit_review(cycle.cycle_id, review)
        
        stats = self.manager.get_venue_cycle_statistics("venue1")
        
        assert stats['venue_id'] == "venue1"
        assert stats['total_cycles'] == 3
        assert stats['active_cycles'] == 1
        assert stats['completed_cycles'] == 2
        assert stats['acceptance_rate'] == 0.5  # 1 accepted out of 2 completed
        assert stats['average_rounds'] == 1.0  # All cycles have 1 round
    
    def test_cleanup_old_cycles(self):
        """Test cleaning up old completed cycles."""
        now = datetime.now()
        old_date = now - timedelta(days=400)
        
        # Create and complete an old cycle
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids={"reviewer1", "reviewer2"}
        )
        
        # Complete the cycle
        for reviewer_id in ["reviewer1", "reviewer2"]:
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id="paper1",
                venue_id="venue1",
                recommendation=ReviewDecision.ACCEPT,
                criteria_scores=EnhancedReviewCriteria(
                    novelty=8.0, technical_quality=8.0, clarity=8.0,
                    significance=8.0, reproducibility=8.0, related_work=8.0
                )
            )
            self.manager.submit_review(cycle.cycle_id, review)
        
        # Manually set old final decision date
        cycle.final_decision_date = old_date
        
        assert len(self.manager.completed_cycles) == 1
        
        # Clean up old cycles
        self.manager.cleanup_old_cycles(days_old=365)
        
        assert len(self.manager.completed_cycles) == 0
    
    def test_force_decision(self):
        """Test forcing a decision on a cycle."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        # Force accept decision
        self.manager.force_decision(cycle.cycle_id, ReviewDecision.ACCEPT, "Test reason")
        
        assert cycle.is_complete()
        assert cycle.final_decision == ReviewDecision.ACCEPT
        assert cycle.current_status == RevisionStatus.ACCEPTED
        assert cycle.cycle_id in self.manager.completed_cycles
    
    def test_update_reviewer_assignments(self):
        """Test updating reviewer assignments."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        new_reviewers = {"reviewer4", "reviewer5"}
        self.manager.update_reviewer_assignments(cycle.cycle_id, new_reviewers)
        
        current_round = cycle.get_current_round()
        assert current_round.reviewer_assignments == new_reviewers
    
    def test_extend_deadline(self):
        """Test extending deadlines."""
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=self.venue,
            author_ids=self.author_ids,
            reviewer_ids=self.reviewer_ids
        )
        
        current_round = cycle.get_current_round()
        original_deadline = current_round.review_deadline
        
        # Extend review deadline by 7 days
        self.manager.extend_deadline(cycle.cycle_id, 7, "review")
        
        new_deadline = current_round.review_deadline
        assert new_deadline == original_deadline + timedelta(days=7)
    
    def test_max_revision_rounds_exceeded(self):
        """Test behavior when maximum revision rounds are exceeded."""
        # Set max revision rounds to 1
        venue = EnhancedVenue(
            id="venue1",
            name="Test Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI",
            review_deadline_weeks=4,
            revision_cycles_allowed=1
        )
        
        cycle = self.manager.start_review_cycle(
            paper_id="paper1",
            venue=venue,
            author_ids=self.author_ids,
            reviewer_ids={"reviewer1", "reviewer2"}
        )
        
        # Complete first round with major revision
        for reviewer_id in ["reviewer1", "reviewer2"]:
            review = StructuredReview(
                reviewer_id=reviewer_id,
                paper_id="paper1",
                venue_id="venue1",
                recommendation=ReviewDecision.MAJOR_REVISION
            )
            self.manager.submit_review(cycle.cycle_id, review)
        
        # Should be rejected since max revision rounds is 1 and we requested major revision
        assert cycle.is_complete()
        assert cycle.final_decision == ReviewDecision.REJECT


if __name__ == "__main__":
    pytest.main([__file__])