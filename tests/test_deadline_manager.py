"""
Unit tests for the DeadlineManager class.

Tests venue-specific deadlines (2-8 weeks), review submission timing tracking,
penalty system for late submissions, and comprehensive deadline management functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.enhancements.deadline_manager import (
    DeadlineManager, DeadlineAssignment, DeadlineStatus, PenaltyConfiguration
)
from src.data.enhanced_models import (
    EnhancedVenue, VenueType, StructuredReview, EnhancedResearcher, 
    ResearcherLevel, ReviewQualityMetric
)
from src.core.exceptions import ValidationError


class TestDeadlineAssignment:
    """Test DeadlineAssignment functionality."""
    
    def test_deadline_assignment_creation(self):
        """Test creating a deadline assignment."""
        now = datetime.now()
        deadline = now + timedelta(weeks=4)
        
        assignment = DeadlineAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline
        )
        
        assert assignment.reviewer_id == "reviewer1"
        assert assignment.paper_id == "paper1"
        assert assignment.venue_id == "venue1"
        assert assignment.assigned_date == now
        assert assignment.deadline == deadline
        assert assignment.status == DeadlineStatus.ON_TIME
        assert assignment.days_late == 0
        assert assignment.penalty_applied == 0.0
    
    def test_on_time_submission(self):
        """Test on-time submission status calculation."""
        now = datetime.now()
        deadline = now + timedelta(weeks=4)
        submission = deadline - timedelta(hours=1)  # 1 hour before deadline
        
        assignment = DeadlineAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline,
            submission_date=submission
        )
        
        assert assignment.status == DeadlineStatus.ON_TIME
        assert assignment.days_late == 0
    
    def test_late_submission(self):
        """Test late submission status calculation."""
        now = datetime.now()
        deadline = now + timedelta(weeks=4)
        submission = deadline + timedelta(days=2)  # 2 days late
        
        assignment = DeadlineAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline,
            submission_date=submission
        )
        
        assert assignment.status == DeadlineStatus.LATE
        assert assignment.days_late == 2
    
    def test_very_late_submission(self):
        """Test very late submission status calculation."""
        now = datetime.now()
        deadline = now + timedelta(weeks=4)
        submission = deadline + timedelta(days=5)  # 5 days late
        
        assignment = DeadlineAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline,
            submission_date=submission
        )
        
        assert assignment.status == DeadlineStatus.VERY_LATE
        assert assignment.days_late == 5
    
    def test_missed_submission(self):
        """Test missed submission status calculation."""
        now = datetime.now()
        deadline = now + timedelta(weeks=4)
        submission = deadline + timedelta(days=10)  # 10 days late
        
        assignment = DeadlineAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline,
            submission_date=submission
        )
        
        assert assignment.status == DeadlineStatus.MISSED
        assert assignment.days_late == 10
    
    def test_is_overdue(self):
        """Test overdue detection."""
        now = datetime.now()
        deadline = now - timedelta(days=1)  # Yesterday
        
        assignment = DeadlineAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now - timedelta(weeks=4),
            deadline=deadline
        )
        
        assert assignment.is_overdue(now)
        
        # Not overdue if submitted
        assignment.submission_date = now
        assert not assignment.is_overdue(now)
    
    def test_days_until_deadline(self):
        """Test days until deadline calculation."""
        now = datetime.now()
        deadline = now + timedelta(days=5)
        
        assignment = DeadlineAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline
        )
        
        assert assignment.days_until_deadline(now) == 5
        
        # Past deadline should return 0
        past_deadline = now - timedelta(days=1)
        assignment.deadline = past_deadline
        assert assignment.days_until_deadline(now) == 0


class TestPenaltyConfiguration:
    """Test PenaltyConfiguration functionality."""
    
    def test_default_penalty_config(self):
        """Test default penalty configuration."""
        config = PenaltyConfiguration()
        
        assert config.base_penalty_per_day == 0.05
        assert config.max_penalty == 0.5
        assert config.grace_period_days == 1
        assert config.escalation_factor == 1.5
    
    def test_penalty_calculation_within_grace_period(self):
        """Test penalty calculation within grace period."""
        config = PenaltyConfiguration()
        
        # Within grace period
        assert config.calculate_penalty(0) == 0.0
        assert config.calculate_penalty(1) == 0.0
    
    def test_penalty_calculation_basic(self):
        """Test basic penalty calculation."""
        config = PenaltyConfiguration()
        
        # 2 days late (1 day after grace period)
        penalty = config.calculate_penalty(2)
        expected = 1 * 0.05  # 1 effective day * 5%
        assert penalty == expected
        
        # 5 days late (4 days after grace period)
        penalty = config.calculate_penalty(5)
        expected = 4 * 0.05  # 4 effective days * 5%
        assert penalty == expected
    
    def test_penalty_calculation_with_escalation(self):
        """Test penalty calculation with escalation for very late submissions."""
        config = PenaltyConfiguration()
        
        # 10 days late (9 days after grace period, 3 days with escalation)
        penalty = config.calculate_penalty(10)
        base_penalty = 6 * 0.05  # First 6 effective days (up to 7 total days late)
        escalation_penalty = 3 * 0.05 * 1.5  # 3 days with escalation
        expected = base_penalty + escalation_penalty
        # The penalty is capped at max_penalty (0.5), so it should be 0.5
        assert penalty == min(expected, config.max_penalty)
    
    def test_penalty_calculation_max_penalty(self):
        """Test penalty calculation respects maximum penalty."""
        config = PenaltyConfiguration()
        
        # Very late submission that would exceed max penalty
        penalty = config.calculate_penalty(50)
        assert penalty == config.max_penalty


class TestDeadlineManager:
    """Test DeadlineManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DeadlineManager()
        self.venue = EnhancedVenue(
            id="venue1",
            name="Test Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI"
        )
        self.researcher = EnhancedResearcher(
            id="researcher1",
            name="Test Researcher",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF
        )
    
    def test_initialization(self):
        """Test DeadlineManager initialization."""
        manager = DeadlineManager()
        
        assert isinstance(manager.penalty_config, PenaltyConfiguration)
        assert len(manager.deadline_assignments) == 0
        assert VenueType.TOP_CONFERENCE in manager.venue_deadline_weeks
        assert manager.venue_deadline_weeks[VenueType.TOP_CONFERENCE] == 6
        assert manager.venue_deadline_weeks[VenueType.MID_CONFERENCE] == 4
        assert manager.venue_deadline_weeks[VenueType.LOW_CONFERENCE] == 3
        assert manager.venue_deadline_weeks[VenueType.TOP_JOURNAL] == 8
    
    def test_get_venue_deadline_weeks(self):
        """Test getting venue-specific deadline weeks."""
        manager = DeadlineManager()
        
        assert manager.get_venue_deadline_weeks(VenueType.TOP_CONFERENCE) == 6
        assert manager.get_venue_deadline_weeks(VenueType.MID_CONFERENCE) == 4
        assert manager.get_venue_deadline_weeks(VenueType.LOW_CONFERENCE) == 3
        assert manager.get_venue_deadline_weeks(VenueType.TOP_JOURNAL) == 8
        assert manager.get_venue_deadline_weeks(VenueType.SPECIALIZED_JOURNAL) == 6
        assert manager.get_venue_deadline_weeks(VenueType.GENERAL_JOURNAL) == 4
        assert manager.get_venue_deadline_weeks(VenueType.WORKSHOP) == 2
        assert manager.get_venue_deadline_weeks(VenueType.PREPRINT) == 2
    
    def test_assign_review_deadline(self):
        """Test assigning review deadline."""
        now = datetime.now()
        assignment = self.manager.assign_review_deadline(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue=self.venue,
            assignment_date=now
        )
        
        assert assignment.reviewer_id == "reviewer1"
        assert assignment.paper_id == "paper1"
        assert assignment.venue_id == "venue1"
        assert assignment.assigned_date == now
        
        # Should be 4 weeks for MID_CONFERENCE
        expected_deadline = now + timedelta(weeks=4)
        assert abs((assignment.deadline - expected_deadline).total_seconds()) < 1
        
        # Should be stored in manager
        assert assignment.assignment_id in self.manager.deadline_assignments
    
    def test_assign_review_deadline_different_venues(self):
        """Test deadline assignment for different venue types."""
        now = datetime.now()
        
        # Test different venue types
        venues = [
            (VenueType.TOP_CONFERENCE, 6),
            (VenueType.LOW_CONFERENCE, 3),
            (VenueType.TOP_JOURNAL, 8),
            (VenueType.WORKSHOP, 2)
        ]
        
        for venue_type, expected_weeks in venues:
            venue = EnhancedVenue(
                id=f"venue_{venue_type.value}",
                name=f"Test {venue_type.value}",
                venue_type=venue_type,
                field="AI"
            )
            
            assignment = self.manager.assign_review_deadline(
                reviewer_id="reviewer1",
                paper_id="paper1",
                venue=venue,
                assignment_date=now
            )
            
            expected_deadline = now + timedelta(weeks=expected_weeks)
            assert abs((assignment.deadline - expected_deadline).total_seconds()) < 1
    
    def test_submit_review_on_time(self):
        """Test submitting review on time."""
        now = datetime.now()
        assignment = self.manager.assign_review_deadline(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue=self.venue,
            assignment_date=now
        )
        
        # Submit before deadline
        submission_time = assignment.deadline - timedelta(hours=1)
        review = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1"
        )
        
        is_on_time, penalty = self.manager.submit_review(
            assignment.assignment_id,
            review,
            submission_time
        )
        
        assert is_on_time
        assert penalty == 0.0
        assert assignment.status == DeadlineStatus.ON_TIME
        assert assignment.days_late == 0
        assert not review.is_late
    
    def test_submit_review_late(self):
        """Test submitting review late."""
        now = datetime.now()
        assignment = self.manager.assign_review_deadline(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue=self.venue,
            assignment_date=now
        )
        
        # Submit 3 days late
        submission_time = assignment.deadline + timedelta(days=3)
        review = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1"
        )
        
        is_on_time, penalty = self.manager.submit_review(
            assignment.assignment_id,
            review,
            submission_time
        )
        
        assert not is_on_time
        assert penalty > 0.0
        assert assignment.status == DeadlineStatus.LATE
        assert assignment.days_late == 3
        assert review.is_late
        
        # Check penalty calculation (3 days late, 1 day grace period = 2 effective days)
        expected_penalty = 2 * 0.05  # 2 days * 5%
        assert penalty == expected_penalty
    
    def test_submit_review_invalid_assignment(self):
        """Test submitting review with invalid assignment ID."""
        review = StructuredReview(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1"
        )
        
        with pytest.raises(ValidationError):
            self.manager.submit_review("invalid_id", review)
    
    def test_get_overdue_assignments(self):
        """Test getting overdue assignments."""
        now = datetime.now()
        
        # Create assignments with different deadlines
        assignment1 = self.manager.assign_review_deadline(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue=self.venue,
            assignment_date=now - timedelta(weeks=5)  # Should be overdue
        )
        
        assignment2 = self.manager.assign_review_deadline(
            reviewer_id="reviewer2",
            paper_id="paper2",
            venue=self.venue,
            assignment_date=now - timedelta(weeks=2)  # Should not be overdue
        )
        
        # Submit one review to make it not overdue
        review = StructuredReview(
            reviewer_id="reviewer2",
            paper_id="paper2",
            venue_id="venue1"
        )
        self.manager.submit_review(assignment2.assignment_id, review)
        
        overdue = self.manager.get_overdue_assignments(now)
        
        assert len(overdue) == 1
        assert overdue[0].assignment_id == assignment1.assignment_id
    
    def test_get_upcoming_deadlines(self):
        """Test getting upcoming deadlines."""
        now = datetime.now()
        
        # Create assignments with different deadlines
        assignment1 = self.manager.assign_review_deadline(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue=self.venue,
            assignment_date=now - timedelta(weeks=3, days=4)  # Due in 3 days
        )
        
        assignment2 = self.manager.assign_review_deadline(
            reviewer_id="reviewer2",
            paper_id="paper2",
            venue=self.venue,
            assignment_date=now - timedelta(weeks=2)  # Due in 2 weeks
        )
        
        upcoming = self.manager.get_upcoming_deadlines(days_ahead=7, current_time=now)
        
        assert len(upcoming) == 1
        assert upcoming[0].assignment_id == assignment1.assignment_id
    
    def test_apply_reliability_penalty(self):
        """Test applying reliability penalty based on deadline compliance."""
        now = datetime.now()
        
        # Create multiple assignments for the same reviewer
        assignments = []
        for i in range(5):
            assignment = self.manager.assign_review_deadline(
                reviewer_id="reviewer1",
                paper_id=f"paper{i}",
                venue=self.venue,
                assignment_date=now - timedelta(weeks=5)
            )
            assignments.append(assignment)
        
        # Submit some on time, some late
        review_times = [
            assignment.deadline - timedelta(hours=1),  # On time
            assignment.deadline + timedelta(days=2),   # Late
            assignment.deadline - timedelta(hours=2),  # On time
            assignment.deadline + timedelta(days=10),  # Very late (missed)
            assignment.deadline + timedelta(days=1),   # Late
        ]
        
        for assignment, submission_time in zip(assignments, review_times):
            review = StructuredReview(
                reviewer_id="reviewer1",
                paper_id=assignment.paper_id,
                venue_id="venue1"
            )
            self.manager.submit_review(assignment.assignment_id, review, submission_time)
        
        reliability_score = self.manager.apply_reliability_penalty("reviewer1", self.researcher)
        
        # 2 on time out of 5 = 0.4 base reliability
        # 1 very late submission = 0.1 penalty
        # Expected: 0.4 - 0.1 = 0.3
        assert abs(reliability_score - 0.3) < 0.01
        
        # Check that reliability metric was added to researcher
        assert len(self.researcher.review_quality_history) > 0
        latest_metric = self.researcher.review_quality_history[-1]
        assert latest_metric.quality_score == reliability_score
    
    def test_get_reviewer_deadline_stats(self):
        """Test getting reviewer deadline statistics."""
        now = datetime.now()
        
        # Create assignments with different outcomes
        assignments = []
        for i in range(4):
            assignment = self.manager.assign_review_deadline(
                reviewer_id="reviewer1",
                paper_id=f"paper{i}",
                venue=self.venue,
                assignment_date=now - timedelta(weeks=5)
            )
            assignments.append(assignment)
        
        # Submit with different timing
        review_times = [
            assignment.deadline - timedelta(hours=1),  # On time
            assignment.deadline + timedelta(days=2),   # Late
            assignment.deadline - timedelta(hours=2),  # On time
            assignment.deadline + timedelta(days=5),   # Very late
        ]
        
        for assignment, submission_time in zip(assignments, review_times):
            review = StructuredReview(
                reviewer_id="reviewer1",
                paper_id=assignment.paper_id,
                venue_id="venue1"
            )
            self.manager.submit_review(assignment.assignment_id, review, submission_time)
        
        stats = self.manager.get_reviewer_deadline_stats("reviewer1")
        
        assert stats['total_assignments'] == 4
        assert stats['on_time_rate'] == 0.5  # 2 out of 4
        assert stats['average_days_late'] == 1.75  # (0 + 2 + 0 + 5) / 4 = 7/4 = 1.75
        assert stats['status_breakdown']['On Time'] == 2
        assert stats['status_breakdown']['Late'] == 1
        assert stats['status_breakdown']['Very Late'] == 1
    
    def test_get_reviewer_deadline_stats_no_history(self):
        """Test getting reviewer deadline statistics with no history."""
        stats = self.manager.get_reviewer_deadline_stats("nonexistent_reviewer")
        
        assert stats['total_assignments'] == 0
        assert stats['on_time_rate'] == 1.0
        assert stats['average_days_late'] == 0.0
        assert stats['total_penalties'] == 0.0
    
    def test_send_deadline_reminders(self):
        """Test sending deadline reminders."""
        now = datetime.now()
        
        # Create assignments with different deadlines
        assignment1 = self.manager.assign_review_deadline(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue=self.venue,
            assignment_date=now - timedelta(weeks=3, days=4)  # Due in 3 days
        )
        
        assignment2 = self.manager.assign_review_deadline(
            reviewer_id="reviewer2",
            paper_id="paper2",
            venue=self.venue,
            assignment_date=now - timedelta(weeks=2)  # Due in 2 weeks
        )
        
        # Send reminders for deadlines within 3 days
        reminders = self.manager.send_deadline_reminders(days_before=3, current_time=now)
        
        assert len(reminders) == 1
        assert "reviewer1" in reminders
        assert assignment1.reminder_sent
        assert not assignment2.reminder_sent
        
        # Second call should not send reminder again
        reminders2 = self.manager.send_deadline_reminders(days_before=3, current_time=now)
        assert len(reminders2) == 0
    
    def test_get_venue_deadline_performance(self):
        """Test getting venue deadline performance statistics."""
        now = datetime.now()
        
        # Create assignments for the same venue
        assignments = []
        for i in range(3):
            assignment = self.manager.assign_review_deadline(
                reviewer_id=f"reviewer{i}",
                paper_id=f"paper{i}",
                venue=self.venue,
                assignment_date=now - timedelta(weeks=5)
            )
            assignments.append(assignment)
        
        # Submit with different timing
        review_times = [
            assignment.deadline - timedelta(hours=1),  # On time
            assignment.deadline + timedelta(days=2),   # Late
            assignment.deadline - timedelta(hours=2),  # On time
        ]
        
        for assignment, submission_time in zip(assignments, review_times):
            review = StructuredReview(
                reviewer_id=assignment.reviewer_id,
                paper_id=assignment.paper_id,
                venue_id="venue1"
            )
            self.manager.submit_review(assignment.assignment_id, review, submission_time)
        
        performance = self.manager.get_venue_deadline_performance("venue1")
        
        assert performance['total_reviews'] == 3
        assert performance['on_time_rate'] == 2/3  # 2 out of 3
        assert abs(performance['average_days_late'] - 2/3) < 0.01  # (0 + 2 + 0) / 3
        assert performance['status_breakdown']['On Time'] == 2
        assert performance['status_breakdown']['Late'] == 1
    
    def test_update_venue_deadline_weeks(self):
        """Test updating venue deadline weeks."""
        manager = DeadlineManager()
        
        # Update deadline weeks
        manager.update_venue_deadline_weeks(VenueType.MID_CONFERENCE, 5)
        assert manager.venue_deadline_weeks[VenueType.MID_CONFERENCE] == 5
        
        # Test validation
        with pytest.raises(ValidationError):
            manager.update_venue_deadline_weeks(VenueType.MID_CONFERENCE, 1)  # Too low
        
        with pytest.raises(ValidationError):
            manager.update_venue_deadline_weeks(VenueType.MID_CONFERENCE, 10)  # Too high
    
    def test_clear_old_assignments(self):
        """Test clearing old assignments."""
        now = datetime.now()
        
        # Create old and recent assignments
        old_assignment = self.manager.assign_review_deadline(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue=self.venue,
            assignment_date=now - timedelta(days=400)  # Very old
        )
        
        recent_assignment = self.manager.assign_review_deadline(
            reviewer_id="reviewer2",
            paper_id="paper2",
            venue=self.venue,
            assignment_date=now - timedelta(days=30)  # Recent
        )
        
        assert len(self.manager.deadline_assignments) == 2
        
        # Clear assignments older than 365 days
        self.manager.clear_old_assignments(days_old=365)
        
        assert len(self.manager.deadline_assignments) == 1
        assert recent_assignment.assignment_id in self.manager.deadline_assignments
        assert old_assignment.assignment_id not in self.manager.deadline_assignments


if __name__ == "__main__":
    pytest.main([__file__])