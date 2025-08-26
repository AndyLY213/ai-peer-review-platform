"""
Unit tests for the WorkloadTracker class.

Tests reviewer capacity monitoring, availability status checking based on current workload,
logic for maximum reviews per month (2-8 based on seniority), and comprehensive
availability tracking functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.enhancements.workload_tracker import (
    WorkloadTracker, ReviewAssignment, AvailabilityStatus, 
    WorkloadConfiguration, ReviewerWorkload
)
from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel
from src.core.exceptions import ValidationError


class TestReviewAssignment:
    """Test ReviewAssignment functionality."""
    
    def test_review_assignment_creation(self):
        """Test creating a review assignment."""
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        
        assignment = ReviewAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline,
            estimated_hours=10
        )
        
        assert assignment.reviewer_id == "reviewer1"
        assert assignment.paper_id == "paper1"
        assert assignment.venue_id == "venue1"
        assert assignment.assigned_date == now
        assert assignment.deadline == deadline
        assert assignment.estimated_hours == 10
        assert not assignment.is_completed
        assert assignment.actual_hours is None
    
    def test_is_active(self):
        """Test active assignment detection."""
        now = datetime.now()
        deadline = now + timedelta(weeks=1)
        
        assignment = ReviewAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline
        )
        
        # Should be active when not completed and not overdue
        assert assignment.is_active(now)
        
        # Should not be active when completed
        assignment.is_completed = True
        assert not assignment.is_active(now)
        
        # Should not be active when very overdue (>2 weeks)
        assignment.is_completed = False
        very_overdue_time = deadline + timedelta(weeks=3)
        assert not assignment.is_active(very_overdue_time)
        
        # Should still be active when slightly overdue (<2 weeks)
        slightly_overdue_time = deadline + timedelta(days=5)
        assert assignment.is_active(slightly_overdue_time)
    
    def test_days_remaining(self):
        """Test days remaining calculation."""
        now = datetime.now()
        deadline = now + timedelta(days=5)
        
        assignment = ReviewAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=deadline
        )
        
        assert assignment.days_remaining(now) == 5
        
        # Past deadline should return 0
        past_time = deadline + timedelta(days=1)
        assert assignment.days_remaining(past_time) == 0


class TestWorkloadConfiguration:
    """Test WorkloadConfiguration functionality."""
    
    def test_default_configuration(self):
        """Test default workload configuration."""
        config = WorkloadConfiguration()
        
        # Test default max reviews per month
        assert config.get_max_reviews(ResearcherLevel.GRADUATE_STUDENT) == 2
        assert config.get_max_reviews(ResearcherLevel.POSTDOC) == 3
        assert config.get_max_reviews(ResearcherLevel.ASSISTANT_PROF) == 4
        assert config.get_max_reviews(ResearcherLevel.ASSOCIATE_PROF) == 6
        assert config.get_max_reviews(ResearcherLevel.FULL_PROF) == 8
        assert config.get_max_reviews(ResearcherLevel.EMERITUS) == 3
        
        # Test default thresholds
        assert config.busy_threshold == 0.7
        assert config.overloaded_threshold == 1.0
        assert config.base_hours_per_review == 8
    
    def test_get_estimated_hours(self):
        """Test estimated hours calculation for different venue types."""
        config = WorkloadConfiguration()
        
        assert config.get_estimated_hours('TOP_CONFERENCE') == 12  # 8 * 1.5
        assert config.get_estimated_hours('MID_CONFERENCE') == 8   # 8 * 1.0
        assert config.get_estimated_hours('LOW_CONFERENCE') == 6   # 8 * 0.8
        assert config.get_estimated_hours('TOP_JOURNAL') == 16     # 8 * 2.0
        assert config.get_estimated_hours('WORKSHOP') == 4         # 8 * 0.5
        assert config.get_estimated_hours('UNKNOWN_VENUE') == 8    # Default


class TestReviewerWorkload:
    """Test ReviewerWorkload functionality."""
    
    def test_reviewer_workload_creation(self):
        """Test creating reviewer workload."""
        workload = ReviewerWorkload(reviewer_id="reviewer1")
        
        assert workload.reviewer_id == "reviewer1"
        assert len(workload.current_assignments) == 0
        assert workload.completed_this_month == 0
        assert workload.total_hours_this_month == 0
        assert workload.availability_status == AvailabilityStatus.AVAILABLE
        assert workload.unavailable_until is None
        assert workload.sabbatical_period is None
    
    def test_get_active_assignments(self):
        """Test getting active assignments."""
        now = datetime.now()
        workload = ReviewerWorkload(reviewer_id="reviewer1")
        
        # Add active assignment
        active_assignment = ReviewAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=now + timedelta(weeks=1)
        )
        workload.current_assignments.append(active_assignment)
        
        # Add completed assignment
        completed_assignment = ReviewAssignment(
            reviewer_id="reviewer1",
            paper_id="paper2",
            venue_id="venue1",
            assigned_date=now,
            deadline=now + timedelta(weeks=1),
            is_completed=True
        )
        workload.current_assignments.append(completed_assignment)
        
        active = workload.get_active_assignments(now)
        assert len(active) == 1
        assert active[0].paper_id == "paper1"
    
    def test_get_total_estimated_hours(self):
        """Test total estimated hours calculation."""
        now = datetime.now()
        workload = ReviewerWorkload(reviewer_id="reviewer1")
        
        # Add assignments with different hours
        assignment1 = ReviewAssignment(
            reviewer_id="reviewer1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=now,
            deadline=now + timedelta(weeks=1),
            estimated_hours=8
        )
        assignment2 = ReviewAssignment(
            reviewer_id="reviewer1",
            paper_id="paper2",
            venue_id="venue1",
            assigned_date=now,
            deadline=now + timedelta(weeks=1),
            estimated_hours=12
        )
        
        workload.current_assignments.extend([assignment1, assignment2])
        
        assert workload.get_total_estimated_hours(now) == 20
    
    def test_get_workload_ratio(self):
        """Test workload ratio calculation."""
        now = datetime.now()
        workload = ReviewerWorkload(reviewer_id="reviewer1")
        
        # Add 2 active assignments
        for i in range(2):
            assignment = ReviewAssignment(
                reviewer_id="reviewer1",
                paper_id=f"paper{i}",
                venue_id="venue1",
                assigned_date=now,
                deadline=now + timedelta(weeks=1)
            )
            workload.current_assignments.append(assignment)
        
        # With max_reviews=4, ratio should be 2/4 = 0.5
        assert workload.get_workload_ratio(4, now) == 0.5
        
        # With max_reviews=2, ratio should be 2/2 = 1.0
        assert workload.get_workload_ratio(2, now) == 1.0


class TestWorkloadTracker:
    """Test WorkloadTracker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = WorkloadTracker()
        self.researcher = EnhancedResearcher(
            id="researcher1",
            name="Test Researcher",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF
        )
    
    def test_initialization(self):
        """Test WorkloadTracker initialization."""
        tracker = WorkloadTracker()
        
        assert isinstance(tracker.config, WorkloadConfiguration)
        assert len(tracker.reviewer_workloads) == 0
    
    def test_register_reviewer(self):
        """Test registering a reviewer."""
        self.tracker.register_reviewer(self.researcher)
        
        assert "researcher1" in self.tracker.reviewer_workloads
        workload = self.tracker.reviewer_workloads["researcher1"]
        assert workload.reviewer_id == "researcher1"
        assert workload.availability_status == AvailabilityStatus.AVAILABLE
    
    def test_register_reviewer_with_custom_max(self):
        """Test registering reviewer with custom maximum reviews."""
        self.tracker.register_reviewer(self.researcher, custom_max_reviews=6)
        
        workload = self.tracker.reviewer_workloads["researcher1"]
        assert workload.custom_max_reviews == 6
    
    def test_assign_review(self):
        """Test assigning a review."""
        self.tracker.register_reviewer(self.researcher)
        
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        
        assignment = self.tracker.assign_review(
            reviewer_id="researcher1",
            paper_id="paper1",
            venue_id="venue1",
            venue_type="MID_CONFERENCE",
            deadline=deadline,
            assignment_date=now
        )
        
        assert assignment.reviewer_id == "researcher1"
        assert assignment.paper_id == "paper1"
        assert assignment.venue_id == "venue1"
        assert assignment.deadline == deadline
        assert assignment.estimated_hours == 8  # MID_CONFERENCE default
        
        # Check that assignment was added to workload
        workload = self.tracker.reviewer_workloads["researcher1"]
        assert len(workload.current_assignments) == 1
        assert workload.current_assignments[0].assignment_id == assignment.assignment_id
    
    def test_assign_review_unregistered_reviewer(self):
        """Test assigning review to unregistered reviewer raises error."""
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        
        with pytest.raises(ValidationError):
            self.tracker.assign_review(
                reviewer_id="unregistered",
                paper_id="paper1",
                venue_id="venue1",
                venue_type="MID_CONFERENCE",
                deadline=deadline
            )
    
    def test_complete_review(self):
        """Test completing a review."""
        self.tracker.register_reviewer(self.researcher)
        
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        
        assignment = self.tracker.assign_review(
            reviewer_id="researcher1",
            paper_id="paper1",
            venue_id="venue1",
            venue_type="MID_CONFERENCE",
            deadline=deadline,
            assignment_date=now
        )
        
        completion_time = now + timedelta(days=5)
        self.tracker.complete_review(
            reviewer_id="researcher1",
            assignment_id=assignment.assignment_id,
            actual_hours=10,
            completion_date=completion_time
        )
        
        assert assignment.is_completed
        assert assignment.completion_date == completion_time
        assert assignment.actual_hours == 10
        
        # Check monthly statistics were updated
        workload = self.tracker.reviewer_workloads["researcher1"]
        assert workload.completed_this_month == 1
        assert workload.total_hours_this_month == 10
    
    def test_complete_review_invalid_assignment(self):
        """Test completing review with invalid assignment ID."""
        self.tracker.register_reviewer(self.researcher)
        
        with pytest.raises(ValidationError):
            self.tracker.complete_review(
                reviewer_id="researcher1",
                assignment_id="invalid_id",
                actual_hours=10
            )
    
    def test_check_availability_available(self):
        """Test checking availability for available reviewer."""
        self.tracker.register_reviewer(self.researcher)
        
        is_available, status, reason = self.tracker.check_availability(
            "researcher1", self.researcher
        )
        
        assert is_available
        assert status == AvailabilityStatus.AVAILABLE
        assert "Available" in reason
    
    def test_check_availability_busy(self):
        """Test checking availability for busy reviewer."""
        self.tracker.register_reviewer(self.researcher)
        
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        
        # Assign 3 reviews (75% of capacity for assistant prof with max 4)
        for i in range(3):
            self.tracker.assign_review(
                reviewer_id="researcher1",
                paper_id=f"paper{i}",
                venue_id="venue1",
                venue_type="MID_CONFERENCE",
                deadline=deadline
            )
        
        is_available, status, reason = self.tracker.check_availability(
            "researcher1", self.researcher
        )
        
        assert is_available
        assert status == AvailabilityStatus.BUSY
        assert "Busy but available" in reason
    
    def test_check_availability_overloaded(self):
        """Test checking availability for overloaded reviewer."""
        self.tracker.register_reviewer(self.researcher)
        
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        
        # Assign 4 reviews (100% of capacity for assistant prof)
        for i in range(4):
            self.tracker.assign_review(
                reviewer_id="researcher1",
                paper_id=f"paper{i}",
                venue_id="venue1",
                venue_type="MID_CONFERENCE",
                deadline=deadline
            )
        
        is_available, status, reason = self.tracker.check_availability(
            "researcher1", self.researcher
        )
        
        assert not is_available
        assert status == AvailabilityStatus.OVERLOADED
        assert "At capacity" in reason
    
    def test_check_availability_unavailable(self):
        """Test checking availability for explicitly unavailable reviewer."""
        self.tracker.register_reviewer(self.researcher)
        
        future_date = datetime.now() + timedelta(weeks=2)
        self.tracker.set_reviewer_unavailable("researcher1", until_date=future_date)
        
        is_available, status, reason = self.tracker.check_availability(
            "researcher1", self.researcher
        )
        
        assert not is_available
        assert status == AvailabilityStatus.UNAVAILABLE
        assert "Unavailable until" in reason
    
    def test_check_availability_sabbatical(self):
        """Test checking availability for reviewer on sabbatical."""
        self.tracker.register_reviewer(self.researcher)
        
        now = datetime.now()
        start_date = now - timedelta(weeks=1)
        end_date = now + timedelta(weeks=10)
        
        self.tracker.set_reviewer_sabbatical("researcher1", start_date, end_date)
        
        is_available, status, reason = self.tracker.check_availability(
            "researcher1", self.researcher, now
        )
        
        assert not is_available
        assert status == AvailabilityStatus.ON_SABBATICAL
        assert "On sabbatical until" in reason
    
    def test_check_availability_auto_register(self):
        """Test that check_availability auto-registers unknown reviewers."""
        # Don't register reviewer first
        is_available, status, reason = self.tracker.check_availability(
            "researcher1", self.researcher
        )
        
        # Should auto-register and be available
        assert is_available
        assert status == AvailabilityStatus.AVAILABLE
        assert "researcher1" in self.tracker.reviewer_workloads
    
    def test_get_reviewer_capacity(self):
        """Test getting detailed reviewer capacity information."""
        self.tracker.register_reviewer(self.researcher)
        
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        
        # Assign 2 reviews
        for i in range(2):
            self.tracker.assign_review(
                reviewer_id="researcher1",
                paper_id=f"paper{i}",
                venue_id="venue1",
                venue_type="MID_CONFERENCE",
                deadline=deadline
            )
        
        capacity = self.tracker.get_reviewer_capacity("researcher1", self.researcher)
        
        assert capacity['reviewer_id'] == "researcher1"
        assert capacity['seniority_level'] == "Assistant Prof"
        assert capacity['max_reviews_per_month'] == 4
        assert capacity['current_active_reviews'] == 2
        assert capacity['workload_ratio'] == 0.5
        assert capacity['is_available']
        assert len(capacity['active_assignments']) == 2
    
    def test_set_reviewer_unavailable(self):
        """Test setting reviewer as unavailable."""
        self.tracker.register_reviewer(self.researcher)
        
        future_date = datetime.now() + timedelta(weeks=2)
        self.tracker.set_reviewer_unavailable("researcher1", until_date=future_date)
        
        workload = self.tracker.reviewer_workloads["researcher1"]
        assert workload.availability_status == AvailabilityStatus.UNAVAILABLE
        assert workload.unavailable_until == future_date
    
    def test_set_reviewer_sabbatical(self):
        """Test setting reviewer on sabbatical."""
        self.tracker.register_reviewer(self.researcher)
        
        start_date = datetime.now()
        end_date = start_date + timedelta(weeks=26)  # 6 months
        
        self.tracker.set_reviewer_sabbatical("researcher1", start_date, end_date)
        
        workload = self.tracker.reviewer_workloads["researcher1"]
        assert workload.sabbatical_period == (start_date, end_date)
    
    def test_set_reviewer_available(self):
        """Test setting reviewer as available again."""
        self.tracker.register_reviewer(self.researcher)
        
        # First make unavailable
        self.tracker.set_reviewer_unavailable("researcher1")
        
        # Then make available again
        self.tracker.set_reviewer_available("researcher1")
        
        workload = self.tracker.reviewer_workloads["researcher1"]
        assert workload.availability_status == AvailabilityStatus.AVAILABLE
        assert workload.unavailable_until is None
        assert workload.sabbatical_period is None
    
    def test_get_available_reviewers(self):
        """Test getting list of available reviewers."""
        # Create multiple researchers with different levels
        researchers = [
            EnhancedResearcher(
                id=f"researcher{i}",
                name=f"Researcher {i}",
                specialty="AI",
                level=ResearcherLevel.ASSISTANT_PROF
            )
            for i in range(3)
        ]
        
        # Register all researchers
        for researcher in researchers:
            self.tracker.register_reviewer(researcher)
        
        # Make one unavailable
        self.tracker.set_reviewer_unavailable("researcher1")
        
        # Verify the unavailable status is set
        workload1 = self.tracker.reviewer_workloads["researcher1"]
        assert workload1.availability_status == AvailabilityStatus.UNAVAILABLE
        
        # Overload another
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        for i in range(4):  # Max capacity for assistant prof
            self.tracker.assign_review(
                reviewer_id="researcher2",
                paper_id=f"paper{i}",
                venue_id="venue1",
                venue_type="MID_CONFERENCE",
                deadline=deadline
            )
        
        available = self.tracker.get_available_reviewers(researchers)
        
        # Only researcher0 should be available (researcher1 is unavailable, researcher2 is overloaded)
        available_ids = [reviewer_id for reviewer_id, _ in available]
        assert len(available) == 1
        assert "researcher0" in available_ids
        assert available[0][1]['is_available']
    
    def test_get_available_reviewers_min_capacity(self):
        """Test getting available reviewers with minimum capacity requirement."""
        researchers = [
            EnhancedResearcher(
                id=f"researcher{i}",
                name=f"Researcher {i}",
                specialty="AI",
                level=ResearcherLevel.ASSISTANT_PROF
            )
            for i in range(2)
        ]
        
        for researcher in researchers:
            self.tracker.register_reviewer(researcher)
        
        # Load one reviewer to 75% capacity
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        for i in range(3):
            self.tracker.assign_review(
                reviewer_id="researcher0",
                paper_id=f"paper{i}",
                venue_id="venue1",
                venue_type="MID_CONFERENCE",
                deadline=deadline
            )
        
        # With min_capacity=0.5 (50% available), only researcher1 should qualify
        available = self.tracker.get_available_reviewers(researchers, min_capacity=0.5)
        
        assert len(available) == 1
        assert available[0][0] == "researcher1"
    
    def test_reset_monthly_stats(self):
        """Test resetting monthly statistics."""
        self.tracker.register_reviewer(self.researcher)
        
        # Set some monthly stats
        workload = self.tracker.reviewer_workloads["researcher1"]
        workload.completed_this_month = 5
        workload.total_hours_this_month = 40
        
        self.tracker.reset_monthly_stats()
        
        assert workload.completed_this_month == 0
        assert workload.total_hours_this_month == 0
    
    def test_get_workload_statistics(self):
        """Test getting overall workload statistics."""
        # Create multiple researchers
        researchers = [
            EnhancedResearcher(
                id=f"researcher{i}",
                name=f"Researcher {i}",
                specialty="AI",
                level=ResearcherLevel.ASSISTANT_PROF
            )
            for i in range(3)
        ]
        
        for researcher in researchers:
            self.tracker.register_reviewer(researcher)
        
        # Make one unavailable
        self.tracker.set_reviewer_unavailable("researcher1")
        
        # Add some assignments
        now = datetime.now()
        deadline = now + timedelta(weeks=2)
        self.tracker.assign_review(
            reviewer_id="researcher0",
            paper_id="paper1",
            venue_id="venue1",
            venue_type="MID_CONFERENCE",
            deadline=deadline
        )
        
        stats = self.tracker.get_workload_statistics()
        
        assert stats['total_reviewers'] == 3
        assert stats['availability_breakdown']['Available'] == 2
        assert stats['availability_breakdown']['Unavailable'] == 1
        assert stats['total_active_assignments'] == 1
    
    def test_update_max_reviews_for_level(self):
        """Test updating maximum reviews for a seniority level."""
        tracker = WorkloadTracker()
        
        # Update max reviews for assistant professors
        tracker.update_max_reviews_for_level(ResearcherLevel.ASSISTANT_PROF, 6)
        
        assert tracker.config.get_max_reviews(ResearcherLevel.ASSISTANT_PROF) == 6
        
        # Test validation
        with pytest.raises(ValidationError):
            tracker.update_max_reviews_for_level(ResearcherLevel.ASSISTANT_PROF, 1)  # Too low
        
        with pytest.raises(ValidationError):
            tracker.update_max_reviews_for_level(ResearcherLevel.ASSISTANT_PROF, 10)  # Too high
    
    def test_cleanup_old_assignments(self):
        """Test cleaning up old completed assignments."""
        self.tracker.register_reviewer(self.researcher)
        
        now = datetime.now()
        old_date = now - timedelta(days=100)
        recent_date = now - timedelta(days=30)
        
        # Create old completed assignment
        old_assignment = ReviewAssignment(
            reviewer_id="researcher1",
            paper_id="paper1",
            venue_id="venue1",
            assigned_date=old_date,
            deadline=old_date + timedelta(weeks=2),
            is_completed=True,
            completion_date=old_date + timedelta(days=5)
        )
        
        # Create recent completed assignment
        recent_assignment = ReviewAssignment(
            reviewer_id="researcher1",
            paper_id="paper2",
            venue_id="venue1",
            assigned_date=recent_date,
            deadline=recent_date + timedelta(weeks=2),
            is_completed=True,
            completion_date=recent_date + timedelta(days=5)
        )
        
        workload = self.tracker.reviewer_workloads["researcher1"]
        workload.current_assignments.extend([old_assignment, recent_assignment])
        
        assert len(workload.current_assignments) == 2
        
        # Clean up assignments older than 90 days
        self.tracker.cleanup_old_assignments(days_old=90)
        
        assert len(workload.current_assignments) == 1
        assert workload.current_assignments[0].paper_id == "paper2"
    
    def test_different_seniority_levels(self):
        """Test workload limits for different seniority levels."""
        levels_and_limits = [
            (ResearcherLevel.GRADUATE_STUDENT, 2),
            (ResearcherLevel.POSTDOC, 3),
            (ResearcherLevel.ASSISTANT_PROF, 4),
            (ResearcherLevel.ASSOCIATE_PROF, 6),
            (ResearcherLevel.FULL_PROF, 8),
            (ResearcherLevel.EMERITUS, 3)
        ]
        
        for level, expected_max in levels_and_limits:
            researcher = EnhancedResearcher(
                id=f"researcher_{level.value}",
                name=f"Test {level.value}",
                specialty="AI",
                level=level
            )
            
            self.tracker.register_reviewer(researcher)
            capacity = self.tracker.get_reviewer_capacity(researcher.id, researcher)
            
            assert capacity['max_reviews_per_month'] == expected_max


if __name__ == "__main__":
    pytest.main([__file__])