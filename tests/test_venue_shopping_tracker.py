"""
Unit tests for VenueShoppingTracker

Tests venue shopping tracking functionality including submission recording,
pattern detection, and strategic behavior analysis.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.enhancements.venue_shopping_tracker import (
    VenueShoppingTracker, SubmissionRecord, VenueShoppingPattern,
    ResearcherShoppingProfile, SubmissionOutcome
)
from src.data.enhanced_models import VenueType
from src.core.exceptions import ValidationError


class TestSubmissionRecord:
    """Test SubmissionRecord functionality."""
    
    def test_submission_record_creation(self):
        """Test creating a submission record."""
        record = SubmissionRecord(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        assert record.paper_id == "paper_001"
        assert record.researcher_id == "researcher_001"
        assert record.venue_id == "venue_001"
        assert record.venue_type == VenueType.TOP_CONFERENCE
        assert record.venue_prestige == 9
        assert record.outcome == SubmissionOutcome.PENDING
        assert record.average_score == 0.0
    
    def test_submission_record_with_scores(self):
        """Test submission record with review scores."""
        record = SubmissionRecord(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.MID_CONFERENCE,
            venue_prestige=6,
            review_scores=[7.5, 6.0, 8.0]
        )
        
        assert record.review_scores == [7.5, 6.0, 8.0]
        assert abs(record.average_score - 7.166666666666667) < 0.01
        assert abs(record.average_score - 7.166666666666667) < 0.01


class TestVenueShoppingPattern:
    """Test VenueShoppingPattern functionality."""
    
    def test_empty_pattern(self):
        """Test pattern with no submissions."""
        pattern = VenueShoppingPattern(
            paper_id="paper_001",
            researcher_id="researcher_001"
        )
        
        assert pattern.venue_downgrade_count == 0
        assert pattern.total_venues_tried == 0
        assert pattern.time_span_days == 0
        assert pattern.shopping_score == 0.0
    
    def test_single_submission_pattern(self):
        """Test pattern with single submission."""
        submission = SubmissionRecord(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        pattern = VenueShoppingPattern(
            paper_id="paper_001",
            researcher_id="researcher_001",
            submission_sequence=[submission]
        )
        
        assert pattern.venue_downgrade_count == 0
        assert pattern.total_venues_tried == 1
        assert pattern.time_span_days == 0
        assert pattern.shopping_score == 0.0
    
    def test_venue_downgrade_pattern(self):
        """Test pattern with venue downgrades."""
        base_time = datetime.now()
        
        submissions = [
            SubmissionRecord(
                paper_id="paper_001",
                researcher_id="researcher_001",
                venue_id="venue_001",
                venue_type=VenueType.TOP_CONFERENCE,
                venue_prestige=9,
                submission_date=base_time
            ),
            SubmissionRecord(
                paper_id="paper_001",
                researcher_id="researcher_001",
                venue_id="venue_002",
                venue_type=VenueType.MID_CONFERENCE,
                venue_prestige=6,
                submission_date=base_time + timedelta(days=30)
            ),
            SubmissionRecord(
                paper_id="paper_001",
                researcher_id="researcher_001",
                venue_id="venue_003",
                venue_type=VenueType.LOW_CONFERENCE,
                venue_prestige=4,
                submission_date=base_time + timedelta(days=60)
            )
        ]
        
        pattern = VenueShoppingPattern(
            paper_id="paper_001",
            researcher_id="researcher_001",
            submission_sequence=submissions
        )
        
        assert pattern.venue_downgrade_count == 2  # Two downgrades
        assert pattern.total_venues_tried == 3
        assert pattern.time_span_days == 60
        assert pattern.shopping_score > 0.0  # Should have positive shopping score
    
    def test_shopping_score_calculation(self):
        """Test shopping score calculation with various scenarios."""
        base_time = datetime.now()
        
        # High shopping scenario: 5 venues, 3 downgrades, 1 year span
        submissions = []
        prestiges = [9, 7, 5, 3, 2]
        for i, prestige in enumerate(prestiges):
            submissions.append(SubmissionRecord(
                paper_id="paper_001",
                researcher_id="researcher_001",
                venue_id=f"venue_{i+1:03d}",
                venue_type=VenueType.MID_CONFERENCE,
                venue_prestige=prestige,
                submission_date=base_time + timedelta(days=i*90)
            ))
        
        pattern = VenueShoppingPattern(
            paper_id="paper_001",
            researcher_id="researcher_001",
            submission_sequence=submissions
        )
        
        # Should have high shopping score
        assert pattern.shopping_score > 0.7
        assert pattern.venue_downgrade_count == 4
        assert pattern.total_venues_tried == 5


class TestResearcherShoppingProfile:
    """Test ResearcherShoppingProfile functionality."""
    
    def test_empty_profile(self):
        """Test profile with no patterns."""
        profile = ResearcherShoppingProfile(researcher_id="researcher_001")
        profile.update_profile([])
        
        assert profile.total_papers_submitted == 0
        assert profile.papers_with_shopping == 0
        assert profile.average_venues_per_paper == 1.0
        assert profile.shopping_tendency == 0.0
    
    def test_profile_update(self):
        """Test profile update with shopping patterns."""
        # Create patterns with different shopping behaviors
        patterns = []
        
        # Pattern 1: High shopping - manually set values after __post_init__
        pattern1 = VenueShoppingPattern(
            paper_id="paper_001",
            researcher_id="researcher_001"
        )
        pattern1.total_venues_tried = 4
        pattern1.shopping_score = 0.8
        patterns.append(pattern1)
        
        # Pattern 2: Low shopping
        pattern2 = VenueShoppingPattern(
            paper_id="paper_002",
            researcher_id="researcher_001"
        )
        pattern2.total_venues_tried = 2
        pattern2.shopping_score = 0.1
        patterns.append(pattern2)
        
        # Pattern 3: No shopping
        pattern3 = VenueShoppingPattern(
            paper_id="paper_003",
            researcher_id="researcher_001"
        )
        pattern3.total_venues_tried = 1
        pattern3.shopping_score = 0.0
        patterns.append(pattern3)
        
        profile = ResearcherShoppingProfile(researcher_id="researcher_001")
        profile.update_profile(patterns)
        
        assert profile.total_papers_submitted == 3
        assert profile.papers_with_shopping == 1  # Only pattern1 > 0.1 threshold
        assert profile.average_venues_per_paper == (4 + 2 + 1) / 3
        assert profile.shopping_tendency == 1/3  # 1 out of 3 papers
        assert abs(profile.average_shopping_score - 0.3) < 0.01  # (0.8 + 0.1 + 0.0) / 3


class TestVenueShoppingTracker:
    """Test VenueShoppingTracker functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = VenueShoppingTracker(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        assert self.tracker.data_dir.exists()
        assert len(self.tracker.submission_records) == 0
        assert len(self.tracker.shopping_patterns) == 0
        assert len(self.tracker.researcher_profiles) == 0
    
    def test_record_submission(self):
        """Test recording a submission."""
        submission_id = self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        assert submission_id is not None
        assert "paper_001" in self.tracker.submission_records
        assert len(self.tracker.submission_records["paper_001"]) == 1
        
        submission = self.tracker.submission_records["paper_001"][0]
        assert submission.paper_id == "paper_001"
        assert submission.researcher_id == "researcher_001"
        assert submission.venue_id == "venue_001"
        assert submission.venue_type == VenueType.TOP_CONFERENCE
        assert submission.venue_prestige == 9
    
    def test_multiple_submissions_same_paper(self):
        """Test recording multiple submissions for same paper."""
        # First submission
        self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        # Second submission
        self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_002",
            venue_type=VenueType.MID_CONFERENCE,
            venue_prestige=6
        )
        
        assert len(self.tracker.submission_records["paper_001"]) == 2
        
        submissions = self.tracker.submission_records["paper_001"]
        assert submissions[0].venue_prestige == 9
        assert submissions[1].venue_prestige == 6
    
    def test_update_submission_outcome(self):
        """Test updating submission outcome."""
        submission_id = self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        # Update outcome
        self.tracker.update_submission_outcome(
            paper_id="paper_001",
            submission_id=submission_id,
            outcome=SubmissionOutcome.REJECTED,
            review_scores=[5.0, 6.0, 4.0]
        )
        
        submission = self.tracker.submission_records["paper_001"][0]
        assert submission.outcome == SubmissionOutcome.REJECTED
        assert submission.review_scores == [5.0, 6.0, 4.0]
        assert submission.average_score == 5.0
        assert submission.outcome_date is not None
    
    def test_update_nonexistent_submission(self):
        """Test updating nonexistent submission raises error."""
        with pytest.raises(ValidationError):
            self.tracker.update_submission_outcome(
                paper_id="nonexistent",
                submission_id="fake_id",
                outcome=SubmissionOutcome.REJECTED
            )
    
    def test_shopping_pattern_creation(self):
        """Test shopping pattern creation after rejection."""
        # Record first submission
        submission_id1 = self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        # Record second submission
        submission_id2 = self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_002",
            venue_type=VenueType.MID_CONFERENCE,
            venue_prestige=6
        )
        
        # Update first as rejected
        self.tracker.update_submission_outcome(
            paper_id="paper_001",
            submission_id=submission_id1,
            outcome=SubmissionOutcome.REJECTED,
            review_scores=[4.0, 5.0, 3.0]
        )
        
        # Should create shopping pattern
        assert "paper_001" in self.tracker.shopping_patterns
        pattern = self.tracker.shopping_patterns["paper_001"]
        assert pattern.paper_id == "paper_001"
        assert pattern.researcher_id == "researcher_001"
        assert len(pattern.submission_sequence) == 2
    
    def test_detect_venue_shopping(self):
        """Test venue shopping detection."""
        # Create shopping scenario
        paper_id = "paper_001"
        researcher_id = "researcher_001"
        
        # Multiple submissions with downgrades
        submissions = []
        prestiges = [9, 6, 4]
        for i, prestige in enumerate(prestiges):
            submission_id = self.tracker.record_submission(
                paper_id=paper_id,
                researcher_id=researcher_id,
                venue_id=f"venue_{i+1:03d}",
                venue_type=VenueType.MID_CONFERENCE,
                venue_prestige=prestige
            )
            submissions.append(submission_id)
        
        # Mark first two as rejected
        for i in range(2):
            self.tracker.update_submission_outcome(
                paper_id=paper_id,
                submission_id=submissions[i],
                outcome=SubmissionOutcome.REJECTED,
                review_scores=[3.0, 4.0, 2.0]
            )
        
        # Detect shopping
        pattern = self.tracker.detect_venue_shopping(paper_id)
        assert pattern is not None
        assert pattern.venue_downgrade_count > 0
        assert pattern.shopping_score >= self.tracker.min_shopping_score
    
    def test_no_shopping_detection_single_venue(self):
        """Test no shopping detected for single venue submission."""
        submission_id = self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        self.tracker.update_submission_outcome(
            paper_id="paper_001",
            submission_id=submission_id,
            outcome=SubmissionOutcome.ACCEPTED,
            review_scores=[8.0, 9.0, 7.0]
        )
        
        pattern = self.tracker.detect_venue_shopping("paper_001")
        assert pattern is None  # No shopping for single venue
    
    def test_researcher_profile_creation(self):
        """Test researcher profile creation."""
        paper_id = "paper_001"
        researcher_id = "researcher_001"
        
        # Create shopping scenario
        submissions = []
        prestiges = [9, 6, 4]
        for i, prestige in enumerate(prestiges):
            submission_id = self.tracker.record_submission(
                paper_id=paper_id,
                researcher_id=researcher_id,
                venue_id=f"venue_{i+1:03d}",
                venue_type=VenueType.MID_CONFERENCE,
                venue_prestige=prestige
            )
            submissions.append(submission_id)
        
        # Mark as rejected to trigger profile update
        self.tracker.update_submission_outcome(
            paper_id=paper_id,
            submission_id=submissions[0],
            outcome=SubmissionOutcome.REJECTED
        )
        
        # Check profile creation
        profile = self.tracker.get_researcher_shopping_behavior(researcher_id)
        assert profile is not None
        assert profile.researcher_id == researcher_id
        assert profile.total_papers_submitted >= 1
    
    def test_venue_downgrade_patterns_analysis(self):
        """Test venue downgrade patterns analysis."""
        # Create multiple papers with downgrade patterns
        for paper_num in range(3):
            paper_id = f"paper_{paper_num:03d}"
            researcher_id = "researcher_001"
            
            # Create downgrade pattern
            prestiges = [9, 6, 4]
            submissions = []
            for i, prestige in enumerate(prestiges):
                submission_id = self.tracker.record_submission(
                    paper_id=paper_id,
                    researcher_id=researcher_id,
                    venue_id=f"venue_{paper_num}_{i}",
                    venue_type=VenueType.MID_CONFERENCE,
                    venue_prestige=prestige
                )
                submissions.append(submission_id)
            
            # Mark first two as rejected
            for i in range(2):
                self.tracker.update_submission_outcome(
                    paper_id=paper_id,
                    submission_id=submissions[i],
                    outcome=SubmissionOutcome.REJECTED
                )
        
        # Analyze patterns
        patterns = self.tracker.analyze_venue_downgrade_patterns()
        assert "researcher_001" in patterns
        assert len(patterns["researcher_001"]) > 0
    
    def test_shopping_statistics(self):
        """Test shopping statistics calculation."""
        # Initially empty
        stats = self.tracker.get_shopping_statistics()
        assert stats['total_papers'] == 0
        assert stats['shopping_rate'] == 0.0
        
        # Add some papers with shopping
        for paper_num in range(3):
            paper_id = f"paper_{paper_num:03d}"
            researcher_id = "researcher_001"
            
            # Create varying levels of shopping
            num_venues = 2 + paper_num  # 2, 3, 4 venues
            prestiges = list(range(9, 9-num_venues, -1))  # Decreasing prestige
            
            submissions = []
            for i, prestige in enumerate(prestiges):
                submission_id = self.tracker.record_submission(
                    paper_id=paper_id,
                    researcher_id=researcher_id,
                    venue_id=f"venue_{paper_num}_{i}",
                    venue_type=VenueType.MID_CONFERENCE,
                    venue_prestige=prestige
                )
                submissions.append(submission_id)
            
            # Mark all but last as rejected
            for i in range(len(submissions) - 1):
                self.tracker.update_submission_outcome(
                    paper_id=paper_id,
                    submission_id=submissions[i],
                    outcome=SubmissionOutcome.REJECTED
                )
        
        # Check statistics
        stats = self.tracker.get_shopping_statistics()
        assert stats['total_papers'] == 3
        assert stats['shopping_rate'] > 0.0
        assert stats['average_venues_per_paper'] > 1.0
    
    def test_predict_next_venue_choice(self):
        """Test next venue choice prediction."""
        # Create historical pattern for researcher
        paper_id = "paper_001"
        researcher_id = "researcher_001"
        
        # Create pattern: Top -> Mid -> Low
        prestiges = [9, 6, 4]
        venue_types = [VenueType.TOP_CONFERENCE, VenueType.MID_CONFERENCE, VenueType.LOW_CONFERENCE]
        
        submissions = []
        for i, (prestige, vtype) in enumerate(zip(prestiges, venue_types)):
            submission_id = self.tracker.record_submission(
                paper_id=paper_id,
                researcher_id=researcher_id,
                venue_id=f"venue_{i+1:03d}",
                venue_type=vtype,
                venue_prestige=prestige
            )
            submissions.append(submission_id)
        
        # Mark first two as rejected
        for i in range(2):
            self.tracker.update_submission_outcome(
                paper_id=paper_id,
                submission_id=submissions[i],
                outcome=SubmissionOutcome.REJECTED,
                review_scores=[4.0, 5.0, 3.0]
            )
        
        # Predict next venue for similar scenario
        predictions = self.tracker.predict_next_venue_choice(
            researcher_id=researcher_id,
            current_venue_prestige=8,
            rejection_score=4.0
        )
        
        assert len(predictions) > 0
        assert all(isinstance(pred, tuple) and len(pred) == 2 for pred in predictions)
        assert all(isinstance(pred[0], VenueType) and isinstance(pred[1], float) for pred in predictions)
        
        # Probabilities should sum to approximately 1.0
        total_prob = sum(pred[1] for pred in predictions)
        assert abs(total_prob - 1.0) < 0.1
    
    def test_predict_next_venue_no_history(self):
        """Test next venue prediction with no history."""
        predictions = self.tracker.predict_next_venue_choice(
            researcher_id="new_researcher",
            current_venue_prestige=8,
            rejection_score=4.0
        )
        
        # Should return default predictions
        assert len(predictions) > 0
        assert all(isinstance(pred, tuple) and len(pred) == 2 for pred in predictions)
    
    def test_save_and_load_data(self):
        """Test saving and loading data."""
        # Create some data
        submission_id = self.tracker.record_submission(
            paper_id="paper_001",
            researcher_id="researcher_001",
            venue_id="venue_001",
            venue_type=VenueType.TOP_CONFERENCE,
            venue_prestige=9
        )
        
        self.tracker.update_submission_outcome(
            paper_id="paper_001",
            submission_id=submission_id,
            outcome=SubmissionOutcome.REJECTED,
            review_scores=[5.0, 6.0, 4.0]
        )
        
        # Save data
        filename = "test_venue_shopping.json"
        self.tracker.save_data(filename)
        
        # Create new tracker and load data
        new_tracker = VenueShoppingTracker(data_dir=self.temp_dir)
        new_tracker.load_data(filename)
        
        # Verify data loaded correctly
        assert "paper_001" in new_tracker.submission_records
        assert len(new_tracker.submission_records["paper_001"]) == 1
        
        submission = new_tracker.submission_records["paper_001"][0]
        assert submission.paper_id == "paper_001"
        assert submission.outcome == SubmissionOutcome.REJECTED
        assert submission.review_scores == [5.0, 6.0, 4.0]
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        # Should not raise error, just log warning
        self.tracker.load_data("nonexistent.json")
        
        # Data should remain empty
        assert len(self.tracker.submission_records) == 0
        assert len(self.tracker.shopping_patterns) == 0
        assert len(self.tracker.researcher_profiles) == 0


if __name__ == "__main__":
    pytest.main([__file__])