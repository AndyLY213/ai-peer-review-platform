"""
Unit tests for CollaborationNetwork class.

Tests collaboration network tracking, conflict of interest detection,
and network analysis functionality.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch

from src.enhancements.collaboration_network import (
    CollaborationNetwork, CollaborationRecord, ConflictOfInterest, NetworkError
)
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord, ResearcherLevel
from src.core.exceptions import ValidationError


class TestCollaborationRecord:
    """Test CollaborationRecord dataclass."""
    
    def test_valid_collaboration_record(self):
        """Test creating a valid collaboration record."""
        record = CollaborationRecord(
            researcher_1_id="researcher1",
            researcher_2_id="researcher2",
            paper_id="paper1",
            paper_title="Test Paper",
            collaboration_date=date(2023, 1, 1),
            venue="Test Conference"
        )
        
        assert record.researcher_1_id == "researcher1"
        assert record.researcher_2_id == "researcher2"
        assert record.collaboration_type == "co-author"
    
    def test_same_researcher_ids_raises_error(self):
        """Test that same researcher IDs raise ValidationError."""
        with pytest.raises(ValidationError):
            CollaborationRecord(
                researcher_1_id="researcher1",
                researcher_2_id="researcher1",
                paper_id="paper1",
                paper_title="Test Paper",
                collaboration_date=date(2023, 1, 1),
                venue="Test Conference"
            )


class TestCollaborationNetwork:
    """Test CollaborationNetwork class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = CollaborationNetwork(collaboration_window_years=3)
        self.test_date = date(2023, 6, 1)
    
    def test_initialization(self):
        """Test network initialization."""
        assert self.network.collaboration_window_years == 3
        assert len(self.network.collaboration_records) == 0
        assert len(self.network.researcher_collaborations) == 0
    
    def test_add_collaboration(self):
        """Test adding a collaboration."""
        record = self.network.add_collaboration(
            researcher_1_id="researcher1",
            researcher_2_id="researcher2",
            paper_id="paper1",
            paper_title="Test Paper",
            collaboration_date=date(2023, 1, 1),
            venue="Test Conference"
        )
        
        assert len(self.network.collaboration_records) == 1
        assert record.researcher_1_id == "researcher1"
        assert record.researcher_2_id == "researcher2"
        
        # Check bidirectional tracking
        assert "researcher2" in self.network.researcher_collaborations["researcher1"]
        assert "researcher1" in self.network.researcher_collaborations["researcher2"]
    
    def test_add_collaboration_same_researchers_raises_error(self):
        """Test that adding collaboration with same researchers raises error."""
        with pytest.raises(NetworkError):
            self.network.add_collaboration(
                researcher_1_id="researcher1",
                researcher_2_id="researcher1",
                paper_id="paper1",
                paper_title="Test Paper",
                collaboration_date=date(2023, 1, 1),
                venue="Test Conference"
            )
    
    def test_add_advisor_relationship(self):
        """Test adding advisor-student relationship."""
        self.network.add_advisor_relationship("advisor1", "student1")
        
        assert "student1" in self.network.advisor_relationships["advisor1"]
    
    def test_add_institutional_affiliation(self):
        """Test adding institutional affiliation."""
        self.network.add_institutional_affiliation("researcher1", "University A")
        
        assert "researcher1" in self.network.institutional_affiliations["University A"]
    
    def test_get_collaborators_within_window(self):
        """Test getting collaborators within time window."""
        # Add recent collaboration (within window)
        self.network.add_collaboration(
            "researcher1", "researcher2", "paper1", "Paper 1",
            date(2022, 1, 1), "Conference A"
        )
        
        # Add old collaboration (outside window)
        self.network.add_collaboration(
            "researcher1", "researcher3", "paper2", "Paper 2",
            date(2019, 1, 1), "Conference B"
        )
        
        recent_collaborators = self.network.get_collaborators_within_window(
            "researcher1", reference_date=self.test_date
        )
        
        assert "researcher2" in recent_collaborators
        assert "researcher3" not in recent_collaborators
    
    def test_detect_self_review_conflict(self):
        """Test detection of self-review conflict."""
        conflicts = self.network.detect_conflicts_of_interest(
            paper_authors=["author1", "author2"],
            potential_reviewer="author1"
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "self-review"
        assert conflicts[0].conflict_strength == 1.0
    
    def test_detect_advisor_conflict(self):
        """Test detection of advisor-student conflict."""
        self.network.add_advisor_relationship("advisor1", "student1")
        
        conflicts = self.network.detect_conflicts_of_interest(
            paper_authors=["student1"],
            potential_reviewer="advisor1"
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "advisor"
        assert conflicts[0].conflict_strength == 1.0
    
    def test_detect_recent_collaboration_conflict(self):
        """Test detection of recent collaboration conflict."""
        # Add recent collaboration
        self.network.add_collaboration(
            "reviewer1", "author1", "paper1", "Paper 1",
            date(2022, 6, 1), "Conference A"
        )
        
        conflicts = self.network.detect_conflicts_of_interest(
            paper_authors=["author1"],
            potential_reviewer="reviewer1",
            reference_date=self.test_date
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "recent-collaborator"
        assert conflicts[0].conflict_strength > 0.1
    
    def test_detect_institutional_conflict(self):
        """Test detection of institutional conflict."""
        self.network.add_institutional_affiliation("reviewer1", "University A")
        self.network.add_institutional_affiliation("author1", "University A")
        
        conflicts = self.network.detect_conflicts_of_interest(
            paper_authors=["author1"],
            potential_reviewer="reviewer1"
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "institutional"
        assert conflicts[0].conflict_strength == 0.3
    
    def test_has_conflict_of_interest(self):
        """Test conflict of interest checking with threshold."""
        # Add strong conflict (advisor relationship)
        self.network.add_advisor_relationship("advisor1", "student1")
        
        # Strong conflict should be detected
        assert self.network.has_conflict_of_interest(
            paper_authors=["student1"],
            potential_reviewer="advisor1",
            min_conflict_strength=0.5
        )
        
        # Weak conflict should not be detected with high threshold
        self.network.add_institutional_affiliation("reviewer2", "University A")
        self.network.add_institutional_affiliation("author2", "University A")
        
        assert not self.network.has_conflict_of_interest(
            paper_authors=["author2"],
            potential_reviewer="reviewer2",
            min_conflict_strength=0.5
        )
    
    def test_get_collaboration_history(self):
        """Test getting collaboration history for a researcher."""
        # Add multiple collaborations
        self.network.add_collaboration(
            "researcher1", "researcher2", "paper1", "Paper 1",
            date(2023, 1, 1), "Conference A"
        )
        self.network.add_collaboration(
            "researcher1", "researcher3", "paper2", "Paper 2",
            date(2022, 1, 1), "Conference B"
        )
        
        history = self.network.get_collaboration_history("researcher1")
        
        assert len(history) == 2
        # Should be sorted by date (most recent first)
        assert history[0].collaboration_date > history[1].collaboration_date
    
    def test_get_collaboration_strength(self):
        """Test calculating collaboration strength between researchers."""
        # No collaborations
        strength = self.network.get_collaboration_strength("researcher1", "researcher2")
        assert strength == 0.0
        
        # Add collaboration
        self.network.add_collaboration(
            "researcher1", "researcher2", "paper1", "Paper 1",
            date(2023, 1, 1), "Conference A"
        )
        
        strength = self.network.get_collaboration_strength("researcher1", "researcher2")
        assert strength > 0.0
    
    def test_get_network_statistics(self):
        """Test getting network statistics."""
        # Add some data
        self.network.add_collaboration(
            "researcher1", "researcher2", "paper1", "Paper 1",
            date(2023, 1, 1), "Conference A"
        )
        self.network.add_advisor_relationship("advisor1", "student1")
        self.network.add_institutional_affiliation("researcher1", "University A")
        
        stats = self.network.get_network_statistics()
        
        assert stats["total_researchers"] == 2
        assert stats["total_collaborations"] == 1
        assert stats["total_advisor_relationships"] == 1
        assert stats["total_institutions"] == 1
        assert "average_collaborations_per_researcher" in stats
        assert "most_collaborative_researchers" in stats
        assert "collaboration_type_distribution" in stats
    
    def test_build_network_from_researchers(self):
        """Test building network from researcher objects."""
        # Create test researchers with publication history
        researcher1 = EnhancedResearcher(
            id="researcher1",
            name="Researcher 1",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF,
            institutional_affiliations=["University A"],
            advisor_relationships=["advisor1"],
            collaboration_network={"researcher2"},
            publication_history=[
                PublicationRecord(
                    paper_id="paper1",
                    title="Paper 1",
                    venue="Conference A",
                    year=2023
                )
            ]
        )
        
        researcher2 = EnhancedResearcher(
            id="researcher2",
            name="Researcher 2",
            specialty="ML",
            level=ResearcherLevel.POSTDOC,
            collaboration_network={"researcher1"},
            publication_history=[
                PublicationRecord(
                    paper_id="paper1",
                    title="Paper 1",
                    venue="Conference A",
                    year=2023
                )
            ]
        )
        
        self.network.build_network_from_researchers([researcher1, researcher2])
        
        # Check that network was built
        assert len(self.network.collaboration_records) == 1
        assert "University A" in self.network.institutional_affiliations
        assert "researcher1" in self.network.advisor_relationships["advisor1"]
    
    def test_serialization(self):
        """Test network serialization and deserialization."""
        # Add some data
        self.network.add_collaboration(
            "researcher1", "researcher2", "paper1", "Paper 1",
            date(2023, 1, 1), "Conference A"
        )
        self.network.add_advisor_relationship("advisor1", "student1")
        self.network.add_institutional_affiliation("researcher1", "University A")
        
        # Serialize
        data = self.network.to_dict()
        
        # Deserialize
        new_network = CollaborationNetwork.from_dict(data)
        
        assert new_network.collaboration_window_years == self.network.collaboration_window_years
        assert len(new_network.collaboration_records) == len(self.network.collaboration_records)
        assert len(new_network.advisor_relationships) == len(self.network.advisor_relationships)
        assert len(new_network.institutional_affiliations) == len(self.network.institutional_affiliations)
    
    def test_multiple_conflicts_same_reviewer(self):
        """Test detecting multiple conflicts for the same reviewer."""
        # Set up multiple conflict types
        self.network.add_advisor_relationship("reviewer1", "author1")
        self.network.add_institutional_affiliation("reviewer1", "University A")
        self.network.add_institutional_affiliation("author2", "University A")
        
        conflicts = self.network.detect_conflicts_of_interest(
            paper_authors=["author1", "author2"],
            potential_reviewer="reviewer1"
        )
        
        assert len(conflicts) == 2
        conflict_types = {c.conflict_type for c in conflicts}
        assert "advisor" in conflict_types
        assert "institutional" in conflict_types
    
    def test_collaboration_strength_with_multiple_papers(self):
        """Test collaboration strength calculation with multiple papers."""
        # Add multiple collaborations
        for i in range(3):
            self.network.add_collaboration(
                "researcher1", "researcher2", f"paper{i}", f"Paper {i}",
                date(2023, i+1, 1), "Conference A"
            )
        
        strength = self.network.get_collaboration_strength("researcher1", "researcher2")
        
        # Should be higher than single collaboration
        assert strength > 0.3
    
    def test_conflict_strength_decreases_with_time(self):
        """Test that conflict strength decreases with time."""
        # Add old collaboration
        self.network.add_collaboration(
            "reviewer1", "author1", "paper1", "Paper 1",
            date(2020, 1, 1), "Conference A"
        )
        
        conflicts = self.network.detect_conflicts_of_interest(
            paper_authors=["author1"],
            potential_reviewer="reviewer1",
            reference_date=self.test_date
        )
        
        if conflicts:  # Should be weak or no conflict due to age
            assert conflicts[0].conflict_strength < 0.5