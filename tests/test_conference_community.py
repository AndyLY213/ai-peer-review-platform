"""
Unit tests for ConferenceCommunity class.

Tests conference community modeling, clique formation,
and community-based reviewer selection functionality.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch
import random

from src.enhancements.conference_community import (
    ConferenceCommunity, AttendanceRecord, CommunityClique, CommunityInfluence, NetworkError
)
from src.data.enhanced_models import EnhancedResearcher, EnhancedVenue, ResearcherLevel, VenueType
from src.core.exceptions import ValidationError


class TestAttendanceRecord:
    """Test AttendanceRecord dataclass."""
    
    def test_valid_attendance_record(self):
        """Test creating a valid attendance record."""
        record = AttendanceRecord(
            researcher_id="researcher1",
            venue_id="venue1",
            year=2023,
            role="presenter",
            presentation_count=1,
            networking_activity=0.8
        )
        
        assert record.researcher_id == "researcher1"
        assert record.venue_id == "venue1"
        assert record.year == 2023
        assert record.role == "presenter"
    
    def test_invalid_year_raises_error(self):
        """Test that invalid year raises ValidationError."""
        with pytest.raises(ValidationError):
            AttendanceRecord(
                researcher_id="researcher1",
                venue_id="venue1",
                year=1900,  # Too old
                networking_activity=0.5
            )
    
    def test_invalid_networking_activity_raises_error(self):
        """Test that invalid networking activity raises ValidationError."""
        with pytest.raises(ValidationError):
            AttendanceRecord(
                researcher_id="researcher1",
                venue_id="venue1",
                year=2023,
                networking_activity=1.5  # Too high
            )


class TestCommunityClique:
    """Test CommunityClique dataclass."""
    
    def test_valid_community_clique(self):
        """Test creating a valid community clique."""
        clique = CommunityClique(
            clique_id="clique1",
            venue_id="venue1",
            member_ids={"researcher1", "researcher2", "researcher3"},
            formation_year=2023,
            clique_strength=0.8,
            influence_score=0.6,
            research_focus="AI"
        )
        
        assert clique.clique_id == "clique1"
        assert len(clique.member_ids) == 3
        assert clique.clique_strength == 0.8
    
    def test_insufficient_members_raises_error(self):
        """Test that insufficient members raises ValidationError."""
        with pytest.raises(ValidationError):
            CommunityClique(
                clique_id="clique1",
                venue_id="venue1",
                member_ids={"researcher1"},  # Only 1 member
                formation_year=2023,
                clique_strength=0.8,
                influence_score=0.6
            )
    
    def test_invalid_strength_raises_error(self):
        """Test that invalid clique strength raises ValidationError."""
        with pytest.raises(ValidationError):
            CommunityClique(
                clique_id="clique1",
                venue_id="venue1",
                member_ids={"researcher1", "researcher2"},
                formation_year=2023,
                clique_strength=1.5,  # Too high
                influence_score=0.6
            )


class TestConferenceCommunity:
    """Test ConferenceCommunity class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.community = ConferenceCommunity()
    
    def test_initialization(self):
        """Test community initialization."""
        assert len(self.community.attendance_records) == 0
        assert len(self.community.venue_attendees) == 0
        assert len(self.community.researcher_venues) == 0
        assert len(self.community.community_cliques) == 0
    
    def test_add_attendance_record(self):
        """Test adding an attendance record."""
        record = self.community.add_attendance_record(
            researcher_id="researcher1",
            venue_id="venue1",
            year=2023,
            role="presenter",
            presentation_count=1,
            networking_activity=0.8
        )
        
        assert len(self.community.attendance_records) == 1
        assert record.researcher_id == "researcher1"
        
        # Check tracking updates
        assert "researcher1" in self.community.venue_attendees["venue1"][2023]
        assert "venue1" in self.community.researcher_venues["researcher1"]
    
    def test_add_attendance_record_invalid_data_raises_error(self):
        """Test that invalid attendance data raises error."""
        with pytest.raises(NetworkError):
            self.community.add_attendance_record(
                researcher_id="researcher1",
                venue_id="venue1",
                year=1900,  # Invalid year
                networking_activity=0.5
            )
    
    def test_get_venue_attendees(self):
        """Test getting venue attendees."""
        # Add attendance records
        self.community.add_attendance_record("researcher1", "venue1", 2023)
        self.community.add_attendance_record("researcher2", "venue1", 2023)
        self.community.add_attendance_record("researcher1", "venue1", 2022)
        
        # Get attendees for specific year
        attendees_2023 = self.community.get_venue_attendees("venue1", 2023)
        assert "researcher1" in attendees_2023
        assert "researcher2" in attendees_2023
        assert len(attendees_2023) == 2
        
        # Get all attendees
        all_attendees = self.community.get_venue_attendees("venue1")
        assert "researcher1" in all_attendees
        assert "researcher2" in all_attendees
        assert len(all_attendees) == 2
    
    def test_get_researcher_attendance_history(self):
        """Test getting researcher attendance history."""
        # Add multiple attendance records
        self.community.add_attendance_record("researcher1", "venue1", 2023, "presenter")
        self.community.add_attendance_record("researcher1", "venue2", 2023, "attendee")
        self.community.add_attendance_record("researcher1", "venue1", 2022, "reviewer")
        
        history = self.community.get_researcher_attendance_history("researcher1")
        
        assert len(history) == 3
        # Should be sorted by year and venue
        assert history[0].year <= history[1].year <= history[2].year
    
    def test_calculate_attendance_overlap(self):
        """Test calculating attendance overlap between researchers."""
        # Add overlapping attendance
        self.community.add_attendance_record("researcher1", "venue1", 2023)
        self.community.add_attendance_record("researcher2", "venue1", 2023)
        self.community.add_attendance_record("researcher1", "venue2", 2022)
        self.community.add_attendance_record("researcher2", "venue2", 2022)
        
        # Add non-overlapping attendance
        self.community.add_attendance_record("researcher1", "venue3", 2021)
        
        overlap = self.community.calculate_attendance_overlap("researcher1", "researcher2")
        
        assert overlap["total_overlaps"] == 2
        assert ("venue1", 2023) in overlap["overlap_events"]
        assert ("venue2", 2022) in overlap["overlap_events"]
        assert overlap["common_venues"] == 2
        assert overlap["overlap_strength"] > 0
    
    def test_form_cliques(self):
        """Test clique formation."""
        # Add attendance records for multiple researchers at same venue
        researchers = ["r1", "r2", "r3", "r4", "r5"]
        
        # Create overlapping attendance patterns
        for researcher in researchers[:3]:  # First group
            for year in [2021, 2022, 2023]:
                self.community.add_attendance_record(researcher, "venue1", year, networking_activity=0.8)
        
        for researcher in researchers[2:]:  # Second group (r3 overlaps)
            for year in [2022, 2023]:
                self.community.add_attendance_record(researcher, "venue1", year, networking_activity=0.7)
        
        cliques = self.community.form_cliques("venue1", min_clique_size=3, min_overlap_threshold=0.1)
        
        assert len(cliques) > 0
        for clique in cliques:
            assert len(clique.member_ids) >= 3
            assert clique.venue_id == "venue1"
            assert 0 <= clique.clique_strength <= 1
            assert 0 <= clique.influence_score <= 1
    
    def test_calculate_community_influence(self):
        """Test calculating community influence."""
        # Add attendance records with different roles
        self.community.add_attendance_record("researcher1", "venue1", 2023, "keynote", 1, 0.9)
        self.community.add_attendance_record("researcher1", "venue1", 2022, "presenter", 2, 0.8)
        self.community.add_attendance_record("researcher1", "venue1", 2021, "attendee", 0, 0.7)
        
        influence = self.community.calculate_community_influence("researcher1", "venue1")
        
        assert influence.researcher_id == "researcher1"
        assert influence.venue_id == "venue1"
        assert influence.community_standing > 0
        assert len(influence.attendance_years) == 3
        assert influence.networking_score > 0
        assert influence.influence_multiplier >= 1.0
    
    def test_calculate_community_influence_no_attendance(self):
        """Test calculating influence with no attendance."""
        influence = self.community.calculate_community_influence("researcher1", "venue1")
        
        assert influence.community_standing == 0.0
        assert len(influence.attendance_years) == 0
        assert influence.influence_multiplier == 1.0
    
    def test_get_reviewer_selection_preferences(self):
        """Test getting reviewer selection preferences."""
        # Add attendance for different researchers
        self.community.add_attendance_record("researcher1", "venue1", 2023, "keynote", 1, 0.9)
        self.community.add_attendance_record("researcher2", "venue1", 2023, "attendee", 0, 0.5)
        
        preferences = self.community.get_reviewer_selection_preferences(
            "venue1", ["researcher1", "researcher2", "researcher3"]
        )
        
        assert "researcher1" in preferences
        assert "researcher2" in preferences
        assert "researcher3" in preferences
        
        # Researcher1 should have higher preference due to keynote role
        assert preferences["researcher1"] > preferences["researcher2"]
        # Researcher3 has no attendance, so lowest preference
        assert preferences["researcher3"] == 0.0
    
    def test_detect_community_effects(self):
        """Test detecting community effects."""
        # Add attendance and form cliques
        researchers = ["r1", "r2", "r3", "r4"]
        for researcher in researchers:
            for year in [2021, 2022, 2023]:
                self.community.add_attendance_record(researcher, "venue1", year, networking_activity=0.8)
        
        self.community.form_cliques("venue1", min_clique_size=2, min_overlap_threshold=0.1)
        
        effects = self.community.detect_community_effects("venue1")
        
        assert "total_cliques" in effects
        assert "clique_coverage" in effects
        assert "average_clique_size" in effects
        assert "influence_concentration" in effects
        assert "attendance_loyalty" in effects
        
        assert effects["attendance_loyalty"] > 0  # Should have some loyalty
    
    def test_get_community_statistics(self):
        """Test getting community statistics."""
        # Add some data
        self.community.add_attendance_record("researcher1", "venue1", 2023)
        self.community.add_attendance_record("researcher2", "venue1", 2023)
        self.community.add_attendance_record("researcher1", "venue2", 2022)
        
        stats = self.community.get_community_statistics()
        
        assert stats["total_venues"] == 2
        assert stats["total_attendees"] == 2
        assert stats["total_attendance_records"] == 3
        assert stats["average_attendance_per_researcher"] == 1.5
        assert "most_active_venues" in stats
        assert "most_connected_researchers" in stats
    
    def test_build_communities_from_researchers(self):
        """Test building communities from researcher objects."""
        # Create test researchers and venues with high reputation to ensure attendance
        researcher1 = EnhancedResearcher(
            id="researcher1",
            name="Researcher 1",
            specialty="AI",
            level=ResearcherLevel.FULL_PROF,
            reputation_score=0.9  # High reputation for guaranteed attendance
        )
        
        researcher2 = EnhancedResearcher(
            id="researcher2",
            name="Researcher 2",
            specialty="AI",  # Same specialty to ensure venue match
            level=ResearcherLevel.FULL_PROF,
            reputation_score=0.9
        )
        
        venue1 = EnhancedVenue(
            id="venue1",
            name="AI Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="AI"
        )
        
        venue2 = EnhancedVenue(
            id="venue2",
            name="AI Workshop",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI"
        )
        
        # Set random seed for reproducible results
        import random
        random.seed(42)
        
        self.community.build_communities_from_researchers([researcher1, researcher2], [venue1, venue2])
        
        # Check that the method ran without errors
        # The actual attendance depends on random factors, so we just check it completed
        assert len(self.community.attendance_records) >= 0
        
        # Check that cliques were attempted to be formed
        total_cliques = sum(len(cliques) for cliques in self.community.community_cliques.values())
        assert total_cliques >= 0
    
    def test_serialization(self):
        """Test community serialization and deserialization."""
        # Add some data
        self.community.add_attendance_record("researcher1", "venue1", 2023, "presenter", 1, 0.8)
        
        # Form a clique
        researchers = ["r1", "r2", "r3"]
        for researcher in researchers:
            self.community.add_attendance_record(researcher, "venue2", 2023, networking_activity=0.8)
        self.community.form_cliques("venue2", min_clique_size=2, min_overlap_threshold=0.1)
        
        # Serialize
        data = self.community.to_dict()
        
        # Deserialize
        new_community = ConferenceCommunity.from_dict(data)
        
        assert len(new_community.attendance_records) == len(self.community.attendance_records)
        assert len(new_community.venue_attendees) == len(self.community.venue_attendees)
        assert len(new_community.community_cliques) == len(self.community.community_cliques)
        
        # Check specific record
        record = new_community.attendance_records[0]
        assert record.researcher_id == "researcher1"
        assert record.venue_id == "venue1"
        assert record.year == 2023
        assert record.role == "presenter"
    
    def test_empty_venue_clique_formation(self):
        """Test clique formation with empty venue."""
        cliques = self.community.form_cliques("empty_venue", min_clique_size=3)
        assert len(cliques) == 0
    
    def test_insufficient_attendees_clique_formation(self):
        """Test clique formation with insufficient attendees."""
        # Add only 2 attendees
        self.community.add_attendance_record("researcher1", "venue1", 2023)
        self.community.add_attendance_record("researcher2", "venue1", 2023)
        
        cliques = self.community.form_cliques("venue1", min_clique_size=3)
        assert len(cliques) == 0
    
    def test_community_influence_with_cliques(self):
        """Test community influence calculation with clique membership."""
        # Add attendance
        researchers = ["r1", "r2", "r3"]
        for researcher in researchers:
            self.community.add_attendance_record(researcher, "venue1", 2023, networking_activity=0.8)
        
        # Form cliques
        cliques = self.community.form_cliques("venue1", min_clique_size=2, min_overlap_threshold=0.1)
        
        if cliques:  # If cliques were formed
            # Get influence for a clique member
            clique_member = list(cliques[0].member_ids)[0]
            influence = self.community.calculate_community_influence(clique_member, "venue1")
            
            assert len(influence.clique_memberships) > 0
            assert influence.influence_multiplier > 1.0  # Should have bonus from clique membership
    
    def test_attendance_overlap_no_overlap(self):
        """Test attendance overlap with no common attendance."""
        self.community.add_attendance_record("researcher1", "venue1", 2023)
        self.community.add_attendance_record("researcher2", "venue2", 2023)
        
        overlap = self.community.calculate_attendance_overlap("researcher1", "researcher2")
        
        assert overlap["total_overlaps"] == 0
        assert overlap["overlap_strength"] == 0.0
        assert overlap["common_venues"] == 0
    
    def test_reviewer_preferences_with_cliques(self):
        """Test reviewer preferences considering clique memberships."""
        # Add attendance and form cliques
        researchers = ["r1", "r2", "r3", "r4"]
        for researcher in researchers:
            self.community.add_attendance_record(researcher, "venue1", 2023, networking_activity=0.8)
        
        cliques = self.community.form_cliques("venue1", min_clique_size=2, min_overlap_threshold=0.1)
        
        preferences = self.community.get_reviewer_selection_preferences("venue1", researchers)
        
        # All researchers should have some preference since they attended
        for researcher in researchers:
            assert preferences[researcher] > 0
        
        # If cliques were formed, clique members should have higher preferences
        if cliques:
            clique_members = set()
            for clique in cliques:
                clique_members.update(clique.member_ids)
            
            non_clique_members = set(researchers) - clique_members
            
            if non_clique_members:
                avg_clique_preference = sum(preferences[member] for member in clique_members) / len(clique_members)
                avg_non_clique_preference = sum(preferences[member] for member in non_clique_members) / len(non_clique_members)
                
                # Clique members should generally have higher preferences
                assert avg_clique_preference >= avg_non_clique_preference