"""
Unit tests for the venue system module.

Tests venue registry, venue creation from profiles, venue management,
and integration with PeerRead calibration data.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.enhancements.venue_system import (
    VenueRegistry, VenueProfile, get_venue_registry, initialize_standard_venues
)
from src.data.enhanced_models import (
    EnhancedVenue, VenueType, EnhancedResearcher, ResearcherLevel,
    ReviewRequirements, QualityStandards, ReviewerCriteria
)
from src.core.exceptions import ValidationError, DatabaseError


class TestVenueProfile:
    """Test VenueProfile dataclass."""
    
    def test_venue_profile_creation(self):
        """Test creating a venue profile."""
        profile = VenueProfile(
            name="Test Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Machine Learning",
            acceptance_rate=0.25,
            prestige_score=8,
            min_h_index=15,
            score_threshold=3.5,
            review_deadline_weeks=6
        )
        
        assert profile.name == "Test Conference"
        assert profile.venue_type == VenueType.TOP_CONFERENCE
        assert profile.field == "Machine Learning"
        assert profile.acceptance_rate == 0.25
        assert profile.prestige_score == 8
        assert profile.min_h_index == 15
        assert profile.score_threshold == 3.5
        assert profile.review_deadline_weeks == 6
    
    def test_venue_profile_with_peerread_data(self):
        """Test venue profile with PeerRead calibration data."""
        profile = VenueProfile(
            name="ACL",
            venue_type=VenueType.TOP_CONFERENCE,
            field="NLP",
            acceptance_rate=0.25,
            prestige_score=9,
            min_h_index=15,
            score_threshold=3.5,
            review_deadline_weeks=6,
            peerread_venue_code="acl",
            historical_acceptance_rates=[0.23, 0.25, 0.27],
            score_distributions={
                "IMPACT": [2.8, 3.2, 3.5],
                "SUBSTANCE": [3.1, 3.4, 3.6]
            },
            review_patterns={
                "avg_review_length": 650,
                "min_word_count": 400
            }
        )
        
        assert profile.peerread_venue_code == "acl"
        assert len(profile.historical_acceptance_rates) == 3
        assert "IMPACT" in profile.score_distributions
        assert profile.review_patterns["avg_review_length"] == 650


class TestVenueRegistry:
    """Test VenueRegistry class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def registry(self, temp_dir):
        """Create venue registry for testing."""
        return VenueRegistry(temp_dir)
    
    @pytest.fixture
    def sample_venue(self):
        """Create sample venue for testing."""
        return EnhancedVenue(
            id="test-venue-1",
            name="Test Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="Computer Science",
            acceptance_rate=0.30,
            prestige_score=6
        )
    
    @pytest.fixture
    def sample_researcher(self):
        """Create sample researcher for testing."""
        researcher = EnhancedResearcher(
            id="researcher-1",
            name="Dr. Test",
            specialty="Machine Learning",
            level=ResearcherLevel.ASSISTANT_PROF,
            h_index=12,
            years_active=5,
            institution_tier=2,
            total_citations=200  # Add citations to boost reputation
        )
        # Manually set reputation score to ensure it meets requirements
        researcher.reputation_score = 0.4
        return researcher
    
    def test_registry_initialization(self, temp_dir):
        """Test registry initialization."""
        registry = VenueRegistry(temp_dir)
        
        assert registry.data_dir == temp_dir
        assert temp_dir.exists()
        
        # Check that real venue profiles are initialized
        stats = registry.get_registry_stats()
        assert "ACL" in stats["available_profiles"]
        assert "NeurIPS" in stats["available_profiles"]
        assert "ICLR" in stats["available_profiles"]
        assert "CoNLL" in stats["available_profiles"]
    
    def test_create_venue_from_profile_acl(self, registry):
        """Test creating ACL venue from profile."""
        venue = registry.create_venue_from_profile("ACL")
        
        assert venue.name == "ACL"
        assert venue.venue_type == VenueType.TOP_CONFERENCE
        assert venue.field == "Natural Language Processing"
        assert venue.acceptance_rate == 0.25
        assert venue.prestige_score == 9
        assert venue.reviewer_selection_criteria.min_h_index == 15
        assert venue.quality_standards.acceptance_threshold == 3.5
        assert venue.review_deadline_weeks == 6
        
        # Check review requirements
        assert venue.review_requirements.min_word_count == 400
        assert venue.review_requirements.max_word_count == 800
        assert "summary" in venue.review_requirements.required_sections
        assert "strengths" in venue.review_requirements.required_sections
        
        # Check PeerRead calibration data
        assert "IMPACT" in venue.score_distributions
        assert "SUBSTANCE" in venue.score_distributions
        assert venue.review_length_stats["avg_length"] == 650
    
    def test_create_venue_from_profile_neurips(self, registry):
        """Test creating NeurIPS venue from profile."""
        venue = registry.create_venue_from_profile("NeurIPS")
        
        assert venue.name == "NeurIPS"
        assert venue.venue_type == VenueType.TOP_CONFERENCE
        assert venue.field == "Machine Learning"
        assert venue.acceptance_rate == 0.20
        assert venue.prestige_score == 10
        assert venue.reviewer_selection_criteria.min_h_index == 20
        assert venue.quality_standards.acceptance_threshold == 4.0
        assert venue.review_deadline_weeks == 6
        
        # Check that it has higher standards than ACL
        assert venue.reviewer_selection_criteria.min_h_index > 15
        assert venue.quality_standards.acceptance_threshold > 3.5
    
    def test_create_venue_from_profile_iclr(self, registry):
        """Test creating ICLR venue from profile."""
        venue = registry.create_venue_from_profile("ICLR")
        
        assert venue.name == "ICLR"
        assert venue.venue_type == VenueType.TOP_CONFERENCE
        assert venue.field == "Machine Learning"
        assert venue.acceptance_rate == 0.30
        assert venue.prestige_score == 9
        assert venue.reviewer_selection_criteria.min_h_index == 12
        assert venue.quality_standards.acceptance_threshold == 3.5
        assert venue.review_deadline_weeks == 8  # ICLR has longer review cycles
    
    def test_create_venue_from_profile_conll(self, registry):
        """Test creating CoNLL venue from profile."""
        venue = registry.create_venue_from_profile("CoNLL")
        
        assert venue.name == "CoNLL"
        assert venue.venue_type == VenueType.MID_CONFERENCE
        assert venue.field == "Natural Language Processing"
        assert venue.acceptance_rate == 0.35
        assert venue.prestige_score == 7
        assert venue.reviewer_selection_criteria.min_h_index == 8
        assert venue.quality_standards.acceptance_threshold == 3.0
        assert venue.review_deadline_weeks == 4
        
        # Check that it has lower standards than top conferences
        assert venue.reviewer_selection_criteria.min_h_index < 15
        assert venue.quality_standards.acceptance_threshold < 3.5
    
    def test_create_venue_from_invalid_profile(self, registry):
        """Test creating venue from non-existent profile."""
        with pytest.raises(ValidationError) as exc_info:
            registry.create_venue_from_profile("NonExistentVenue")
        
        assert "profile_name" in str(exc_info.value)
    
    def test_register_venue_success(self, registry, sample_venue):
        """Test successful venue registration."""
        result = registry.register_venue(sample_venue)
        
        assert result is True
        assert sample_venue.id in registry._venues
        assert sample_venue.name in registry._venue_by_name
        
        # Check that venue file was created
        venue_file = registry.data_dir / f"{sample_venue.id}.json"
        assert venue_file.exists()
    
    def test_register_venue_duplicate_id(self, registry, sample_venue):
        """Test registering venue with duplicate ID."""
        registry.register_venue(sample_venue)
        
        duplicate_venue = EnhancedVenue(
            id=sample_venue.id,  # Same ID
            name="Different Name",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Different Field"
        )
        
        with pytest.raises(DatabaseError) as exc_info:
            registry.register_venue(duplicate_venue)
        
        assert "already exists" in str(exc_info.value)
    
    def test_register_venue_duplicate_name(self, registry, sample_venue):
        """Test registering venue with duplicate name."""
        registry.register_venue(sample_venue)
        
        duplicate_venue = EnhancedVenue(
            id="different-id",
            name=sample_venue.name,  # Same name
            venue_type=VenueType.TOP_CONFERENCE,
            field="Different Field"
        )
        
        with pytest.raises(DatabaseError) as exc_info:
            registry.register_venue(duplicate_venue)
        
        assert "already exists" in str(exc_info.value)
    
    def test_get_venue_by_id(self, registry, sample_venue):
        """Test getting venue by ID."""
        registry.register_venue(sample_venue)
        
        retrieved_venue = registry.get_venue(sample_venue.id)
        assert retrieved_venue is not None
        assert retrieved_venue.id == sample_venue.id
        assert retrieved_venue.name == sample_venue.name
    
    def test_get_venue_by_id_not_found(self, registry):
        """Test getting non-existent venue by ID."""
        result = registry.get_venue("non-existent-id")
        assert result is None
    
    def test_get_venue_by_name(self, registry, sample_venue):
        """Test getting venue by name."""
        registry.register_venue(sample_venue)
        
        retrieved_venue = registry.get_venue_by_name(sample_venue.name)
        assert retrieved_venue is not None
        assert retrieved_venue.id == sample_venue.id
        assert retrieved_venue.name == sample_venue.name
    
    def test_get_venue_by_name_not_found(self, registry):
        """Test getting non-existent venue by name."""
        result = registry.get_venue_by_name("Non-existent Conference")
        assert result is None
    
    def test_list_venues_all(self, registry):
        """Test listing all venues."""
        # Create and register venues from profiles
        acl_venue = registry.create_venue_from_profile("ACL")
        neurips_venue = registry.create_venue_from_profile("NeurIPS")
        
        registry.register_venue(acl_venue)
        registry.register_venue(neurips_venue)
        
        venues = registry.list_venues()
        assert len(venues) == 2
        venue_names = [v.name for v in venues]
        assert "ACL" in venue_names
        assert "NeurIPS" in venue_names
    
    def test_list_venues_by_type(self, registry):
        """Test listing venues filtered by type."""
        acl_venue = registry.create_venue_from_profile("ACL")
        conll_venue = registry.create_venue_from_profile("CoNLL")
        
        registry.register_venue(acl_venue)
        registry.register_venue(conll_venue)
        
        top_conferences = registry.list_venues(venue_type=VenueType.TOP_CONFERENCE)
        assert len(top_conferences) == 1
        assert top_conferences[0].name == "ACL"
        
        mid_conferences = registry.list_venues(venue_type=VenueType.MID_CONFERENCE)
        assert len(mid_conferences) == 1
        assert mid_conferences[0].name == "CoNLL"
    
    def test_list_venues_by_field(self, registry):
        """Test listing venues filtered by field."""
        acl_venue = registry.create_venue_from_profile("ACL")
        neurips_venue = registry.create_venue_from_profile("NeurIPS")
        
        registry.register_venue(acl_venue)
        registry.register_venue(neurips_venue)
        
        nlp_venues = registry.list_venues(field="Natural Language Processing")
        assert len(nlp_venues) == 1
        assert nlp_venues[0].name == "ACL"
        
        ml_venues = registry.list_venues(field="Machine Learning")
        assert len(ml_venues) == 1
        assert ml_venues[0].name == "NeurIPS"
    
    def test_get_venues_for_researcher_qualified(self, registry, sample_researcher):
        """Test getting venues for qualified researcher."""
        # Create venues with different requirements
        acl_venue = registry.create_venue_from_profile("ACL")  # min_h_index=15
        conll_venue = registry.create_venue_from_profile("CoNLL")  # min_h_index=8
        
        registry.register_venue(acl_venue)
        registry.register_venue(conll_venue)
        
        # Researcher has h_index=12, so should qualify for CoNLL but not ACL
        qualified_venues = registry.get_venues_for_researcher(sample_researcher)
        
        assert len(qualified_venues) == 1
        assert qualified_venues[0].name == "CoNLL"
    
    def test_get_venues_for_researcher_highly_qualified(self, registry):
        """Test getting venues for highly qualified researcher."""
        # Create highly qualified researcher
        senior_researcher = EnhancedResearcher(
            id="senior-researcher",
            name="Dr. Senior",
            specialty="Machine Learning",
            level=ResearcherLevel.FULL_PROF,
            h_index=25,
            years_active=15,
            institution_tier=1
        )
        
        # Create venues
        acl_venue = registry.create_venue_from_profile("ACL")
        neurips_venue = registry.create_venue_from_profile("NeurIPS")
        conll_venue = registry.create_venue_from_profile("CoNLL")
        
        registry.register_venue(acl_venue)
        registry.register_venue(neurips_venue)
        registry.register_venue(conll_venue)
        
        # Should qualify for all venues
        qualified_venues = registry.get_venues_for_researcher(senior_researcher)
        
        assert len(qualified_venues) == 3
        venue_names = [v.name for v in qualified_venues]
        assert "ACL" in venue_names
        assert "NeurIPS" in venue_names
        assert "CoNLL" in venue_names
    
    def test_create_standard_venues(self, registry):
        """Test creating all standard venues."""
        created_venues = registry.create_standard_venues()
        
        assert len(created_venues) == 4  # ACL, NeurIPS, ICLR, CoNLL
        
        venue_names = [v.name for v in created_venues]
        assert "ACL" in venue_names
        assert "NeurIPS" in venue_names
        assert "ICLR" in venue_names
        assert "CoNLL" in venue_names
        
        # Check that all venues are registered
        stats = registry.get_registry_stats()
        assert stats["total_venues"] == 4
    
    def test_venue_persistence(self, temp_dir):
        """Test venue persistence across registry instances."""
        # Create first registry and add venue
        registry1 = VenueRegistry(temp_dir)
        venue = registry1.create_venue_from_profile("ACL")
        registry1.register_venue(venue)
        
        # Create second registry and check venue is loaded
        registry2 = VenueRegistry(temp_dir)
        loaded_venue = registry2.get_venue_by_name("ACL")
        
        assert loaded_venue is not None
        assert loaded_venue.name == "ACL"
        assert loaded_venue.venue_type == VenueType.TOP_CONFERENCE
    
    def test_get_registry_stats(self, registry):
        """Test getting registry statistics."""
        # Create and register venues
        acl_venue = registry.create_venue_from_profile("ACL")
        neurips_venue = registry.create_venue_from_profile("NeurIPS")
        conll_venue = registry.create_venue_from_profile("CoNLL")
        
        registry.register_venue(acl_venue)
        registry.register_venue(neurips_venue)
        registry.register_venue(conll_venue)
        
        stats = registry.get_registry_stats()
        
        assert stats["total_venues"] == 3
        assert stats["venue_types"]["Top Conference"] == 2
        assert stats["venue_types"]["Mid Conference"] == 1
        assert stats["research_fields"]["Natural Language Processing"] == 2
        assert stats["research_fields"]["Machine Learning"] == 1
        assert len(stats["available_profiles"]) == 4
    
    def test_venue_validation_empty_id(self, registry):
        """Test venue validation with empty ID."""
        invalid_venue = EnhancedVenue(
            id="",  # Empty ID
            name="Test Venue",
            venue_type=VenueType.MID_CONFERENCE,
            field="Test Field"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            registry.register_venue(invalid_venue)
        
        assert "venue.id" in str(exc_info.value)
    
    def test_venue_validation_empty_name(self, registry):
        """Test venue validation with empty name."""
        invalid_venue = EnhancedVenue(
            id="test-id",
            name="",  # Empty name
            venue_type=VenueType.MID_CONFERENCE,
            field="Test Field"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            registry.register_venue(invalid_venue)
        
        assert "venue.name" in str(exc_info.value)
    
    def test_venue_validation_invalid_acceptance_rate(self, registry):
        """Test venue validation with invalid acceptance rate."""
        with pytest.raises(ValidationError) as exc_info:
            invalid_venue = EnhancedVenue(
                id="test-id",
                name="Test Venue",
                venue_type=VenueType.MID_CONFERENCE,
                field="Test Field",
                acceptance_rate=1.5  # Invalid rate > 1.0
            )
        
        assert "acceptance_rate" in str(exc_info.value)
    
    def test_venue_validation_invalid_prestige_score(self, registry):
        """Test venue validation with invalid prestige score."""
        invalid_venue = EnhancedVenue(
            id="test-id",
            name="Test Venue",
            venue_type=VenueType.MID_CONFERENCE,
            field="Test Field",
            prestige_score=15  # Invalid score > 10
        )
        
        with pytest.raises(ValidationError) as exc_info:
            registry.register_venue(invalid_venue)
        
        assert "prestige_score" in str(exc_info.value)


class TestGlobalVenueRegistry:
    """Test global venue registry functions."""
    
    def setup_method(self):
        """Reset global registry before each test."""
        import src.enhancements.venue_system
        src.enhancements.venue_system._venue_registry = None
    
    def test_get_venue_registry_singleton(self):
        """Test that get_venue_registry returns singleton."""
        registry1 = get_venue_registry()
        registry2 = get_venue_registry()
        
        assert registry1 is registry2
    
    def test_initialize_standard_venues(self):
        """Test initializing standard venues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venues = initialize_standard_venues(Path(temp_dir))
            
            assert len(venues) == 4
            venue_names = [v.name for v in venues]
            assert "ACL" in venue_names
            assert "NeurIPS" in venue_names
            assert "ICLR" in venue_names
            assert "CoNLL" in venue_names


class TestVenueSystemIntegration:
    """Integration tests for venue system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_full_venue_workflow(self, temp_dir):
        """Test complete venue management workflow."""
        # Initialize registry
        registry = VenueRegistry(temp_dir)
        
        # Create venues from profiles
        venues = registry.create_standard_venues()
        assert len(venues) == 4
        
        # Create researchers with different qualifications
        junior_researcher = EnhancedResearcher(
            id="junior",
            name="Dr. Junior",
            specialty="Machine Learning",
            level=ResearcherLevel.ASSISTANT_PROF,
            h_index=5,
            years_active=2,
            institution_tier=3
        )
        
        senior_researcher = EnhancedResearcher(
            id="senior",
            name="Dr. Senior",
            specialty="Machine Learning",
            level=ResearcherLevel.FULL_PROF,
            h_index=30,
            years_active=20,
            institution_tier=1
        )
        
        # Check venue qualifications
        junior_venues = registry.get_venues_for_researcher(junior_researcher)
        senior_venues = registry.get_venues_for_researcher(senior_researcher)
        
        # Junior researcher should qualify for fewer venues
        assert len(junior_venues) < len(senior_venues)
        
        # Senior researcher should qualify for top venues
        senior_venue_names = [v.name for v in senior_venues]
        assert "NeurIPS" in senior_venue_names
        
        # Test venue retrieval
        neurips = registry.get_venue_by_name("NeurIPS")
        assert neurips is not None
        assert neurips.venue_type == VenueType.TOP_CONFERENCE
        
        # Test filtering
        top_conferences = registry.list_venues(venue_type=VenueType.TOP_CONFERENCE)
        assert len(top_conferences) == 3  # ACL, NeurIPS, ICLR
        
        ml_venues = registry.list_venues(field="Machine Learning")
        assert len(ml_venues) == 2  # NeurIPS, ICLR
        
        # Test statistics
        stats = registry.get_registry_stats()
        assert stats["total_venues"] == 4
        assert "Top Conference" in stats["venue_types"]
        assert "Mid Conference" in stats["venue_types"]


if __name__ == "__main__":
    pytest.main([__file__])