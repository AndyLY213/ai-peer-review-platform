"""
Unit tests for NetworkInfluenceCalculator class.

Tests network distance calculations, review weight adjustments,
and network-based bias effects functionality.
"""

import pytest
from datetime import date
from unittest.mock import Mock, MagicMock

from src.enhancements.network_influence import (
    NetworkInfluenceCalculator, NetworkDistance, NetworkInfluenceEffect
)
from src.enhancements.collaboration_network import CollaborationNetwork
from src.enhancements.citation_network import CitationNetwork
from src.enhancements.conference_community import ConferenceCommunity
from src.core.exceptions import ValidationError


class TestNetworkDistance:
    """Test NetworkDistance dataclass."""
    
    def test_valid_network_distance(self):
        """Test creating a valid network distance."""
        distance = NetworkDistance(
            researcher_1_id="researcher1",
            researcher_2_id="researcher2",
            collaboration_distance=0.3,
            citation_distance=0.5,
            community_distance=0.2,
            overall_distance=0.35,
            connection_strength=0.65
        )
        
        assert distance.researcher_1_id == "researcher1"
        assert distance.researcher_2_id == "researcher2"
        assert distance.overall_distance == 0.35
    
    def test_invalid_distance_values_raise_error(self):
        """Test that invalid distance values raise ValidationError."""
        with pytest.raises(ValidationError):
            NetworkDistance(
                researcher_1_id="researcher1",
                researcher_2_id="researcher2",
                collaboration_distance=1.5,  # Too high
                citation_distance=0.5,
                community_distance=0.2,
                overall_distance=0.35,
                connection_strength=0.65
            )


class TestNetworkInfluenceCalculator:
    """Test NetworkInfluenceCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock networks
        self.mock_collaboration = Mock(spec=CollaborationNetwork)
        self.mock_citation = Mock(spec=CitationNetwork)
        self.mock_community = Mock(spec=ConferenceCommunity)
        
        self.calculator = NetworkInfluenceCalculator(
            collaboration_network=self.mock_collaboration,
            citation_network=self.mock_citation,
            conference_community=self.mock_community
        )
    
    def setup_basic_mocks(self):
        """Set up basic mock return values for common operations."""
        self.mock_collaboration.get_collaboration_strength.return_value = 0.0
        self.mock_collaboration.researcher_collaborations = {}
        self.mock_citation.get_author_citation_relationship.return_value = 0
        self.mock_citation.author_papers = {}
        self.mock_community.calculate_attendance_overlap.return_value = {
            "total_overlaps": 0,
            "overlap_strength": 0.0
        }
        self.mock_community.researcher_venues = {}
        self.mock_community.get_venue_attendees.return_value = set()
        
        # Mock network statistics
        self.mock_collaboration.get_network_statistics.return_value = {
            "total_researchers": 0,
            "total_collaborations": 0
        }
        self.mock_citation.get_citation_network_statistics.return_value = {
            "total_authors": 0,
            "total_citations": 0
        }
        self.mock_community.get_community_statistics.return_value = {
            "total_attendees": 0,
            "total_venues": 0
        }
    
    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator.collaboration_network is not None
        assert self.calculator.citation_network is not None
        assert self.calculator.conference_community is not None
        assert len(self.calculator.distance_cache) == 0
    
    def test_initialization_without_networks(self):
        """Test calculator initialization without networks."""
        calculator = NetworkInfluenceCalculator()
        
        assert calculator.collaboration_network is None
        assert calculator.citation_network is None
        assert calculator.conference_community is None
    
    def test_calculate_collaboration_distance_direct(self):
        """Test calculating direct collaboration distance."""
        # Mock direct collaboration
        self.mock_collaboration.get_collaboration_strength.return_value = 0.8
        
        distance = self.calculator.calculate_collaboration_distance("researcher1", "researcher2")
        
        assert distance == pytest.approx(0.2, abs=0.01)  # 1.0 - 0.8
        self.mock_collaboration.get_collaboration_strength.assert_called_once_with("researcher1", "researcher2")
    
    def test_calculate_collaboration_distance_indirect(self):
        """Test calculating indirect collaboration distance."""
        # Mock no direct collaboration but indirect through common collaborator
        self.mock_collaboration.get_collaboration_strength.side_effect = [0.0, 0.6, 0.7]  # direct, then indirect
        self.mock_collaboration.researcher_collaborations = {
            "researcher1": {"common_collaborator"},
            "researcher2": {"common_collaborator"}
        }
        
        distance = self.calculator.calculate_collaboration_distance("researcher1", "researcher2")
        
        # Should calculate indirect connection: (0.6 * 0.7) * 0.5 = 0.21, distance = 1.0 - 0.21 = 0.79
        assert distance == pytest.approx(0.79, abs=0.01)
    
    def test_calculate_collaboration_distance_no_network(self):
        """Test collaboration distance calculation without network."""
        calculator = NetworkInfluenceCalculator()
        distance = calculator.calculate_collaboration_distance("researcher1", "researcher2")
        assert distance == 1.0
    
    def test_calculate_citation_distance_direct(self):
        """Test calculating direct citation distance."""
        # Mock direct citations
        self.mock_citation.get_author_citation_relationship.side_effect = [3, 2]  # 3 citations one way, 2 the other
        
        distance = self.calculator.calculate_citation_distance("researcher1", "researcher2")
        
        # Total citations = 5, strength = log(6)/log(10) ≈ 0.778, distance ≈ 0.222
        assert distance < 0.5  # Should be relatively close due to citations
    
    def test_calculate_citation_distance_indirect(self):
        """Test calculating indirect citation distance."""
        # Mock no direct citations but common citing papers
        self.mock_citation.get_author_citation_relationship.return_value = 0
        self.mock_citation.author_papers = {
            "researcher1": {"paper1"},
            "researcher2": {"paper2"}
        }
        self.mock_citation.get_paper_citations.side_effect = [{"citing_paper1"}, {"citing_paper1"}]
        
        distance = self.calculator.calculate_citation_distance("researcher1", "researcher2")
        
        # Should have some indirect connection
        assert distance < 1.0
    
    def test_calculate_citation_distance_no_network(self):
        """Test citation distance calculation without network."""
        calculator = NetworkInfluenceCalculator()
        distance = calculator.calculate_citation_distance("researcher1", "researcher2")
        assert distance == 1.0
    
    def test_calculate_community_distance_same_venue(self):
        """Test calculating community distance for same venue."""
        # Mock both researchers attend the venue
        self.mock_community.get_venue_attendees.return_value = {"researcher1", "researcher2"}
        self.mock_community.calculate_community_influence.side_effect = [
            Mock(clique_memberships=["clique1"], community_standing=0.8),
            Mock(clique_memberships=["clique1"], community_standing=0.7)
        ]
        
        distance = self.calculator.calculate_community_distance("researcher1", "researcher2", "venue1")
        
        assert distance == 0.1  # Very close due to same clique
    
    def test_calculate_community_distance_different_attendance(self):
        """Test calculating community distance with different attendance."""
        # Mock only one researcher attends
        self.mock_community.get_venue_attendees.return_value = {"researcher1"}
        
        distance = self.calculator.calculate_community_distance("researcher1", "researcher2", "venue1")
        
        assert distance == 0.7  # Moderate distance
    
    def test_calculate_community_distance_no_network(self):
        """Test community distance calculation without network."""
        calculator = NetworkInfluenceCalculator()
        distance = calculator.calculate_community_distance("researcher1", "researcher2")
        assert distance == 1.0
    
    def test_calculate_network_distance(self):
        """Test calculating overall network distance."""
        # Mock individual distance calculations
        self.mock_collaboration.get_collaboration_strength.return_value = 0.5
        self.mock_collaboration.researcher_collaborations = {}
        self.mock_citation.get_author_citation_relationship.return_value = 0
        self.mock_citation.author_papers = {}
        self.mock_community.calculate_attendance_overlap.return_value = {
            "total_overlaps": 0,
            "overlap_strength": 0.0
        }
        self.mock_community.researcher_venues = {}
        
        distance = self.calculator.calculate_network_distance("researcher1", "researcher2")
        
        assert isinstance(distance, NetworkDistance)
        assert distance.researcher_1_id == "researcher1"
        assert distance.researcher_2_id == "researcher2"
        assert 0.0 <= distance.overall_distance <= 1.0
        assert 0.0 <= distance.connection_strength <= 1.0
    
    def test_calculate_network_distance_caching(self):
        """Test that network distance calculations are cached."""
        # Mock individual distance calculations
        self.mock_collaboration.get_collaboration_strength.return_value = 0.5
        self.mock_collaboration.researcher_collaborations = {}
        self.mock_citation.get_author_citation_relationship.return_value = 0
        self.mock_citation.author_papers = {}
        self.mock_community.calculate_attendance_overlap.return_value = {
            "total_overlaps": 0,
            "overlap_strength": 0.0
        }
        self.mock_community.researcher_venues = {}
        
        # Calculate distance twice
        distance1 = self.calculator.calculate_network_distance("researcher1", "researcher2")
        distance2 = self.calculator.calculate_network_distance("researcher1", "researcher2")
        
        # Should be the same object from cache
        assert distance1 is distance2
        assert len(self.calculator.distance_cache) == 1
    
    def test_calculate_review_weight_adjustment_self_review(self):
        """Test review weight adjustment for self-review."""
        adjusted_weight, effects = self.calculator.calculate_review_weight_adjustment(
            "researcher1", ["researcher1"], base_weight=1.0
        )
        
        assert adjusted_weight == 0.1  # Minimum weight due to self-review
        assert len(effects) == 1
        assert effects[0].influence_type == "self-review"
        assert effects[0].weight_adjustment == -0.9
    
    def test_calculate_review_weight_adjustment_close_connection(self):
        """Test review weight adjustment for close network connection."""
        self.setup_basic_mocks()
        # Mock very close network connection (need high overall connection strength > 0.7)
        self.mock_collaboration.get_collaboration_strength.return_value = 0.95  # Very high collaboration
        # Also mock some citation connection to boost overall connection strength
        self.mock_citation.get_author_citation_relationship.side_effect = [5, 3]  # Strong citation relationship
        
        adjusted_weight, effects = self.calculator.calculate_review_weight_adjustment(
            "reviewer1", ["author1"], base_weight=1.0
        )
        
        assert adjusted_weight < 1.0  # Should be reduced due to close connection
        assert len(effects) > 0
        assert any(effect.influence_type == "proximity-bias" for effect in effects)
    
    def test_calculate_network_bias_adjustment_self_review(self):
        """Test network bias adjustment for self-review."""
        adjusted_score, effects = self.calculator.calculate_network_bias_adjustment(
            "researcher1", ["researcher1"], base_score=5.0
        )
        
        assert adjusted_score > 5.0  # Should be positively biased
        assert len(effects) == 1
        assert effects[0].influence_type == "self-review"
        assert effects[0].score_adjustment == 2.0
    
    def test_calculate_network_bias_adjustment_collaboration(self):
        """Test network bias adjustment for collaboration."""
        self.setup_basic_mocks()
        # Mock close collaboration
        self.mock_collaboration.get_collaboration_strength.return_value = 0.9
        
        adjusted_score, effects = self.calculator.calculate_network_bias_adjustment(
            "reviewer1", ["author1"], base_score=5.0
        )
        
        assert adjusted_score > 5.0  # Should be positively biased
        collaboration_effects = [e for e in effects if e.influence_type == "collaboration-bias"]
        assert len(collaboration_effects) > 0
    
    def test_calculate_network_bias_adjustment_score_clamping(self):
        """Test that bias adjustments are properly clamped."""
        # Test upper bound
        adjusted_score, _ = self.calculator.calculate_network_bias_adjustment(
            "researcher1", ["researcher1"], base_score=9.0  # High base + bias should be clamped
        )
        assert adjusted_score <= 10.0
        
        # Test lower bound
        self.setup_basic_mocks()
        adjusted_score, _ = self.calculator.calculate_network_bias_adjustment(
            "reviewer1", ["author1"], base_score=1.0  # Low base should stay above 1.0
        )
        assert adjusted_score >= 1.0
    
    def test_get_network_statistics(self):
        """Test getting network statistics."""
        self.setup_basic_mocks()
        stats = self.calculator.get_network_statistics()
        
        assert "networks_available" in stats
        assert stats["networks_available"]["collaboration"] is True
        assert stats["networks_available"]["citation"] is True
        assert stats["networks_available"]["community"] is True
        assert "distance_cache_size" in stats
        assert "network_coverage" in stats
    
    def test_get_network_statistics_with_mock_data(self):
        """Test network statistics with mock network data."""
        # Mock network statistics
        self.mock_collaboration.get_network_statistics.return_value = {
            "total_researchers": 10,
            "total_collaborations": 25
        }
        self.mock_citation.get_citation_network_statistics.return_value = {
            "total_authors": 15,
            "total_citations": 50
        }
        self.mock_community.get_community_statistics.return_value = {
            "total_attendees": 20,
            "total_venues": 5
        }
        
        stats = self.calculator.get_network_statistics()
        
        assert stats["network_coverage"]["collaboration"]["total_researchers"] == 10
        assert stats["network_coverage"]["citation"]["total_authors"] == 15
        assert stats["network_coverage"]["community"]["total_attendees"] == 20
    
    def test_clear_distance_cache(self):
        """Test clearing the distance cache."""
        self.setup_basic_mocks()
        # Add something to cache first
        self.mock_collaboration.get_collaboration_strength.return_value = 0.5
        
        self.calculator.calculate_network_distance("researcher1", "researcher2")
        assert len(self.calculator.distance_cache) == 1
        
        self.calculator.clear_distance_cache()
        assert len(self.calculator.distance_cache) == 0
    
    def test_analyze_network_effects(self):
        """Test analyzing network effects across multiple pairs."""
        self.setup_basic_mocks()
        # Mock network responses
        self.mock_collaboration.get_collaboration_strength.return_value = 0.3
        
        reviewer_author_pairs = [
            ("reviewer1", ["author1", "author2"]),
            ("reviewer2", ["author3"])
        ]
        
        analysis = self.calculator.analyze_network_effects(reviewer_author_pairs)
        
        assert analysis["total_pairs"] == 2
        assert "distance_distribution" in analysis
        assert "influence_types" in analysis
        assert "average_weight_adjustment" in analysis
        assert "average_score_adjustment" in analysis
        assert "high_influence_pairs" in analysis
    
    def test_multiple_author_weight_adjustment(self):
        """Test weight adjustment with multiple authors."""
        self.setup_basic_mocks()
        # Mock different connections to different authors - need very high for weight adjustment
        self.mock_collaboration.get_collaboration_strength.side_effect = [0.95, 0.1]  # Very strong, weak
        # Also add citation connections for the first author
        self.mock_citation.get_author_citation_relationship.side_effect = [8, 2, 0, 0]  # Strong for author1, none for author2
        
        adjusted_weight, effects = self.calculator.calculate_review_weight_adjustment(
            "reviewer1", ["author1", "author2"], base_weight=1.0
        )
        
        # Should have effects for the strongly connected author
        assert len(effects) >= 1  # At least one effect from strong connection
        assert adjusted_weight < 1.0  # Should be reduced due to strong connection to author1
    
    def test_multiple_author_bias_adjustment(self):
        """Test bias adjustment with multiple authors."""
        self.setup_basic_mocks()
        # Mock different connections to different authors
        self.mock_collaboration.get_collaboration_strength.side_effect = [0.9, 0.1]  # Strong, weak
        
        adjusted_score, effects = self.calculator.calculate_network_bias_adjustment(
            "reviewer1", ["author1", "author2"], base_score=5.0
        )
        
        # Should have positive bias due to strong connection to author1
        assert adjusted_score > 5.0
        collaboration_effects = [e for e in effects if e.influence_type == "collaboration-bias"]
        assert len(collaboration_effects) >= 1
    
    def test_network_distance_with_venue_context(self):
        """Test network distance calculation with venue context."""
        # Mock venue-specific community distance
        self.mock_collaboration.get_collaboration_strength.return_value = 0.5
        self.mock_collaboration.researcher_collaborations = {}
        self.mock_citation.get_author_citation_relationship.return_value = 0
        self.mock_citation.author_papers = {}
        self.mock_community.get_venue_attendees.return_value = {"researcher1", "researcher2"}
        self.mock_community.calculate_community_influence.side_effect = [
            Mock(clique_memberships=[], community_standing=0.5),
            Mock(clique_memberships=[], community_standing=0.6)
        ]
        self.mock_community.calculate_attendance_overlap.return_value = {"overlap_strength": 0.3}
        
        distance = self.calculator.calculate_network_distance("researcher1", "researcher2", "venue1")
        
        assert isinstance(distance, NetworkDistance)
        # Community distance should be calculated for the specific venue
        assert distance.community_distance < 1.0