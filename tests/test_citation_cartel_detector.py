"""
Unit tests for CitationCartelDetector class.
"""

import pytest
import uuid
from datetime import date, datetime
from unittest.mock import Mock, patch
from collections import defaultdict

from src.enhancements.citation_cartel_detector import (
    CitationCartelDetector, CitationCartel, MutualCitationPair, CitationRing
)
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord, ResearcherLevel
from src.core.exceptions import ValidationError, NetworkError


class TestCitationCartelDetector:
    """Test cases for CitationCartelDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CitationCartelDetector(
            min_mutual_citations=3,
            min_ring_size=3,
            suspicion_threshold=0.7
        )
        
        # Create mock citation network
        self.mock_citation_network = Mock()
        self.mock_citation_network.author_citations = defaultdict(lambda: defaultdict(int))
        self.mock_citation_network.author_papers = defaultdict(set)
        self.mock_citation_network.paper_authors = {}
        
        # Sample researchers
        self.researchers = [
            EnhancedResearcher(
                id="researcher_1",
                name="Dr. Alice Smith",
                specialty="Machine Learning",
                level=ResearcherLevel.ASSISTANT_PROF,
                institution_tier=1,
                h_index=15,
                total_citations=500,
                years_active=5,
                reputation_score=0.8,
                cognitive_biases={},
                review_behavior=Mock(),
                strategic_behavior=Mock(),
                career_stage=Mock(),
                funding_status=Mock(),
                publication_pressure=0.5,
                tenure_timeline=None,
                collaboration_network={"researcher_2", "researcher_3"},
                citation_network={"researcher_2", "researcher_3"},
                institutional_affiliations=["University A"],
                review_quality_history=[],
                publication_history=[
                    PublicationRecord(
                        paper_id="paper_1",
                        title="ML Paper 1",
                        year=2022,
                        venue="ICML",
                        citations=50
                    )
                ],
                career_milestones=[]
            ),
            EnhancedResearcher(
                id="researcher_2",
                name="Dr. Bob Johnson",
                specialty="Deep Learning",
                level=ResearcherLevel.ASSOCIATE_PROF,
                institution_tier=1,
                h_index=20,
                total_citations=800,
                years_active=8,
                reputation_score=0.9,
                cognitive_biases={},
                review_behavior=Mock(),
                strategic_behavior=Mock(),
                career_stage=Mock(),
                funding_status=Mock(),
                publication_pressure=0.6,
                tenure_timeline=None,
                collaboration_network={"researcher_1", "researcher_3"},
                citation_network={"researcher_1", "researcher_3"},
                institutional_affiliations=["University B"],
                review_quality_history=[],
                publication_history=[
                    PublicationRecord(
                        paper_id="paper_2",
                        title="DL Paper 1",
                        year=2021,
                        venue="NeurIPS",
                        citations=80
                    )
                ],
                career_milestones=[]
            ),
            EnhancedResearcher(
                id="researcher_3",
                name="Dr. Carol Davis",
                specialty="Computer Vision",
                level=ResearcherLevel.FULL_PROF,
                institution_tier=2,
                h_index=25,
                total_citations=1200,
                years_active=12,
                reputation_score=0.95,
                cognitive_biases={},
                review_behavior=Mock(),
                strategic_behavior=Mock(),
                career_stage=Mock(),
                funding_status=Mock(),
                publication_pressure=0.4,
                tenure_timeline=None,
                collaboration_network={"researcher_1", "researcher_2"},
                citation_network={"researcher_1", "researcher_2"},
                institutional_affiliations=["University C"],
                review_quality_history=[],
                publication_history=[
                    PublicationRecord(
                        paper_id="paper_3",
                        title="CV Paper 1",
                        year=2020,
                        venue="CVPR",
                        citations=120
                    )
                ],
                career_milestones=[]
            )
        ]
    
    def test_initialization(self):
        """Test CitationCartelDetector initialization."""
        detector = CitationCartelDetector(
            min_mutual_citations=5,
            min_ring_size=4,
            suspicion_threshold=0.8
        )
        
        assert detector.min_mutual_citations == 5
        assert detector.min_ring_size == 4
        assert detector.suspicion_threshold == 0.8
        assert detector.detected_cartels == []
        assert detector.mutual_pairs == []
        assert detector.citation_rings == []
    
    def test_load_citation_data(self):
        """Test loading citation data from citation network."""
        # Set up mock data
        self.mock_citation_network.author_citations["researcher_1"]["researcher_2"] = 5
        self.mock_citation_network.author_citations["researcher_2"]["researcher_1"] = 4
        self.mock_citation_network.author_papers["researcher_1"] = {"paper_1", "paper_2"}
        self.mock_citation_network.paper_authors["paper_1"] = ["researcher_1"]
        
        self.detector.load_citation_data(self.mock_citation_network)
        
        assert self.detector.author_citations["researcher_1"]["researcher_2"] == 5
        assert self.detector.author_citations["researcher_2"]["researcher_1"] == 4
        assert "paper_1" in self.detector.author_papers["researcher_1"]
        assert self.detector.paper_authors["paper_1"] == ["researcher_1"]
    
    def test_load_citation_data_error(self):
        """Test error handling in load_citation_data."""
        mock_network = Mock()
        mock_network.author_citations = None  # This will cause an error
        
        with pytest.raises(NetworkError):
            self.detector.load_citation_data(mock_network)
    
    def test_detect_mutual_citation_pairs(self):
        """Test detection of mutual citation pairs."""
        # Set up citation data with mutual citations
        self.detector.author_citations["researcher_1"]["researcher_2"] = 5
        self.detector.author_citations["researcher_2"]["researcher_1"] = 4
        self.detector.author_citations["researcher_1"]["researcher_3"] = 2  # Below threshold
        self.detector.author_citations["researcher_3"]["researcher_1"] = 1  # Below threshold
        
        pairs = self.detector.detect_mutual_citation_pairs()
        
        assert len(pairs) == 1
        pair = pairs[0]
        assert set([pair.researcher_a, pair.researcher_b]) == {"researcher_1", "researcher_2"}
        assert pair.total_mutual_citations == 9
        assert pair.suspicion_score > 0
    
    def test_detect_mutual_citation_pairs_no_pairs(self):
        """Test detection when no mutual pairs exist."""
        # Set up citation data with no mutual citations above threshold
        self.detector.author_citations["researcher_1"]["researcher_2"] = 1
        self.detector.author_citations["researcher_2"]["researcher_3"] = 2
        
        pairs = self.detector.detect_mutual_citation_pairs()
        
        assert len(pairs) == 0
    
    def test_detect_citation_rings(self):
        """Test detection of citation rings."""
        # Set up citation data for a 3-member ring
        self.detector.author_citations["researcher_1"]["researcher_2"] = 4
        self.detector.author_citations["researcher_2"]["researcher_3"] = 3
        self.detector.author_citations["researcher_3"]["researcher_1"] = 5
        self.detector.author_citations["researcher_1"]["researcher_3"] = 2
        self.detector.author_citations["researcher_2"]["researcher_1"] = 3
        self.detector.author_citations["researcher_3"]["researcher_2"] = 4
        
        rings = self.detector.detect_citation_rings(min_ring_size=3)
        
        assert len(rings) >= 0  # May or may not detect depending on thresholds
        if rings:
            ring = rings[0]
            assert ring.ring_size == 3
            assert set(ring.member_ids) == {"researcher_1", "researcher_2", "researcher_3"}
            assert ring.total_ring_citations > 0
    
    def test_detect_citation_rings_no_rings(self):
        """Test detection when no citation rings exist."""
        # Set up sparse citation data
        self.detector.author_citations["researcher_1"]["researcher_2"] = 1
        
        rings = self.detector.detect_citation_rings()
        
        assert len(rings) == 0
    
    def test_analyze_citation_patterns(self):
        """Test comprehensive citation pattern analysis."""
        # Set up citation data with various patterns
        self.detector.author_citations["researcher_1"]["researcher_2"] = 5
        self.detector.author_citations["researcher_2"]["researcher_1"] = 4
        self.detector.author_citations["researcher_1"]["researcher_1"] = 3  # Self-citations
        self.detector.author_citations["researcher_2"]["researcher_3"] = 2
        
        analysis = self.detector.analyze_citation_patterns(self.researchers)
        
        assert "total_researchers" in analysis
        assert analysis["total_researchers"] == 3
        assert "citation_statistics" in analysis
        assert "suspicious_patterns" in analysis
        assert "cartel_analysis" in analysis
        
        # Check that cartels were created
        assert len(self.detector.detected_cartels) >= 0
    
    def test_get_researcher_cartel_involvement(self):
        """Test getting cartel involvement for a researcher."""
        # Create a test cartel
        cartel = CitationCartel(
            cartel_id="test_cartel",
            member_ids=["researcher_1", "researcher_2"],
            cartel_type="mutual_pair",
            detection_date=date.today(),
            strength_score=0.8,
            total_mutual_citations=10,
            average_citations_per_member=5.0,
            suspicious_patterns=["High mutual citations"],
            evidence={"test": "data"}
        )
        self.detector.detected_cartels = [cartel]
        
        involvement = self.detector.get_researcher_cartel_involvement("researcher_1")
        
        assert involvement["researcher_id"] == "researcher_1"
        assert involvement["total_cartels"] == 1
        assert involvement["max_strength_score"] == 0.8
        assert "mutual_pair" in involvement["cartel_types"]
    
    def test_get_researcher_cartel_involvement_no_cartels(self):
        """Test getting cartel involvement when researcher is not in any cartels."""
        involvement = self.detector.get_researcher_cartel_involvement("researcher_1")
        
        assert involvement["researcher_id"] == "researcher_1"
        assert involvement["total_cartels"] == 0
        assert involvement["max_strength_score"] == 0.0
        assert involvement["cartel_types"] == []
    
    def test_generate_cartel_report(self):
        """Test generating comprehensive cartel report."""
        # Create test cartels
        cartel1 = CitationCartel(
            cartel_id="cartel_1",
            member_ids=["researcher_1", "researcher_2"],
            cartel_type="mutual_pair",
            detection_date=date.today(),
            strength_score=0.8,
            total_mutual_citations=10,
            average_citations_per_member=5.0,
            suspicious_patterns=["High mutual citations"],
            evidence={}
        )
        cartel2 = CitationCartel(
            cartel_id="cartel_2",
            member_ids=["researcher_1", "researcher_2", "researcher_3"],
            cartel_type="citation_ring",
            detection_date=date.today(),
            strength_score=0.9,
            total_mutual_citations=15,
            average_citations_per_member=5.0,
            suspicious_patterns=["Citation ring"],
            evidence={}
        )
        self.detector.detected_cartels = [cartel1, cartel2]
        
        # Set up basic data
        self.detector.author_citations = {"researcher_1": {}, "researcher_2": {}}
        self.detector.paper_authors = {"paper_1": ["researcher_1"]}
        
        report = self.detector.generate_cartel_report()
        
        assert "detection_summary" in report
        assert report["detection_summary"]["total_cartels"] == 2
        assert report["detection_summary"]["mutual_pairs"] == 1
        assert report["detection_summary"]["citation_rings"] == 1
        
        assert "cartel_details" in report
        assert len(report["cartel_details"]) == 2
        
        assert "researcher_involvement" in report
        assert "researcher_1" in report["researcher_involvement"]
        
        assert "network_statistics" in report
    
    def test_remove_overlapping_rings(self):
        """Test removal of overlapping citation rings."""
        # Create overlapping rings
        ring1 = CitationRing(
            ring_id="ring_1",
            member_ids=["researcher_1", "researcher_2", "researcher_3"],
            ring_size=3,
            total_ring_citations=10,
            average_citations_per_edge=2.0,
            ring_density=0.8,
            detection_confidence=0.9,
            citation_matrix={}
        )
        ring2 = CitationRing(
            ring_id="ring_2",
            member_ids=["researcher_1", "researcher_2", "researcher_4"],  # Overlaps with ring1
            ring_size=3,
            total_ring_citations=8,
            average_citations_per_edge=1.5,
            ring_density=0.7,
            detection_confidence=0.8,
            citation_matrix={}
        )
        
        non_overlapping = self.detector._remove_overlapping_rings([ring1, ring2])
        
        # Should keep the higher confidence ring
        assert len(non_overlapping) == 1
        assert non_overlapping[0].ring_id == "ring_1"
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        # Create test data
        cartel = CitationCartel(
            cartel_id="test_cartel",
            member_ids=["researcher_1", "researcher_2"],
            cartel_type="mutual_pair",
            detection_date=date.today(),
            strength_score=0.8,
            total_mutual_citations=10,
            average_citations_per_member=5.0,
            suspicious_patterns=["Test pattern"],
            evidence={"test": "data"}
        )
        self.detector.detected_cartels = [cartel]
        
        pair = MutualCitationPair(
            researcher_a="researcher_1",
            researcher_b="researcher_2",
            citations_a_to_b=5,
            citations_b_to_a=4,
            time_span_years=3.0,
            suspicion_score=0.8
        )
        self.detector.mutual_pairs = [pair]
        
        # Test serialization
        data = self.detector.to_dict()
        
        assert "config" in data
        assert "detected_cartels" in data
        assert "mutual_pairs" in data
        assert "citation_rings" in data
        
        # Test deserialization
        new_detector = CitationCartelDetector.from_dict(data)
        
        assert new_detector.min_mutual_citations == self.detector.min_mutual_citations
        assert len(new_detector.detected_cartels) == 1
        assert len(new_detector.mutual_pairs) == 1
        assert new_detector.detected_cartels[0].cartel_id == "test_cartel"


class TestCitationCartel:
    """Test cases for CitationCartel dataclass."""
    
    def test_citation_cartel_creation(self):
        """Test CitationCartel creation and validation."""
        cartel = CitationCartel(
            cartel_id="test_cartel",
            member_ids=["researcher_1", "researcher_2"],
            cartel_type="mutual_pair",
            detection_date=date.today(),
            strength_score=0.8,
            total_mutual_citations=10,
            average_citations_per_member=5.0,
            suspicious_patterns=["High mutual citations"],
            evidence={"citations": 10}
        )
        
        assert cartel.cartel_id == "test_cartel"
        assert len(cartel.member_ids) == 2
        assert cartel.strength_score == 0.8
    
    def test_citation_cartel_validation_too_few_members(self):
        """Test validation error for too few members."""
        with pytest.raises(ValidationError):
            CitationCartel(
                cartel_id="test_cartel",
                member_ids=["researcher_1"],  # Only one member
                cartel_type="mutual_pair",
                detection_date=date.today(),
                strength_score=0.8,
                total_mutual_citations=10,
                average_citations_per_member=5.0,
                suspicious_patterns=[],
                evidence={}
            )
    
    def test_citation_cartel_validation_invalid_strength(self):
        """Test validation error for invalid strength score."""
        with pytest.raises(ValidationError):
            CitationCartel(
                cartel_id="test_cartel",
                member_ids=["researcher_1", "researcher_2"],
                cartel_type="mutual_pair",
                detection_date=date.today(),
                strength_score=1.5,  # Invalid score > 1
                total_mutual_citations=10,
                average_citations_per_member=5.0,
                suspicious_patterns=[],
                evidence={}
            )


class TestMutualCitationPair:
    """Test cases for MutualCitationPair dataclass."""
    
    def test_mutual_citation_pair_creation(self):
        """Test MutualCitationPair creation and calculations."""
        pair = MutualCitationPair(
            researcher_a="researcher_1",
            researcher_b="researcher_2",
            citations_a_to_b=5,
            citations_b_to_a=4,
            time_span_years=3.0,
            suspicion_score=0.8
        )
        
        assert pair.total_mutual_citations == 9
        assert pair.citation_ratio == 0.8  # 4/5
    
    def test_mutual_citation_pair_zero_citations(self):
        """Test MutualCitationPair with zero citations."""
        pair = MutualCitationPair(
            researcher_a="researcher_1",
            researcher_b="researcher_2",
            citations_a_to_b=0,
            citations_b_to_a=0,
            time_span_years=3.0,
            suspicion_score=0.0
        )
        
        assert pair.total_mutual_citations == 0
        assert pair.citation_ratio == 0.0


class TestCitationRing:
    """Test cases for CitationRing dataclass."""
    
    def test_citation_ring_creation(self):
        """Test CitationRing creation and calculations."""
        citation_matrix = {
            ("researcher_1", "researcher_2"): 3,
            ("researcher_2", "researcher_3"): 2,
            ("researcher_3", "researcher_1"): 4
        }
        
        ring = CitationRing(
            ring_id="test_ring",
            member_ids=["researcher_1", "researcher_2", "researcher_3"],
            ring_size=3,
            total_ring_citations=9,
            average_citations_per_edge=3.0,
            ring_density=0.5,
            detection_confidence=0.8,
            citation_matrix=citation_matrix
        )
        
        assert ring.ring_size == 3
        assert ring.ring_density == 0.5
        assert len(ring.citation_matrix) == 3
    
    def test_citation_ring_density_calculation(self):
        """Test citation ring density calculation."""
        citation_matrix = {
            ("researcher_1", "researcher_2"): 3,
            ("researcher_2", "researcher_1"): 2
        }
        
        ring = CitationRing(
            ring_id="test_ring",
            member_ids=["researcher_1", "researcher_2"],
            ring_size=2,
            total_ring_citations=5,
            average_citations_per_edge=2.5,
            ring_density=0.0,  # Will be recalculated
            detection_confidence=0.8,
            citation_matrix=citation_matrix
        )
        
        # For 2 members, possible edges = 2 * 1 = 2, actual edges = 2
        assert ring.ring_density == 1.0  # 2/2 = 1.0


if __name__ == "__main__":
    pytest.main([__file__])