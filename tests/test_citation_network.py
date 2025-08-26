"""
Unit tests for CitationNetwork class.

Tests citation network modeling, citation bias effects,
and citation pattern analysis functionality.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch

from src.enhancements.citation_network import (
    CitationNetwork, CitationRecord, CitationBiasEffect, NetworkError
)
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord, ResearcherLevel
from src.core.exceptions import ValidationError


class TestCitationRecord:
    """Test CitationRecord dataclass."""
    
    def test_valid_citation_record(self):
        """Test creating a valid citation record."""
        record = CitationRecord(
            citing_paper_id="paper1",
            cited_paper_id="paper2",
            citing_author_ids=["author1"],
            cited_author_ids=["author2"],
            citation_date=date(2023, 1, 1)
        )
        
        assert record.citing_paper_id == "paper1"
        assert record.cited_paper_id == "paper2"
        assert record.citation_type == "reference"
    
    def test_same_paper_ids_raises_error(self):
        """Test that same paper IDs raise ValidationError."""
        with pytest.raises(ValidationError):
            CitationRecord(
                citing_paper_id="paper1",
                cited_paper_id="paper1",
                citing_author_ids=["author1"],
                cited_author_ids=["author2"],
                citation_date=date(2023, 1, 1)
            )
    
    def test_empty_author_lists_raise_error(self):
        """Test that empty author lists raise ValidationError."""
        with pytest.raises(ValidationError):
            CitationRecord(
                citing_paper_id="paper1",
                cited_paper_id="paper2",
                citing_author_ids=[],
                cited_author_ids=["author2"],
                citation_date=date(2023, 1, 1)
            )


class TestCitationNetwork:
    """Test CitationNetwork class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = CitationNetwork()
    
    def test_initialization(self):
        """Test network initialization."""
        assert len(self.network.citation_records) == 0
        assert len(self.network.paper_citations) == 0
        assert len(self.network.author_citations) == 0
    
    def test_add_citation(self):
        """Test adding a citation."""
        record = self.network.add_citation(
            citing_paper_id="paper1",
            cited_paper_id="paper2",
            citing_author_ids=["author1"],
            cited_author_ids=["author2"],
            citation_date=date(2023, 1, 1),
            citation_context="Related work"
        )
        
        assert len(self.network.citation_records) == 1
        assert record.citing_paper_id == "paper1"
        assert record.cited_paper_id == "paper2"
        
        # Check citation tracking
        assert "paper1" in self.network.paper_citations["paper2"]
        assert self.network.author_citations["author1"]["author2"] == 1
    
    def test_add_citation_same_papers_raises_error(self):
        """Test that adding citation with same papers raises error."""
        with pytest.raises(NetworkError):
            self.network.add_citation(
                citing_paper_id="paper1",
                cited_paper_id="paper1",
                citing_author_ids=["author1"],
                cited_author_ids=["author2"],
                citation_date=date(2023, 1, 1)
            )
    
    def test_get_paper_citations(self):
        """Test getting papers that cite a given paper."""
        self.network.add_citation(
            "paper1", "paper2", ["author1"], ["author2"], date(2023, 1, 1)
        )
        self.network.add_citation(
            "paper3", "paper2", ["author3"], ["author2"], date(2023, 2, 1)
        )
        
        citations = self.network.get_paper_citations("paper2")
        assert "paper1" in citations
        assert "paper3" in citations
        assert len(citations) == 2
    
    def test_get_paper_references(self):
        """Test getting papers referenced by a given paper."""
        self.network.add_citation(
            "paper1", "paper2", ["author1"], ["author2"], date(2023, 1, 1)
        )
        self.network.add_citation(
            "paper1", "paper3", ["author1"], ["author3"], date(2023, 1, 1)
        )
        
        references = self.network.get_paper_references("paper1")
        assert "paper2" in references
        assert "paper3" in references
        assert len(references) == 2
    
    def test_get_citation_count(self):
        """Test getting citation count for a paper."""
        assert self.network.get_citation_count("paper1") == 0
        
        self.network.add_citation(
            "paper2", "paper1", ["author2"], ["author1"], date(2023, 1, 1)
        )
        self.network.add_citation(
            "paper3", "paper1", ["author3"], ["author1"], date(2023, 2, 1)
        )
        
        assert self.network.get_citation_count("paper1") == 2
    
    def test_get_author_citation_relationship(self):
        """Test getting citation relationship between authors."""
        assert self.network.get_author_citation_relationship("author1", "author2") == 0
        
        self.network.add_citation(
            "paper1", "paper2", ["author1"], ["author2"], date(2023, 1, 1)
        )
        self.network.add_citation(
            "paper3", "paper2", ["author1"], ["author2"], date(2023, 2, 1)
        )
        
        assert self.network.get_author_citation_relationship("author1", "author2") == 2
    
    def test_identify_citation_connections(self):
        """Test identifying citation connections between reviewer and authors."""
        # Add citations
        self.network.add_citation(
            "paper1", "paper2", ["reviewer1"], ["author1"], date(2023, 1, 1)
        )
        self.network.add_citation(
            "paper3", "paper4", ["author1"], ["reviewer1"], date(2023, 2, 1)
        )
        
        connections = self.network.identify_citation_connections("reviewer1", ["author1"])
        
        assert connections["total_citations_to_authors"] == 1
        assert connections["total_citations_from_authors"] == 1
        assert connections["mutual_citations"] == 1
        assert "author1" in connections["author_connections"]
        assert connections["author_connections"]["author1"]["mutual"] is True
    
    def test_identify_self_citation_connections(self):
        """Test identifying self-citation connections."""
        connections = self.network.identify_citation_connections("author1", ["author1", "author2"])
        
        assert connections["self_citations"] == 1
    
    def test_calculate_citation_bias_positive(self):
        """Test calculating positive citation bias."""
        # Add citation from reviewer to author
        self.network.add_citation(
            "paper1", "paper2", ["reviewer1"], ["author1"], date(2023, 1, 1)
        )
        
        adjusted_score, bias_effects = self.network.calculate_citation_bias(
            "reviewer1", ["author1"], base_score=5.0
        )
        
        assert adjusted_score > 5.0  # Should be positively biased
        assert len(bias_effects) == 1
        assert bias_effects[0].bias_type == "positive-citation"
        assert bias_effects[0].bias_strength > 0
    
    def test_calculate_citation_bias_reciprocal(self):
        """Test calculating reciprocal citation bias."""
        # Add citation from author to reviewer
        self.network.add_citation(
            "paper1", "paper2", ["author1"], ["reviewer1"], date(2023, 1, 1)
        )
        
        adjusted_score, bias_effects = self.network.calculate_citation_bias(
            "reviewer1", ["author1"], base_score=5.0
        )
        
        assert adjusted_score > 5.0  # Should be positively biased (smaller effect)
        assert len(bias_effects) == 1
        assert bias_effects[0].bias_type == "reciprocal-citation"
    
    def test_calculate_citation_bias_self_review(self):
        """Test calculating self-review citation bias."""
        adjusted_score, bias_effects = self.network.calculate_citation_bias(
            "author1", ["author1"], base_score=5.0
        )
        
        assert adjusted_score > 5.0  # Should be strongly positively biased
        assert len(bias_effects) == 1
        assert bias_effects[0].bias_type == "self-citation"
        assert bias_effects[0].bias_strength == 1.0
    
    def test_calculate_citation_bias_multiple_citations(self):
        """Test citation bias with multiple citations."""
        # Add multiple citations from reviewer to author
        for i in range(3):
            self.network.add_citation(
                f"paper{i}", "paper_target", ["reviewer1"], ["author1"], date(2023, i+1, 1)
            )
        
        adjusted_score, bias_effects = self.network.calculate_citation_bias(
            "reviewer1", ["author1"], base_score=5.0
        )
        
        # Should have stronger bias due to multiple citations
        assert adjusted_score > 5.0
        assert len(bias_effects) == 1
        assert bias_effects[0].citation_count == 3
    
    def test_get_most_cited_papers(self):
        """Test getting most cited papers."""
        # Add citations
        self.network.add_citation("paper1", "paper_target", ["author1"], ["target_author"], date(2023, 1, 1))
        self.network.add_citation("paper2", "paper_target", ["author2"], ["target_author"], date(2023, 2, 1))
        self.network.add_citation("paper3", "paper_other", ["author3"], ["other_author"], date(2023, 3, 1))
        
        most_cited = self.network.get_most_cited_papers(limit=2)
        
        assert len(most_cited) == 2
        assert most_cited[0][0] == "paper_target"  # Most cited
        assert most_cited[0][1] == 2  # Citation count
        assert most_cited[1][0] == "paper_other"
        assert most_cited[1][1] == 1
    
    def test_get_most_cited_authors(self):
        """Test getting most cited authors."""
        # Add citations
        self.network.add_citation("paper1", "paper_target1", ["author1"], ["target_author"], date(2023, 1, 1))
        self.network.add_citation("paper2", "paper_target2", ["author2"], ["target_author"], date(2023, 2, 1))
        self.network.add_citation("paper3", "paper_other", ["author3"], ["other_author"], date(2023, 3, 1))
        
        most_cited = self.network.get_most_cited_authors(limit=2)
        
        assert len(most_cited) == 2
        assert most_cited[0][0] == "target_author"  # Most cited
        assert most_cited[0][1] == 2  # Citation count
    
    def test_detect_citation_patterns(self):
        """Test detecting suspicious citation patterns."""
        # Add mutual citations
        for i in range(4):
            self.network.add_citation(f"paper{i}", f"paper_b{i}", ["author1"], ["author2"], date(2023, i+1, 1))
            self.network.add_citation(f"paper_b{i}", f"paper{i}", ["author2"], ["author1"], date(2023, i+1, 1))
        
        # Add self-citations
        for i in range(4):
            self.network.add_citation(f"paper_self{i}", f"paper_self_target{i}", ["author3"], ["author3"], date(2023, i+1, 1))
        
        patterns = self.network.detect_citation_patterns(min_citations=3)
        
        # Should detect both mutual citation pairs and self-citations
        # The self-citations also count as mutual pairs (author3 -> author3)
        assert len(patterns["mutual_citation_pairs"]) == 2  # author1<->author2 and author3<->author3
        assert len(patterns["high_self_citers"]) == 1  # author3
        
        # Find the non-self mutual citation pair
        non_self_pairs = [pair for pair in patterns["mutual_citation_pairs"] if pair[0] != pair[1]]
        assert len(non_self_pairs) == 1
        assert non_self_pairs[0][2] == 8  # Total mutual citations between author1 and author2
    
    def test_get_citation_network_statistics(self):
        """Test getting network statistics."""
        # Add some citations
        self.network.add_citation("paper1", "paper2", ["author1"], ["author2"], date(2023, 1, 1))
        self.network.add_citation("paper3", "paper2", ["author3"], ["author2"], date(2023, 2, 1))
        
        stats = self.network.get_citation_network_statistics()
        
        assert stats["total_papers"] == 3
        assert stats["total_citations"] == 2
        assert stats["total_authors"] == 3
        assert stats["average_citations_per_paper"] > 0
        assert "citation_distribution" in stats
        assert "most_cited_papers" in stats
        assert "most_cited_authors" in stats
        assert "suspicious_patterns" in stats
    
    def test_build_network_from_researchers(self):
        """Test building network from researcher objects."""
        # Create test researchers with publication history
        researcher1 = EnhancedResearcher(
            id="researcher1",
            name="Researcher 1",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF,
            citation_network={"researcher2"},
            publication_history=[
                PublicationRecord(
                    paper_id="paper1",
                    title="Paper 1",
                    venue="Conference A",
                    year=2023,
                    citations=10
                )
            ]
        )
        
        researcher2 = EnhancedResearcher(
            id="researcher2",
            name="Researcher 2",
            specialty="ML",
            level=ResearcherLevel.POSTDOC,
            citation_network={"researcher1"},
            publication_history=[
                PublicationRecord(
                    paper_id="paper2",
                    title="Paper 2",
                    venue="Conference B",
                    year=2022,
                    citations=20
                )
            ]
        )
        
        self.network.build_network_from_researchers([researcher1, researcher2])
        
        # Check that network was built
        assert len(self.network.citation_records) == 1
        assert len(self.network.paper_authors) >= 2
    
    def test_serialization(self):
        """Test network serialization and deserialization."""
        # Add some data
        self.network.add_citation(
            "paper1", "paper2", ["author1"], ["author2"], date(2023, 1, 1),
            citation_context="Related work", citation_type="comparison"
        )
        
        # Serialize
        data = self.network.to_dict()
        
        # Deserialize
        new_network = CitationNetwork.from_dict(data)
        
        assert len(new_network.citation_records) == len(self.network.citation_records)
        assert len(new_network.paper_citations) == len(self.network.paper_citations)
        assert len(new_network.author_citations) == len(self.network.author_citations)
        
        # Check specific record
        record = new_network.citation_records[0]
        assert record.citing_paper_id == "paper1"
        assert record.cited_paper_id == "paper2"
        assert record.citation_context == "Related work"
        assert record.citation_type == "comparison"
    
    def test_bias_score_clamping(self):
        """Test that bias adjustments are properly clamped."""
        # Test upper bound clamping
        adjusted_score, _ = self.network.calculate_citation_bias(
            "author1", ["author1"], base_score=9.0  # High base score
        )
        assert adjusted_score <= 10.0  # Should be clamped to max
        
        # Test lower bound clamping (though unlikely with positive bias)
        adjusted_score, _ = self.network.calculate_citation_bias(
            "reviewer1", ["author1"], base_score=1.0  # Low base score
        )
        assert adjusted_score >= 1.0  # Should be clamped to min
    
    def test_empty_network_statistics(self):
        """Test statistics on empty network."""
        stats = self.network.get_citation_network_statistics()
        
        assert stats["total_papers"] == 0
        assert stats["total_citations"] == 0
        assert stats["total_authors"] == 0
        assert stats["average_citations_per_paper"] == 0.0
        assert len(stats["most_cited_papers"]) == 0
        assert len(stats["most_cited_authors"]) == 0
    
    def test_multiple_author_citation_tracking(self):
        """Test citation tracking with multiple authors."""
        self.network.add_citation(
            "paper1", "paper2", 
            ["author1", "author2"], ["author3", "author4"], 
            date(2023, 1, 1)
        )
        
        # All citing authors should have citations to all cited authors
        assert self.network.get_author_citation_relationship("author1", "author3") == 1
        assert self.network.get_author_citation_relationship("author1", "author4") == 1
        assert self.network.get_author_citation_relationship("author2", "author3") == 1
        assert self.network.get_author_citation_relationship("author2", "author4") == 1