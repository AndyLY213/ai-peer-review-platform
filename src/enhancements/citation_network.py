"""
Citation Network Modeling System

This module implements the CitationNetwork class to track paper citation relationships,
identify citation-based connections between researchers, and implement citation bias
effects on review scores.
"""

import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import math

from src.core.exceptions import ValidationError, NetworkError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord


logger = get_logger(__name__)


@dataclass
class CitationRecord:
    """Record of a citation between papers."""
    citing_paper_id: str
    cited_paper_id: str
    citing_author_ids: List[str]
    cited_author_ids: List[str]
    citation_date: date
    citation_context: str = ""  # Context where citation appears
    citation_type: str = "reference"  # reference, comparison, criticism, etc.
    
    def __post_init__(self):
        """Validate citation record."""
        if self.citing_paper_id == self.cited_paper_id:
            raise ValidationError("paper_ids", "same", "different paper IDs")
        if not self.citing_author_ids or not self.cited_author_ids:
            raise ValidationError("author_ids", "empty", "non-empty author lists")


@dataclass
class CitationBiasEffect:
    """Represents citation bias effect on a review."""
    reviewer_id: str
    author_id: str
    bias_type: str  # "positive-citation", "negative-citation", "self-citation"
    bias_strength: float  # 0-1 scale
    citation_count: int
    most_recent_citation: Optional[date] = None
    description: str = ""


class CitationNetwork:
    """
    Manages citation networks and tracks paper citation relationships.
    
    This class implements comprehensive citation tracking including:
    - Paper-to-paper citation relationships
    - Author-to-author citation networks
    - Citation bias effects on reviews
    - Citation pattern analysis
    """
    
    def __init__(self):
        """Initialize citation network."""
        self.citation_records: List[CitationRecord] = []
        self.paper_citations: Dict[str, Set[str]] = defaultdict(set)  # cited_paper -> {citing_papers}
        self.author_citations: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # citing_author -> {cited_author: count}
        self.paper_authors: Dict[str, List[str]] = {}  # paper_id -> [author_ids]
        self.author_papers: Dict[str, Set[str]] = defaultdict(set)  # author_id -> {paper_ids}
        
        logger.info("Initialized CitationNetwork")
    
    def add_citation(self, citing_paper_id: str, cited_paper_id: str,
                    citing_author_ids: List[str], cited_author_ids: List[str],
                    citation_date: date, citation_context: str = "",
                    citation_type: str = "reference") -> CitationRecord:
        """
        Add a citation record between papers.
        
        Args:
            citing_paper_id: ID of the citing paper
            cited_paper_id: ID of the cited paper
            citing_author_ids: IDs of citing paper authors
            cited_author_ids: IDs of cited paper authors
            citation_date: Date of citation
            citation_context: Context where citation appears
            citation_type: Type of citation
            
        Returns:
            CitationRecord: The created citation record
            
        Raises:
            ValidationError: If paper IDs are the same or author lists are empty
        """
        try:
            citation = CitationRecord(
                citing_paper_id=citing_paper_id,
                cited_paper_id=cited_paper_id,
                citing_author_ids=citing_author_ids,
                cited_author_ids=cited_author_ids,
                citation_date=citation_date,
                citation_context=citation_context,
                citation_type=citation_type
            )
            
            self.citation_records.append(citation)
            
            # Update paper citation tracking
            self.paper_citations[cited_paper_id].add(citing_paper_id)
            
            # Update paper-author mappings
            self.paper_authors[citing_paper_id] = citing_author_ids
            self.paper_authors[cited_paper_id] = cited_author_ids
            
            # Update author-paper mappings
            for author_id in citing_author_ids:
                self.author_papers[author_id].add(citing_paper_id)
            for author_id in cited_author_ids:
                self.author_papers[author_id].add(cited_paper_id)
            
            # Update author citation counts
            for citing_author in citing_author_ids:
                for cited_author in cited_author_ids:
                    self.author_citations[citing_author][cited_author] += 1
            
            logger.debug(f"Added citation from {citing_paper_id} to {cited_paper_id}")
            
            return citation
            
        except Exception as e:
            logger.error(f"Failed to add citation: {e}")
            raise NetworkError(f"Failed to add citation: {e}")
    
    def get_paper_citations(self, paper_id: str) -> Set[str]:
        """
        Get all papers that cite a given paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Set[str]: Set of citing paper IDs
        """
        return self.paper_citations.get(paper_id, set())
    
    def get_paper_references(self, paper_id: str) -> Set[str]:
        """
        Get all papers referenced by a given paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Set[str]: Set of referenced paper IDs
        """
        references = set()
        for record in self.citation_records:
            if record.citing_paper_id == paper_id:
                references.add(record.cited_paper_id)
        return references
    
    def get_citation_count(self, paper_id: str) -> int:
        """
        Get the citation count for a paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            int: Number of citations
        """
        return len(self.paper_citations.get(paper_id, set()))
    
    def get_author_citation_relationship(self, citing_author: str, cited_author: str) -> int:
        """
        Get the number of times one author has cited another.
        
        Args:
            citing_author: ID of the citing author
            cited_author: ID of the cited author
            
        Returns:
            int: Number of citations
        """
        return self.author_citations.get(citing_author, {}).get(cited_author, 0)
    
    def identify_citation_connections(self, reviewer_id: str, author_ids: List[str]) -> Dict[str, Any]:
        """
        Identify citation-based connections between a reviewer and paper authors.
        
        Args:
            reviewer_id: ID of the reviewer
            author_ids: IDs of the paper authors
            
        Returns:
            Dict[str, Any]: Citation connection information
        """
        connections = {
            "total_citations_to_authors": 0,
            "total_citations_from_authors": 0,
            "author_connections": {},
            "mutual_citations": 0,
            "self_citations": 0
        }
        
        for author_id in author_ids:
            # Citations from reviewer to author
            citations_to = self.get_author_citation_relationship(reviewer_id, author_id)
            # Citations from author to reviewer
            citations_from = self.get_author_citation_relationship(author_id, reviewer_id)
            
            connections["total_citations_to_authors"] += citations_to
            connections["total_citations_from_authors"] += citations_from
            
            if citations_to > 0 or citations_from > 0:
                connections["author_connections"][author_id] = {
                    "citations_to": citations_to,
                    "citations_from": citations_from,
                    "mutual": citations_to > 0 and citations_from > 0
                }
                
                if citations_to > 0 and citations_from > 0:
                    connections["mutual_citations"] += 1
            
            # Check for self-citations (same author)
            if reviewer_id == author_id:
                connections["self_citations"] += 1
        
        return connections
    
    def calculate_citation_bias(self, reviewer_id: str, author_ids: List[str],
                              base_score: float) -> Tuple[float, List[CitationBiasEffect]]:
        """
        Calculate citation bias effects on a review score.
        
        Args:
            reviewer_id: ID of the reviewer
            author_ids: IDs of the paper authors
            base_score: Base review score before bias
            
        Returns:
            Tuple[float, List[CitationBiasEffect]]: Adjusted score and bias effects
        """
        bias_effects = []
        total_bias_adjustment = 0.0
        
        connections = self.identify_citation_connections(reviewer_id, author_ids)
        
        for author_id in author_ids:
            if author_id in connections["author_connections"]:
                conn = connections["author_connections"][author_id]
                
                # Positive bias from citing the author's work
                if conn["citations_to"] > 0:
                    # Bias strength based on citation count (logarithmic scale)
                    bias_strength = min(0.3, 0.1 * math.log(conn["citations_to"] + 1))
                    bias_adjustment = bias_strength * 1.0  # Positive adjustment
                    
                    bias_effects.append(CitationBiasEffect(
                        reviewer_id=reviewer_id,
                        author_id=author_id,
                        bias_type="positive-citation",
                        bias_strength=bias_strength,
                        citation_count=conn["citations_to"],
                        description=f"Reviewer has cited author {conn['citations_to']} times"
                    ))
                    
                    total_bias_adjustment += bias_adjustment
                
                # Potential negative bias from being cited by the author
                if conn["citations_from"] > 0:
                    # Smaller effect, potential for reciprocal bias
                    bias_strength = min(0.1, 0.05 * math.log(conn["citations_from"] + 1))
                    bias_adjustment = bias_strength * 0.5  # Smaller positive adjustment
                    
                    bias_effects.append(CitationBiasEffect(
                        reviewer_id=reviewer_id,
                        author_id=author_id,
                        bias_type="reciprocal-citation",
                        bias_strength=bias_strength,
                        citation_count=conn["citations_from"],
                        description=f"Author has cited reviewer {conn['citations_from']} times"
                    ))
                    
                    total_bias_adjustment += bias_adjustment
            
            # Self-citation bias (if reviewer is also an author)
            if reviewer_id == author_id:
                bias_effects.append(CitationBiasEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    bias_type="self-citation",
                    bias_strength=1.0,
                    citation_count=0,
                    description="Reviewer is also an author (self-review)"
                ))
                # This should be caught by conflict of interest, but add strong positive bias
                total_bias_adjustment += 2.0
        
        # Apply bias adjustment to score (capped at reasonable limits)
        adjusted_score = base_score + min(2.0, max(-1.0, total_bias_adjustment))
        adjusted_score = max(1.0, min(10.0, adjusted_score))  # Keep within valid range
        
        logger.debug(f"Citation bias adjustment for reviewer {reviewer_id}: {total_bias_adjustment}")
        
        return adjusted_score, bias_effects
    
    def get_most_cited_papers(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most cited papers in the network.
        
        Args:
            limit: Maximum number of papers to return
            
        Returns:
            List[Tuple[str, int]]: List of (paper_id, citation_count) tuples
        """
        paper_counts = [(paper_id, len(citing_papers)) 
                       for paper_id, citing_papers in self.paper_citations.items()]
        paper_counts.sort(key=lambda x: x[1], reverse=True)
        return paper_counts[:limit]
    
    def get_most_cited_authors(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most cited authors in the network.
        
        Args:
            limit: Maximum number of authors to return
            
        Returns:
            List[Tuple[str, int]]: List of (author_id, citation_count) tuples
        """
        author_counts = defaultdict(int)
        
        for cited_paper_id, citing_papers in self.paper_citations.items():
            if cited_paper_id in self.paper_authors:
                citation_count = len(citing_papers)
                for author_id in self.paper_authors[cited_paper_id]:
                    author_counts[author_id] += citation_count
        
        author_list = list(author_counts.items())
        author_list.sort(key=lambda x: x[1], reverse=True)
        return author_list[:limit]
    
    def detect_citation_patterns(self, min_citations: int = 3) -> Dict[str, List[Tuple[str, str, int]]]:
        """
        Detect suspicious citation patterns.
        
        Args:
            min_citations: Minimum citations to consider suspicious
            
        Returns:
            Dict[str, List[Tuple[str, str, int]]]: Detected patterns
        """
        patterns = {
            "mutual_citation_pairs": [],
            "high_self_citers": [],
            "citation_clusters": []
        }
        
        # Find mutual citation pairs
        for citing_author, cited_authors in self.author_citations.items():
            for cited_author, count in cited_authors.items():
                if count >= min_citations:
                    reverse_count = self.author_citations.get(cited_author, {}).get(citing_author, 0)
                    if reverse_count >= min_citations:
                        # Mutual citation pair
                        pair = tuple(sorted([citing_author, cited_author]))
                        if (pair[0], pair[1], count + reverse_count) not in patterns["mutual_citation_pairs"]:
                            patterns["mutual_citation_pairs"].append((pair[0], pair[1], count + reverse_count))
        
        # Find high self-citers (authors citing themselves frequently)
        for author_id in self.author_citations:
            self_citations = self.author_citations[author_id].get(author_id, 0)
            if self_citations >= min_citations:
                patterns["high_self_citers"].append((author_id, author_id, self_citations))
        
        return patterns
    
    def get_citation_network_statistics(self) -> Dict[str, Any]:
        """
        Get citation network statistics and metrics.
        
        Returns:
            Dict[str, Any]: Network statistics
        """
        total_papers = len(set(list(self.paper_citations.keys()) + 
                              [record.citing_paper_id for record in self.citation_records]))
        total_citations = len(self.citation_records)
        total_authors = len(self.author_papers)
        
        # Calculate average citations per paper
        if total_papers > 0:
            avg_citations_per_paper = total_citations / total_papers
        else:
            avg_citations_per_paper = 0.0
        
        # Calculate citation distribution
        citation_counts = [len(citing_papers) for citing_papers in self.paper_citations.values()]
        if citation_counts:
            max_citations = max(citation_counts)
            min_citations = min(citation_counts)
            avg_citations = sum(citation_counts) / len(citation_counts)
        else:
            max_citations = min_citations = avg_citations = 0
        
        # Get most cited papers and authors
        most_cited_papers = self.get_most_cited_papers(5)
        most_cited_authors = self.get_most_cited_authors(5)
        
        # Detect patterns
        patterns = self.detect_citation_patterns()
        
        return {
            "total_papers": total_papers,
            "total_citations": total_citations,
            "total_authors": total_authors,
            "average_citations_per_paper": avg_citations_per_paper,
            "citation_distribution": {
                "max": max_citations,
                "min": min_citations,
                "average": avg_citations
            },
            "most_cited_papers": most_cited_papers,
            "most_cited_authors": most_cited_authors,
            "suspicious_patterns": {
                "mutual_citation_pairs": len(patterns["mutual_citation_pairs"]),
                "high_self_citers": len(patterns["high_self_citers"])
            }
        }
    
    def build_network_from_researchers(self, researchers: List[EnhancedResearcher]):
        """
        Build citation network from researcher publication histories.
        
        Args:
            researchers: List of enhanced researchers
        """
        logger.info(f"Building citation network from {len(researchers)} researchers")
        
        # Create researcher and paper lookups
        researcher_lookup = {r.id: r for r in researchers}
        paper_lookup = {}
        
        # Build paper lookup from all publications
        for researcher in researchers:
            for pub in researcher.publication_history:
                if pub.paper_id not in paper_lookup:
                    paper_lookup[pub.paper_id] = {
                        "title": pub.title,
                        "authors": [researcher.id],
                        "year": pub.year,
                        "venue": pub.venue,
                        "citations": pub.citations
                    }
                else:
                    # Add co-author
                    if researcher.id not in paper_lookup[pub.paper_id]["authors"]:
                        paper_lookup[pub.paper_id]["authors"].append(researcher.id)
        
        # Simulate citation relationships based on citation networks
        processed_pairs = set()
        
        for researcher in researchers:
            for cited_researcher_id in researcher.citation_network:
                if cited_researcher_id in researcher_lookup:
                    pair = tuple(sorted([researcher.id, cited_researcher_id]))
                    if pair not in processed_pairs:
                        # Find papers to create citation relationship
                        citing_papers = [p for p in researcher.publication_history if p.year >= 2020]
                        cited_papers = [p for p in researcher_lookup[cited_researcher_id].publication_history 
                                      if p.year < 2023]  # Cited papers should be older
                        
                        if citing_papers and cited_papers:
                            # Create citation from most recent citing paper to most cited paper
                            citing_paper = max(citing_papers, key=lambda p: p.year)
                            cited_paper = max(cited_papers, key=lambda p: p.citations)
                            
                            self.add_citation(
                                citing_paper_id=citing_paper.paper_id,
                                cited_paper_id=cited_paper.paper_id,
                                citing_author_ids=paper_lookup[citing_paper.paper_id]["authors"],
                                cited_author_ids=paper_lookup[cited_paper.paper_id]["authors"],
                                citation_date=date(citing_paper.year, 6, 1),  # Approximate date
                                citation_context="Related work",
                                citation_type="reference"
                            )
                        
                        processed_pairs.add(pair)
        
        logger.info(f"Built citation network with {len(self.citation_records)} citation records")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "citation_records": [
                {
                    "citing_paper_id": record.citing_paper_id,
                    "cited_paper_id": record.cited_paper_id,
                    "citing_author_ids": record.citing_author_ids,
                    "cited_author_ids": record.cited_author_ids,
                    "citation_date": record.citation_date.isoformat(),
                    "citation_context": record.citation_context,
                    "citation_type": record.citation_type
                }
                for record in self.citation_records
            ],
            "paper_authors": self.paper_authors,
            "author_papers": {
                author_id: list(papers) 
                for author_id, papers in self.author_papers.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CitationNetwork':
        """Create from dictionary."""
        network = cls()
        
        # Load citation records
        for record_data in data.get("citation_records", []):
            record = CitationRecord(
                citing_paper_id=record_data["citing_paper_id"],
                cited_paper_id=record_data["cited_paper_id"],
                citing_author_ids=record_data["citing_author_ids"],
                cited_author_ids=record_data["cited_author_ids"],
                citation_date=date.fromisoformat(record_data["citation_date"]),
                citation_context=record_data.get("citation_context", ""),
                citation_type=record_data.get("citation_type", "reference")
            )
            network.citation_records.append(record)
            
            # Update tracking dictionaries
            network.paper_citations[record.cited_paper_id].add(record.citing_paper_id)
            
            # Update author citation counts
            for citing_author in record.citing_author_ids:
                for cited_author in record.cited_author_ids:
                    network.author_citations[citing_author][cited_author] += 1
        
        # Load paper-author mappings
        network.paper_authors = data.get("paper_authors", {})
        
        # Load author-paper mappings
        for author_id, papers in data.get("author_papers", {}).items():
            network.author_papers[author_id] = set(papers)
        
        return network